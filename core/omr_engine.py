from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from models.template import Template, Zone, ZoneType


@dataclass
class OMRIssue:
    code: str
    message: str
    zone_id: str | None = None


@dataclass
class OMRResult:
    image_path: str
    student_id: str = ""
    exam_code: str = ""
    mcq_answers: dict[int, str] = field(default_factory=dict)
    true_false_answers: dict[int, dict[str, bool]] = field(default_factory=dict)
    numeric_answers: dict[int, str] = field(default_factory=dict)
    answer_string: str = ""
    confidence_scores: dict[str, float] = field(default_factory=dict)
    recognition_errors: list[str] = field(default_factory=list)
    issues: list[OMRIssue] = field(default_factory=list)
    processing_time_sec: float = 0.0
    debug_image_path: str = ""

    # backward compatible aliases
    confidence: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def sync_legacy_aliases(self) -> None:
        self.confidence = self.confidence_scores
        self.errors = self.recognition_errors


@dataclass
class RecognitionContext:
    detected_anchors: list[tuple[float, float]] = field(default_factory=list)
    detected_digit_anchors: list[tuple[float, float]] = field(default_factory=list)
    bubble_states_by_zone: dict[str, list[bool]] = field(default_factory=dict)
    semantic_grids: dict[str, object] = field(default_factory=dict)
    recognized_answers: dict[str, dict] = field(default_factory=dict)
    digit_zone_debug: dict[str, dict[str, object]] = field(default_factory=dict)
    deadline_monotonic: float = 0.0
    collect_diagnostics: bool = True
    force_identifier_recognition: bool = False
    fast_production_test: bool = True
    debug_deep: bool = False

    def reset(self) -> None:
        self.detected_anchors = []
        self.detected_digit_anchors = []
        self.bubble_states_by_zone = {}
        self.semantic_grids = {}
        self.digit_zone_debug = {}
        self.deadline_monotonic = 0.0
        self.recognized_answers = {
            "student_id": "",
            "exam_code": "",
            "mcq_answers": {},
            "true_false_answers": {},
            "numeric_answers": {},
        }


@dataclass
class TemplateRuntimePlan:
    key: tuple[int, int, int, int, int]
    target_width: int
    target_height: int
    alignment_profile: str
    template_match_mode: str
    scan_profile: str
    sorted_zones: list[Zone] = field(default_factory=list)
    anchor_points_px: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    zone_bounds_px: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    zone_centers_px: dict[str, np.ndarray] = field(default_factory=dict)
    zone_bubble_radius: dict[str, int] = field(default_factory=dict)
    identifier_zone_ids: set[str] = field(default_factory=set)


class OMRProcessor:
    def __init__(
        self,
        fill_threshold: float = 0.45,
        empty_threshold: float = 0.20,
        certainty_margin: float = 0.08,
        debug_mode: bool = False,
        debug_dir: str | Path | None = None,
    ):
        self.fill_threshold = fill_threshold
        self.empty_threshold = empty_threshold
        self.certainty_margin = certainty_margin
        self.debug_mode = debug_mode
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self._mask_cache: dict[int, tuple[np.ndarray, float]] = {}
        self.alignment_profile: str = "auto"
        self.standard_width: int = 1200
        self._last_alignment_debug: dict[str, object] = {}
        self.processing_time_limit_sec: float = 8.0
        self._processing_deadline_monotonic: float | None = None
        self._batch_worker_local = threading.local()
        self._batch_orientation_hint: int | None = None
        self._batch_orientation_hint_score: float | None = None
        self._template_anchor_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        self._template_plan_cache: dict[tuple[int, int, int, int, int], TemplateRuntimePlan] = {}
        self._scanner_locked_plan_cache: dict[tuple[int, int, int, int, int], dict[str, object]] = {}
        self._active_point_shift: tuple[float, float] = (0.0, 0.0)

    def _time_budget_exceeded(self) -> bool:
        deadline = self._processing_deadline_monotonic
        return bool(deadline and time.monotonic() >= deadline)

    @staticmethod
    def _zone_recognition_priority(zone: Zone) -> tuple[int, str]:
        zone_type = getattr(zone, "zone_type", None)
        if zone_type == ZoneType.MCQ_BLOCK:
            return (0, str(getattr(zone, "id", "") or ""))
        if zone_type == ZoneType.TRUE_FALSE_BLOCK:
            return (1, str(getattr(zone, "id", "") or ""))
        if zone_type == ZoneType.NUMERIC_BLOCK:
            return (2, str(getattr(zone, "id", "") or ""))
        if zone_type == ZoneType.STUDENT_ID_BLOCK:
            return (3, str(getattr(zone, "id", "") or ""))
        if zone_type == ZoneType.EXAM_CODE_BLOCK:
            return (4, str(getattr(zone, "id", "") or ""))
        return (5, str(getattr(zone, "id", "") or ""))

    @staticmethod
    def _identifier_recognition_enabled(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        return bool(meta.get("enable_identifier_recognition", False))

    @staticmethod
    def _is_fast_200dpi_mode(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        return bool(meta.get("fast_200dpi_mode", False))

    @staticmethod
    def _is_fast_scan_mode(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        return bool(meta.get("fast_scan_mode", False) or meta.get("fast_200dpi_mode", False))

    @staticmethod
    def _should_skip_rotation(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        return bool(meta.get("skip_rotation_for_scanner", False))

    @staticmethod
    def _should_skip_heavy_identifier_fallback(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        return bool(meta.get("skip_heavy_identifier_fallback", False))

    @staticmethod
    def _poor_image_fast_fail_enabled(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        raw = meta.get("poor_image_fast_fail", True)
        if isinstance(raw, bool):
            return raw
        return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _allow_heavy_identifier_retry_on_poor_image(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        raw = meta.get("allow_heavy_identifier_retry_on_poor_image", False)
        if isinstance(raw, bool):
            return raw
        return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _poor_image_quality_threshold(template: Template) -> float:
        meta = getattr(template, "metadata", {}) or {}
        raw = meta.get("poor_image_quality_threshold", 0.42)
        try:
            return float(raw)
        except Exception:
            return 0.42

    @staticmethod
    def _identifier_timeout_sec(template: Template) -> float:
        meta = getattr(template, "metadata", {}) or {}
        raw = meta.get("identifier_timeout_sec", 2.0)
        try:
            value = float(raw)
        except Exception:
            value = 2.0
        return max(0.5, value)

    @staticmethod
    def _identifier_timeout_sec_fast(template: Template) -> float:
        meta = getattr(template, "metadata", {}) or {}
        raw = meta.get("identifier_timeout_sec_fast", 0.25)
        try:
            value = float(raw)
        except Exception:
            value = 0.25
        return max(0.08, value)

    @staticmethod
    def _max_identifier_fallback_steps(template: Template, fast_mode: bool) -> int:
        meta = getattr(template, "metadata", {}) or {}
        default_steps = 1 if fast_mode else 3
        raw = meta.get("max_identifier_fallback_steps", default_steps)
        try:
            return max(0, int(raw))
        except Exception:
            return default_steps

    @staticmethod
    def _should_force_grayscale_load(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        return bool(meta.get("force_grayscale_load", False))

    @staticmethod
    def _use_shape_based_dpi_normalization(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        return bool(meta.get("use_shape_based_dpi_normalization", False))

    @staticmethod
    def _fast_alignment_profile(template: Template) -> str:
        meta = getattr(template, "metadata", {}) or {}
        val = str(meta.get("fast_alignment_profile", "border") or "border").strip().lower()
        return val if val in {"border", "one_side", "legacy", "hybrid"} else "border"

    @staticmethod
    def _scan_profile(template: Template) -> str:
        meta = getattr(template, "metadata", {}) or {}
        return str(meta.get("scan_profile", "") or "").strip().lower()

    @staticmethod
    def _template_match_mode(template: Template) -> str:
        meta = getattr(template, "metadata", {}) or {}
        return str(meta.get("template_match_mode", "generic") or "generic").strip().lower()

    @staticmethod
    def _scanner_locked_mode(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        raw = meta.get("scanner_locked_mode", False)
        if isinstance(raw, bool):
            return raw
        return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _scanner_locked_allow_full_fallback(template: Template) -> bool:
        meta = getattr(template, "metadata", {}) or {}
        raw = meta.get("scanner_locked_allow_full_fallback", True)
        if isinstance(raw, bool):
            return raw
        return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _scanner_locked_max_shift_px(template: Template) -> float:
        meta = getattr(template, "metadata", {}) or {}
        raw = meta.get("scanner_locked_max_shift_px", 24.0)
        try:
            return max(4.0, float(raw))
        except Exception:
            return 24.0

    def _is_fixed_scan_profile(self, template: Template) -> bool:
        profile = self._scan_profile(template)
        match_mode = self._template_match_mode(template)
        return profile.startswith("fixed_") or match_mode == "precompiled_anchors"

    def _template_plan_key(self, template: Template) -> tuple[int, int, int, int, int]:
        return (id(template), int(template.width), int(template.height), len(template.zones), len(template.anchors))

    def _build_template_runtime_plan(self, template: Template) -> TemplateRuntimePlan:
        key = self._template_plan_key(template)
        sorted_zones = sorted(template.zones, key=self._zone_recognition_priority)
        zone_bounds_px: dict[str, tuple[int, int, int, int]] = {}
        zone_centers_px: dict[str, np.ndarray] = {}
        zone_bubble_radius: dict[str, int] = {}
        identifier_zone_ids: set[str] = set()
        for zone in sorted_zones:
            x = int(round((zone.x * template.width) if zone.x <= 1.0 else zone.x))
            y = int(round((zone.y * template.height) if zone.y <= 1.0 else zone.y))
            w = int(round((zone.width * template.width) if zone.width <= 1.0 else zone.width))
            h = int(round((zone.height * template.height) if zone.height <= 1.0 else zone.height))
            zone_bounds_px[zone.id] = (x, y, w, h)
            positions = getattr(getattr(zone, "grid", None), "bubble_positions", None) or []
            if positions:
                centers = np.array(
                    [
                        (bx * template.width, by * template.height) if bx <= 1.0 and by <= 1.0 else (bx, by)
                        for bx, by in positions
                    ],
                    dtype=np.float32,
                )
            else:
                centers = np.empty((0, 2), dtype=np.float32)
            zone_centers_px[zone.id] = centers
            zone_bubble_radius[zone.id] = int(zone.metadata.get("bubble_radius", 9))
            if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
                identifier_zone_ids.add(zone.id)
        return TemplateRuntimePlan(
            key=key,
            target_width=int(template.width),
            target_height=int(template.height),
            alignment_profile=self._fast_alignment_profile(template),
            template_match_mode=self._template_match_mode(template),
            scan_profile=self._scan_profile(template),
            sorted_zones=sorted_zones,
            anchor_points_px=self._template_anchor_points_px(template),
            zone_bounds_px=zone_bounds_px,
            zone_centers_px=zone_centers_px,
            zone_bubble_radius=zone_bubble_radius,
            identifier_zone_ids=identifier_zone_ids,
        )

    def _get_template_runtime_plan(self, template: Template) -> TemplateRuntimePlan:
        key = self._template_plan_key(template)
        cached = self._template_plan_cache.get(key)
        if cached is not None:
            return cached
        plan = self._build_template_runtime_plan(template)
        self._template_plan_cache[key] = plan
        return plan

    def _build_scanner_locked_plan(self, template: Template) -> dict[str, object]:
        plan = self._get_template_runtime_plan(template)
        reference_gray, reference_scan_path = self._extract_reference_scan_gray(template)
        registration_patches = self._build_registration_patches(template, reference_gray)
        reference_digit_points_px: dict[str, np.ndarray] = {}
        for zone in plan.sorted_zones:
            if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
                reference_digit_points_px[zone.id] = np.array(plan.zone_centers_px.get(zone.id, np.empty((0, 2), dtype=np.float32)), dtype=np.float32)
        anchor_expected_px: dict[str, np.ndarray] = {}
        for idx, anc in enumerate(list(template.anchors or [])):
            name = str(getattr(anc, "name", "") or f"A{idx+1}")
            x = float(anc.x * template.width if anc.x <= 1.0 else anc.x)
            y = float(anc.y * template.height if anc.y <= 1.0 else anc.y)
            anchor_expected_px[name] = np.array([x, y], dtype=np.float32)
        strip_anchor_groups = {
            "header": ["A1", "A2", "A5", "A3"],
            "mcq": ["A5", "A3", "A6", "A4"],
            "tf": ["A6", "A4", "A7", "A10"],
            "numeric": ["A7", "A10", "A8", "A9"],
        }
        return {
            "calibrated_reference_size": (int(template.width), int(template.height)),
            "reference_scan_path": reference_scan_path,
            "reference_gray": reference_gray,
            "reference_zone_bounds_px": dict(plan.zone_bounds_px),
            "reference_zone_centers_px": {k: np.array(v, dtype=np.float32) for k, v in plan.zone_centers_px.items()},
            "reference_digit_points_px": reference_digit_points_px,
            "registration_patches": registration_patches,
            "per_zone_radius": dict(plan.zone_bubble_radius),
            "identifier_zone_ids": set(plan.identifier_zone_ids),
            "scanner_locked_registration_mode": str((template.metadata or {}).get("scanner_locked_registration_mode", "translation_first") or "translation_first"),
            "scanner_locked_use_affine_if_needed": bool((template.metadata or {}).get("scanner_locked_use_affine_if_needed", True)),
            "max_registration_patches": int((template.metadata or {}).get("scanner_locked_max_registration_patches", 6) or 6),
            # scanner-locked plan: precompiled anchors and strip groups for local transform.
            "anchor_expected_px": anchor_expected_px,
            "digit_anchor_expected_px": dict(anchor_expected_px),
            "strip_anchor_groups": strip_anchor_groups,
        }

    def _get_scanner_locked_plan(self, template: Template) -> dict[str, object]:
        key = self._template_plan_key(template)
        cached = self._scanner_locked_plan_cache.get(key)
        if cached is not None:
            return cached
        built = self._build_scanner_locked_plan(template)
        self._scanner_locked_plan_cache[key] = built
        return built

    @staticmethod
    def _scanner_locked_plan_ready(scanner_plan: dict[str, object]) -> bool:
        ref = scanner_plan.get("reference_gray", None)
        patches = list(scanner_plan.get("registration_patches", []) or [])
        return isinstance(ref, np.ndarray) and ref.size > 0 and len(patches) >= 2

    def _preprocess_fast(self, image: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(blur, 60, 160)
        return {"gray": gray, "blur": blur, "binary": binary, "edges": edges}

    @staticmethod
    def _append_issue_once(result: OMRResult, code: str, message: str, zone_id: str | None = None) -> None:
        code_u = str(code or "").strip().upper()
        zid = str(zone_id or "").strip()
        for issue in (result.issues or []):
            if str(getattr(issue, "code", "") or "").strip().upper() == code_u and str(getattr(issue, "zone_id", "") or "").strip() == zid:
                return
        result.issues.append(OMRIssue(code_u, message, zone_id=zone_id))

    def _estimate_image_quality(
        self,
        aligned_image: np.ndarray,
        aligned_binary: np.ndarray,
        template: Template,
    ) -> dict[str, object]:
        md = getattr(template, "metadata", {}) or {}
        threshold = self._poor_image_quality_threshold(template)
        match_count = int(self._last_alignment_debug.get("used_anchor_match_count", self._last_alignment_debug.get("fast_match_count", 0)) or 0)
        mean_error = float(self._last_alignment_debug.get("fast_mean_error", 16.0) or 16.0)
        align_score = float(self._last_alignment_debug.get("alignment_score", 0.0) or 0.0)
        h, w = aligned_binary.shape[:2]
        header_zone = aligned_binary[: max(1, int(h * 0.22)), :]
        header_density = float(np.mean(header_zone > 0)) if header_zone.size else 0.0
        blur_var = 0.0
        try:
            gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY) if len(aligned_image.shape) == 3 else aligned_image
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            blur_var = 0.0
        id_zones = [z for z in template.zones if z.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK)]
        usable_ratios: list[float] = []
        for z in id_zones:
            centers = self._resolve_zone_centers(aligned_binary, z, template)
            if centers is None or len(centers) == 0:
                continue
            within = (centers[:, 0] >= 1) & (centers[:, 0] < w - 1) & (centers[:, 1] >= 1) & (centers[:, 1] < h - 1)
            usable_ratios.append(float(np.mean(within)))
        usable_ratio = float(np.mean(usable_ratios)) if usable_ratios else 0.0

        anchor_component = min(1.0, max(0.0, match_count / 6.0))
        err_component = min(1.0, max(0.0, (18.0 - mean_error) / 18.0))
        blur_component = min(1.0, max(0.0, blur_var / 120.0))
        align_component = min(1.0, max(0.0, align_score / 180.0))
        quality_score = float(
            (0.28 * anchor_component)
            + (0.20 * err_component)
            + (0.18 * align_component)
            + (0.16 * min(1.0, max(0.0, header_density / 0.22)))
            + (0.10 * usable_ratio)
            + (0.08 * blur_component)
        )
        poor_scan = quality_score < threshold
        poor_identifier_zone = bool(usable_ratio < 0.55 or header_density < 0.04)
        reason_bits: list[str] = []
        if match_count < 3:
            reason_bits.append("low_anchor_match")
        if mean_error > 16.0:
            reason_bits.append("high_alignment_error")
        if header_density < 0.04:
            reason_bits.append("weak_header_density")
        if usable_ratio < 0.55:
            reason_bits.append("weak_identifier_centers")
        if blur_var < 18.0:
            reason_bits.append("high_blur")
        if not reason_bits:
            reason_bits.append("ok")
        return {
            "quality_score": quality_score,
            "poor_scan": bool(poor_scan),
            "poor_identifier_zone": bool(poor_identifier_zone),
            "reason": ",".join(reason_bits),
            "threshold": float(md.get("poor_image_quality_threshold", threshold) or threshold),
            "blur_var": float(blur_var),
            "header_density": float(header_density),
            "usable_ratio": float(usable_ratio),
            "match_count": int(match_count),
            "mean_error": float(mean_error),
            "align_score": float(align_score),
        }

    @staticmethod
    def _has_meaningful_result(result: OMRResult) -> bool:
        sid = str(getattr(result, "student_id", "") or "").strip()
        code = str(getattr(result, "exam_code", "") or "").strip()
        has_identity = bool((sid and sid != "-") or (code and code != "-"))
        return has_identity or bool((result.mcq_answers or {}) or (result.true_false_answers or {}) or (result.numeric_answers or {}))

    def _extract_reference_scan_gray(self, template: Template) -> tuple[np.ndarray | None, str]:
        meta = getattr(template, "metadata", {}) or {}
        raw_path = (
            meta.get("scanner_locked_reference_scan_path")
            or meta.get("reference_scan_path")
            or meta.get("scanner_reference_scan")
            or ""
        )
        scan_path = str(raw_path or "").strip()
        if not scan_path:
            return None, ""
        candidates = [Path(scan_path)]
        if not Path(scan_path).is_absolute():
            candidates.append(Path.cwd() / scan_path)
        image_gray = None
        used_path = ""
        for cand in candidates:
            if not cand.exists():
                continue
            loaded = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
            if loaded is None:
                continue
            image_gray = loaded
            used_path = str(cand)
            break
        if image_gray is None:
            return None, scan_path
        if image_gray.shape[1] != int(template.width) or image_gray.shape[0] != int(template.height):
            image_gray = cv2.resize(image_gray, (int(template.width), int(template.height)), interpolation=cv2.INTER_LINEAR)
        return image_gray, used_path

    @staticmethod
    def _zone_band_name(zone: Zone) -> str:
        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            return "header"
        if zone.zone_type == ZoneType.MCQ_BLOCK:
            return "mcq"
        if zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
            return "tf"
        if zone.zone_type == ZoneType.NUMERIC_BLOCK:
            return "numeric"
        return "other"

    def _build_registration_patches(self, template: Template, reference_gray: np.ndarray | None) -> list[dict[str, object]]:
        if reference_gray is None:
            return []
        plan = self._get_template_runtime_plan(template)
        zone_bounds = dict(plan.zone_bounds_px)
        groups: dict[str, list[tuple[int, int, int, int]]] = {"header": [], "mcq": [], "tf": [], "numeric": []}
        for zone in plan.sorted_zones:
            if zone.id not in zone_bounds:
                continue
            band = self._zone_band_name(zone)
            if band in groups:
                groups[band].append(tuple(zone_bounds[zone.id]))
        def union_rect(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
            if not rects:
                return None
            x0 = min(r[0] for r in rects)
            y0 = min(r[1] for r in rects)
            x1 = max(r[2] for r in rects)
            y1 = max(r[3] for r in rects)
            return (x0, y0, x1, y1)
        bands = {k: union_rect(v) for k, v in groups.items()}
        h, w = reference_gray.shape[:2]
        patch_specs: list[tuple[str, str, tuple[int, int, int, int], float]] = []
        if bands.get("header"):
            x0, y0, x1, y1 = bands["header"]  # type: ignore[index]
            pw = max(40, int((x1 - x0) * 0.28))
            ph = max(36, int((y1 - y0) * 0.55))
            patch_specs.append(("header_left_patch", "header", (x0, y0, pw, ph), 1.2))
            patch_specs.append(("header_right_patch", "header", (max(x0, x1 - pw), y0, pw, ph), 1.2))
        if bands.get("mcq"):
            x0, y0, x1, y1 = bands["mcq"]  # type: ignore[index]
            pw = max(48, int((x1 - x0) * 0.20))
            ph = max(48, int((y1 - y0) * 0.20))
            patch_specs.append(("mcq_left_patch", "mcq", (x0, y0, pw, ph), 1.0))
            patch_specs.append(("mcq_right_patch", "mcq", (max(x0, x1 - pw), y0, pw, ph), 1.0))
        if bands.get("tf"):
            x0, y0, x1, y1 = bands["tf"]  # type: ignore[index]
            pw = max(48, int((x1 - x0) * 0.22))
            ph = max(42, int((y1 - y0) * 0.22))
            patch_specs.append(("tf_patch", "tf", (x0, y0, pw, ph), 0.9))
        if bands.get("numeric"):
            x0, y0, x1, y1 = bands["numeric"]  # type: ignore[index]
            pw = max(48, int((x1 - x0) * 0.22))
            ph = max(42, int((y1 - y0) * 0.22))
            patch_specs.append(("numeric_patch", "numeric", (x0, y0, pw, ph), 0.9))
        fallback_specs = [
            ("fallback_top_center", "header", (int(w * 0.36), int(h * 0.04), int(w * 0.20), int(h * 0.08)), 0.5),
            ("fallback_mid_center", "mcq", (int(w * 0.35), int(h * 0.38), int(w * 0.20), int(h * 0.10)), 0.5),
        ]
        patch_specs.extend(fallback_specs)
        patches: list[dict[str, object]] = []
        margin_default = int((template.metadata or {}).get("scanner_locked_patch_search_margin", 24) or 24)
        for name, band, (x, y, pw, ph), weight in patch_specs:
            x = int(np.clip(x, 0, max(0, w - 2)))
            y = int(np.clip(y, 0, max(0, h - 2)))
            pw = int(np.clip(pw, 16, max(16, w - x)))
            ph = int(np.clip(ph, 16, max(16, h - y)))
            patch = reference_gray[y : y + ph, x : x + pw]
            if patch.size == 0:
                continue
            patches.append(
                {
                    "name": name,
                    "band": band,
                    "bbox": [int(x), int(y), int(pw), int(ph)],
                    "search_margin": int(margin_default),
                    "weight": float(weight),
                    "image": patch.copy(),
                    "center_ref": np.array([x + (pw * 0.5), y + (ph * 0.5)], dtype=np.float32),
                }
            )
        return patches

    def _match_registration_patch(self, image_gray: np.ndarray, patch_cfg: dict[str, object]) -> dict[str, object]:
        ref_patch = patch_cfg.get("image")
        if not isinstance(ref_patch, np.ndarray) or ref_patch.size == 0:
            return {"success": False, "reason": "empty_patch", "name": patch_cfg.get("name", "")}
        h, w = image_gray.shape[:2]
        x, y, pw, ph = [int(v) for v in (patch_cfg.get("bbox") or [0, 0, 0, 0])]
        margin = int(patch_cfg.get("search_margin", 24) or 24)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w, x + pw + margin)
        y1 = min(h, y + ph + margin)
        search = image_gray[y0:y1, x0:x1]
        if search.size == 0 or search.shape[0] < ph or search.shape[1] < pw:
            return {"success": False, "reason": "search_small", "name": patch_cfg.get("name", "")}
        res = cv2.matchTemplate(search, ref_patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        ox = float(x0 + max_loc[0] + (pw * 0.5))
        oy = float(y0 + max_loc[1] + (ph * 0.5))
        ref_center = np.array([x + (pw * 0.5), y + (ph * 0.5)], dtype=np.float32)
        success = bool(max_val >= 0.20)
        return {
            "success": success,
            "name": str(patch_cfg.get("name", "")),
            "band": str(patch_cfg.get("band", "")),
            "score": float(max_val),
            "weight": float(patch_cfg.get("weight", 1.0) or 1.0),
            "center_ref": ref_center,
            "center_obs": np.array([ox, oy], dtype=np.float32),
            "dx": float(ox - ref_center[0]),
            "dy": float(oy - ref_center[1]),
            "reason": "" if success else "low_score",
        }

    def _estimate_scanner_locked_transform(
        self,
        image_gray: np.ndarray,
        scanner_plan: dict[str, object],
    ) -> dict[str, object]:
        match_started = time.perf_counter()
        patches = list(scanner_plan.get("registration_patches", []) or [])
        max_patches = int(((scanner_plan.get("max_registration_patches", 6)) or 6))
        if len(patches) > max_patches:
            patches = patches[:max_patches]
        patch_matches: list[dict[str, object]] = []
        for cfg in patches:
            patch_matches.append(self._match_registration_patch(image_gray, cfg))
        registration_patch_time_sec = float(time.perf_counter() - match_started)
        valid = [m for m in patch_matches if bool(m.get("success", False))]
        valid_sorted = sorted(valid, key=lambda m: float(m.get("score", 0.0) or 0.0), reverse=True)
        fit_started = time.perf_counter()
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        inlier_count = 0
        use_affine = bool(scanner_plan.get("scanner_locked_use_affine_if_needed", True))
        mode = str(scanner_plan.get("scanner_locked_registration_mode", "translation_first") or "translation_first").strip().lower()
        if len(valid_sorted) >= 2 and mode == "translation_first":
            top = valid_sorted[: min(4, len(valid_sorted))]
            dx = float(np.mean([float(v.get("dx", 0.0) or 0.0) for v in top]))
            dy = float(np.mean([float(v.get("dy", 0.0) or 0.0) for v in top]))
            matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            inlier_count = len(top)
            mean_score = float(np.mean([float(v.get("score", 0.0) or 0.0) for v in top])) if top else 0.0
            if use_affine and (mean_score < 0.55):
                src_pts = np.array([m["center_ref"] for m in valid_sorted], dtype=np.float32)
                dst_pts = np.array([m["center_obs"] for m in valid_sorted], dtype=np.float32)
                est, inliers = cv2.estimateAffinePartial2D(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=400,
                )
                if est is not None:
                    matrix = est.astype(np.float32)
                    inlier_count = int(np.sum(inliers)) if inliers is not None else len(valid_sorted)
        elif len(valid_sorted) >= 2:
            src_pts = np.array([m["center_ref"] for m in valid_sorted], dtype=np.float32)
            dst_pts = np.array([m["center_obs"] for m in valid_sorted], dtype=np.float32)
            est, inliers = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=400,
            )
            if est is not None:
                matrix = est.astype(np.float32)
                inlier_count = int(np.sum(inliers)) if inliers is not None else len(valid_sorted)
            else:
                dx = float(np.mean([float(v.get("dx", 0.0) or 0.0) for v in valid_sorted]))
                dy = float(np.mean([float(v.get("dy", 0.0) or 0.0) for v in valid_sorted]))
                matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
                inlier_count = len(valid_sorted)
        elif len(valid_sorted) == 1:
            dx = float(valid_sorted[0].get("dx", 0.0) or 0.0)
            dy = float(valid_sorted[0].get("dy", 0.0) or 0.0)
            matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            inlier_count = 1
        transform_fit_time_sec = float(time.perf_counter() - fit_started)
        theta_deg = float(np.degrees(np.arctan2(float(matrix[1, 0]), float(matrix[0, 0]))))
        scale = float(np.sqrt((float(matrix[0, 0]) ** 2) + (float(matrix[1, 0]) ** 2)))
        success = len(valid_sorted) >= 2 and inlier_count >= 2
        scores = [float(m.get("score", 0.0) or 0.0) for m in patch_matches]
        patch_score_mean = float(np.mean(scores)) if scores else 0.0
        patch_score_min = float(np.min(scores)) if scores else 0.0
        patch_score_max = float(np.max(scores)) if scores else 0.0
        per_patch = [
            {
                "name": str(m.get("name", "")),
                "band": str(m.get("band", "")),
                "score": float(m.get("score", 0.0) or 0.0),
                "success": bool(m.get("success", False)),
                "dx": float(m.get("dx", 0.0) or 0.0),
                "dy": float(m.get("dy", 0.0) or 0.0),
                "reason": str(m.get("reason", "") or ""),
            }
            for m in patch_matches
        ]
        return {
            "success": bool(success),
            "matrix": matrix,
            "patch_matches": patch_matches,
            "valid_patch_count": int(len(valid_sorted)),
            "inlier_count": int(inlier_count),
            "patch_total_count": int(len(patch_matches)),
            "patch_score_mean": float(patch_score_mean),
            "patch_score_min": float(patch_score_min),
            "patch_score_max": float(patch_score_max),
            "per_patch": per_patch,
            "registration_patch_time_sec": registration_patch_time_sec,
            "transform_fit_time_sec": transform_fit_time_sec,
            "dtheta_deg": theta_deg,
            "scale": scale,
        }

    @staticmethod
    def _transform_points_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.transform(pts, np.asarray(matrix, dtype=np.float32)).reshape(-1, 2)
        return out.astype(np.float32)

    @staticmethod
    def _transform_rect_affine(rect: tuple[int, int, int, int], matrix: np.ndarray) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        mapped = cv2.transform(corners.reshape(-1, 1, 2), np.asarray(matrix, dtype=np.float32)).reshape(-1, 2)
        min_xy = np.min(mapped, axis=0)
        max_xy = np.max(mapped, axis=0)
        return (int(round(min_xy[0])), int(round(min_xy[1])), int(round(max_xy[0])), int(round(max_xy[1])))

    def _locate_anchor_in_roi(self, image_gray: np.ndarray, expected_xy: np.ndarray, roi_margin: int = 26) -> dict[str, object]:
        # ROI anchor search: local threshold + contour centroid, no full-page anchor detection.
        h, w = image_gray.shape[:2]
        cx, cy = float(expected_xy[0]), float(expected_xy[1])
        m = int(max(10, roi_margin))
        x0 = max(0, int(round(cx - m)))
        y0 = max(0, int(round(cy - m)))
        x1 = min(w, int(round(cx + m + 1)))
        y1 = min(h, int(round(cy + m + 1)))
        roi = image_gray[y0:y1, x0:x1]
        if roi.size == 0:
            return {"success": False, "reason": "empty_roi", "point": np.array([cx, cy], dtype=np.float32)}
        _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"success": False, "reason": "no_blob", "point": np.array([cx, cy], dtype=np.float32)}
        best = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(best))
        if area < 9.0:
            return {"success": False, "reason": "blob_too_small", "point": np.array([cx, cy], dtype=np.float32)}
        mmt = cv2.moments(best)
        if float(mmt.get("m00", 0.0) or 0.0) <= 1e-6:
            return {"success": False, "reason": "degenerate_blob", "point": np.array([cx, cy], dtype=np.float32)}
        px = float((mmt["m10"] / mmt["m00"]) + x0)
        py = float((mmt["m01"] / mmt["m00"]) + y0)
        return {"success": True, "point": np.array([px, py], dtype=np.float32), "reason": ""}

    def _estimate_strip_transforms(
        self,
        image_gray: np.ndarray,
        scanner_plan: dict[str, object],
        base_matrix: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # strip transform: local anchor fit per band for stable scanner-locked decode.
        anchors = dict(scanner_plan.get("anchor_expected_px", {}) or {})
        groups = dict(scanner_plan.get("strip_anchor_groups", {}) or {})
        roi_margin = int(((scanner_plan.get("anchor_roi_margin", 26)) or 26))
        out: dict[str, np.ndarray] = {}
        for band, names in groups.items():
            src_pts: list[np.ndarray] = []
            dst_pts: list[np.ndarray] = []
            for nm in names:
                exp = anchors.get(nm)
                if exp is None:
                    continue
                pred = self._transform_points_affine(np.asarray([exp], dtype=np.float32), base_matrix)
                if len(pred) == 0:
                    continue
                loc = self._locate_anchor_in_roi(image_gray, pred[0], roi_margin=roi_margin)
                if not bool(loc.get("success", False)):
                    continue
                src_pts.append(np.asarray(exp, dtype=np.float32))
                dst_pts.append(np.asarray(loc.get("point"), dtype=np.float32))
            if len(src_pts) >= 2:
                est, _ = cv2.estimateAffinePartial2D(np.asarray(src_pts, dtype=np.float32), np.asarray(dst_pts, dtype=np.float32), method=cv2.LMEDS)
                out[band] = est.astype(np.float32) if est is not None else np.asarray(base_matrix, dtype=np.float32)
            else:
                out[band] = np.asarray(base_matrix, dtype=np.float32)
        return out

    def _estimate_band_transform(
        self,
        image_gray: np.ndarray,
        template: Template,
        band_name: str,
        scanner_plan: dict[str, object],
    ) -> dict[str, float | bool]:
        reg = self._estimate_scanner_locked_transform(image_gray, scanner_plan)
        matrix = np.asarray(reg.get("matrix"), dtype=np.float32)
        cx = float(template.width) * 0.5
        cy = float(template.height) * 0.5
        if str(band_name) == "header":
            cy = float(template.height) * 0.12
        elif str(band_name) == "mcq":
            cy = float(template.height) * 0.50
        elif str(band_name) == "tf":
            cy = float(template.height) * 0.65
        elif str(band_name) == "numeric":
            cy = float(template.height) * 0.80
        src = np.array([[[cx, cy]]], dtype=np.float32)
        dst = cv2.transform(src, matrix).reshape(2)
        return {
            "success": bool(reg.get("success", False)),
            "dx": float(dst[0] - cx),
            "dy": float(dst[1] - cy),
            "dtheta_deg": float(reg.get("dtheta_deg", 0.0) or 0.0),
            "scale": float(reg.get("scale", 1.0) or 1.0),
            "band": str(band_name),
            "max_shift_px": float(self._scanner_locked_max_shift_px(template)),
        }

    @staticmethod
    def _apply_band_transform_to_centers(centers: np.ndarray, transform: dict[str, float | bool]) -> np.ndarray:
        if centers.size == 0:
            return centers
        dx = float(transform.get("dx", 0.0) or 0.0)
        dy = float(transform.get("dy", 0.0) or 0.0)
        out = centers.astype(np.float32).copy()
        out[:, 0] += dx
        out[:, 1] += dy
        return out

    @staticmethod
    def _build_integral_sampler(binary: np.ndarray) -> np.ndarray:
        return cv2.integral((binary > 0).astype(np.uint8))

    @staticmethod
    def _calc_unaccounted_time(timing: dict[str, float], path_used: str = "") -> float:
        # timing accounting without double count: use path-level blocks, not nested breakdowns.
        if str(path_used) == "scanner_locked_fast":
            accounted = (
                float(timing.get("load_time_sec", 0.0) or 0.0)
                + float(timing.get("normalize_time_sec", 0.0) or 0.0)
                + float(timing.get("registration_patch_time_sec", 0.0) or 0.0)
                + float(timing.get("transform_fit_time_sec", 0.0) or 0.0)
                + float(timing.get("roi_decode_time_sec", 0.0) or 0.0)
                + float(timing.get("diagnostics_time_sec", 0.0) or 0.0)
            )
        else:
            accounted = (
                float(timing.get("load_time_sec", 0.0) or 0.0)
                + float(timing.get("normalize_time_sec", 0.0) or 0.0)
                + float(timing.get("alignment_time_sec", 0.0) or 0.0)
                + float(timing.get("diagnostics_time_sec", 0.0) or 0.0)
                + float(timing.get("scanner_locked_match_time_sec", 0.0) or 0.0)
            )
        total = float(timing.get("total_time_sec", 0.0) or 0.0)
        return float(total - accounted)

    @staticmethod
    def _sample_bubbles_fast(integral: np.ndarray, centers: np.ndarray, radius: int, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        if centers is None or len(centers) == 0:
            return np.zeros((0,), dtype=np.float32)
        r = max(1, int(radius))
        pts = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1, 2)
        xs = np.rint(pts[:, 0]).astype(np.int32)
        ys = np.rint(pts[:, 1]).astype(np.int32)
        x0 = np.clip(xs - r, 0, max(0, w - 1))
        y0 = np.clip(ys - r, 0, max(0, h - 1))
        x1 = np.clip(xs + r, 0, max(0, w - 1))
        y1 = np.clip(ys + r, 0, max(0, h - 1))
        area = np.maximum(1, (x1 - x0 + 1) * (y1 - y0 + 1)).astype(np.float32)
        sums = (
            integral[y1 + 1, x1 + 1]
            - integral[y0, x1 + 1]
            - integral[y1 + 1, x0]
            + integral[y0, x0]
        ).astype(np.float32)
        return (sums / area).astype(np.float32)

    def _decode_identifier_scanner_locked_fast(
        self,
        mat: np.ndarray,
        zone: Zone,
        result: OMRResult,
    ) -> None:
        zone_debug = dict(getattr(result, "digit_zone_debug", {}) or {})
        rows, cols = mat.shape
        col_logs: list[dict[str, object]] = []
        default_digit_map = list(range(10))
        digit_map = zone.metadata.get("digit_map", default_digit_map)
        for c in range(cols):
            col = np.asarray(mat[:, c], dtype=np.float32)
            order = np.argsort(col)[::-1] if col.size else np.array([], dtype=np.int32)
            top_i = int(order[0]) if len(order) else 0
            second_i = int(order[1]) if len(order) > 1 else top_i
            top_score = float(col[top_i]) if col.size else 0.0
            second_score = float(col[second_i]) if col.size else 0.0
            threshold_used = float(
                max(
                    self.empty_threshold + 0.10,
                    min(min(self.fill_threshold, 0.52), float(np.mean(col) + (0.5 * np.std(col))) if col.size else self.fill_threshold),
                )
            )
            status = "ok"
            if len(order) > 1 and abs(top_score - second_score) <= 0.015:
                status = "multiple_equal"
            elif top_score < threshold_used:
                status = "below_threshold"
            elif (top_score - second_score) <= 0.0:
                status = "ambiguous"
            chosen_digit = digit_map[top_i] if top_i < len(digit_map) else top_i
            col_logs.append(
                {
                    "zone_id": str(zone.id),
                    "recognition_path": "scanner_locked_direct",
                    "column_index": int(c),
                    "top_score": float(top_score),
                    "second_score": float(second_score),
                    "margin": float(top_score - second_score),
                    "threshold_used": float(threshold_used),
                    "chosen_digit": str(chosen_digit),
                    "status": str(status),
                    "fallback_attempted": False,
                    "fallback_kind": "",
                }
            )
        value, confs = self._decode_column_digits(mat, zone, zone.grid, result)
        key = "student_id" if zone.zone_type == ZoneType.STUDENT_ID_BLOCK else "exam_code"
        if key == "student_id":
            result.student_id = value
        else:
            result.exam_code = value
        result.confidence_scores[key] = float(np.mean(confs)) if confs else 0.0
        result.confidence_scores[f"{key}:{zone.id}"] = float(np.mean(confs)) if confs else 0.0
        zone_debug[zone.id] = dict(zone_debug.get(zone.id, {})) | {
            "zone_id": str(zone.id),
            "recognition_path": "scanner_locked_direct",
            "columns": col_logs,
        }
        setattr(result, "digit_zone_debug", zone_debug)

    def _decode_mcq_scanner_locked_fast(self, mat: np.ndarray, zone: Zone, result: OMRResult) -> None:
        grid = zone.grid
        if grid is None:
            return
        rows, cols = mat.shape
        labels = grid.options if grid.options else [chr(65 + i) for i in range(cols)]
        confs: list[float] = []
        q_count = min(grid.question_count or rows, rows)
        for r in range(q_count):
            qno = grid.question_start + r
            row_scores = mat[r, :]
            best_idx, confidence, reason = self._pick_best_mcq_option(row_scores, self.fill_threshold)
            if best_idx is None:
                result.recognition_errors.append(f"MCQ Q{qno}: {reason or 'uncertain'}")
                continue
            result.mcq_answers[qno] = labels[best_idx]
            confs.append(confidence)
        result.confidence_scores[f"mcq:{zone.id}"] = float(np.mean(confs)) if confs else 0.0

    def _decode_tf_scanner_locked_fast(self, mat: np.ndarray, zone: Zone, result: OMRResult) -> None:
        grid = zone.grid
        if grid is None:
            return
        qpb = int(zone.metadata.get("questions_per_block", 2))
        spq = int(zone.metadata.get("statements_per_question", 4))
        cps = int(zone.metadata.get("choices_per_statement", 2))
        labels = [chr(ord("a") + i) for i in range(spq)]
        confs: list[float] = []
        for q in range(qpb):
            qno = grid.question_start + q
            result.true_false_answers[qno] = {}
            for sidx in range(spq):
                row_idx = q * spq + sidx
                if row_idx >= mat.shape[0]:
                    break
                row = np.asarray(mat[row_idx, : max(1, cps)], dtype=np.float32)
                if row.size == 0:
                    continue
                order = np.argsort(row)[::-1]
                top_i = int(order[0])
                second_i = int(order[1]) if len(order) > 1 else top_i
                top = float(row[top_i])
                second = float(row[second_i]) if len(order) > 1 else 0.0
                if top < self.fill_threshold or (top - second) <= self.certainty_margin:
                    result.recognition_errors.append(f"TF Q{qno}{labels[sidx]}: uncertain")
                    continue
                result.true_false_answers[qno][labels[sidx]] = bool(top_i == 0)
                confs.append(top - second)
        result.confidence_scores[f"tf:{zone.id}"] = float(np.mean(confs)) if confs else 0.0

    def _decode_numeric_scanner_locked_fast(self, mat: np.ndarray, zone: Zone, result: OMRResult) -> None:
        grid = zone.grid
        if grid is None:
            return
        rows, cols = mat.shape
        digits_per_answer = int(zone.metadata.get("digits_per_answer", 3))
        question_count = int(zone.metadata.get("questions_per_block", zone.metadata.get("total_questions", 1)))
        digit_map = zone.metadata.get("digit_map", list(range(max(0, rows - 2))))
        digit_start_row = max(0, int(zone.metadata.get("digit_start_row", 3)) - 1)
        confs: list[float] = []
        for q in range(question_count):
            chars: list[str] = []
            question_base = q * digits_per_answer
            for d in range(digits_per_answer):
                c = question_base + d
                if c >= cols:
                    break
                col = np.asarray(mat[digit_start_row:, c], dtype=np.float32)
                if col.size == 0:
                    chars.append("?")
                    continue
                order = np.argsort(col)[::-1]
                top_i = int(order[0])
                second_i = int(order[1]) if len(order) > 1 else top_i
                top = float(col[top_i])
                second = float(col[second_i]) if len(order) > 1 else 0.0
                if top < self.fill_threshold or (top - second) <= self.certainty_margin:
                    chars.append("?")
                else:
                    mapped = digit_map[top_i] if top_i < len(digit_map) else top_i
                    chars.append(str(mapped))
                confs.append(max(0.0, top - second))
            result.numeric_answers[grid.question_start + q] = "".join(chars).replace("?", "")
        result.confidence_scores[f"numeric:{zone.id}"] = float(np.mean(confs)) if confs else 0.0

    def _decode_zone_scanner_locked_fast(
        self,
        binary: np.ndarray,
        integral: np.ndarray,
        zone: Zone,
        scanner_plan: dict[str, object],
        affine_matrix: np.ndarray,
        result: OMRResult,
    ) -> dict[str, object]:
        if zone.grid is None:
            return {"bubble_count": 0, "recognized_count": 0, "blank_count": 0, "uncertain_count": 0}
        ref_centers = np.array((scanner_plan.get("reference_zone_centers_px", {}) or {}).get(zone.id, np.empty((0, 2), dtype=np.float32)), dtype=np.float32)
        if ref_centers.size == 0:
            return {"bubble_count": 0, "recognized_count": 0, "blank_count": 0, "uncertain_count": 0}
        # direct ROI decode: transform zone centers only, no full-page warp in fast path.
        centers = self._transform_points_affine(ref_centers, affine_matrix)
        radius = int((scanner_plan.get("per_zone_radius", {}) or {}).get(zone.id, int(zone.metadata.get("bubble_radius", 9)) or 9))
        ratios = self._sample_bubbles_fast(integral, centers, radius, binary.shape[:2])
        rows = max(1, int(getattr(zone.grid, "rows", 1) or 1))
        cols = max(1, int(getattr(zone.grid, "cols", 1) or 1))
        need = rows * cols
        if len(ratios) < need:
            ratios = np.pad(ratios, (0, need - len(ratios)), constant_values=0.0)
        mat = ratios[:need].reshape(rows, cols)
        recognized_count = int(np.sum(mat > self.fill_threshold))
        blank_count = int(np.sum(mat < self.empty_threshold))
        uncertain_count = int(max(0, mat.size - recognized_count - blank_count))
        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            self._decode_identifier_scanner_locked_fast(mat, zone, result)
        elif zone.zone_type == ZoneType.MCQ_BLOCK:
            self._decode_mcq_scanner_locked_fast(mat, zone, result)
        elif zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
            self._decode_tf_scanner_locked_fast(mat, zone, result)
        elif zone.zone_type == ZoneType.NUMERIC_BLOCK:
            self._decode_numeric_scanner_locked_fast(mat, zone, result)
        return {
            "bubble_count": int(mat.size),
            "recognized_count": int(recognized_count),
            "blank_count": int(blank_count),
            "uncertain_count": int(uncertain_count),
        }

    def _recognize_sheet_scanner_locked(
        self,
        src: np.ndarray,
        template: Template,
        result: OMRResult,
        context: RecognitionContext,
        started: float,
        load_time_sec: float,
        normalize_time_sec: float,
        dpi_message: str,
        original_shape: tuple[int, ...],
    ) -> OMRResult:
        # template-locked registration: patch match against reference scan, then transform points only.
        scanner_plan = self._get_scanner_locked_plan(template)
        working = src
        if working.shape[1] != template.width or working.shape[0] != template.height:
            working = cv2.resize(working, (template.width, template.height), interpolation=cv2.INTER_LINEAR)
        prep = self._preprocess_fast(working)
        gray = prep["gray"]
        aligned = working
        aligned_binary = prep["binary"]
        reg_payload = self._estimate_scanner_locked_transform(gray, scanner_plan)
        registration_patch_time_sec = float(reg_payload.get("registration_patch_time_sec", 0.0) or 0.0)
        transform_fit_time_sec = float(reg_payload.get("transform_fit_time_sec", 0.0) or 0.0)
        affine_matrix = np.asarray(reg_payload.get("matrix"), dtype=np.float32)
        registration_ok = bool(reg_payload.get("success", False))
        self._active_point_shift = (0.0, 0.0)
        setattr(result, "_scanner_locked_registration_ok", bool(registration_ok))
        final_shape = tuple(int(v) for v in working.shape)
        if not registration_ok:
            # generic fallback hook: allow caller to run full pipeline only when registration fails.
            self._append_issue_once(result, "SCANNER_LOCK_FAIL", "Template-locked registration failed")
            timing = {
                "load_time_sec": float(load_time_sec),
                "normalize_time_sec": float(normalize_time_sec),
                "scanner_locked_match_time_sec": float(registration_patch_time_sec + transform_fit_time_sec),
                "registration_patch_time_sec": float(registration_patch_time_sec),
                "transform_fit_time_sec": float(transform_fit_time_sec),
                "header_decode_time_sec": 0.0,
                "mcq_decode_time_sec": 0.0,
                "tf_decode_time_sec": 0.0,
                "numeric_decode_time_sec": 0.0,
                "roi_decode_time_sec": 0.0,
                "identifier_time_sec": 0.0,
                "identifier_fallback_time_sec": 0.0,
                "diagnostics_time_sec": 0.0,
                "total_time_sec": float(time.perf_counter() - started),
            }
            timing["unaccounted_time_sec"] = self._calc_unaccounted_time(timing, "scanner_locked_fast")
            setattr(result, "alignment_debug", {
                "alignment_mode": "scanner_locked_fast",
                "recognition_mode": "scanner_locked",
                "path_used": "scanner_locked_fast",
                "scanner_locked_enabled": True,
                "registration_ok": False,
                "fallback_used": True,
                "fallback_reason": "registration_failed",
                "reference_scan_path": str(scanner_plan.get("reference_scan_path", "") or ""),
                "original_shape": tuple(int(v) for v in original_shape),
                "final_shape": final_shape,
                "dpi_message": str(dpi_message or ""),
                "patch_total_count": int(reg_payload.get("patch_total_count", 0) or 0),
                "patch_valid_count": int(reg_payload.get("valid_patch_count", 0) or 0),
                "patch_inlier_count": int(reg_payload.get("inlier_count", 0) or 0),
                "dtheta_deg": float(reg_payload.get("dtheta_deg", 0.0) or 0.0),
                "scale": float(reg_payload.get("scale", 1.0) or 1.0),
                "transform_matrix": affine_matrix.tolist(),
                "patch_score_mean": float(reg_payload.get("patch_score_mean", 0.0) or 0.0),
                "patch_score_min": float(reg_payload.get("patch_score_min", 0.0) or 0.0),
                "patch_score_max": float(reg_payload.get("patch_score_max", 0.0) or 0.0),
                "per_patch": list(reg_payload.get("per_patch", []) or []),
                "patch_matches": list(reg_payload.get("patch_matches", []) or []),
                "unaccounted_time_sec": float(timing["unaccounted_time_sec"]),
                "timing_breakdown": timing,
            })
            result.processing_time_sec = float(time.perf_counter() - started)
            result.sync_legacy_aliases()
            return result

        quality_gate = self._estimate_image_quality(aligned, aligned_binary, template)
        poor_image = bool(quality_gate.get("poor_scan", False))
        poor_identifier_zone = bool(quality_gate.get("poor_identifier_zone", False))
        poor_fast_fail = self._poor_image_fast_fail_enabled(template)
        if poor_image:
            self._append_issue_once(result, "POOR_IMAGE", "Image quality is below recognition threshold")
        if poor_identifier_zone:
            self._append_issue_once(result, "POOR_IDENTIFIER_ZONE", "Identifier area quality is weak")

        id_timeout = self._identifier_timeout_sec_fast(template)
        setattr(result, "_identifier_deadline_monotonic", float(time.monotonic() + id_timeout))
        setattr(result, "_identifier_time_sec", 0.0)
        setattr(result, "_identifier_fallback_time_sec", 0.0)
        setattr(result, "_quality_gate", dict(quality_gate))
        setattr(result, "_poor_image_fast_fail", bool(poor_fast_fail))
        setattr(result, "_fast_production_mode", True)
        setattr(result, "_debug_deep", False)

        mcq_time_sec = 0.0
        tf_time_sec = 0.0
        numeric_time_sec = 0.0
        header_time_sec = 0.0
        strip_matrices = self._estimate_strip_transforms(gray, scanner_plan, affine_matrix)
        integral = self._build_integral_sampler(aligned_binary)
        roi_decode_started = time.perf_counter()
        plan = self._get_template_runtime_plan(template)
        zone_metrics: list[dict[str, object]] = []
        for zone in plan.sorted_zones:
            if zone.zone_type == ZoneType.ANCHOR:
                continue
            if (
                zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK)
                and not self._identifier_recognition_enabled(template)
                and not bool(getattr(context, "force_identifier_recognition", False))
            ):
                continue
            z_started = time.perf_counter()
            band_name = self._zone_band_name(zone)
            zone_matrix = np.asarray(strip_matrices.get(band_name, affine_matrix), dtype=np.float32)
            zone_stat = self._decode_zone_scanner_locked_fast(
                binary=aligned_binary,
                integral=integral,
                zone=zone,
                scanner_plan=scanner_plan,
                affine_matrix=zone_matrix,
                result=result,
            )
            elapsed = float(time.perf_counter() - z_started)
            zone_metrics.append(
                {
                    "zone_id": str(zone.id),
                    "zone_type": str(getattr(zone.zone_type, "value", zone.zone_type)),
                    "bubble_count": int(zone_stat.get("bubble_count", 0) or 0),
                    "decode_time_sec": float(elapsed),
                    "recognized_count": int(zone_stat.get("recognized_count", 0) or 0),
                    "blank_count": int(zone_stat.get("blank_count", 0) or 0),
                    "uncertain_count": int(zone_stat.get("uncertain_count", 0) or 0),
                }
            )
            if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
                header_time_sec += elapsed
            elif zone.zone_type == ZoneType.MCQ_BLOCK:
                mcq_time_sec += elapsed
            elif zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
                tf_time_sec += elapsed
            elif zone.zone_type == ZoneType.NUMERIC_BLOCK:
                numeric_time_sec += elapsed
        roi_decode_time_sec = float(time.perf_counter() - roi_decode_started)
        identifier_time_sec = float(getattr(result, "_identifier_time_sec", 0.0) or 0.0)

        setattr(result, "aligned_image", aligned)
        setattr(result, "aligned_binary", aligned_binary)
        setattr(result, "detected_anchors", [])
        timing = {
            "load_time_sec": float(load_time_sec),
            "normalize_time_sec": float(normalize_time_sec),
            "scanner_locked_match_time_sec": float(registration_patch_time_sec + transform_fit_time_sec),
            "registration_patch_time_sec": float(registration_patch_time_sec),
            "transform_fit_time_sec": float(transform_fit_time_sec),
            "header_decode_time_sec": float(header_time_sec),
            "mcq_decode_time_sec": float(mcq_time_sec),
            "tf_decode_time_sec": float(tf_time_sec),
            "numeric_decode_time_sec": float(numeric_time_sec),
            "roi_decode_time_sec": float(roi_decode_time_sec),
            "mcq_time_sec": float(mcq_time_sec),
            "tf_time_sec": float(tf_time_sec),
            "numeric_time_sec": float(numeric_time_sec),
            "identifier_time_sec": float(identifier_time_sec),
            "identifier_fallback_time_sec": float(getattr(result, "_identifier_fallback_time_sec", 0.0) or 0.0),
            "diagnostics_time_sec": 0.0,
            "total_time_sec": float(time.perf_counter() - started),
            "poor_image_fast_fail": bool(poor_fast_fail),
        }
        timing["unaccounted_time_sec"] = self._calc_unaccounted_time(timing, "scanner_locked_fast")
        setattr(result, "alignment_debug", {
            "alignment_mode": "scanner_locked_fast",
            "recognition_mode": "scanner_locked",
            "path_used": "scanner_locked_fast",
            "scanner_locked_enabled": True,
            "registration_ok": True,
            "fallback_used": False,
            "fallback_reason": "",
            "reference_scan_path": str(scanner_plan.get("reference_scan_path", "") or ""),
            "original_shape": tuple(int(v) for v in original_shape),
            "final_shape": final_shape,
            "dpi_message": str(dpi_message or ""),
            "transform_matrix": affine_matrix.tolist(),
            "patch_total_count": int(reg_payload.get("patch_total_count", 0) or 0),
            "patch_valid_count": int(reg_payload.get("valid_patch_count", 0) or 0),
            "patch_inlier_count": int(reg_payload.get("inlier_count", 0) or 0),
            "dtheta_deg": float(reg_payload.get("dtheta_deg", 0.0) or 0.0),
            "scale": float(reg_payload.get("scale", 1.0) or 1.0),
            "patch_score_mean": float(reg_payload.get("patch_score_mean", 0.0) or 0.0),
            "patch_score_min": float(reg_payload.get("patch_score_min", 0.0) or 0.0),
            "patch_score_max": float(reg_payload.get("patch_score_max", 0.0) or 0.0),
            "per_patch": list(reg_payload.get("per_patch", []) or []),
            "patch_matches": list(reg_payload.get("patch_matches", []) or []),
            "quality_gate": dict(quality_gate),
            "band_locking": {
                "header": {"success": True, "matrix": np.asarray(strip_matrices.get("header", affine_matrix)).tolist()},
                "mcq": {"success": True, "matrix": np.asarray(strip_matrices.get("mcq", affine_matrix)).tolist()},
                "tf": {"success": True, "matrix": np.asarray(strip_matrices.get("tf", affine_matrix)).tolist()},
                "numeric": {"success": True, "matrix": np.asarray(strip_matrices.get("numeric", affine_matrix)).tolist()},
            },
            "quality_score": float(quality_gate.get("quality_score", 0.0) or 0.0),
            "poor_image": bool(poor_image),
            "poor_identifier_zone": bool(poor_identifier_zone),
            "quality_reason": str(quality_gate.get("reason", "") or ""),
            "blur_var": float(quality_gate.get("blur_var", 0.0) or 0.0),
            "header_density": float(quality_gate.get("header_density", 0.0) or 0.0),
            "usable_ratio": float(quality_gate.get("usable_ratio", 0.0) or 0.0),
            "match_count": int(quality_gate.get("match_count", 0) or 0),
            "mean_error": float(quality_gate.get("mean_error", 0.0) or 0.0),
            "align_score": float(quality_gate.get("align_score", 0.0) or 0.0),
            "zone_metrics": zone_metrics,
            "fast_production_mode": True,
            "diagnostics_collected": False,
            "unaccounted_time_sec": float(timing["unaccounted_time_sec"]),
            "timing_breakdown": timing,
        })
        result.processing_time_sec = float(time.perf_counter() - started)
        result.sync_legacy_aliases()
        return result

    def recognize_sheet(self, image: str | Path | np.ndarray, template: Template, context: RecognitionContext | None = None) -> OMRResult:
        started = time.perf_counter()
        image_path = str(image) if isinstance(image, (str, Path)) else "<in-memory>"
        result = OMRResult(image_path=image_path)
        context = context or RecognitionContext()
        context.reset()

        result.mcq_answers = {}
        result.true_false_answers = {}
        result.numeric_answers = {}

        template_profile = str((template.metadata or {}).get("alignment_profile", "") or "").strip().lower()
        prev_profile = self.alignment_profile
        if template_profile in {"auto", "legacy", "border", "hybrid", "one_side"}:
            self.alignment_profile = template_profile
        timeout_raw = (template.metadata or {}).get("recognition_timeout_sec", None)
        timeout_sec = float(timeout_raw) if isinstance(timeout_raw, (int, float)) else 0.0
        if timeout_sec <= 0.0:
            timeout_sec = 0.0
        context.deadline_monotonic = (time.monotonic() + max(1.0, timeout_sec)) if timeout_sec > 0.0 else 0.0
        self._processing_deadline_monotonic = context.deadline_monotonic

        try:
            fast_mode = self._is_fast_scan_mode(template)
            fixed_profile = self._is_fixed_scan_profile(template)
            fast_production_mode = bool(getattr(context, "fast_production_test", True) and not getattr(context, "debug_deep", False))
            if fast_production_mode:
                fast_mode = True
                context.collect_diagnostics = False
            anchor_detect_time_sec = 0.0
            alignment_time_sec = 0.0
            diagnostics_time_sec = 0.0
            identifier_time_sec = 0.0
            identifier_fallback_time_sec = 0.0
            load_time_sec = 0.0
            normalize_time_sec = 0.0
            scanner_locked_match_time_sec = 0.0
            roi_decode_time_sec = 0.0
            mcq_time_sec = 0.0
            tf_time_sec = 0.0
            numeric_time_sec = 0.0
            src_load_started = time.perf_counter()
            dpi_msg = ""
            if isinstance(image, np.ndarray):
                src = image.copy()
            else:
                src, dpi_msg = self._load_image_normalized_to_200_dpi(str(image), template=template)
                if dpi_msg:
                    result.issues.append(OMRIssue("DPI", dpi_msg))
                if src is None:
                    result.issues.append(OMRIssue("FILE", "Unable to load image"))
                    result.sync_legacy_aliases()
                    return result
            load_time_sec = float(time.perf_counter() - src_load_started)
            original_shape = tuple(int(v) for v in src.shape) if src is not None else tuple()

            normalize_started = time.perf_counter()
            if src.shape[1] != template.width or src.shape[0] != template.height:
                src = cv2.resize(src, (template.width, template.height), interpolation=cv2.INTER_LINEAR)
            normalize_time_sec = float(time.perf_counter() - normalize_started)
            final_shape = tuple(int(v) for v in src.shape)

            if self._scanner_locked_mode(template) and fast_production_mode:
                # force scanner-locked production path when plan has usable reference scan + patches.
                scanner_plan = self._get_scanner_locked_plan(template)
                plan_ready = self._scanner_locked_plan_ready(scanner_plan)
                if not plan_ready:
                    self._append_issue_once(result, "SCANNER_LOCK_PLAN_MISSING", "Scanner-locked plan missing reference scan or patches")
                    if not self._scanner_locked_allow_full_fallback(template):
                        setattr(result, "_scanner_locked_registration_ok", False)
                        result.alignment_debug = {
                            "path_used": "scanner_locked_fast",
                            "recognition_mode": "scanner_locked",
                            "scanner_locked_enabled": True,
                            "registration_ok": False,
                            "fallback_used": False,
                            # generic fallback reason
                            "fallback_reason": "plan_not_ready",
                            "reference_scan_path": str(scanner_plan.get("reference_scan_path", "") or ""),
                            "original_shape": tuple(int(v) for v in original_shape),
                            "final_shape": tuple(int(v) for v in final_shape),
                            "dpi_message": str(dpi_msg or ""),
                            "timing_breakdown": {
                                "load_time_sec": float(load_time_sec),
                                "normalize_time_sec": float(normalize_time_sec),
                                "registration_patch_time_sec": 0.0,
                                "transform_fit_time_sec": 0.0,
                                "header_decode_time_sec": 0.0,
                                "mcq_decode_time_sec": 0.0,
                                "tf_decode_time_sec": 0.0,
                                "numeric_decode_time_sec": 0.0,
                                "roi_decode_time_sec": 0.0,
                                "identifier_time_sec": 0.0,
                                "identifier_fallback_time_sec": 0.0,
                                "diagnostics_time_sec": 0.0,
                                "total_time_sec": float(time.perf_counter() - started),
                                "unaccounted_time_sec": float(time.perf_counter() - started - load_time_sec - normalize_time_sec),
                            },
                        }
                        result.processing_time_sec = float(time.perf_counter() - started)
                        result.sync_legacy_aliases()
                        return result
                fast_res = self._recognize_sheet_scanner_locked(
                    src=src,
                    template=template,
                    result=result,
                    context=context,
                    started=started,
                    load_time_sec=load_time_sec,
                    normalize_time_sec=normalize_time_sec,
                    dpi_message=dpi_msg,
                    original_shape=original_shape,
                )
                fast_timing = dict((getattr(fast_res, "alignment_debug", {}) or {}).get("timing_breakdown", {}) or {})
                scanner_locked_match_time_sec = float(fast_timing.get("scanner_locked_match_time_sec", fast_timing.get("registration_patch_time_sec", 0.0)) or 0.0)
                registration_ok = bool(getattr(fast_res, "_scanner_locked_registration_ok", False))
                if registration_ok or (not self._scanner_locked_allow_full_fallback(template)):
                    return fast_res
                self._append_issue_once(result, "SCANNER_LOCK_FAIL", "Scanner-locked registration failed; fallback to generic path")
                self._append_issue_once(result, "SAFE_FALLBACK_USED", "Fell back to generic recognition pipeline")

            rotated = src if ((fast_mode or fixed_profile) and self._should_skip_rotation(template)) else self._correct_rotation(src)
            if self._time_budget_exceeded():
                result.issues.append(OMRIssue("TIMEOUT", "Recognition time budget exceeded during rotation correction"))
                setattr(result, "aligned_image", rotated)
                result.processing_time_sec = time.perf_counter() - started
                result.sync_legacy_aliases()
                return result
            prepared = self._preprocess_fast(rotated) if fast_mode else self._preprocess(rotated)
            if self._time_budget_exceeded():
                result.issues.append(OMRIssue("TIMEOUT", "Recognition time budget exceeded during preprocessing"))
                setattr(result, "aligned_image", rotated)
                setattr(result, "aligned_binary", prepared.get("binary"))
                result.processing_time_sec = time.perf_counter() - started
                result.sync_legacy_aliases()
                return result

            align_started = time.perf_counter()
            aligned, aligned_binary = self.correct_perspective(rotated, prepared["binary"], template, result)
            alignment_time_sec = float(time.perf_counter() - align_started)
            anchor_detect_time_sec = float((self._last_alignment_debug.get("anchor_detect_time_sec", 0.0) or 0.0))
            if self._time_budget_exceeded():
                result.issues.append(OMRIssue("TIMEOUT", "Recognition time budget exceeded during alignment"))
                setattr(result, "aligned_image", aligned)
                setattr(result, "aligned_binary", aligned_binary if aligned_binary is not None else prepared.get("binary"))
                result.processing_time_sec = time.perf_counter() - started
                result.sync_legacy_aliases()
                return result
            if aligned_binary is None:
                aligned_binary = self._preprocess(aligned)["binary"]

            quality_gate = self._estimate_image_quality(aligned, aligned_binary, template)
            poor_image = bool(quality_gate.get("poor_scan", False))
            poor_identifier_zone = bool(quality_gate.get("poor_identifier_zone", False))
            quality_reason = str(quality_gate.get("reason", "") or "")
            poor_fast_fail = self._poor_image_fast_fail_enabled(template)
            if poor_image:
                self._append_issue_once(result, "POOR_IMAGE", "Image quality is below recognition threshold")
            if poor_identifier_zone:
                self._append_issue_once(result, "POOR_IDENTIFIER_ZONE", "Identifier area quality is weak")

            id_timeout = self._identifier_timeout_sec_fast(template) if fast_production_mode else self._identifier_timeout_sec(template)
            identifier_timeout_deadline = time.monotonic() + id_timeout
            setattr(result, "_identifier_deadline_monotonic", float(identifier_timeout_deadline))
            setattr(result, "_identifier_time_sec", 0.0)
            setattr(result, "_identifier_fallback_time_sec", 0.0)
            setattr(result, "_quality_gate", dict(quality_gate))
            setattr(result, "_poor_image_fast_fail", bool(poor_fast_fail))
            setattr(result, "_fast_production_mode", bool(fast_production_mode))
            setattr(result, "_debug_deep", bool(getattr(context, "debug_deep", False)))

            debug_overlay = aligned.copy() if self.debug_mode else None
            if debug_overlay is not None:
                self._draw_alignment_debug(debug_overlay, template)

            plan = self._get_template_runtime_plan(template)
            zone_metrics: list[dict[str, object]] = []
            for zone in plan.sorted_zones:
                if self._time_budget_exceeded():
                    result.issues.append(OMRIssue("TIMEOUT", "Recognition time budget exceeded; skipped remaining zones"))
                    break
                if zone.zone_type == ZoneType.ANCHOR:
                    continue
                if (
                    zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK)
                    and not self._identifier_recognition_enabled(template)
                    and not bool(getattr(context, "force_identifier_recognition", False))
                ):
                    continue
                if zone.grid is not None:
                    context.semantic_grids[zone.id] = zone.grid
                z_started = time.perf_counter()
                err_before = len(result.recognition_errors)
                mcq_before = len(result.mcq_answers)
                tf_before = sum(len(v or {}) for v in result.true_false_answers.values())
                num_before = len(result.numeric_answers)
                self.recognize_block(aligned_binary, zone, template, result, debug_overlay)
                z_elapsed = float(time.perf_counter() - z_started)
                mcq_after = len(result.mcq_answers)
                tf_after = sum(len(v or {}) for v in result.true_false_answers.values())
                num_after = len(result.numeric_answers)
                rec_count = max(0, (mcq_after - mcq_before) + (tf_after - tf_before) + (num_after - num_before))
                uncertain_count = max(0, len(result.recognition_errors) - err_before)
                bubble_count = int((zone.grid.rows * zone.grid.cols) if zone.grid is not None else 0)
                blank_count = max(0, bubble_count - rec_count - uncertain_count)
                zone_metrics.append(
                    {
                        "zone_id": str(zone.id),
                        "zone_type": str(getattr(zone.zone_type, "value", zone.zone_type)),
                        "bubble_count": int(bubble_count),
                        "decode_time_sec": float(z_elapsed),
                        "recognized_count": int(rec_count),
                        "blank_count": int(blank_count),
                        "uncertain_count": int(uncertain_count),
                    }
                )
                if zone.zone_type == ZoneType.MCQ_BLOCK:
                    mcq_time_sec += z_elapsed
                elif zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
                    tf_time_sec += z_elapsed
                elif zone.zone_type == ZoneType.NUMERIC_BLOCK:
                    numeric_time_sec += z_elapsed
                roi_decode_time_sec += z_elapsed

            if debug_overlay is not None:
                out_dir = self.debug_dir or Path(result.image_path).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{Path(result.image_path).stem}_debug.png"
                cv2.imwrite(str(out_path), debug_overlay)
                result.debug_image_path = str(out_path)

            setattr(result, "aligned_image", aligned)
            setattr(result, "aligned_binary", aligned_binary)
            setattr(result, "alignment_debug", dict(self._last_alignment_debug))
            if context.collect_diagnostics:
                diag_started = time.perf_counter()
                cached_anchors = list((self._last_alignment_debug.get("used_detected_anchors", []) or []))
                if not cached_anchors:
                    cached_anchors = list((self._last_alignment_debug.get("matched_detected_anchors", []) or []))
                if not cached_anchors and aligned_binary is not None:
                    cached_anchors = self.detect_anchors(aligned_binary, use_border_padding=True, relaxed_polygon=True, max_points=16, fast_mode=True)
                context.detected_anchors = cached_anchors
                setattr(result, "detected_anchors", context.detected_anchors)
                if self.debug_mode:
                    context.detected_digit_anchors = self._detect_digit_anchor_ruler(aligned_binary, template)
                    setattr(result, "detected_digit_anchors", context.detected_digit_anchors)
                    context.bubble_states_by_zone = self.extract_bubble_states(aligned_binary, template)
                    setattr(result, "bubble_states_by_zone", context.bubble_states_by_zone)
                diagnostics_time_sec = float(time.perf_counter() - diag_started)
            else:
                # diagnostics skip: keep production path minimal when diagnostics are disabled.
                diagnostics_time_sec = 0.0
            context.digit_zone_debug = dict(getattr(result, "digit_zone_debug", {}) or {})
            setattr(result, "digit_zone_debug", context.digit_zone_debug)

            context.recognized_answers = {
                "student_id": result.student_id,
                "exam_code": result.exam_code,
                "mcq_answers": dict(result.mcq_answers or {}),
                "true_false_answers": dict(result.true_false_answers or {}),
                "numeric_answers": dict(result.numeric_answers or {}),
            }

            result.processing_time_sec = time.perf_counter() - started
            identifier_time_sec = float(getattr(result, "_identifier_time_sec", 0.0) or 0.0)
            identifier_fallback_time_sec = float(getattr(result, "_identifier_fallback_time_sec", 0.0) or 0.0)
            if not isinstance(getattr(result, "alignment_debug", None), dict):
                setattr(result, "alignment_debug", {})
            result.alignment_debug["quality_gate"] = dict(quality_gate)
            result.alignment_debug["quality_score"] = float(quality_gate.get("quality_score", 0.0) or 0.0)
            result.alignment_debug["poor_image"] = bool(poor_image)
            result.alignment_debug["poor_identifier_zone"] = bool(poor_identifier_zone)
            result.alignment_debug["quality_reason"] = quality_reason
            result.alignment_debug["blur_var"] = float(quality_gate.get("blur_var", 0.0) or 0.0)
            result.alignment_debug["header_density"] = float(quality_gate.get("header_density", 0.0) or 0.0)
            result.alignment_debug["usable_ratio"] = float(quality_gate.get("usable_ratio", 0.0) or 0.0)
            result.alignment_debug["match_count"] = int(quality_gate.get("match_count", 0) or 0)
            result.alignment_debug["mean_error"] = float(quality_gate.get("mean_error", 0.0) or 0.0)
            result.alignment_debug["align_score"] = float(quality_gate.get("align_score", 0.0) or 0.0)
            result.alignment_debug["fast_production_mode"] = bool(fast_production_mode)
            result.alignment_debug["diagnostics_collected"] = bool(context.collect_diagnostics)
            result.alignment_debug["zone_metrics"] = zone_metrics
            path_used = "generic_fallback" if self._scanner_locked_mode(template) and fast_production_mode else "generic_fallback"
            fallback_used = bool(self._scanner_locked_mode(template) and fast_production_mode)
            fallback_reason = "registration_failed" if fallback_used else ""
            timing = {
                "load_time_sec": float(load_time_sec),
                "normalize_time_sec": float(normalize_time_sec),
                "anchor_detect_time_sec": float(anchor_detect_time_sec),
                "alignment_time_sec": float(alignment_time_sec),
                "scanner_locked_match_time_sec": float(scanner_locked_match_time_sec),
                "registration_patch_time_sec": 0.0,
                "transform_fit_time_sec": 0.0,
                "header_decode_time_sec": 0.0,
                "mcq_decode_time_sec": float(mcq_time_sec),
                "tf_decode_time_sec": float(tf_time_sec),
                "numeric_decode_time_sec": float(numeric_time_sec),
                "scanner_locked_header_time_sec": 0.0,
                "scanner_locked_mcq_time_sec": 0.0,
                "scanner_locked_tf_time_sec": 0.0,
                "scanner_locked_numeric_time_sec": 0.0,
                "roi_decode_time_sec": float(roi_decode_time_sec),
                "mcq_time_sec": float(mcq_time_sec),
                "tf_time_sec": float(tf_time_sec),
                "numeric_time_sec": float(numeric_time_sec),
                "identifier_time_sec": float(identifier_time_sec),
                "identifier_fallback_time_sec": float(identifier_fallback_time_sec),
                "diagnostics_time_sec": float(diagnostics_time_sec),
                "poor_image_fast_fail": bool(poor_fast_fail),
                "total_time_sec": float(result.processing_time_sec),
            }
            timing["unaccounted_time_sec"] = self._calc_unaccounted_time(timing, path_used)
            result.alignment_debug["path_used"] = str(path_used)
            result.alignment_debug["scanner_locked_enabled"] = bool(self._scanner_locked_mode(template))
            result.alignment_debug["registration_ok"] = bool(not fallback_used)
            result.alignment_debug["fallback_used"] = bool(fallback_used)
            result.alignment_debug["fallback_reason"] = str(fallback_reason)
            result.alignment_debug["reference_scan_path"] = str((self._get_scanner_locked_plan(template).get("reference_scan_path", "") if self._scanner_locked_mode(template) else "") or "")
            result.alignment_debug["original_shape"] = tuple(int(v) for v in original_shape)
            result.alignment_debug["final_shape"] = tuple(int(v) for v in final_shape)
            result.alignment_debug["dpi_message"] = str(dpi_msg or "")
            result.alignment_debug["patch_total_count"] = int(result.alignment_debug.get("patch_total_count", 0) or 0)
            result.alignment_debug["patch_valid_count"] = int(result.alignment_debug.get("patch_valid_count", 0) or 0)
            result.alignment_debug["patch_inlier_count"] = int(result.alignment_debug.get("patch_inlier_count", 0) or 0)
            result.alignment_debug["unaccounted_time_sec"] = float(timing["unaccounted_time_sec"])
            result.alignment_debug["timing_breakdown"] = timing
            setattr(result, "answers", result.mcq_answers)
            result.sync_legacy_aliases()
            return result
        finally:
            self.alignment_profile = prev_profile
            self._processing_deadline_monotonic = None
            self._active_point_shift = (0.0, 0.0)

    def recognize_sheet_production_fast(self, image: str | Path | np.ndarray, template: Template, context: RecognitionContext | None = None) -> OMRResult:
        # debug deep mode switch: this path forces production-fast behavior by default.
        ctx = context or RecognitionContext()
        ctx.fast_production_test = True
        ctx.debug_deep = False
        ctx.collect_diagnostics = False
        return self.recognize_sheet(image, template, ctx)

    def run_recognition_test(
        self,
        image: str | Path | np.ndarray,
        template: Template,
        context: RecognitionContext | None = None,
        *,
        fast_production_test: bool = True,
        debug_deep: bool = False,
    ) -> OMRResult:
        ctx = context or RecognitionContext()
        ctx.force_identifier_recognition = True
        ctx.fast_production_test = bool(fast_production_test)
        ctx.debug_deep = bool(debug_deep)
        if fast_production_test and not debug_deep:
            ctx.collect_diagnostics = False
        return self.recognize_sheet(image, template, ctx)

    def extract_bubble_states(self, binary_image: np.ndarray, template: Template) -> dict[str, list[bool]]:
        states_by_zone: dict[str, list[bool]] = {}
        if binary_image is None or template is None:
            return states_by_zone
        h, w = binary_image.shape[:2]
        for z in template.zones:
            if not z.grid:
                continue
            states: list[bool] = []
            centers = self._resolve_zone_centers(binary_image, z, template)
            if centers is None or len(centers) == 0:
                centers = np.array([(bx * w, by * h) if bx <= 1.0 and by <= 1.0 else (bx, by) for bx, by in z.grid.bubble_positions], dtype=np.float32)
            radius = int(z.metadata.get("bubble_radius", 9))
            ratios = self.detect_bubbles(binary_image, centers, radius)
            for ratio in ratios:
                states.append(float(ratio) >= self.fill_threshold)
            states_by_zone[z.id] = states
        return states_by_zone

    def process_image(self, image_path: str | Path, template: Template) -> OMRResult:
        return self.recognize_sheet(image_path, template, RecognitionContext())

    def _make_batch_worker(self) -> "OMRProcessor":
        worker = OMRProcessor(
            fill_threshold=self.fill_threshold,
            empty_threshold=self.empty_threshold,
            certainty_margin=self.certainty_margin,
            debug_mode=self.debug_mode,
            debug_dir=self.debug_dir,
        )
        worker.alignment_profile = self.alignment_profile
        worker.standard_width = self.standard_width
        worker.processing_time_limit_sec = self.processing_time_limit_sec
        worker._template_plan_cache = dict(self._template_plan_cache)
        return worker

    def _get_thread_batch_worker(self) -> "OMRProcessor":
        worker = getattr(self._batch_worker_local, "worker", None)
        if worker is None:
            worker = self._make_batch_worker()
            self._batch_worker_local.worker = worker
        return worker

    def _run_batch_item(self, image_path: str, template: Template):
        worker = self._get_thread_batch_worker()
        started = time.perf_counter()
        if self._scanner_locked_mode(template):
            res = worker.recognize_sheet_production_fast(image_path, template, RecognitionContext(collect_diagnostics=False))
        else:
            res = worker.run_recognition_test(image_path, template, RecognitionContext(collect_diagnostics=False))
        return res, float(time.perf_counter() - started)

    def _template_anchor_points_px(self, template: Template) -> np.ndarray:
        anchors = list(template.anchors or [])
        key = (id(template), int(template.width), int(template.height), len(anchors))
        cached = self._template_anchor_cache.get(key)
        if cached is not None:
            return cached
        pts = np.array(
            [
                (a.x * template.width, a.y * template.height) if a.x <= 1.0 and a.y <= 1.0 else (a.x, a.y)
                for a in anchors
            ],
            dtype=np.float32,
        )
        self._template_anchor_cache[key] = pts
        return pts

    @staticmethod
    def _auto_batch_worker_count(total: int, template: Template) -> int:
        meta = (getattr(template, "metadata", None) or {})
        auto_parallel = bool(meta.get("batch_auto_parallel", True))
        if not auto_parallel:
            return 1
        min_items = int(meta.get("batch_auto_parallel_min_items", 24) or 24)
        if total < max(2, min_items):
            return 1
        cpu_count = max(1, int(os.cpu_count() or 1))
        target = max(2, cpu_count // 2)
        max_auto = int(meta.get("batch_auto_parallel_max_workers", 4) or 4)
        return max(1, min(total, target, max_auto))

    @staticmethod
    def _write_batch_timing_log(log_path: Path, rows: list[dict[str, object]], total_count: int) -> None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            lines.append(
                "idx\tfile\tseconds\tpath_used\tregistration_ok\tfallback_used\tstudent_id_ok\texam_code_ok\t"
                "total_time_sec\tidentifier_time_sec\tregistration_patch_time_sec\theader_decode_time_sec\tquality_score\t"
                "patch_valid_count\tpatch_inlier_count"
            )
            for row in rows:
                lines.append(
                    f"{int(row.get('idx', 0) or 0)}\t{row.get('file', '')}\t{float(row.get('seconds', 0.0) or 0.0):.4f}\t"
                    f"{row.get('path_used', '')}\t{int(bool(row.get('registration_ok', False)))}\t{int(bool(row.get('fallback_used', False)))}\t"
                    f"{int(bool(row.get('student_id_ok', False)))}\t{int(bool(row.get('exam_code_ok', False)))}\t"
                    f"{float(row.get('total_time_sec', 0.0) or 0.0):.4f}\t{float(row.get('identifier_time_sec', 0.0) or 0.0):.4f}\t"
                    f"{float(row.get('registration_patch_time_sec', 0.0) or 0.0):.4f}\t{float(row.get('header_decode_time_sec', 0.0) or 0.0):.4f}\t"
                    f"{float(row.get('quality_score', 0.0) or 0.0):.4f}\t{int(row.get('patch_valid_count', 0) or 0)}\t"
                    f"{int(row.get('patch_inlier_count', 0) or 0)}"
                )
            valid_secs = [float(r.get("seconds", 0.0) or 0.0) for r in rows if float(r.get("seconds", 0.0) or 0.0) >= 0.0]
            if valid_secs:
                avg = float(sum(valid_secs) / len(valid_secs))
                est_500 = avg * 500.0
                est_600 = avg * 600.0
                totals = np.array([float(r.get("total_time_sec", 0.0) or 0.0) for r in rows], dtype=np.float32)
                identifier_secs = [float(r.get("identifier_time_sec", 0.0) or 0.0) for r in rows]
                reg_secs = [float(r.get("registration_patch_time_sec", 0.0) or 0.0) for r in rows]
                header_secs = [float(r.get("header_decode_time_sec", 0.0) or 0.0) for r in rows]
                success_count = sum(1 for r in rows if bool(r.get("student_id_ok", False)) and bool(r.get("exam_code_ok", False)))
                sid_fail_count = sum(1 for r in rows if not bool(r.get("student_id_ok", False)))
                exam_fail_count = sum(1 for r in rows if not bool(r.get("exam_code_ok", False)))
                generic_fallback_count = sum(1 for r in rows if bool(r.get("fallback_used", False)))
                poor_image_count = sum(1 for r in rows if bool(r.get("poor_image", False)))
                slowest = sorted(rows, key=lambda r: float(r.get("total_time_sec", 0.0) or 0.0), reverse=True)[:20]
                worst_identifier = sorted(rows, key=lambda r: float(r.get("identifier_time_sec", 0.0) or 0.0), reverse=True)[:20]
                worst_registration = sorted(rows, key=lambda r: float(r.get("registration_patch_time_sec", 0.0) or 0.0), reverse=True)[:20]
                lines.append("")
                lines.append(f"avg_seconds_per_file\t{avg:.4f}")
                lines.append(f"estimated_500_files_seconds\t{est_500:.2f}")
                lines.append(f"estimated_600_files_seconds\t{est_600:.2f}")
                lines.append(f"target_under_120_seconds_for_500\t{'YES' if est_500 <= 120.0 else 'NO'}")
                lines.append(f"success_count\t{success_count}")
                lines.append(f"student_id_fail_count\t{sid_fail_count}")
                lines.append(f"exam_code_fail_count\t{exam_fail_count}")
                lines.append(f"generic_fallback_count\t{generic_fallback_count}")
                lines.append(f"poor_image_count\t{poor_image_count}")
                lines.append(f"avg_total_sec\t{float(np.mean(totals)) if totals.size else 0.0:.4f}")
                lines.append(f"p50_total_sec\t{float(np.percentile(totals, 50)) if totals.size else 0.0:.4f}")
                lines.append(f"p90_total_sec\t{float(np.percentile(totals, 90)) if totals.size else 0.0:.4f}")
                lines.append(f"p95_total_sec\t{float(np.percentile(totals, 95)) if totals.size else 0.0:.4f}")
                lines.append(f"avg_identifier_sec\t{(sum(identifier_secs) / len(identifier_secs)) if identifier_secs else 0.0:.4f}")
                lines.append(f"avg_registration_patch_sec\t{(sum(reg_secs) / len(reg_secs)) if reg_secs else 0.0:.4f}")
                lines.append(f"avg_header_decode_sec\t{(sum(header_secs) / len(header_secs)) if header_secs else 0.0:.4f}")
                lines.append("slowest_20_files")
                for r in slowest:
                    lines.append(f"- {r.get('file', '')}\t{float(r.get('total_time_sec', 0.0) or 0.0):.4f}")
                lines.append("worst_identifier_20_files")
                for r in worst_identifier:
                    lines.append(f"- {r.get('file', '')}\t{float(r.get('identifier_time_sec', 0.0) or 0.0):.4f}")
                lines.append("worst_registration_20_files")
                for r in worst_registration:
                    lines.append(f"- {r.get('file', '')}\t{float(r.get('registration_patch_time_sec', 0.0) or 0.0):.4f}")
            lines.append(f"total_files\t{total_count}")
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _batch_timing_row(idx: int, image_path: str, seconds: float, result: OMRResult) -> dict[str, object]:
        dbg = dict(getattr(result, "alignment_debug", {}) or {})
        timing = dict(dbg.get("timing_breakdown", {}) or {})
        sid = str(getattr(result, "student_id", "") or "").strip()
        code = str(getattr(result, "exam_code", "") or "").strip()
        return {
            "idx": int(idx),
            "file": str(image_path),
            "seconds": float(seconds),
            "path_used": str(dbg.get("path_used", dbg.get("alignment_mode", "")) or ""),
            "registration_ok": bool(dbg.get("registration_ok", False)),
            "fallback_used": bool(dbg.get("fallback_used", False)),
            "student_id_ok": bool(sid and sid != "-" and "?" not in sid),
            "exam_code_ok": bool(code and code != "-" and "?" not in code),
            "total_time_sec": float(timing.get("total_time_sec", getattr(result, "processing_time_sec", 0.0)) or 0.0),
            "identifier_time_sec": float(timing.get("identifier_time_sec", 0.0) or 0.0),
            "registration_patch_time_sec": float(timing.get("registration_patch_time_sec", 0.0) or 0.0),
            "header_decode_time_sec": float(timing.get("header_decode_time_sec", 0.0) or 0.0),
            "quality_score": float(dbg.get("quality_score", 0.0) or 0.0),
            "patch_valid_count": int(dbg.get("patch_valid_count", 0) or 0),
            "patch_inlier_count": int(dbg.get("patch_inlier_count", 0) or 0),
            "poor_image": bool(dbg.get("poor_image", False)),
        }

    @staticmethod
    def _rotate_pair(image: np.ndarray, binary: np.ndarray, rotation: int, template: Template) -> tuple[np.ndarray, np.ndarray]:
        rot = int(rotation) % 360
        if rot == 90:
            img_r = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            bin_r = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            img_r = cv2.rotate(image, cv2.ROTATE_180)
            bin_r = cv2.rotate(binary, cv2.ROTATE_180)
        elif rot == 270:
            img_r = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bin_r = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img_r = image
            bin_r = binary
        if img_r.shape[1] != template.width or img_r.shape[0] != template.height:
            img_r = cv2.resize(img_r, (template.width, template.height), interpolation=cv2.INTER_LINEAR)
            bin_r = cv2.resize(bin_r, (template.width, template.height), interpolation=cv2.INTER_NEAREST)
        return img_r, bin_r

    def process_batch(self, image_paths: list[str], template: Template, progress_callback: Callable[[int, int, str], None] | None = None) -> list[OMRResult]:
        total = len(image_paths)
        if total == 0:
            return []
        self._get_template_runtime_plan(template)
        # Reset cross-file orientation hints at the start of each batch.
        # Sequential batches (single worker) will repopulate these values from
        # per-file alignment diagnostics as they progress.
        self._batch_orientation_hint = None
        self._batch_orientation_hint_score = None
        timing_rows: list[dict[str, object]] = []
        configured_workers = int(((template.metadata or {}).get("batch_workers", 0) if getattr(template, "metadata", None) else 0) or 0)
        default_workers = self._auto_batch_worker_count(total, template)
        worker_count = configured_workers if configured_workers > 0 else default_workers
        worker_count = max(1, min(worker_count, total))
        log_path_text = str((template.metadata or {}).get("batch_timing_log_path", "") or "")
        log_enabled = bool(log_path_text.strip())
        results: list[OMRResult] = []
        if worker_count <= 1:
            for idx, image_path in enumerate(image_paths, start=1):
                context = RecognitionContext()
                context.collect_diagnostics = False
                started = time.perf_counter()
                if self._scanner_locked_mode(template):
                    result = self.recognize_sheet_production_fast(image_path, template, context)
                else:
                    result = self.run_recognition_test(image_path, template, context)
                results.append(result)
                elapsed = float(time.perf_counter() - started)
                timing_rows.append(self._batch_timing_row(idx, str(image_path), elapsed, result))
                debug_payload = dict(getattr(result, "alignment_debug", {}) or {})
                orient = debug_payload.get("orientation_rotation", None)
                if isinstance(orient, (int, float)):
                    self._batch_orientation_hint = int(orient) % 360
                score = debug_payload.get("alignment_score", None)
                if isinstance(score, (int, float)):
                    self._batch_orientation_hint_score = float(score)
                if progress_callback:
                    progress_callback(idx, total, image_path)
            if log_enabled:
                self._write_batch_timing_log(Path(log_path_text), timing_rows, total)
            return results

        ordered_results: list[OMRResult | None] = [None] * total
        ordered_secs: list[float] = [0.0] * total
        completed = 0
        self._batch_worker_local = threading.local()
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_map = {
                pool.submit(self._run_batch_item, image_path, template): (idx, image_path)
                for idx, image_path in enumerate(image_paths)
            }
            for fut in as_completed(future_map):
                idx, image_path = future_map[fut]
                res, elapsed = fut.result()
                ordered_results[idx] = res
                ordered_secs[idx] = float(elapsed)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, image_path)
        for idx, image_path in enumerate(image_paths, start=1):
            res = ordered_results[idx - 1]
            if res is None:
                continue
            timing_rows.append(self._batch_timing_row(idx, str(image_path), float(ordered_secs[idx - 1]), res))
        if log_enabled:
            self._write_batch_timing_log(Path(log_path_text), timing_rows, total)
        return [res for res in ordered_results if res is not None]

    def _load_image_normalized_to_200_dpi(self, path: str, template: Template | None = None) -> tuple[np.ndarray | None, str]:
        if template is not None and self._should_force_grayscale_load(template):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return None, ""
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if self._use_shape_based_dpi_normalization(template):
                h, w = img.shape[:2]
                # A4 portrait expected near 1240x1754 (150dpi) or 1654x2339 (200dpi).
                # Fast production path: upscale only when image is close to 150dpi shape.
                if max(h, w) < 2100:
                    scale = 4.0 / 3.0
                    out = cv2.resize(
                        img,
                        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    return out, "Shape-based normalization: assumed ~150 DPI, scaled to ~200 DPI."
                return img, ""
        dpi = 0
        try:
            with Image.open(path) as im_meta:
                dpi_info = im_meta.info.get("dpi")
                if isinstance(dpi_info, tuple) and dpi_info and dpi_info[0]:
                    dpi = int(round(float(dpi_info[0])))
        except Exception:
            dpi = 0
        img = cv2.imread(path)
        if img is None:
            return None, ""
        if dpi == 200:
            return img, ""
        if dpi <= 0:
            return img, "Image DPI metadata missing; expected 200 DPI."
        scale = 200.0 / dpi
        out = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
        return out, f"Input DPI={dpi}. Normalized to 200 DPI scale."

    def _resize_for_processing(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        if w <= self.standard_width:
            return image, 1.0
        scale = self.standard_width / float(w)
        resized = cv2.resize(image, (self.standard_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(blur, 50, 150)
        return {"gray": gray, "blur": blur, "binary": binary, "edges": edges}

    def _correct_rotation(self, image: np.ndarray) -> np.ndarray:
        if self._time_budget_exceeded():
            return image
        angle = self._estimate_rotation_angle(image)
        if abs(angle) < 0.15:
            return image
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _downscale_for_rotation_estimation(image: np.ndarray, max_side: int = 1400) -> np.ndarray:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return image
        scale = max_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _estimate_rotation_angle(self, image: np.ndarray) -> float:
        probe = self._downscale_for_rotation_estimation(image)
        gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=170)
        if lines is not None and len(lines) > 0:
            angles = []
            for ln in lines[:25]:
                theta = ln[0][1]
                angle = (theta * 180.0 / np.pi) - 90.0
                if -45.0 <= angle <= 45.0:
                    angles.append(angle)
            if angles:
                return float(np.median(angles))

        if self._time_budget_exceeded():
            return 0.0
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        page = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(page)
        angle = rect[2]
        if angle < -45:
            angle += 90
        return float(angle)

    def detect_anchors(
        self,
        binary: np.ndarray,
        use_border_padding: bool = True,
        relaxed_polygon: bool = True,
        max_points: int = 40,
        fast_mode: bool = False,
    ) -> list[tuple[float, float]]:
        pad = 8 if use_border_padding else 0
        padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) if pad else binary
        contours, _ = cv2.findContours(padded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = binary.shape[:2]
        min_area = max(50, int((h * w) * 0.00003))
        border_margin = max(24.0, min(w, h) * 0.18)
        anchors: list[tuple[float, float, float]] = []
        max_keep = max(4, int(max_points or 40))
        if fast_mode:
            # fast fixed-form anchor detection: keep very few high-quality border anchors only.
            max_keep = min(max_keep, 8)
        max_candidates = max(20, max_keep * (3 if use_border_padding else 2)) if fast_mode else max(120, max_keep * (12 if use_border_padding else 8))
        if len(contours) > max_candidates:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_candidates]

        for idx, cnt in enumerate(contours):
            if (idx % 24 == 0) and self._time_budget_exceeded():
                break
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if relaxed_polygon:
                if len(approx) < 4 or len(approx) > 8:
                    continue
            elif len(approx) != 4:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            if bh == 0 or bw == 0:
                continue
            ratio = bw / float(bh)
            if not (0.75 <= ratio <= 1.25):
                continue
            rect_area = float(bw * bh)
            fill_ratio = area / rect_area if rect_area else 0.0
            if fill_ratio < 0.65:
                continue
            roi = padded[y : y + bh, x : x + bw]
            darkness = float(cv2.countNonZero(roi)) / rect_area
            if darkness < 0.50:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = (m["m10"] / m["m00"]) - float(pad)
            cy = (m["m01"] / m["m00"]) - float(pad)
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            if use_border_padding:
                near_border = min(cx, cy, (w - 1) - cx, (h - 1) - cy)
                if near_border > border_margin:
                    continue
                if fast_mode and near_border > (border_margin * 0.72):
                    continue
                border_bonus = 1.0 + max(0.0, (border_margin - near_border) / max(1.0, border_margin))
            else:
                border_bonus = 1.0
            score = area * fill_ratio * darkness * border_bonus
            anchors.append((cx, cy, score))
            if fast_mode and len(anchors) >= max_keep:
                # fast fixed-form anchor detection: stop early when enough good anchors collected.
                break

        anchors.sort(key=lambda p: p[2], reverse=True)
        if use_border_padding:
            max_keep = min(max_keep, 16 if fast_mode else 24)
        return [(x, y) for x, y, _ in anchors[:max_keep]]

    def _template_has_border_anchors(self, template: Template) -> bool:
        if not template.anchors:
            return False
        margin = 0.06
        px_margin_x = max(20.0, template.width * margin)
        px_margin_y = max(20.0, template.height * margin)
        for a in template.anchors:
            ax = a.x * template.width if a.x <= 1.0 else a.x
            ay = a.y * template.height if a.y <= 1.0 else a.y
            if ax <= px_margin_x or ay <= px_margin_y or ax >= (template.width - px_margin_x) or ay >= (template.height - px_margin_y):
                return True
        return False

    def _template_has_one_side_anchor_ruler(self, template: Template) -> bool:
        anchors = list(template.anchors or [])
        if len(anchors) < 4:
            return False
        pts = np.array([[a.x * template.width, a.y * template.height] if a.x <= 1.0 and a.y <= 1.0 else [a.x, a.y] for a in anchors], dtype=np.float32)
        x_span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        y_span = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        if y_span <= 1.0:
            return False
        if x_span / y_span > 0.20:
            return False
        mean_x = float(np.mean(pts[:, 0]))
        return mean_x <= template.width * 0.18 or mean_x >= template.width * 0.82

    def _detect_page_corners(self, image: np.ndarray) -> np.ndarray | None:
        working, scale = self._resize_for_processing(image)
        prep = self._preprocess(working)
        edges = prep["edges"]
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:10]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return self._order_points((approx.reshape(4, 2) / scale).astype(np.float32))
            hull = cv2.convexHull(cnt)
            peri_h = cv2.arcLength(hull, True)
            approx_h = cv2.approxPolyDP(hull, 0.02 * peri_h, True)
            if len(approx_h) == 4:
                return self._order_points((approx_h.reshape(4, 2) / scale).astype(np.float32))

        line_pts = self._detect_page_corners_from_hough(edges)
        if line_pts is not None:
            return self._order_points((line_pts / scale).astype(np.float32))
        return None

    def _detect_page_corners_from_hough(self, edges: np.ndarray) -> np.ndarray | None:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=120, minLineLength=max(80, edges.shape[1] // 4), maxLineGap=25)
        if lines is None:
            return None
        vertical: list[np.ndarray] = []
        horizontal: list[np.ndarray] = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, line)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:
                horizontal.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            elif 70 < angle < 110:
                vertical.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        if len(horizontal) < 2 or len(vertical) < 2:
            return None

        def _line_pos(line: np.ndarray, axis: int) -> float:
            return float((line[axis] + line[axis + 2]) * 0.5)

        top = min(horizontal, key=lambda ln: _line_pos(ln, 1))
        bottom = max(horizontal, key=lambda ln: _line_pos(ln, 1))
        left = min(vertical, key=lambda ln: _line_pos(ln, 0))
        right = max(vertical, key=lambda ln: _line_pos(ln, 0))
        corners = [self._line_intersection(left, top), self._line_intersection(right, top), self._line_intersection(right, bottom), self._line_intersection(left, bottom)]
        if any(pt is None for pt in corners):
            return None
        return np.array(corners, dtype=np.float32)

    def _line_intersection(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
        x1, y1, x2, y2 = map(float, a)
        x3, y3, x4, y4 = map(float, b)
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return float(px), float(py)

    def _detect_anchors_by_profile(self, binary: np.ndarray, profile: str, fast_fixed: bool = False) -> list[tuple[float, float]]:
        p = str(profile or "auto").strip().lower()
        if fast_fixed:
            # fast fixed-form anchor detection profile
            if p == "one_side":
                return self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=8, fast_mode=True)
            return self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=6, fast_mode=True)
        if p == "legacy":
            return self.detect_anchors(binary, use_border_padding=False, relaxed_polygon=False)
        if p == "border":
            return self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=60)
        if p == "one_side":
            return self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=120)
        if p == "hybrid":
            a = self.detect_anchors(binary, use_border_padding=False, relaxed_polygon=False)
            b = self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True)
            merged: list[tuple[float, float]] = []
            for pt in a + b:
                if not any((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 < 36.0 for q in merged):
                    merged.append(pt)
            return merged[:20]
        return self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=60)

    @staticmethod
    def _fast_alignment_score_from_matches(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[float, int, float]:
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return -1e9, 0, 1e9
        h = cv2.getPerspectiveTransform(src_pts[:4].astype(np.float32), dst_pts[:4].astype(np.float32))
        probe = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2).astype(np.float32), h).reshape(-1, 2)
        err = np.linalg.norm(probe - dst_pts.astype(np.float32), axis=1)
        mean_err = float(np.mean(err)) if len(err) else 1e9
        median_err = float(np.median(err)) if len(err) else 1e9
        score = float(len(src_pts)) * 22.0 - (mean_err * 3.0) - (median_err * 1.6)
        return score, int(len(src_pts)), mean_err

    def _try_anchor_alignment(self, image: np.ndarray, binary: np.ndarray, template: Template, profile: str) -> tuple[np.ndarray, np.ndarray] | None:
        detected = self._detect_anchors_by_profile(binary, profile)
        template_pts = self._template_anchor_points_px(template)
        self._last_alignment_debug["last_detected_anchors"] = [(float(x), float(y)) for x, y in detected[: min(20, len(detected))]]
        self._last_alignment_debug["last_template_anchors"] = [(float(x), float(y)) for x, y in template_pts[: min(20, len(template_pts))]]
        self._last_alignment_debug["last_anchor_profile"] = str(profile)
        if len(detected) < 4 or len(template_pts) < 4:
            self._last_alignment_debug["last_anchor_match_count"] = 0
            return None
        src_pts, dst_pts = self._match_anchor_sets(np.array(detected, dtype=np.float32), template_pts)
        self._last_alignment_debug["last_anchor_match_count"] = int(min(len(src_pts), len(dst_pts)))
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None
        h = cv2.getPerspectiveTransform(self._order_points(src_pts[:4]), self._order_points(dst_pts[:4]))
        aligned = cv2.warpPerspective(image, h, (template.width, template.height))
        aligned_binary = cv2.warpPerspective(binary, h, (template.width, template.height))
        return aligned, aligned_binary

    def _refine_alignment_with_one_side_anchors(self, aligned: np.ndarray, aligned_binary: np.ndarray, template: Template) -> tuple[np.ndarray, np.ndarray]:
        anchors = list(template.anchors or [])
        if len(anchors) < 4:
            return aligned, aligned_binary

        tpl = np.array([[a.x * template.width, a.y * template.height] if a.x <= 1.0 and a.y <= 1.0 else [a.x, a.y] for a in anchors], dtype=np.float32)
        mean_x = float(np.mean(tpl[:, 0]))
        right_side = mean_x >= (template.width * 0.5)

        detected = np.array(self.detect_anchors(aligned_binary, use_border_padding=True, relaxed_polygon=True, max_points=120), dtype=np.float32)
        if len(detected) < 4:
            return aligned, aligned_binary

        margin = template.width * 0.24
        if right_side:
            det_side = detected[detected[:, 0] >= (template.width - margin)]
            tpl_side = tpl[tpl[:, 0] >= (template.width - margin)]
        else:
            det_side = detected[detected[:, 0] <= margin]
            tpl_side = tpl[tpl[:, 0] <= margin]
        if len(det_side) < 4 or len(tpl_side) < 4:
            return aligned, aligned_binary

        det_side = det_side[np.argsort(det_side[:, 1])]
        tpl_side = tpl_side[np.argsort(tpl_side[:, 1])]

        def _trim(arr: np.ndarray) -> np.ndarray:
            if len(arr) <= 10:
                return arr
            lo = max(0, int(len(arr) * 0.05))
            hi = max(lo + 4, int(len(arr) * 0.95))
            return arr[lo:hi]

        det_side = _trim(det_side)
        tpl_side = _trim(tpl_side)

        n = min(len(det_side), len(tpl_side))
        if n < 4:
            return aligned, aligned_binary
        if len(det_side) != n:
            idx = np.linspace(0, len(det_side) - 1, n).astype(int)
            det_side = det_side[idx]
        if len(tpl_side) != n:
            idx = np.linspace(0, len(tpl_side) - 1, n).astype(int)
            tpl_side = tpl_side[idx]

        m, inliers = cv2.estimateAffinePartial2D(det_side, tpl_side, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if m is None:
            return aligned, aligned_binary
        if inliers is not None and int(inliers.sum()) < max(4, int(0.55 * n)):
            return aligned, aligned_binary

        sx = float(np.hypot(m[0, 0], m[1, 0]))
        sy = float(np.hypot(m[0, 1], m[1, 1]))
        if sx < 0.985 or sx > 1.015 or sy < 0.985 or sy > 1.015:
            return aligned, aligned_binary
        if abs(float(m[0, 2])) > template.width * 0.10 or abs(float(m[1, 2])) > template.height * 0.12:
            return aligned, aligned_binary

        dx = float(np.median(tpl_side[:, 0] - det_side[:, 0]))
        dy = float(np.median(tpl_side[:, 1] - det_side[:, 1]))
        m_shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

        probe = cv2.transform(det_side.reshape(1, -1, 2), m_shift).reshape(-1, 2)
        base_err = float(np.mean(np.abs(det_side[:, 1] - tpl_side[:, 1])))
        new_err = float(np.mean(np.abs(probe[:, 1] - tpl_side[:, 1])))
        if new_err > base_err - 0.2:
            return aligned, aligned_binary

        refined = cv2.warpAffine(aligned, m_shift, (template.width, template.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        refined_bin = cv2.warpAffine(aligned_binary, m_shift, (template.width, template.height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        return refined, refined_bin

    def correct_perspective(self, image: np.ndarray, binary: np.ndarray, template: Template, result: OMRResult) -> tuple[np.ndarray, np.ndarray | None]:
        self._last_alignment_debug = {}
        aligned: np.ndarray | None = None
        aligned_binary: np.ndarray | None = None
        fast_mode = self._is_fast_scan_mode(template)
        fixed_profile = self._is_fixed_scan_profile(template)
        if fixed_profile:
            fast_mode = True
        mode = str(getattr(self, "alignment_profile", "auto") or "auto").strip().lower()
        candidates: list[str] = []
        if fixed_profile:
            mode_fast = self._template_match_mode(template)
            candidates = ["one_side"] if mode_fast == "one_side_ruler" else [self._fast_alignment_profile(template)]
        elif fast_mode:
            candidates = [self._fast_alignment_profile(template)]
        elif mode == "auto":
            if self._template_has_one_side_anchor_ruler(template):
                candidates = ["one_side", "border", "hybrid", "legacy"]
            elif self._template_has_border_anchors(template):
                candidates = ["border", "hybrid", "legacy"]
            else:
                candidates = ["legacy", "hybrid", "border"]
        elif mode in {"legacy", "border", "hybrid", "one_side"}:
            candidates = [mode]
        else:
            candidates = ["hybrid"]

        base_image = aligned if aligned is not None else image
        base_binary = aligned_binary if aligned_binary is not None else binary
        best_attempt: tuple[float, str, np.ndarray, np.ndarray, dict[str, object]] | None = None

        if fixed_profile:
            # fast fixed-form path: single short alignment path for fixed-form scan.
            candidate = candidates[0] if candidates else "border"
            det_started = time.perf_counter()
            detected = self._detect_anchors_by_profile(base_binary, candidate, fast_fixed=True)
            self._last_alignment_debug["anchor_detect_time_sec"] = float(time.perf_counter() - det_started)
            template_pts = self._template_anchor_points_px(template)
            if len(detected) >= 4 and len(template_pts) >= 4:
                src_pts, dst_pts = self._match_anchor_sets(np.array(detected, dtype=np.float32), template_pts)
                score, match_count, mean_err = self._fast_alignment_score_from_matches(src_pts, dst_pts)
                if match_count >= 4 and mean_err <= 14.0:
                    h = cv2.getPerspectiveTransform(self._order_points(src_pts[:4]), self._order_points(dst_pts[:4]))
                    fast_img = cv2.warpPerspective(base_image, h, (template.width, template.height))
                    fast_bin = cv2.warpPerspective(base_binary, h, (template.width, template.height))
                    self._last_alignment_debug["alignment_mode"] = f"fixed_fast:{candidate}"
                    self._last_alignment_debug["alignment_score"] = float(score)
                    self._last_alignment_debug["fast_match_count"] = int(match_count)
                    self._last_alignment_debug["fast_mean_error"] = float(mean_err)
                    self._last_alignment_debug["matched_detected_anchors"] = [(float(x), float(y)) for x, y in detected[: min(12, len(detected))]]
                    self._last_alignment_debug["used_detected_anchors"] = [(float(x), float(y)) for x, y in detected[: min(20, len(detected))]]
                    self._last_alignment_debug["used_template_anchors"] = [(float(x), float(y)) for x, y in template_pts[: min(20, len(template_pts))]]
                    self._last_alignment_debug["used_anchor_match_count"] = int(match_count)
                    self._last_alignment_debug["used_anchor_profile"] = str(candidate)
                    return fast_img, fast_bin
            # safe fallback for fixed-form when fast accept fails.
            attempt = self._try_anchor_alignment(base_image, base_binary, template, candidate)
            if attempt is not None:
                safe_img, safe_bin = attempt
                self._last_alignment_debug["alignment_mode"] = f"fixed_safe:{candidate}"
                self._last_alignment_debug["alignment_score"] = float(self._orientation_score(safe_bin, template))
                return safe_img, safe_bin
            contour_img, contour_bin = self._fallback_align_page_contour(image, template, conservative=True)
            self._last_alignment_debug["alignment_mode"] = "fixed_safe:page_contour"
            self._last_alignment_debug["alignment_score"] = float(self._orientation_score(contour_bin, template))
            return contour_img, contour_bin

        for candidate in candidates:
            if self._time_budget_exceeded():
                break
            det_started = time.perf_counter()
            if candidate == "one_side":
                coarse_img, coarse_bin = self._fallback_align_page_contour(image, template, conservative=True)
                refined_img, refined_bin = self._refine_alignment_with_one_side_anchors(coarse_img, coarse_bin, template)
            else:
                attempt = self._try_anchor_alignment(base_image, base_binary, template, candidate)
                if attempt is None:
                    continue
                coarse_img, coarse_bin = attempt
                refined_img, refined_bin = (coarse_img, coarse_bin) if (candidate == "legacy" or fast_mode) else self._refine_alignment_with_template_anchors(coarse_img, coarse_bin, template)
            self._last_alignment_debug["anchor_detect_time_sec"] = self._last_alignment_debug.get("anchor_detect_time_sec", 0.0) + float(time.perf_counter() - det_started)
            if fast_mode:
                candidate_score = self._orientation_score(refined_bin, template)
                affine_img, affine_bin = refined_img, refined_bin
                if candidate_score < 150.0:
                    # Fast-path not confident enough -> light fallback to safe orientation/refine.
                    oriented_img, oriented_bin = self._auto_orient(refined_img, refined_bin, template)
                    shifted_img, shifted_bin = self._refine_corner_translation(oriented_img, oriented_bin, template)
                    affine_img, affine_bin = self._refine_alignment_with_affine_anchors(shifted_img, shifted_bin, template)
                    candidate_score = self._orientation_score(affine_bin, template)
            else:
                oriented_img, oriented_bin = self._auto_orient(refined_img, refined_bin, template)
                shifted_img, shifted_bin = self._refine_corner_translation(oriented_img, oriented_bin, template)
                affine_img, affine_bin = self._refine_alignment_with_affine_anchors(shifted_img, shifted_bin, template)
                candidate_score = self._orientation_score(affine_bin, template)
            if fixed_profile and candidate_score >= 135.0:
                self._last_alignment_debug["alignment_mode"] = f"fixed:{candidate}"
                self._last_alignment_debug["alignment_score"] = float(candidate_score)
                self._last_alignment_debug["template_match_mode"] = self._template_match_mode(template)
                self._last_alignment_debug["used_detected_anchors"] = list(self._last_alignment_debug.get("last_detected_anchors", []) or [])
                self._last_alignment_debug["used_template_anchors"] = list(self._last_alignment_debug.get("last_template_anchors", []) or [])
                self._last_alignment_debug["used_anchor_match_count"] = int(self._last_alignment_debug.get("last_anchor_match_count", 0) or 0)
                self._last_alignment_debug["used_anchor_profile"] = str(self._last_alignment_debug.get("last_anchor_profile", candidate) or candidate)
                return affine_img, affine_bin
            if best_attempt is None or candidate_score > best_attempt[0]:
                snapshot = {
                    "used_detected_anchors": list(self._last_alignment_debug.get("last_detected_anchors", []) or []),
                    "used_template_anchors": list(self._last_alignment_debug.get("last_template_anchors", []) or []),
                    "used_anchor_match_count": int(self._last_alignment_debug.get("last_anchor_match_count", 0) or 0),
                    "used_anchor_profile": str(self._last_alignment_debug.get("last_anchor_profile", candidate) or candidate),
                }
                best_attempt = (candidate_score, candidate, affine_img, affine_bin, snapshot)

        if best_attempt is not None:
            score, candidate, best_img, best_bin, snapshot = best_attempt
            self._last_alignment_debug["alignment_mode"] = candidate
            self._last_alignment_debug["alignment_score"] = float(score)
            self._last_alignment_debug["used_detected_anchors"] = list(snapshot.get("used_detected_anchors", []) or [])
            self._last_alignment_debug["used_template_anchors"] = list(snapshot.get("used_template_anchors", []) or [])
            self._last_alignment_debug["used_anchor_match_count"] = int(snapshot.get("used_anchor_match_count", 0) or 0)
            self._last_alignment_debug["used_anchor_profile"] = str(snapshot.get("used_anchor_profile", candidate) or candidate)
            return best_img, best_bin

        result.issues.append(OMRIssue("MISSING_ANCHORS", "Anchor detection failed; using page contour fallback"))
        aligned, aligned_binary = self._fallback_align_page_contour(image, template)
        refined_img, refined_bin = self._refine_alignment_with_template_anchors(aligned, aligned_binary, template)
        oriented_img, oriented_bin = self._auto_orient(refined_img, refined_bin, template)
        shifted_img, shifted_bin = self._refine_corner_translation(oriented_img, oriented_bin, template)
        affine_img, affine_bin = self._refine_alignment_with_affine_anchors(shifted_img, shifted_bin, template)
        self._last_alignment_debug.setdefault("alignment_mode", "page_contour")
        self._last_alignment_debug["alignment_score"] = float(self._orientation_score(affine_bin, template))
        return affine_img, affine_bin

    def _auto_orient(self, aligned: np.ndarray, aligned_binary: np.ndarray, template: Template) -> tuple[np.ndarray, np.ndarray]:
        hint = self._batch_orientation_hint
        if hint is not None:
            hinted_img, hinted_bin = self._rotate_pair(aligned, aligned_binary, int(hint), template)
            hinted_score = self._orientation_score(hinted_bin, template)
            baseline = self._batch_orientation_hint_score
            if baseline is None or hinted_score >= (float(baseline) - 35.0):
                self._last_alignment_debug["orientation_rotation"] = int(hint) % 360
                return hinted_img, hinted_bin

        candidates = []
        for rotation in (0, 90, 180, 270):
            img_r, bin_r = self._rotate_pair(aligned, aligned_binary, rotation, template)
            score = self._orientation_score(bin_r, template)
            candidates.append((score, rotation, img_r, bin_r))
        by_rotation = {rotation: (score, img_r, bin_r) for score, rotation, img_r, bin_r in candidates}
        zero_score = by_rotation[0][0]
        best_score, best_rotation, best_img, best_bin = max(candidates, key=lambda item: item[0])
        flip_score = by_rotation[180][0]

        chosen_rotation = 0
        chosen_img = by_rotation[0][1]
        chosen_bin = by_rotation[0][2]

        if flip_score > zero_score + 12.0:
            chosen_rotation = 180
            chosen_img = by_rotation[180][1]
            chosen_bin = by_rotation[180][2]

        if best_rotation in {90, 270} and best_score > max(zero_score, flip_score) + 40.0:
            chosen_rotation = int(best_rotation)
            chosen_img = best_img
            chosen_bin = best_bin

        self._last_alignment_debug["orientation_rotation"] = int(chosen_rotation)
        return chosen_img, chosen_bin

    def _orientation_score(self, binary: np.ndarray, template: Template) -> float:
        score = 0.0
        anchors = np.array(self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=24, fast_mode=True), dtype=np.float32)
        if len(anchors) >= 4 and len(template.anchors) >= 4:
            tpl = self._template_anchor_points_px(template)
            tpl = self._pick_corner_like_points(tpl)
            total = 0.0
            for pt in tpl:
                d2 = np.sum((anchors - pt) ** 2, axis=1)
                total += float(np.sqrt(float(np.min(d2))))
            score -= total
            score += float(min(len(anchors), 12)) * 6.0
        vert = np.sum(binary > 0, axis=0).astype(np.float32)
        horz = np.sum(binary > 0, axis=1).astype(np.float32)
        score += float(np.std(vert) + np.std(horz)) * 0.02
        m = cv2.moments(binary)
        if abs(m.get("mu02", 0.0)) > 1e-6:
            angle = 0.5 * np.degrees(np.arctan2(2.0 * m.get("mu11", 0.0), m.get("mu20", 0.0) - m.get("mu02", 0.0)))
            score -= abs(float(angle)) * 0.1
        return score

    def _refine_corner_translation(self, aligned: np.ndarray, aligned_binary: np.ndarray, template: Template) -> tuple[np.ndarray, np.ndarray]:
        if len(template.anchors) < 4:
            return aligned, aligned_binary
        template_pts = self._template_anchor_points_px(template)
        ordered_template = self._order_points(self._pick_corner_like_points(template_pts))
        detected = np.array(self.detect_anchors(aligned_binary, use_border_padding=True, relaxed_polygon=True, max_points=120), dtype=np.float32)
        if len(detected) < 4:
            return aligned, aligned_binary
        ordered_detected = self._order_points(self._pick_corner_like_points(detected))
        delta = ordered_template - ordered_detected
        dx = float(np.median(delta[:, 0]))
        dy = float(np.median(delta[:, 1]))
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            return aligned, aligned_binary
        shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(aligned, shift, (template.width, template.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        shifted_bin = cv2.warpAffine(aligned_binary, shift, (template.width, template.height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        self._last_alignment_debug["fine_offset"] = {"dx": dx, "dy": dy}
        return shifted, shifted_bin

    def _refine_alignment_with_affine_anchors(self, aligned: np.ndarray, aligned_binary: np.ndarray, template: Template) -> tuple[np.ndarray, np.ndarray]:
        if len(template.anchors) < 4:
            return aligned, aligned_binary
        template_pts = self._template_anchor_points_px(template)
        detected = np.array(self.detect_anchors(aligned_binary, use_border_padding=True, relaxed_polygon=True, max_points=120), dtype=np.float32)
        if len(detected) < 4:
            return aligned, aligned_binary

        max_dist = max(10.0, 0.03 * float(np.hypot(template.width, template.height)))
        src_matches: list[np.ndarray] = []
        dst_matches: list[np.ndarray] = []
        used: set[int] = set()
        for t in template_pts:
            d2 = np.sum((detected - t) ** 2, axis=1)
            idx = int(np.argmin(d2))
            if idx in used:
                continue
            dist = float(np.sqrt(float(d2[idx])))
            if dist <= max_dist:
                used.add(idx)
                src_matches.append(detected[idx])
                dst_matches.append(t)
        if len(src_matches) < 4:
            return aligned, aligned_binary

        src = np.array(src_matches, dtype=np.float32)
        dst = np.array(dst_matches, dtype=np.float32)
        affine, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.5)
        if affine is None:
            return aligned, aligned_binary
        if inliers is not None and int(inliers.sum()) < 4:
            return aligned, aligned_binary

        linear = affine[:, :2].astype(np.float64)
        det = float(np.linalg.det(linear))
        if not np.isfinite(det) or det <= 0.0:
            return aligned, aligned_binary
        if det < 0.90 or det > 1.10:
            return aligned, aligned_binary
        if float(np.max(np.abs(linear - np.eye(2)))) > 0.18:
            return aligned, aligned_binary

        proj = cv2.transform(src.reshape(1, -1, 2), affine.astype(np.float32)).reshape(-1, 2)
        base_err = float(np.mean(np.min(np.sqrt(np.sum((detected[:, None, :] - template_pts[None, :, :]) ** 2, axis=2)), axis=0)))
        new_err = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
        if not np.isfinite(new_err) or new_err > max(2.5, base_err - 0.3):
            return aligned, aligned_binary

        refined = cv2.warpAffine(aligned, affine.astype(np.float32), (template.width, template.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        refined_bin = cv2.warpAffine(aligned_binary, affine.astype(np.float32), (template.width, template.height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        self._last_alignment_debug["affine_refine"] = {
            "matrix": affine.astype(float).tolist(),
            "rmse": new_err,
            "match_count": len(src_matches),
        }
        return refined, refined_bin

    def _refine_alignment_with_template_anchors(self, aligned: np.ndarray, aligned_binary: np.ndarray, template: Template) -> tuple[np.ndarray, np.ndarray]:
        if len(template.anchors) < 4:
            return aligned, aligned_binary
        detected = self.detect_anchors(aligned_binary)
        if len(detected) < 4:
            return aligned, aligned_binary

        template_pts = self._template_anchor_points_px(template)
        detected_arr = np.array(detected, dtype=np.float32)
        if len(template_pts) < 4:
            return aligned, aligned_binary

        max_dist = max(8.0, 0.02 * float(np.hypot(template.width, template.height)))
        src_matches: list[np.ndarray] = []
        dst_matches: list[np.ndarray] = []
        used: set[int] = set()
        for t in template_pts:
            d2 = np.sum((detected_arr - t) ** 2, axis=1)
            idx = int(np.argmin(d2))
            if idx in used:
                continue
            dist = float(np.sqrt(float(d2[idx])))
            if dist <= max_dist:
                used.add(idx)
                src_matches.append(detected_arr[idx])
                dst_matches.append(t)
        if len(src_matches) < 5:
            return aligned, aligned_binary

        src = np.array(src_matches, dtype=np.float32)
        dst = np.array(dst_matches, dtype=np.float32)
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=4.0)
        if h is None:
            return aligned, aligned_binary
        if mask is not None and int(mask.sum()) < 5:
            return aligned, aligned_binary

        h = h.astype(np.float64)
        if abs(float(h[2, 0])) > 2e-4 or abs(float(h[2, 1])) > 2e-4:
            return aligned, aligned_binary

        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
        if not np.isfinite(rmse) or rmse > 3.5:
            return aligned, aligned_binary

        corners = np.array([[[0.0, 0.0]], [[template.width - 1.0, 0.0]], [[template.width - 1.0, template.height - 1.0]], [[0.0, template.height - 1.0]]], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(corners, h).reshape(4, 2)
        orig_area = float(template.width * template.height)
        warped_area = abs(float(cv2.contourArea(warped_corners.astype(np.float32))))
        if orig_area <= 0:
            return aligned, aligned_binary
        area_ratio = warped_area / orig_area
        if area_ratio < 0.90 or area_ratio > 1.10:
            return aligned, aligned_binary

        center = np.array([template.width / 2.0, template.height / 2.0], dtype=np.float32).reshape(1, 1, 2)
        warped_center = cv2.perspectiveTransform(center, h).reshape(2)
        if float(np.linalg.norm(warped_center - center.reshape(2))) > 25.0:
            return aligned, aligned_binary

        refined = cv2.warpPerspective(aligned, h, (template.width, template.height))
        refined_bin = cv2.warpPerspective(aligned_binary, h, (template.width, template.height))

        refined_detected = self.detect_anchors(refined_bin)
        if len(refined_detected) >= 4:
            ref_arr = np.array(refined_detected, dtype=np.float32)
            base_arr = detected_arr

            def _nearest_avg(det_arr: np.ndarray, tpl_arr: np.ndarray) -> float:
                vals: list[float] = []
                for p in tpl_arr:
                    d2 = np.sum((det_arr - p) ** 2, axis=1)
                    vals.append(float(np.sqrt(float(np.min(d2)))))
                return float(np.mean(vals)) if vals else 1e9

            base_err = _nearest_avg(base_arr, template_pts)
            refined_err = _nearest_avg(ref_arr, template_pts)
            if not np.isfinite(refined_err) or refined_err > (base_err - 0.5):
                return aligned, aligned_binary

        return refined, refined_bin

    def _fallback_align_page_contour(self, image: np.ndarray, template: Template, conservative: bool = False) -> tuple[np.ndarray, np.ndarray]:
        corners = self._detect_page_corners(image)
        if corners is None:
            resized = cv2.resize(image, (template.width, template.height))
            return resized, self._preprocess(resized)["binary"]
        if not self._is_reasonable_page_warp(corners, image.shape[:2], template):
            resized = cv2.resize(image, (template.width, template.height))
            return resized, self._preprocess(resized)["binary"]
        self._last_alignment_debug["page_corners"] = corners.tolist()
        dst = np.array([[0, 0], [template.width - 1, 0], [template.width - 1, template.height - 1], [0, template.height - 1]], dtype=np.float32)
        h = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        if conservative:
            h64 = h.astype(np.float64)
            if abs(float(h64[2, 0])) > 1.2e-4 or abs(float(h64[2, 1])) > 1.2e-4:
                resized = cv2.resize(image, (template.width, template.height))
                return resized, self._preprocess(resized)["binary"]
            warped_corners = cv2.perspectiveTransform(np.array([[[0.0, 0.0]], [[template.width - 1.0, 0.0]], [[template.width - 1.0, template.height - 1.0]], [[0.0, template.height - 1.0]]], dtype=np.float32), h64.astype(np.float32)).reshape(4, 2)
            warped_area = abs(float(cv2.contourArea(warped_corners.astype(np.float32))))
            base_area = float(template.width * template.height)
            if base_area <= 0:
                resized = cv2.resize(image, (template.width, template.height))
                return resized, self._preprocess(resized)["binary"]
            area_ratio = warped_area / base_area
            if area_ratio < 0.96 or area_ratio > 1.04:
                resized = cv2.resize(image, (template.width, template.height))
                return resized, self._preprocess(resized)["binary"]
        aligned = cv2.warpPerspective(image, h, (template.width, template.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aligned_binary = cv2.warpPerspective(self._preprocess(image)["binary"], h, (template.width, template.height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        return aligned, aligned_binary

    def _is_reasonable_page_warp(self, corners: np.ndarray, image_shape: tuple[int, int], template: Template) -> bool:
        img_h, img_w = image_shape
        if len(corners) != 4:
            return False
        area = abs(float(cv2.contourArea(corners.astype(np.float32))))
        image_area = float(img_w * img_h)
        if image_area <= 0 or area < image_area * 0.45:
            return False
        width_top = float(np.linalg.norm(corners[1] - corners[0]))
        width_bottom = float(np.linalg.norm(corners[2] - corners[3]))
        height_left = float(np.linalg.norm(corners[3] - corners[0]))
        height_right = float(np.linalg.norm(corners[2] - corners[1]))
        avg_width = max(1.0, 0.5 * (width_top + width_bottom))
        avg_height = max(1.0, 0.5 * (height_left + height_right))
        source_ratio = avg_width / avg_height
        template_ratio = float(template.width) / max(1.0, float(template.height))
        ratio_delta = abs(np.log(max(source_ratio, 1e-6) / max(template_ratio, 1e-6)))
        return ratio_delta <= 0.35

    def _match_anchor_sets(self, detected: np.ndarray, template_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        det = self._order_points(self._pick_corner_like_points(detected))
        dst = self._order_points(self._pick_corner_like_points(template_pts))
        return det.astype(np.float32), dst.astype(np.float32)

    def _pick_corner_like_points(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) <= 4:
            return pts[:4]
        sums = pts[:, 0] + pts[:, 1]
        diffs = pts[:, 0] - pts[:, 1]
        picks = [pts[np.argmin(sums)], pts[np.argmax(diffs)], pts[np.argmax(sums)], pts[np.argmin(diffs)]]
        return np.array(picks, dtype=np.float32)

    def _estimate_local_fill_threshold(self, binary: np.ndarray, center: np.ndarray, radius: int, fallback: float) -> float:
        x, y = map(int, center)
        pad = max(radius * 2, 12)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(binary.shape[1], x + pad + 1), min(binary.shape[0], y + pad + 1)
        roi = binary[y0:y1, x0:x1]
        if roi.size == 0:
            return fallback
        local_density = float(np.count_nonzero(roi)) / float(roi.size)
        return float(np.clip(0.65 * fallback + 0.35 * max(self.empty_threshold + 0.02, local_density * 1.15), self.empty_threshold + 0.02, 0.85))

    def detect_bubbles(self, binary: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        mask = self._get_x_mask(radius)
        r = max(3, int(radius))
        h, w = binary.shape[:2]
        ratios = np.zeros((len(centers),), dtype=np.float32)
        if len(centers) == 0 or h <= 0 or w <= 0:
            return ratios
        centers_int = centers.astype(np.int32)
        valid = (
            (centers_int[:, 0] >= 0)
            & (centers_int[:, 0] < w)
            & (centers_int[:, 1] >= 0)
            & (centers_int[:, 1] < h)
        )
        if not np.any(valid):
            return ratios

        bin01 = (binary > 0).astype(np.float32)
        kernel = mask.astype(np.float32)
        den = float(np.count_nonzero(mask))
        if den <= 0.0:
            return ratios
        cross_sum = cv2.filter2D(bin01, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        square_mean = cv2.boxFilter(
            bin01,
            ddepth=-1,
            ksize=((2 * r) + 1, (2 * r) + 1),
            normalize=True,
            borderType=cv2.BORDER_CONSTANT,
        )

        xs = centers_int[valid, 0]
        ys = centers_int[valid, 1]
        fill = cross_sum[ys, xs] / den
        mean = square_mean[ys, xs]
        merged = np.clip((0.8 * fill) + (0.2 * mean), 0.0, 1.0).astype(np.float32)
        ratios[np.where(valid)[0]] = merged
        return ratios

    def _detect_center_core_marks(self, binary: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        r = max(2, int(round(radius * 0.45)))
        h, w = binary.shape[:2]
        ratios = np.zeros((len(centers),), dtype=np.float32)
        centers_int = centers.astype(np.int32)
        for i, (x, y) in enumerate(centers_int):
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(w, x + r + 1), min(h, y + r + 1)
            roi = binary[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            ratios[i] = float(np.count_nonzero(roi)) / float(roi.size)
        return ratios

    def _detect_core_ring_contrast(self, binary: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        core_r = max(2, int(round(radius * 0.35)))
        ring_r = max(core_r + 1, int(round(radius * 0.80)))
        h, w = binary.shape[:2]
        scores = np.zeros((len(centers),), dtype=np.float32)
        centers_int = centers.astype(np.int32)
        yy, xx = np.indices(((2 * ring_r) + 1, (2 * ring_r) + 1))
        cx = cy = ring_r
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        core_mask = dist2 <= (core_r ** 2)
        ring_mask = (dist2 > (core_r ** 2)) & (dist2 <= (ring_r ** 2))
        for i, (x, y) in enumerate(centers_int):
            x0, y0 = x - ring_r, y - ring_r
            x1, y1 = x + ring_r + 1, y + ring_r + 1
            if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
                continue
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(w, x1), min(h, y1)
            roi = binary[y0c:y1c, x0c:x1c]
            local_core = core_mask[(y0c - y0):(y1c - y0), (x0c - x0):(x1c - x0)]
            local_ring = ring_mask[(y0c - y0):(y1c - y0), (x0c - x0):(x1c - x0)]
            if roi.size == 0 or not np.any(local_core):
                continue
            core_ratio = float(np.count_nonzero(roi[local_core])) / float(np.count_nonzero(local_core))
            ring_ratio = 0.0 if not np.any(local_ring) else float(np.count_nonzero(roi[local_ring])) / float(np.count_nonzero(local_ring))
            scores[i] = float(np.clip(core_ratio - (0.45 * ring_ratio), 0.0, 1.0))
        return scores

    def _detect_eroded_mark_density(self, binary: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        r = max(3, int(round(radius * 0.85)))
        kernel = np.ones((3, 3), np.uint8)
        h, w = binary.shape[:2]
        scores = np.zeros((len(centers),), dtype=np.float32)
        centers_int = centers.astype(np.int32)
        for i, (x, y) in enumerate(centers_int):
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(w, x + r + 1), min(h, y + r + 1)
            roi = binary[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            eroded = cv2.erode(roi, kernel, iterations=1)
            scores[i] = float(np.count_nonzero(eroded)) / float(eroded.size)
        return scores

    def _detect_square_mark_density(self, binary: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        r = max(2, int(round(radius * 0.70)))
        h, w = binary.shape[:2]
        scores = np.zeros((len(centers),), dtype=np.float32)
        centers_int = centers.astype(np.int32)
        for i, (x, y) in enumerate(centers_int):
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(w, x + r + 1), min(h, y + r + 1)
            roi = binary[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            scores[i] = float(np.count_nonzero(roi)) / float(roi.size)
        return scores

    def _detect_digit_zone_multi_probe_marks(self, binary: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        probe_r = max(2, int(round(radius * 0.62)))
        offset = max(1, int(round(radius * 0.38)))
        offsets = ((0, 0), (offset, 0), (-offset, 0), (0, offset), (0, -offset))
        h, w = binary.shape[:2]
        scores = np.zeros((len(centers),), dtype=np.float32)
        centers_int = centers.astype(np.int32)
        for i, (x, y) in enumerate(centers_int):
            probe_scores: list[float] = []
            for dx, dy in offsets:
                cx = int(x + dx)
                cy = int(y + dy)
                x0, y0 = max(0, cx - probe_r), max(0, cy - probe_r)
                x1, y1 = min(w, cx + probe_r + 1), min(h, cy + probe_r + 1)
                roi = binary[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                probe_scores.append(float(np.count_nonzero(roi)) / float(roi.size))
            if not probe_scores:
                continue
            probe_scores.sort(reverse=True)
            best = probe_scores[0]
            top2 = float(np.mean(probe_scores[: min(2, len(probe_scores))]))
            scores[i] = float(np.clip((0.65 * best) + (0.35 * top2), 0.0, 1.0))
        return scores

    def _detect_digit_zone_peak_window_marks(self, binary: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        peak_r = max(2, int(round(radius * 0.52)))
        search_r = max(1, int(round(radius * 0.45)))
        h, w = binary.shape[:2]
        scores = np.zeros((len(centers),), dtype=np.float32)
        centers_int = centers.astype(np.int32)
        for i, (x, y) in enumerate(centers_int):
            best_density = 0.0
            ring_density = 0.0
            for dy in range(-search_r, search_r + 1, max(1, peak_r // 2)):
                for dx in range(-search_r, search_r + 1, max(1, peak_r // 2)):
                    cx = int(x + dx)
                    cy = int(y + dy)
                    x0, y0 = max(0, cx - peak_r), max(0, cy - peak_r)
                    x1, y1 = min(w, cx + peak_r + 1), min(h, cy + peak_r + 1)
                    roi = binary[y0:y1, x0:x1]
                    if roi.size == 0:
                        continue
                    density = float(np.count_nonzero(roi)) / float(roi.size)
                    if density > best_density:
                        best_density = density
            ring_r = max(peak_r + 1, int(round(radius * 0.95)))
            x0, y0 = max(0, x - ring_r), max(0, y - ring_r)
            x1, y1 = min(w, x + ring_r + 1), min(h, y + ring_r + 1)
            outer = binary[y0:y1, x0:x1]
            if outer.size:
                yy, xx = np.indices(outer.shape)
                cx0 = int(x - x0)
                cy0 = int(y - y0)
                dist2 = (xx - cx0) ** 2 + (yy - cy0) ** 2
                core_mask = dist2 <= (peak_r ** 2)
                ring_mask = (dist2 > (peak_r ** 2)) & (dist2 <= (ring_r ** 2))
                if np.any(ring_mask):
                    ring_density = float(np.count_nonzero(outer[ring_mask])) / float(np.count_nonzero(ring_mask))
            scores[i] = float(np.clip((0.85 * best_density) - (0.25 * ring_density), 0.0, 1.0))
        return scores

    def classify_bubble(self, ratio: float) -> str:
        if ratio > self.fill_threshold:
            return "filled"
        if ratio < self.empty_threshold:
            return "empty"
        return "uncertain"

    def _resolve_zone_centers(self, binary: np.ndarray, zone: Zone, template: Template) -> np.ndarray:
        grid = zone.grid
        if not grid or not grid.bubble_positions:
            return np.empty((0, 2), dtype=np.float32)
        h, w = binary.shape[:2]
        plan = self._get_template_runtime_plan(template)
        expected = np.array(plan.zone_centers_px.get(zone.id, np.empty((0, 2), dtype=np.float32)), dtype=np.float32)
        if expected.size == 0:
            expected = np.array([(x * w, y * h) if x <= 1.0 and y <= 1.0 else (x, y) for x, y in grid.bubble_positions], dtype=np.float32)
        elif expected.shape[0] and (w != template.width or h != template.height):
            sx = float(w) / float(max(1, template.width))
            sy = float(h) / float(max(1, template.height))
            expected = expected.copy()
            expected[:, 0] *= sx
            expected[:, 1] *= sy
        shift_x, shift_y = self._active_point_shift
        if expected.shape[0] and (abs(float(shift_x)) > 0.01 or abs(float(shift_y)) > 0.01):
            expected = expected.copy()
            expected[:, 0] += float(shift_x)
            expected[:, 1] += float(shift_y)
        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            if zone.zone_type == ZoneType.EXAM_CODE_BLOCK:
                modeled_expected, _ = self._digit_model_expected_points(template, zone)
                if len(modeled_expected) == len(expected):
                    expected = modeled_expected
                else:
                    expected, _ = self._digit_zone_guidance(binary, expected, zone, template)
            else:
                expected, _ = self._digit_zone_guidance(binary, expected, zone, template)
            return self._resolve_column_digit_centers(binary, expected, grid, float(zone.metadata.get("bubble_radius", 9)))
        if self._is_fast_scan_mode(template) and self._is_fixed_scan_profile(template):
            return expected.astype(np.float32)
        bubble_radius = float(zone.metadata.get("bubble_radius", 9))
        min_area = max(6.0, 0.25 * np.pi * (bubble_radius ** 2))
        max_area = max(min_area * 3.5, 4.0 * np.pi * (bubble_radius ** 2))
        search_pad = max(10, int(round(bubble_radius * 2.2)))
        refined: list[np.ndarray] = []
        for exp_x, exp_y in expected:
            x0 = int(max(0, exp_x - search_pad))
            y0 = int(max(0, exp_y - search_pad))
            x1 = int(min(w, exp_x + search_pad + 1))
            y1 = int(min(h, exp_y + search_pad + 1))
            roi = binary[y0:y1, x0:x1]
            if roi.size == 0:
                refined.append(np.array([exp_x, exp_y], dtype=np.float32))
                continue
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
            best_pt: np.ndarray | None = None
            best_score = float("inf")
            for idx in range(1, num_labels):
                area = float(stats[idx, cv2.CC_STAT_AREA])
                if area < min_area or area > max_area:
                    continue
                cx, cy = centroids[idx]
                cand = np.array([cx + x0, cy + y0], dtype=np.float32)
                dist = float(np.linalg.norm(cand - np.array([exp_x, exp_y], dtype=np.float32)))
                if dist > max(6.0, bubble_radius * 1.6):
                    continue
                if dist < best_score:
                    best_score = dist
                    best_pt = cand
            refined.append(best_pt if best_pt is not None else np.array([exp_x, exp_y], dtype=np.float32))
        return np.array(refined, dtype=np.float32)

    def _apply_anchor_ruler_to_digit_zone(self, binary: np.ndarray, expected: np.ndarray, zone: Zone, template: Template) -> np.ndarray:
        guided, _ = self._digit_zone_guidance(binary, expected, zone, template)
        return guided

    def _build_digit_row_lines(self, anchor_ys: np.ndarray, rows: int) -> list[float]:
        ys = np.sort(np.asarray(anchor_ys, dtype=np.float32))
        if rows <= 0 or ys.size < 2:
            return []
        if ys.size == rows + 1:
            step = float((ys[-1] - ys[0]) / max(1, rows - 1)) if rows > 1 else 0.0
            row_tops = ys.astype(np.float32)
            bottom = float(row_tops[-1] + step)
            return [float(v) for v in row_tops.tolist()] + [bottom]
        if ys.size >= rows + 2:
            row_tops = ys[1 : rows + 1]
            step = float(np.median(np.diff(row_tops))) if row_tops.size >= 2 else 0.0
            if step <= 0.0:
                step = float((row_tops[-1] - row_tops[0]) / max(1, rows - 1)) if rows > 1 else 0.0
            bottom = float(row_tops[-1] + step)
            return [float(v) for v in row_tops.tolist()] + [bottom]
        return np.linspace(float(ys[0]), float(ys[-1]), rows + 1, dtype=np.float32).astype(float).tolist()

    def _digit_zone_guidance(self, binary: np.ndarray, expected: np.ndarray, zone: Zone, template: Template) -> tuple[np.ndarray, dict[str, object]]:
        if len(expected) == 0 or not zone.grid:
            return expected, {}
        template_pts = np.array(self._get_manual_digit_anchor_points(template, zone), dtype=np.float32)
        if len(template_pts) < 2:
            return expected, {}
        detected_pts: list[tuple[float, float]] = []
        guide_regions: list[tuple[float, float, float, float]] = []
        for pt in template_pts:
            matched, region = self._find_digit_anchor_match(binary, pt)
            if region is not None:
                guide_regions.append(tuple(float(v) for v in region))
            if matched is not None:
                detected_pts.append((float(matched[0]), float(matched[1])))
        detected = np.array(detected_pts, dtype=np.float32)
        if len(detected) < 2:
            return expected, {
                "guide_points": detected_pts,
                "guide_regions": guide_regions,
                "template_guide_points": [(float(x), float(y)) for x, y in template_pts],
            }

        n = min(len(detected), len(template_pts))
        if n < 2:
            return expected, {}
        if len(detected) != n:
            det_idx = np.linspace(0, len(detected) - 1, n).astype(int)
            detected = detected[det_idx]
        if len(template_pts) != n:
            tpl_idx = np.linspace(0, len(template_pts) - 1, n).astype(int)
            template_pts = template_pts[tpl_idx]

        order_tpl = np.argsort(template_pts[:, 1])
        order_det = np.argsort(detected[:, 1])
        template_pts = template_pts[order_tpl]
        detected = detected[order_det]

        try:
            a, b = np.polyfit(template_pts[:, 1], detected[:, 1], deg=1)
        except Exception:
            return expected, {}
        if not np.isfinite(a) or not np.isfinite(b):
            return expected, {}

        guided = expected.copy()
        guided[:, 1] = (a * expected[:, 1]) + b

        rows = max(1, int(zone.grid.rows))
        cols = max(1, int(zone.grid.cols))
        row_lines = self._build_digit_row_lines(detected[:, 1] if len(detected) else np.array([], dtype=np.float32), rows)
        anchor_layout = "interpolated"
        if len(detected) == rows + 1:
            anchor_layout = "row_tops"
        elif len(detected) >= rows + 2:
            anchor_layout = "blocker_plus_row_tops"
        if len(guided) == rows * cols:
            if len(row_lines) == rows + 1:
                row_centers = [float((row_lines[r] + row_lines[r + 1]) * 0.5) for r in range(rows)]
            else:
                row_centers = [float(np.median(guided[r * cols:(r + 1) * cols, 1])) for r in range(rows)]
            for r in range(rows):
                row_slice = slice(r * cols, (r + 1) * cols)
                guided[row_slice, 1] = row_centers[r]
            if not row_lines and len(row_centers) >= 2:
                mid_steps = np.diff(np.array(row_centers, dtype=np.float32))
                step = float(np.median(mid_steps)) if len(mid_steps) else 0.0
                row_lines = [float(row_centers[0] - (step * 0.5))]
                for center_y in row_centers:
                    row_lines.append(float(center_y + (step * 0.5)))

        fitted_line = []
        if len(template_pts):
            y0 = float(np.min(template_pts[:, 1]))
            y1 = float(np.max(template_pts[:, 1]))
            x_line = float(np.median(template_pts[:, 0]))
            fitted_line = [(x_line, float((a * y0) + b)), (x_line, float((a * y1) + b))]

        debug: dict[str, object] = {
            "guide_points": [(float(x), float(y)) for x, y in detected],
            "guide_regions": guide_regions,
            "template_guide_points": [(float(x), float(y)) for x, y in template_pts],
            "fitted_line": fitted_line,
            "row_lines": row_lines,
            "rows": rows,
            "cols": cols,
            "anchor_layout": anchor_layout,
            "fit_coefficients": {"a": float(a), "b": float(b)},
        }
        return guided.astype(np.float32), debug

    def _warp_digit_block(
        self,
        aligned_image: np.ndarray | None,
        aligned_binary: np.ndarray,
        zone: Zone,
        template: Template,
    ) -> tuple[np.ndarray | None, np.ndarray, dict[str, object]]:
        detected = np.array(self._detect_digit_anchor_ruler(aligned_binary, template, zone), dtype=np.float32)
        template_pts = np.array(self._get_manual_digit_anchor_points(template, zone), dtype=np.float32)
        debug: dict[str, object] = {
            "detected_digit_anchors": [(float(x), float(y)) for x, y in detected] if len(detected) else [],
            "template_digit_anchors": [(float(x), float(y)) for x, y in template_pts] if len(template_pts) else [],
            "warp_applied": False,
        }
        if len(detected) < 4 or len(template_pts) < 4:
            return aligned_image, aligned_binary, debug

        if len(detected) != len(template_pts):
            n = min(len(detected), len(template_pts))
            if n < 4:
                return aligned_image, aligned_binary, debug
            det_idx = np.linspace(0, len(detected) - 1, n).astype(int)
            tpl_idx = np.linspace(0, len(template_pts) - 1, n).astype(int)
            detected = detected[det_idx]
            template_pts = template_pts[tpl_idx]

        h_matrix, mask = cv2.findHomography(
            detected.astype(np.float32),
            template_pts.astype(np.float32),
            cv2.RANSAC,
            3.0,
        )
        if h_matrix is None:
            return aligned_image, aligned_binary, debug

        warped_img = (
            None
            if aligned_image is None
            else cv2.warpPerspective(
                aligned_image,
                h_matrix,
                (template.width, template.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
        warped_bin = cv2.warpPerspective(
            aligned_binary,
            h_matrix,
            (template.width, template.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        debug["warp_applied"] = True
        debug["homography_mask_inliers"] = int(mask.sum()) if mask is not None else 0
        return warped_img, warped_bin, debug

    def _detect_digit_anchor_ruler(self, binary: np.ndarray, template: Template, zone: Zone | None = None) -> list[tuple[float, float]]:
        manual_pts = self._get_manual_digit_anchor_points(template, zone)
        if len(manual_pts) == 0:
            return []
        detected: list[tuple[float, float]] = []
        for pt in manual_pts:
            matched = self._find_digit_anchor_from_manual_point(binary, pt)
            if matched is not None:
                detected.append((float(matched[0]), float(matched[1])))
        return detected

    def _select_digit_anchor_cluster(self, pts: np.ndarray, template: Template, zone: Zone | None = None) -> np.ndarray:
        pts = np.array(pts, dtype=np.float32)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = pts[np.argsort(pts[:, 1])]
        if zone is None or len(pts) <= 1:
            return pts
        zone_left = float(zone.x * template.width if zone.x <= 1.0 else zone.x)
        zone_top = float(zone.y * template.height if zone.y <= 1.0 else zone.y)
        zone_width = float(zone.width * template.width if zone.width <= 1.0 else zone.width)
        zone_height = float(zone.height * template.height if zone.height <= 1.0 else zone.height)
        zone_right = zone_left + zone_width
        zone_bottom = zone_top + zone_height
        zone_center_x = zone_left + (zone_width * 0.5)
        desired_count = max(2, int(getattr(zone.grid, "rows", 0) or 0) + 1)

        x_tol = max(10.0, template.width * 0.03)
        x_sorted = pts[np.argsort(pts[:, 0])]
        clusters: list[np.ndarray] = []
        current: list[np.ndarray] = [x_sorted[0]]
        for pt in x_sorted[1:]:
            if abs(float(pt[0]) - float(current[-1][0])) <= x_tol:
                current.append(pt)
            else:
                clusters.append(np.array(current, dtype=np.float32))
                current = [pt]
        clusters.append(np.array(current, dtype=np.float32))

        def _cluster_score(cluster: np.ndarray) -> tuple[float, float, float]:
            mean_x = float(np.mean(cluster[:, 0]))
            edge_dist = min(abs(mean_x - zone_left), abs(mean_x - zone_right), abs(mean_x - zone_center_x))
            vertical_penalty = 0.0
            if len(cluster):
                top = float(np.min(cluster[:, 1]))
                bottom = float(np.max(cluster[:, 1]))
                vertical_penalty = max(0.0, zone_top - bottom) + max(0.0, top - zone_bottom)
            count_penalty = 0.0 if len(cluster) >= desired_count else float((desired_count - len(cluster)) * max(8.0, zone_height * 0.1))
            return (edge_dist + (vertical_penalty * 0.35) + count_penalty, count_penalty, abs(len(cluster) - desired_count))

        cluster = min(clusters, key=_cluster_score)
        cluster = cluster[np.argsort(cluster[:, 1])]
        if len(clusters) == 1:
            return cluster[np.argsort(cluster[:, 1])]

        expanded_top = zone_top - max(10.0, zone_height * 0.20)
        expanded_bottom = zone_bottom + max(10.0, zone_height * 0.20)
        in_band = cluster[(cluster[:, 1] >= expanded_top) & (cluster[:, 1] <= expanded_bottom)]
        if len(in_band) >= desired_count:
            cluster = in_band
        return cluster[np.argsort(cluster[:, 1])]

    def _expand_digit_anchor_points(self, indexed_points: list[tuple[int, np.ndarray]], total_count: int) -> np.ndarray:
        if len(indexed_points) < 2:
            return np.array([pt for _, pt in indexed_points], dtype=np.float32) if indexed_points else np.empty((0, 2), dtype=np.float32)
        indexed_points = sorted(indexed_points, key=lambda item: item[0])
        series: dict[int, np.ndarray] = {idx: np.asarray(pt, dtype=np.float32) for idx, pt in indexed_points}
        for (idx_a, pt_a), (idx_b, pt_b) in zip(indexed_points, indexed_points[1:]):
            if idx_b <= idx_a + 1:
                continue
            for idx in range(idx_a + 1, idx_b):
                alpha = float((idx - idx_a) / float(idx_b - idx_a))
                series[idx] = ((1.0 - alpha) * pt_a) + (alpha * pt_b)
        keys = sorted(k for k in series.keys() if 1 <= k <= total_count)
        return np.array([series[k] for k in keys], dtype=np.float32)

    def _get_manual_digit_anchor_points(self, template: Template, zone: Zone | None = None) -> np.ndarray:
        anchors = [a for a in (template.anchors or []) if str(getattr(a, "name", "") or "").startswith("DIGIT_ANCHOR_")]
        if not anchors:
            return np.empty((0, 2), dtype=np.float32)
        indexed_points: list[tuple[int, np.ndarray]] = []
        for a in anchors:
            name = str(getattr(a, "name", "") or "")
            try:
                idx = int(name.rsplit("_", 1)[-1])
            except Exception:
                continue
            pt = np.array([
                a.x * template.width if a.x <= 1.0 else a.x,
                a.y * template.height if a.y <= 1.0 else a.y,
            ], dtype=np.float32)
            indexed_points.append((idx, pt))
        if not indexed_points:
            return np.empty((0, 2), dtype=np.float32)
        total_count = max(2, int(getattr(zone.grid, "rows", 0) or 0) + 1) if zone is not None and getattr(zone, 'grid', None) else max(idx for idx, _ in indexed_points)
        pts = self._expand_digit_anchor_points(indexed_points, total_count)
        return self._select_digit_anchor_cluster(pts, template, zone)

    def _get_detected_corner_anchor_pairs(self, binary: np.ndarray, template: Template) -> tuple[np.ndarray, np.ndarray]:
        template_pts = []
        for a in (template.anchors or []):
            if str(getattr(a, "name", "") or "").startswith("DIGIT_ANCHOR_"):
                continue
            template_pts.append([
                a.x * template.width if a.x <= 1.0 else a.x,
                a.y * template.height if a.y <= 1.0 else a.y,
            ])
        if len(template_pts) < 4:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        template_pts_arr = np.array(template_pts, dtype=np.float32)
        detected_all = np.array(self.detect_anchors(binary, max_points=120), dtype=np.float32)
        if len(detected_all) < 4:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        def _corner_set(pts: np.ndarray, width: float, height: float) -> np.ndarray:
            corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float32)
            picks = []
            for corner in corners:
                d2 = np.sum((pts - corner) ** 2, axis=1)
                picks.append(pts[int(np.argmin(d2))])
            return np.array(picks, dtype=np.float32)

        template_corner_pts = _corner_set(template_pts_arr, float(template.width), float(template.height))
        detected_corner_pts = _corner_set(detected_all, float(template.width), float(template.height))
        return template_corner_pts, detected_corner_pts

    def _find_digit_anchor_match(self, binary: np.ndarray, expected_pt: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
        h, w = binary.shape[:2]
        exp_x, exp_y = float(expected_pt[0]), float(expected_pt[1])
        search_pad_x = max(20, int(round(w * 0.025)))
        search_pad_y = max(20, int(round(h * 0.025)))
        x0 = int(max(0, exp_x - search_pad_x))
        y0 = int(max(0, exp_y - search_pad_y))
        x1 = int(min(w, exp_x + search_pad_x + 1))
        y1 = int(min(h, exp_y + search_pad_y + 1))
        roi = binary[y0:y1, x0:x1]
        region = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        if roi.size == 0:
            return None, region

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        best_pt: np.ndarray | None = None
        best_score = float("-inf")
        for idx in range(1, num_labels):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area < 18:
                continue
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if bw <= 0 or bh <= 0:
                continue
            cx = x0 + float(centroids[idx][0])
            cy = y0 + float(centroids[idx][1])
            dist = float(np.hypot(cx - exp_x, cy - exp_y))
            if dist > max(search_pad_x, search_pad_y) * 1.15:
                continue
            density = area / float(max(1, bw * bh))
            score = (area * 1.4) + (density * 18.0) - (dist * 2.2)
            if score > best_score:
                best_score = score
                best_pt = np.array([cx, cy], dtype=np.float32)

        if best_pt is not None:
            return best_pt, region

        ys, xs = np.where(roi > 0)
        if len(xs) < 16:
            return None, region
        cx = x0 + float(np.mean(xs))
        cy = y0 + float(np.mean(ys))
        if float(np.hypot(cx - exp_x, cy - exp_y)) > max(search_pad_x, search_pad_y) * 1.15:
            return None, region
        return np.array([cx, cy], dtype=np.float32), region

    def _find_digit_anchor_from_manual_point(self, binary: np.ndarray, expected_pt: np.ndarray) -> np.ndarray | None:
        best_pt, _ = self._find_digit_anchor_match(binary, expected_pt)
        return best_pt

    def _resolve_column_digit_centers(
        self,
        binary: np.ndarray,
        expected: np.ndarray,
        grid,
        bubble_radius: float,
    ) -> np.ndarray:
        rows, cols = max(1, int(grid.rows)), max(1, int(grid.cols))
        if len(expected) != rows * cols:
            return expected
        min_area = max(6.0, 0.25 * np.pi * (bubble_radius ** 2))
        max_area = max(min_area * 3.5, 4.0 * np.pi * (bubble_radius ** 2))
        search_pad = max(10, int(round(bubble_radius * 2.4)))
        max_dist = max(6.0, bubble_radius * 1.75)
        refined = expected.copy()
        for c in range(cols):
            col_pts = expected[c::cols]
            offsets_x: list[float] = []
            for exp_x, exp_y in col_pts:
                best_offset = self._find_local_component_offset(
                    binary,
                    np.array([exp_x, exp_y], dtype=np.float32),
                    search_pad,
                    min_area,
                    max_area,
                    max_dist,
                )
                if best_offset is not None:
                    offsets_x.append(float(best_offset[0]))
            if not offsets_x:
                continue
            median_shift_x = float(np.median(np.array(offsets_x, dtype=np.float32)))
            median_shift_x = float(np.clip(median_shift_x, -3.0, 3.0))
            for r in range(rows):
                idx = (r * cols) + c
                refined[idx, 0] = expected[idx, 0] + median_shift_x

        refined_centers = refined.astype(np.float32)
        return refined_centers

    def _find_local_component(
        self,
        binary: np.ndarray,
        center: np.ndarray,
        search_pad: int,
        min_area: float,
        max_area: float,
        max_dist: float,
    ) -> dict[str, float] | None:
        h, w = binary.shape[:2]
        exp_x, exp_y = float(center[0]), float(center[1])
        x0 = int(max(0, exp_x - search_pad))
        y0 = int(max(0, exp_y - search_pad))
        x1 = int(min(w, exp_x + search_pad + 1))
        y1 = int(min(h, exp_y + search_pad + 1))
        roi = binary[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        best: dict[str, float] | None = None
        best_score = float("inf")
        for idx in range(1, num_labels):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            cx, cy = centroids[idx]
            cand = np.array([cx + x0, cy + y0], dtype=np.float32)
            offset = cand - np.array([exp_x, exp_y], dtype=np.float32)
            dist = float(np.linalg.norm(offset))
            if dist > max_dist:
                continue
            bw = max(1.0, float(stats[idx, cv2.CC_STAT_WIDTH]))
            bh = max(1.0, float(stats[idx, cv2.CC_STAT_HEIGHT]))
            extent = float(area / max(1.0, bw * bh))
            aspect = float(min(bw, bh) / max(bw, bh))
            score = dist + (0.35 * search_pad * max(0.0, 0.45 - (aspect * extent)))
            if score < best_score:
                best_score = score
                best = {
                    "offset_x": float(offset[0]),
                    "offset_y": float(offset[1]),
                    "distance": dist,
                    "area": area,
                    "width": bw,
                    "height": bh,
                    "extent": extent,
                    "aspect": aspect,
                }
        return best

    def _find_local_component_offset(
        self,
        binary: np.ndarray,
        center: np.ndarray,
        search_pad: int,
        min_area: float,
        max_area: float,
        max_dist: float,
    ) -> np.ndarray | None:
        component = self._find_local_component(binary, center, search_pad, min_area, max_area, max_dist)
        if component is None:
            return None
        return np.array([component["offset_x"], component["offset_y"]], dtype=np.float32)

    def _detect_digit_zone_component_marks(
        self,
        binary: np.ndarray,
        centers: np.ndarray,
        bubble_radius: int,
    ) -> np.ndarray:
        detected = np.zeros(len(centers), dtype=np.float32)
        if len(centers) == 0:
            return detected
        min_area = max(5.0, 0.14 * np.pi * (bubble_radius ** 2))
        max_area = max(min_area * 6.0, 6.0 * np.pi * (bubble_radius ** 2))
        search_pad = max(10, int(round(bubble_radius * 2.6)))
        max_dist = max(8.0, bubble_radius * 2.2)
        target_area = max(min_area, 0.65 * np.pi * (bubble_radius ** 2))
        for idx, center in enumerate(np.asarray(centers, dtype=np.float32)):
            component = self._find_local_component(binary, center, search_pad, min_area, max_area, max_dist)
            if component is None:
                continue
            dist_score = float(np.clip(1.0 - (component["distance"] / max(max_dist, 1e-6)), 0.0, 1.0))
            shape_score = float(np.clip((component["aspect"] ** 0.5) * (component["extent"] / 0.45), 0.0, 1.0))
            area_score = float(np.clip(component["area"] / max(target_area, 1.0), 0.0, 1.0))
            detected[idx] = float(np.clip((0.45 * dist_score) + (0.35 * shape_score) + (0.20 * area_score), 0.0, 1.0))
        return detected

    def _refit_digit_grid_from_clear_points(
        self,
        binary: np.ndarray,
        centers: np.ndarray,
        grid,
        bubble_radius: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        rows, cols = max(1, int(getattr(grid, "rows", 1) or 1)), max(1, int(getattr(grid, "cols", 1) or 1))
        expected = np.asarray(centers, dtype=np.float32)
        if expected.shape != (rows * cols, 2):
            return expected, {}
        multi_scores = self._detect_digit_zone_multi_probe_marks(binary, expected, int(round(bubble_radius)))
        core_scores = self._detect_center_core_marks(binary, expected, int(round(bubble_radius)))
        combined_scores = np.maximum(multi_scores, core_scores)
        strong_threshold = max(0.52, float(np.mean(combined_scores) + (0.55 * np.std(combined_scores))))
        min_area = max(5.0, 0.15 * np.pi * (bubble_radius ** 2))
        max_area = max(min_area * 6.0, 6.0 * np.pi * (bubble_radius ** 2))
        search_pad = max(10, int(round(bubble_radius * 2.8)))
        max_dist = max(8.0, bubble_radius * 2.4)
        observations: list[tuple[int, int, float, float, float]] = []
        for idx, score in enumerate(combined_scores.tolist()):
            if float(score) < strong_threshold:
                continue
            offset = self._find_local_component_offset(binary, expected[idx], search_pad, min_area, max_area, max_dist)
            if offset is None:
                continue
            row = idx // cols
            col = idx % cols
            observed = expected[idx] + offset
            observations.append((row, col, float(observed[0]), float(observed[1]), float(score)))
        if not observations:
            return expected, {"grid_fit_observations": [], "grid_fit_applied": False}

        default_x = np.median(expected.reshape(rows, cols, 2)[:, :, 0], axis=0)
        default_y = np.median(expected.reshape(rows, cols, 2)[:, :, 1], axis=1)
        default_x_step = float(np.median(np.diff(default_x))) if cols > 1 else 0.0
        default_y_step = float(np.median(np.diff(default_y))) if rows > 1 else 0.0

        def _fit_axis(indices: list[int], values: list[float], default_step: float) -> tuple[float, float]:
            uniq = sorted(set(indices))
            if len(uniq) >= 2 and len(set(indices)) >= 2:
                step, base = np.polyfit(np.asarray(indices, dtype=np.float32), np.asarray(values, dtype=np.float32), deg=1)
                if not np.isfinite(step) or not np.isfinite(base):
                    step = default_step
                    base = float(np.median(np.asarray(values, dtype=np.float32) - (np.asarray(indices, dtype=np.float32) * default_step)))
            else:
                step = default_step
                base = float(values[0] - (indices[0] * default_step))
            return float(base), float(step)

        row_idx = [row for row, _, _, _, _ in observations]
        col_idx = [col for _, col, _, _, _ in observations]
        obs_x = [x for _, _, x, _, _ in observations]
        obs_y = [y for _, _, _, y, _ in observations]
        fit_x0, fit_x_step = _fit_axis(col_idx, obs_x, default_x_step)
        fit_y0, fit_y_step = _fit_axis(row_idx, obs_y, default_y_step)
        if cols > 1 and default_x_step > 0.0 and abs(fit_x_step - default_x_step) > (0.35 * abs(default_x_step)):
            fit_x_step = default_x_step
            fit_x0 = float(np.median(np.asarray(obs_x, dtype=np.float32) - (np.asarray(col_idx, dtype=np.float32) * fit_x_step)))
        if rows > 1 and default_y_step > 0.0 and abs(fit_y_step - default_y_step) > (0.35 * abs(default_y_step)):
            fit_y_step = default_y_step
            fit_y0 = float(np.median(np.asarray(obs_y, dtype=np.float32) - (np.asarray(row_idx, dtype=np.float32) * fit_y_step)))

        rebuilt = expected.copy()
        for row in range(rows):
            for col in range(cols):
                idx = (row * cols) + col
                rebuilt[idx, 0] = fit_x0 + (col * fit_x_step)
                rebuilt[idx, 1] = fit_y0 + (row * fit_y_step)
        debug = {
            "grid_fit_applied": True,
            "grid_fit_observations": [(int(r), int(c), float(x), float(y), float(s)) for r, c, x, y, s in observations],
            "grid_fit_threshold": float(strong_threshold),
            "grid_fit_x": {"x0": float(fit_x0), "step": float(fit_x_step)},
            "grid_fit_y": {"y0": float(fit_y0), "step": float(fit_y_step)},
        }
        return rebuilt.astype(np.float32), debug

    def _detect_digit_bubble_centers(
        self,
        binary: np.ndarray,
        centers: np.ndarray,
        bubble_radius: float,
    ) -> np.ndarray:
        detected = np.asarray(centers, dtype=np.float32).copy()
        if detected.size == 0:
            return detected
        min_area = max(6.0, 0.25 * np.pi * (bubble_radius ** 2))
        max_area = max(min_area * 3.5, 4.0 * np.pi * (bubble_radius ** 2))
        search_pad = max(10, int(round(bubble_radius * 2.4)))
        max_dist = max(6.0, bubble_radius * 1.75)
        for idx, center in enumerate(detected):
            best_offset = self._find_local_component_offset(
                binary,
                center,
                search_pad,
                min_area,
                max_area,
                max_dist,
            )
            if best_offset is not None:
                detected[idx] = center + best_offset
        return detected.astype(np.float32)

    def _sample_digit_grid(
        self,
        binary: np.ndarray,
        centers: np.ndarray,
        rows: int,
        cols: int,
        bubble_radius: float,
    ) -> np.ndarray:
        if rows <= 0 or cols <= 0:
            return np.zeros((max(1, rows), max(1, cols)), dtype=np.float32)
        blurred = cv2.GaussianBlur(binary, (5, 5), 0)
        sample_r = max(3, int(round(bubble_radius * 0.9)))
        bg_r = max(sample_r + 2, int(round(bubble_radius * 1.6)))
        yy, xx = np.ogrid[-bg_r : bg_r + 1, -bg_r : bg_r + 1]
        inner_mask = (xx * xx + yy * yy) <= (sample_r * sample_r)
        ring_mask = ((xx * xx + yy * yy) <= (bg_r * bg_r)) & (~inner_mask)
        h, w = blurred.shape[:2]
        mat = np.zeros((rows, cols), dtype=np.float32)
        for idx, (cx, cy) in enumerate(np.asarray(centers[: rows * cols], dtype=np.float32)):
            x0 = max(0, int(round(cx)) - bg_r)
            y0 = max(0, int(round(cy)) - bg_r)
            x1 = min(w, int(round(cx)) + bg_r + 1)
            y1 = min(h, int(round(cy)) + bg_r + 1)
            roi = blurred[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            mask_y0 = bg_r - (int(round(cy)) - y0)
            mask_x0 = bg_r - (int(round(cx)) - x0)
            mask_y1 = mask_y0 + roi.shape[0]
            mask_x1 = mask_x0 + roi.shape[1]
            inner = inner_mask[mask_y0:mask_y1, mask_x0:mask_x1]
            ring = ring_mask[mask_y0:mask_y1, mask_x0:mask_x1]
            inner_mean = float(np.mean(roi[inner])) / 255.0 if np.any(inner) else 0.0
            ring_mean = float(np.mean(roi[ring])) / 255.0 if np.any(ring) else 0.0
            score = float(np.clip(inner_mean - (0.5 * ring_mean), 0.0, 1.0))
            r = idx // cols
            c = idx % cols
            if r < rows and c < cols:
                mat[r, c] = score
        return mat

    def _preprocess_digit_sampling(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    def _digit_zone_bbox(self, zone: Zone, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = image_shape[:2]
        zx = float(zone.x * w) if zone.x <= 1.0 else float(zone.x)
        zy = float(zone.y * h) if zone.y <= 1.0 else float(zone.y)
        zw = float(zone.width * w) if zone.width <= 1.0 else float(zone.width)
        zh = float(zone.height * h) if zone.height <= 1.0 else float(zone.height)
        zone_bbox = (int(round(zx)), int(round(zy)), int(round(zw)), int(round(zh)))
        grid = getattr(zone, "grid", None)
        positions = getattr(grid, "bubble_positions", None) or []
        if not positions:
            return zone_bbox
        xs = [float(x * w) if x <= 1.0 else float(x) for x, _ in positions]
        ys = [float(y * h) if y <= 1.0 else float(y) for _, y in positions]
        if not xs or not ys:
            return zone_bbox
        cols = max(1, int(getattr(grid, "cols", 1) or 1))
        rows = 10 if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK) else max(1, int(getattr(grid, "rows", 1) or 1))
        x_pad = ((max(xs) - min(xs)) / max(1, cols - 1)) * 0.5 if cols > 1 else max(6.0, zw * 0.1)
        y_pad = ((max(ys) - min(ys)) / max(1, rows - 1)) * 0.5 if rows > 1 else max(6.0, zh * 0.1)
        x0 = max(zone_bbox[0], int(round(min(xs) - x_pad)))
        y0 = max(zone_bbox[1], int(round(min(ys) - y_pad)))
        x1 = min(zone_bbox[0] + zone_bbox[2], int(round(max(xs) + x_pad)))
        y1 = min(zone_bbox[1] + zone_bbox[3], int(round(max(ys) + y_pad)))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def _sample_digit_cell(self, img: np.ndarray, cx: float, cy: float, cell_w: float, cell_h: float) -> float:
        outer_r = max(2, int(min(cell_w, cell_h) * 0.26))
        inner_r = max(1, int(round(outer_r * 0.55)))

        h, w = img.shape[:2]
        x1 = max(int(cx - outer_r), 0)
        x2 = min(int(cx + outer_r + 1), w)
        y1 = max(int(cy - outer_r), 0)
        y2 = min(int(cy + outer_r + 1), h)

        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0

        _, th = cv2.threshold(
            patch,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        yy, xx = np.indices(th.shape)
        local_cx = float(cx - x1)
        local_cy = float(cy - y1)
        dist2 = (xx - local_cx) ** 2 + (yy - local_cy) ** 2
        core_mask = dist2 <= float(inner_r ** 2)
        ring_mask = (dist2 > float(inner_r ** 2)) & (dist2 <= float(outer_r ** 2))

        if not np.any(core_mask):
            return 0.0
        core_density = float(np.count_nonzero(th[core_mask])) / float(np.count_nonzero(core_mask))
        ring_density = 0.0
        if np.any(ring_mask):
            ring_density = float(np.count_nonzero(th[ring_mask])) / float(np.count_nonzero(ring_mask))
        patch_density = float(np.count_nonzero(th)) / float(th.size)
        search_step = max(1, inner_r // 2)
        peak_density = core_density
        for dy in range(-search_step, search_step + 1, search_step):
            for dx in range(-search_step, search_step + 1, search_step):
                shifted_dist2 = (xx - (local_cx + dx)) ** 2 + (yy - (local_cy + dy)) ** 2
                shifted_core = shifted_dist2 <= float(inner_r ** 2)
                if np.any(shifted_core):
                    peak_density = max(
                        peak_density,
                        float(np.count_nonzero(th[shifted_core])) / float(np.count_nonzero(shifted_core)),
                    )
        score = (0.60 * core_density) + (0.25 * peak_density) + (0.15 * patch_density) - (0.45 * ring_density)
        return float(np.clip(score, 0.0, 1.0))

    def _evaluate_digit_grid_offset(
        self,
        img: np.ndarray,
        points: np.ndarray,
        num_cols: int,
        num_rows: int,
        cell_w: float,
        cell_h: float,
        dx: float,
        dy: float,
    ) -> float:
        total = 0.0
        for col in range(num_cols):
            col_max = 0.0
            for row in range(num_rows):
                cx, cy = points[row, col]
                val = self._sample_digit_cell(img, float(cx + (dx * cell_w)), float(cy + (dy * cell_h)), cell_w, cell_h)
                col_max = max(col_max, float(val))
            total += col_max
        return float(total)

    def _auto_align_digit_grid(
        self,
        img: np.ndarray,
        points: np.ndarray,
        num_cols: int,
        num_rows: int,
        cell_w: float,
        cell_h: float,
    ) -> tuple[float, float, float]:
        best_dx = 0.0
        best_dy = 0.0
        best_score = -1.0
        for dx in np.linspace(-0.2, 0.2, 9):
            for dy in np.linspace(-0.2, 0.2, 9):
                score = self._evaluate_digit_grid_offset(img, points, num_cols, num_rows, cell_w, cell_h, float(dx), float(dy))
                if score > best_score:
                    best_score = float(score)
                    best_dx = float(dx)
                    best_dy = float(dy)
        return best_dx, best_dy, best_score

    def _read_digit_grid_sampling(
        self,
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
        num_cols: int,
        num_rows: int,
        threshold: float,
        centers: np.ndarray | None = None,
    ) -> tuple[list[int | None | str], np.ndarray, dict[str, object]]:
        x0, y0, bw, bh = bbox
        cell_w = bw / float(max(1, num_cols))
        cell_h = bh / float(max(1, num_rows))
        mat = np.zeros((num_rows, num_cols), dtype=np.float32)
        debug_centers: list[tuple[float, float]] = []
        sampled_points: list[tuple[float, float]] = []
        center_array = None
        if centers is not None:
            arr = np.asarray(centers, dtype=np.float32)
            if arr.shape == (num_rows * num_cols, 2):
                center_array = arr.reshape(num_rows, num_cols, 2)
        if center_array is not None:
            base_points = center_array.astype(np.float32)
        else:
            base_points = np.zeros((num_rows, num_cols, 2), dtype=np.float32)
            for col in range(num_cols):
                for row in range(num_rows):
                    base_points[row, col, 0] = float(x0 + (col * cell_w) + (cell_w * 0.5))
                    base_points[row, col, 1] = float(y0 + (row * cell_h) + (cell_h * 0.5))

        offset_x, offset_y, align_score = self._auto_align_digit_grid(img, base_points, num_cols, num_rows, cell_w, cell_h)

        for col in range(num_cols):
            for row in range(num_rows):
                cx = float(base_points[row, col, 0] + (offset_x * cell_w))
                cy = float(base_points[row, col, 1] + (offset_y * cell_h))
                debug_centers.append((float(base_points[row, col, 0]), float(base_points[row, col, 1])))
                sampled_points.append((cx, cy))
                mat[row, col] = self._sample_digit_cell(img, cx, cy, cell_w, cell_h)
        results: list[int | None | str] = []
        recognized_points: list[tuple[float, float]] = []
        for col in range(num_cols):
            column_scores = [(row, float(mat[row, col])) for row in range(num_rows)]
            if self.debug_mode:
                print(f"Column {col} -> {column_scores}")
            best_row = int(np.argmax([score for _, score in column_scores])) if column_scores else None
            scores = [score for _, score in column_scores]
            sorted_scores = sorted(scores, reverse=True) if scores else []
            top_score = float(sorted_scores[0]) if sorted_scores else 0.0
            second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            margin_ok = top_score >= (second_score + 0.08)
            ratio_ok = second_score <= 1e-6 or top_score >= (1.18 * second_score)
            if best_row is None or not scores or top_score < float(threshold) or not margin_ok or not ratio_ok:
                results.append(None)
            else:
                results.append(best_row)
                recognized_points.append((
                    float(base_points[best_row, col, 0] + (offset_x * cell_w)),
                    float(base_points[best_row, col, 1] + (offset_y * cell_h)),
                ))
        debug = {
            "bbox": bbox,
            "centers": debug_centers,
            "sample_points": sampled_points,
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "alignment_score": float(align_score),
            "row_lines": [float(y0 + (row * cell_h) + (cell_h * 0.5)) for row in range(num_rows)],
            "col_lines": [float(x0 + (col * cell_w) + (cell_w * 0.5)) for col in range(num_cols)],
            "scores": mat.tolist(),
            "recognized_points": recognized_points,
        }
        return results, mat, debug

    def _validate_sampled_digits(self, digits: list[int | None | str]) -> bool:
        return not any(d == "INVALID" for d in digits)

    def _decode_sampled_digit_grid(self, mat: np.ndarray, zone: Zone, grid, result: OMRResult) -> tuple[str, list[float]]:
        rows, cols = mat.shape
        default_digit_map = list(range(10)) if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK) else list(range(rows))
        digit_map = zone.metadata.get("digit_map", default_digit_map)
        digits: list[str] = []
        confs: list[float] = []
        for c in range(cols):
            col_scores = np.asarray(mat[:, c], dtype=np.float32)
            col_threshold = max(0.18, float(np.mean(col_scores) + (0.5 * np.std(col_scores))))
            filled_rows = [idx for idx, score in enumerate(col_scores.tolist()) if float(score) > col_threshold]
            order = np.argsort(col_scores)[::-1]
            top_i = int(order[0]) if len(order) else 0
            second_i = int(order[1]) if len(order) > 1 else top_i
            top = float(col_scores[top_i]) if len(order) else 0.0
            second = float(col_scores[second_i]) if len(order) > 1 else 0.0
            confs.append(1.0 if len(filled_rows) == 1 else 0.0)
            if len(filled_rows) == 1:
                mapped = digit_map[filled_rows[0]] if filled_rows[0] < len(digit_map) else filled_rows[0]
                digits.append(str(mapped))
            elif len(filled_rows) >= 2:
                if second > 0.0 and top > (1.4 * second):
                    mapped = digit_map[top_i] if top_i < len(digit_map) else top_i
                    digits.append(str(mapped))
                else:
                    digits.append("?")
                    result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: multiple marks")
            else:
                digits.append("?")
                result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: blank")
        return "".join(digits), confs


    def _digit_offset_candidates(self) -> list[tuple[float, float]]:
        return [
            (0.0, 0.0),
            (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0),
            (2.0, 2.0), (2.0, -2.0), (-2.0, 2.0), (-2.0, -2.0),
        ]

    def _shift_digit_centers(self, centers: np.ndarray, dx: float, dy: float) -> np.ndarray:
        shifted = np.asarray(centers, dtype=np.float32).copy()
        if shifted.size == 0:
            return shifted
        shifted[:, 0] += float(dx)
        shifted[:, 1] += float(dy)
        return shifted

    def _read_digit_zone_with_offset_fallback(
        self,
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
        cols: int,
        rows: int,
        threshold: float,
        centers: np.ndarray,
    ) -> tuple[list[int | None | str], np.ndarray, dict[str, object]]:
        fallback_debug: dict[str, object] = {"offset_trials": []}
        last_digits: list[int | None | str] = []
        last_mat = np.zeros((rows, cols), dtype=np.float32)
        last_debug: dict[str, object] = {}
        for dx, dy in self._digit_offset_candidates():
            shifted_centers = self._shift_digit_centers(centers, dx, dy)
            digits, mat, sampling_debug = self._read_digit_grid_sampling(
                img,
                bbox,
                cols,
                rows,
                threshold=threshold,
                centers=shifted_centers,
            )
            trial_complete = len(digits) == cols and all(isinstance(d, int) for d in digits)
            fallback_debug["offset_trials"].append({
                "dx": float(dx),
                "dy": float(dy),
                "complete": bool(trial_complete),
                "digits": [None if d is None else int(d) if isinstance(d, int) else str(d) for d in digits],
            })
            last_digits, last_mat, last_debug = digits, mat, sampling_debug
            if trial_complete:
                sampling_debug["applied_center_offset"] = {"dx": float(dx), "dy": float(dy)}
                return digits, mat, sampling_debug | fallback_debug
        if last_debug is not None:
            last_debug["applied_center_offset"] = {"dx": 0.0, "dy": 0.0}
        return last_digits, last_mat, (last_debug | fallback_debug)

    def _decode_identifier_zone_from_centers(
        self,
        binary: np.ndarray,
        zone: Zone,
        template: Template,
        result: OMRResult,
        centers: np.ndarray,
        radius: int,
    ) -> tuple[str, list[float], np.ndarray, np.ndarray, dict[str, object]]:
        rows = 10
        cols = max(1, int(zone.grid.cols))
        need = rows * cols
        refined_centers = self._detect_digit_bubble_centers(binary, centers, float(radius))
        ratios = self.detect_bubbles(binary, refined_centers, radius)
        core_ratios = self._detect_center_core_marks(binary, refined_centers, radius)
        quick_ratios = np.clip((0.55 * ratios) + (0.45 * core_ratios), 0.0, 1.0)
        quick_mat = self._cluster_digit_columns(refined_centers, quick_ratios, rows, cols, 0.0)
        class _TmpResult:
            def __init__(self):
                self.recognition_errors: list[str] = []
                self.confidence_scores: dict[str, float] = {}
        tmp_result = _TmpResult()
        quick_value, quick_confs = self._decode_column_digits(quick_mat, zone, zone.grid, tmp_result)
        quick_valid = (
            zone.zone_type == ZoneType.EXAM_CODE_BLOCK
            and
            isinstance(quick_value, str)
            and len(quick_value) == cols
            and ("?" not in quick_value)
            and (float(np.mean(quick_confs or [0.0])) >= 0.55)
        )
        if quick_valid:
            debug = {
                "centers": [(float(x), float(y)) for x, y in refined_centers],
                "direct_scores": quick_mat.tolist(),
                "direct_local_fill": [],
                "direct_radius": int(radius),
                "peak_window_scores": [],
                "component_scores": [],
                "fast_path_used": True,
            }
            return quick_value, quick_confs, refined_centers, quick_mat, debug
        advanced_mode = bool(((template.metadata or {}).get("digit_advanced_mode", False) if getattr(template, "metadata", None) else False))
        multi_probe_ratios = self._detect_digit_zone_multi_probe_marks(binary, refined_centers, radius) if advanced_mode else np.zeros_like(ratios)
        peak_window_ratios = np.zeros_like(ratios)
        component_ratios = self._detect_digit_zone_component_marks(binary, refined_centers, radius) if advanced_mode else np.zeros_like(ratios)
        if zone.zone_type == ZoneType.STUDENT_ID_BLOCK:
            square_ratios = self._detect_square_mark_density(binary, refined_centers, radius)
            eroded_ratios = self._detect_eroded_mark_density(binary, refined_centers, radius)
            legacy_ratios = np.clip((0.20 * ratios) + (0.32 * core_ratios) + (0.26 * square_ratios) + (0.22 * eroded_ratios), 0.0, 1.0)
            widened_ratios = np.clip((0.06 * ratios) + (0.14 * core_ratios) + (0.16 * square_ratios) + (0.14 * eroded_ratios) + (0.18 * multi_probe_ratios) + (0.14 * peak_window_ratios) + (0.18 * component_ratios), 0.0, 1.0) if advanced_mode else legacy_ratios
            ratios = np.maximum(legacy_ratios, widened_ratios)
        else:
            legacy_ratios = np.clip((0.58 * ratios) + (0.42 * core_ratios), 0.0, 1.0)
            widened_ratios = np.clip((0.18 * ratios) + (0.16 * core_ratios) + (0.22 * multi_probe_ratios) + (0.18 * peak_window_ratios) + (0.26 * component_ratios), 0.0, 1.0) if advanced_mode else legacy_ratios
            ratios = np.maximum(legacy_ratios, widened_ratios)
        weak_mask = ratios < max(self.empty_threshold + 0.16, min(self.fill_threshold * 0.82, 0.44)) if advanced_mode else np.zeros_like(ratios, dtype=bool)
        if advanced_mode and np.any(weak_mask):
            peak_window_ratios = self._detect_digit_zone_peak_window_marks(binary, refined_centers, radius)
            rescue_scores = np.clip((0.55 * ratios) + (0.45 * peak_window_ratios), 0.0, 1.0)
            ratios = np.where(weak_mask, np.maximum(ratios, rescue_scores), ratios)
        dynamic_thresholds = np.array([
            self._estimate_local_fill_threshold(binary, center, radius, self.fill_threshold)
            for center in refined_centers
        ], dtype=np.float32)
        if len(ratios) < need:
            ratios = np.pad(ratios, (0, need - len(ratios)), constant_values=0.0)
            dynamic_thresholds = np.pad(dynamic_thresholds, (0, need - len(dynamic_thresholds)), constant_values=self.fill_threshold)
        mat = self._cluster_digit_columns(refined_centers, ratios, rows, cols, 0.0)
        local_fill = self._cluster_digit_columns(refined_centers, dynamic_thresholds, rows, cols, self.fill_threshold)
        mat = np.clip(mat - (0.5 * np.clip(local_fill - self.empty_threshold, 0.0, 1.0)), 0.0, 1.0)

        orig_fill = self.fill_threshold
        try:
            self.fill_threshold = float(np.clip(np.median(local_fill), orig_fill - 0.08, orig_fill + 0.08))
            value, confs = self._decode_column_digits(mat, zone, zone.grid, result)
        finally:
            self.fill_threshold = orig_fill
        debug = {
            "centers": [(float(x), float(y)) for x, y in refined_centers],
            "direct_scores": mat.tolist(),
            "direct_local_fill": local_fill.tolist(),
            "direct_radius": int(radius),
            "peak_window_scores": peak_window_ratios.tolist(),
            "component_scores": component_ratios.tolist(),
            "fast_path_used": False,
        }
        return value, confs, refined_centers, mat, debug

    @staticmethod
    def _identifier_expected_len(key: str, zone: Zone) -> int:
        if key == "student_id":
            return 8
        if key == "exam_code":
            return 4
        return max(1, int(getattr(zone.grid, "cols", 1) or 1))

    def _decode_identifier_by_anchor_axis(
        self,
        binary: np.ndarray,
        zone: Zone,
        template: Template,
        centers: np.ndarray,
        radius: int,
    ) -> tuple[str, list[float], dict[str, object]]:
        rows = 10
        cols = max(1, int(zone.grid.cols or 1))
        if centers.size == 0:
            return "", [], {"axis_mode": "empty_centers"}

        scores = self.detect_bubbles(binary, centers, radius)
        core_scores = self._detect_center_core_marks(binary, centers, radius)
        component_scores = self._detect_digit_zone_component_marks(binary, centers, radius)
        mark_scores = np.clip((0.30 * scores) + (0.30 * core_scores) + (0.40 * component_scores), 0.0, 1.0)

        detected_anchors = np.array(self._detect_digit_anchor_ruler(binary, template, zone), dtype=np.float32)
        if len(detected_anchors) < 2:
            detected_anchors = np.array(self._get_manual_digit_anchor_points(template, zone), dtype=np.float32)

        axis_mode = "anchor_ruler"
        if len(detected_anchors) >= 2:
            order = np.argsort(detected_anchors[:, 1])
            axis_start = detected_anchors[order[0]]
            axis_end = detected_anchors[order[-1]]
        else:
            axis_mode = "pca_fallback"
            pts = centers.astype(np.float32)
            mean_pt = np.mean(pts, axis=0)
            cov = np.cov((pts - mean_pt).T) if len(pts) >= 2 else np.eye(2, dtype=np.float32)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            vec = eig_vecs[:, int(np.argmax(eig_vals))]
            if float(abs(vec[1])) < float(abs(vec[0])):
                vec = np.array([0.0, 1.0], dtype=np.float32)
            if vec[1] < 0:
                vec = -vec
            axis_start = mean_pt - (120.0 * vec)
            axis_end = mean_pt + (120.0 * vec)

        axis_vec = (axis_end - axis_start).astype(np.float32)
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm <= 1e-6:
            return "", [], {"axis_mode": "degenerate_axis"}
        axis_y = axis_vec / axis_norm
        axis_x = np.array([axis_y[1], -axis_y[0]], dtype=np.float32)
        rel = centers.astype(np.float32) - axis_start
        proj_y = rel @ axis_y
        proj_x = rel @ axis_x

        x_order = np.argsort(proj_x)
        x_chunks = np.array_split(proj_x[x_order], cols)
        col_centers = np.array([float(np.median(ch)) if len(ch) else 0.0 for ch in x_chunks], dtype=np.float32)
        col_indices = np.array([int(np.argmin(np.abs(col_centers - px))) for px in proj_x], dtype=np.int32)

        if len(detected_anchors) >= 2:
            anchor_proj = np.sort(((detected_anchors - axis_start) @ axis_y).astype(np.float32))
            if len(anchor_proj) >= rows + 1:
                pick = np.linspace(0, len(anchor_proj) - 1, rows + 1).round().astype(int)
                anchor_proj = anchor_proj[pick]
                row_proj = 0.5 * (anchor_proj[:-1] + anchor_proj[1:])
            else:
                lo, hi = float(np.min(anchor_proj)), float(np.max(anchor_proj))
                step = (hi - lo) / float(max(rows, 1))
                row_proj = np.array([lo + ((idx + 0.5) * step) for idx in range(rows)], dtype=np.float32)
        else:
            y_order = np.argsort(proj_y)
            y_chunks = np.array_split(proj_y[y_order], rows)
            row_proj = np.array([float(np.median(ch)) if len(ch) else 0.0 for ch in y_chunks], dtype=np.float32)

        mat = np.zeros((rows, cols), dtype=np.float32)
        for idx, score in enumerate(mark_scores):
            c_idx = int(col_indices[idx])
            r_idx = int(np.argmin(np.abs(row_proj - proj_y[idx])))
            if 0 <= r_idx < rows and 0 <= c_idx < cols:
                mat[r_idx, c_idx] = max(float(mat[r_idx, c_idx]), float(score))

        digit_map = zone.metadata.get("digit_map", list(range(rows)))
        digits: list[str] = []
        confs: list[float] = []
        threshold = max(0.22, float(self.fill_threshold) * 0.72)
        for c_idx in range(cols):
            col = mat[:, c_idx]
            top_idx = int(np.argmax(col))
            top = float(col[top_idx])
            second = float(np.partition(col, -2)[-2]) if len(col) > 1 else 0.0
            if top < threshold or (second >= top * 0.92):
                digits.append("?")
                confs.append(max(0.0, top - second))
                continue
            mapped = digit_map[top_idx] if top_idx < len(digit_map) else top_idx
            digits.append(str(mapped))
            confs.append(max(0.0, top - second))

        debug = {
            "axis_mode": axis_mode,
            "axis_line": [(float(axis_start[0]), float(axis_start[1])), (float(axis_end[0]), float(axis_end[1]))],
            "axis_col_centers": [float(x) for x in col_centers],
            "axis_row_centers": [float(y) for y in row_proj],
            "axis_scores": mat.tolist(),
        }
        return "".join(digits), confs, debug

    def _finalize_identifier_value(
        self,
        zone: Zone,
        key: str,
        raw_digits: list[int | None | str] | None = None,
        value: str | None = None,
        confs: list[float] | None = None,
        result: OMRResult | None = None,
    ) -> tuple[str, list[float]]:
        rows = 10
        if raw_digits is not None:
            digit_map = zone.metadata.get("digit_map", list(range(rows)))
            mapped_digits: list[int | None | str] = []
            for d in raw_digits:
                if isinstance(d, int):
                    mapped_digits.append(digit_map[d] if 0 <= d < len(digit_map) else d)
                else:
                    mapped_digits.append(d)
            conf_list = [1.0 if isinstance(d, int) else 0.0 for d in mapped_digits]
            complete_digits = len(mapped_digits) == max(1, int(zone.grid.cols)) and all(isinstance(d, int) for d in mapped_digits)
            final_value = "".join(str(d) for d in mapped_digits) if complete_digits and self._validate_sampled_digits(mapped_digits) else "-"
            if final_value == "-" and result is not None:
                result.recognition_errors.append(f"{zone.zone_type.value}: LOW_CONFIDENCE")
        else:
            final_value = value or ""
            conf_list = list(confs or [])
            expected_len = self._identifier_expected_len(key, zone)
            if len(final_value) > expected_len:
                final_value = final_value[:expected_len]
            missing_digits = final_value.count("?")
            is_valid = len(final_value) == expected_len and missing_digits == 0
            global_score = float(expected_len - missing_digits) / float(expected_len or 1)
            valid_without_ambiguity = len(final_value) == expected_len and ("?" not in final_value)
            if not is_valid and valid_without_ambiguity:
                is_valid = True
            if (not is_valid or global_score < 0.60) and result is not None:
                result.recognition_errors.append(f"{zone.zone_type.value}: LOW_CONFIDENCE")
                if not is_valid:
                    result.recognition_errors.append(f"{zone.zone_type.value}: invalid length or ambiguous digit sequence")
                    if key == "student_id":
                        result.recognition_errors.append(f"{zone.zone_type.value}: Lỗi SBD")
                    elif key == "exam_code":
                        result.recognition_errors.append(f"{zone.zone_type.value}: Lỗi Mã đề")
                    final_value = ""
        if key == "student_id" and final_value and final_value != "-":
            longest_run = 1
            current_run = 1
            for idx in range(1, len(final_value)):
                if final_value[idx] == final_value[idx - 1]:
                    current_run += 1
                    longest_run = max(longest_run, current_run)
                else:
                    current_run = 1
            if len(set(final_value)) == 1 or (len(final_value) >= 6 and longest_run >= 4):
                if result is not None:
                    result.recognition_errors.append(f"{zone.zone_type.value}: invalid repeated-digit pattern")
                final_value = "-" if raw_digits is not None else ""
        return final_value, conf_list

    def _digit_model_expected_points(
        self,
        template: Template,
        zone: Zone,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if zone.zone_type != ZoneType.EXAM_CODE_BLOCK or not zone.grid:
            return np.empty((0, 2), dtype=np.float32), {}
        model = dict((getattr(template, "metadata", {}) or {}).get("digit_model", {}) or {})
        if not model:
            return np.empty((0, 2), dtype=np.float32), {}
        anchors = np.array(self._get_manual_digit_anchor_points(template, zone), dtype=np.float32)
        rows = max(1, int(zone.grid.rows))
        cols = max(1, int(zone.grid.cols))
        if len(anchors) < 2 or rows <= 0 or cols <= 0:
            return np.empty((0, 2), dtype=np.float32), {}

        ref_anchor = anchors[0].copy()
        ruler_anchors = np.array(anchors[1:] if len(anchors) > 2 else anchors, dtype=np.float32)
        required_anchor_count = rows + 1
        if len(ruler_anchors) > required_anchor_count:
            ruler_anchors = ruler_anchors[:required_anchor_count]
        elif len(ruler_anchors) < required_anchor_count and len(ruler_anchors) >= 2:
            interp = []
            for idx in range(required_anchor_count):
                alpha = 0.0 if required_anchor_count <= 1 else float(idx / max(1, required_anchor_count - 1))
                interp.append(((1.0 - alpha) * ruler_anchors[0]) + (alpha * ruler_anchors[-1]))
            ruler_anchors = np.array(interp, dtype=np.float32)
        if len(ruler_anchors) == 0:
            return np.empty((0, 2), dtype=np.float32), {}

        anchor_vec = ruler_anchors[-1] - ruler_anchors[0] if len(ruler_anchors) >= 2 else np.array([0.0, 1.0], dtype=np.float32)
        norm = float(np.linalg.norm(anchor_vec))
        if norm < 1e-6:
            return np.empty((0, 2), dtype=np.float32), {}
        row_unit = anchor_vec / norm
        ang = np.deg2rad(float(model.get("rotation_deg", 0.0) or 0.0))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
        row_unit = rot @ row_unit
        col_unit = np.array([-row_unit[1], row_unit[0]], dtype=np.float32)

        zone_width = float(zone.width * template.width if zone.width <= 1.0 else zone.width)
        zone_height = float(zone.height * template.height if zone.height <= 1.0 else zone.height)
        base_col_spacing = (zone_width / max(1, cols - 1)) if cols > 1 else max(1.0, zone_width)
        base_row_spacing = (zone_height / max(1, rows)) if rows > 0 else max(1.0, zone_height)
        col_spacing = base_col_spacing * float(model.get("exam_col_spacing_scale", 1.0) or 1.0)
        row_spacing_scale = float(model.get("exam_row_spacing_scale", 1.0) or 1.0)
        row_spacing = base_row_spacing * row_spacing_scale
        off = model.get("offset_exam", [0.0, 0.0]) or [0.0, 0.0]
        off_x = float(off[0] if len(off) > 0 else 0.0)
        off_y = float(off[1] if len(off) > 1 else 0.0)

        median_gap = row_spacing
        if len(ruler_anchors) >= 2:
            gaps = [float(np.dot(ruler_anchors[i + 1] - ruler_anchors[i], row_unit)) for i in range(len(ruler_anchors) - 1)]
            valid = [g for g in gaps if g > 1e-3]
            if valid:
                median_gap = float(np.median(valid))

        base_row_midpoints: list[np.ndarray] = []
        if len(ruler_anchors) >= 2:
            base_row_midpoints = [((ruler_anchors[i] + ruler_anchors[i + 1]) * 0.5) for i in range(len(ruler_anchors) - 1)]
            if len(base_row_midpoints) < rows:
                last_gap_vec = ruler_anchors[-1] - ruler_anchors[-2]
                base_row_midpoints.append(ruler_anchors[-1] + (0.5 * last_gap_vec))
        else:
            base_row_midpoints = [ruler_anchors[0] + (0.5 * median_gap * row_unit)]
        row_origin = base_row_midpoints[0] if base_row_midpoints else ref_anchor

        row_centers: list[np.ndarray] = []
        row_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for r in range(rows):
            if r < len(base_row_midpoints):
                center_base = base_row_midpoints[r].copy()
            elif base_row_midpoints:
                center_base = row_origin + (r * median_gap * row_unit)
            else:
                center_base = ref_anchor + ((r + 0.5) * median_gap * row_unit)
            delta = center_base - row_origin
            along = float(np.dot(delta, row_unit))
            perp = delta - (along * row_unit)
            scaled_base = row_origin + perp + ((along * row_spacing_scale) * row_unit)
            center = scaled_base + (off_x * col_unit) + (off_y * row_unit)
            row_centers.append(center)
            line_center = scaled_base + (off_y * row_unit)
            row_start = line_center.copy()
            row_end = row_start - (zone_width * col_unit)
            row_segments.append((tuple(float(v) for v in row_start.tolist()), tuple(float(v) for v in row_end.tolist())))

        points: list[np.ndarray] = []
        for r in range(rows):
            for c in range(cols):
                points.append(row_centers[r] + (c * col_spacing * col_unit))
        col_lines = []
        for c in range(cols):
            start = row_centers[0] + (c * col_spacing * col_unit)
            end = row_centers[-1] + (c * col_spacing * col_unit)
            col_lines.append((tuple(float(v) for v in start.tolist()), tuple(float(v) for v in end.tolist())))
        debug = {
            "digit_model_applied": True,
            "anchor_points": [(float(x), float(y)) for x, y in anchors],
            "anchor_line": [tuple(float(v) for v in ref_anchor.tolist()), tuple(float(v) for v in (ref_anchor + (row_unit * max(median_gap * rows, 1.0))).tolist())],
            "col_segments": col_lines,
            "row_segments": row_segments,
            "bubble_centers": [(float(pt[0]), float(pt[1])) for pt in points],
        }
        return np.array(points, dtype=np.float32), debug

    def _adjust_identifier_points_by_anchor_distance(
        self,
        binary: np.ndarray,
        template: Template,
        zone: Zone,
        points: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if points.size == 0:
            return points, {"identifier_anchor_distance_ratio_applied": False, "reason": "empty_points"}
        manual = np.array(self._get_manual_digit_anchor_points(template, zone), dtype=np.float32)
        detected = np.array(self._detect_digit_anchor_ruler(binary, template, zone), dtype=np.float32)
        if len(manual) < 2 or len(detected) < 2:
            return points, {
                "identifier_anchor_distance_ratio_applied": False,
                "manual_anchor_count": int(len(manual)),
                "detected_anchor_count": int(len(detected)),
            }

        manual_order = np.argsort(manual[:, 1])
        detected_order = np.argsort(detected[:, 1])
        manual_start, manual_end = manual[manual_order[0]], manual[manual_order[-1]]
        detected_start, detected_end = detected[detected_order[0]], detected[detected_order[-1]]

        manual_axis = (manual_end - manual_start).astype(np.float32)
        detected_axis = (detected_end - detected_start).astype(np.float32)
        manual_len = float(np.linalg.norm(manual_axis))
        detected_len = float(np.linalg.norm(detected_axis))
        if manual_len <= 1e-6 or detected_len <= 1e-6:
            return points, {"identifier_anchor_distance_ratio_applied": False, "reason": "degenerate_axis"}

        row_ratio = float(np.clip(detected_len / manual_len, 0.75, 1.35))
        row_unit_manual = manual_axis / manual_len
        row_unit_detected = detected_axis / detected_len
        col_unit_manual = np.array([-row_unit_manual[1], row_unit_manual[0]], dtype=np.float32)
        col_unit_detected = np.array([-row_unit_detected[1], row_unit_detected[0]], dtype=np.float32)

        zone_x = float(zone.x * template.width if zone.x <= 1.0 else zone.x)
        zone_y = float(zone.y * template.height if zone.y <= 1.0 else zone.y)
        zone_w = float(zone.width * template.width if zone.width <= 1.0 else zone.width)
        zone_h = float(zone.height * template.height if zone.height <= 1.0 else zone.height)
        zone_center = np.array([zone_x + (0.5 * zone_w), zone_y + (0.5 * zone_h)], dtype=np.float32)
        manual_dist = float(np.mean(np.linalg.norm(manual - zone_center, axis=1)))
        detected_dist = float(np.mean(np.linalg.norm(detected - zone_center, axis=1)))
        distance_ratio = float(np.clip((detected_dist / max(manual_dist, 1e-6)), 0.75, 1.35))

        origin_manual = np.mean(manual, axis=0).astype(np.float32)
        origin_detected = np.mean(detected, axis=0).astype(np.float32)
        adjusted: list[np.ndarray] = []
        for pt in points.astype(np.float32):
            delta = pt - origin_manual
            row_comp = float(np.dot(delta, row_unit_manual))
            col_comp = float(np.dot(delta, col_unit_manual))
            adj = origin_detected + ((row_comp * row_ratio) * row_unit_detected) + ((col_comp * distance_ratio) * col_unit_detected)
            adjusted.append(adj.astype(np.float32))

        debug = {
            "identifier_anchor_distance_ratio_applied": True,
            "identifier_anchor_row_ratio": row_ratio,
            "identifier_anchor_distance_ratio": distance_ratio,
            "identifier_anchor_zone": str(getattr(zone.zone_type, "value", "") or ""),
            "manual_anchor_distance_mean": manual_dist,
            "detected_anchor_distance_mean": detected_dist,
            "manual_anchor_count": int(len(manual)),
            "detected_anchor_count": int(len(detected)),
        }
        return np.array(adjusted, dtype=np.float32), debug

    def _should_retry_exam_code_with_sampling(
        self,
        recognition_errors: list[str],
        error_mark: int,
        confs: list[float] | None,
    ) -> bool:
        """Backward-compatible shim for older exam-code retry call sites.

        The extra retry path was removed to avoid slowing down mã đề
        recognition. Keep the method so older code paths do not crash.
        """
        _ = recognition_errors, error_mark, confs
        return False

    def _pick_best_mcq_option(
        self,
        row_scores: np.ndarray,
        row_threshold: float,
        row_raw_scores: np.ndarray | None = None,
        row_core_scores: np.ndarray | None = None,
        row_eroded_scores: np.ndarray | None = None,
    ) -> tuple[int | None, float, str | None]:
        scores = np.asarray(row_scores, dtype=np.float32)
        if scores.size == 0:
            return None, 0.0, "uncertain"
        order = np.argsort(scores)[::-1]
        top_i = int(order[0])
        second_i = int(order[1]) if len(order) > 1 else top_i
        top = float(scores[top_i])
        second = float(scores[second_i]) if len(order) > 1 else 0.0
        margin = top - second
        if top < max(self.empty_threshold + 0.08, row_threshold - 0.10):
            return None, 0.0, "uncertain"
        tie_tol = 0.015
        if len(order) > 1 and abs(top - second) <= tie_tol:
            return None, 0.0, "multiple_equal"
        if margin <= 0:
            return None, 0.0, "multiple_equal"
        _ = row_raw_scores, row_core_scores, row_eroded_scores
        return top_i, margin, "max_pick"
        return None, 0.0, "uncertain"

    def recognize_block(self, binary: np.ndarray, zone: Zone, template: Template, result: OMRResult, debug_overlay: np.ndarray | None = None) -> None:
        grid = zone.grid
        if not grid or not grid.bubble_positions:
            return

        working_binary = binary
        working_image = getattr(result, "aligned_image", None)
        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            identifier_started = time.perf_counter()
            expected = self._resolve_zone_centers(working_binary, zone, template).astype(np.float32)
            bubble_radius = float(zone.metadata.get("bubble_radius", 9))
            sid_ratio_debug: dict[str, object] = {}
            sid_ratio_guided: np.ndarray | None = None
            sid_base_guided: np.ndarray | None = None
            exam_digit_model_applied = False
            if zone.zone_type == ZoneType.EXAM_CODE_BLOCK:
                modeled_expected, model_debug = self._digit_model_expected_points(template, zone)
                if len(modeled_expected) == len(expected):
                    adjusted_expected, ratio_debug = self._adjust_identifier_points_by_anchor_distance(
                        working_binary,
                        template,
                        zone,
                        modeled_expected.astype(np.float32),
                    )
                    guided, digit_debug = adjusted_expected, (model_debug | ratio_debug)
                    exam_digit_model_applied = True
                else:
                    guided, digit_debug = self._digit_zone_guidance(working_binary, expected, zone, template)
            else:
                guided, digit_debug = self._digit_zone_guidance(working_binary, expected, zone, template)
            if zone.zone_type == ZoneType.STUDENT_ID_BLOCK:
                sid_base_guided = guided.astype(np.float32)
                guided = sid_base_guided
            column_guided = self._resolve_column_digit_centers(working_binary, guided.astype(np.float32), grid, bubble_radius)
            centers, grid_fit_debug = self._refit_digit_grid_from_clear_points(
                working_binary,
                column_guided.astype(np.float32),
                grid,
                bubble_radius,
            )
            key = "student_id" if zone.zone_type == ZoneType.STUDENT_ID_BLOCK else "exam_code"
            error_mark = len(result.recognition_errors)
            quality_gate = dict(getattr(result, "_quality_gate", {}) or {})
            poor_image = bool(quality_gate.get("poor_scan", False))
            poor_identifier_zone = bool(quality_gate.get("poor_identifier_zone", False))
            poor_fast_fail = bool(getattr(result, "_poor_image_fast_fail", self._poor_image_fast_fail_enabled(template)))
            allow_heavy_on_poor = self._allow_heavy_identifier_retry_on_poor_image(template)
            identifier_deadline = float(getattr(result, "_identifier_deadline_monotonic", 0.0) or 0.0)

            def identifier_budget_ok(min_left: float = 0.0) -> bool:
                if self._time_budget_exceeded():
                    return False
                if identifier_deadline > 0.0 and (time.monotonic() + min_left) > identifier_deadline:
                    return False
                return True

            direct_value, direct_confs, direct_centers, direct_mat, direct_debug = self._decode_identifier_zone_from_centers(
                working_binary,
                zone,
                template,
                result,
                centers,
                int(round(bubble_radius)),
            )
            final_value, final_confs = self._finalize_identifier_value(
                zone,
                key,
                value=direct_value,
                confs=direct_confs,
                result=result,
            )
            # identifier risk gate: cap fallback depth when direct decode looks ambiguous.
            col_ambiguity = 0
            weak_cols = 0
            if isinstance(direct_mat, np.ndarray) and direct_mat.size > 0 and direct_mat.ndim == 2:
                for c in range(direct_mat.shape[1]):
                    col = np.asarray(direct_mat[:, c], dtype=np.float32)
                    if col.size == 0:
                        continue
                    top2 = np.sort(col)[-2:] if col.size >= 2 else np.array([0.0, col.max()], dtype=np.float32)
                    margin = float(top2[-1] - top2[-2]) if top2.size >= 2 else float(top2[-1])
                    if margin < 0.12:
                        col_ambiguity += 1
                    if float(top2[-1]) < 0.34:
                        weak_cols += 1
            direct_conf_mean = float(np.mean(final_confs or [0.0]))
            id_risk_score = float(min(1.0, (0.55 - direct_conf_mean) + (0.10 * col_ambiguity) + (0.06 * weak_cols)))
            poor_identifier_risk = bool((direct_conf_mean < 0.30) or (col_ambiguity >= 1) or id_risk_score >= 0.45)
            fallback_budget_ok = identifier_budget_ok(0.12)
            poor_blocks_heavy = poor_fast_fail and poor_image and (not allow_heavy_on_poor)
            fast_production_mode = bool(getattr(result, "_fast_production_mode", False))
            max_fallback_steps = self._max_identifier_fallback_steps(template, fast_production_mode)
            fallback_steps_used = 0
            allow_fallbacks = fallback_budget_ok and (not poor_blocks_heavy) and not (bool(final_value) and direct_conf_mean >= 0.55)
            if poor_blocks_heavy:
                self._append_issue_once(result, "FAST_FAIL_POOR_SCAN", "Skipped heavy identifier fallback due to poor image", zone_id=zone.id)
                result.recognition_errors.append(f"{zone.zone_type.value}: skipped heavy fallback due to poor image")
            axis_final = ""
            axis_debug: dict[str, object] = {}
            axis_needed = (
                allow_fallbacks
                and (not poor_image)
                and (not poor_identifier_zone)
                and (fallback_steps_used < max_fallback_steps)
                and identifier_budget_ok(0.10)
                and ((not final_value) or (direct_conf_mean < 0.32))
            )
            if axis_needed:
                axis_value, axis_confs, axis_debug = self._decode_identifier_by_anchor_axis(
                    working_binary,
                    zone,
                    template,
                    direct_centers,
                    int(round(bubble_radius)),
                )
                axis_final, axis_final_confs = self._finalize_identifier_value(
                    zone,
                    key,
                    value=axis_value,
                    confs=axis_confs,
                    result=None,
                )
                fallback_steps_used += 1
                if axis_final and (not final_value or float(np.mean(axis_final_confs or [0.0])) >= float(np.mean(final_confs or [0.0]))):
                    del result.recognition_errors[error_mark:]
                    final_value, final_confs = axis_final, axis_final_confs
            allow_ratio_retry = bool((template.metadata or {}).get("student_id_allow_ratio_retry", not fast_production_mode))
            if (
                allow_fallbacks
                and
                key == "student_id"
                and allow_ratio_retry
                and (not poor_image)
                and (not poor_identifier_risk if fast_production_mode else True)
                and (fallback_steps_used < max_fallback_steps)
                and not final_value
                and sid_base_guided is not None
                and identifier_budget_ok(0.16)
                and direct_conf_mean < 0.35
            ):
                sid_ratio_guided, sid_ratio_debug = self._adjust_identifier_points_by_anchor_distance(
                    working_binary,
                    template,
                    zone,
                    sid_base_guided,
                )
                if not (
                    sid_ratio_guided is not None
                    and bool(sid_ratio_debug.get("identifier_anchor_distance_ratio_applied"))
                    and int(sid_ratio_debug.get("detected_anchor_count", 0)) >= 3
                ):
                    sid_ratio_debug["student_ratio_retry_attempted"] = False
                else:
                    fallback_steps_used += 1
                    retry_column_guided = self._resolve_column_digit_centers(working_binary, sid_ratio_guided.astype(np.float32), grid, bubble_radius)
                    retry_centers, retry_grid_fit_debug = self._refit_digit_grid_from_clear_points(
                        working_binary,
                        retry_column_guided.astype(np.float32),
                        grid,
                        bubble_radius,
                    )
                    retry_value, retry_confs, retry_direct_centers, retry_mat, retry_direct_debug = self._decode_identifier_zone_from_centers(
                        working_binary,
                        zone,
                        template,
                        result,
                        retry_centers,
                        int(round(bubble_radius)),
                    )
                    retry_final, retry_final_confs = self._finalize_identifier_value(
                        zone,
                        key,
                        value=retry_value,
                        confs=retry_confs,
                        result=None,
                    )
                    sid_ratio_debug["student_ratio_retry_attempted"] = True
                    sid_ratio_debug["student_ratio_retry_success"] = bool(retry_final)
                    if retry_final:
                        del result.recognition_errors[error_mark:]
                        final_value, final_confs = retry_final, retry_final_confs
                        fallback_steps_used += 1
                        direct_centers = retry_direct_centers
                        direct_mat = retry_mat
                        direct_debug = dict(direct_debug) | dict(retry_direct_debug) | dict(retry_grid_fit_debug) | {
                            "student_ratio_applied": True,
                            "recognition_path": "student_ratio_retry",
                        }
            if sid_ratio_debug:
                digit_debug = dict(digit_debug) | sid_ratio_debug
            used_sampling = False
            zone_debug = dict(getattr(result, "digit_zone_debug", {}) or {})
            final_col_lines = [float(np.median(direct_centers[c::grid.cols, 0])) for c in range(grid.cols)] if grid.cols > 0 else []
            zone_debug[zone.id] = digit_debug | grid_fit_debug | direct_debug | axis_debug | {
                "col_lines": final_col_lines,
                "col_segments": [],
                "recognition_path": "axis_projection" if bool(axis_final) else "direct",
                "poor_image": bool(poor_image),
                "poor_identifier_zone": bool(poor_identifier_zone),
                "identifier_risk_score": float(id_risk_score),
                "poor_identifier_risk": bool(poor_identifier_risk),
                "ambiguous_columns": int(col_ambiguity),
                "weak_columns": int(weak_cols),
            }
            allow_exam_sampling = bool((template.metadata or {}).get("exam_code_allow_sampling_fallback", not fast_production_mode))
            allow_student_sampling = bool((template.metadata or {}).get("student_id_allow_sampling_fallback", not fast_production_mode))
            allow_sampling_for_key = allow_exam_sampling if key == "exam_code" else allow_student_sampling
            skip_sampling_fallback = bool(
                (key == "exam_code" and exam_digit_model_applied)
                or (self._is_fast_scan_mode(template) and self._should_skip_heavy_identifier_fallback(template))
                or (poor_fast_fail and poor_image and (not allow_heavy_on_poor))
                or (not allow_sampling_for_key)
                or (fallback_steps_used >= max_fallback_steps)
                or (fast_production_mode and poor_identifier_risk)
            )
            if allow_fallbacks and not final_value and not skip_sampling_fallback and identifier_budget_ok(0.18) and not poor_identifier_zone:
                fallback_steps_used += 1
                rows, cols = 10, max(1, grid.cols)
                source_img = self._preprocess_digit_sampling(working_image if working_image is not None else working_binary)
                bbox = self._digit_zone_bbox(zone, source_img.shape[:2])
                digits, mat, sampling_debug = self._read_digit_zone_with_offset_fallback(
                    source_img,
                    bbox,
                    cols,
                    rows,
                    threshold=0.25,
                    centers=direct_centers,
                )
                sampled_error_mark = len(result.recognition_errors)
                sampled_value, sampled_confs = self._finalize_identifier_value(
                    zone,
                    key,
                    raw_digits=digits,
                    result=result,
                )
                chosen_value = final_value
                chosen_confs = final_confs
                chosen_path = "direct"
                if key == "exam_code":
                    chosen_value, chosen_confs, chosen_path = self._pick_best_exam_code_result(
                        final_value,
                        final_confs,
                        sampled_value,
                        sampled_confs,
                    )
                elif sampled_value not in ("", "-"):
                    chosen_value, chosen_confs, chosen_path = sampled_value, sampled_confs, "sampling_fallback"
                zone_debug[zone.id] = dict(zone_debug.get(zone.id, {})) | sampling_debug | {
                    "sampling_scores": mat.tolist(),
                    "recognition_path": chosen_path if chosen_path == "sampling_fallback" else zone_debug.get(zone.id, {}).get("recognition_path", "direct"),
                }
                if chosen_path == "sampling_fallback":
                    del result.recognition_errors[error_mark:]
                    del result.recognition_errors[sampled_error_mark:]
                    final_value, final_confs = chosen_value, chosen_confs
                    used_sampling = True
                    fallback_steps_used += 1
                elif final_value == "":
                    final_value = "-"
            elif not final_value and skip_sampling_fallback:
                final_value = "-"
            elif not final_value and not identifier_budget_ok():
                self._append_issue_once(result, "IDENTIFIER_TIMEOUT", "Identifier timeout budget exceeded", zone_id=zone.id)
                result.recognition_errors.append(f"{zone.zone_type.value}: identifier timeout budget exceeded")
                final_value = "-"
            if allow_fallbacks and fallback_steps_used >= max_fallback_steps and (not final_value):
                # identifier hard cap: never exceed capped fallback depth in fast mode.
                self._append_issue_once(result, "IDENTIFIER_FAST_CAP", "Identifier fallback capped for fast mode", zone_id=zone.id)
                if key == "student_id":
                    self._append_issue_once(result, "STUDENT_ID_FAST_FAIL", "Student ID stopped by fast fallback cap", zone_id=zone.id)
                else:
                    self._append_issue_once(result, "EXAM_CODE_FAST_FAIL", "Exam code stopped by fast fallback cap", zone_id=zone.id)
            if key == "student_id" and poor_image and (not final_value or direct_conf_mean < 0.30):
                final_value = "-"
            setattr(result, "digit_zone_debug", zone_debug)
            if self.debug_mode and working_image is not None:
                digit_overlay = working_image.copy()
                for x, y in digit_debug.get("guide_points", []):
                    cv2.circle(digit_overlay, (int(round(x)), int(round(y))), 6, (0, 255, 255), 2)
                fitted_line = digit_debug.get("fitted_line", [])
                if len(fitted_line) == 2:
                    pt0 = tuple(int(round(v)) for v in fitted_line[0])
                    pt1 = tuple(int(round(v)) for v in fitted_line[1])
                    cv2.line(digit_overlay, pt0, pt1, (255, 255, 0), 2)
                for x, y in direct_centers:
                    cv2.circle(digit_overlay, (int(round(x)), int(round(y))), 5, (0, 0, 255), 1)
                if used_sampling:
                    for x, y in zone_debug.get(zone.id, {}).get("sample_points", []):
                        cv2.circle(digit_overlay, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
                out_dir = self.debug_dir or Path(result.image_path).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{Path(result.image_path).stem}_{zone.id}_digit_debug.png"
                cv2.imwrite(str(out_path), digit_overlay)
            if key == "student_id":
                result.student_id = final_value
            else:
                result.exam_code = final_value
            confidence_value = float(np.mean(final_confs)) if final_confs else 0.0
            result.confidence_scores[f"{key}:{zone.id}"] = confidence_value
            result.confidence_scores[key] = max(confidence_value, float(result.confidence_scores.get(key, 0.0) or 0.0))
            if self.debug_mode:
                print(
                    f"[OMR][IDENTIFIER] key={key} direct={direct_value!r} final={final_value!r} "
                    f"conf={confidence_value:.3f} path={zone_debug.get(zone.id, {}).get('recognition_path', '-')}"
                )
            consumed = float(time.perf_counter() - identifier_started)
            setattr(result, "_identifier_time_sec", float(getattr(result, "_identifier_time_sec", 0.0) or 0.0) + consumed)
            if axis_needed or bool(sid_ratio_debug.get("student_ratio_retry_attempted")) or used_sampling:
                setattr(result, "_identifier_fallback_time_sec", float(getattr(result, "_identifier_fallback_time_sec", 0.0) or 0.0) + consumed)
            return
        else:
            centers = self._resolve_zone_centers(working_binary, zone, template)
        radius = int(zone.metadata.get("bubble_radius", 9))
        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            centers = self._detect_digit_bubble_centers(working_binary, centers, float(radius))
        ratios = self.detect_bubbles(working_binary, centers, radius)
        if zone.zone_type == ZoneType.MCQ_BLOCK:
            raw_mcq_ratios = np.asarray(ratios, dtype=np.float32).copy()
            core_ratios = self._detect_center_core_marks(working_binary, centers, radius)
            eroded_ratios = self._detect_eroded_mark_density(working_binary, centers, radius)
            core_boosted = np.clip((0.62 * ratios) + (0.38 * core_ratios), 0.0, 1.0)
            xmark_boosted = np.clip((0.50 * ratios) + (0.30 * core_ratios) + (0.20 * eroded_ratios), 0.0, 1.0)
            ratios = np.maximum(ratios, np.maximum(core_boosted, xmark_boosted))
        elif zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            core_ratios = self._detect_center_core_marks(working_binary, centers, radius)
            advanced_mode = bool(((template.metadata or {}).get("digit_advanced_mode", False) if getattr(template, "metadata", None) else False))
            multi_probe_ratios = self._detect_digit_zone_multi_probe_marks(working_binary, centers, radius) if advanced_mode else np.zeros_like(ratios)
            if zone.zone_type == ZoneType.STUDENT_ID_BLOCK:
                square_ratios = self._detect_square_mark_density(working_binary, centers, radius)
                eroded_ratios = self._detect_eroded_mark_density(working_binary, centers, radius)
                legacy_ratios = np.clip((0.20 * ratios) + (0.34 * core_ratios) + (0.26 * square_ratios) + (0.20 * eroded_ratios), 0.0, 1.0)
                widened_ratios = np.clip((0.10 * ratios) + (0.22 * core_ratios) + (0.22 * square_ratios) + (0.18 * eroded_ratios) + (0.28 * multi_probe_ratios), 0.0, 1.0) if advanced_mode else legacy_ratios
                ratios = np.maximum(legacy_ratios, widened_ratios)
            else:
                legacy_ratios = np.clip((0.60 * ratios) + (0.40 * core_ratios), 0.0, 1.0)
                widened_ratios = np.clip((0.30 * ratios) + (0.28 * core_ratios) + (0.42 * multi_probe_ratios), 0.0, 1.0) if advanced_mode else legacy_ratios
                ratios = np.maximum(legacy_ratios, widened_ratios)
        dynamic_thresholds = np.array([self._estimate_local_fill_threshold(working_binary, center, radius, self.fill_threshold) for center in centers], dtype=np.float32)

        rows, cols = max(1, grid.rows), max(1, grid.cols)
        need = rows * cols
        if len(ratios) < need:
            ratios = np.pad(ratios, (0, need - len(ratios)), constant_values=0.0)
            dynamic_thresholds = np.pad(dynamic_thresholds, (0, need - len(dynamic_thresholds)), constant_values=self.fill_threshold)
        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            mat = self._cluster_digit_columns(centers, ratios, rows, cols, 0.0)
            local_fill = self._cluster_digit_columns(centers, dynamic_thresholds, rows, cols, self.fill_threshold)
            mat = np.clip(mat - (0.5 * np.clip(local_fill - self.empty_threshold, 0.0, 1.0)), 0.0, 1.0)
        else:
            mat = ratios[:need].reshape(rows, cols)
            local_fill = dynamic_thresholds[:need].reshape(rows, cols)
            raw_mcq_mat = np.asarray(raw_mcq_ratios[:need], dtype=np.float32).reshape(rows, cols) if zone.zone_type == ZoneType.MCQ_BLOCK else None
            core_mcq_mat = np.asarray(core_ratios[:need], dtype=np.float32).reshape(rows, cols) if zone.zone_type == ZoneType.MCQ_BLOCK else None
            eroded_mcq_mat = np.asarray(eroded_ratios[:need], dtype=np.float32).reshape(rows, cols) if zone.zone_type == ZoneType.MCQ_BLOCK else None

        orig_fill = self.fill_threshold
        self.fill_threshold = float(np.clip(np.median(local_fill), orig_fill - 0.08, orig_fill + 0.08))
        try:
            if debug_overlay is not None:
                self._draw_debug_overlay(debug_overlay, centers, ratios, radius, zone)

            if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
                value, confs = self._decode_column_digits(mat, zone, grid, result)
                key = "student_id" if zone.zone_type == ZoneType.STUDENT_ID_BLOCK else "exam_code"
                expected_len = max(1, cols)
                missing_digits = value.count("?")
                is_valid = len(value) == expected_len and missing_digits == 0
                global_score = float(expected_len - missing_digits) / float(expected_len or 1)
                if not is_valid or global_score < 0.85:
                    result.recognition_errors.append(f"{zone.zone_type.value}: LOW_CONFIDENCE")
                    if not is_valid:
                        result.recognition_errors.append(f"{zone.zone_type.value}: invalid length or ambiguous digit sequence")
                    value = ""
                if key == "student_id" and value:
                    longest_run = 1
                    current_run = 1
                    for idx in range(1, len(value)):
                        if value[idx] == value[idx - 1]:
                            current_run += 1
                            longest_run = max(longest_run, current_run)
                        else:
                            current_run = 1
                    if len(set(value)) == 1 or (len(value) >= 6 and longest_run >= 4):
                        result.recognition_errors.append(f"{zone.zone_type.value}: invalid repeated-digit pattern")
                        value = ""
                elif global_score < 0.85:
                    result.recognition_errors.append(f"{zone.zone_type.value}: LOW_CONFIDENCE")
                if key == "student_id":
                    result.student_id = value
                else:
                    result.exam_code = value
                result.confidence_scores[f"{key}:{zone.id}"] = float(np.mean(confs)) if confs else 0.0
                return

            if zone.zone_type == ZoneType.MCQ_BLOCK:
                labels = grid.options if grid.options else [chr(65 + i) for i in range(cols)]
                confs: list[float] = []
                q_count = min(grid.question_count or rows, rows)
                for r in range(q_count):
                    qno = grid.question_start + r
                    row_scores = mat[r, :]
                    row_threshold = float(np.median(local_fill[r, :])) if local_fill.shape[1] else self.fill_threshold
                    best_idx, confidence, reason = self._pick_best_mcq_option(
                        row_scores,
                        row_threshold,
                        row_raw_scores=raw_mcq_mat[r, :] if raw_mcq_mat is not None else None,
                        row_core_scores=core_mcq_mat[r, :] if core_mcq_mat is not None else None,
                        row_eroded_scores=eroded_mcq_mat[r, :] if eroded_mcq_mat is not None else None,
                    )
                    if best_idx is None:
                        if reason == "multiple":
                            result.recognition_errors.append(f"MCQ Q{qno}: multiple answer")
                        else:
                            result.recognition_errors.append(f"MCQ Q{qno}: uncertain")
                        continue
                    result.mcq_answers[qno] = labels[best_idx]
                    confs.append(confidence)
                result.confidence_scores[f"mcq:{zone.id}"] = float(np.mean(confs)) if confs else 0.0
                return

            if zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
                self._decode_true_false(mat, zone, grid, result)
                return

            if zone.zone_type == ZoneType.NUMERIC_BLOCK:
                self._decode_numeric(mat, zone, grid, result)
                return
        finally:
            self.fill_threshold = orig_fill

    def _decode_column_digits(self, mat: np.ndarray, zone: Zone, grid, result: OMRResult) -> tuple[str, list[float]]:
        rows, cols = mat.shape
        default_digit_map = list(range(10)) if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK) else list(range(rows))
        digit_map = zone.metadata.get("digit_map", default_digit_map)
        digits: list[str] = []
        confs: list[float] = []
        soft_fallbacks: list[tuple[int, int, float]] = []
        for c in range(cols):
            col_scores = np.asarray(mat[:, c], dtype=np.float32)
            raw_max = float(np.max(col_scores)) if col_scores.size else 0.0
            fill_cap = self.fill_threshold
            if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
                fill_cap = min(fill_cap, 0.52)
            col_threshold = max(
                self.empty_threshold + 0.10,
                min(
                    fill_cap,
                    float(np.mean(col_scores) + (0.5 * np.std(col_scores))),
                ),
            )
            activated_scores = np.clip(col_scores - col_threshold, 0.0, None)
            activated_max = float(np.max(activated_scores)) if activated_scores.size else 0.0
            if activated_max > 1e-6:
                norm_scores = activated_scores / activated_max
            else:
                norm_scores = np.zeros_like(activated_scores)
            top3 = np.sort(col_scores)[-min(3, len(col_scores)) :] if len(col_scores) else np.array([], dtype=np.float32)
            top3_mean = float(np.mean(top3)) if top3.size else 0.0
            order = np.argsort(norm_scores)[::-1]
            top_i, second_i = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
            mapped = digit_map[top_i] if top_i < len(digit_map) else top_i
            raw_second = float(col_scores[second_i]) if len(order) > 1 else 0.0
            confidence = (raw_max / top3_mean) if top3_mean > 1e-6 else 0.0
            fallback_ok = raw_max > (col_threshold * 1.2)
            if raw_max < col_threshold:
                if fallback_ok:
                    digits.append(str(mapped))
                    result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: fallback accepted")
                else:
                    digits.append("?")
                    result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: below threshold")
            elif raw_max <= (1.4 * raw_second):
                if fallback_ok:
                    digits.append(str(mapped))
                    result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: fallback accepted")
                else:
                    digits.append("?")
                    result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: ambiguous")
            elif confidence < 1.2:
                if fallback_ok:
                    digits.append(str(mapped))
                    result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: fallback accepted")
                else:
                    digits.append("?")
                    result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: low confidence")
            else:
                digits.append(str(mapped))
            soft_fallbacks.append((c, int(mapped), raw_max))
            confs.append(confidence)
        return "".join(digits), confs

    def _cluster_digit_columns(
        self,
        centers: np.ndarray,
        values: np.ndarray,
        rows: int,
        cols: int,
        fill_value: float,
    ) -> np.ndarray:
        count = rows * cols
        if count <= 0:
            return np.zeros((max(1, rows), max(1, cols)), dtype=np.float32)
        actual_count = min(count, len(centers), len(values))
        pts = np.asarray(centers[:actual_count], dtype=np.float32)
        vals = np.asarray(values[:actual_count], dtype=np.float32)
        if pts.size == 0:
            return np.full((rows, cols), float(fill_value), dtype=np.float32)

        xs = pts[:, 0].astype(np.float32)
        order_x = np.argsort(xs)
        sorted_xs = xs[order_x]
        if cols <= 1:
            labels = np.zeros(len(xs), dtype=np.int32)
            split_xs = np.array([float(np.mean(sorted_xs)) if sorted_xs.size else 0.0], dtype=np.float32)
        else:
            gaps = np.diff(sorted_xs) if sorted_xs.size > 1 else np.array([], dtype=np.float32)
            cut_positions = np.argsort(gaps)[-(cols - 1) :] + 1 if gaps.size else np.array([], dtype=np.int32)
            cut_positions = np.sort(cut_positions[: cols - 1])
            segment_bounds = [0] + cut_positions.tolist() + [len(sorted_xs)]
            split_xs = []
            sorted_labels = np.zeros(len(sorted_xs), dtype=np.int32)
            for seg_idx in range(min(cols, len(segment_bounds) - 1)):
                start = int(segment_bounds[seg_idx])
                end = int(segment_bounds[seg_idx + 1])
                segment = sorted_xs[start:end]
                if segment.size == 0:
                    continue
                split_xs.append(float(np.mean(segment)))
                sorted_labels[start:end] = seg_idx
            if len(split_xs) < cols:
                fill = split_xs[-1] if split_xs else (float(np.mean(sorted_xs)) if sorted_xs.size else 0.0)
                split_xs.extend([fill] * (cols - len(split_xs)))
            split_xs = np.array(split_xs[:cols], dtype=np.float32)
            labels = np.zeros(len(xs), dtype=np.int32)
            labels[order_x] = sorted_labels[: len(order_x)]
            non_empty = [idx for idx in range(cols) if np.any(labels == idx)]
            if non_empty:
                if len(non_empty) >= 2:
                    spacing = float(np.median(np.diff(split_xs[non_empty])))
                else:
                    spacing = 1.0
                for idx in range(cols):
                    if np.any(labels == idx):
                        continue
                    left = max([j for j in non_empty if j < idx], default=None)
                    right = min([j for j in non_empty if j > idx], default=None)
                    if left is not None and right is not None:
                        split_xs[idx] = float((split_xs[left] + split_xs[right]) * 0.5)
                    elif left is not None:
                        split_xs[idx] = float(split_xs[left] + spacing * (idx - left))
                    elif right is not None:
                        split_xs[idx] = float(split_xs[right] - spacing * (right - idx))
                dists = np.abs(xs[:, None] - split_xs[None, :])
                labels = np.argmin(dists, axis=1).astype(np.int32)

        ys = pts[:, 1].astype(np.float32)
        order_y_all = np.argsort(ys)
        sorted_ys = ys[order_y_all]
        row_bounds = np.array_split(sorted_ys, rows)
        row_centers = np.array(
            [float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in row_bounds],
            dtype=np.float32,
        )
        row_centers = np.sort(row_centers)
        row_spacing = float(np.median(np.diff(row_centers))) if len(row_centers) > 1 else 1.0
        col_spacing = float(np.median(np.diff(split_xs))) if len(split_xs) > 1 else 1.0
        max_dx = max(4.0, col_spacing * 0.45)
        max_dy = max(4.0, row_spacing * 0.45)

        mat = np.full((rows, cols), float(fill_value), dtype=np.float32)
        for idx, pt in enumerate(pts):
            col_idx = int(np.argmin(np.abs(split_xs - pt[0])))
            row_idx = int(np.argmin(np.abs(row_centers - pt[1])))
            if abs(float(pt[0] - split_xs[col_idx])) > max_dx:
                continue
            if abs(float(pt[1] - row_centers[row_idx])) > max_dy:
                continue
            mat[row_idx, col_idx] = max(mat[row_idx, col_idx], float(vals[idx]))
        return mat

    def _decode_true_false(self, mat: np.ndarray, zone: Zone, grid, result: OMRResult) -> None:
        qpb = int(zone.metadata.get("questions_per_block", 2))
        spq = int(zone.metadata.get("statements_per_question", 4))
        cps = int(zone.metadata.get("choices_per_statement", 2))
        labels = [chr(ord("a") + i) for i in range(spq)]
        confs: list[float] = []

        for q in range(qpb):
            qno = grid.question_start + q
            result.true_false_answers[qno] = {}
            for sidx in range(spq):
                row_idx = q * spq + sidx
                if row_idx >= mat.shape[0]:
                    break
                row = mat[row_idx, : max(1, cps)]
                order = np.argsort(row)[::-1]
                top_i, second_i = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
                top, second = float(row[top_i]), float(row[second_i]) if len(order) > 1 else 0.0
                if len(np.where(row > self.fill_threshold)[0]) > 1:
                    result.recognition_errors.append(f"TF Q{qno}{labels[sidx]}: multiple answer")
                    continue
                if self.classify_bubble(top) != "filled" or (top - second) <= self.certainty_margin:
                    result.recognition_errors.append(f"TF Q{qno}{labels[sidx]}: uncertain")
                    continue
                result.true_false_answers[qno][labels[sidx]] = bool(top_i == 0)
                confs.append(top - second)

        result.confidence_scores[f"tf:{zone.id}"] = float(np.mean(confs)) if confs else 0.0

    def _decode_numeric(self, mat: np.ndarray, zone: Zone, grid, result: OMRResult) -> None:
        rows, cols = mat.shape
        digits_per_answer = int(zone.metadata.get("digits_per_answer", 3))
        question_count = int(zone.metadata.get("questions_per_block", zone.metadata.get("total_questions", 1)))
        digit_map = zone.metadata.get("digit_map", list(range(max(0, rows - 2))))
        sign_row = max(0, int(zone.metadata.get("sign_row", 1)) - 1)
        decimal_row = max(0, int(zone.metadata.get("decimal_row", 2)) - 1)
        digit_start_row = max(0, int(zone.metadata.get("digit_start_row", 3)) - 1)
        sign_columns = self._parse_index_list(zone.metadata.get("sign_columns", [1]), digits_per_answer)
        decimal_columns = self._parse_index_list(zone.metadata.get("decimal_columns", [2, 3]), digits_per_answer)
        sign_symbol = str(zone.metadata.get("sign_symbol", "-"))
        decimal_symbol = str(zone.metadata.get("decimal_symbol", "."))
        confs: list[float] = []

        for q in range(question_count):
            chars: list[str] = []
            question_base = q * digits_per_answer

            sign_mark = self._pick_row_mark(mat, sign_row, question_base, sign_columns)
            decimal_mark = self._pick_row_mark(mat, decimal_row, question_base, decimal_columns)

            for d in range(digits_per_answer):
                c = question_base + d
                if c >= cols:
                    break
                col = mat[:, c]
                digit_slice = col[digit_start_row:]
                if digit_slice.size == 0:
                    chars.append("?")
                    result.recognition_errors.append(f"NUM Q{grid.question_start+q} digit {d+1}: invalid digit rows")
                    continue
                order = np.argsort(digit_slice)[::-1]
                top_local = int(order[0])
                second_local = int(order[1]) if len(order) > 1 else int(order[0])
                top = float(digit_slice[top_local])
                second = float(digit_slice[second_local]) if len(order) > 1 else 0.0
                filled = np.where(digit_slice > self.fill_threshold)[0]
                if len(filled) > 1:
                    result.recognition_errors.append(f"NUM Q{grid.question_start+q} digit {d+1}: multiple answer")
                if self.classify_bubble(top) != "filled" or (top - second) <= self.certainty_margin:
                    chars.append("?")
                    result.recognition_errors.append(f"NUM Q{grid.question_start+q} digit {d+1}: uncertain")
                else:
                    mapped = digit_map[top_local] if top_local < len(digit_map) else top_local
                    chars.append(str(mapped))
                confs.append(top - second)

            value = "".join(chars)
            if decimal_mark is not None and decimal_mark > 0 and decimal_mark <= len(value):
                value = value[:decimal_mark] + decimal_symbol + value[decimal_mark:]
                value = value.replace("?", "")
            else:
                value = value.rstrip("?")
            if sign_mark is not None:
                value = value.lstrip("?")
                value = sign_symbol + value
            result.numeric_answers[grid.question_start + q] = value

        result.confidence_scores[f"num:{zone.id}"] = float(np.mean(confs)) if confs else 0.0

    def _parse_index_list(self, raw: object, width: int) -> list[int]:
        if raw is None:
            return []
        items: list[int] = []
        if isinstance(raw, str):
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            for tok in tokens:
                if tok.lstrip("-").isdigit():
                    items.append(int(tok) - 1)
        elif isinstance(raw, (list, tuple)):
            for val in raw:
                try:
                    items.append(int(val) - 1)
                except Exception:
                    continue
        return sorted({i for i in items if 0 <= i < max(1, width)})

    def _pick_row_mark(self, mat: np.ndarray, row: int, q_base: int, allowed_cols: list[int]) -> int | None:
        if row < 0 or row >= mat.shape[0] or not allowed_cols:
            return None
        picks: list[tuple[float, int]] = []
        for rel_col in allowed_cols:
            col = q_base + rel_col
            if col >= mat.shape[1]:
                continue
            picks.append((float(mat[row, col]), rel_col))
        if not picks:
            return None
        picks.sort(key=lambda x: x[0], reverse=True)
        best_ratio, best_col = picks[0]
        if self.classify_bubble(best_ratio) != "filled":
            return None
        if len(picks) > 1 and picks[1][0] > self.fill_threshold:
            return None
        return best_col

    def _get_x_mask(self, radius: int) -> np.ndarray:
        r = max(3, int(radius))
        cached = self._mask_cache.get(r)
        if cached is not None:
            return cached[0]
        size = (2 * r) + 1
        y, x = np.indices((size, size))
        d1 = np.abs(y - x)
        d2 = np.abs(y - (size - 1 - x))
        thickness = max(1, int(round(r * 0.20)))
        mask = (d1 <= thickness) | (d2 <= thickness)
        self._mask_cache[r] = (mask, float(np.count_nonzero(mask)))
        return mask

    def _draw_alignment_debug(self, image: np.ndarray, template: Template) -> None:
        corners = self._last_alignment_debug.get("page_corners") or []
        if len(corners) == 4:
            for idx, pt in enumerate(np.array(corners, dtype=np.int32)):
                cv2.circle(image, tuple(pt), 8, (0, 165, 255), -1)
                cv2.putText(image, f"C{idx+1}", (int(pt[0]) + 8, int(pt[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        plan = self._get_template_runtime_plan(template)
        for zone in template.zones:
            x, y, ww, hh = plan.zone_bounds_px.get(zone.id, (0, 0, 0, 0))
            cv2.rectangle(image, (x, y), (x + ww, y + hh), (255, 128, 0), 1)

    def _draw_debug_overlay(self, image: np.ndarray, centers: np.ndarray, ratios: np.ndarray, radius: int, zone: Zone | None = None) -> None:
        if zone is not None:
            x = int(zone.x * image.shape[1])
            y = int(zone.y * image.shape[0])
            w = int(zone.width * image.shape[1])
            h = int(zone.height * image.shape[0])
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 128, 0), 1)
        for (x, y), ratio in zip(centers.astype(np.int32), ratios):
            state = self.classify_bubble(float(ratio))
            color = (0, 255, 0) if state == "filled" else (0, 0, 255) if state == "empty" else (255, 255, 0)
            cv2.circle(image, (int(x), int(y)), radius, (255, 0, 0), 1)
            cv2.circle(image, (int(x), int(y)), max(2, radius // 4), color, -1)
            cv2.putText(image, f"{ratio:.2f}", (int(x + radius + 2), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) < 4:
            out = np.zeros((4, 2), dtype=np.float32)
            out[: len(pts)] = pts[: len(pts)]
            return out
        rect = np.zeros((4, 2), dtype=np.float32)
        sums = pts.sum(axis=1)
        rect[0] = pts[np.argmin(sums)]
        rect[2] = pts[np.argmax(sums)]
        diffs = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diffs)]
        rect[3] = pts[np.argmax(diffs)]
        return rect
