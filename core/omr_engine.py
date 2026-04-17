from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

try:
    from models.template import Template, Zone, ZoneType
except Exception:  # pragma: no cover
    from dataclasses import dataclass as _dc
    from enum import Enum

    class ZoneType(str, Enum):
        ANCHOR = "ANCHOR"
        STUDENT_ID_BLOCK = "STUDENT_ID_BLOCK"
        EXAM_CODE_BLOCK = "EXAM_CODE_BLOCK"
        MCQ_BLOCK = "MCQ_BLOCK"
        TRUE_FALSE_BLOCK = "TRUE_FALSE_BLOCK"
        NUMERIC_BLOCK = "NUMERIC_BLOCK"

    @_dc
    class _Grid:
        rows: int = 0
        cols: int = 0
        question_start: int = 1
        question_count: int = 0
        options: list[str] = field(default_factory=list)
        semantic_layout: dict = field(default_factory=dict)
        bubble_positions: list[tuple[float, float]] = field(default_factory=list)

    @_dc
    class Zone:
        id: str
        name: str
        zone_type: ZoneType
        x: float
        y: float
        width: float
        height: float
        metadata: dict = field(default_factory=dict)
        grid: _Grid | None = None

    @_dc
    class _Anchor:
        x: float
        y: float
        name: str = ""

    @_dc
    class Template:
        name: str
        image_path: str
        width: int
        height: int
        anchors: list[_Anchor] = field(default_factory=list)
        zones: list[Zone] = field(default_factory=list)
        metadata: dict = field(default_factory=dict)


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
class ZonePlan:
    zone: Zone
    bounds_px: tuple[int, int, int, int]
    centers_px: np.ndarray
    rows: int
    cols: int
    radius: int
    options: list[str]
    spatial_order_idx: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    ordered_centers_px: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))


@dataclass
class TemplateRuntimePlan:
    key: tuple[int, int, int, int, int]
    target_width: int
    target_height: int
    sorted_zones: list[Zone] = field(default_factory=list)
    anchor_points_px: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    anchor_names: list[str] = field(default_factory=list)
    digit_anchor_points_px: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    digit_anchor_names: list[str] = field(default_factory=list)
    zone_plans: dict[str, ZonePlan] = field(default_factory=dict)


class OMRProcessor:
    """
    Hybrid scanner-locked OMR engine.

    Principles:
    - Template is fixed and fully annotated.
    - Registration uses anchor search in small ROIs near expected positions.
    - A light global affine is estimated first.
    - Each zone then gets a tiny local dx/dy refinement.
    - Header zones get an extra digit-anchor ruler refinement.
    - Bubble reading is direct from template centers using integral-image scores.
    - Correctness is preferred over aggressive speed hacks.
    """

    def __init__(
        self,
        fill_threshold: float = 0.45,
        empty_threshold: float = 0.20,
        certainty_margin: float = 0.08,
        debug_mode: bool = False,
        debug_dir: str | Path | None = None,
    ):
        self.fill_threshold = float(fill_threshold)
        self.empty_threshold = float(empty_threshold)
        self.certainty_margin = float(certainty_margin)
        self.debug_mode = bool(debug_mode)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.processing_time_limit_sec: float = 8.0
        self.alignment_profile: str = "scanner_locked"
        self._processing_deadline_monotonic: float | None = None
        self._template_plan_cache: dict[tuple[int, int, int, int, int], TemplateRuntimePlan] = {}
        self._batch_worker_local = None
        self._last_alignment_debug: dict[str, object] = {}

        # UI sliders are typically expressed as 0..1 values.
        # Normalize once and reuse these conservative gates everywhere so
        # blank bubbles are less likely to be classified as marked.
        self._fill_level = self._normalized_slider_value(self.fill_threshold, default=0.45)
        self._empty_level = self._normalized_slider_value(self.empty_threshold, default=0.20)
        self._certainty_level = self._normalized_slider_value(self.certainty_margin, default=0.08)

    # ------------------------------------------------------------------
    # Metadata / mode helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _md(template: Template) -> dict:
        return dict(getattr(template, "metadata", {}) or {})

    def _is_scanner_locked_mode(self, template: Template) -> bool:
        md = self._md(template)
        profile = str(md.get("alignment_profile", "") or "").strip().lower()
        return bool(md.get("scanner_locked_mode", False) or profile in {"scanner_locked", "hybrid"})

    @staticmethod
    def _scanner_locked_allow_full_fallback(template: Template) -> bool:
        md = getattr(template, "metadata", {}) or {}
        raw = md.get("scanner_locked_allow_full_fallback", False)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _identifier_recognition_enabled(template: Template) -> bool:
        md = getattr(template, "metadata", {}) or {}
        raw = md.get("enable_identifier_recognition", True)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() not in {"0", "false", "off", "no"}

    @staticmethod
    def _should_force_grayscale_load(template: Template) -> bool:
        md = getattr(template, "metadata", {}) or {}
        raw = md.get("force_grayscale_load", True)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _use_shape_based_dpi_normalization(template: Template) -> bool:
        # Kept only for backward compatibility with existing callers / metadata.
        # Production path now avoids pre-normalizing DPI because the image is resized
        # exactly once to the template size after loading.
        return False

    @staticmethod
    def _zone_priority(zone: Zone) -> tuple[int, str]:
        zt = getattr(zone, "zone_type", None)
        order = {
            ZoneType.MCQ_BLOCK: 0,
            ZoneType.TRUE_FALSE_BLOCK: 1,
            ZoneType.NUMERIC_BLOCK: 2,
            ZoneType.STUDENT_ID_BLOCK: 3,
            ZoneType.EXAM_CODE_BLOCK: 4,
        }
        return (order.get(zt, 5), str(getattr(zone, "id", "") or ""))

    def _time_budget_exceeded(self) -> bool:
        return bool(self._processing_deadline_monotonic and time.monotonic() >= self._processing_deadline_monotonic)

    @staticmethod
    def _normalized_slider_value(value: float | int | None, default: float) -> float:
        try:
            v = float(value)
        except Exception:
            v = float(default)
        if v < 0.0:
            v = 0.0
        if v <= 1.0:
            return float(max(0.0, min(1.0, v)))
        if v <= 100.0:
            return float(max(0.0, min(1.0, v / 100.0)))
        if v <= 255.0:
            return float(max(0.0, min(1.0, v / 255.0)))
        return float(max(0.0, min(1.0, default)))

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _zone_score_floor(self, zone: Zone, zone_type_name: str) -> float:
        md = dict(getattr(zone, "metadata", {}) or {})
        if "score_floor" in md:
            return max(0.0, self._safe_float(md.get("score_floor"), 0.0))
        level = self._fill_level
        defaults = {
            "MCQ_BLOCK": 16.0 + 22.0 * level,
            "TRUE_FALSE_BLOCK": 15.0 + 20.0 * level,
            "NUMERIC_BLOCK": 15.0 + 20.0 * level,
            "STUDENT_ID_BLOCK": 14.0 + 18.0 * level,
            "EXAM_CODE_BLOCK": 14.0 + 18.0 * level,
        }
        return float(defaults.get(zone_type_name, 16.0 + 20.0 * level))

    def _zone_delta_floor(self, zone: Zone, zone_type_name: str) -> float:
        md = dict(getattr(zone, "metadata", {}) or {})
        if "delta_floor" in md:
            return max(0.0, self._safe_float(md.get("delta_floor"), 0.0))
        level = self._certainty_level
        defaults = {
            "MCQ_BLOCK": 5.0 + 24.0 * level,
            "TRUE_FALSE_BLOCK": 4.0 + 20.0 * level,
            "NUMERIC_BLOCK": 4.0 + 20.0 * level,
            "STUDENT_ID_BLOCK": 4.0 + 18.0 * level,
            "EXAM_CODE_BLOCK": 4.0 + 18.0 * level,
        }
        return float(defaults.get(zone_type_name, 4.0 + 20.0 * level))

    def _zone_ratio_floor(self, zone: Zone, zone_type_name: str) -> float:
        md = dict(getattr(zone, "metadata", {}) or {})
        if "ratio_floor" in md:
            return max(1.0, self._safe_float(md.get("ratio_floor"), 1.0))
        level = self._certainty_level
        defaults = {
            "MCQ_BLOCK": 1.18 + 1.05 * level,
            "TRUE_FALSE_BLOCK": 1.10 + 0.95 * level,
            "NUMERIC_BLOCK": 1.10 + 0.95 * level,
            "STUDENT_ID_BLOCK": 1.08 + 0.85 * level,
            "EXAM_CODE_BLOCK": 1.08 + 0.85 * level,
        }
        return float(defaults.get(zone_type_name, 1.10 + 0.95 * level))

    def _filled_state_threshold(self, zone: Zone, zone_type_name: str) -> float:
        md = dict(getattr(zone, "metadata", {}) or {})
        if "state_threshold" in md:
            return max(0.0, self._safe_float(md.get("state_threshold"), 0.0))
        base = self._zone_score_floor(zone, zone_type_name)
        return float(max(10.0, base * 0.88))

    def _blank_state_threshold(self, zone: Zone, zone_type_name: str) -> float:
        md = dict(getattr(zone, "metadata", {}) or {})
        if "blank_threshold" in md:
            return max(0.0, self._safe_float(md.get("blank_threshold"), 0.0))
        empty = max(0.05, self._empty_level)
        return float(max(6.0, (8.0 + 18.0 * empty)))

    # ------------------------------------------------------------------
    # Template plan
    # ------------------------------------------------------------------
    @staticmethod
    def _template_plan_key(template: Template) -> tuple[int, int, int, int, int]:
        return (id(template), int(template.width), int(template.height), len(getattr(template, "zones", []) or []), len(getattr(template, "anchors", []) or []))

    @staticmethod
    def _anchor_point_px(anchor, template: Template) -> tuple[float, float]:
        x = float(anchor.x * template.width if float(anchor.x) <= 1.0 else anchor.x)
        y = float(anchor.y * template.height if float(anchor.y) <= 1.0 else anchor.y)
        return x, y

    @staticmethod
    def _zone_bounds_px(zone: Zone, template: Template) -> tuple[int, int, int, int]:
        x = int(round(zone.x * template.width if float(zone.x) <= 1.0 else zone.x))
        y = int(round(zone.y * template.height if float(zone.y) <= 1.0 else zone.y))
        w = int(round(zone.width * template.width if float(zone.width) <= 1.0 else zone.width))
        h = int(round(zone.height * template.height if float(zone.height) <= 1.0 else zone.height))
        return x, y, max(1, w), max(1, h)

    @staticmethod
    def _grid_rows_cols(zone: Zone) -> tuple[int, int]:
        grid = getattr(zone, "grid", None)
        rows = int(getattr(grid, "rows", 0) or 0)
        cols = int(getattr(grid, "cols", 0) or 0)
        if rows <= 0 and cols > 0 and getattr(grid, "bubble_positions", None):
            rows = max(1, len(grid.bubble_positions) // cols)
        if cols <= 0 and rows > 0 and getattr(grid, "bubble_positions", None):
            cols = max(1, len(grid.bubble_positions) // rows)
        return max(1, rows), max(1, cols)

    def _zone_centers_px(self, zone: Zone, template: Template) -> np.ndarray:
        grid = getattr(zone, "grid", None)
        positions = getattr(grid, "bubble_positions", None) or []
        pts: list[tuple[float, float]] = []
        for x, y in positions:
            px = float(x * template.width if float(x) <= 1.0 else x)
            py = float(y * template.height if float(y) <= 1.0 else y)
            pts.append((px, py))
        return np.asarray(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)

    @staticmethod
    def _compute_spatial_order_index(centers: np.ndarray, rows: int, cols: int) -> np.ndarray:
        need = max(0, int(rows) * int(cols))
        if need <= 0:
            return np.empty((0,), dtype=np.int32)
        if centers.size == 0:
            return np.zeros((need,), dtype=np.int32)
        pts = np.asarray(centers, dtype=np.float32)
        if pts.shape[0] < need:
            pad = np.repeat(pts[-1:, :], need - pts.shape[0], axis=0)
            pts = np.vstack([pts, pad])
        pts = pts[:need]
        order_x = np.argsort(pts[:, 0], kind="mergesort")
        col_groups = np.array_split(order_x, cols)
        matrix_idx = np.zeros((rows, cols), dtype=np.int32)
        for c in range(cols):
            grp = np.asarray(col_groups[c], dtype=np.int32)
            if grp.size == 0:
                fill = int(col_groups[c - 1][-1]) if c > 0 and len(col_groups[c - 1]) else 0
                matrix_idx[:, c] = fill
                continue
            grp = grp[np.argsort(pts[grp, 1], kind="mergesort")]
            if grp.size < rows:
                grp = np.concatenate([grp, np.full((rows - grp.size,), int(grp[-1]), dtype=np.int32)])
            matrix_idx[:, c] = grp[:rows]
        return matrix_idx.reshape(-1).astype(np.int32)

    @staticmethod
    def _estimate_radius(zone: Zone, centers: np.ndarray, rows: int, cols: int) -> int:
        md = getattr(zone, "metadata", {}) or {}
        if "bubble_radius" in md:
            try:
                return max(2, int(round(float(md.get("bubble_radius", 9)))))
            except Exception:
                pass
        if centers.shape[0] >= 2:
            pts = centers.reshape(rows, cols, 2) if centers.shape[0] == rows * cols else centers
            dx = []
            dy = []
            if centers.shape[0] == rows * cols:
                if cols > 1:
                    dx = np.diff(pts[:, :, 0], axis=1).ravel().tolist()
                if rows > 1:
                    dy = np.diff(pts[:, :, 1], axis=0).ravel().tolist()
            spacings = [v for v in dx + dy if v > 1.0]
            if spacings:
                return max(2, int(round(0.22 * float(np.median(spacings)))))
        return 9

    def _get_manual_digit_anchor_points(self, template: Template) -> tuple[np.ndarray, list[str]]:
        pts = []
        names = []
        for a in getattr(template, "anchors", []) or []:
            name = str(getattr(a, "name", "") or "")
            if name.startswith("DIGIT_ANCHOR_"):
                pts.append(self._anchor_point_px(a, template))
                names.append(name)
        if not pts:
            return np.empty((0, 2), dtype=np.float32), []
        order = np.argsort(np.asarray(pts, dtype=np.float32)[:, 1])
        pts_arr = np.asarray(pts, dtype=np.float32)[order]
        names_sorted = [names[i] for i in order.tolist()]
        return pts_arr, names_sorted

    def _build_template_runtime_plan(self, template: Template) -> TemplateRuntimePlan:
        key = self._template_plan_key(template)
        anchors = []
        anchor_names = []
        for a in getattr(template, "anchors", []) or []:
            name = str(getattr(a, "name", "") or "")
            if name.startswith("DIGIT_ANCHOR_"):
                continue
            anchors.append(self._anchor_point_px(a, template))
            anchor_names.append(name or f"A{len(anchor_names)+1}")
        digit_pts, digit_names = self._get_manual_digit_anchor_points(template)
        zone_plans: dict[str, ZonePlan] = {}
        zones_sorted = sorted(list(getattr(template, "zones", []) or []), key=self._zone_priority)
        for z in zones_sorted:
            if getattr(z, "grid", None) is None:
                continue
            zt = getattr(z, "zone_type", None)
            if zt == ZoneType.ANCHOR:
                continue
            rows, cols = self._grid_rows_cols(z)
            centers = self._zone_centers_px(z, template)
            bounds = self._zone_bounds_px(z, template)
            radius = self._estimate_radius(z, centers, rows, cols)
            options = list(getattr(getattr(z, "grid", None), "options", []) or [])
            spatial_order_idx = self._compute_spatial_order_index(centers, rows, cols)
            ordered_centers = centers[spatial_order_idx] if centers.size and spatial_order_idx.size else centers
            zone_plans[str(getattr(z, "id", "") or "")] = ZonePlan(
                zone=z,
                bounds_px=bounds,
                centers_px=centers,
                rows=rows,
                cols=cols,
                radius=radius,
                options=options,
                spatial_order_idx=spatial_order_idx,
                ordered_centers_px=ordered_centers.astype(np.float32) if ordered_centers.size else np.empty((0, 2), dtype=np.float32),
            )
        return TemplateRuntimePlan(
            key=key,
            target_width=int(template.width),
            target_height=int(template.height),
            sorted_zones=zones_sorted,
            anchor_points_px=np.asarray(anchors, dtype=np.float32) if anchors else np.empty((0, 2), dtype=np.float32),
            anchor_names=anchor_names,
            digit_anchor_points_px=digit_pts,
            digit_anchor_names=digit_names,
            zone_plans=zone_plans,
        )

    def _get_template_runtime_plan(self, template: Template) -> TemplateRuntimePlan:
        key = self._template_plan_key(template)
        cached = self._template_plan_cache.get(key)
        if cached is not None:
            return cached
        plan = self._build_template_runtime_plan(template)
        self._template_plan_cache[key] = plan
        return plan

    # ------------------------------------------------------------------
    # Image preprocess / registration
    # ------------------------------------------------------------------
    def _load_image_normalized_to_200_dpi(self, path: str, template: Template | None = None) -> tuple[np.ndarray | None, str]:
        # Production path for this project: input sheets are already standardized to
        # A4 at either 150 DPI or 200 DPI. To avoid double-normalization cost and
        # geometric drift, load once and resize exactly once later to template size.
        force_gray = True if template is None else self._should_force_grayscale_load(template)
        if force_gray:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return None, ""
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "A4 input assumed: 150/200 DPI; deferred direct resize to template."
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None, ""
        return img, "A4 input assumed: 150/200 DPI; deferred direct resize to template."

    @staticmethod
    def _preprocess_fast(image: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        dark = cv2.normalize(255 - blur, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return {"gray": gray, "blur": blur, "dark": dark, "binary": binary}

    @staticmethod
    def _build_integral(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.float32:
            arr = img.astype(np.float32)
        else:
            arr = img
        return cv2.integral(arr)

    @staticmethod
    def _sample_box_mean(integral: np.ndarray, centers: np.ndarray, radius: int, shape: tuple[int, int]) -> np.ndarray:
        if centers.size == 0:
            return np.zeros((0,), dtype=np.float32)
        h, w = shape
        r = max(1, int(radius))
        pts = np.rint(centers).astype(np.int32)
        x0 = np.clip(pts[:, 0] - r, 0, w - 1)
        y0 = np.clip(pts[:, 1] - r, 0, h - 1)
        x1 = np.clip(pts[:, 0] + r, 0, w - 1)
        y1 = np.clip(pts[:, 1] + r, 0, h - 1)
        area = np.maximum(1, (x1 - x0 + 1) * (y1 - y0 + 1)).astype(np.float32)
        s = integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
        return (s.astype(np.float32) / area).astype(np.float32)

    @staticmethod
    def _sample_ring_score(integral: np.ndarray, centers: np.ndarray, radius: int, shape: tuple[int, int]) -> np.ndarray:
        if centers.size == 0:
            return np.zeros((0,), dtype=np.float32)
        inner = OMRProcessor._sample_box_mean(integral, centers, max(1, int(round(radius * 0.60))), shape)
        outer = OMRProcessor._sample_box_mean(integral, centers, max(2, int(round(radius * 1.10))), shape)
        return np.clip(inner - (0.55 * np.maximum(0.0, outer - inner)), 0.0, 255.0)

    @staticmethod
    def _small_offsets(kind: str) -> np.ndarray:
        if kind == "header":
            vals = []
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    vals.append((dx, dy))
            return np.asarray(vals, dtype=np.float32)
        if kind == "numeric":
            vals = []
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    vals.append((dx, dy))
            return np.asarray(vals, dtype=np.float32)
        if kind == "mcq":
            vals = []
            for dy in (-3, -2, -1, 0, 1, 2, 3):
                for dx in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
                    vals.append((dx, dy))
            return np.asarray(vals, dtype=np.float32)
        return np.asarray([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)], dtype=np.float32)

    @staticmethod
    def _scores_to_spatial_matrix(points: np.ndarray, scores: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
        """Fallback reorder without cached plan index."""
        points = np.asarray(points, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        need = max(0, int(rows) * int(cols))
        if need <= 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 0, 2), dtype=np.float32)
        if points.shape[0] < need:
            pad_pts = np.zeros((need - points.shape[0], 2), dtype=np.float32)
            points = np.vstack([points, pad_pts]) if points.size else pad_pts
        if scores.shape[0] < need:
            scores = np.pad(scores, (0, need - scores.shape[0]), constant_values=0.0)
        points = points[:need]
        scores = scores[:need]
        order_x = np.argsort(points[:, 0], kind="mergesort")
        px = points[order_x]
        ps = scores[order_x]
        col_groups_pts = np.array_split(px, cols)
        col_groups_scores = np.array_split(ps, cols)
        mat = np.zeros((rows, cols), dtype=np.float32)
        mat_pts = np.zeros((rows, cols, 2), dtype=np.float32)
        for c in range(cols):
            gpts = np.asarray(col_groups_pts[c], dtype=np.float32)
            gs = np.asarray(col_groups_scores[c], dtype=np.float32)
            if gpts.size == 0:
                continue
            order_y = np.argsort(gpts[:, 1], kind="mergesort")
            gpts = gpts[order_y]
            gs = gs[order_y]
            if gpts.shape[0] < rows:
                if gpts.shape[0] == 0:
                    continue
                pad_pts = np.repeat(gpts[-1:, :], rows - gpts.shape[0], axis=0)
                pad_scores = np.zeros((rows - gs.shape[0],), dtype=np.float32)
                gpts = np.vstack([gpts, pad_pts])
                gs = np.concatenate([gs, pad_scores])
            mat[:, c] = gs[:rows]
            mat_pts[:, c, :] = gpts[:rows]
        return mat, mat_pts

    @staticmethod
    def _reorder_sampled_matrix(points: np.ndarray, scores: np.ndarray, rows: int, cols: int, order_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        need = max(0, int(rows) * int(cols))
        if need <= 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32)
        sc = np.asarray(scores, dtype=np.float32).reshape(-1)
        if pts.shape[0] < need:
            pad_pts = np.zeros((need - pts.shape[0], 2), dtype=np.float32)
            pts = np.vstack([pts, pad_pts]) if pts.size else pad_pts
        if sc.shape[0] < need:
            sc = np.pad(sc, (0, need - sc.shape[0]), constant_values=0.0)
        pts = pts[:need]
        sc = sc[:need]
        idx = np.asarray(order_idx if order_idx is not None else [], dtype=np.int32).reshape(-1)
        if idx.size >= need and pts.shape[0] > 0:
            safe_idx = np.clip(idx[:need], 0, pts.shape[0] - 1)
            ordered_points = pts[safe_idx]
            ordered_scores = sc[safe_idx]
            return ordered_scores.reshape(rows, cols), ordered_points.reshape(rows, cols, 2)
        return OMRProcessor._scores_to_spatial_matrix(pts, sc, rows, cols)

    def _sample_with_local_recenter(self, integral: np.ndarray, centers: np.ndarray, radius: int, shape: tuple[int, int], *, kind: str) -> tuple[np.ndarray, np.ndarray]:
        if centers.size == 0:
            return np.zeros((0,), dtype=np.float32), centers.astype(np.float32)
        offsets = self._small_offsets(kind)
        best_scores = np.full((len(centers),), -1e9, dtype=np.float32)
        best_centers = centers.astype(np.float32).copy()
        for dx, dy in offsets:
            shifted = centers.astype(np.float32).copy()
            shifted[:, 0] += float(dx)
            shifted[:, 1] += float(dy)
            scores = self._sample_ring_score(integral, shifted, radius, shape)
            mask = scores > best_scores
            best_scores[mask] = scores[mask]
            best_centers[mask] = shifted[mask]
        return best_scores.astype(np.float32), best_centers.astype(np.float32)

    @staticmethod
    def _strong_score_mean(scores: np.ndarray) -> float:
        arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return -1e18
        take = max(2, int(round(arr.size * 0.5)))
        if take >= arr.size:
            return float(np.mean(arr))
        part = np.partition(arr, -take)[-take:]
        return float(np.mean(part))

    def _sample_mcq_with_axis_refine(
        self,
        integral: np.ndarray,
        centers: np.ndarray,
        radius: int,
        shape: tuple[int, int],
        rows: int,
        cols: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if centers.size == 0:
            return np.zeros((0,), dtype=np.float32), centers.astype(np.float32)
        pts = centers.astype(np.float32).reshape(-1, 2)
        if pts.shape[0] != int(rows) * int(cols):
            return self._sample_with_local_recenter(integral, pts, radius, shape, kind="mcq")
        grid = pts.reshape(rows, cols, 2).astype(np.float32).copy()

        # Refine each answer column horizontally. This is especially important for
        # right-side MCQ blocks where a tiny residual affine drift can confidently
        # push sampling into the neighbouring bubble.
        for c in range(cols):
            base = grid[:, c, :].copy()
            best_dx = 0.0
            best_score = -1e18
            for dx in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
                probe = base.copy()
                probe[:, 0] += float(dx)
                score = self._strong_score_mean(self._sample_ring_score(integral, probe, radius, shape))
                if score > best_score:
                    best_score = score
                    best_dx = float(dx)
            grid[:, c, 0] += best_dx

        # Refine each question row vertically so lower rows in the block do not
        # inherit a stale y-position after the global affine fit.
        for r in range(rows):
            base = grid[r, :, :].copy()
            best_dy = 0.0
            best_score = -1e18
            for dy in (-3, -2, -1, 0, 1, 2, 3):
                probe = base.copy()
                probe[:, 1] += float(dy)
                score = self._strong_score_mean(self._sample_ring_score(integral, probe, radius, shape))
                if score > best_score:
                    best_score = score
                    best_dy = float(dy)
            grid[r, :, 1] += best_dy

        flat = grid.reshape(-1, 2)
        return self._sample_with_local_recenter(integral, flat, radius, shape, kind="mcq")

    def _find_anchor_in_roi(self, binary: np.ndarray, expected_xy: tuple[float, float], search_radius: int = 24) -> tuple[float, float] | None:
        h, w = binary.shape[:2]
        ex, ey = expected_xy
        x0 = max(0, int(round(ex - search_radius)))
        y0 = max(0, int(round(ey - search_radius)))
        x1 = min(w, int(round(ex + search_radius + 1)))
        y1 = min(h, int(round(ey + search_radius + 1)))
        roi = binary[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), connectivity=8)
        best = None
        best_score = -1e18
        for i in range(1, num_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < 8:
                continue
            bw = float(max(1, stats[i, cv2.CC_STAT_WIDTH]))
            bh = float(max(1, stats[i, cv2.CC_STAT_HEIGHT]))
            aspect = min(bw, bh) / max(bw, bh)
            density = area / max(1.0, bw * bh)
            cx = float(x0 + centroids[i, 0])
            cy = float(y0 + centroids[i, 1])
            dist = math.hypot(cx - ex, cy - ey)
            score = (area * 2.0) + (density * 15.0) + (aspect * 12.0) - (dist * 2.5)
            if score > best_score:
                best_score = score
                best = (cx, cy)
        return best

    def _estimate_affine(self, binary: np.ndarray, plan: TemplateRuntimePlan) -> tuple[np.ndarray, list[tuple[float, float]], dict[str, object]]:
        found_src: list[tuple[float, float]] = []
        found_dst: list[tuple[float, float]] = []
        detected_points: list[tuple[float, float]] = []
        per_anchor: list[dict[str, object]] = []
        for pt, name in zip(plan.anchor_points_px.tolist(), plan.anchor_names):
            matched = self._find_anchor_in_roi(binary, (float(pt[0]), float(pt[1])), search_radius=26)
            row = {"name": str(name), "expected": (float(pt[0]), float(pt[1])), "success": bool(matched)}
            if matched is not None:
                found_src.append((float(matched[0]), float(matched[1])))
                found_dst.append((float(pt[0]), float(pt[1])))
                detected_points.append((float(matched[0]), float(matched[1])))
                row["detected"] = (float(matched[0]), float(matched[1]))
            per_anchor.append(row)
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        success = False
        mean_error = 0.0
        if len(found_src) >= 3:
            src = np.asarray(found_src, dtype=np.float32)
            dst = np.asarray(found_dst, dtype=np.float32)
            aff, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if aff is not None:
                matrix = aff.astype(np.float32)
                proj = cv2.transform(src.reshape(1, -1, 2), matrix).reshape(-1, 2)
                err = np.linalg.norm(proj - dst, axis=1)
                mean_error = float(np.mean(err)) if len(err) else 0.0
                success = bool(np.isfinite(mean_error) and mean_error <= 8.0)
        debug = {
            "registration_ok": bool(success),
            "patch_valid_count": int(len(found_src)),
            "patch_inlier_count": int(len(found_src)),
            "patch_score_mean": float(max(0.0, 1.0 - (mean_error / 12.0))) if len(found_src) >= 3 else 0.0,
            "mean_error": float(mean_error),
            "transform_matrix": matrix.astype(float).tolist(),
            "per_patch": per_anchor,
            "patch_matches": [dict(row) for row in per_anchor],
        }
        return matrix, detected_points, debug

    @staticmethod
    def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points.astype(np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        return cv2.transform(pts, matrix).reshape(-1, 2).astype(np.float32)

    def _detect_digit_anchor_ruler(self, binary: np.ndarray, plan: TemplateRuntimePlan, matrix: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
        if plan.digit_anchor_points_px.size == 0:
            return np.empty((0, 2), dtype=np.float32), []
        expected = self._transform_points(plan.digit_anchor_points_px, matrix)
        found = []
        for pt in expected.tolist():
            m = self._find_anchor_in_roi(binary, (float(pt[0]), float(pt[1])), search_radius=18)
            if m is not None:
                found.append((float(m[0]), float(m[1])))
        if not found:
            return np.empty((0, 2), dtype=np.float32), []
        arr = np.asarray(found, dtype=np.float32)
        arr = arr[np.argsort(arr[:, 1])]
        return arr, [(float(x), float(y)) for x, y in arr]

    def _estimate_zone_shift(self, integral: np.ndarray, centers: np.ndarray, radius: int, shape: tuple[int, int], zone_type_name: str) -> tuple[float, float, float]:
        if centers.size == 0:
            return 0.0, 0.0, 0.0
        sample_idx = np.linspace(0, len(centers) - 1, min(len(centers), 36)).astype(int)
        sample = centers[sample_idx].astype(np.float32)
        if zone_type_name in {"STUDENT_ID_BLOCK", "EXAM_CODE_BLOCK"}:
            kind = "header"
        elif zone_type_name == "NUMERIC_BLOCK":
            kind = "numeric"
        elif zone_type_name == "MCQ_BLOCK":
            kind = "mcq"
        else:
            kind = "zone"
        offsets = self._small_offsets(kind)
        best_score = -1e18
        best_dx = 0.0
        best_dy = 0.0
        for dx, dy in offsets:
            shifted = sample.copy()
            shifted[:, 0] += float(dx)
            shifted[:, 1] += float(dy)
            scores = self._sample_ring_score(integral, shifted, radius, shape)
            if scores.size == 0:
                continue
            k = max(4, int(round(scores.size * 0.35)))
            part = np.partition(scores, -k)[-k:]
            score = float(np.mean(part))
            if score > best_score:
                best_score = score
                best_dx = float(dx)
                best_dy = float(dy)
        return best_dx, best_dy, float(best_score)

    # ------------------------------------------------------------------
    # Zone decoding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _zone_type_name(zone: Zone) -> str:
        zt = getattr(zone, "zone_type", None)
        return str(getattr(zt, "value", zt) or "")

    @staticmethod
    def _pick_best_index(
        scores: np.ndarray,
        threshold: float,
        margin: float,
        *,
        absolute_floor: float = 0.0,
        delta_floor: float = 0.0,
        ratio_floor: float = 1.0,
    ) -> tuple[int | None, float, str]:
        if scores.size == 0:
            return None, 0.0, "empty"
        arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        order = np.argsort(arr)[::-1]
        top_i = int(order[0])
        top = float(arr[top_i])
        second = float(arr[int(order[1])]) if len(order) > 1 else 0.0
        effective_threshold = max(float(threshold), float(absolute_floor))
        effective_margin = max(float(margin), float(delta_floor))
        if top < effective_threshold:
            return None, max(0.0, top - effective_threshold), "below threshold"
        if (top - second) < effective_margin:
            return None, max(0.0, top - second), "ambiguous"
        if second > 1e-6 and (top / second) < max(1.0, float(ratio_floor)):
            return None, max(0.0, top / second), "weak ratio"
        return top_i, float(top - second), "ok"

    def _row_pick_threshold(self, scores: np.ndarray, zone: Zone, zone_type_name: str, *, is_column: bool = False) -> float:
        arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return self._zone_score_floor(zone, zone_type_name)
        mean_v = float(np.mean(arr))
        median_v = float(np.median(arr))
        max_v = float(np.max(arr))
        if is_column:
            return float(max(
                self._zone_score_floor(zone, zone_type_name),
                mean_v + 3.0,
                median_v + 4.0,
                max_v * (0.60 + 0.12 * self._fill_level),
            ))
        return float(max(
            self._zone_score_floor(zone, zone_type_name),
            mean_v + 3.5,
            median_v + 5.0,
            max_v * (0.60 + 0.14 * self._fill_level),
        ))

    def _bubble_states_from_scores(self, scores: np.ndarray, threshold: float | None = None) -> list[bool]:
        t = (15.0 + 20.0 * self._fill_level) if threshold is None else float(threshold)
        return [bool(v >= t) for v in np.asarray(scores, dtype=np.float32).tolist()]

    def _zone_weak_decision_count(self, mat: np.ndarray, zone: Zone, zone_type_name: str) -> tuple[int, int]:
        rows, cols = mat.shape if mat.ndim == 2 else (0, 0)
        if rows <= 0 or cols <= 0:
            return 0, 0
        blank_thr = self._blank_state_threshold(zone, zone_type_name)
        score_floor = self._zone_score_floor(zone, zone_type_name)
        delta_floor = self._zone_delta_floor(zone, zone_type_name)
        ratio_floor = self._zone_ratio_floor(zone, zone_type_name)
        weak = 0
        total = 0
        if zone_type_name in {"NUMERIC_BLOCK", "STUDENT_ID_BLOCK", "EXAM_CODE_BLOCK"}:
            vectors = [mat[:, c] for c in range(cols)]
        else:
            vectors = [mat[r, :] for r in range(rows)]
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            total += 1
            order = np.argsort(arr)[::-1]
            top = float(arr[int(order[0])])
            second = float(arr[int(order[1])]) if arr.size > 1 else 0.0
            if top <= blank_thr:
                continue
            ratio = float(top / max(second, 1e-6)) if second > 0.0 else 9999.0
            if top < score_floor or (top - second) < delta_floor or ratio < ratio_floor:
                weak += 1
        return weak, total

    def _zone_result_needs_fallback(self, mat: np.ndarray, zone: Zone, zone_type_name: str) -> tuple[bool, int, int]:
        weak, total = self._zone_weak_decision_count(mat, zone, zone_type_name)
        if total <= 0:
            return False, weak, total
        if zone_type_name == "NUMERIC_BLOCK":
            return weak > 0, weak, total
        return weak > max(1, int(round(total * 0.12))), weak, total

    def _identifier_result_needs_fallback(self, mat: np.ndarray, zone: Zone, value: str, expected_cols: int) -> tuple[bool, int, int]:
        weak, total = self._zone_weak_decision_count(mat, zone, self._zone_type_name(zone))
        value_ok = bool(value and value != "-" and len(value) == expected_cols)
        return (not value_ok) or weak > 0, weak, total

    def _mcq_row_edge_boost(self, row_idx: int, q_count: int) -> float:
        if q_count <= 0:
            return 1.0
        if row_idx == 0:
            return 1.08
        if row_idx == q_count - 1:
            return 1.04
        if row_idx == 1:
            return 1.03
        return 1.0

    def _mcq_soft_fallback_pick(
        self,
        row: np.ndarray,
        zone: Zone,
        zone_type_name: str,
        row_idx: int,
        q_count: int,
    ) -> tuple[int | None, float, str]:
        # Soft fallback is intentionally limited to the first MCQ row,
        # which is the row most likely to be clipped by the block title / border.
        if row_idx != 0:
            return None, 0.0, "disabled"
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None, 0.0, "empty"
        order = np.argsort(arr)[::-1]
        top_i = int(order[0])
        top = float(arr[top_i])
        second = float(arr[int(order[1])]) if len(order) > 1 else 0.0
        delta = float(top - second)
        ratio = float(top / max(second, 1e-6))

        strict_thr = self._row_pick_threshold(arr, zone, zone_type_name, is_column=False)
        score_floor = self._zone_score_floor(zone, zone_type_name)
        delta_floor = self._zone_delta_floor(zone, zone_type_name)
        ratio_floor = self._zone_ratio_floor(zone, zone_type_name)

        soft_thr = max(score_floor * 0.82, strict_thr * 0.86, float(np.mean(arr) + 4.0), float(np.median(arr) + 5.0))
        soft_delta = max(2.2, delta_floor * 0.78)
        soft_ratio = max(1.10, ratio_floor * 0.92)

        if top < soft_thr:
            return None, max(0.0, top - soft_thr), "below soft threshold"
        if delta < soft_delta:
            return None, max(0.0, delta - soft_delta), "soft ambiguous"
        if ratio < soft_ratio:
            return None, max(0.0, ratio - soft_ratio), "soft weak ratio"
        return top_i, delta, "soft ok"

    def _decode_mcq_zone(self, mat: np.ndarray, zone: Zone, result: OMRResult) -> None:
        grid = zone.grid
        if grid is None:
            return
        rows, cols = mat.shape
        ztype = self._zone_type_name(zone)
        labels = list(getattr(grid, "options", []) or [chr(65 + i) for i in range(cols)])
        if len(labels) < cols:
            labels.extend(chr(65 + i) for i in range(len(labels), cols))
        confs: list[float] = []
        q_count = min(int(getattr(grid, "question_count", 0) or rows), rows)
        for r in range(q_count):
            row = np.asarray(mat[r], dtype=np.float32).reshape(-1)
            edge_boost = self._mcq_row_edge_boost(r, q_count)
            boosted_row = row * edge_boost if edge_boost != 1.0 else row
            thr = self._row_pick_threshold(boosted_row, zone, ztype, is_column=False)
            idx, conf, reason = self._pick_best_index(
                boosted_row,
                thr,
                self._zone_delta_floor(zone, ztype),
                absolute_floor=self._zone_score_floor(zone, ztype),
                delta_floor=self._zone_delta_floor(zone, ztype),
                ratio_floor=self._zone_ratio_floor(zone, ztype),
            )
            if idx is None:
                idx, conf, reason = self._mcq_soft_fallback_pick(boosted_row, zone, ztype, r, q_count)
            qno = int(getattr(grid, "question_start", 1) or 1) + r
            if idx is None:
                continue
            result.mcq_answers[qno] = str(labels[idx])
            confs.append(float(conf))
        result.confidence_scores[f"mcq:{zone.id}"] = float(np.mean(confs) / 255.0) if confs else 0.0

    def _decode_true_false_zone(self, mat: np.ndarray, zone: Zone, result: OMRResult) -> None:
        grid = zone.grid
        if grid is None:
            return
        ztype = self._zone_type_name(zone)
        qpb = max(1, int(zone.metadata.get("questions_per_block", 2) or 2))
        spq = max(1, int(zone.metadata.get("statements_per_question", 4) or 4))
        cps = max(1, int(zone.metadata.get("choices_per_statement", 2) or 2))
        labels = [chr(ord("a") + i) for i in range(spq)]
        confs: list[float] = []
        for q in range(qpb):
            qno = int(getattr(grid, "question_start", 1) or 1) + q
            result.true_false_answers[qno] = {}
            for sidx in range(spq):
                row_idx = q * spq + sidx
                if row_idx >= mat.shape[0]:
                    break
                row = mat[row_idx, :cps]
                thr = self._row_pick_threshold(row, zone, ztype, is_column=False)
                idx, conf, reason = self._pick_best_index(
                    row,
                    thr,
                    self._zone_delta_floor(zone, ztype),
                    absolute_floor=self._zone_score_floor(zone, ztype),
                    delta_floor=self._zone_delta_floor(zone, ztype),
                    ratio_floor=self._zone_ratio_floor(zone, ztype),
                )
                if idx is None:
                    continue
                result.true_false_answers[qno][labels[sidx]] = bool(idx == 0)
                confs.append(float(conf))
        result.confidence_scores[f"tf:{zone.id}"] = float(np.mean(confs) / 255.0) if confs else 0.0

    @staticmethod
    def _parse_index_list(raw: Iterable[int] | str, max_len: int) -> set[int]:
        if isinstance(raw, str):
            items = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            items = list(raw or [])
        out: set[int] = set()
        for item in items:
            try:
                idx = int(item) - 1
            except Exception:
                continue
            if 0 <= idx < max_len:
                out.add(idx)
        return out

    def _decode_numeric_zone(self, mat: np.ndarray, zone: Zone, result: OMRResult) -> None:
        grid = zone.grid
        if grid is None:
            return
        rows, cols = mat.shape
        ztype = self._zone_type_name(zone)
        md = dict(getattr(zone, "metadata", {}) or {})
        digits_per_answer = max(1, int(md.get("digits_per_answer", cols) or cols))
        question_count = max(1, int(md.get("questions_per_block", md.get("total_questions", 1)) or 1))
        sign_row = max(0, int(md.get("sign_row", 1) or 1) - 1)
        decimal_row = max(0, int(md.get("decimal_row", 2) or 2) - 1)
        digit_start_row = max(0, int(md.get("digit_start_row", 3) or 3) - 1)
        sign_cols = self._parse_index_list(md.get("sign_columns", [1]), digits_per_answer)
        decimal_cols = self._parse_index_list(md.get("decimal_columns", [2, 3]), digits_per_answer)
        sign_symbol = str(md.get("sign_symbol", "-") or "-")
        decimal_symbol = str(md.get("decimal_symbol", ".") or ".")
        digit_map = list(md.get("digit_map", list(range(max(1, rows - digit_start_row)))) or list(range(max(1, rows - digit_start_row))))
        options = [str(v) for v in (getattr(grid, "options", []) or [])]

        def token_for_row(row_idx: int, col_in_answer: int) -> str:
            if col_in_answer in sign_cols and row_idx == sign_row:
                return sign_symbol
            if col_in_answer in decimal_cols and row_idx == decimal_row:
                return decimal_symbol
            if options and 0 <= row_idx < len(options):
                tok = str(options[row_idx])
                if tok:
                    return tok
            digit_offset = row_idx - digit_start_row
            if 0 <= digit_offset < len(digit_map):
                return str(digit_map[digit_offset])
            return ""

        confs: list[float] = []
        for q in range(question_count):
            parts: list[str] = []
            for local_c in range(digits_per_answer):
                c = q * digits_per_answer + local_c
                if c >= cols:
                    break
                col = mat[:, c]
                thr = self._row_pick_threshold(col, zone, ztype, is_column=True)
                idx, conf, _ = self._pick_best_index(
                    col,
                    thr,
                    self._zone_delta_floor(zone, ztype),
                    absolute_floor=self._zone_score_floor(zone, ztype),
                    delta_floor=self._zone_delta_floor(zone, ztype),
                    ratio_floor=self._zone_ratio_floor(zone, ztype),
                )
                if idx is None:
                    continue
                tok = token_for_row(idx, local_c)
                if not tok:
                    continue
                if tok == decimal_symbol and decimal_symbol in parts:
                    continue
                parts.append(tok)
                confs.append(float(conf))
            value = "".join(parts).strip().replace("..", ".")
            if value in {"", sign_symbol, decimal_symbol}:
                value = ""
            qno = int(getattr(grid, "question_start", 1) or 1) + q
            result.numeric_answers[qno] = value
        result.confidence_scores[f"numeric:{zone.id}"] = float(np.mean(confs) / 255.0) if confs else 0.0

    def _decode_column_digits(self, mat: np.ndarray, zone: Zone, result: OMRResult, *, record_errors: bool = True) -> tuple[str, list[float]]:
        rows, cols = mat.shape
        digit_map = list((getattr(zone, "metadata", {}) or {}).get("digit_map", list(range(rows))) or list(range(rows)))
        digits: list[str] = []
        confs: list[float] = []
        for c in range(cols):
            col = mat[:, c]
            thr = max(16.0, float(np.mean(col) + 2.0), float(np.max(col) * 0.55))
            idx, conf, reason = self._pick_best_index(col, thr, max(2.0, self.certainty_margin * 180.0))
            if idx is None:
                digits.append("?")
                if record_errors:
                    result.recognition_errors.append(f"{self._zone_type_name(zone)} column {c+1}: {reason}")
            else:
                digits.append(str(digit_map[idx] if idx < len(digit_map) else idx))
                confs.append(float(conf))
        value = "".join(d for d in digits if d != "?")
        if "?" in digits or len(value) != cols:
            return value or "-", confs
        return value, confs

    def _align_identifier_by_digit_anchors(self, centers: np.ndarray, zone: Zone, plan: TemplateRuntimePlan, detected_digit_anchors: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        if centers.size == 0 or detected_digit_anchors.size < 2 or plan.digit_anchor_points_px.size < 2:
            return centers, {}
        expected_digit = plan.digit_anchor_points_px
        # sort both by y, fit y = a*y + b and x translation from medians.
        exp = expected_digit[np.argsort(expected_digit[:, 1])]
        det = detected_digit_anchors[np.argsort(detected_digit_anchors[:, 1])]
        n = min(len(exp), len(det))
        exp = exp[:n]
        det = det[:n]
        if n < 2:
            return centers, {}
        coeff = np.polyfit(exp[:, 1], det[:, 1], deg=1)
        a = float(coeff[0])
        b = float(coeff[1])
        dx = float(np.median(det[:, 0] - exp[:, 0]))
        out = centers.astype(np.float32).copy()
        out[:, 0] += dx
        out[:, 1] = (a * out[:, 1]) + b
        rows = max(1, int(getattr(zone.grid, "rows", 1) or 1))
        cols = max(1, int(getattr(zone.grid, "cols", 1) or 1))
        debug = {
            "guide_points": [(float(x), float(y)) for x, y in det],
            "row_lines": [float((a * y) + b) for y in exp[:, 1].tolist()],
            "anchor_points": [(float(x), float(y)) for x, y in det],
            "rows": rows,
            "cols": cols,
        }
        return out, debug

    def _decode_identifier_zone_deep(self, dark_integral: np.ndarray, shape: tuple[int, int], zplan: ZonePlan, plan: TemplateRuntimePlan, transformed_centers: np.ndarray, detected_digit_anchors: np.ndarray, result: OMRResult, context: RecognitionContext, *, store_bubble_states: bool = True) -> tuple[dict[str, object], np.ndarray, np.ndarray, list[float], str]:
        centers, debug = self._align_identifier_by_digit_anchors(transformed_centers, zplan.zone, plan, detected_digit_anchors)
        rows, cols = zplan.rows, zplan.cols
        if centers.shape[0] != rows * cols:
            centers = transformed_centers
        centers = centers.reshape(rows, cols, 2).astype(np.float32)
        for c in range(cols):
            base = centers[:, c, :].copy()
            best_dx = 0.0
            best_score = -1e18
            for dx in (-3, -2, -1, 0, 1, 2, 3):
                probe = base.copy()
                probe[:, 0] += float(dx)
                scores, _ = self._sample_with_local_recenter(dark_integral, probe, zplan.radius, shape, kind="header")
                score = float(np.mean(np.sort(scores)[-max(2, len(scores)//2):])) if scores.size else -1e18
                if score > best_score:
                    best_score = score
                    best_dx = float(dx)
            centers[:, c, 0] += best_dx
        flat = centers.reshape(-1, 2)
        scores, best_pts = self._sample_with_local_recenter(dark_integral, flat, zplan.radius, shape, kind="header")
        mat, mat_pts = self._reorder_sampled_matrix(best_pts, scores, rows, cols, zplan.spatial_order_idx)
        value, confs = self._decode_column_digits(mat, zplan.zone, result, record_errors=True)
        zone_key = "student_id" if self._zone_type_name(zplan.zone) == "STUDENT_ID_BLOCK" else "exam_code"
        if zone_key == "student_id":
            result.student_id = value
        else:
            result.exam_code = value
        mean_conf = float(np.mean(confs) / 255.0) if confs else 0.0
        result.confidence_scores[zone_key] = mean_conf
        result.confidence_scores[f"{zone_key}:{zplan.zone.id}"] = mean_conf
        if store_bubble_states:
            zone_debug = dict(context.digit_zone_debug.get(str(zplan.zone.id), {}))
            zone_debug.update(debug)
            zone_debug.update({
                "bubble_centers": [(float(x), float(y)) for x, y in mat_pts.reshape(-1, 2).tolist()],
                "recognized_points": [(float(x), float(y)) for x, y in mat_pts.reshape(-1, 2).tolist()],
                "scores": mat.tolist(),
            })
            context.digit_zone_debug[str(zplan.zone.id)] = zone_debug
            result.digit_zone_debug = dict(context.digit_zone_debug)
            ztype = self._zone_type_name(zplan.zone)
            filled_thr = self._filled_state_threshold(zplan.zone, ztype)
            context.bubble_states_by_zone[str(zplan.zone.id)] = self._bubble_states_from_scores(scores, threshold=filled_thr)
            result.bubble_states_by_zone = dict(context.bubble_states_by_zone)
        return debug, mat, mat_pts, confs, value

    def _decode_identifier_zone(self, dark_integral: np.ndarray, shape: tuple[int, int], zplan: ZonePlan, plan: TemplateRuntimePlan, transformed_centers: np.ndarray, detected_digit_anchors: np.ndarray, result: OMRResult, context: RecognitionContext, *, store_bubble_states: bool = True) -> dict[str, object]:
        centers, debug = self._align_identifier_by_digit_anchors(transformed_centers, zplan.zone, plan, detected_digit_anchors)
        rows, cols = zplan.rows, zplan.cols
        if centers.shape[0] != rows * cols:
            centers = transformed_centers
        centers = centers.reshape(rows, cols, 2).astype(np.float32)
        for c in range(cols):
            base = centers[:, c, :].copy()
            best_dx = 0.0
            best_score = -1e18
            for dx in (-3, -2, -1, 0, 1, 2, 3):
                probe = base.copy()
                probe[:, 0] += float(dx)
                scores = self._sample_ring_score(dark_integral, probe, zplan.radius, shape)
                score = float(np.mean(np.sort(scores)[-max(2, len(scores)//2):])) if scores.size else -1e18
                if score > best_score:
                    best_score = score
                    best_dx = float(dx)
            centers[:, c, 0] += best_dx
        flat = centers.reshape(-1, 2)
        scores = self._sample_ring_score(dark_integral, flat, zplan.radius, shape)
        mat, mat_pts = self._reorder_sampled_matrix(flat, scores, rows, cols, zplan.spatial_order_idx)
        fast_value, fast_confs = self._decode_column_digits(mat, zplan.zone, result, record_errors=False)
        need_fallback, weak_count, total_checks = self._identifier_result_needs_fallback(mat, zplan.zone, fast_value, cols)
        fallback_used = False
        if need_fallback:
            fallback_used = True
            debug, mat, mat_pts, fast_confs, fast_value = self._decode_identifier_zone_deep(
                dark_integral,
                shape,
                zplan,
                plan,
                transformed_centers,
                detected_digit_anchors,
                result,
                context,
                store_bubble_states=store_bubble_states,
            )
        else:
            zone_key = "student_id" if self._zone_type_name(zplan.zone) == "STUDENT_ID_BLOCK" else "exam_code"
            if zone_key == "student_id":
                result.student_id = fast_value
            else:
                result.exam_code = fast_value
            mean_conf = float(np.mean(fast_confs) / 255.0) if fast_confs else 0.0
            result.confidence_scores[zone_key] = mean_conf
            result.confidence_scores[f"{zone_key}:{zplan.zone.id}"] = mean_conf
            if store_bubble_states:
                zone_debug = dict(context.digit_zone_debug.get(str(zplan.zone.id), {}))
                zone_debug.update(debug)
                zone_debug.update({
                    "bubble_centers": [(float(x), float(y)) for x, y in mat_pts.reshape(-1, 2).tolist()],
                    "recognized_points": [(float(x), float(y)) for x, y in mat_pts.reshape(-1, 2).tolist()],
                    "scores": mat.tolist(),
                })
                context.digit_zone_debug[str(zplan.zone.id)] = zone_debug
                result.digit_zone_debug = dict(context.digit_zone_debug)
                ztype = self._zone_type_name(zplan.zone)
                filled_thr = self._filled_state_threshold(zplan.zone, ztype)
                context.bubble_states_by_zone[str(zplan.zone.id)] = self._bubble_states_from_scores(scores, threshold=filled_thr)
                result.bubble_states_by_zone = dict(context.bubble_states_by_zone)
        ztype = self._zone_type_name(zplan.zone)
        filled_thr = self._filled_state_threshold(zplan.zone, ztype)
        blank_thr = self._blank_state_threshold(zplan.zone, ztype)
        final_scores = mat.reshape(-1)
        bubble_count = int(final_scores.size)
        recognized = int(np.sum(final_scores >= filled_thr))
        blank = int(np.sum(final_scores <= blank_thr))
        uncertain = int(max(0, bubble_count - recognized - blank))
        return {
            "zone_id": str(zplan.zone.id),
            "zone_type": self._zone_type_name(zplan.zone),
            "bubble_count": bubble_count,
            "recognized_count": recognized,
            "blank_count": blank,
            "uncertain_count": uncertain,
            "fast_path_used": True,
            "fallback_used": bool(fallback_used),
            "weak_count": int(weak_count),
            "decision_count": int(total_checks),
        }

    def _decode_regular_zone(self, dark_integral: np.ndarray, shape: tuple[int, int], zplan: ZonePlan, transformed_centers: np.ndarray, result: OMRResult, context: RecognitionContext, *, store_bubble_states: bool = True) -> dict[str, object]:
        ztype = self._zone_type_name(zplan.zone)
        dx, dy, _ = self._estimate_zone_shift(dark_integral, transformed_centers, zplan.radius, shape, ztype)
        shifted = transformed_centers.astype(np.float32).copy()
        shifted[:, 0] += dx
        shifted[:, 1] += dy
        rows, cols = zplan.rows, zplan.cols
        need = rows * cols
        if ztype == "MCQ_BLOCK":
            scores, best_pts = self._sample_mcq_with_axis_refine(dark_integral, shifted, zplan.radius, shape, rows, cols)
            mat, mat_pts = self._reorder_sampled_matrix(best_pts, scores, rows, cols, zplan.spatial_order_idx)
        else:
            scores = self._sample_ring_score(dark_integral, shifted, zplan.radius, shape)
            mat, mat_pts = self._reorder_sampled_matrix(shifted, scores, rows, cols, zplan.spatial_order_idx)
        need_fallback, weak_count, total_checks = self._zone_result_needs_fallback(mat, zplan.zone, ztype)
        fallback_used = False
        if need_fallback:
            fallback_used = True
            if ztype == "MCQ_BLOCK":
                scores, best_pts = self._sample_mcq_with_axis_refine(dark_integral, shifted, zplan.radius, shape, rows, cols)
            else:
                kind = "numeric" if ztype == "NUMERIC_BLOCK" else "zone"
                scores, best_pts = self._sample_with_local_recenter(dark_integral, shifted, zplan.radius, shape, kind=kind)
            mat, mat_pts = self._reorder_sampled_matrix(best_pts, scores, rows, cols, zplan.spatial_order_idx)
        if ztype == "MCQ_BLOCK":
            self._decode_mcq_zone(mat, zplan.zone, result)
        elif ztype == "TRUE_FALSE_BLOCK":
            self._decode_true_false_zone(mat, zplan.zone, result)
        elif ztype == "NUMERIC_BLOCK":
            self._decode_numeric_zone(mat, zplan.zone, result)
        filled_thr = self._filled_state_threshold(zplan.zone, ztype)
        blank_thr = self._blank_state_threshold(zplan.zone, ztype)
        if store_bubble_states:
            context.bubble_states_by_zone[str(zplan.zone.id)] = self._bubble_states_from_scores(mat.reshape(-1), threshold=filled_thr)
        bubble_count = int(need)
        flat_scores = mat.reshape(-1)
        recognized = int(np.sum(flat_scores >= filled_thr))
        blank = int(np.sum(flat_scores <= blank_thr))
        uncertain = int(max(0, bubble_count - recognized - blank))
        return {
            "zone_id": str(zplan.zone.id),
            "zone_type": ztype,
            "bubble_count": bubble_count,
            "recognized_count": recognized,
            "blank_count": blank,
            "uncertain_count": uncertain,
            "zone_shift": {"dx": float(dx), "dy": float(dy)},
            "sampled_points": [(float(x), float(y)) for x, y in mat_pts.reshape(-1, 2).tolist()] if store_bubble_states else [],
            "fast_path_used": True,
            "fallback_used": bool(fallback_used),
            "weak_count": int(weak_count),
            "decision_count": int(total_checks),
        }

    # ------------------------------------------------------------------
    # Quality / debug helpers
    # ------------------------------------------------------------------
    def _estimate_image_quality(self, gray: np.ndarray, binary: np.ndarray, reg_debug: dict[str, object]) -> dict[str, object]:
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else 0.0
        header = binary[: max(1, int(binary.shape[0] * 0.22)), :]
        header_density = float(np.mean(header > 0)) if header.size else 0.0
        match_count = int(reg_debug.get("patch_valid_count", 0) or 0)
        mean_error = float(reg_debug.get("mean_error", 999.0) or 999.0)
        quality = (
            0.35 * min(1.0, match_count / 6.0)
            + 0.20 * max(0.0, min(1.0, blur_var / 120.0))
            + 0.20 * max(0.0, min(1.0, header_density / 0.18))
            + 0.25 * max(0.0, min(1.0, (8.0 - mean_error) / 8.0))
        )
        reason = "ok"
        if match_count < 3:
            reason = "weak_registration"
        elif mean_error > 8.0:
            reason = "registration_error"
        elif blur_var < 12.0:
            reason = "blur"
        elif header_density < 0.03:
            reason = "weak_header"
        return {
            "quality_score": float(quality),
            "poor_scan": bool(quality < 0.32),
            "poor_identifier_zone": bool(header_density < 0.03),
            "reason": reason,
            "blur_var": float(blur_var),
            "header_density": float(header_density),
            "usable_ratio": 1.0,
            "match_count": int(match_count),
            "mean_error": float(mean_error),
            "align_score": float(max(0.0, 1.0 - mean_error / 10.0)),
        }

    @staticmethod
    def _calc_unaccounted_time(timing: dict[str, float]) -> float:
        total = float(timing.get("total_time_sec", 0.0) or 0.0)
        accounted = (
            float(timing.get("load_time_sec", 0.0) or 0.0)
            + float(timing.get("normalize_time_sec", 0.0) or 0.0)
            + float(timing.get("registration_patch_time_sec", 0.0) or 0.0)
            + float(timing.get("transform_fit_time_sec", 0.0) or 0.0)
            + float(timing.get("roi_decode_time_sec", 0.0) or 0.0)
            + float(timing.get("diagnostics_time_sec", 0.0) or 0.0)
        )
        return float(total - accounted)

    # ------------------------------------------------------------------
    # Public recognition API
    # ------------------------------------------------------------------
    def recognize_sheet(self, image: str | Path | np.ndarray, template: Template, context: RecognitionContext | None = None) -> OMRResult:
        started = time.perf_counter()
        image_path = str(image) if isinstance(image, (str, Path)) else "<in-memory>"
        result = OMRResult(image_path=image_path)
        result.aligned_image = None
        result.aligned_binary = None
        result.alignment_debug = {}
        result.detected_anchors = []
        result.detected_digit_anchors = []
        result.bubble_states_by_zone = {}
        result.digit_zone_debug = {}
        context = context or RecognitionContext()
        context.reset()
        timeout_raw = self._md(template).get("recognition_timeout_sec", None)
        timeout_sec = float(timeout_raw) if isinstance(timeout_raw, (int, float)) else 0.0
        self._processing_deadline_monotonic = time.monotonic() + max(1.0, timeout_sec) if timeout_sec > 0.0 else None

        try:
            need_artifacts = bool(self.debug_mode or context.collect_diagnostics or context.debug_deep or context.force_identifier_recognition or (not context.fast_production_test))
            need_quality = bool(need_artifacts)
            need_identifier = bool(self._identifier_recognition_enabled(template) or context.force_identifier_recognition)

            t0 = time.perf_counter()
            if isinstance(image, np.ndarray):
                src = image.copy()
                dpi_message = ""
            else:
                src, dpi_message = self._load_image_normalized_to_200_dpi(str(image), template)
            load_time_sec = float(time.perf_counter() - t0)
            if src is None:
                result.issues.append(OMRIssue("FILE", "Unable to load image"))
                result.processing_time_sec = float(time.perf_counter() - started)
                result.sync_legacy_aliases()
                return result
            original_shape = tuple(int(v) for v in src.shape)

            t1 = time.perf_counter()
            plan = self._get_template_runtime_plan(template)
            if src.shape[1] != plan.target_width or src.shape[0] != plan.target_height:
                src = cv2.resize(src, (plan.target_width, plan.target_height), interpolation=cv2.INTER_LINEAR)
            normalize_time_sec = float(time.perf_counter() - t1)
            final_shape = tuple(int(v) for v in src.shape)

            prep = self._preprocess_fast(src)
            gray = prep["gray"]
            binary = prep["binary"]
            dark = prep["dark"]

            t2 = time.perf_counter()
            affine_matrix, detected_anchors, reg_debug = self._estimate_affine(binary, plan)
            registration_patch_time_sec = float(time.perf_counter() - t2)
            transform_fit_time_sec = 0.0
            identity_affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            affine_is_identity = bool(np.allclose(affine_matrix, identity_affine))
            if affine_is_identity:
                aligned_dark = dark
                aligned_binary = binary if (need_identifier or need_artifacts) else None
                aligned_gray = gray if need_quality else None
                aligned_image = src if need_artifacts else None
            else:
                aligned_dark = cv2.warpAffine(dark, affine_matrix, (plan.target_width, plan.target_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                aligned_binary = cv2.warpAffine(binary, affine_matrix, (plan.target_width, plan.target_height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) if (need_identifier or need_artifacts) else None
                aligned_gray = cv2.warpAffine(gray, affine_matrix, (plan.target_width, plan.target_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if need_quality else None
                aligned_image = cv2.warpAffine(src, affine_matrix, (plan.target_width, plan.target_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if need_artifacts else None
            alignment_time_sec = float(registration_patch_time_sec + transform_fit_time_sec)

            digit_anchor_arr = np.empty((0, 2), dtype=np.float32)
            digit_anchor_list: list[tuple[float, float]] = []
            if need_identifier and aligned_binary is not None:
                digit_anchor_arr, digit_anchor_list = self._detect_digit_anchor_ruler(aligned_binary, plan, identity_affine)
            integral_dark = self._build_integral(aligned_dark)

            if need_quality and aligned_gray is not None and aligned_binary is not None:
                quality_gate = self._estimate_image_quality(aligned_gray, aligned_binary, reg_debug)
            else:
                quality_gate = {
                    "quality_score": 0.0,
                    "poor_scan": False,
                    "poor_identifier_zone": False,
                    "reason": "skipped_production_fast",
                    "blur_var": 0.0,
                    "header_density": 0.0,
                    "usable_ratio": 1.0,
                    "match_count": int(reg_debug.get("patch_valid_count", 0) or 0),
                    "mean_error": float(reg_debug.get("mean_error", 0.0) or 0.0),
                    "align_score": float(max(0.0, 1.0 - float(reg_debug.get("mean_error", 0.0) or 0.0) / 10.0)),
                }

            roi_decode_started = time.perf_counter()
            zone_metrics: list[dict[str, object]] = []
            header_decode_time_sec = 0.0
            mcq_decode_time_sec = 0.0
            tf_decode_time_sec = 0.0
            numeric_decode_time_sec = 0.0
            fast_zone_decode_time_sec = 0.0
            fallback_zone_decode_time_sec = 0.0
            for zone in plan.sorted_zones:
                zplan = plan.zone_plans.get(str(getattr(zone, "id", "") or ""))
                if zplan is None:
                    continue
                zt = self._zone_type_name(zone)
                if need_artifacts:
                    context.semantic_grids[str(zone.id)] = getattr(zone, "grid", None)
                transformed = zplan.centers_px.astype(np.float32, copy=True)
                z_started = time.perf_counter()
                if zt in {"STUDENT_ID_BLOCK", "EXAM_CODE_BLOCK"}:
                    if not need_identifier:
                        continue
                    metric = self._decode_identifier_zone(
                        integral_dark,
                        aligned_dark.shape[:2],
                        zplan,
                        plan,
                        transformed,
                        digit_anchor_arr,
                        result,
                        context,
                        store_bubble_states=need_artifacts,
                    )
                    header_decode_time_sec += float(time.perf_counter() - z_started)
                else:
                    metric = self._decode_regular_zone(
                        integral_dark,
                        aligned_dark.shape[:2],
                        zplan,
                        transformed,
                        result,
                        context,
                        store_bubble_states=need_artifacts,
                    )
                    elapsed = float(time.perf_counter() - z_started)
                    if zt == "MCQ_BLOCK":
                        mcq_decode_time_sec += elapsed
                    elif zt == "TRUE_FALSE_BLOCK":
                        tf_decode_time_sec += elapsed
                    elif zt == "NUMERIC_BLOCK":
                        numeric_decode_time_sec += elapsed
                if bool(metric.get("fallback_used", False)):
                    fallback_zone_decode_time_sec += float(time.perf_counter() - z_started)
                else:
                    fast_zone_decode_time_sec += float(time.perf_counter() - z_started)
                zone_metrics.append(metric)
            roi_decode_time_sec = float(time.perf_counter() - roi_decode_started)

            diagnostics_time_sec = 0.0
            if need_artifacts:
                t_diag = time.perf_counter()
                context.detected_anchors = list(detected_anchors)
                context.detected_digit_anchors = list(digit_anchor_list)
                diagnostics_time_sec = float(time.perf_counter() - t_diag)
            result.detected_anchors = list(detected_anchors) if need_artifacts else []
            result.detected_digit_anchors = list(digit_anchor_list) if need_artifacts else []

            if need_artifacts:
                result.aligned_image = aligned_image
                result.aligned_binary = aligned_binary
                result.bubble_states_by_zone = dict(context.bubble_states_by_zone)
                result.digit_zone_debug = dict(context.digit_zone_debug)

            context.recognized_answers = {
                "student_id": result.student_id,
                "exam_code": result.exam_code,
                "mcq_answers": dict(result.mcq_answers),
                "true_false_answers": dict(result.true_false_answers),
                "numeric_answers": dict(result.numeric_answers),
            }

            result.processing_time_sec = float(time.perf_counter() - started)
            timing = {
                "load_time_sec": float(load_time_sec),
                "normalize_time_sec": float(normalize_time_sec),
                "registration_patch_time_sec": float(registration_patch_time_sec),
                "transform_fit_time_sec": float(transform_fit_time_sec),
                "alignment_time_sec": float(alignment_time_sec),
                "anchor_detect_time_sec": 0.0,
                "header_decode_time_sec": float(header_decode_time_sec),
                "identifier_time_sec": float(header_decode_time_sec),
                "identifier_fallback_time_sec": 0.0,
                "mcq_decode_time_sec": float(mcq_decode_time_sec),
                "tf_decode_time_sec": float(tf_decode_time_sec),
                "numeric_decode_time_sec": float(numeric_decode_time_sec),
                "roi_decode_time_sec": float(roi_decode_time_sec),
                "scanner_locked_match_time_sec": float(registration_patch_time_sec),
                "scanner_locked_header_time_sec": float(header_decode_time_sec),
                "scanner_locked_mcq_time_sec": float(mcq_decode_time_sec),
                "scanner_locked_tf_time_sec": float(tf_decode_time_sec),
                "scanner_locked_numeric_time_sec": float(numeric_decode_time_sec),
                "mcq_time_sec": float(mcq_decode_time_sec),
                "tf_time_sec": float(tf_decode_time_sec),
                "numeric_time_sec": float(numeric_decode_time_sec),
                "fast_zone_decode_time_sec": float(fast_zone_decode_time_sec),
                "fallback_zone_decode_time_sec": float(fallback_zone_decode_time_sec),
                "artifact_time_sec": float(diagnostics_time_sec),
                "diagnostics_time_sec": float(diagnostics_time_sec),
                "poor_image_fast_fail": bool(quality_gate.get("poor_scan", False)),
                "total_time_sec": float(result.processing_time_sec),
            }
            timing["unaccounted_time_sec"] = self._calc_unaccounted_time(timing)

            result.alignment_debug = {
                **reg_debug,
                "alignment_mode": "scanner_locked",
                "recognition_mode": "scanner_locked",
                "band_locking": "hybrid_scanner_locked",
                "path_used": "scanner_locked_fast_selective_fallback",
                "registration_ok": bool(reg_debug.get("registration_ok", False)),
                "fallback_used": bool(any(bool(m.get("fallback_used", False)) for m in zone_metrics)),
                "fallback_reason": "selective_zone_retry" if any(bool(m.get("fallback_used", False)) for m in zone_metrics) else "",
                "quality_gate": dict(quality_gate),
                "quality_score": float(quality_gate.get("quality_score", 0.0) or 0.0),
                "poor_image": bool(quality_gate.get("poor_scan", False)),
                "poor_identifier_zone": bool(quality_gate.get("poor_identifier_zone", False)),
                "quality_reason": str(quality_gate.get("reason", "") or ""),
                "fast_production_mode": bool(context.fast_production_test and not context.debug_deep),
                "diagnostics_collected": bool(need_artifacts),
                "original_shape": original_shape,
                "final_shape": final_shape,
                "dpi_message": dpi_message,
                "zone_metrics": zone_metrics,
                "timing_breakdown": timing,
            }
            self._last_alignment_debug = dict(result.alignment_debug)
            setattr(result, "answers", result.mcq_answers)
            result.sync_legacy_aliases()
            return result
        finally:
            self._processing_deadline_monotonic = None

    def recognize_sheet_production_fast(self, image: str | Path | np.ndarray, template: Template, context: RecognitionContext | None = None) -> OMRResult:
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
        ctx.collect_diagnostics = True
        return self.recognize_sheet(image, template, ctx)

    def extract_bubble_states(self, binary_image: np.ndarray, template: Template) -> dict[str, list[bool]]:
        # lightweight helper for UI overlay; use plan centers directly.
        plan = self._get_template_runtime_plan(template)
        prep = self._preprocess_fast(binary_image if binary_image.ndim == 3 else cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR))
        dark = prep["dark"]
        integral_dark = self._build_integral(dark)
        out: dict[str, list[bool]] = {}
        for zplan in plan.zone_plans.values():
            ztype = self._zone_type_name(zplan.zone)
            kind = "header" if ztype in {"STUDENT_ID_BLOCK", "EXAM_CODE_BLOCK"} else ("numeric" if ztype == "NUMERIC_BLOCK" else "zone")
            scores, _ = self._sample_with_local_recenter(integral_dark, zplan.centers_px, zplan.radius, dark.shape[:2], kind=kind)
            out[str(zplan.zone.id)] = self._bubble_states_from_scores(scores, threshold=self._filled_state_threshold(zplan.zone, ztype))
        return out

    def process_image(self, image_path: str | Path, template: Template) -> OMRResult:
        return self.recognize_sheet(image_path, template, RecognitionContext())

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def _make_batch_worker(self) -> "OMRProcessor":
        worker = OMRProcessor(
            fill_threshold=self.fill_threshold,
            empty_threshold=self.empty_threshold,
            certainty_margin=self.certainty_margin,
            debug_mode=self.debug_mode,
            debug_dir=self.debug_dir,
        )
        worker.processing_time_limit_sec = self.processing_time_limit_sec
        worker._template_plan_cache = dict(self._template_plan_cache)
        return worker

    def _get_thread_batch_worker(self) -> "OMRProcessor":
        worker = getattr(self, "_batch_worker", None)
        if worker is None:
            worker = self._make_batch_worker()
            self._batch_worker = worker
        return worker

    @staticmethod
    def _auto_batch_worker_count(total: int, template: Template) -> int:
        meta = getattr(template, "metadata", {}) or {}
        auto_parallel = bool(meta.get("batch_auto_parallel", True))
        if not auto_parallel:
            return 1
        min_items = int(meta.get("batch_auto_parallel_min_items", 20) or 20)
        if total < max(2, min_items):
            return 1
        cpu_count = max(1, int(os.cpu_count() or 1))
        max_auto = int(meta.get("batch_auto_parallel_max_workers", min(8, cpu_count)) or min(8, cpu_count))
        return max(1, min(total, max_auto))

    def _run_batch_item(self, image_path: str, template: Template) -> tuple[OMRResult, float]:
        worker = self._get_thread_batch_worker()
        started = time.perf_counter()
        res = worker.recognize_sheet_production_fast(image_path, template, RecognitionContext(collect_diagnostics=False))
        return res, float(time.perf_counter() - started)

    @staticmethod
    def _write_batch_timing_log(log_path: Path, rows: list[tuple[int, str, float]], total_count: int) -> None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lines = ["idx\tfile\tseconds"]
            for idx, image_path, sec in rows:
                lines.append(f"{idx}\t{image_path}\t{sec:.4f}")
            valid = [sec for _, _, sec in rows if sec >= 0.0]
            if valid:
                avg = float(sum(valid) / len(valid))
                lines.extend([
                    "",
                    f"avg_seconds_per_file\t{avg:.4f}",
                    f"estimated_500_files_seconds\t{avg * 500.0:.2f}",
                    f"estimated_600_files_seconds\t{avg * 600.0:.2f}",
                ])
            lines.append(f"total_files\t{total_count}")
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def process_batch(self, image_paths: list[str], template: Template, progress_callback: Callable[[int, int, str], None] | None = None) -> list[OMRResult]:
        total = len(image_paths)
        if total == 0:
            return []
        self._get_template_runtime_plan(template)
        worker_count = self._auto_batch_worker_count(total, template)
        log_path_text = str((getattr(template, "metadata", {}) or {}).get("batch_timing_log_path", "") or "")
        log_enabled = bool(log_path_text.strip())
        timing_rows: list[tuple[int, str, float]] = []

        if worker_count <= 1:
            results: list[OMRResult] = []
            for idx, image_path in enumerate(image_paths, start=1):
                started = time.perf_counter()
                res = self.recognize_sheet_production_fast(image_path, template, RecognitionContext(collect_diagnostics=False))
                elapsed = float(time.perf_counter() - started)
                results.append(res)
                timing_rows.append((idx, str(image_path), elapsed))
                if progress_callback:
                    progress_callback(idx, total, image_path)
            if log_enabled:
                self._write_batch_timing_log(Path(log_path_text), timing_rows, total)
            return results

        ordered_results: list[OMRResult | None] = [None] * total
        ordered_secs: list[float] = [0.0] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_map = {pool.submit(self._run_batch_item, image_path, template): (idx, image_path) for idx, image_path in enumerate(image_paths)}
            for fut in as_completed(future_map):
                idx, image_path = future_map[fut]
                res, elapsed = fut.result()
                ordered_results[idx] = res
                ordered_secs[idx] = float(elapsed)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, image_path)
        for idx, image_path in enumerate(image_paths, start=1):
            timing_rows.append((idx, str(image_path), float(ordered_secs[idx - 1])))
        if log_enabled:
            self._write_batch_timing_log(Path(log_path_text), timing_rows, total)
        return [r for r in ordered_results if r is not None]


__all__ = [
    "OMRIssue",
    "OMRResult",
    "RecognitionContext",
    "TemplateRuntimePlan",
    "OMRProcessor",
]
