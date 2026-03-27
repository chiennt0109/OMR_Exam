from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
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
        timeout_sec = float((template.metadata or {}).get("recognition_timeout_sec", self.processing_time_limit_sec) or self.processing_time_limit_sec)
        context.deadline_monotonic = time.monotonic() + max(1.0, timeout_sec)
        self._processing_deadline_monotonic = context.deadline_monotonic

        try:
            if isinstance(image, np.ndarray):
                src = image.copy()
            else:
                norm_path, dpi_msg = self._normalize_to_200_dpi(str(image))
                if dpi_msg:
                    result.issues.append(OMRIssue("DPI", dpi_msg))
                src = cv2.imread(norm_path)
                if src is None:
                    result.issues.append(OMRIssue("FILE", "Unable to load image"))
                    result.sync_legacy_aliases()
                    return result

            rotated = self._correct_rotation(src)
            prepared = self._preprocess(rotated)

            aligned, aligned_binary = self.correct_perspective(rotated, prepared["binary"], template, result)
            aligned_pre = self._preprocess(aligned)
            if aligned_binary is None:
                aligned_binary = aligned_pre["binary"]

            aligned_h, aligned_w = aligned_binary.shape[:2]
            template.width = int(aligned_w)
            template.height = int(aligned_h)

            debug_overlay = aligned.copy() if self.debug_mode else None
            if debug_overlay is not None:
                self._draw_alignment_debug(debug_overlay, template)

            for zone in sorted(template.zones, key=self._zone_recognition_priority):
                if self._time_budget_exceeded():
                    result.issues.append(OMRIssue("TIMEOUT", "Recognition time budget exceeded; skipped remaining zones"))
                    break
                if zone.zone_type == ZoneType.ANCHOR:
                    continue
                if zone.grid is not None:
                    context.semantic_grids[zone.id] = zone.grid
                self.recognize_block(aligned_binary, zone, template, result, debug_overlay)

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
                context.detected_anchors = self.detect_anchors(aligned_binary, max_points=120)
                setattr(result, "detected_anchors", context.detected_anchors)
                context.detected_digit_anchors = self._detect_digit_anchor_ruler(aligned_binary, template)
                setattr(result, "detected_digit_anchors", context.detected_digit_anchors)

                context.bubble_states_by_zone = self.extract_bubble_states(aligned_binary, template)
                setattr(result, "bubble_states_by_zone", context.bubble_states_by_zone)
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
            setattr(result, "answers", result.mcq_answers)
            result.sync_legacy_aliases()
            return result
        finally:
            self.alignment_profile = prev_profile
            self._processing_deadline_monotonic = None

    def run_recognition_test(self, image: str | Path | np.ndarray, template: Template, context: RecognitionContext | None = None) -> OMRResult:
        return self.recognize_sheet(image, template, context)

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

    def process_batch(self, image_paths: list[str], template: Template, progress_callback: Callable[[int, int, str], None] | None = None) -> list[OMRResult]:
        total = len(image_paths)
        results: list[OMRResult] = []
        for idx, image_path in enumerate(image_paths, start=1):
            template_copy = deepcopy(template)
            context = RecognitionContext()
            context.collect_diagnostics = False
            results.append(self.run_recognition_test(image_path, template_copy, context))
            if progress_callback:
                progress_callback(idx, total, image_path)
        return results

    def _normalize_to_200_dpi(self, path: str) -> tuple[str, str]:
        dpi = 0
        try:
            with Image.open(path) as im_meta:
                dpi_info = im_meta.info.get("dpi")
                if isinstance(dpi_info, tuple) and dpi_info and dpi_info[0]:
                    dpi = int(round(float(dpi_info[0])))
        except Exception:
            dpi = 0
        if dpi == 200:
            return path, ""
        if dpi <= 0:
            return path, "Image DPI metadata missing; expected 200 DPI."
        img = cv2.imread(path)
        if img is None:
            return path, ""
        scale = 200.0 / dpi
        out = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
        out_path = str(Path(path).with_name(f"{Path(path).stem}_200dpi.png"))
        cv2.imwrite(out_path, out)
        return out_path, f"Input DPI={dpi}. Normalized to 200 DPI scale."

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
        angle = self._estimate_rotation_angle(image)
        if abs(angle) < 0.15:
            return image
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _estimate_rotation_angle(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=180)
        if lines is not None and len(lines) > 0:
            angles = []
            for ln in lines[:25]:
                theta = ln[0][1]
                angle = (theta * 180.0 / np.pi) - 90.0
                if -45.0 <= angle <= 45.0:
                    angles.append(angle)
            if angles:
                return float(np.median(angles))

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

    def detect_anchors(self, binary: np.ndarray, use_border_padding: bool = True, relaxed_polygon: bool = True, max_points: int = 40) -> list[tuple[float, float]]:
        pad = 8 if use_border_padding else 0
        padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) if pad else binary
        contours, _ = cv2.findContours(padded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = binary.shape[:2]
        min_area = max(50, int((h * w) * 0.00003))
        border_margin = max(24.0, min(w, h) * 0.18)
        anchors: list[tuple[float, float, float]] = []
        max_keep = max(4, int(max_points or 40))
        max_candidates = max(200, max_keep * (18 if use_border_padding else 12))
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
                border_bonus = 1.0 + max(0.0, (border_margin - near_border) / max(1.0, border_margin))
            else:
                border_bonus = 1.0
            score = area * fill_ratio * darkness * border_bonus
            anchors.append((cx, cy, score))

        anchors.sort(key=lambda p: p[2], reverse=True)
        if use_border_padding:
            max_keep = min(max_keep, 24)
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

    def _detect_anchors_by_profile(self, binary: np.ndarray, profile: str) -> list[tuple[float, float]]:
        p = str(profile or "auto").strip().lower()
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

    def _try_anchor_alignment(self, image: np.ndarray, binary: np.ndarray, template: Template, profile: str) -> tuple[np.ndarray, np.ndarray] | None:
        detected = self._detect_anchors_by_profile(binary, profile)
        template_pts = np.array([(a.x * template.width, a.y * template.height) if a.x <= 1.0 and a.y <= 1.0 else (a.x, a.y) for a in template.anchors], dtype=np.float32)
        if len(detected) < 4 or len(template_pts) < 4:
            return None
        src_pts, dst_pts = self._match_anchor_sets(np.array(detected, dtype=np.float32), template_pts)
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
        mode = str(getattr(self, "alignment_profile", "auto") or "auto").strip().lower()
        candidates: list[str] = []
        if mode == "auto":
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
        best_attempt: tuple[float, str, np.ndarray, np.ndarray] | None = None

        for candidate in candidates:
            if self._time_budget_exceeded():
                break
            if candidate == "one_side":
                coarse_img, coarse_bin = self._fallback_align_page_contour(image, template, conservative=True)
                refined_img, refined_bin = self._refine_alignment_with_one_side_anchors(coarse_img, coarse_bin, template)
            else:
                attempt = self._try_anchor_alignment(base_image, base_binary, template, candidate)
                if attempt is None:
                    continue
                coarse_img, coarse_bin = attempt
                refined_img, refined_bin = (coarse_img, coarse_bin) if candidate == "legacy" else self._refine_alignment_with_template_anchors(coarse_img, coarse_bin, template)
            oriented_img, oriented_bin = self._auto_orient(refined_img, refined_bin, template)
            shifted_img, shifted_bin = self._refine_corner_translation(oriented_img, oriented_bin, template)
            affine_img, affine_bin = self._refine_alignment_with_affine_anchors(shifted_img, shifted_bin, template)
            candidate_score = self._orientation_score(affine_bin, template)
            if best_attempt is None or candidate_score > best_attempt[0]:
                best_attempt = (candidate_score, candidate, affine_img, affine_bin)

        if best_attempt is not None:
            score, candidate, best_img, best_bin = best_attempt
            self._last_alignment_debug["alignment_mode"] = candidate
            self._last_alignment_debug["alignment_score"] = float(score)
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
        candidates = []
        for rotation in (0, 90, 180, 270):
            if rotation == 0:
                img_r = aligned
                bin_r = aligned_binary
            elif rotation == 90:
                img_r = cv2.rotate(aligned, cv2.ROTATE_90_CLOCKWISE)
                bin_r = cv2.rotate(aligned_binary, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                img_r = cv2.rotate(aligned, cv2.ROTATE_180)
                bin_r = cv2.rotate(aligned_binary, cv2.ROTATE_180)
            else:
                img_r = cv2.rotate(aligned, cv2.ROTATE_90_COUNTERCLOCKWISE)
                bin_r = cv2.rotate(aligned_binary, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if img_r.shape[1] != template.width or img_r.shape[0] != template.height:
                img_r = cv2.resize(img_r, (template.width, template.height), interpolation=cv2.INTER_LINEAR)
                bin_r = cv2.resize(bin_r, (template.width, template.height), interpolation=cv2.INTER_NEAREST)
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
        anchors = np.array(self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=80), dtype=np.float32)
        if len(anchors) >= 4 and len(template.anchors) >= 4:
            tpl = np.array([(a.x * template.width, a.y * template.height) if a.x <= 1.0 and a.y <= 1.0 else (a.x, a.y) for a in template.anchors], dtype=np.float32)
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
        template_pts = np.array([(a.x * template.width, a.y * template.height) if a.x <= 1.0 and a.y <= 1.0 else (a.x, a.y) for a in template.anchors], dtype=np.float32)
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
        template_pts = np.array([(a.x * template.width, a.y * template.height) if a.x <= 1.0 and a.y <= 1.0 else (a.x, a.y) for a in template.anchors], dtype=np.float32)
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

        template_pts = np.array([(a.x * template.width, a.y * template.height) if a.x <= 1.0 and a.y <= 1.0 else (a.x, a.y) for a in template.anchors], dtype=np.float32)
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
        centers_int = centers.astype(np.int32)

        for i, (x, y) in enumerate(centers_int):
            x0, y0 = x - r, y - r
            x1, y1 = x + r + 1, y + r + 1
            if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
                continue
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(w, x1), min(h, y1)
            roi = binary[y0c:y1c, x0c:x1c]
            local_mask = mask[(y0c - y0):(y1c - y0), (x0c - x0):(x1c - x0)]
            den = float(np.count_nonzero(local_mask))
            if roi.size == 0 or den == 0:
                continue
            dark_pixels = cv2.countNonZero(roi[local_mask])
            fill_ratio = float(dark_pixels) / den
            mean_density = float(np.count_nonzero(roi)) / float(roi.size)
            ratios[i] = float(np.clip(0.8 * fill_ratio + 0.2 * mean_density, 0.0, 1.0))
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
        expected = np.array([(x * w, y * h) if x <= 1.0 and y <= 1.0 else (x, y) for x, y in grid.bubble_positions], dtype=np.float32)
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
        multi_probe_ratios = self._detect_digit_zone_multi_probe_marks(binary, refined_centers, radius)
        peak_window_ratios = np.zeros_like(ratios)
        component_ratios = self._detect_digit_zone_component_marks(binary, refined_centers, radius)
        if zone.zone_type == ZoneType.STUDENT_ID_BLOCK:
            square_ratios = self._detect_square_mark_density(binary, refined_centers, radius)
            eroded_ratios = self._detect_eroded_mark_density(binary, refined_centers, radius)
            legacy_ratios = np.clip((0.12 * ratios) + (0.24 * core_ratios) + (0.24 * square_ratios) + (0.20 * eroded_ratios) + (0.20 * component_ratios), 0.0, 1.0)
            widened_ratios = np.clip((0.06 * ratios) + (0.14 * core_ratios) + (0.16 * square_ratios) + (0.14 * eroded_ratios) + (0.18 * multi_probe_ratios) + (0.14 * peak_window_ratios) + (0.18 * component_ratios), 0.0, 1.0)
            ratios = np.maximum(legacy_ratios, widened_ratios)
        else:
            legacy_ratios = np.clip((0.42 * ratios) + (0.28 * core_ratios) + (0.30 * component_ratios), 0.0, 1.0)
            widened_ratios = np.clip((0.18 * ratios) + (0.16 * core_ratios) + (0.22 * multi_probe_ratios) + (0.18 * peak_window_ratios) + (0.26 * component_ratios), 0.0, 1.0)
            ratios = np.maximum(legacy_ratios, widened_ratios)
        weak_mask = ratios < max(self.empty_threshold + 0.16, min(self.fill_threshold * 0.82, 0.44))
        if np.any(weak_mask):
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
            if (not is_valid or global_score < 0.85) and result is not None:
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
        filled = np.where(scores > row_threshold)[0]
        margin = top - second
        if len(filled) > 1:
            top_support = top
            second_support = second
            if row_raw_scores is not None and row_core_scores is not None:
                raw_scores = np.asarray(row_raw_scores, dtype=np.float32)
                core_scores = np.asarray(row_core_scores, dtype=np.float32)
                eroded_scores = np.asarray(row_eroded_scores if row_eroded_scores is not None else raw_scores, dtype=np.float32)
                top_support = max(float(raw_scores[top_i]), float(core_scores[top_i]), float(eroded_scores[top_i]))
                second_support = max(float(raw_scores[second_i]), float(core_scores[second_i]), float(eroded_scores[second_i]))
            support_similarity = second_support / max(top_support, 1e-6)
            if top >= row_threshold and support_similarity < 0.95:
                return top_i, max(margin, top_support - second_support), "row_max_fallback"
            return None, 0.0, "multiple"
        if top > row_threshold and margin > self.certainty_margin:
            return top_i, margin, None
        relaxed_threshold = max(self.empty_threshold + 0.10, row_threshold - 0.10)
        strong_margin = max(self.certainty_margin * 0.75, 0.05)
        strong_ratio = second <= 1e-6 or top >= (1.20 * second)
        if len(filled) <= 1 and top >= relaxed_threshold and margin >= strong_margin and strong_ratio:
            return top_i, margin, "best_fallback"
        return None, 0.0, "uncertain"

    def recognize_block(self, binary: np.ndarray, zone: Zone, template: Template, result: OMRResult, debug_overlay: np.ndarray | None = None) -> None:
        grid = zone.grid
        if not grid or not grid.bubble_positions:
            return

        working_binary = binary
        working_image = getattr(result, "aligned_image", None)
        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            expected = np.array(
                [(x * working_binary.shape[1], y * working_binary.shape[0]) if x <= 1.0 and y <= 1.0 else (x, y) for x, y in grid.bubble_positions],
                dtype=np.float32,
            )
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
            direct_value, direct_confs, direct_centers, direct_mat, direct_debug = self._decode_identifier_zone_from_centers(
                working_binary,
                zone,
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
            direct_conf_mean = float(np.mean(final_confs or [0.0]))
            fallback_budget_ok = not self._time_budget_exceeded()
            allow_fallbacks = fallback_budget_ok and not (bool(final_value) and direct_conf_mean >= 0.55)
            axis_final = ""
            axis_debug: dict[str, object] = {}
            axis_needed = allow_fallbacks and ((not final_value) or (direct_conf_mean < 0.32))
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
                if axis_final and (not final_value or float(np.mean(axis_final_confs or [0.0])) >= float(np.mean(final_confs or [0.0]))):
                    del result.recognition_errors[error_mark:]
                    final_value, final_confs = axis_final, axis_final_confs
            if (
                allow_fallbacks
                and
                key == "student_id"
                and not final_value
                and sid_base_guided is not None
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
            }
            skip_sampling_fallback = key == "exam_code" and exam_digit_model_applied
            if allow_fallbacks and not final_value and not skip_sampling_fallback:
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
                elif final_value == "":
                    final_value = "-"
            elif not final_value and skip_sampling_fallback:
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
            result.confidence_scores[f"{key}:{zone.id}"] = float(np.mean(final_confs)) if final_confs else 0.0
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
            multi_probe_ratios = self._detect_digit_zone_multi_probe_marks(working_binary, centers, radius)
            if zone.zone_type == ZoneType.STUDENT_ID_BLOCK:
                square_ratios = self._detect_square_mark_density(working_binary, centers, radius)
                eroded_ratios = self._detect_eroded_mark_density(working_binary, centers, radius)
                legacy_ratios = np.clip((0.15 * ratios) + (0.30 * core_ratios) + (0.30 * square_ratios) + (0.25 * eroded_ratios), 0.0, 1.0)
                widened_ratios = np.clip((0.10 * ratios) + (0.22 * core_ratios) + (0.22 * square_ratios) + (0.18 * eroded_ratios) + (0.28 * multi_probe_ratios), 0.0, 1.0)
                ratios = np.maximum(legacy_ratios, widened_ratios)
            else:
                legacy_ratios = np.clip((0.55 * ratios) + (0.45 * core_ratios), 0.0, 1.0)
                widened_ratios = np.clip((0.30 * ratios) + (0.28 * core_ratios) + (0.42 * multi_probe_ratios), 0.0, 1.0)
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
        for zone in template.zones:
            x = int(zone.x * template.width)
            y = int(zone.y * template.height)
            ww = int(zone.width * template.width)
            hh = int(zone.height * template.height)
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
