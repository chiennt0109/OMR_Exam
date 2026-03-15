from __future__ import annotations

from dataclasses import dataclass, field
import math
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
        # alignment_profile: auto | legacy | border | hybrid | one_side
        self.alignment_profile: str = "auto"

    def process_image(self, image_path: str | Path, template: Template) -> OMRResult:
        started = time.perf_counter()
        result = OMRResult(image_path=str(image_path))
        template_profile = str((template.metadata or {}).get("alignment_profile", "") or "").strip().lower()
        prev_profile = self.alignment_profile
        if template_profile in {"auto", "legacy", "border", "hybrid", "one_side"}:
            self.alignment_profile = template_profile
        norm_path, dpi_msg = self._normalize_to_200_dpi(str(image_path))
        if dpi_msg:
            result.issues.append(OMRIssue("DPI", dpi_msg))

        src = cv2.imread(norm_path)
        if src is None:
            result.issues.append(OMRIssue("FILE", "Unable to load image"))
            self.alignment_profile = prev_profile
            result.sync_legacy_aliases()
            return result

        rotated = self._correct_rotation(src)
        prepared = self._preprocess(rotated)

        aligned, aligned_binary = self.correct_perspective(rotated, prepared["binary"], template, result)
        self.alignment_profile = prev_profile
        aligned_pre = self._preprocess(aligned)
        if aligned_binary is None:
            aligned_binary = aligned_pre["binary"]

        debug_overlay = aligned.copy() if self.debug_mode else None

        for zone in template.zones:
            if zone.zone_type == ZoneType.ANCHOR:
                continue
            self.recognize_block(aligned_binary, zone, template, result, debug_overlay)

        if debug_overlay is not None:
            out_dir = self.debug_dir or Path(result.image_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(result.image_path).stem}_debug.png"
            cv2.imwrite(str(out_path), debug_overlay)
            result.debug_image_path = str(out_path)

        result.processing_time_sec = time.perf_counter() - started
        setattr(result, "answers", result.mcq_answers)
        result.sync_legacy_aliases()
        return result

    def process_batch(
        self,
        image_paths: list[str],
        template: Template,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[OMRResult]:
        total = len(image_paths)
        results: list[OMRResult] = []
        for idx, image_path in enumerate(image_paths, start=1):
            results.append(self.process_image(image_path, template))
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

    def _preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            5,
        )
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return {"gray": gray, "blur": blur, "binary": binary}

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
        # Pad image before contour extraction so anchors touching page border are not clipped.
        pad = 8 if use_border_padding else 0
        padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) if pad else binary
        contours, _ = cv2.findContours(padded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = binary.shape[:2]
        min_area = max(50, int((h * w) * 0.00003))
        anchors: list[tuple[float, float, float]] = []

        for cnt in contours:
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
            score = area * fill_ratio * darkness
            anchors.append((cx, cy, score))

        anchors.sort(key=lambda p: p[2], reverse=True)
        max_keep = max(4, int(max_points or 40))
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
        pts = np.array(
            [
                [a.x * template.width, a.y * template.height] if a.x <= 1.0 and a.y <= 1.0 else [a.x, a.y]
                for a in anchors
            ],
            dtype=np.float32,
        )
        x_span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        y_span = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        if y_span <= 1.0:
            return False
        # Vertical ruler: x spread small compared to y spread.
        if x_span / y_span > 0.20:
            return False
        mean_x = float(np.mean(pts[:, 0]))
        # Prefer truly side anchors.
        return mean_x <= template.width * 0.18 or mean_x >= template.width * 0.82

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
        # auto fallback defaults to border-safe detector.
        return self.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=60)

    def _try_anchor_alignment(self, image: np.ndarray, binary: np.ndarray, template: Template, profile: str) -> tuple[np.ndarray, np.ndarray] | None:
        detected = self._detect_anchors_by_profile(binary, profile)
        template_pts = np.array(
            [
                (a.x * template.width, a.y * template.height)
                if a.x <= 1.0 and a.y <= 1.0
                else (a.x, a.y)
                for a in template.anchors
            ],
            dtype=np.float32,
        )
        if len(detected) < 4 or len(template_pts) < 4:
            return None
        src_pts, dst_pts = self._match_anchor_sets(np.array(detected, dtype=np.float32), template_pts)
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None
        h = cv2.getPerspectiveTransform(self._order_points(src_pts[:4]), self._order_points(dst_pts[:4]))
        aligned = cv2.warpPerspective(image, h, (template.width, template.height))
        aligned_binary = cv2.warpPerspective(binary, h, (template.width, template.height))
        return aligned, aligned_binary

    def _refine_alignment_with_one_side_anchors(
        self,
        aligned: np.ndarray,
        aligned_binary: np.ndarray,
        template: Template,
    ) -> tuple[np.ndarray, np.ndarray]:
        anchors = list(template.anchors or [])
        if len(anchors) < 4:
            return aligned, aligned_binary

        tpl = np.array(
            [
                [a.x * template.width, a.y * template.height] if a.x <= 1.0 and a.y <= 1.0 else [a.x, a.y]
                for a in anchors
            ],
            dtype=np.float32,
        )
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

        # Top-down row-wise pairing by y-order.
        det_side = det_side[np.argsort(det_side[:, 1])]
        tpl_side = tpl_side[np.argsort(tpl_side[:, 1])]

        # Trim outliers on both ends if very dense.
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

        # Robust affine from row markers (used only to estimate stable shift;
        # keep rendering geometry unchanged to avoid stretch artifacts).
        m, inliers = cv2.estimateAffinePartial2D(det_side, tpl_side, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if m is None:
            return aligned, aligned_binary
        if inliers is not None and int(inliers.sum()) < max(4, int(0.55 * n)):
            return aligned, aligned_binary

        sx = float(np.hypot(m[0, 0], m[1, 0]))
        sy = float(np.hypot(m[0, 1], m[1, 1]))
        # Reject any noticeable scaling to preserve original sheet geometry.
        if sx < 0.985 or sx > 1.015 or sy < 0.985 or sy > 1.015:
            return aligned, aligned_binary
        if abs(float(m[0, 2])) > template.width * 0.10 or abs(float(m[1, 2])) > template.height * 0.12:
            return aligned, aligned_binary

        # Use translation-only correction from matched anchors to avoid stretch.
        dx = float(np.median(tpl_side[:, 0] - det_side[:, 0]))
        dy = float(np.median(tpl_side[:, 1] - det_side[:, 1]))
        m_shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

        # Improvement gate on y-fit.
        probe = cv2.transform(det_side.reshape(1, -1, 2), m_shift).reshape(-1, 2)
        base_err = float(np.mean(np.abs(det_side[:, 1] - tpl_side[:, 1])))
        new_err = float(np.mean(np.abs(probe[:, 1] - tpl_side[:, 1])))
        if new_err > base_err - 0.2:
            return aligned, aligned_binary

        refined = cv2.warpAffine(
            aligned,
            m_shift,
            (template.width, template.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        refined_bin = cv2.warpAffine(
            aligned_binary,
            m_shift,
            (template.width, template.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return refined, refined_bin

    def correct_perspective(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        template: Template,
        result: OMRResult,
    ) -> tuple[np.ndarray, np.ndarray | None]:
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

        for candidate in candidates:
            if candidate == "one_side":
                # One-side ruler templates are sensitive to page warping artifacts.
                # Use a conservative coarse alignment that falls back to plain resize
                # when contour homography looks non-physical.
                coarse_img, coarse_bin = self._fallback_align_page_contour(image, template, conservative=True)
                refined_img, refined_bin = self._refine_alignment_with_one_side_anchors(coarse_img, coarse_bin, template)
                return refined_img, refined_bin

            attempt = self._try_anchor_alignment(image, binary, template, candidate)
            if attempt is None:
                continue
            aligned, aligned_binary = attempt
            if candidate == "legacy":
                return aligned, aligned_binary
            refined_img, refined_bin = self._refine_alignment_with_template_anchors(aligned, aligned_binary, template)
            return refined_img, refined_bin

        result.issues.append(OMRIssue("MISSING_ANCHORS", "Anchor detection failed; using page contour fallback"))
        fallback_img, fallback_bin = self._fallback_align_page_contour(image, template)
        refined_img, refined_bin = self._refine_alignment_with_template_anchors(fallback_img, fallback_bin, template)
        return refined_img, refined_bin

    def _refine_alignment_with_template_anchors(
        self,
        aligned: np.ndarray,
        aligned_binary: np.ndarray,
        template: Template,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(template.anchors) < 4:
            return aligned, aligned_binary
        detected = self.detect_anchors(aligned_binary)
        if len(detected) < 4:
            return aligned, aligned_binary

        template_pts = np.array(
            [
                (a.x * template.width, a.y * template.height)
                if a.x <= 1.0 and a.y <= 1.0
                else (a.x, a.y)
                for a in template.anchors
            ],
            dtype=np.float32,
        )
        detected_arr = np.array(detected, dtype=np.float32)
        if len(template_pts) < 4:
            return aligned, aligned_binary

        # Match by nearest position in template coordinate space after coarse alignment.
        # Keep this gate strict to avoid over-correcting already-good legacy templates.
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

        # Guard rails: refinement should be a small correction in template space.
        h = h.astype(np.float64)
        if abs(float(h[2, 0])) > 2e-4 or abs(float(h[2, 1])) > 2e-4:
            return aligned, aligned_binary

        # Reprojection quality check on matched points.
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
        if not np.isfinite(rmse) or rmse > 3.5:
            return aligned, aligned_binary

        # Scale/shift sanity to prevent large accidental warps on legacy templates.
        corners = np.array(
            [[[0.0, 0.0]], [[template.width - 1.0, 0.0]], [[template.width - 1.0, template.height - 1.0]], [[0.0, template.height - 1.0]]],
            dtype=np.float32,
        )
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

        # Final acceptance: refined result must not degrade anchor fit.
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
            # Require meaningful improvement; otherwise keep coarse alignment.
            if not np.isfinite(refined_err) or refined_err > (base_err - 0.5):
                return aligned, aligned_binary

        return refined, refined_bin

    def _fallback_align_page_contour(self, image: np.ndarray, template: Template, conservative: bool = False) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            resized = cv2.resize(image, (template.width, template.height))
            return resized, self._preprocess(resized)["binary"]
        page = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(page, True)
        approx = cv2.approxPolyDP(page, 0.02 * peri, True)
        if len(approx) != 4:
            resized = cv2.resize(image, (template.width, template.height))
            return resized, self._preprocess(resized)["binary"]
        src = self._order_points(approx.reshape(4, 2).astype(np.float32))
        dst = np.array(
            [[0, 0], [template.width - 1, 0], [template.width - 1, template.height - 1], [0, template.height - 1]],
            dtype=np.float32,
        )
        h = cv2.getPerspectiveTransform(src, dst)
        if conservative:
            # Gate perspective warps aggressively for one-side profile to avoid
            # visible stretch/keystone distortions in recognition preview.
            h64 = h.astype(np.float64)
            if abs(float(h64[2, 0])) > 1.2e-4 or abs(float(h64[2, 1])) > 1.2e-4:
                resized = cv2.resize(image, (template.width, template.height))
                return resized, self._preprocess(resized)["binary"]

            corners = np.array(
                [[[0.0, 0.0]], [[template.width - 1.0, 0.0]], [[template.width - 1.0, template.height - 1.0]], [[0.0, template.height - 1.0]]],
                dtype=np.float32,
            )
            warped_corners = cv2.perspectiveTransform(corners, h64.astype(np.float32)).reshape(4, 2)
            warped_area = abs(float(cv2.contourArea(warped_corners.astype(np.float32))))
            base_area = float(template.width * template.height)
            if base_area <= 0:
                resized = cv2.resize(image, (template.width, template.height))
                return resized, self._preprocess(resized)["binary"]
            area_ratio = warped_area / base_area
            if area_ratio < 0.96 or area_ratio > 1.04:
                resized = cv2.resize(image, (template.width, template.height))
                return resized, self._preprocess(resized)["binary"]

        aligned = cv2.warpPerspective(image, h, (template.width, template.height))
        return aligned, cv2.warpPerspective(self._preprocess(image)["binary"], h, (template.width, template.height))

    def _match_anchor_sets(self, detected: np.ndarray, template_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # stable match by relative quadrant ordering; robust to extra detections
        det = self._order_points(self._pick_corner_like_points(detected))
        dst = self._order_points(self._pick_corner_like_points(template_pts))
        return det.astype(np.float32), dst.astype(np.float32)

    def _pick_corner_like_points(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) <= 4:
            return pts[:4]
        sums = pts[:, 0] + pts[:, 1]
        diffs = pts[:, 0] - pts[:, 1]
        picks = [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ]
        return np.array(picks, dtype=np.float32)

    def detect_bubbles(self, binary: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        mask, mask_area = self._get_circular_mask(radius)
        r = radius
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
            local_mask = mask[(y0c - y0) : (y1c - y0), (x0c - x0) : (x1c - x0)]
            den = float(np.count_nonzero(local_mask))
            if roi.size == 0 or den == 0:
                continue
            dark_pixels = cv2.countNonZero(roi[local_mask])
            ratios[i] = float(dark_pixels) / den
        return ratios

    def classify_bubble(self, ratio: float) -> str:
        if ratio > self.fill_threshold:
            return "filled"
        if ratio < self.empty_threshold:
            return "empty"
        return "uncertain"

    def recognize_block(
        self,
        binary: np.ndarray,
        zone: Zone,
        template: Template,
        result: OMRResult,
        debug_overlay: np.ndarray | None = None,
    ) -> None:
        grid = zone.grid
        if not grid or not grid.bubble_positions:
            return

        centers = np.array(
            [
                (x * template.width, y * template.height) if x <= 1.0 and y <= 1.0 else (x, y)
                for x, y in grid.bubble_positions
            ],
            dtype=np.float32,
        )
        radius = int(zone.metadata.get("bubble_radius", 9))
        ratios = self.detect_bubbles(binary, centers, radius)

        rows, cols = max(1, grid.rows), max(1, grid.cols)
        need = rows * cols
        if len(ratios) < need:
            ratios = np.pad(ratios, (0, need - len(ratios)), constant_values=0.0)
        mat = ratios[:need].reshape(rows, cols)

        if debug_overlay is not None:
            self._draw_debug_overlay(debug_overlay, centers, ratios, radius)

        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            value, confs = self._decode_column_digits(mat, zone, grid, result)
            key = "student_id" if zone.zone_type == ZoneType.STUDENT_ID_BLOCK else "exam_code"
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
                order = np.argsort(row_scores)[::-1]
                top_i, second_i = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
                top, second = float(row_scores[top_i]), float(row_scores[second_i]) if len(order) > 1 else 0.0
                filled = np.where(row_scores > self.fill_threshold)[0]
                if len(filled) > 1:
                    result.recognition_errors.append(f"MCQ Q{qno}: multiple answer")
                    continue
                if self.classify_bubble(top) != "filled" or (top - second) <= self.certainty_margin:
                    result.recognition_errors.append(f"MCQ Q{qno}: uncertain")
                    continue
                result.mcq_answers[qno] = labels[top_i]
                confs.append(top - second)
            result.confidence_scores[f"mcq:{zone.id}"] = float(np.mean(confs)) if confs else 0.0
            return

        if zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
            self._decode_true_false(mat, zone, grid, result)
            return

        if zone.zone_type == ZoneType.NUMERIC_BLOCK:
            self._decode_numeric(mat, zone, grid, result)
            return

    def _decode_column_digits(self, mat: np.ndarray, zone: Zone, grid, result: OMRResult) -> tuple[str, list[float]]:
        rows, cols = mat.shape
        default_digit_map = list(range(10)) if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK) else list(range(rows))
        digit_map = zone.metadata.get("digit_map", default_digit_map)
        digits: list[str] = []
        confs: list[float] = []
        for c in range(cols):
            col_scores = mat[:, c]
            order = np.argsort(col_scores)[::-1]
            top_i, second_i = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
            top, second = float(col_scores[top_i]), float(col_scores[second_i]) if len(order) > 1 else 0.0
            if len(np.where(col_scores > self.fill_threshold)[0]) > 1:
                result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: double mark")
            if self.classify_bubble(top) != "filled" or (top - second) <= self.certainty_margin:
                digits.append("?")
                result.recognition_errors.append(f"{zone.zone_type.value} column {c+1}: uncertain")
            else:
                mapped = digit_map[top_i] if top_i < len(digit_map) else top_i
                digits.append(str(mapped))
            confs.append(top - second)
        return "".join(digits), confs

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

    def _get_circular_mask(self, radius: int) -> tuple[np.ndarray, float]:
        r = max(3, int(radius))
        if r in self._mask_cache:
            return self._mask_cache[r]
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        mask = (x * x + y * y) <= (r * r)
        area = float(math.pi * r * r)
        self._mask_cache[r] = (mask, area)
        return mask, area

    def _draw_debug_overlay(self, image: np.ndarray, centers: np.ndarray, ratios: np.ndarray, radius: int) -> None:
        for (x, y), ratio in zip(centers.astype(np.int32), ratios):
            state = self.classify_bubble(float(ratio))
            color = (0, 255, 0) if state == "filled" else (0, 0, 255) if state == "empty" else (255, 255, 0)
            cv2.circle(image, (int(x), int(y)), radius, (255, 0, 0), 1)
            cv2.circle(image, (int(x), int(y)), max(2, radius // 4), color, -1)
            cv2.putText(
                image,
                f"{ratio:.2f}",
                (int(x + radius + 2), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

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
