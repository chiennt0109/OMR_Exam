from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from models.template import AnchorPoint, BubbleGrid, Template, Zone, ZoneType


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
    answers: dict[int, str] = field(default_factory=dict)
    true_false_answers: dict[int, dict[str, bool]] = field(default_factory=dict)
    numeric_answers: dict[int, str] = field(default_factory=dict)
    issues: list[OMRIssue] = field(default_factory=list)


class OMRProcessor:
    def __init__(self, fill_threshold: float = 0.42):
        self.fill_threshold = fill_threshold

    def process_image(self, image_path: str | Path, template: Template) -> OMRResult:
        result = OMRResult(image_path=str(image_path))
        norm_path, dpi_msg = self._normalize_to_200_dpi(str(image_path))
        if dpi_msg:
            result.issues.append(OMRIssue("DPI", dpi_msg))

        src = cv2.imread(norm_path)
        if src is None:
            result.issues.append(OMRIssue("FILE", "Unable to load image"))
            return result

        sheet = self._detect_exam_sheet(src)
        aligned = self._align_via_anchors(sheet, template, result)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        for zone in template.zones:
            if zone.zone_type == ZoneType.ANCHOR:
                continue
            if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK, ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_BLOCK, ZoneType.NUMERIC_BLOCK):
                self._recognize_zone(gray, zone, template, result)

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
        s = 200.0 / dpi
        out = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_LINEAR)
        out_path = str(Path(path).with_name(f"{Path(path).stem}_200dpi.png"))
        cv2.imwrite(out_path, out)
        return out_path, f"Input DPI={dpi}. Normalized to 200 DPI scale."

    def _detect_exam_sheet(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        page = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(page, True)
        approx = cv2.approxPolyDP(page, 0.02 * peri, True)
        if len(approx) == 4:
            return self._four_point_transform(image, approx.reshape(4, 2).astype("float32"))
        return image

    def _align_via_anchors(self, image: np.ndarray, template: Template, result: OMRResult) -> np.ndarray:
        detected = self._detect_anchor_markers(image)
        template_pts = np.array([(a.x * template.width, a.y * template.height) if a.x <= 1.0 and a.y <= 1.0 else (a.x, a.y) for a in template.anchors], dtype=np.float32)

        if len(template_pts) < 4:
            result.issues.append(OMRIssue("MISSING_ANCHORS", "Template has fewer than 4 anchors"))
            return cv2.resize(image, (template.width, template.height))

        if len(detected) < 4:
            result.issues.append(OMRIssue("MISSING_ANCHORS", "Could not detect enough anchors on scan"))
            return cv2.resize(image, (template.width, template.height))

        src_pts, dst_pts = self._match_anchor_sets(np.array(detected, dtype=np.float32), template_pts)
        if len(src_pts) < 4:
            result.issues.append(OMRIssue("MISSING_ANCHORS", "Anchor matching failed"))
            return cv2.resize(image, (template.width, template.height))

        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if h is None:
            result.issues.append(OMRIssue("ALIGNMENT", "Homography estimation failed"))
            return cv2.resize(image, (template.width, template.height))

        return cv2.warpPerspective(image, h, (template.width, template.height))

    def _detect_anchor_markers(self, image: np.ndarray) -> list[tuple[float, float]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        anchors: list[tuple[float, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue
            ratio = w / float(h)
            if 0.75 <= ratio <= 1.25:
                m = cv2.moments(cnt)
                if m["m00"] != 0:
                    anchors.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
        anchors = sorted(anchors, key=lambda p: p[0] + p[1])
        return anchors[:12]

    def _match_anchor_sets(self, detected: np.ndarray, template_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = min(len(detected), len(template_pts), 12)
        det = detected[:n]
        tmp = template_pts[:n]
        # nearest-neighbor matching around centroid-normalized coordinates
        det_c = det - det.mean(axis=0)
        tmp_c = tmp - tmp.mean(axis=0)
        src = []
        dst = []
        used = set()
        for p in tmp_c:
            dists = np.linalg.norm(det_c - p, axis=1)
            idxs = np.argsort(dists)
            pick = None
            for idx in idxs:
                if int(idx) not in used:
                    pick = int(idx)
                    used.add(pick)
                    break
            if pick is not None:
                src.append(det[pick])
                dst.append(tmp[len(dst)])
        return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

    def _recognize_zone(self, gray: np.ndarray, zone: Zone, template: Template, result: OMRResult) -> None:
        grid = zone.grid
        if not grid or not grid.bubble_positions:
            return

        pts = np.array([
            (bx * template.width, by * template.height) if bx <= 1.0 and by <= 1.0 else (bx, by)
            for bx, by in grid.bubble_positions
        ], dtype=np.float32)

        filled_scores = self._measure_fill_scores(gray, pts)
        options = max(1, len(grid.options))
        question_count = len(filled_scores) // options

        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            digits = []
            for q in range(question_count):
                row_scores = filled_scores[q * options:(q + 1) * options]
                idx = int(np.argmax(row_scores))
                digits.append(str(idx))
            val = "".join(digits)
            if zone.zone_type == ZoneType.EXAM_CODE_BLOCK:
                result.exam_code = val
            else:
                result.student_id = val
            return

        if zone.zone_type == ZoneType.MCQ_BLOCK:
            for q in range(question_count):
                row_scores = filled_scores[q * options:(q + 1) * options]
                idx = int(np.argmax(row_scores))
                result.answers[grid.question_start + q] = grid.options[idx]
            return

        if zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
            qpb = int(zone.metadata.get("questions_per_block", 2))
            spq = int(zone.metadata.get("statements_per_question", 4))
            cps = int(zone.metadata.get("choices_per_statement", 2))
            statement_labels = [chr(ord("a") + i) for i in range(spq)]
            for q in range(qpb):
                result.true_false_answers[grid.question_start + q] = {}
                for s in range(spq):
                    idx = q * spq + s
                    if idx >= question_count:
                        break
                    row_scores = filled_scores[idx * options:(idx + 1) * options]
                    pick = int(np.argmax(row_scores))
                    result.true_false_answers[grid.question_start + q][statement_labels[s]] = bool(pick == 0) if cps == 2 else bool(pick)
            return

        if zone.zone_type == ZoneType.NUMERIC_BLOCK:
            digits_per_answer = int(zone.metadata.get("digits_per_answer", 5))
            q_count = int(zone.metadata.get("questions", 5))
            for q in range(q_count):
                digs = []
                for d in range(digits_per_answer):
                    idx = q * digits_per_answer + d
                    if idx >= question_count:
                        break
                    row_scores = filled_scores[idx * options:(idx + 1) * options]
                    digs.append(str(int(np.argmax(row_scores))))
                result.numeric_answers[grid.question_start + q] = "".join(digs)
            return

    def _measure_fill_scores(self, gray: np.ndarray, centers: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
        r = 6
        scores = np.zeros((len(centers),), dtype=np.float32)
        for i, (x, y) in enumerate(centers.astype(int)):
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(th.shape[1], x + r), min(th.shape[0], y + r)
            roi = th[y0:y1, x0:x1]
            scores[i] = 0.0 if roi.size == 0 else float(np.count_nonzero(roi)) / float(roi.size)
        return scores

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = max(int(width_a), int(width_b))
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = max(int(height_a), int(height_b))
        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
        m = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, m, (max_w, max_h))

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
