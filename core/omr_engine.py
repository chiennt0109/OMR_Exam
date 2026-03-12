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
    mcq_answers: dict[int, str] = field(default_factory=dict)
    true_false_answers: dict[int, dict[str, bool]] = field(default_factory=dict)
    numeric_answers: dict[int, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    issues: list[OMRIssue] = field(default_factory=list)


class OMRProcessor:
    def __init__(self, fill_threshold: float = 0.40, empty_threshold: float = 0.20, certainty_margin: float = 0.08):
        self.fill_threshold = fill_threshold
        self.empty_threshold = empty_threshold
        self.certainty_margin = certainty_margin

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

        # Backward-compatible alias
        setattr(result, "answers", result.mcq_answers)
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

        scores = self._measure_fill_scores(gray, pts)
        rows, cols = max(1, grid.rows), max(1, grid.cols)
        mat = scores.reshape(rows, cols) if len(scores) >= rows * cols else np.pad(scores, (0, rows * cols - len(scores)), constant_values=0).reshape(rows, cols)

        if zone.zone_type in (ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK):
            digits = []
            confs: list[float] = []
            for c in range(cols):
                col_scores = mat[:, c]
                order = np.argsort(col_scores)[::-1]
                top_i = int(order[0])
                top = float(col_scores[top_i])
                second = float(col_scores[int(order[1])]) if len(order) > 1 else 0.0
                if top < self.fill_threshold or (top - second) <= self.certainty_margin:
                    digits.append("?")
                    result.errors.append(f"{zone.zone_type.value} column {c+1}: uncertain")
                else:
                    digits.append(str(top_i))
                confs.append(top - second)
            value = "".join(digits)
            if zone.zone_type == ZoneType.EXAM_CODE_BLOCK:
                result.exam_code = value
                result.confidence[f"exam_code:{zone.id}"] = float(np.mean(confs)) if confs else 0.0
            else:
                result.student_id = value
                result.confidence[f"student_id:{zone.id}"] = float(np.mean(confs)) if confs else 0.0
            return

        if zone.zone_type == ZoneType.MCQ_BLOCK:
            confs: list[float] = []
            q_count = min(grid.question_count, rows)
            for r in range(q_count):
                row_scores = mat[r, :]
                order = np.argsort(row_scores)[::-1]
                top_idx = int(order[0])
                top = float(row_scores[top_idx])
                second = float(row_scores[int(order[1])]) if len(order) > 1 else 0.0
                qno = grid.question_start + r
                if top < self.fill_threshold or (top - second) <= self.certainty_margin:
                    result.errors.append(f"MCQ Q{qno}: uncertain")
                    continue
                if second > self.fill_threshold:
                    result.errors.append(f"MCQ Q{qno}: double mark")
                    continue
                labels = grid.options if grid.options else [chr(65 + i) for i in range(cols)]
                result.mcq_answers[qno] = labels[top_idx]
                confs.append(top - second)
            if confs:
                result.confidence[f"mcq:{zone.id}"] = float(np.mean(confs))
            return

        if zone.zone_type == ZoneType.TRUE_FALSE_BLOCK:
            qpb = int(zone.metadata.get("questions_per_block", 2))
            spq = int(zone.metadata.get("statements_per_question", 4))
            cps = int(zone.metadata.get("choices_per_statement", 2))
            confs: list[float] = []
            labels = [chr(ord("a") + i) for i in range(spq)]
            for q in range(qpb):
                qno = grid.question_start + q
                result.true_false_answers[qno] = {}
                for sidx in range(spq):
                    r = q * spq + sidx
                    if r >= rows:
                        break
                    row_scores = mat[r, :max(1, cps)]
                    order = np.argsort(row_scores)[::-1]
                    top = float(row_scores[int(order[0])])
                    second = float(row_scores[int(order[1])]) if len(order) > 1 else 0.0
                    if top < self.fill_threshold or second > self.fill_threshold:
                        result.errors.append(f"TF Q{qno}{labels[sidx]}: uncertain/double")
                        continue
                    pick = int(order[0])
                    result.true_false_answers[qno][labels[sidx]] = bool(pick == 0)
                    confs.append(top - second)
            if confs:
                result.confidence[f"tf:{zone.id}"] = float(np.mean(confs))
            return

        if zone.zone_type == ZoneType.NUMERIC_BLOCK:
            digits_per_answer = int(zone.metadata.get("digits_per_answer", 5))
            q_count = int(zone.metadata.get("total_questions", zone.metadata.get("questions", 1)))
            confs: list[float] = []
            for q in range(q_count):
                digits = []
                for d in range(digits_per_answer):
                    c = q * digits_per_answer + d
                    if c >= cols:
                        break
                    col_scores = mat[:, c]
                    order = np.argsort(col_scores)[::-1]
                    top_i = int(order[0])
                    top = float(col_scores[top_i])
                    second = float(col_scores[int(order[1])]) if len(order) > 1 else 0.0
                    if top < self.fill_threshold or (top - second) <= self.certainty_margin:
                        digits.append("?")
                        result.errors.append(f"NUM Q{grid.question_start+q} digit {d+1}: uncertain")
                    else:
                        digits.append(str(top_i))
                    confs.append(top - second)
                result.numeric_answers[grid.question_start + q] = "".join(digits)
            if confs:
                result.confidence[f"num:{zone.id}"] = float(np.mean(confs))
            return

    def _measure_fill_scores(self, gray: np.ndarray, centers: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
        r = 7
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        mask = (xx * xx + yy * yy) <= (r * r)
        area = float(np.count_nonzero(mask))

        scores = np.zeros((len(centers),), dtype=np.float32)
        h, w = th.shape[:2]
        for i, (x, y) in enumerate(centers.astype(int)):
            x0, y0 = x - r, y - r
            x1, y1 = x + r + 1, y + r + 1
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                x0c, y0c = max(0, x0), max(0, y0)
                x1c, y1c = min(w, x1), min(h, y1)
                roi = th[y0c:y1c, x0c:x1c]
                m = mask[(y0c - y0):(y1c - y0), (x0c - x0):(x1c - x0)]
                den = float(np.count_nonzero(m))
                scores[i] = 0.0 if roi.size == 0 or den == 0 else float(np.count_nonzero(roi[m])) / den
            else:
                roi = th[y0:y1, x0:x1]
                scores[i] = float(np.count_nonzero(roi[mask])) / area
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
