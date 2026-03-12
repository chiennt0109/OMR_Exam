from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import cv2
import numpy as np

from models.template import Template, ZoneType, Zone


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
    issues: list[OMRIssue] = field(default_factory=list)


class OMRProcessor:
    def __init__(self, fill_threshold: float = 0.42):
        self.fill_threshold = fill_threshold

    def process_image(self, image_path: str | Path, template: Template) -> OMRResult:
        result = OMRResult(image_path=str(image_path))
        src = cv2.imread(str(image_path))
        if src is None:
            result.issues.append(OMRIssue("FILE", "Unable to load image"))
            return result

        sheet = self._detect_exam_sheet(src)
        aligned = self._align_to_template(sheet, template, result)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        for zone in template.zones:
            zx, zy, zw, zh = self._zone_to_abs(zone, template.width, template.height)
            crop = gray[zy : zy + zh, zx : zx + zw]
            if crop.size == 0:
                result.issues.append(OMRIssue("TEMPLATE_MISMATCH", "Zone outside image", zone.id))
                continue

            if zone.zone_type == ZoneType.STUDENT_ID:
                result.student_id = self._read_digit_bubbles(crop) or result.student_id
                if not result.student_id:
                    result.issues.append(OMRIssue("UNREADABLE_STUDENT_ID", "Student ID could not be read", zone.id))
            elif zone.zone_type == ZoneType.EXAM_CODE:
                result.exam_code = self._read_digit_bubbles(crop) or result.exam_code
            elif zone.zone_type in (ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_BLOCK, ZoneType.NUMERIC_BLOCK, ZoneType.ID_BLOCK) and zone.grid:
                answer, issue = self._detect_single_answer(crop, zone.grid.options)
                q_no = zone.grid.question_start
                if answer:
                    result.answers[q_no] = answer
                if issue:
                    result.issues.append(OMRIssue(issue, f"Issue in question {q_no}", zone.id))

        if not result.exam_code:
            result.issues.append(OMRIssue("TEMPLATE_MISMATCH", "Exam code missing or unreadable"))
        return result

    def process_batch(
        self,
        image_paths: list[str],
        template: Template,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[OMRResult]:
        results: list[OMRResult] = []
        total = len(image_paths)
        for idx, image_path in enumerate(image_paths, start=1):
            res = self.process_image(image_path, template)
            results.append(res)
            if progress_callback:
                progress_callback(idx, total, image_path)
        return results

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

    def _align_to_template(self, image: np.ndarray, template: Template, result: OMRResult) -> np.ndarray:
        h, w = image.shape[:2]
        if abs(w - template.width) < 8 and abs(h - template.height) < 8:
            return image
        if template.width <= 0 or template.height <= 0:
            result.issues.append(OMRIssue("MISSING_ANCHORS", "Template dimensions invalid"))
            return image
        return cv2.resize(image, (template.width, template.height), interpolation=cv2.INTER_LINEAR)

    def _read_digit_bubbles(self, crop: np.ndarray) -> str:
        thresh = cv2.adaptiveThreshold(
            cv2.GaussianBlur(crop, (5, 5), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            4,
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ""
        filled = [cnt for cnt in contours if cv2.contourArea(cnt) > 20]
        return str(len(filled)) if filled else ""

    def _detect_single_answer(self, crop: np.ndarray, options: list[str]) -> tuple[str | None, str | None]:
        gray = cv2.GaussianBlur(crop, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            6,
        )
        step = max(1, thresh.shape[1] // len(options))
        fill_ratios: list[float] = []
        for i in range(len(options)):
            x0 = i * step
            x1 = thresh.shape[1] if i == len(options) - 1 else (i + 1) * step
            bubble_roi = thresh[:, x0:x1]
            ratio = float(np.count_nonzero(bubble_roi)) / float(bubble_roi.size)
            fill_ratios.append(ratio)

        marks = [i for i, ratio in enumerate(fill_ratios) if ratio >= self.fill_threshold]
        if not marks:
            return None, "BLANK_ANSWER"
        if len(marks) > 1:
            return None, "MULTIPLE_ANSWERS"
        return options[marks[0]], None


    def _zone_to_abs(self, zone: Zone, width: int, height: int) -> tuple[int, int, int, int]:
        if zone.width <= 1.0 and zone.height <= 1.0:
            x = int(zone.x * width)
            y = int(zone.y * height)
            w = max(1, int(zone.width * width))
            h = max(1, int(zone.height * height))
            return x, y, w, h
        return int(zone.x), int(zone.y), int(zone.width), int(zone.height)

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = max(int(height_a), int(height_b))

        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype="float32",
        )

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
