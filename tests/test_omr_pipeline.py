import unittest

import cv2
import numpy as np

from core.omr_engine import OMRProcessor
from models.template import AnchorPoint, BubbleGrid, Template, Zone, ZoneType


class OMRPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = OMRProcessor(debug_mode=False)

    def test_classify_bubble_thresholds(self):
        self.assertEqual(self.processor.classify_bubble(0.6), "filled")
        self.assertEqual(self.processor.classify_bubble(0.1), "empty")
        self.assertEqual(self.processor.classify_bubble(0.3), "uncertain")

    def test_detect_anchors_square_markers(self):
        img = np.zeros((400, 300), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (45, 45), 255, -1)
        cv2.rectangle(img, (250, 20), (275, 45), 255, -1)
        cv2.rectangle(img, (20, 350), (45, 375), 255, -1)
        cv2.rectangle(img, (250, 350), (275, 375), 255, -1)

        anchors = self.processor.detect_anchors(img)
        self.assertGreaterEqual(len(anchors), 4)

    def test_detect_bubbles_ratio(self):
        binary = np.zeros((120, 120), dtype=np.uint8)
        cv2.circle(binary, (40, 60), 10, 255, -1)
        cv2.circle(binary, (80, 60), 10, 255, 1)
        centers = np.array([[40, 60], [80, 60]], dtype=np.float32)
        ratios = self.processor.detect_bubbles(binary, centers, 10)
        self.assertGreater(ratios[0], 0.75)
        self.assertLess(ratios[1], 0.35)

    def test_mcq_recognition(self):
        template = Template(
            name="t",
            image_path="",
            width=200,
            height=120,
            anchors=[AnchorPoint(0.05, 0.05), AnchorPoint(0.95, 0.05), AnchorPoint(0.95, 0.95), AnchorPoint(0.05, 0.95)],
            zones=[
                Zone(
                    id="mcq1",
                    name="mcq",
                    zone_type=ZoneType.MCQ_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(
                        rows=1,
                        cols=4,
                        question_start=1,
                        question_count=1,
                        options=["A", "B", "C", "D"],
                        bubble_positions=[(40, 60), (80, 60), (120, 60), (160, 60)],
                    ),
                    metadata={"bubble_radius": 10},
                )
            ],
        )
        binary = np.zeros((120, 200), dtype=np.uint8)
        cv2.circle(binary, (120, 60), 10, 255, -1)  # C filled
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()

        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.mcq_answers.get(1), "C")

    def test_legacy_template_absolute_coordinates_are_converted(self):
        raw = {
            "name": "legacy",
            "image_path": "sheet.png",
            "width": 1000,
            "height": 2000,
            "metadata": {"coordinate_mode": "absolute"},
            "anchors": [{"x": 100, "y": 200, "name": "A1"}],
            "zones": [
                {
                    "id": "z1",
                    "name": "Z",
                    "zone_type": "MCQ_BLOCK",
                    "x": 100,
                    "y": 300,
                    "width": 400,
                    "height": 500,
                    "grid": {
                        "rows": 1,
                        "cols": 2,
                        "question_start": 1,
                        "question_count": 1,
                        "options": ["A", "B"],
                        "bubble_positions": [[200, 500], [300, 500]],
                    },
                    "metadata": {},
                }
            ],
        }

        tpl = Template.from_dict(raw)
        self.assertAlmostEqual(tpl.anchors[0].x, 0.1, places=4)
        self.assertAlmostEqual(tpl.anchors[0].y, 0.1, places=4)
        z = tpl.zones[0]
        self.assertAlmostEqual(z.x, 0.1, places=4)
        self.assertAlmostEqual(z.y, 0.15, places=4)
        self.assertAlmostEqual(z.width, 0.4, places=4)
        self.assertAlmostEqual(z.height, 0.25, places=4)
        self.assertAlmostEqual(z.grid.bubble_positions[0][0], 0.2, places=4)
        self.assertAlmostEqual(z.grid.bubble_positions[0][1], 0.25, places=4)

    def test_numeric_block_signed_decimal_layout(self):
        rows, cols = 12, 3
        bubbles = []
        for r in range(rows):
            for c in range(cols):
                bubbles.append((30 + c * 40, 20 + r * 16))

        template = Template(
            name="num",
            image_path="",
            width=200,
            height=240,
            anchors=[AnchorPoint(0.05, 0.05), AnchorPoint(0.95, 0.05), AnchorPoint(0.95, 0.95), AnchorPoint(0.05, 0.95)],
            zones=[
                Zone(
                    id="num1",
                    name="num",
                    zone_type=ZoneType.NUMERIC_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(
                        rows=rows,
                        cols=cols,
                        question_start=1,
                        question_count=1,
                        options=[],
                        bubble_positions=bubbles,
                    ),
                    metadata={
                        "bubble_radius": 5,
                        "questions_per_block": 1,
                        "digits_per_answer": 3,
                        "sign_row": 1,
                        "decimal_row": 2,
                        "digit_start_row": 3,
                        "sign_columns": [1],
                        "decimal_columns": [2, 3],
                        "digit_map": list(range(10)),
                        "sign_symbol": "-",
                        "decimal_symbol": ".",
                    },
                )
            ],
        )

        binary = np.zeros((240, 200), dtype=np.uint8)

        def fill(row: int, col: int) -> None:
            idx = row * cols + col
            x, y = bubbles[idx]
            cv2.circle(binary, (int(x), int(y)), 5, 255, -1)

        fill(0, 0)   # sign row, column 1 -> negative
        fill(11, 0)  # digit 9 in first column (row index 11 => 11-2=9)
        fill(3, 1)   # digit 1 in second column
        fill(1, 2)   # decimal row on third column -> place decimal before col 3
        fill(5, 2)   # digit 3 in third column

        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.numeric_answers.get(1), "-91.3")


if __name__ == "__main__":
    unittest.main()
