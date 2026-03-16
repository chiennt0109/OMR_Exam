import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

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

    def test_template_dict_persists_template_coordinate_space(self):
        tpl = Template(name="t", image_path="x.png", width=1234, height=1754)
        payload = tpl.to_dict()
        self.assertEqual(payload["metadata"]["coordinate_mode"], "relative")
        self.assertEqual(payload["metadata"]["template_width"], 1234)
        self.assertEqual(payload["metadata"]["template_height"], 1754)

    def test_editor_and_batch_pipeline_share_same_recognition_entrypoint(self):
        template = Template(name="t", image_path="", width=200, height=100, anchors=[], zones=[])
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "sheet.png"
            cv2.imwrite(str(img_path), np.zeros((100, 200, 3), dtype=np.uint8))

            with patch.object(self.processor, "_normalize_to_200_dpi", return_value=(str(img_path), "")), \
                patch.object(self.processor, "_correct_rotation", side_effect=lambda x: x), \
                patch.object(self.processor, "_preprocess", return_value={"binary": np.zeros((100, 200), dtype=np.uint8)}), \
                patch.object(self.processor, "correct_perspective", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((100, 200), dtype=np.uint8))), \
                patch.object(self.processor, "detect_anchors", return_value=[]):
                editor_res = self.processor.recognize_sheet(str(img_path), template)
                batch_res = self.processor.process_batch([str(img_path)], template)[0]

        self.assertEqual(editor_res.mcq_answers, batch_res.mcq_answers)
        self.assertEqual(editor_res.true_false_answers, batch_res.true_false_answers)
        self.assertEqual(editor_res.numeric_answers, batch_res.numeric_answers)

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

    def test_numeric_block_removes_unfilled_placeholders(self):
        rows, cols = 12, 3
        bubbles = []
        for r in range(rows):
            for c in range(cols):
                bubbles.append((30 + c * 40, 20 + r * 16))

        template = Template(
            name="num2",
            image_path="",
            width=200,
            height=240,
            anchors=[AnchorPoint(0.05, 0.05), AnchorPoint(0.95, 0.05), AnchorPoint(0.95, 0.95), AnchorPoint(0.05, 0.95)],
            zones=[
                Zone(
                    id="num2",
                    name="num",
                    zone_type=ZoneType.NUMERIC_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=rows, cols=cols, question_start=1, question_count=1, options=[], bubble_positions=bubbles),
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
                        "decimal_symbol": ",",
                    },
                )
            ],
        )
        binary = np.zeros((240, 200), dtype=np.uint8)

        def fill(row: int, col: int) -> None:
            idx = row * cols + col
            x, y = bubbles[idx]
            cv2.circle(binary, (int(x), int(y)), 5, 255, -1)

        fill(0, 0)   # sign
        fill(11, 0)  # first digit = 9
        # middle digit intentionally left empty => would become '?'
        fill(1, 2)   # decimal marker on col 3
        fill(5, 2)   # third digit = 3

        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.numeric_answers.get(1), "-9,3")

    def test_numeric_block_drops_leading_placeholder_after_sign(self):
        rows, cols = 12, 3
        bubbles = []
        for r in range(rows):
            for c in range(cols):
                bubbles.append((30 + c * 40, 20 + r * 16))

        template = Template(
            name="num_sign_leading_blank",
            image_path="",
            width=200,
            height=240,
            anchors=[AnchorPoint(0.05, 0.05), AnchorPoint(0.95, 0.05), AnchorPoint(0.95, 0.95), AnchorPoint(0.05, 0.95)],
            zones=[
                Zone(
                    id="num_sign_leading_blank",
                    name="num",
                    zone_type=ZoneType.NUMERIC_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=rows, cols=cols, question_start=1, question_count=1, options=[], bubble_positions=bubbles),
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
                        "decimal_symbol": ",",
                    },
                )
            ],
        )
        binary = np.zeros((240, 200), dtype=np.uint8)

        def fill(row: int, col: int) -> None:
            idx = row * cols + col
            x, y = bubbles[idx]
            cv2.circle(binary, (int(x), int(y)), 5, 255, -1)

        fill(0, 0)   # sign
        # first digit intentionally blank => '?'
        fill(11, 1)  # second digit = 9
        fill(5, 2)   # third digit = 3

        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.numeric_answers.get(1), "-93")

    def test_numeric_block_keeps_placeholder_when_no_decimal_mark(self):
        rows, cols = 12, 3
        bubbles = []
        for r in range(rows):
            for c in range(cols):
                bubbles.append((30 + c * 40, 20 + r * 16))

        template = Template(
            name="num3",
            image_path="",
            width=200,
            height=240,
            anchors=[AnchorPoint(0.05, 0.05), AnchorPoint(0.95, 0.05), AnchorPoint(0.95, 0.95), AnchorPoint(0.05, 0.95)],
            zones=[
                Zone(
                    id="num3",
                    name="num",
                    zone_type=ZoneType.NUMERIC_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=rows, cols=cols, question_start=1, question_count=1, options=[], bubble_positions=bubbles),
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
                        "decimal_symbol": ",",
                    },
                )
            ],
        )
        binary = np.zeros((240, 200), dtype=np.uint8)

        def fill(row: int, col: int) -> None:
            idx = row * cols + col
            x, y = bubbles[idx]
            cv2.circle(binary, (int(x), int(y)), 5, 255, -1)

        fill(0, 0)
        fill(11, 0)
        # no decimal mark filled
        fill(5, 2)

        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.numeric_answers.get(1), "-9?3")

    def test_student_id_custom_digit_map(self):
        template = Template(
            name="sid",
            image_path="",
            width=200,
            height=200,
            anchors=[AnchorPoint(0.05, 0.05), AnchorPoint(0.95, 0.05), AnchorPoint(0.95, 0.95), AnchorPoint(0.05, 0.95)],
            zones=[
                Zone(
                    id="sid1",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(
                        rows=10,
                        cols=2,
                        question_start=1,
                        question_count=2,
                        options=[],
                        bubble_positions=[(40 + c * 60, 20 + r * 16) for r in range(10) for c in range(2)],
                    ),
                    metadata={"bubble_radius": 5, "digit_map": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]},
                )
            ],
        )
        binary = np.zeros((200, 200), dtype=np.uint8)
        # mark row 0 col 0 => digit 9, row 9 col 1 => digit 0
        cv2.circle(binary, (40, 20), 5, 255, -1)
        cv2.circle(binary, (100, 164), 5, 255, -1)
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.student_id, "90")

    def test_student_id_default_digit_map_matches_exam_code(self):
        template = Template(
            name="sid_default",
            image_path="",
            width=200,
            height=200,
            anchors=[AnchorPoint(0.05, 0.05), AnchorPoint(0.95, 0.05), AnchorPoint(0.95, 0.95), AnchorPoint(0.05, 0.95)],
            zones=[
                Zone(
                    id="sid_default",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(
                        rows=10,
                        cols=1,
                        question_start=1,
                        question_count=1,
                        options=[],
                        bubble_positions=[(40, 20 + r * 16) for r in range(10)],
                    ),
                    metadata={"bubble_radius": 5},
                )
            ],
        )
        binary = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(binary, (40, 20 + 7 * 16), 5, 255, -1)
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.student_id, "7")

    def test_numeric_block_removes_only_trailing_placeholders_without_decimal(self):
        rows, cols = 12, 4
        bubbles = []
        for r in range(rows):
            for c in range(cols):
                bubbles.append((25 + c * 35, 20 + r * 14))

        template = Template(
            name="num4",
            image_path="",
            width=220,
            height=220,
            anchors=[AnchorPoint(0.05, 0.05), AnchorPoint(0.95, 0.05), AnchorPoint(0.95, 0.95), AnchorPoint(0.05, 0.95)],
            zones=[
                Zone(
                    id="num4",
                    name="num",
                    zone_type=ZoneType.NUMERIC_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=rows, cols=cols, question_start=1, question_count=1, options=[], bubble_positions=bubbles),
                    metadata={
                        "bubble_radius": 5,
                        "questions_per_block": 1,
                        "digits_per_answer": 4,
                        "sign_row": 1,
                        "decimal_row": 2,
                        "digit_start_row": 3,
                        "sign_columns": [1],
                        "decimal_columns": [2, 3],
                        "digit_map": list(range(10)),
                        "sign_symbol": "-",
                        "decimal_symbol": ",",
                    },
                )
            ],
        )
        binary = np.zeros((220, 220), dtype=np.uint8)

        def fill(row: int, col: int) -> None:
            idx = row * cols + col
            x, y = bubbles[idx]
            cv2.circle(binary, (int(x), int(y)), 5, 255, -1)

        fill(0, 0)    # sign
        fill(11, 0)   # first digit 9
        fill(5, 2)    # third digit 3 (second digit left blank => internal '?')
        # fourth digit left blank => trailing '?', should be removed

        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.numeric_answers.get(1), "-9?3")


if __name__ == "__main__":
    unittest.main()
