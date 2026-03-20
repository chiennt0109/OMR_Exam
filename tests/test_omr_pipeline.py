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

    def test_detect_anchors_ignores_inner_bubbles_when_border_mode_enabled(self):
        img = np.zeros((400, 300), dtype=np.uint8)
        for x, y in [(20, 20), (260, 20), (20, 360), (260, 360)]:
            cv2.rectangle(img, (x, y), (x + 18, y + 18), 255, -1)
        for x in range(80, 220, 35):
            for y in range(80, 320, 45):
                cv2.circle(img, (x, y), 10, 255, 2)

        anchors = self.processor.detect_anchors(img, use_border_padding=True, relaxed_polygon=True, max_points=40)
        self.assertLessEqual(len(anchors), 8)

    def test_student_id_zone_keeps_expected_centers(self):
        template = Template(
            name="sid",
            image_path="",
            width=200,
            height=200,
            anchors=[],
            zones=[
                Zone(
                    id="sid",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(40 + c * 60, 20 + r * 16) for r in range(10) for c in range(2)]),
                    metadata={"bubble_radius": 5},
                )
            ],
        )
        binary = np.zeros((200, 200), dtype=np.uint8)
        centers = self.processor._resolve_zone_centers(binary, template.zones[0], template)
        expected = np.array(template.zones[0].grid.bubble_positions, dtype=np.float32)
        self.assertTrue(np.allclose(centers, expected))

    def test_student_id_zone_applies_column_offset_for_tilt(self):
        grid = BubbleGrid(rows=4, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(40 + c * 40, 20 + r * 20) for r in range(4) for c in range(2)])
        template = Template(
            name="sid_tilt",
            image_path="",
            width=140,
            height=120,
            anchors=[],
            zones=[Zone(id="sid_tilt", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0, y=0, width=1, height=1, grid=grid, metadata={"bubble_radius": 5})],
        )
        binary = np.zeros((120, 140), dtype=np.uint8)
        expected = np.array(grid.bubble_positions, dtype=np.float32)
        shifted = expected.copy()
        shifted[0::2] += np.array([2.0, 1.0], dtype=np.float32)
        shifted[1::2] += np.array([6.0, 1.0], dtype=np.float32)
        for x, y in shifted.astype(np.int32):
            cv2.circle(binary, (int(x), int(y)), 5, 255, 1)

        centers = self.processor._resolve_zone_centers(binary, template.zones[0], template)
        self.assertLess(float(np.mean(np.linalg.norm(centers - shifted, axis=1))), 3.5)

    def test_student_id_zone_refines_each_row_after_column_shift(self):
        grid = BubbleGrid(rows=4, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(40, 20 + r * 20) for r in range(4)])
        template = Template(
            name="sid_row_refine",
            image_path="",
            width=100,
            height=120,
            anchors=[],
            zones=[Zone(id="sid_row_refine", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0, y=0, width=1, height=1, grid=grid, metadata={"bubble_radius": 5})],
        )
        binary = np.zeros((120, 100), dtype=np.uint8)
        shifted = np.array([[42, 20], [43, 41], [42, 60], [44, 81]], dtype=np.int32)
        for x, y in shifted:
            cv2.circle(binary, (int(x), int(y)), 5, 255, 1)

        centers = self.processor._resolve_zone_centers(binary, template.zones[0], template)
        self.assertLess(float(np.mean(np.linalg.norm(centers - shifted.astype(np.float32), axis=1))), 3.0)

    def test_detect_bubbles_ratio(self):
        binary = np.zeros((120, 120), dtype=np.uint8)
        cv2.circle(binary, (40, 60), 10, 255, -1)
        cv2.circle(binary, (80, 60), 10, 255, 1)
        centers = np.array([[40, 60], [80, 60]], dtype=np.float32)
        ratios = self.processor.detect_bubbles(binary, centers, 10)
        self.assertGreater(ratios[0], 0.75)
        self.assertLess(ratios[1], 0.35)

    def test_detect_center_core_marks_prefers_filled_center_over_outline(self):
        binary = np.zeros((120, 120), dtype=np.uint8)
        cv2.circle(binary, (40, 60), 10, 255, -1)
        cv2.circle(binary, (80, 60), 10, 255, 1)
        centers = np.array([[40, 60], [80, 60]], dtype=np.float32)
        ratios = self.processor._detect_center_core_marks(binary, centers, 10)
        self.assertGreater(ratios[0], 0.8)
        self.assertLess(ratios[1], 0.25)

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

    def test_student_id_decode_uses_softer_column_margin(self):
        zone = Zone(
            id="sid_margin",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[]),
            metadata={},
        )
        mat = np.array([[0.58], [0.18], [0.48], [0.16], [0.15], [0.14], [0.13], [0.12], [0.11], [0.10]], dtype=np.float32)
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        digits, _ = self.processor._decode_column_digits(mat, zone, zone.grid, result_stub)
        self.assertEqual(digits, "0")

    def test_student_id_recognize_block_weights_center_core_more_than_outline_ratio(self):
        template = Template(
            name="sid_weight",
            image_path="",
            width=100,
            height=100,
            anchors=[],
            zones=[
                Zone(
                    id="sid_weight",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(50, 10 + r * 8) for r in range(10)]),
                    metadata={"bubble_radius": 5},
                )
            ],
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        with patch.object(self.processor, "_resolve_zone_centers", return_value=np.array([[50, 10 + r * 8] for r in range(10)], dtype=np.float32)), \
            patch.object(self.processor, "detect_bubbles", return_value=np.array([0.42, 0.35, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float32)), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=np.array([0.86, 0.30, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)):
            self.processor.recognize_block(np.zeros((100, 100), dtype=np.uint8), template.zones[0], template, result_stub)
        self.assertEqual(result_stub.student_id, "0")

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
    def _build_sheet_with_corner_markers(self, width: int = 600, height: int = 900) -> np.ndarray:
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (80, 80), (0, 0, 0), -1)
        cv2.rectangle(img, (width - 80, 30), (width - 30, 80), (0, 0, 0), -1)
        cv2.rectangle(img, (width - 80, height - 80), (width - 30, height - 30), (0, 0, 0), -1)
        cv2.rectangle(img, (30, height - 80), (80, height - 30), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 0, 0), 4)
        return img

    def test_detect_page_corners_from_sheet_border(self):
        image = self._build_sheet_with_corner_markers()
        corners = self.processor._detect_page_corners(image)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (4, 2))
        self.assertLess(np.linalg.norm(corners[0] - np.array([9.0, 9.0], dtype=np.float32)), 40.0)
        self.assertLess(np.linalg.norm(corners[2] - np.array([590.0, 890.0], dtype=np.float32)), 40.0)

    def test_correct_perspective_handles_rotation_and_perspective_distortion(self):
        template = Template(
            name="sheet",
            image_path="",
            width=400,
            height=600,
            anchors=[AnchorPoint(0.08, 0.08), AnchorPoint(0.92, 0.08), AnchorPoint(0.92, 0.92), AnchorPoint(0.08, 0.92)],
            zones=[],
        )
        base = np.full((600, 400, 3), 255, dtype=np.uint8)
        cv2.rectangle(base, (0, 0), (399, 599), (0, 0, 0), 4)
        for x, y in [(32, 48), (368, 48), (368, 552), (32, 552)]:
            cv2.rectangle(base, (x - 14, y - 14), (x + 14, y + 14), (0, 0, 0), -1)

        src = np.array([[0, 0], [399, 0], [399, 599], [0, 599]], dtype=np.float32)
        dst = np.array([[30, 20], [360, 5], [390, 590], [10, 560]], dtype=np.float32)
        warped = cv2.warpPerspective(base, cv2.getPerspectiveTransform(src, dst), (400, 600), borderValue=(255, 255, 255))
        rotated = cv2.warpAffine(warped, cv2.getRotationMatrix2D((200, 300), 7.0, 1.0), (400, 600), borderValue=(255, 255, 255))
        binary = self.processor._preprocess(rotated)["binary"]
        result_stub = type("R", (), {"issues": []})()

        aligned, aligned_binary = self.processor.correct_perspective(rotated, binary, template, result_stub)
        anchors = np.array(self.processor.detect_anchors(aligned_binary, max_points=20), dtype=np.float32)
        expected = np.array([[32, 48], [368, 48], [368, 552], [32, 552]], dtype=np.float32)
        distances = []
        for pt in expected:
            d2 = np.sum((anchors - pt) ** 2, axis=1)
            distances.append(np.sqrt(np.min(d2)))
        self.assertLess(float(np.mean(distances)), 20.0)
        self.assertEqual(aligned.shape[:2], (600, 400))

    def test_resolve_zone_centers_uses_connected_components_grid_relaxation(self):
        template = Template(
            name="mcq",
            image_path="",
            width=200,
            height=120,
            anchors=[],
            zones=[
                Zone(
                    id="mcq1",
                    name="mcq",
                    zone_type=ZoneType.MCQ_BLOCK,
                    x=0.0,
                    y=0.0,
                    width=1.0,
                    height=1.0,
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
        actual = np.array([[36, 58], [79, 61], [123, 59], [165, 62]], dtype=np.int32)
        for x, y in actual:
            cv2.circle(binary, (int(x), int(y)), 8, 255, -1)

        centers = self.processor._resolve_zone_centers(binary, template.zones[0], template)
        self.assertEqual(centers.shape, (4, 2))
        self.assertLess(float(np.mean(np.linalg.norm(centers - actual.astype(np.float32), axis=1))), 8.0)

    def test_auto_orient_keeps_original_when_90_degree_score_is_ambiguous(self):
        template = Template(
            name="sheet",
            image_path="",
            width=400,
            height=600,
            anchors=[AnchorPoint(0.08, 0.08), AnchorPoint(0.92, 0.08), AnchorPoint(0.92, 0.92), AnchorPoint(0.08, 0.92)],
            zones=[],
        )
        aligned = np.full((600, 400, 3), 255, dtype=np.uint8)
        aligned_binary = np.zeros((600, 400), dtype=np.uint8)
        with patch.object(self.processor, "_orientation_score", side_effect=[120.0, 135.0, 118.0, 110.0]):
            out_img, out_bin = self.processor._auto_orient(aligned, aligned_binary, template)

        self.assertEqual(out_img.shape[:2], (600, 400))
        self.assertEqual(out_bin.shape[:2], (600, 400))
        self.assertEqual(self.processor._last_alignment_debug["orientation_rotation"], 0)

    def test_reasonable_page_warp_rejects_small_inner_contours(self):
        template = Template(name="sheet", image_path="", width=400, height=600, anchors=[], zones=[])
        inner = np.array([[80, 120], [320, 120], [320, 480], [80, 480]], dtype=np.float32)
        self.assertFalse(self.processor._is_reasonable_page_warp(inner, (600, 400), template))

    def test_correct_perspective_fallback_path_does_not_use_uninitialized_alignment(self):
        template = Template(name="sheet", image_path="", width=400, height=600, anchors=[], zones=[])
        image = np.zeros((600, 400, 3), dtype=np.uint8)
        binary = np.zeros((600, 400), dtype=np.uint8)
        result_stub = type("R", (), {"issues": []})()
        fallback_img = np.zeros_like(image)
        fallback_bin = np.zeros_like(binary)

        with patch.object(self.processor, "_fallback_align_page_contour", return_value=(fallback_img, fallback_bin)), \
            patch.object(self.processor, "_refine_alignment_with_template_anchors", side_effect=lambda img, bin_img, _tpl: (img, bin_img)), \
            patch.object(self.processor, "_auto_orient", side_effect=lambda img, bin_img, _tpl: (img, bin_img)), \
            patch.object(self.processor, "_refine_corner_translation", side_effect=lambda img, bin_img, _tpl: (img, bin_img)):
            aligned, aligned_binary = self.processor.correct_perspective(image, binary, template, result_stub)

        self.assertEqual(aligned.shape, image.shape)
        self.assertEqual(aligned_binary.shape, binary.shape)
        self.assertEqual(result_stub.issues[0].code, "MISSING_ANCHORS")


if __name__ == "__main__":
    unittest.main()
