import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import time

import cv2
import numpy as np

from core.omr_engine import OMRProcessor, RecognitionContext
from core.template_engine import TemplateEngine
from models.template import AnchorPoint, BubbleGrid, Template, Zone, ZoneType


class OMRPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = OMRProcessor(debug_mode=False)
        self.template_engine = TemplateEngine()

    def test_classify_bubble_thresholds(self):
        self.assertEqual(self.processor.classify_bubble(0.6), "filled")
        self.assertEqual(self.processor.classify_bubble(0.1), "empty")
        self.assertEqual(self.processor.classify_bubble(0.3), "uncertain")

    def test_zone_recognition_priority_reads_answer_blocks_before_ids(self):
        mcq = Zone(id="mcq", name="mcq", zone_type=ZoneType.MCQ_BLOCK, x=0, y=0, width=1, height=1, grid=None)
        tf = Zone(id="tf", name="tf", zone_type=ZoneType.TRUE_FALSE_BLOCK, x=0, y=0, width=1, height=1, grid=None)
        num = Zone(id="num", name="num", zone_type=ZoneType.NUMERIC_BLOCK, x=0, y=0, width=1, height=1, grid=None)
        sid = Zone(id="sid", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0, y=0, width=1, height=1, grid=None)
        exam = Zone(id="exam", name="exam", zone_type=ZoneType.EXAM_CODE_BLOCK, x=0, y=0, width=1, height=1, grid=None)

        ordered = sorted([exam, sid, num, tf, mcq], key=self.processor._zone_recognition_priority)

        self.assertEqual([z.id for z in ordered], ["mcq", "tf", "num", "sid", "exam"])

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

    def test_detect_anchors_caps_candidate_contours_on_noisy_images(self):
        binary = np.zeros((120, 120), dtype=np.uint8)
        contours = [
            np.array([[[idx, 0]], [[idx, 1]], [[idx + 1, 1]], [[idx + 1, 0]]], dtype=np.int32)
            for idx in range(1000)
        ]
        approx_calls = {"count": 0}

        def _approx(cnt, epsilon, closed):
            approx_calls["count"] += 1
            return np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)

        with patch.object(cv2, "findContours", return_value=(contours, None)), \
            patch.object(cv2, "contourArea", return_value=120.0), \
            patch.object(cv2, "arcLength", return_value=44.0), \
            patch.object(cv2, "approxPolyDP", side_effect=_approx), \
            patch.object(cv2, "boundingRect", return_value=(0, 0, 10, 10)), \
            patch.object(cv2, "countNonZero", return_value=90), \
            patch.object(cv2, "moments", return_value={"m00": 1.0, "m10": 8.0, "m01": 8.0}):
            anchors = self.processor.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=10)

        self.assertLessEqual(approx_calls["count"], 200)
        self.assertEqual(len(anchors), 10)

    def test_detect_anchors_stops_immediately_when_time_budget_is_exceeded(self):
        binary = np.zeros((120, 120), dtype=np.uint8)
        contours = [
            np.array([[[idx, 0]], [[idx, 1]], [[idx + 1, 1]], [[idx + 1, 0]]], dtype=np.int32)
            for idx in range(100)
        ]
        self.processor._processing_deadline_monotonic = time.monotonic() - 1.0
        with patch.object(cv2, "findContours", return_value=(contours, None)), \
            patch.object(cv2, "approxPolyDP", side_effect=AssertionError("detect_anchors should stop before polygon approximation when timed out")):
            anchors = self.processor.detect_anchors(binary, use_border_padding=True, relaxed_polygon=True, max_points=10)
        self.processor._processing_deadline_monotonic = None

        self.assertEqual(anchors, [])

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
        expected_x = np.array(grid.bubble_positions, dtype=np.float32)[:, 0]
        expected_x[0::2] += 2.0
        expected_x[1::2] += 4.0
        self.assertTrue(np.allclose(centers[:, 0], expected_x, atol=0.5))
        self.assertLess(float(np.mean(np.abs(centers[:, 1] - shifted[:, 1]))), 2.5)

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

    def test_student_id_zone_edge_columns_allow_larger_shift(self):
        grid = BubbleGrid(rows=4, cols=3, question_start=1, question_count=3, options=[], bubble_positions=[(30 + c * 30, 20 + r * 20) for r in range(4) for c in range(3)])
        template = Template(
            name="sid_edge_shift",
            image_path="",
            width=140,
            height=120,
            anchors=[],
            zones=[Zone(id="sid_edge_shift", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0, y=0, width=1, height=1, grid=grid, metadata={"bubble_radius": 5})],
        )
        binary = np.zeros((120, 140), dtype=np.uint8)
        shifted = np.array(grid.bubble_positions, dtype=np.float32)
        shifted[0::3] += np.array([-5.0, 0.5], dtype=np.float32)
        shifted[2::3] += np.array([5.0, 0.5], dtype=np.float32)
        for x, y in shifted.astype(np.int32):
            cv2.circle(binary, (int(x), int(y)), 5, 255, 1)

        centers = self.processor._resolve_zone_centers(binary, template.zones[0], template)
        self.assertLess(float(np.mean(np.linalg.norm(centers - shifted, axis=1))), 4.0)

    def test_student_id_zone_uses_right_anchor_ruler_before_component_refine(self):
        grid = BubbleGrid(rows=4, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(130 + c * 24, 30 + r * 20) for r in range(4) for c in range(2)])
        anchors = [AnchorPoint(176 / 220, y / 140, f"R{i}") for i, y in enumerate((24, 44, 64, 84, 104, 124), start=1)]
        zone = Zone(id="sid_ruler", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=120 / 220, y=20 / 140, width=48 / 220, height=90 / 140, grid=grid, metadata={"bubble_radius": 5})
        template = Template(name="sid_ruler", image_path="", width=220, height=140, anchors=anchors, zones=[zone])

        binary = np.zeros((140, 220), dtype=np.uint8)
        shifted = np.array(grid.bubble_positions, dtype=np.float32) + np.array([10.0, 2.0], dtype=np.float32)
        for x, y in shifted.astype(np.int32):
            cv2.circle(binary, (int(x), int(y)), 5, 255, 1)

        detected_ruler = [(186.0, 26.0), (186.0, 46.0), (186.0, 66.0), (186.0, 86.0), (186.0, 106.0), (186.0, 126.0)]
        with patch.object(self.processor, "detect_anchors", return_value=detected_ruler):
            centers = self.processor._resolve_zone_centers(binary, zone, template)

        self.assertTrue(np.allclose(centers, np.array(grid.bubble_positions, dtype=np.float32), atol=0.5))

    def test_exam_code_zone_uses_right_anchor_ruler_for_local_shift(self):
        grid = BubbleGrid(rows=4, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(150, 30 + r * 18) for r in range(4)])
        anchors = [AnchorPoint(188 / 240, y / 120, f"DIGIT_ANCHOR_{i:02d}") for i, y in enumerate((20, 38, 56, 74, 92, 110), start=1)]
        zone = Zone(id="exam_ruler", name="exam", zone_type=ZoneType.EXAM_CODE_BLOCK, x=142 / 240, y=18 / 120, width=28 / 240, height=76 / 120, grid=grid, metadata={"bubble_radius": 5})
        template = Template(name="exam_ruler", image_path="", width=240, height=120, anchors=anchors, zones=[zone])

        binary = np.zeros((120, 240), dtype=np.uint8)
        shifted = np.array(grid.bubble_positions, dtype=np.float32) + np.array([8.0, 1.5], dtype=np.float32)
        for x, y in shifted.astype(np.int32):
            cv2.circle(binary, (int(x), int(y)), 5, 255, 1)

        detected_ruler = [(196.0, 21.5), (196.0, 39.5), (196.0, 57.5), (196.0, 75.5), (196.0, 93.5), (196.0, 111.5)]
        with patch.object(self.processor, "detect_anchors", return_value=detected_ruler):
            centers = self.processor._resolve_zone_centers(binary, zone, template)

        self.assertTrue(np.allclose(centers[:, 0], np.array(grid.bubble_positions, dtype=np.float32)[:, 0] + 3.0, atol=0.5))
        self.assertLess(float(np.mean(np.abs(centers[:, 1] - shifted[:, 1]))), 2.0)

    def test_detect_digit_anchor_ruler_finds_actual_positions_near_manual_points(self):
        template = Template(
            name="digit_template",
            image_path="",
            width=240,
            height=140,
            anchors=[
                AnchorPoint(0.80, 0.18, "DIGIT_ANCHOR_01"),
                AnchorPoint(0.80, 0.42, "DIGIT_ANCHOR_02"),
                AnchorPoint(0.80, 0.66, "DIGIT_ANCHOR_03"),
                AnchorPoint(0.80, 0.90, "DIGIT_ANCHOR_04"),
            ],
            zones=[],
        )
        binary = np.zeros((140, 240), dtype=np.uint8)
        for x, y in [(196, 30), (197, 63), (196, 95), (197, 127)]:
            cv2.rectangle(binary, (x - 5, y - 4), (x + 5, y + 4), 255, -1)

        detected = self.processor._detect_digit_anchor_ruler(binary, template)

        self.assertEqual(len(detected), 4)
        self.assertLess(float(np.mean([abs(x - 196.5) for x, _ in detected])), 4.0)
        self.assertLess(float(np.mean([abs(y - tgt) for (_, y), tgt in zip(detected, [30.0, 63.0, 95.0, 127.0])])), 4.0)


    def test_digit_zone_uses_nearest_manual_anchor_cluster_per_zone(self):
        processor = self.processor
        sid_grid = BubbleGrid(rows=4, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(50, 20 + r * 20) for r in range(4)])
        exam_grid = BubbleGrid(rows=4, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(170, 20 + r * 20) for r in range(4)])
        sid_zone = Zone(id="sid_cluster", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=40 / 240, y=10 / 140, width=24 / 240, height=90 / 140, grid=sid_grid, metadata={"bubble_radius": 5})
        exam_zone = Zone(id="exam_cluster", name="exam", zone_type=ZoneType.EXAM_CODE_BLOCK, x=160 / 240, y=10 / 140, width=24 / 240, height=90 / 140, grid=exam_grid, metadata={"bubble_radius": 5})
        template = Template(
            name="dual_digit_clusters",
            image_path="",
            width=240,
            height=140,
            anchors=[],
            zones=[sid_zone, exam_zone],
        )
        template.anchors = [
            AnchorPoint(72 / 240, y / 140, f"DIGIT_ANCHOR_{i:02d}")
            for i, y in enumerate((12, 32, 52, 72, 92), start=1)
        ] + [
            AnchorPoint(196 / 240, y / 140, f"DIGIT_ANCHOR_{i:02d}")
            for i, y in enumerate((18, 38, 58, 78, 98), start=6)
        ]

        sid_guides = processor._get_manual_digit_anchor_points(template, sid_zone)
        exam_guides = processor._get_manual_digit_anchor_points(template, exam_zone)

        self.assertEqual(len(sid_guides), 5)
        self.assertEqual(len(exam_guides), 5)
        self.assertTrue(np.all(sid_guides[:, 0] < 120))
        self.assertTrue(np.all(exam_guides[:, 0] > 120))
    def test_manual_digit_anchor_guides_define_row_centers(self):
        grid = BubbleGrid(rows=4, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(100 + c * 20, 20 + r * 20) for r in range(4) for c in range(2)])
        zone = Zone(id="sid_guides", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0.4, y=0.1, width=0.2, height=0.6, grid=grid, metadata={"bubble_radius": 5})
        template = Template(
            name="digit_guides",
            image_path="",
            width=240,
            height=160,
            anchors=[
                AnchorPoint(0.84, 0.10, "DIGIT_ANCHOR_01"),
                AnchorPoint(0.84, 0.22, "DIGIT_ANCHOR_02"),
                AnchorPoint(0.84, 0.34, "DIGIT_ANCHOR_03"),
                AnchorPoint(0.84, 0.46, "DIGIT_ANCHOR_04"),
                AnchorPoint(0.84, 0.58, "DIGIT_ANCHOR_05"),
            ],
            zones=[zone],
        )
        expected = np.array(grid.bubble_positions, dtype=np.float32)

        binary = np.zeros((160, 240), dtype=np.uint8)
        for x, y in [(202, 18), (203, 38), (202, 58), (203, 78), (202, 98)]:
            cv2.rectangle(binary, (x - 4, y - 4), (x + 4, y + 4), 255, -1)

        guided = self.processor._apply_anchor_ruler_to_digit_zone(binary, expected, zone, template)

        target_centers = [48.0, 68.0, 88.0, 108.0]
        for row_idx, center_y in enumerate(target_centers):
            self.assertAlmostEqual(float(guided[row_idx * 2][1]), float(center_y), places=3)

    def test_manual_digit_anchor_blocker_is_ignored_when_building_row_centers(self):
        grid = BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(120, 20 + r * 12) for r in range(10)])
        zone = Zone(id="sid_blocker", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0.4, y=0.1, width=0.2, height=0.7, grid=grid, metadata={"bubble_radius": 5})
        template = Template(
            name="digit_blocker",
            image_path="",
            width=240,
            height=220,
            anchors=[
                AnchorPoint(0.84, 0.05, "DIGIT_ANCHOR_00"),
                AnchorPoint(0.84, 0.10, "DIGIT_ANCHOR_01"),
                AnchorPoint(0.84, 0.16, "DIGIT_ANCHOR_02"),
                AnchorPoint(0.84, 0.22, "DIGIT_ANCHOR_03"),
                AnchorPoint(0.84, 0.28, "DIGIT_ANCHOR_04"),
                AnchorPoint(0.84, 0.34, "DIGIT_ANCHOR_05"),
                AnchorPoint(0.84, 0.40, "DIGIT_ANCHOR_06"),
                AnchorPoint(0.84, 0.46, "DIGIT_ANCHOR_07"),
                AnchorPoint(0.84, 0.52, "DIGIT_ANCHOR_08"),
                AnchorPoint(0.84, 0.58, "DIGIT_ANCHOR_09"),
                AnchorPoint(0.84, 0.64, "DIGIT_ANCHOR_10"),
            ],
            zones=[zone],
        )
        expected = np.array(grid.bubble_positions, dtype=np.float32)

        guided, debug = self.processor._digit_zone_guidance(np.zeros((220, 240), dtype=np.uint8), expected, zone, template)

        first_center = float(guided[0][1])
        self.assertAlmostEqual(first_center, ((0.10 + 0.13) * 220), places=3)
        self.assertEqual(debug.get("anchor_layout"), "blocker_plus_row_tops")

    def test_manual_digit_anchor_guides_use_positions_2_to_11_as_row_tops(self):
        grid = BubbleGrid(rows=4, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(120, 20 + r * 20) for r in range(4)])
        zone = Zone(id="sid_row_tops", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0.4, y=0.1, width=0.2, height=0.6, grid=grid, metadata={"bubble_radius": 5})
        template = Template(
            name="digit_row_tops",
            image_path="",
            width=240,
            height=160,
            anchors=[
                AnchorPoint(0.84, 0.08, "DIGIT_ANCHOR_01"),
                AnchorPoint(0.84, 0.16, "DIGIT_ANCHOR_02"),
                AnchorPoint(0.84, 0.28, "DIGIT_ANCHOR_03"),
                AnchorPoint(0.84, 0.40, "DIGIT_ANCHOR_04"),
                AnchorPoint(0.84, 0.52, "DIGIT_ANCHOR_05"),
            ],
            zones=[zone],
        )
        expected = np.array(grid.bubble_positions, dtype=np.float32)

        guided = self.processor._apply_anchor_ruler_to_digit_zone(np.zeros((160, 240), dtype=np.uint8), expected, zone, template)

        expected_centers = [(0.16 + 0.06) * 160, (0.28 + 0.06) * 160, (0.40 + 0.06) * 160, (0.52 + 0.06) * 160]
        for row_idx, center_y in enumerate(expected_centers):
            self.assertAlmostEqual(float(guided[row_idx][1]), float(center_y), places=3)

    def test_manual_digit_anchor_row_tops_are_regularized_to_even_spacing(self):
        grid = BubbleGrid(rows=4, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(120, 20 + r * 20) for r in range(4)])
        zone = Zone(id="sid_regularized", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0.4, y=0.1, width=0.2, height=0.6, grid=grid, metadata={"bubble_radius": 5})
        template = Template(
            name="digit_regularized",
            image_path="",
            width=240,
            height=160,
            anchors=[
                AnchorPoint(0.84, 0.08, "DIGIT_ANCHOR_01"),
                AnchorPoint(0.84, 0.16, "DIGIT_ANCHOR_02"),
                AnchorPoint(0.84, 0.29, "DIGIT_ANCHOR_03"),
                AnchorPoint(0.84, 0.39, "DIGIT_ANCHOR_04"),
                AnchorPoint(0.84, 0.53, "DIGIT_ANCHOR_05"),
            ],
            zones=[zone],
        )
        expected = np.array(grid.bubble_positions, dtype=np.float32)

        guided = self.processor._apply_anchor_ruler_to_digit_zone(np.zeros((160, 240), dtype=np.uint8), expected, zone, template)

        spacings = np.diff([float(guided[i][1]) for i in range(4)])
        self.assertLess(float(np.max(spacings) - np.min(spacings)), 1.0)

    def test_digit_zone_guidance_no_longer_affine_warps_digit_grid(self):
        grid = BubbleGrid(rows=4, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(100 + c * 20, 30 + r * 18) for r in range(4) for c in range(2)])
        zone = Zone(id="sid_tilt_combo", name="sid", zone_type=ZoneType.STUDENT_ID_BLOCK, x=0.35, y=0.1, width=0.25, height=0.55, grid=grid, metadata={"bubble_radius": 5})
        template = Template(
            name="tilt_combo",
            image_path="",
            width=240,
            height=160,
            anchors=[
                AnchorPoint(0.05, 0.05, "A1"),
                AnchorPoint(0.95, 0.05, "A2"),
                AnchorPoint(0.95, 0.95, "A3"),
                AnchorPoint(0.05, 0.95, "A4"),
                AnchorPoint(0.84, 0.08, "DIGIT_ANCHOR_01"),
                AnchorPoint(0.84, 0.18, "DIGIT_ANCHOR_02"),
                AnchorPoint(0.84, 0.30, "DIGIT_ANCHOR_03"),
                AnchorPoint(0.84, 0.42, "DIGIT_ANCHOR_04"),
                AnchorPoint(0.84, 0.54, "DIGIT_ANCHOR_05"),
            ],
            zones=[zone],
        )
        expected = np.array(grid.bubble_positions, dtype=np.float32)
        binary = np.zeros((160, 240), dtype=np.uint8)
        detected_corners = [(16.0, 10.0), (225.0, 18.0), (231.0, 150.0), (12.0, 145.0)]
        detected_digit = [(202.0, 16.0), (204.0, 34.0), (206.0, 54.0), (208.0, 74.0), (210.0, 94.0)]

        with patch.object(self.processor, "detect_anchors", return_value=detected_corners), \
            patch.object(self.processor, "_detect_digit_anchor_ruler", return_value=detected_digit):
            guided = self.processor._apply_anchor_ruler_to_digit_zone(binary, expected, zone, template)

        self.assertTrue(np.allclose(guided[:, 0], expected[:, 0]))
        self.assertGreater(float(np.mean(guided[:, 1] - expected[:, 1])), 0.0)

    def test_digit_columns_are_regularized_to_even_spacing_with_tilt(self):
        grid = BubbleGrid(rows=4, cols=4, question_start=1, question_count=4, options=[], bubble_positions=[(90 + c * 22, 30 + r * 20) for r in range(4) for c in range(4)])
        expected = np.array(grid.bubble_positions, dtype=np.float32)
        binary = np.zeros((180, 240), dtype=np.uint8)
        for r in range(4):
            for c in range(4):
                x = 92 + (c * 23) + r
                y = 30 + (r * 20)
                cv2.circle(binary, (x, y), 5, 255, 1)

        centers = self.processor._resolve_column_digit_centers(binary, expected, grid, 5.0)

        for r in range(4):
            row = centers[r * 4:(r + 1) * 4, 0]
            diffs = np.diff(row)
            self.assertLess(float(np.max(diffs) - np.min(diffs)), 1.0)

    def test_digit_columns_keep_template_spacing_while_applying_row_shift(self):
        grid = BubbleGrid(rows=3, cols=4, question_start=1, question_count=4, options=[], bubble_positions=[(80 + c * 24, 30 + r * 24) for r in range(3) for c in range(4)])
        expected = np.array(grid.bubble_positions, dtype=np.float32)
        binary = np.zeros((160, 220), dtype=np.uint8)
        for r in range(3):
            x_shift = 6 + r
            for c in range(4):
                x = int(expected[r * 4 + c][0] + x_shift)
                y = int(expected[r * 4 + c][1])
                cv2.circle(binary, (x, y), 5, 255, 1)

        centers = self.processor._resolve_column_digit_centers(binary, expected, grid, 5.0)

        for r in range(3):
            row_slice = slice(r * 4, (r + 1) * 4)
            self.assertTrue(np.allclose(np.diff(centers[row_slice, 0]), np.diff(expected[row_slice, 0]), atol=1.0))

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

    def test_detect_core_ring_contrast_penalizes_outline_only_bubbles(self):
        binary = np.zeros((120, 120), dtype=np.uint8)
        cv2.circle(binary, (40, 60), 10, 255, -1)
        cv2.circle(binary, (80, 60), 10, 255, 1)
        centers = np.array([[40, 60], [80, 60]], dtype=np.float32)
        scores = self.processor._detect_core_ring_contrast(binary, centers, 10)
        self.assertGreater(scores[0], 0.45)
        self.assertLess(scores[1], 0.10)

    def test_detect_eroded_mark_density_suppresses_outline_only_bubbles(self):
        binary = np.zeros((120, 120), dtype=np.uint8)
        cv2.line(binary, (34, 54), (46, 66), 255, 3)
        cv2.line(binary, (46, 54), (34, 66), 255, 3)
        cv2.circle(binary, (80, 60), 10, 255, 1)
        centers = np.array([[40, 60], [80, 60]], dtype=np.float32)
        scores = self.processor._detect_eroded_mark_density(binary, centers, 10)
        self.assertGreater(scores[0], 0.12)
        self.assertLess(scores[1], 0.05)

    def test_detect_square_mark_density_prefers_cross_mark_for_student_id(self):
        binary = np.zeros((120, 120), dtype=np.uint8)
        cv2.line(binary, (32, 52), (48, 68), 255, 3)
        cv2.line(binary, (48, 52), (32, 68), 255, 3)
        cv2.circle(binary, (80, 60), 10, 255, 1)
        centers = np.array([[40, 60], [80, 60]], dtype=np.float32)
        scores = self.processor._detect_square_mark_density(binary, centers, 10)
        self.assertGreater(scores[0], 0.15)
        self.assertLess(scores[1], 0.12)

    def test_digit_zone_multi_probe_marks_catch_shifted_filled_bubble_better_than_outline(self):
        binary = np.zeros((140, 140), dtype=np.uint8)
        cv2.circle(binary, (47, 60), 9, 255, -1)
        cv2.circle(binary, (100, 60), 9, 255, 1)
        centers = np.array([[40, 60], [100, 60]], dtype=np.float32)
        scores = self.processor._detect_digit_zone_multi_probe_marks(binary, centers, 10)
        self.assertGreater(scores[0], 0.40)
        self.assertLess(scores[1], 0.20)

    def test_digit_zone_peak_window_marks_catch_off_center_blob_better_than_outline(self):
        binary = np.zeros((140, 140), dtype=np.uint8)
        cv2.circle(binary, (48, 58), 7, 255, -1)
        cv2.circle(binary, (100, 60), 9, 255, 1)
        centers = np.array([[40, 60], [100, 60]], dtype=np.float32)
        scores = self.processor._detect_digit_zone_peak_window_marks(binary, centers, 10)
        self.assertGreater(scores[0], 0.45)
        self.assertLess(scores[1], 0.20)

    def test_digit_zone_component_marks_prefer_compact_bubble_over_tall_stroke(self):
        binary = np.zeros((140, 160), dtype=np.uint8)
        cv2.circle(binary, (42, 62), 8, 255, -1)
        cv2.rectangle(binary, (96, 40), (101, 84), 255, -1)
        centers = np.array([[40, 60], [100, 60]], dtype=np.float32)
        scores = self.processor._detect_digit_zone_component_marks(binary, centers, 10)
        self.assertGreater(scores[0], 0.70)
        self.assertLess(scores[1], 0.55)

    def test_exam_code_recognize_block_uses_multi_probe_digit_signal_for_shifted_mark(self):
        template = Template(
            name="exam_multi_probe",
            image_path="",
            width=120,
            height=140,
            anchors=[],
            zones=[
                Zone(
                    id="exam_multi_probe",
                    name="exam",
                    zone_type=ZoneType.EXAM_CODE_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(60, 20 + r * 10) for r in range(10)]),
                    metadata={"bubble_radius": 6},
                )
            ],
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        with patch.object(self.processor, "_resolve_column_digit_centers", return_value=np.array([[60, 20 + r * 10] for r in range(10)], dtype=np.float32)), \
            patch.object(self.processor, "detect_bubbles", return_value=np.array([0.30, 0.24, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08], dtype=np.float32)), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=np.array([0.35, 0.28, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06], dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=np.array([0.88, 0.22, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)):
            self.processor.recognize_block(np.zeros((140, 120), dtype=np.uint8), template.zones[0], template, result_stub)
        self.assertEqual(result_stub.exam_code, "0")

    def test_exam_code_recognize_block_component_signal_suppresses_upper_stroke_bias(self):
        template = Template(
            name="exam_component_bias",
            image_path="",
            width=120,
            height=140,
            anchors=[],
            zones=[
                Zone(
                    id="exam_component_bias",
                    name="exam",
                    zone_type=ZoneType.EXAM_CODE_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(60, 20 + r * 10) for r in range(10)]),
                    metadata={"bubble_radius": 6},
                )
            ],
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": "", "digit_zone_debug": {}})()
        with patch.object(self.processor, "_resolve_column_digit_centers", return_value=np.array([[60, 20 + r * 10] for r in range(10)], dtype=np.float32)), \
            patch.object(self.processor, "detect_bubbles", return_value=np.array([0.56, 0.10, 0.10, 0.10, 0.10, 0.58, 0.10, 0.10, 0.10, 0.10], dtype=np.float32)), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=np.array([0.52, 0.12, 0.12, 0.12, 0.12, 0.54, 0.12, 0.12, 0.12, 0.12], dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=np.array([0.54, 0.08, 0.08, 0.08, 0.08, 0.60, 0.08, 0.08, 0.08, 0.08], dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_peak_window_marks", return_value=np.array([0.52, 0.07, 0.07, 0.07, 0.07, 0.61, 0.07, 0.07, 0.07, 0.07], dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_component_marks", return_value=np.array([0.22, 0.04, 0.04, 0.04, 0.04, 0.92, 0.04, 0.04, 0.04, 0.04], dtype=np.float32)), \
            patch.object(self.processor, "_estimate_local_fill_threshold", return_value=0.45):
            self.processor.recognize_block(np.zeros((140, 120), dtype=np.uint8), template.zones[0], template, result_stub)
        self.assertEqual(result_stub.exam_code, "5")

    def test_exam_code_recognize_block_uses_peak_window_signal_for_off_center_mark(self):
        template = Template(
            name="exam_peak_window",
            image_path="",
            width=120,
            height=140,
            anchors=[],
            zones=[
                Zone(
                    id="exam_peak_window",
                    name="exam",
                    zone_type=ZoneType.EXAM_CODE_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(60, 20 + r * 10) for r in range(10)]),
                    metadata={"bubble_radius": 6},
                )
            ],
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        with patch.object(self.processor, "_resolve_column_digit_centers", return_value=np.array([[60, 20 + r * 10] for r in range(10)], dtype=np.float32)), \
            patch.object(self.processor, "detect_bubbles", return_value=np.array([0.20, 0.19, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07], dtype=np.float32)), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=np.array([0.21, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=np.array([0.26, 0.22, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_peak_window_marks", return_value=np.array([0.92, 0.18, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04], dtype=np.float32)):
            self.processor.recognize_block(np.zeros((140, 120), dtype=np.uint8), template.zones[0], template, result_stub)
        self.assertEqual(result_stub.exam_code, "0")

    def test_exam_code_decode_caps_digit_threshold_when_ui_fill_threshold_is_high(self):
        processor = OMRProcessor(fill_threshold=0.62, debug_mode=False)
        zone = Zone(
            id="exam_threshold_cap",
            name="exam",
            zone_type=ZoneType.EXAM_CODE_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[]),
            metadata={},
        )
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        mat = np.zeros((10, 1), dtype=np.float32)
        mat[2, 0] = 0.53

        digits, confs = processor._decode_column_digits(mat, zone, zone.grid, result_stub)

        self.assertEqual(digits, "2")
        self.assertTrue(confs and confs[0] > 1.0)

    def test_digit_zone_guidance_keeps_template_positions(self):
        expected = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        zone = Zone(
            id="sid_guidance_passthrough",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=2, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[]),
            metadata={},
        )
        template = Template(name="t", image_path="", width=100, height=100, anchors=[], zones=[zone])
        guided, debug = self.processor._digit_zone_guidance(np.zeros((100, 100), dtype=np.uint8), expected, zone, template)
        self.assertTrue(np.array_equal(guided, expected))
        self.assertEqual(debug, {})

    def test_digit_zone_guidance_fits_y_only_from_digit_anchor_ruler(self):
        zone = Zone(
            id="sid_warp",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=4, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(20, 20), (20, 40), (20, 60), (20, 80)]),
            metadata={},
        )
        template = Template(name="t", image_path="", width=120, height=120, anchors=[], zones=[zone])
        expected = np.array(zone.grid.bubble_positions, dtype=np.float32)
        detected = np.array([[90.0, 18.0], [90.0, 42.0], [90.0, 66.0], [90.0, 90.0]], dtype=np.float32)
        template_pts = np.array([[90.0, 20.0], [90.0, 40.0], [90.0, 60.0], [90.0, 80.0]], dtype=np.float32)

        with patch.object(self.processor, "_detect_digit_anchor_ruler", return_value=detected.tolist()), \
            patch.object(self.processor, "_get_manual_digit_anchor_points", return_value=template_pts):
            guided, debug = self.processor._digit_zone_guidance(np.zeros((120, 120), dtype=np.uint8), expected, zone, template)

        self.assertTrue(np.allclose(guided[:, 0], expected[:, 0]))
        self.assertTrue(np.allclose(guided[:, 1], np.array([18.0, 42.0, 66.0, 90.0], dtype=np.float32)))
        self.assertIn("fitted_line", debug)

    def test_resolve_column_digit_centers_caps_x_drift_to_three_pixels(self):
        grid = BubbleGrid(rows=4, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(20 + c * 30, 20 + r * 18) for r in range(4) for c in range(2)])
        expected = np.array(grid.bubble_positions, dtype=np.float32)
        binary = np.zeros((120, 120), dtype=np.uint8)
        offsets = [np.array([7.0, 0.0], dtype=np.float32)] * 4 + [np.array([-8.0, 0.0], dtype=np.float32)] * 4
        with patch.object(self.processor, "_find_local_component_offset", side_effect=offsets):
            centers = self.processor._resolve_column_digit_centers(binary, expected, grid, 5.0)

        self.assertTrue(np.allclose(centers[0::2, 0], expected[0::2, 0] + 3.0))
        self.assertTrue(np.allclose(centers[1::2, 0], expected[1::2, 0] - 3.0))
        self.assertTrue(np.allclose(centers[:, 1], expected[:, 1]))

    def test_exam_code_decode_rejects_ambiguous_last_column(self):
        zone = Zone(
            id="exam_edge_fallback",
            name="exam",
            zone_type=ZoneType.EXAM_CODE_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=4, question_start=1, question_count=4, options=[], bubble_positions=[]),
            metadata={},
        )
        mat = np.array(
            [
                [0.10, 0.12, 0.11, 0.18],
                [0.11, 0.13, 0.12, 0.17],
                [0.12, 0.14, 0.13, 0.16],
                [0.70, 0.15, 0.14, 0.15],
                [0.13, 0.73, 0.15, 0.14],
                [0.12, 0.14, 0.75, 0.13],
                [0.11, 0.13, 0.12, 0.48],
                [0.10, 0.12, 0.11, 0.45],
                [0.09, 0.11, 0.10, 0.12],
                [0.08, 0.10, 0.09, 0.11],
            ],
            dtype=np.float32,
        )
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        digits, _ = self.processor._decode_column_digits(mat, zone, zone.grid, result_stub)
        self.assertEqual(digits, "3456")
        self.assertIn("EXAM_CODE_BLOCK column 4: fallback accepted", result_stub.recognition_errors)

    def test_sample_digit_grid_scores_filled_cell_above_background(self):
        centers = np.array([(20 + c * 20, 20 + r * 18) for r in range(10) for c in range(1)], dtype=np.float32)
        binary = np.zeros((220, 80), dtype=np.uint8)
        cv2.circle(binary, (20, 20 + 7 * 18), 6, 255, -1)
        mat = self.processor._sample_digit_grid(binary, centers, rows=10, cols=1, bubble_radius=6.0)
        self.assertEqual(int(np.argmax(mat[:, 0])), 7)
        self.assertGreater(float(mat[7, 0]), float(np.max(np.delete(mat[:, 0], 7))))

    def test_refit_digit_grid_from_single_clear_point_shifts_equal_grid(self):
        guided = np.array([(20 + c * 30, 20 + r * 18) for r in range(10) for c in range(2)], dtype=np.float32)
        binary = np.zeros((240, 180), dtype=np.uint8)
        cv2.circle(binary, (24, 20 + 2 * 18), 6, 255, -1)
        refined, debug = self.processor._refit_digit_grid_from_clear_points(
            binary,
            guided,
            BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            bubble_radius=6.0,
        )
        self.assertTrue(debug.get("grid_fit_applied"))
        self.assertAlmostEqual(float(refined[2 * 2 + 0, 0]), 24.0, delta=1.5)
        self.assertAlmostEqual(float(refined[2 * 2 + 1, 0] - refined[2 * 2 + 0, 0]), 30.0, delta=1.0)
        self.assertAlmostEqual(float(refined[(3 * 2) + 0, 1] - refined[(2 * 2) + 0, 1]), 18.0, delta=1.0)

    def test_read_digit_grid_sampling_prefers_guided_centers_over_bbox_grid(self):
        zone = Zone(
            id="sid_guided_sampling",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            metadata={"bubble_radius": 6},
        )
        binary = np.zeros((220, 140), dtype=np.uint8)
        guided = np.array([(20 + c * 30, 20 + r * 18) for r in range(10) for c in range(2)], dtype=np.float32)
        cv2.circle(binary, (20, 20 + 2 * 18), 6, 255, -1)
        cv2.circle(binary, (50, 20 + 8 * 18), 6, 255, -1)
        source = self.processor._preprocess_digit_sampling(binary)
        digits, mat, _ = self.processor._read_digit_grid_sampling(
            source,
            bbox=(0, 0, 140, 220),
            num_cols=2,
            num_rows=10,
            threshold=max(0.40, self.processor.fill_threshold * 0.7),
            centers=guided,
        )
        self.assertEqual(digits, [2, 8])
        self.assertEqual(int(np.argmax(mat[:, 0])), 2)
        self.assertEqual(int(np.argmax(mat[:, 1])), 8)

    def test_decode_sampled_digit_grid_accepts_single_filled_row(self):
        zone = Zone(
            id="sid_sampled",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            metadata={},
        )
        mat = np.zeros((10, 2), dtype=np.float32)
        mat[3, 0] = 0.62
        mat[7, 1] = 0.66
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        digits, confs = self.processor._decode_sampled_digit_grid(mat, zone, zone.grid, result_stub)
        self.assertEqual(digits, "37")
        self.assertEqual(confs, [1.0, 1.0])

    def test_recognize_block_prefers_direct_identifier_decode_before_sampling(self):
        template = Template(
            name="sid_direct_first",
            image_path="",
            width=120,
            height=220,
            anchors=[],
            zones=[
                Zone(
                    id="sid_direct_first",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(40, 20 + r * 18) for r in range(10)]),
                    metadata={"bubble_radius": 6},
                )
            ],
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": "", "digit_zone_debug": {}})()
        with patch.object(
            self.processor,
            "_decode_identifier_zone_from_centers",
            return_value=("7", [1.4], np.array([[40, 20 + r * 18] for r in range(10)], dtype=np.float32), np.zeros((10, 1), dtype=np.float32), {"direct_scores": [[0.0]]}),
        ), patch.object(
            self.processor,
            "_read_digit_zone_with_offset_fallback",
            side_effect=AssertionError("sampling should not run when direct decode is valid"),
        ):
            self.processor.recognize_block(np.zeros((220, 120), dtype=np.uint8), template.zones[0], template, result_stub)
        self.assertEqual(result_stub.student_id, "7")
        self.assertEqual(result_stub.digit_zone_debug["sid_direct_first"].get("recognition_path"), "direct")

    def test_recognize_block_reads_student_id_from_template_grid_sampling(self):
        template = Template(
            name="sid_grid_sample",
            image_path="",
            width=140,
            height=220,
            anchors=[],
            zones=[
                Zone(
                    id="sid_grid_sample",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(20 + c * 30, 20 + r * 18) for r in range(10) for c in range(2)]),
                    metadata={"bubble_radius": 6},
                )
            ],
        )
        binary = np.zeros((220, 140), dtype=np.uint8)
        cv2.circle(binary, (20, 20 + 2 * 18), 6, 255, -1)
        cv2.circle(binary, (50, 20 + 8 * 18), 6, 255, -1)
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.student_id, "28")

    def test_digit_model_expected_points_apply_exam_offsets_only(self):
        exam_zone = Zone(
            id="exam_model_offsets",
            name="exam",
            zone_type=ZoneType.EXAM_CODE_BLOCK,
            x=160 / 240,
            y=10 / 140,
            width=24 / 240,
            height=90 / 140,
            grid=BubbleGrid(rows=4, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(170 + c * 12, 20 + r * 20) for r in range(4) for c in range(2)]),
            metadata={"bubble_radius": 5},
        )
        template = Template(
            name="exam_digit_model",
            image_path="",
            width=240,
            height=140,
            anchors=[
                AnchorPoint(196 / 240, y / 140, f"DIGIT_ANCHOR_{i:02d}")
                for i, y in enumerate((18, 38, 58, 78, 98), start=1)
            ],
            zones=[exam_zone],
            metadata={"digit_model": {"exam_col_spacing_scale": 0.8, "exam_row_spacing_scale": 1.1, "offset_exam": [6.0, -2.0]}},
        )

        points, debug = self.processor._digit_model_expected_points(template, exam_zone)

        self.assertEqual(points.shape, (8, 2))
        self.assertTrue(debug.get("digit_model_applied"))
        self.assertGreater(float(points[0, 0]), 170.0)
        self.assertLess(float(points[0, 1]), 28.0)

    def test_recognize_block_uses_exam_digit_model_positions_before_center_refine(self):
        exam_zone = Zone(
            id="exam_model_guided",
            name="exam",
            zone_type=ZoneType.EXAM_CODE_BLOCK,
            x=160 / 240,
            y=10 / 140,
            width=24 / 240,
            height=90 / 140,
            grid=BubbleGrid(rows=4, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(170 + c * 12, 20 + r * 20) for r in range(4) for c in range(2)]),
            metadata={"bubble_radius": 5},
        )
        template = Template(
            name="exam_model_guided",
            image_path="",
            width=240,
            height=140,
            anchors=[
                AnchorPoint(196 / 240, y / 140, f"DIGIT_ANCHOR_{i:02d}")
                for i, y in enumerate((18, 38, 58, 78, 98), start=1)
            ],
            zones=[exam_zone],
            metadata={"digit_model": {"exam_col_spacing_scale": 0.8, "exam_row_spacing_scale": 1.1, "offset_exam": [6.0, -2.0]}},
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": "", "digit_zone_debug": {}})()

        modeled_points, _ = self.processor._digit_model_expected_points(template, exam_zone)
        with patch.object(self.processor, "_resolve_column_digit_centers", side_effect=lambda binary, expected, grid, bubble_radius: expected) as resolve_mock, \
            patch.object(self.processor, "_refit_digit_grid_from_clear_points", side_effect=lambda binary, guided, grid, bubble_radius: (guided, {})), \
            patch.object(self.processor, "_decode_identifier_zone_from_centers", return_value=("23", [1.5, 1.5], modeled_points, np.zeros((10, 2), dtype=np.float32), {"direct_scores": [[0.0, 0.0]]})):
            self.processor.recognize_block(np.zeros((140, 240), dtype=np.uint8), exam_zone, template, result_stub)

        guided_arg = resolve_mock.call_args.args[1]
        self.assertTrue(np.allclose(guided_arg, modeled_points))
        self.assertEqual(result_stub.exam_code, "23")

    def test_recognize_block_skips_exam_sampling_fallback_when_digit_model_is_active(self):
        exam_zone = Zone(
            id="exam_model_no_sampling",
            name="exam",
            zone_type=ZoneType.EXAM_CODE_BLOCK,
            x=160 / 240,
            y=10 / 140,
            width=24 / 240,
            height=90 / 140,
            grid=BubbleGrid(rows=4, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(170 + c * 12, 20 + r * 20) for r in range(4) for c in range(2)]),
            metadata={"bubble_radius": 5},
        )
        template = Template(
            name="exam_model_no_sampling",
            image_path="",
            width=240,
            height=140,
            anchors=[
                AnchorPoint(196 / 240, y / 140, f"DIGIT_ANCHOR_{i:02d}")
                for i, y in enumerate((18, 38, 58, 78, 98), start=1)
            ],
            zones=[exam_zone],
            metadata={"digit_model": {"exam_col_spacing_scale": 0.8, "exam_row_spacing_scale": 1.1, "offset_exam": [6.0, -2.0]}},
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": "", "digit_zone_debug": {}})()

        modeled_points, _ = self.processor._digit_model_expected_points(template, exam_zone)
        with patch.object(self.processor, "_resolve_column_digit_centers", side_effect=lambda binary, expected, grid, bubble_radius: expected), \
            patch.object(self.processor, "_refit_digit_grid_from_clear_points", side_effect=lambda binary, guided, grid, bubble_radius: (guided, {})), \
            patch.object(self.processor, "_decode_identifier_zone_from_centers", return_value=("", [0.0, 0.0], modeled_points, np.zeros((10, 2), dtype=np.float32), {"direct_scores": [[0.0, 0.0]]})), \
            patch.object(self.processor, "_read_digit_zone_with_offset_fallback", side_effect=AssertionError("exam digit-model path should skip sampling fallback")):
            self.processor.recognize_block(np.zeros((140, 240), dtype=np.uint8), exam_zone, template, result_stub)

        self.assertEqual(result_stub.exam_code, "-")

    def test_recognize_block_rejects_exam_code_when_column_has_multiple_sampled_marks(self):
        template = Template(
            name="exam_grid_sample",
            image_path="",
            width=120,
            height=220,
            anchors=[],
            zones=[
                Zone(
                    id="exam_grid_sample",
                    name="exam",
                    zone_type=ZoneType.EXAM_CODE_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(40, 20 + r * 18) for r in range(10)]),
                    metadata={"bubble_radius": 6},
                )
            ],
        )
        binary = np.zeros((220, 120), dtype=np.uint8)
        cv2.circle(binary, (40, 20 + 1 * 18), 6, 255, -1)
        cv2.circle(binary, (40, 20 + 2 * 18), 6, 255, -1)
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        self.processor.recognize_block(binary, template.zones[0], template, result_stub)
        self.assertEqual(result_stub.exam_code, "")
        self.assertIn("EXAM_CODE_BLOCK: LOW_CONFIDENCE", result_stub.recognition_errors)

    def test_resolve_column_digit_centers_returns_refined_array_without_name_error(self):
        grid = BubbleGrid(
            rows=4,
            cols=2,
            question_start=1,
            question_count=2,
            options=[],
            bubble_positions=[(40 + c * 30, 20 + r * 18) for r in range(4) for c in range(2)],
        )
        expected = np.array(grid.bubble_positions, dtype=np.float32)
        binary = np.zeros((120, 120), dtype=np.uint8)
        centers = self.processor._resolve_column_digit_centers(binary, expected, grid, 5.0)
        self.assertEqual(centers.shape, expected.shape)

    def test_cluster_digit_columns_groups_by_x_before_sorting_y(self):
        centers = np.array(
            [
                [82.0, 48.0],
                [24.0, 30.0],
                [54.0, 20.0],
                [84.0, 18.0],
                [22.0, 10.0],
                [52.0, 40.0],
            ],
            dtype=np.float32,
        )
        values = np.array([0.82, 0.21, 0.63, 0.91, 0.71, 0.73], dtype=np.float32)
        mat = self.processor._cluster_digit_columns(centers, values, rows=2, cols=3, fill_value=0.0)
        expected = np.array(
            [
                [0.71, 0.63, 0.91],
                [0.21, 0.73, 0.82],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(mat, expected))

    def test_cluster_digit_columns_reassigns_points_to_stabilized_centers(self):
        centers = np.array(
            [
                [20.0, 10.0],
                [21.5, 20.0],
                [22.0, 30.0],
                [48.0, 10.0],
                [49.0, 20.0],
                [78.0, 10.0],
                [79.0, 20.0],
                [80.0, 30.0],
            ],
            dtype=np.float32,
        )
        values = np.array([0.91, 0.81, 0.71, 0.62, 0.52, 0.93, 0.83, 0.73], dtype=np.float32)
        mat = self.processor._cluster_digit_columns(centers, values, rows=3, cols=3, fill_value=0.0)
        self.assertTrue(np.allclose(mat[:, 0], np.array([0.91, 0.81, 0.71], dtype=np.float32)))
        self.assertGreater(mat[0, 1], 0.0)
        self.assertTrue(np.allclose(mat[:, 2], np.array([0.93, 0.83, 0.73], dtype=np.float32)))

    def test_detect_digit_bubble_centers_uses_local_components_per_bubble(self):
        centers = np.array([[20.0, 20.0], [50.0, 20.0], [20.0, 40.0], [50.0, 40.0]], dtype=np.float32)
        offsets = [
            np.array([1.5, -0.5], dtype=np.float32),
            None,
            np.array([2.0, 1.0], dtype=np.float32),
            np.array([-1.0, 0.5], dtype=np.float32),
        ]
        with patch.object(self.processor, "_find_local_component_offset", side_effect=offsets):
            detected = self.processor._detect_digit_bubble_centers(np.zeros((80, 80), dtype=np.uint8), centers, 5.0)

        expected = np.array([[21.5, 19.5], [50.0, 20.0], [22.0, 41.0], [49.0, 40.5]], dtype=np.float32)
        self.assertTrue(np.allclose(detected, expected))

    def test_recognize_block_clusters_digit_columns_under_skewed_x_positions(self):
        template = Template(
            name="exam_clustered",
            image_path="",
            width=120,
            height=120,
            anchors=[],
            zones=[
                Zone(
                    id="exam_clustered",
                    name="exam",
                    zone_type=ZoneType.EXAM_CODE_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(20 + c * 30, 10 + r * 8) for r in range(10) for c in range(2)]),
                    metadata={"bubble_radius": 5},
                )
            ],
        )
        skewed_centers = np.array(
            [[18.0, 10.0], [47.0, 11.0], [19.0, 18.0], [48.0, 19.0], [20.0, 26.0], [49.0, 27.0], [21.0, 34.0], [50.0, 35.0], [22.0, 42.0], [51.0, 43.0],
             [23.0, 50.0], [52.0, 51.0], [24.0, 58.0], [53.0, 59.0], [25.0, 66.0], [54.0, 67.0], [26.0, 74.0], [55.0, 75.0], [27.0, 82.0], [56.0, 83.0]],
            dtype=np.float32,
        )
        ratios = np.array(
            [0.12, 0.11, 0.13, 0.12, 0.88, 0.10, 0.11, 0.84, 0.10, 0.12, 0.09, 0.11, 0.08, 0.10, 0.07, 0.09, 0.06, 0.08, 0.05, 0.07],
            dtype=np.float32,
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        with patch.object(self.processor, "_resolve_column_digit_centers", return_value=skewed_centers), \
            patch.object(self.processor, "detect_bubbles", return_value=ratios), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=ratios), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=ratios), \
            patch.object(self.processor, "_estimate_local_fill_threshold", return_value=0.45):
            self.processor.recognize_block(np.zeros((120, 120), dtype=np.uint8), template.zones[0], template, result_stub)

        self.assertEqual(result_stub.exam_code, "23")

    def test_recognize_block_uses_detected_digit_bubble_centers_before_grouping(self):
        template = Template(
            name="sid_detected_centers",
            image_path="",
            width=120,
            height=120,
            anchors=[],
            zones=[
                Zone(
                    id="sid_detected_centers",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[(40, 12 + r * 8) for r in range(10)]),
                    metadata={"bubble_radius": 5},
                )
            ],
        )
        original_centers = np.array(template.zones[0].grid.bubble_positions, dtype=np.float32)
        shifted_centers = original_centers.copy()
        shifted_centers[:, 0] += 2.0
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        with patch.object(self.processor, "_resolve_column_digit_centers", return_value=original_centers), \
            patch.object(self.processor, "_detect_digit_bubble_centers", return_value=shifted_centers) as detect_centers, \
            patch.object(self.processor, "detect_bubbles", return_value=np.array([0.11, 0.12, 0.82, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04], dtype=np.float32)), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=np.array([0.10, 0.11, 0.88, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03], dtype=np.float32)), \
            patch.object(self.processor, "_detect_square_mark_density", return_value=np.array([0.09, 0.10, 0.86, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02], dtype=np.float32)), \
            patch.object(self.processor, "_detect_eroded_mark_density", return_value=np.array([0.08, 0.09, 0.84, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01], dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=np.array([0.12, 0.13, 0.90, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05], dtype=np.float32)), \
            patch.object(self.processor, "_estimate_local_fill_threshold", return_value=0.45):
            self.processor.recognize_block(np.zeros((120, 120), dtype=np.uint8), template.zones[0], template, result_stub)

        detect_centers.assert_called_once()
        self.assertEqual(result_stub.student_id, "")
        self.assertIn("STUDENT_ID_BLOCK: invalid repeated-digit pattern", result_stub.recognition_errors)

    def test_recognize_block_keeps_template_x_and_uses_digit_guidance_y(self):
        template = Template(
            name="sid_guided_block",
            image_path="sheet.png",
            width=100,
            height=100,
            anchors=[],
            zones=[
                Zone(
                    id="sid_guided_block",
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
        result_stub = type(
            "R",
            (),
            {
                "mcq_answers": {},
                "recognition_errors": [],
                "confidence_scores": {},
                "true_false_answers": {},
                "numeric_answers": {},
                "student_id": "",
                "exam_code": "",
                "image_path": "sheet.png",
                "aligned_image": np.zeros((100, 100, 3), dtype=np.uint8),
            },
        )()
        guided_centers = np.array([[50, 12 + r * 8] for r in range(10)], dtype=np.float32)
        with patch.object(self.processor, "_resolve_column_digit_centers", return_value=guided_centers), \
            patch.object(self.processor, "detect_bubbles", return_value=np.array([0.80, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], dtype=np.float32)) as detect_bubbles_mock, \
            patch.object(self.processor, "_detect_center_core_marks", return_value=np.array([0.90, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)), \
            patch.object(self.processor, "_detect_square_mark_density", return_value=np.array([0.80, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)), \
            patch.object(self.processor, "_detect_eroded_mark_density", return_value=np.array([0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=np.array([0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)):
            self.processor.recognize_block(np.zeros((100, 100), dtype=np.uint8), template.zones[0], template, result_stub)

        self.assertEqual(result_stub.student_id, "0")
        self.assertEqual(detect_bubbles_mock.call_args.args[1].tolist(), guided_centers.tolist())

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

            with patch.object(self.processor, "_load_image_normalized_to_200_dpi", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), "")), \
                patch.object(self.processor, "_correct_rotation", side_effect=lambda x: x), \
                patch.object(self.processor, "_preprocess", return_value={"binary": np.zeros((100, 200), dtype=np.uint8)}), \
                patch.object(self.processor, "correct_perspective", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((100, 200), dtype=np.uint8))), \
                patch.object(self.processor, "detect_anchors", return_value=[]):
                editor_res = self.processor.recognize_sheet(str(img_path), template)
                batch_res = self.processor.process_batch([str(img_path)], template)[0]

        self.assertEqual(editor_res.mcq_answers, batch_res.mcq_answers)
        self.assertEqual(editor_res.true_false_answers, batch_res.true_false_answers)
        self.assertEqual(editor_res.numeric_answers, batch_res.numeric_answers)

    def test_process_batch_reuses_template_instance_for_speed(self):
        template = Template(name="t", image_path="", width=200, height=100, anchors=[], zones=[], metadata={"batch_workers": 1})
        fake_result = type("FakeResult", (), {"image_path": "x.png"})()
        with patch.object(self.processor, "run_recognition_test", return_value=fake_result) as run_mock:
            self.processor.process_batch(["a.png", "b.png"], template)
        self.assertEqual(run_mock.call_count, 2)
        self.assertIs(run_mock.call_args_list[0].args[1], template)
        self.assertIs(run_mock.call_args_list[1].args[1], template)

    def test_process_batch_parallel_keeps_input_order(self):
        template = Template(name="t", image_path="", width=200, height=100, anchors=[], zones=[], metadata={"batch_workers": 2})

        class FakeWorker:
            def run_recognition_test(self, image_path, _template, _context):
                if str(image_path).endswith("a.png"):
                    time.sleep(0.02)
                return type("R", (), {"image_path": str(image_path)})()

        with patch.object(self.processor, "_make_batch_worker", side_effect=[FakeWorker(), FakeWorker()]):
            out = self.processor.process_batch(["a.png", "b.png"], template)
        self.assertEqual([getattr(x, "image_path", "") for x in out], ["a.png", "b.png"])

    def test_recognize_sheet_keeps_template_dimensions_stable(self):
        template = Template(name="t", image_path="", width=200, height=100, anchors=[], zones=[])
        original_size = (template.width, template.height)
        with patch.object(self.processor, "_load_image_normalized_to_200_dpi", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), "")), \
            patch.object(self.processor, "_correct_rotation", side_effect=lambda x: x), \
            patch.object(self.processor, "_preprocess", return_value={"binary": np.zeros((100, 200), dtype=np.uint8)}), \
            patch.object(self.processor, "correct_perspective", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((100, 200), dtype=np.uint8))), \
            patch.object(self.processor, "detect_anchors", return_value=[]):
            self.processor.recognize_sheet("x.png", template)
        self.assertEqual((template.width, template.height), original_size)

    def test_load_image_normalized_to_200_dpi_does_not_create_extra_files(self):
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "sheet.png"
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

            class _FakeImage:
                info = {"dpi": (300, 300)}
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    return False

            with patch("PIL.Image.open", return_value=_FakeImage()):
                loaded, msg = self.processor._load_image_normalized_to_200_dpi(str(img_path))

            self.assertIsNotNone(loaded)
            self.assertEqual(getattr(loaded, "shape", (0, 0))[1], 66)
            self.assertIn("Normalized to 200 DPI", msg)
            self.assertFalse((Path(td) / "sheet_200dpi.png").exists())

    def test_fast_200dpi_mode_skips_dpi_normalization_and_rotation(self):
        template = Template(name="t", image_path="", width=200, height=100, anchors=[], zones=[], metadata={"fast_200dpi_mode": True})
        with patch("cv2.imread", return_value=np.zeros((100, 200, 3), dtype=np.uint8)), \
            patch.object(self.processor, "_load_image_normalized_to_200_dpi", side_effect=AssertionError("fast mode should bypass DPI normalization")), \
            patch.object(self.processor, "_correct_rotation", side_effect=AssertionError("fast mode should skip rotation by default")), \
            patch.object(self.processor, "_preprocess_fast", return_value={"binary": np.zeros((100, 200), dtype=np.uint8)}), \
            patch.object(self.processor, "correct_perspective", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((100, 200), dtype=np.uint8))), \
            patch.object(self.processor, "detect_anchors", return_value=[]):
            self.processor.recognize_sheet("x.png", template)

    def test_fast_200dpi_mode_can_reenable_rotation(self):
        template = Template(
            name="t",
            image_path="",
            width=200,
            height=100,
            anchors=[],
            zones=[],
            metadata={"fast_200dpi_mode": True, "skip_rotation_in_fast_mode": False},
        )
        with patch("cv2.imread", return_value=np.zeros((100, 200, 3), dtype=np.uint8)), \
            patch.object(self.processor, "_correct_rotation", side_effect=lambda x: x) as rot_mock, \
            patch.object(self.processor, "_preprocess_fast", return_value={"binary": np.zeros((100, 200), dtype=np.uint8)}), \
            patch.object(self.processor, "correct_perspective", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((100, 200), dtype=np.uint8))), \
            patch.object(self.processor, "detect_anchors", return_value=[]):
            self.processor.recognize_sheet("x.png", template)
        self.assertTrue(rot_mock.called)

    def test_fast_200dpi_direct_match_bypasses_perspective_correction(self):
        template = Template(
            name="t",
            image_path="",
            width=200,
            height=100,
            anchors=[],
            zones=[],
            metadata={"fast_200dpi_mode": True, "fast_direct_match": True},
        )
        with patch("cv2.imread", return_value=np.zeros((100, 200, 3), dtype=np.uint8)), \
            patch.object(self.processor, "_preprocess_fast", return_value={"binary": np.zeros((100, 200), dtype=np.uint8)}), \
            patch.object(self.processor, "correct_perspective", side_effect=AssertionError("fast_direct_match should bypass perspective correction")), \
            patch.object(self.processor, "detect_anchors", return_value=[]):
            self.processor.recognize_sheet("x.png", template)

    def test_process_batch_writes_per_file_timing_log(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "timing.tsv"
            template = Template(name="t", image_path="", width=200, height=100, anchors=[], zones=[], metadata={"batch_workers": 1, "batch_timing_log_path": str(log_path)})
            fake_result = type("FakeResult", (), {"image_path": "x.png"})()
            with patch.object(self.processor, "run_recognition_test", return_value=fake_result):
                self.processor.process_batch(["a.png", "b.png", "c.png"], template)
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("idx\tfile\tseconds", text)
            self.assertIn("estimated_500_files_seconds", text)
            self.assertIn("target_under_120_seconds_for_500", text)

    def test_batch_context_skips_expensive_diagnostics_collection(self):
        template = Template(name="t", image_path="", width=200, height=100, anchors=[], zones=[])
        context = RecognitionContext()
        context.collect_diagnostics = False
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "sheet.png"
            cv2.imwrite(str(img_path), np.zeros((100, 200, 3), dtype=np.uint8))

            with patch.object(self.processor, "_load_image_normalized_to_200_dpi", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), "")), \
                patch.object(self.processor, "_correct_rotation", side_effect=lambda x: x), \
                patch.object(self.processor, "_preprocess", return_value={"binary": np.zeros((100, 200), dtype=np.uint8)}), \
                patch.object(self.processor, "correct_perspective", return_value=(np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((100, 200), dtype=np.uint8))), \
                patch.object(self.processor, "extract_bubble_states", side_effect=AssertionError("batch/lightweight context should skip diagnostics extraction")):
                res = self.processor.recognize_sheet(str(img_path), template, context)

        self.assertEqual(getattr(res, "bubble_states_by_zone", {}), {})

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

    def test_mcq_recognize_block_uses_core_boost_for_x_mark_answer(self):
        template = Template(
            name="mcq_core_boost",
            image_path="",
            width=200,
            height=120,
            anchors=[],
            zones=[
                Zone(
                    id="mcq_core_boost",
                    name="mcq",
                    zone_type=ZoneType.MCQ_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=1, cols=4, question_start=1, question_count=1, options=["A", "B", "C", "D"], bubble_positions=[(40, 60), (80, 60), (120, 60), (160, 60)]),
                    metadata={"bubble_radius": 10},
                )
            ],
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        with patch.object(
            self.processor,
            "_resolve_zone_centers",
            return_value=np.array([(40, 60), (80, 60), (120, 60), (160, 60)], dtype=np.float32),
        ), patch.object(
            self.processor,
            "detect_bubbles",
            return_value=np.array([0.22, 0.26, 0.41, 0.20], dtype=np.float32),
        ), patch.object(
            self.processor,
            "_detect_center_core_marks",
            return_value=np.array([0.14, 0.18, 0.92, 0.12], dtype=np.float32),
        ), patch.object(
            self.processor,
            "_detect_eroded_mark_density",
            return_value=np.array([0.08, 0.09, 0.74, 0.07], dtype=np.float32),
        ), patch.object(
            self.processor,
            "_estimate_local_fill_threshold",
            side_effect=[0.62, 0.62, 0.62, 0.62],
        ):
            self.processor.recognize_block(np.zeros((120, 200), dtype=np.uint8), template.zones[0], template, result_stub)
        self.assertEqual(result_stub.mcq_answers.get(1), "C")

    def test_mcq_best_fallback_keeps_strongest_row_answer_below_strict_threshold(self):
        zone = Zone(
            id="mcq_best_fallback",
            name="mcq",
            zone_type=ZoneType.MCQ_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=1, cols=4, question_start=1, question_count=1, options=["A", "B", "C", "D"], bubble_positions=[]),
            metadata={},
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}})()
        best_idx, confidence, reason = self.processor._pick_best_mcq_option(np.array([0.29, 0.31, 0.56, 0.28], dtype=np.float32), 0.62)
        self.assertEqual(best_idx, 2)
        self.assertGreater(confidence, 0.20)
        self.assertEqual(reason, "max_pick")

    def test_mcq_dominant_fallback_accepts_single_best_answer_even_when_two_choices_cross_threshold(self):
        best_idx, confidence, reason = self.processor._pick_best_mcq_option(np.array([0.14, 0.71, 0.92, 0.15], dtype=np.float32), 0.62)
        self.assertEqual(best_idx, 2)
        self.assertGreater(confidence, 0.18)
        self.assertEqual(reason, "max_pick")

    def test_mcq_dominant_fallback_accepts_much_darker_choice_when_second_is_borderline(self):
        best_idx, confidence, reason = self.processor._pick_best_mcq_option(np.array([0.18, 0.63, 0.92, 0.14], dtype=np.float32), 0.62)
        self.assertEqual(best_idx, 2)
        self.assertGreater(confidence, 0.20)
        self.assertEqual(reason, "max_pick")

    def test_mcq_best_fallback_rejects_multiple_filled_choices(self):
        best_idx, confidence, reason = self.processor._pick_best_mcq_option(np.array([0.64, 0.66, 0.18, 0.17], dtype=np.float32), 0.62)
        self.assertEqual(best_idx, 1)
        self.assertGreater(confidence, 0.0)
        self.assertEqual(reason, "max_pick")

    def test_mcq_row_max_fallback_ignores_boosted_secondary_noise_without_core_support(self):
        best_idx, confidence, reason = self.processor._pick_best_mcq_option(
            np.array([0.18, 0.69, 0.91, 0.16], dtype=np.float32),
            0.62,
            row_raw_scores=np.array([0.10, 0.24, 0.86, 0.09], dtype=np.float32),
            row_core_scores=np.array([0.08, 0.22, 0.88, 0.07], dtype=np.float32),
            row_eroded_scores=np.array([0.09, 0.26, 0.82, 0.08], dtype=np.float32),
        )
        self.assertEqual(best_idx, 2)
        self.assertGreater(confidence, 0.20)
        self.assertEqual(reason, "max_pick")

    def test_mcq_row_max_fallback_keeps_darker_choice_when_second_mark_is_not_95_percent_similar(self):
        best_idx, confidence, reason = self.processor._pick_best_mcq_option(
            np.array([0.18, 0.74, 0.92, 0.16], dtype=np.float32),
            0.62,
            row_raw_scores=np.array([0.10, 0.70, 0.88, 0.09], dtype=np.float32),
            row_core_scores=np.array([0.08, 0.68, 0.90, 0.07], dtype=np.float32),
            row_eroded_scores=np.array([0.09, 0.66, 0.84, 0.08], dtype=np.float32),
        )
        self.assertEqual(best_idx, 2)
        self.assertGreater(confidence, 0.15)
        self.assertEqual(reason, "max_pick")

    def test_mcq_row_max_fallback_still_marks_multiple_when_two_choices_are_95_percent_similar(self):
        best_idx, confidence, reason = self.processor._pick_best_mcq_option(
            np.array([0.18, 0.88, 0.91, 0.16], dtype=np.float32),
            0.62,
            row_raw_scores=np.array([0.10, 0.84, 0.88, 0.09], dtype=np.float32),
            row_core_scores=np.array([0.08, 0.85, 0.89, 0.07], dtype=np.float32),
            row_eroded_scores=np.array([0.09, 0.86, 0.87, 0.08], dtype=np.float32),
        )
        self.assertEqual(best_idx, 2)
        self.assertGreater(confidence, 0.0)
        self.assertEqual(reason, "max_pick")

    def test_mcq_equal_top_scores_are_treated_as_multiple_equal(self):
        best_idx, confidence, reason = self.processor._pick_best_mcq_option(
            np.array([0.20, 0.80, 0.80, 0.10], dtype=np.float32),
            0.62,
        )
        self.assertIsNone(best_idx)
        self.assertEqual(confidence, 0.0)
        self.assertEqual(reason, "multiple_equal")

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

    def test_student_id_decode_selects_single_max_digit(self):
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

    def test_student_id_decode_rejects_small_top_second_gap(self):
        zone = Zone(
            id="sid_small_gap",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=1, question_start=1, question_count=1, options=[], bubble_positions=[]),
            metadata={},
        )
        mat = np.array([[0.57], [0.18], [0.52], [0.17], [0.16], [0.15], [0.14], [0.13], [0.12], [0.11]], dtype=np.float32)
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        digits, _ = self.processor._decode_column_digits(mat, zone, zone.grid, result_stub)
        self.assertEqual(digits, "0")
        self.assertIn("STUDENT_ID_BLOCK column 1: fallback accepted", result_stub.recognition_errors)

    def test_student_id_decode_rejects_ambiguous_column_without_guessing(self):
        zone = Zone(
            id="sid_promote",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            metadata={},
        )
        mat = np.array(
            [
                [0.58, 0.18],
                [0.20, 0.17],
                [0.52, 0.16],
                [0.16, 0.15],
                [0.15, 0.14],
                [0.14, 0.60],
                [0.13, 0.18],
                [0.12, 0.17],
                [0.11, 0.16],
                [0.10, 0.15],
            ],
            dtype=np.float32,
        )
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        digits, _ = self.processor._decode_column_digits(mat, zone, zone.grid, result_stub)
        self.assertEqual(digits, "05")
        self.assertIn("STUDENT_ID_BLOCK column 1: fallback accepted", result_stub.recognition_errors)

    def test_student_id_decode_rejects_ambiguous_column_before_confidence(self):
        zone = Zone(
            id="sid_edge",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            metadata={},
        )
        mat = np.array(
            [
                [0.56, 0.15],
                [0.18, 0.14],
                [0.53, 0.13],
                [0.17, 0.12],
                [0.16, 0.11],
                [0.15, 0.10],
                [0.14, 0.09],
                [0.13, 0.08],
                [0.12, 0.07],
                [0.11, 0.55],
            ],
            dtype=np.float32,
        )
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        digits, _ = self.processor._decode_column_digits(mat, zone, zone.grid, result_stub)
        self.assertEqual(digits, "09")
        self.assertIn("STUDENT_ID_BLOCK column 1: fallback accepted", result_stub.recognition_errors)

    def test_digit_decode_uses_column_normalization_to_overcome_global_bias(self):
        zone = Zone(
            id="sid_norm",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            metadata={},
        )
        mat = np.array(
            [
                [0.82, 0.42],
                [0.20, 0.18],
                [0.19, 0.17],
                [0.18, 0.16],
                [0.17, 0.15],
                [0.16, 0.14],
                [0.15, 0.13],
                [0.14, 0.12],
                [0.13, 0.11],
                [0.12, 0.10],
            ],
            dtype=np.float32,
        )
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        digits, confs = self.processor._decode_column_digits(mat, zone, zone.grid, result_stub)
        self.assertEqual(digits, "00")
        self.assertGreaterEqual(min(confs), 0.5)

    def test_identifier_anchor_axis_reads_digits_left_to_right(self):
        zone = Zone(
            id="sid_axis",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            metadata={},
        )
        template = Template(name="sid_axis_tpl", image_path="", width=200, height=160, anchors=[], zones=[zone])
        centers = np.array([(40.0, 20.0 + (r * 10.0)) for r in range(10)] + [(80.0, 20.0 + (r * 10.0)) for r in range(10)], dtype=np.float32)
        scores = np.zeros((20,), dtype=np.float32)
        scores[3] = 0.95
        scores[10 + 7] = 0.93
        with patch.object(self.processor, "detect_bubbles", return_value=scores), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=np.zeros((20,), dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_component_marks", return_value=np.zeros((20,), dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_anchor_ruler", return_value=[(20.0, 10.0), (20.0, 110.0)]):
            value, _, debug = self.processor._decode_identifier_by_anchor_axis(np.zeros((160, 200), dtype=np.uint8), zone, template, centers, 6)
        self.assertEqual(value, "37")
        self.assertEqual(debug.get("axis_mode"), "anchor_ruler")

    def test_identifier_decode_fast_path_skips_heavy_component_and_multi_probe_scoring(self):
        zone = Zone(
            id="sid_fast",
            name="exam",
            zone_type=ZoneType.EXAM_CODE_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            metadata={},
        )
        centers = np.array([(30.0, 20.0 + (r * 8.0)) for r in range(10)] + [(50.0, 20.0 + (r * 8.0)) for r in range(10)], dtype=np.float32)
        ratios = np.zeros((20,), dtype=np.float32)
        ratios[2] = 0.95
        ratios[10 + 7] = 0.93
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        with patch.object(self.processor, "_detect_digit_bubble_centers", return_value=centers), \
            patch.object(self.processor, "detect_bubbles", return_value=ratios), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=ratios), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", side_effect=AssertionError("fast path should skip multi-probe")), \
            patch.object(self.processor, "_detect_digit_zone_component_marks", side_effect=AssertionError("fast path should skip component")):
            value, confs, _, _, debug = self.processor._decode_identifier_zone_from_centers(
                np.zeros((140, 120), dtype=np.uint8),
                zone,
                result_stub,
                centers,
                5,
            )
        self.assertEqual(value, "27")
        self.assertGreaterEqual(min(confs), 0.55)
        self.assertTrue(bool(debug.get("fast_path_used")))

    def test_student_id_decode_does_not_use_fast_path(self):
        zone = Zone(
            id="sid_no_fast",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[]),
            metadata={},
        )
        centers = np.array([(30.0, 20.0 + (r * 8.0)) for r in range(10)] + [(50.0, 20.0 + (r * 8.0)) for r in range(10)], dtype=np.float32)
        ratios = np.zeros((20,), dtype=np.float32)
        ratios[2] = 0.95
        ratios[10 + 7] = 0.93
        result_stub = type("R", (), {"recognition_errors": [], "confidence_scores": {}})()
        with patch.object(self.processor, "_detect_digit_bubble_centers", return_value=centers), \
            patch.object(self.processor, "detect_bubbles", return_value=ratios), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=ratios), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=np.zeros((20,), dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_component_marks", return_value=np.zeros((20,), dtype=np.float32)), \
            patch.object(self.processor, "_detect_square_mark_density", return_value=np.zeros((20,), dtype=np.float32)), \
            patch.object(self.processor, "_detect_eroded_mark_density", return_value=np.zeros((20,), dtype=np.float32)), \
            patch.object(self.processor, "_estimate_local_fill_threshold", return_value=0.45):
            _, _, _, _, debug = self.processor._decode_identifier_zone_from_centers(
                np.zeros((140, 120), dtype=np.uint8),
                zone,
                result_stub,
                centers,
                5,
            )
        self.assertFalse(bool(debug.get("fast_path_used")))

    def test_finalize_identifier_enforces_fixed_lengths_for_student_id_and_exam_code(self):
        sid_zone = Zone(
            id="sid_len",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=12, question_start=1, question_count=12, options=[], bubble_positions=[]),
            metadata={},
        )
        exam_zone = Zone(
            id="exam_len",
            name="exam",
            zone_type=ZoneType.EXAM_CODE_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=12, question_start=1, question_count=12, options=[], bubble_positions=[]),
            metadata={},
        )
        sid_result = type("R", (), {"recognition_errors": []})()
        sid_value, _ = self.processor._finalize_identifier_value(sid_zone, "student_id", value="123456789012", confs=[1.0] * 12, result=sid_result)
        self.assertEqual(sid_value, "12345678")

        exam_result = type("R", (), {"recognition_errors": []})()
        exam_value, _ = self.processor._finalize_identifier_value(exam_zone, "exam_code", value="567890", confs=[1.0] * 6, result=exam_result)
        self.assertEqual(exam_value, "5678")

        short_result = type("R", (), {"recognition_errors": []})()
        short_value, _ = self.processor._finalize_identifier_value(sid_zone, "student_id", value="1234", confs=[1.0] * 4, result=short_result)
        self.assertEqual(short_value, "")
        self.assertTrue(any("Lỗi SBD" in err for err in short_result.recognition_errors))

    def test_exam_code_model_points_are_adjusted_by_anchor_distance_ratio(self):
        zone = Zone(
            id="exam_ratio",
            name="exam",
            zone_type=ZoneType.EXAM_CODE_BLOCK,
            x=120,
            y=20,
            width=60,
            height=100,
            grid=BubbleGrid(rows=10, cols=4, question_start=1, question_count=4, options=[], bubble_positions=[]),
            metadata={},
        )
        template = Template(name="exam_ratio_tpl", image_path="", width=240, height=160, anchors=[], zones=[zone])
        modeled_points = np.array([[120.0, 40.0], [150.0, 70.0], [180.0, 100.0]], dtype=np.float32)
        manual = np.array([(210.0, 20.0), (210.0, 120.0)], dtype=np.float32)
        detected = np.array([(210.0, 10.0), (210.0, 140.0)], dtype=np.float32)
        with patch.object(self.processor, "_get_manual_digit_anchor_points", return_value=manual), \
            patch.object(self.processor, "_detect_digit_anchor_ruler", return_value=detected.tolist()):
            adjusted, debug = self.processor._adjust_identifier_points_by_anchor_distance(
                np.zeros((160, 240), dtype=np.uint8),
                template,
                zone,
                modeled_points,
            )
        self.assertTrue(bool(debug.get("identifier_anchor_distance_ratio_applied")))
        self.assertGreater(float(debug.get("identifier_anchor_row_ratio", 0.0)), 1.0)
        self.assertEqual(adjusted.shape, modeled_points.shape)
        self.assertFalse(np.allclose(adjusted, modeled_points))

    def test_student_id_recognize_block_applies_anchor_distance_ratio_adjustment(self):
        centers = np.array([(30.0 + (c * 20.0), 20.0 + (r * 8.0)) for r in range(10) for c in range(8)], dtype=np.float32)
        zone = Zone(
            id="sid_ratio",
            name="sid",
            zone_type=ZoneType.STUDENT_ID_BLOCK,
            x=0,
            y=0,
            width=1,
            height=1,
            grid=BubbleGrid(rows=10, cols=8, question_start=1, question_count=8, options=[], bubble_positions=[tuple(pt) for pt in centers]),
            metadata={"bubble_radius": 5},
        )
        template = Template(name="sid_ratio_tpl", image_path="", width=220, height=140, anchors=[], zones=[zone])
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": "", "digit_zone_debug": {}})()

        with patch.object(self.processor, "_digit_zone_guidance", return_value=(centers, {})), \
            patch.object(self.processor, "_adjust_identifier_points_by_anchor_distance", return_value=(centers, {"identifier_anchor_distance_ratio_applied": True})) as patched_adjust, \
            patch.object(self.processor, "_resolve_column_digit_centers", return_value=centers), \
            patch.object(self.processor, "_refit_digit_grid_from_clear_points", return_value=(centers, {})), \
            patch.object(self.processor, "_decode_identifier_zone_from_centers", return_value=("00012029", [1.0] * 8, centers, np.zeros((10, 8), dtype=np.float32), {})):
            self.processor.recognize_block(np.zeros((140, 220), dtype=np.uint8), zone, template, result_stub)

        self.assertTrue(patched_adjust.called)

    def test_recognize_block_invalidates_student_id_when_any_digit_is_ambiguous(self):
        template = Template(
            name="sid_invalid",
            image_path="",
            width=100,
            height=100,
            anchors=[],
            zones=[
                Zone(
                    id="sid_invalid",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(20 + c * 20, 15 + r * 8) for r in range(10) for c in range(2)]),
                    metadata={"bubble_radius": 5},
                )
            ],
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        with patch.object(self.processor, "_resolve_column_digit_centers", return_value=np.array(template.zones[0].grid.bubble_positions, dtype=np.float32)), \
            patch.object(self.processor, "detect_bubbles", return_value=np.array([0.20] * 20, dtype=np.float32)), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=np.array([0.20] * 20, dtype=np.float32)), \
            patch.object(self.processor, "_detect_square_mark_density", return_value=np.array([0.20] * 20, dtype=np.float32)), \
            patch.object(self.processor, "_detect_eroded_mark_density", return_value=np.array([0.20] * 20, dtype=np.float32)), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=np.array([0.20] * 20, dtype=np.float32)), \
            patch.object(self.processor, "_estimate_local_fill_threshold", return_value=0.45):
            self.processor.recognize_block(np.zeros((120, 80), dtype=np.uint8), template.zones[0], template, result_stub)

        self.assertEqual(result_stub.student_id, "")
        self.assertIn("STUDENT_ID_BLOCK: invalid length or ambiguous digit sequence", result_stub.recognition_errors)

    def test_recognize_block_rejects_single_missing_digit_in_low_confidence_mode(self):
        template = Template(
            name="sid_soft_mode",
            image_path="",
            width=100,
            height=100,
            anchors=[],
            zones=[
                Zone(
                    id="sid_soft_mode",
                    name="sid",
                    zone_type=ZoneType.STUDENT_ID_BLOCK,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    grid=BubbleGrid(rows=10, cols=2, question_start=1, question_count=2, options=[], bubble_positions=[(20 + c * 20, 15 + r * 8) for r in range(10) for c in range(2)]),
                    metadata={"bubble_radius": 5},
                )
            ],
        )
        result_stub = type("R", (), {"mcq_answers": {}, "recognition_errors": [], "confidence_scores": {}, "true_false_answers": {}, "numeric_answers": {}, "student_id": "", "exam_code": ""})()
        ratios = np.array([0.20, 0.82, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float32)
        with patch.object(self.processor, "_resolve_column_digit_centers", return_value=np.array(template.zones[0].grid.bubble_positions, dtype=np.float32)), \
            patch.object(self.processor, "detect_bubbles", return_value=ratios), \
            patch.object(self.processor, "_detect_center_core_marks", return_value=ratios), \
            patch.object(self.processor, "_detect_square_mark_density", return_value=ratios), \
            patch.object(self.processor, "_detect_eroded_mark_density", return_value=ratios), \
            patch.object(self.processor, "_detect_digit_zone_multi_probe_marks", return_value=ratios), \
            patch.object(self.processor, "_estimate_local_fill_threshold", return_value=0.45):
            self.processor.recognize_block(np.zeros((120, 80), dtype=np.uint8), template.zones[0], template, result_stub)

        self.assertEqual(result_stub.student_id, "")
        self.assertIn("STUDENT_ID_BLOCK: LOW_CONFIDENCE", result_stub.recognition_errors)

    def test_student_id_recognize_block_rejects_when_global_confidence_is_too_low(self):
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
        self.assertEqual(result_stub.student_id, "")
        self.assertIn("STUDENT_ID_BLOCK: LOW_CONFIDENCE", result_stub.recognition_errors)

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

    def test_affine_anchor_refine_improves_mild_sheet_skew(self):
        template = Template(
            name="sheet_affine",
            image_path="",
            width=400,
            height=600,
            anchors=[AnchorPoint(0.08, 0.08), AnchorPoint(0.92, 0.08), AnchorPoint(0.92, 0.92), AnchorPoint(0.08, 0.92)],
            zones=[],
        )
        aligned = np.full((600, 400, 3), 255, dtype=np.uint8)
        aligned_binary = np.zeros((600, 400), dtype=np.uint8)
        expected = np.array([[32, 48], [368, 48], [368, 552], [32, 552]], dtype=np.float32)
        skewed = np.array([[18, 54], [356, 40], [381, 548], [44, 560]], dtype=np.float32)
        for x, y in skewed.astype(np.int32):
            cv2.rectangle(aligned_binary, (x - 13, y - 13), (x + 13, y + 13), 255, -1)
            cv2.rectangle(aligned, (x - 13, y - 13), (x + 13, y + 13), (0, 0, 0), -1)

        refined_img, refined_bin = self.processor._refine_alignment_with_affine_anchors(aligned, aligned_binary, template)
        refined_anchors = np.array(self.processor.detect_anchors(refined_bin, max_points=20), dtype=np.float32)
        distances = []
        for pt in expected:
            d2 = np.sum((refined_anchors - pt) ** 2, axis=1)
            distances.append(np.sqrt(np.min(d2)))

        self.assertLess(float(np.mean(distances)), 12.0)
        self.assertIn("affine_refine", self.processor._last_alignment_debug)

    def test_correct_perspective_auto_mode_chooses_best_alignment_candidate(self):
        template = Template(
            name="sheet_auto_best",
            image_path="",
            width=400,
            height=600,
            anchors=[AnchorPoint(0.08, 0.08), AnchorPoint(0.92, 0.08), AnchorPoint(0.92, 0.92), AnchorPoint(0.08, 0.92)],
            zones=[],
        )
        image = np.zeros((600, 400, 3), dtype=np.uint8)
        binary = np.zeros((600, 400), dtype=np.uint8)
        result_stub = type("R", (), {"issues": []})()
        legacy_img = np.full_like(image, 11)
        border_img = np.full_like(image, 22)
        hybrid_img = np.full_like(image, 33)
        legacy_bin = np.full((600, 400), 11, dtype=np.uint8)
        border_bin = np.full((600, 400), 22, dtype=np.uint8)
        hybrid_bin = np.full((600, 400), 33, dtype=np.uint8)

        attempts = {
            "border": (border_img, border_bin),
            "hybrid": (hybrid_img, hybrid_bin),
            "legacy": (legacy_img, legacy_bin),
        }
        scores = {11: 90.0, 22: 150.0, 33: 120.0}
        with patch.object(self.processor, "_template_has_one_side_anchor_ruler", return_value=False),             patch.object(self.processor, "_template_has_border_anchors", return_value=True),             patch.object(self.processor, "_try_anchor_alignment", side_effect=lambda _img, _bin, _tpl, candidate: attempts.get(candidate)),             patch.object(self.processor, "_refine_alignment_with_template_anchors", side_effect=lambda img, bin_img, _tpl: (img, bin_img)),             patch.object(self.processor, "_auto_orient", side_effect=lambda img, bin_img, _tpl: (img, bin_img)),             patch.object(self.processor, "_refine_corner_translation", side_effect=lambda img, bin_img, _tpl: (img, bin_img)),             patch.object(self.processor, "_refine_alignment_with_affine_anchors", side_effect=lambda img, bin_img, _tpl: (img, bin_img)),             patch.object(self.processor, "_orientation_score", side_effect=lambda bin_img, _tpl: scores[int(bin_img[0, 0])]):
            aligned, aligned_binary = self.processor.correct_perspective(image, binary, template, result_stub)

        self.assertEqual(int(aligned[0, 0, 0]), 22)
        self.assertEqual(int(aligned_binary[0, 0]), 22)
        self.assertEqual(self.processor._last_alignment_debug["alignment_mode"], "border")
        self.assertEqual(self.processor._last_alignment_debug["alignment_score"], 150.0)

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
