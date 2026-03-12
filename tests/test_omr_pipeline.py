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


if __name__ == "__main__":
    unittest.main()
