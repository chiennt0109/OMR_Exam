from __future__ import annotations

import unittest

from core.omr_engine import OMRResult
from core.scoring_engine import ScoringEngine
from models.answer_key import SubjectKey


class ScoringEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ScoringEngine()

    def test_tf_string_exact_match(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0111",
            answers={},
            true_false_answers={16: "ĐĐĐS"},
            numeric_answers={},
        )
        omr = OMRResult(
            image_path="x.png",
            true_false_answers={16: "ĐĐĐS"},
        )
        cfg = {
            "score_mode": "Điểm theo câu",
            "question_scores": {
                "TF": {"1": 0.1, "2": 0.25, "3": 0.5, "4": 1.0},
            },
        }

        row = self.engine.score(omr, key, subject_config=cfg)
        self.assertEqual(row.tf_correct, 1)
        self.assertEqual(row.correct, 1)
        self.assertAlmostEqual(row.score, 1.0, places=4)

    def test_tf_partial_string_matching_by_position(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0112",
            answers={},
            true_false_answers={16: "ĐĐĐS"},
            numeric_answers={},
        )
        omr = OMRResult(
            image_path="x.png",
            true_false_answers={16: "ĐSĐS"},
        )
        cfg = {
            "score_mode": "Điểm theo câu",
            "question_scores": {
                "TF": {"1": 0.1, "2": 0.25, "3": 0.5, "4": 1.0},
            },
        }

        row = self.engine.score(omr, key, subject_config=cfg)
        self.assertEqual(row.tf_correct, 0)
        self.assertEqual(row.correct, 0)
        self.assertEqual(row.wrong, 1)
        self.assertAlmostEqual(row.score, 0.5, places=4)

    def test_numeric_string_matching_normalized_separator_only(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0114",
            answers={},
            true_false_answers={},
            numeric_answers={18: "1347", 19: "40", 20: "113", 21: "0,56"},
        )
        omr = OMRResult(
            image_path="x.png",
            numeric_answers={18: "01347", 19: "40.0", 20: "+113", 21: "0.56"},
        )
        cfg = {
            "score_mode": "Điểm theo câu",
            "question_scores": {
                "NUMERIC": {"per_question": 1.0},
            },
        }

        row = self.engine.score(omr, key, subject_config=cfg)
        # String matching with minimal normalization: '+', spaces and decimal separator are normalized.
        self.assertEqual(row.numeric_correct, 2)
        self.assertEqual(row.correct, 2)
        self.assertEqual(row.wrong, 2)
        self.assertAlmostEqual(row.score, 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
