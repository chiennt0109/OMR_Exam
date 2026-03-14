from __future__ import annotations

import unittest

from core.omr_engine import OMRResult
from core.scoring_engine import ScoringEngine
from models.answer_key import SubjectKey


class ScoringEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ScoringEngine()

    def test_tf_string_boolean_tokens_are_parsed_correctly(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0111",
            answers={},
            true_false_answers={16: {"a": True, "b": True, "c": True, "d": False}},
            numeric_answers={},
        )
        omr = OMRResult(
            image_path="x.png",
            true_false_answers={16: {"a": "Đ", "b": "TRUE", "c": "1", "d": "S"}},
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

    def test_numeric_normalization_comma_dot_and_leading_zero(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0114",
            answers={},
            true_false_answers={},
            numeric_answers={18: "1347", 19: "40", 20: "113", 21: "0,56"},
        )
        omr = OMRResult(
            image_path="x.png",
            numeric_answers={18: "01347", 19: "40.0", 20: "+113", 21: "0.560"},
        )
        cfg = {
            "score_mode": "Điểm theo câu",
            "question_scores": {
                "NUMERIC": {"per_question": 1.0},
            },
        }

        row = self.engine.score(omr, key, subject_config=cfg)
        self.assertEqual(row.numeric_correct, 4)
        self.assertEqual(row.correct, 4)
        self.assertAlmostEqual(row.score, 4.0, places=4)


if __name__ == "__main__":
    unittest.main()
