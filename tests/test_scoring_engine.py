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
        self.assertIn("Q16:ĐĐĐS|ĐĐĐS", row.tf_compare)

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
        self.assertEqual(row.numeric_correct, 3)
        self.assertEqual(row.correct, 3)
        self.assertEqual(row.wrong, 1)
        self.assertAlmostEqual(row.score, 3.0, places=4)

    def test_compare_columns_still_have_debug_when_marked_missing(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0113",
            answers={},
            true_false_answers={16: "ĐĐĐS"},
            numeric_answers={18: "1347"},
        )
        omr = OMRResult(
            image_path="x.png",
            true_false_answers={},
            numeric_answers={},
        )
        cfg = {
            "score_mode": "Điểm theo câu",
            "question_scores": {
                "TF": {"1": 0.1, "2": 0.25, "3": 0.5, "4": 1.0},
                "NUMERIC": {"per_question": 1.0},
            },
        }

        row = self.engine.score(omr, key, subject_config=cfg)
        self.assertIn("Q16:ĐĐĐS|-", row.tf_compare)
        self.assertIn("Q18:1347|-", row.numeric_compare)

    def test_shifted_question_numbers_still_show_marked_compare(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0112",
            answers={},
            true_false_answers={13: "ĐĐĐS", 14: "ĐSĐĐ"},
            numeric_answers={16: "1347", 17: "40"},
        )
        omr = OMRResult(
            image_path="x.png",
            true_false_answers={1: "ĐĐĐS", 2: "ĐSĐĐ"},
            numeric_answers={1: "1347", 2: "40"},
        )
        cfg = {
            "score_mode": "Điểm theo câu",
            "question_scores": {
                "TF": {"1": 0.1, "2": 0.25, "3": 0.5, "4": 1.0},
                "NUMERIC": {"per_question": 1.0},
            },
        }

        row = self.engine.score(omr, key, subject_config=cfg)
        self.assertIn("Q13:ĐĐĐS|ĐĐĐS", row.tf_compare)
        self.assertIn("Q14:ĐSĐĐ|ĐSĐĐ", row.tf_compare)
        self.assertIn("Q16:1347|1347", row.numeric_compare)
        self.assertIn("Q17:40|40", row.numeric_compare)


    def test_mcq_shifted_question_numbers_are_aligned_by_position(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0114",
            answers={13: "A", 14: "B", 15: "C"},
            true_false_answers={},
            numeric_answers={},
        )
        omr = OMRResult(
            image_path="x.png",
            mcq_answers={1: "A", 2: "B", 3: "C"},
        )

        row = self.engine.score(omr, key)
        self.assertEqual(row.correct, 3)
        self.assertEqual(row.wrong, 0)
        self.assertEqual(row.blank, 0)
        self.assertAlmostEqual(row.score, 3.0, places=4)
        self.assertIn("Q13:A|A", row.mcq_compare)
        self.assertIn("Q14:B|B", row.mcq_compare)
        self.assertIn("Q15:C|C", row.mcq_compare)

    def test_aligned_marked_answers_skips_non_numeric_keys(self):
        aligned = self.engine._aligned_marked_answers(
            key_answers={"section": "", "1": "A", 2: "B"},
            marked_answers={1: "A", 2: "B"},
        )
        self.assertEqual(aligned, {1: "A", 2: "B"})

    def test_full_credit_questions_award_points_even_with_wrong_or_blank_answers(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0999",
            answers={1: "A"},
            true_false_answers={2: "ĐĐĐS"},
            numeric_answers={3: "12,5"},
            full_credit_questions={"MCQ": [1], "TF": [2], "NUMERIC": [3]},
        )
        omr = OMRResult(
            image_path="x.png",
            mcq_answers={1: "D"},
            true_false_answers={2: "SSSS"},
            numeric_answers={},
        )
        cfg = {
            "score_mode": "Điểm theo câu",
            "question_scores": {
                "MCQ": {"per_question": 1.0},
                "NUMERIC": {"per_question": 1.0},
                "TF": {"1": 0.1, "2": 0.25, "3": 0.5, "4": 1.0},
            },
        }
        row = self.engine.score(omr, key, subject_config=cfg)
        self.assertEqual(row.correct, 3)
        self.assertEqual(row.wrong, 0)
        self.assertEqual(row.blank, 0)
        self.assertAlmostEqual(row.score, 3.0, places=4)


    def test_answer_string_is_built_by_section_order_with_blank_placeholders(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0211",
            answers={13: "A", 14: "B"},
            true_false_answers={16: "ĐĐĐS"},
            numeric_answers={18: "0,56"},
        )
        omr = OMRResult(
            image_path="x.png",
            mcq_answers={1: "A"},
            true_false_answers={1: {"a": True, "c": False}},
            numeric_answers={1: "0.5"},
        )

        row = self.engine.score(omr, key)
        self.assertEqual(omr.answer_string, "A_ĐS__0,5_")
        self.assertEqual(row.mcq_compare, "Q13:A|A; Q14:B|-")
        self.assertIn("Q16:ĐĐĐS|ĐS--", row.tf_compare)
        self.assertIn("Q18:0,56|0,5", row.numeric_compare)

    def test_prebuilt_answer_string_is_scored_by_fixed_width_slices(self):
        key = SubjectKey(
            subject="Hoa_hoc_11",
            exam_code="0212",
            answers={1: "A", 2: "B"},
            true_false_answers={3: "ĐĐĐS"},
            numeric_answers={4: "12,5"},
        )
        omr = OMRResult(
            image_path="x.png",
            answer_string="ABĐĐĐS12,5",
        )

        row = self.engine.score(omr, key)
        self.assertEqual(row.correct, 4)
        self.assertEqual(row.wrong, 0)
        self.assertEqual(row.blank, 0)
        self.assertAlmostEqual(row.score, 4.0, places=4)


if __name__ == "__main__":
    unittest.main()
