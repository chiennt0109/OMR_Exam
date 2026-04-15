from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.answer_key_importer import import_answer_key


class AnswerKeyImporterTests(unittest.TestCase):
    def test_import_mcq_answer_column_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mcq.csv"
            pd.DataFrame(
                {
                    "Question": [1, 2, 3],
                    "Answer": ["B", "C", "A"],
                }
            ).to_csv(path, index=False)
            package = import_answer_key(path)
            result = package.exam_keys["DEFAULT"]
            self.assertEqual(result.mcq_answers, {1: "B", 2: "C", 3: "A"})

    def test_import_exam_matrix_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "matrix.csv"
            pd.DataFrame(
                {
                    "Question": [1, 2, 1, 2, 1, 2],
                    "0101": ["B", "C", "ĐĐSĐ", "SSĐĐ", "245", "0,61"],
                    "0102": ["A", "D", "SĐĐĐ", "SĐSĐ", "103", "-49,6"],
                }
            ).to_csv(path, index=False)
            package = import_answer_key(path)
            self.assertEqual(package.exam_keys["0101"].mcq_answers, {1: "B", 2: "C"})
            self.assertEqual(package.exam_keys["0102"].mcq_answers, {1: "A", 2: "D"})
            self.assertEqual(package.exam_keys["0101"].true_false_answers[1], {"a": True, "b": True, "c": False, "d": True})
            self.assertEqual(package.exam_keys["0102"].numeric_answers[2], "-49,6")

    def test_import_mcq_matrix_xlsx(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mcq.xlsx"
            pd.DataFrame(
                {
                    "Question": [1, 2, 3],
                    "0101": ["B", "C", "A"],
                    "0102": ["D", "D", "D"],
                }
            ).to_excel(path, index=False)
            package = import_answer_key(path)
            self.assertEqual(package.exam_keys["0101"].mcq_answers, {1: "B", 2: "C", 3: "A"})

    def test_import_numeric_answer_column(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "num.csv"
            pd.DataFrame(
                {
                    "Question": [1, 2, 3],
                    "Answer": ["245", "-103", "0,61"],
                }
            ).to_csv(path, index=False)
            package = import_answer_key(path)
            result = package.exam_keys["DEFAULT"]
            self.assertEqual(result.numeric_answers, {1: "245", 2: "-103", 3: "0,61"})

    def test_import_invalid_value_reports_row(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.csv"
            pd.DataFrame(
                {
                    "Question": [1],
                    "0101": ["INVALID"],
                    "0102": ["A"],
                }
            ).to_csv(path, index=False)
            with self.assertRaises(ImportError) as cm:
                import_answer_key(path)
            self.assertIn("Row 2", str(cm.exception))

    def test_import_invalid_value_can_continue_and_mark_full_credit(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad_continue.csv"
            pd.DataFrame(
                {
                    "Question": [1],
                    "0101": ["EDEE"],
                    "0102": ["A"],
                }
            ).to_csv(path, index=False)
            package = import_answer_key(path, strict=False, award_full_credit_for_invalid=True)
            self.assertTrue(package.warnings)
            self.assertEqual(package.exam_keys["0102"].mcq_answers, {1: "A"})
            self.assertEqual(package.exam_keys["0101"].full_credit_questions.get("MCQ"), [1])

    def test_import_numeric_values_with_four_characters_not_misread_as_tf(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "numeric_len4.csv"
            pd.DataFrame(
                {
                    "Question": [27, 28],
                    "3001": ["22", "4,44"],
                    "3002": ["4,44", "0,36"],
                    "3003": ["0,36", "35,4"],
                    "3004": ["35,4", "4,44"],
                }
            ).to_csv(path, index=False)
            package = import_answer_key(path)
            self.assertEqual(package.exam_keys["3001"].numeric_answers[27], "22")
            self.assertEqual(package.exam_keys["3001"].numeric_answers[28], "4,44")

    def test_import_positional_matrix_without_header_and_reindex_per_section(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "positional.xlsx"
            pd.DataFrame(
                [
                    ["tiêu đề tự do", "", ""],
                    ["stt", "0101", "0102"],
                    [1, "A", "B"],
                    [2, "C", "D"],
                    [3, "ĐSĐS", "SSĐĐ"],
                    [4, "SĐĐS", "ĐĐSS"],
                    [5, "12,5", "13"],
                    [6, "-4", "7,2"],
                ]
            ).to_excel(path, index=False, header=False)
            package = import_answer_key(path)
            key_0101 = package.exam_keys["0101"]
            key_0102 = package.exam_keys["0102"]
            self.assertEqual(key_0101.mcq_answers, {1: "A", 2: "C"})
            self.assertEqual(key_0102.mcq_answers, {1: "B", 2: "D"})
            self.assertEqual(key_0101.true_false_answers[1], {"a": True, "b": False, "c": True, "d": False})
            self.assertEqual(key_0102.true_false_answers[2], {"a": True, "b": True, "c": False, "d": False})
            self.assertEqual(key_0101.numeric_answers, {1: "12,5", 2: "-4"})
            self.assertEqual(key_0102.numeric_answers, {1: "13", 2: "7,2"})


if __name__ == "__main__":
    unittest.main()
