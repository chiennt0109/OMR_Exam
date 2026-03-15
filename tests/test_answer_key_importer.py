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


if __name__ == "__main__":
    unittest.main()
