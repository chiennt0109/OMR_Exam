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
            result = import_answer_key(path)
            self.assertEqual(result.mcq_answers, {1: "B", 2: "C", 3: "A"})

    def test_import_mcq_matrix_xlsx(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mcq.xlsx"
            pd.DataFrame(
                {
                    "Question": [1, 2, 3],
                    "A": ["", "", "X"],
                    "B": ["X", "", ""],
                    "C": ["", "X", ""],
                    "D": ["", "", ""],
                }
            ).to_excel(path, index=False)
            result = import_answer_key(path)
            self.assertEqual(result.mcq_answers, {1: "B", 2: "C", 3: "A"})

    def test_import_true_false_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tf.csv"
            pd.DataFrame(
                {
                    "Question": [1],
                    "a": ["T"],
                    "b": ["F"],
                    "c": ["D"],
                    "d": ["S"],
                }
            ).to_csv(path, index=False)
            result = import_answer_key(path)
            self.assertEqual(result.true_false_answers, {1: {"a": True, "b": False, "c": True, "d": False}})

    def test_import_numeric_answer_column(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "num.csv"
            pd.DataFrame(
                {
                    "Question": [1, 2],
                    "Answer": ["245", "103"],
                }
            ).to_csv(path, index=False)
            result = import_answer_key(path)
            self.assertEqual(result.numeric_answers, {1: "245", 2: "103"})

    def test_import_invalid_value_reports_row(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.csv"
            pd.DataFrame(
                {
                    "Question": [1],
                    "Answer": ["Z"],
                }
            ).to_csv(path, index=False)
            with self.assertRaises(ImportError) as cm:
                import_answer_key(path)
            self.assertIn("Row 2", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
