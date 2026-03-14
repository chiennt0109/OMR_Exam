from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from models.answer_key import SubjectKey
from core.omr_engine import OMRResult


@dataclass
class ScoreResult:
    student_id: str
    name: str
    subject: str
    exam_code: str
    correct: int
    wrong: int
    blank: int
    score: float


class ScoringEngine:
    @staticmethod
    def _is_countable_mcq_key(value: str | None) -> bool:
        text = str(value or "").strip()
        return text not in {"", "-", "?"}

    @staticmethod
    def _is_countable_numeric_key(value: str | None) -> bool:
        text = str(value or "").strip()
        return text not in {"", "-", "?"}

    @staticmethod
    def _is_countable_tf_key(value: dict[str, bool] | None) -> bool:
        if not isinstance(value, dict) or not value:
            return False
        # Count only when at least one option is explicitly defined.
        return any(k in value for k in ["a", "b", "c", "d"])

    def score(self, omr: OMRResult, subject_key: SubjectKey, student_name: str = "") -> ScoreResult:
        correct = wrong = blank = 0
        score = 0.0

        for q_no, key_answer in subject_key.answers.items():
            if not self._is_countable_mcq_key(key_answer):
                continue
            marked = (omr.mcq_answers or {}).get(q_no)
            if not marked:
                blank += 1
                continue
            if marked == key_answer:
                correct += 1
                score += subject_key.points_for_question(q_no)
            else:
                wrong += 1

        for q_no, key_answer in (subject_key.true_false_answers or {}).items():
            if not self._is_countable_tf_key(key_answer):
                continue
            marked = (omr.true_false_answers or {}).get(q_no)
            if not marked:
                blank += 1
                continue
            if marked == key_answer:
                correct += 1
                score += subject_key.points_for_question(q_no)
            else:
                wrong += 1

        for q_no, key_answer in (subject_key.numeric_answers or {}).items():
            if not self._is_countable_numeric_key(key_answer):
                continue
            marked = (omr.numeric_answers or {}).get(q_no)
            if marked is None or str(marked).strip() == "":
                blank += 1
                continue
            if str(marked).strip() == str(key_answer).strip():
                correct += 1
                score += subject_key.points_for_question(q_no)
            else:
                wrong += 1

        return ScoreResult(
            student_id=omr.student_id,
            name=student_name,
            subject=subject_key.subject,
            exam_code=subject_key.exam_code,
            correct=correct,
            wrong=wrong,
            blank=blank,
            score=round(score, 2),
        )

    def export_csv(self, rows: list[ScoreResult], output_path: str | Path) -> None:
        path = Path(output_path)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["StudentID", "Name", "Subject", "Score", "Correct", "Wrong", "Blank", "ExamCode"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "StudentID": row.student_id,
                        "Name": row.name,
                        "Subject": row.subject,
                        "Score": row.score,
                        "Correct": row.correct,
                        "Wrong": row.wrong,
                        "Blank": row.blank,
                        "ExamCode": row.exam_code,
                    }
                )

    def export_json(self, rows: list[ScoreResult], output_path: str | Path) -> None:
        Path(output_path).write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    def export_xml(self, rows: list[ScoreResult], output_path: str | Path) -> None:
        root = ET.Element("results")
        for row in rows:
            item = ET.SubElement(root, "result")
            for k, v in asdict(row).items():
                node = ET.SubElement(item, k)
                node.text = str(v)
        ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)

    def export_excel(self, rows: list[ScoreResult], output_path: str | Path) -> None:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "StudentID": r.student_id,
                    "Name": r.name,
                    "Subject": r.subject,
                    "Score": r.score,
                    "Correct": r.correct,
                    "Wrong": r.wrong,
                    "Blank": r.blank,
                    "ExamCode": r.exam_code,
                }
                for r in rows
            ]
        )
        df.to_excel(output_path, index=False)
