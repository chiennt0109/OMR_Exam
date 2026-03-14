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
        return any(k in value for k in ["a", "b", "c", "d"])

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            return float(str(value).strip().replace(",", "."))
        except Exception:
            return default

    def _score_profile(self, subject_key: SubjectKey, subject_config: dict | None) -> dict:
        cfg = subject_config if isinstance(subject_config, dict) else {}
        mode = str(cfg.get("score_mode", "") or "").strip() or "Điểm theo phần"
        section_scores = cfg.get("section_scores", {}) if isinstance(cfg.get("section_scores", {}), dict) else {}
        question_scores = cfg.get("question_scores", {}) if isinstance(cfg.get("question_scores", {}), dict) else {}

        mcq_count = sum(1 for _, ans in (subject_key.answers or {}).items() if self._is_countable_mcq_key(ans))
        num_count = sum(1 for _, ans in (subject_key.numeric_answers or {}).items() if self._is_countable_numeric_key(ans))

        mcq_default = subject_key.points_for_question(1)
        num_default = subject_key.points_for_question(1)

        if mode == "Điểm theo câu":
            mcq_per = self._to_float(((question_scores.get("MCQ", {}) or {}).get("per_question", mcq_default)), mcq_default)
            num_per = self._to_float(((question_scores.get("NUMERIC", {}) or {}).get("per_question", num_default)), num_default)
            tf_map_raw = (question_scores.get("TF", {}) or {})
        else:
            mcq_total = self._to_float(((section_scores.get("MCQ", {}) or {}).get("total_points", 0.0)), 0.0)
            num_total = self._to_float(((section_scores.get("NUMERIC", {}) or {}).get("total_points", 0.0)), 0.0)
            mcq_per = (mcq_total / mcq_count) if mcq_count > 0 and mcq_total > 0 else mcq_default
            num_per = (num_total / num_count) if num_count > 0 and num_total > 0 else num_default
            tf_map_raw = ((section_scores.get("TF", {}) or {}).get("rule_per_question", {}) or {})

        tf_points_by_correct = {
            int(k): self._to_float(v, 0.0)
            for k, v in (tf_map_raw.items() if isinstance(tf_map_raw, dict) else [])
            if str(k).strip().isdigit()
        }
        if not tf_points_by_correct:
            tf_points_by_correct = {0: 0.0, 1: 0.1, 2: 0.25, 3: 0.5, 4: 1.0}

        return {
            "mcq_per": mcq_per,
            "num_per": num_per,
            "tf_points": tf_points_by_correct,
        }

    def score(self, omr: OMRResult, subject_key: SubjectKey, student_name: str = "", subject_config: dict | None = None) -> ScoreResult:
        correct = wrong = blank = 0
        score = 0.0
        profile = self._score_profile(subject_key, subject_config)

        for q_no, key_answer in subject_key.answers.items():
            if not self._is_countable_mcq_key(key_answer):
                continue
            marked = (omr.mcq_answers or {}).get(q_no)
            if not marked:
                blank += 1
                continue
            if marked == key_answer:
                correct += 1
                score += profile["mcq_per"]
            else:
                wrong += 1

        for q_no, key_answer in (subject_key.true_false_answers or {}).items():
            if not self._is_countable_tf_key(key_answer):
                continue
            marked = (omr.true_false_answers or {}).get(q_no)
            if not marked:
                blank += 1
                continue

            key_map = {str(k).lower(): bool(v) for k, v in (key_answer or {}).items() if str(k).lower() in {"a", "b", "c", "d"}}
            marked_map = {str(k).lower(): bool(v) for k, v in (marked or {}).items() if str(k).lower() in {"a", "b", "c", "d"}}
            if not key_map:
                continue

            matched = sum(1 for opt, val in key_map.items() if opt in marked_map and marked_map.get(opt) == val)
            score += profile["tf_points"].get(matched, 0.0)
            if matched == len(key_map):
                correct += 1
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
                score += profile["num_per"]
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
            score=round(score, 4),
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
