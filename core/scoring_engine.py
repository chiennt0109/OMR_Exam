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
    mcq_correct: int = 0
    tf_correct: int = 0
    numeric_correct: int = 0
    bonus_full_credit_count: int = 0
    bonus_full_credit_points: float = 0.0
    mcq_compare: str = ""
    tf_compare: str = ""
    numeric_compare: str = ""


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
    def _is_countable_tf_key(value: object) -> bool:
        if isinstance(value, dict) and value:
            return any(str(k).lower() in {"a", "b", "c", "d"} for k in value.keys())
        text = str(value or "").strip()
        return text not in {"", "-", "?"}

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            return float(str(value).strip().replace(",", "."))
        except Exception:
            return default

    @staticmethod
    def _to_bool_mark(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().upper()
        if not text:
            return None
        if text in {"1", "T", "TRUE", "D", "Đ"}:
            return True
        if text in {"0", "F", "FALSE", "S"}:
            return False
        return None

    @staticmethod
    def _normalize_numeric_text(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = text.replace(" ", "")
        if text.startswith("+"):
            text = text[1:]
        return text

    def _build_tf_compare_text(self, key_tf: str, marked_tf: str, q_no: int) -> str:
        if not key_tf and not marked_tf:
            return ""
        return f"Q{q_no}:{key_tf or '-'}|{marked_tf or '-'}"

    def _build_mcq_compare_text(self, key_mcq: str, marked_mcq: str, q_no: int) -> str:
        if not key_mcq and not marked_mcq:
            return ""
        return f"Q{q_no}:{key_mcq or '-'}|{marked_mcq or '-'}"

    def _build_numeric_compare_text(self, key_num: str, marked_num: str, q_no: int) -> str:
        if not key_num and not marked_num:
            return ""
        return f"Q{q_no}:{key_num or '-'}|{marked_num or '-'}"

    @staticmethod
    def _exact_answer_lookup(raw_answers: object, q_no: int) -> object:
        if not isinstance(raw_answers, dict):
            return ""
        if q_no in raw_answers:
            return raw_answers[q_no]
        q_text = str(q_no)
        if q_text in raw_answers:
            return raw_answers[q_text]
        return ""

    def _tf_to_canonical_string(self, value: object) -> str:
        if isinstance(value, dict):
            out: list[str] = []
            normalized = {str(k).lower(): v for k, v in value.items()}
            for opt in ["a", "b", "c", "d"]:
                if opt not in normalized:
                    continue
                parsed = self._to_bool_mark(normalized.get(opt))
                if parsed is None:
                    continue
                out.append("Đ" if parsed else "S")
            return "".join(out)

        raw = str(value or "").strip().upper().replace(" ", "")
        if not raw:
            return ""
        chars: list[str] = []
        for ch in raw:
            if ch == "E":
                chars.append("E")
                continue
            parsed = self._to_bool_mark(ch)
            if parsed is None:
                continue
            chars.append("Đ" if parsed else "S")
        return "".join(chars)

    def describe_formula(self, subject_key: SubjectKey, subject_config: dict | None = None) -> str:
        return "Công thức cố định: MCQ đúng × 0.25 + NUMERIC đúng × 0.25 + TF[0→0, 1→0.1, 2→0.25, 3→0.5, 4→1.0]"

    def score(self, omr: OMRResult, subject_key: SubjectKey, student_name: str = "", subject_config: dict | None = None) -> ScoreResult:
        correct = wrong = blank = 0
        score = 0.0
        mcq_correct = tf_correct = numeric_correct = 0
        bonus_full_credit_count = 0
        bonus_full_credit_points = 0.0
        mcq_compare_items: list[str] = []
        tf_compare_items: list[str] = []
        numeric_compare_items: list[str] = []
        tf_points = {0: 0.0, 1: 0.1, 2: 0.25, 3: 0.5, 4: 1.0}

        print("=== OMR INPUT ===")
        print(omr.mcq_answers)
        print(omr.true_false_answers)
        print(omr.numeric_answers)

        for q_no in sorted(int(q) for q in (subject_key.answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            key = str((subject_key.answers or {}).get(q_no, "") or "").strip().upper()
            if not self._is_countable_mcq_key(key):
                continue
            student = str(self._exact_answer_lookup(omr.mcq_answers, q_no) or "").strip().upper()
            print(f"[MCQ] Q{q_no}: {key} | {student}")
            mcq_compare_items.append(self._build_mcq_compare_text(key, student, q_no))
            if key == "E":
                if student != "":
                    correct += 1
                    mcq_correct += 1
                    score += 0.25
                else:
                    blank += 1
            elif student == "":
                blank += 1
            elif student == key:
                correct += 1
                mcq_correct += 1
                score += 0.25
            else:
                wrong += 1

        for q_no in sorted(int(q) for q in (subject_key.true_false_answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            key_value = (subject_key.true_false_answers or {}).get(q_no, {})
            if not self._is_countable_tf_key(key_value):
                continue
            key_tf = self._tf_to_canonical_string(key_value)
            student_tf = self._tf_to_canonical_string(self._exact_answer_lookup(omr.true_false_answers, q_no))
            if not key_tf:
                continue
            print(f"[TF ] Q{q_no}: {key_tf} | {student_tf}")
            tf_compare_items.append(self._build_tf_compare_text(key_tf, student_tf, q_no))
            correct_count = 0
            for key_ch, student_ch in zip(key_tf, student_tf):
                if key_ch == student_ch:
                    correct_count += 1
                elif key_ch == "E":
                    correct_count += 1
            score += tf_points.get(correct_count, 0.0)
            tf_correct += correct_count
            if correct_count == len(key_tf) and len(student_tf) == len(key_tf):
                correct += 1
            elif student_tf == "":
                blank += 1
            else:
                wrong += 1

        for q_no in sorted(int(q) for q in (subject_key.numeric_answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            key = self._normalize_numeric_text((subject_key.numeric_answers or {}).get(q_no, ""))
            if not self._is_countable_numeric_key(key):
                continue
            student = self._normalize_numeric_text(self._exact_answer_lookup(omr.numeric_answers, q_no))
            print(f"[NUM] Q{q_no}: {key} | {student}")
            numeric_compare_items.append(self._build_numeric_compare_text(key, student, q_no))
            if key == "E":
                if student != "":
                    correct += 1
                    numeric_correct += 1
                    score += 0.25
                else:
                    blank += 1
            elif student == "":
                blank += 1
            elif student == key:
                correct += 1
                numeric_correct += 1
                score += 0.25
            else:
                wrong += 1

        print(f"[FINAL] correct={correct}, score={score}")
        return ScoreResult(
            student_id=omr.student_id,
            name=student_name,
            subject=subject_key.subject,
            exam_code=subject_key.exam_code,
            correct=correct,
            wrong=wrong,
            blank=blank,
            score=round(score, 2),
            mcq_correct=mcq_correct,
            tf_correct=tf_correct,
            numeric_correct=numeric_correct,
            bonus_full_credit_count=bonus_full_credit_count,
            bonus_full_credit_points=round(bonus_full_credit_points, 4),
            mcq_compare="; ".join(x for x in mcq_compare_items if x) or "[Không có MCQ trong đáp án hoặc dữ liệu nhận dạng]",
            tf_compare="; ".join(x for x in tf_compare_items if x) or "[Không có TF trong đáp án hoặc dữ liệu nhận dạng]",
            numeric_compare="; ".join(x for x in numeric_compare_items if x) or "[Không có NUMERIC trong đáp án hoặc dữ liệu nhận dạng]",
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
