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
    def _aligned_marked_answers(key_answers: dict, marked_answers: dict) -> dict[int, object]:
        key_numbers = [int(q) for q in key_answers.keys() if str(q).strip().lstrip("-").isdigit()]
        marked_numbers = [int(q) for q in marked_answers.keys() if str(q).strip().lstrip("-").isdigit()]
        key_numbers.sort()
        marked_numbers.sort()
        aligned: dict[int, object] = {}
        for idx, key_q in enumerate(key_numbers):
            if key_q in marked_answers:
                aligned[key_q] = marked_answers[key_q]
            elif idx < len(marked_numbers):
                aligned[key_q] = marked_answers[marked_numbers[idx]]
        return aligned

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
        return text.replace(".", ",")

    @staticmethod
    def _strict_mcq_match(key: str, student: str) -> bool:
        key_text = str(key or "")
        student_text = str(student or "")[:len(key_text)]
        return len(student_text) == len(key_text) and all(a == b for a, b in zip(key_text, student_text))

    @staticmethod
    def _strict_tf_match(key: str, student: str) -> tuple[int, bool]:
        key_text = str(key or "")
        student_text = str(student or "")[:len(key_text)]
        correct = sum(1 for a, b in zip(key_text, student_text) if a == b or a == "E")
        return correct, len(student_text) == len(key_text) and correct == len(key_text)

    @staticmethod
    def _strict_numeric_match(key: str, student: str, delimiter: str = "|") -> bool:
        key_parts = [x for x in str(key or "").split(delimiter)]
        student_parts = [x for x in str(student or "").split(delimiter)][:len(key_parts)]
        if len(student_parts) != len(key_parts):
            return False
        return all(a == b for a, b in zip(key_parts, student_parts))

    @staticmethod
    def _slice_answer_string(answer_string: str, start: int, width: int) -> str:
        if width <= 0:
            return ""
        text = str(answer_string or "")
        chunk = text[start:start + width]
        if len(chunk) < width:
            chunk = chunk + ("_" * (width - len(chunk)))
        return chunk

    def _answer_string_for_scoring(self, omr: OMRResult, subject_key: SubjectKey) -> str:
        existing = str(getattr(omr, "answer_string", "") or "")
        if existing:
            return existing

        def _sorted_items(raw_answers: object) -> list[tuple[int, object]]:
            if not isinstance(raw_answers, dict):
                return []
            out: list[tuple[int, object]] = []
            for key, value in raw_answers.items():
                if str(key).strip().lstrip("-").isdigit():
                    out.append((int(key), value))
            out.sort(key=lambda item: item[0])
            return out

        parts: list[str] = []
        mcq_items = _sorted_items(omr.mcq_answers)
        mcq_idx = 0
        for q_no in sorted(int(q) for q in (subject_key.answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            value = self._exact_answer_lookup(omr.mcq_answers, q_no)
            if value == "" and mcq_idx < len(mcq_items):
                value = mcq_items[mcq_idx][1]
                mcq_idx += 1
            elif value != "":
                mcq_idx += 1
            token = str(value or "").strip().upper()[:1]
            parts.append(token or "_")

        tf_items = _sorted_items(omr.true_false_answers)
        tf_idx = 0
        for q_no in sorted(int(q) for q in (subject_key.true_false_answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            value = self._exact_answer_lookup(omr.true_false_answers, q_no)
            if value == "" and tf_idx < len(tf_items):
                value = tf_items[tf_idx][1]
                tf_idx += 1
            elif value != "":
                tf_idx += 1
            canonical = self._tf_to_canonical_string(value)[:4]
            if len(canonical) < 4:
                canonical = canonical + ("_" * (4 - len(canonical)))
            parts.append(canonical)

        numeric_items = _sorted_items(omr.numeric_answers)
        numeric_idx = 0
        for q_no in sorted(int(q) for q in (subject_key.numeric_answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            key_text = self._normalize_numeric_text((subject_key.numeric_answers or {}).get(q_no, ""))
            width = len(key_text)
            if width <= 0:
                continue
            value = self._exact_answer_lookup(omr.numeric_answers, q_no)
            if value == "" and numeric_idx < len(numeric_items):
                value = numeric_items[numeric_idx][1]
                numeric_idx += 1
            elif value != "":
                numeric_idx += 1
            student = self._normalize_numeric_text(value)
            if len(student) < width:
                student = student + ("_" * (width - len(student)))
            parts.append(student[:width])
        built = "".join(parts)
        omr.answer_string = built
        return built

    @staticmethod
    def _strict_mcq_match(key: str, student: str) -> bool:
        key_text = str(key or "")
        student_text = str(student or "")[:len(key_text)]
        return len(student_text) == len(key_text) and all(a == b for a, b in zip(key_text, student_text))

    @staticmethod
    def _strict_tf_match(key: str, student: str) -> tuple[int, bool]:
        key_text = str(key or "")
        student_text = str(student or "")[:len(key_text)]
        correct = sum(1 for a, b in zip(key_text, student_text) if a == b or a == "E")
        return correct, len(student_text) == len(key_text) and correct == len(key_text)

    @staticmethod
    def _strict_numeric_match(key: str, student: str, delimiter: str = "|") -> bool:
        key_parts = [x for x in str(key or "").split(delimiter)]
        student_parts = [x for x in str(student or "").split(delimiter)][:len(key_parts)]
        if len(student_parts) != len(key_parts):
            return False
        return all(a == b for a, b in zip(key_parts, student_parts))

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

        answer_string = self._answer_string_for_scoring(omr, subject_key)
        cursor = 0
        full_credit_map = {
            "MCQ": {int(x) for x in (subject_key.full_credit_questions or {}).get("MCQ", []) if str(x).strip().lstrip("-").isdigit()},
            "TF": {int(x) for x in (subject_key.full_credit_questions or {}).get("TF", []) if str(x).strip().lstrip("-").isdigit()},
            "NUMERIC": {int(x) for x in (subject_key.full_credit_questions or {}).get("NUMERIC", []) if str(x).strip().lstrip("-").isdigit()},
        }

        for q_no in sorted(int(q) for q in (subject_key.answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            key = str((subject_key.answers or {}).get(q_no, "") or "").strip().upper()
            if not self._is_countable_mcq_key(key):
                continue
            student = self._slice_answer_string(answer_string, cursor, len(key)).strip().upper().replace("_", "")
            cursor += len(key)
            mcq_compare_items.append(self._build_mcq_compare_text(key, student, q_no))
            if q_no in full_credit_map["MCQ"]:
                correct += 1
                bonus_full_credit_count += 1
                bonus_full_credit_points += 1.0
                score += 1.0
            elif key == "E":
                if student != "":
                    correct += 1
                    mcq_correct += 1
                    score += 1.0
                else:
                    blank += 1
            elif student == "":
                blank += 1
            elif self._strict_mcq_match(key, student):
                correct += 1
                mcq_correct += 1
                score += 1.0
            else:
                wrong += 1

        for q_no in sorted(int(q) for q in (subject_key.true_false_answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            key_value = (subject_key.true_false_answers or {}).get(q_no, {})
            if not self._is_countable_tf_key(key_value):
                continue
            key_tf = self._tf_to_canonical_string(key_value)
            if not key_tf:
                continue
            raw_student = self._slice_answer_string(answer_string, cursor, len(key_tf))
            cursor += len(key_tf)
            student_tf = raw_student.replace("_", "") if set(raw_student) == {"_"} else raw_student
            tf_compare_items.append(self._build_tf_compare_text(key_tf, student_tf.replace('_', '-'), q_no))
            correct_count, tf_full_match = self._strict_tf_match(key_tf, student_tf)
            if q_no in full_credit_map["TF"]:
                correct += 1
                bonus_full_credit_count += 1
                bonus_full_credit_points += 1.0
                score += 1.0
            else:
                score += tf_points.get(correct_count, 0.0)
                if tf_full_match:
                    tf_correct += 1
                    correct += 1
                elif raw_student == "_" * len(key_tf):
                    blank += 1
                else:
                    wrong += 1

        for q_no in sorted(int(q) for q in (subject_key.numeric_answers or {}).keys() if str(q).strip().lstrip("-").isdigit()):
            key = self._normalize_numeric_text((subject_key.numeric_answers or {}).get(q_no, ""))
            if not self._is_countable_numeric_key(key):
                continue
            raw_student = self._slice_answer_string(answer_string, cursor, len(key))
            cursor += len(key)
            student = self._normalize_numeric_text(raw_student.replace("_", ""))
            numeric_compare_items.append(self._build_numeric_compare_text(key, student, q_no))
            if q_no in full_credit_map["NUMERIC"]:
                correct += 1
                bonus_full_credit_count += 1
                bonus_full_credit_points += 1.0
                score += 1.0
            elif key == "E":
                if student != "":
                    correct += 1
                    numeric_correct += 1
                    score += 1.0
                else:
                    blank += 1
            elif student == "":
                blank += 1
            elif key == student:
                correct += 1
                numeric_correct += 1
                score += 1.0
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
