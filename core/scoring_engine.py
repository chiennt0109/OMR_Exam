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

    @staticmethod
    def _sorted_numeric_keys(raw: object) -> list[int]:
        if not isinstance(raw, dict):
            return []
        return sorted(int(k) for k in raw.keys() if str(k).strip().lstrip("-").isdigit())

    @staticmethod
    def _question_score(section: str, q_no: int, subject_key: SubjectKey, subject_config: dict | None = None) -> float:
        cfg = subject_config or {}
        q_scores = (cfg.get("question_scores", {}) or {}).get(section, {}) if isinstance(cfg, dict) else {}
        if section in {"MCQ", "NUMERIC"}:
            if isinstance(q_scores, dict) and "per_question" in q_scores:
                return max(0.0, ScoringEngine._to_float(q_scores.get("per_question"), subject_key.points_for_question(q_no)))
            return max(0.0, float(subject_key.points_for_question(q_no) or 0.0))
        if isinstance(q_scores, dict) and q_scores:
            if "per_question" in q_scores:
                return max(0.0, ScoringEngine._to_float(q_scores.get("per_question"), subject_key.points_for_question(q_no)))
            vals = [ScoringEngine._to_float(v, 0.0) for v in q_scores.values()]
            if vals:
                return max(0.0, max(vals))
        return max(0.0, float(subject_key.points_for_question(q_no) or 0.0))

    def _question_definitions(self, subject_key: SubjectKey) -> dict[str, list[dict[str, object]]]:
        invalid = subject_key.invalid_answer_rows or {}
        defs: dict[str, list[dict[str, object]]] = {"MCQ": [], "TF": [], "NUMERIC": []}

        mcq_qs = sorted(set(self._sorted_numeric_keys(subject_key.answers) + self._sorted_numeric_keys(invalid.get("MCQ", {}))))
        for q_no in mcq_qs:
            raw_display = str((subject_key.answers or {}).get(q_no, (invalid.get("MCQ", {}) or {}).get(q_no, "")) or "").strip().upper()
            defs["MCQ"].append({"q_no": q_no, "display": raw_display or "-", "match": raw_display or "", "width": 1, "auto_full": raw_display == "G"})

        tf_qs = sorted(set(self._sorted_numeric_keys(subject_key.true_false_answers) + self._sorted_numeric_keys(invalid.get("TF", {}))))
        for q_no in tf_qs:
            key_value = (subject_key.true_false_answers or {}).get(q_no, None)
            raw_invalid = str((invalid.get("TF", {}) or {}).get(q_no, "") or "").strip()
            display = self._tf_to_canonical_string(key_value) if key_value is not None and key_value != "" else raw_invalid.upper().replace(" ", "")
            defs["TF"].append({"q_no": q_no, "display": display or "-", "match": display or "", "width": 4, "auto_full": str(raw_invalid or display).strip().upper() == "G"})

        num_qs = sorted(set(self._sorted_numeric_keys(subject_key.numeric_answers) + self._sorted_numeric_keys(invalid.get("NUMERIC", {}))))
        for q_no in num_qs:
            raw_display = str((subject_key.numeric_answers or {}).get(q_no, (invalid.get("NUMERIC", {}) or {}).get(q_no, "")) or "").strip()
            normalized = self._normalize_numeric_text(raw_display)
            width = len(normalized) if normalized else max(1, len(raw_display.strip()))
            defs["NUMERIC"].append({"q_no": q_no, "display": raw_display or "-", "match": normalized, "width": width, "auto_full": raw_display.strip().upper() == "G"})
        return defs

    def _answer_string_for_scoring(self, omr: OMRResult, subject_key: SubjectKey) -> str:
        existing = str(getattr(omr, "answer_string", "") or "")
        has_recognized_maps = bool((omr.mcq_answers or {}) or (omr.true_false_answers or {}) or (omr.numeric_answers or {}))
        if existing and not has_recognized_maps:
            return existing

        defs = self._question_definitions(subject_key)
        parts: list[str] = []
        mcq_aligned = self._aligned_marked_answers({int(item["q_no"]): "" for item in defs["MCQ"]}, omr.mcq_answers or {})
        tf_aligned = self._aligned_marked_answers({int(item["q_no"]): "" for item in defs["TF"]}, omr.true_false_answers or {})
        numeric_aligned = self._aligned_marked_answers({int(item["q_no"]): "" for item in defs["NUMERIC"]}, omr.numeric_answers or {})

        for item in defs["MCQ"]:
            q_no = int(item["q_no"])
            value = self._exact_answer_lookup(omr.mcq_answers, q_no)
            if value == "":
                value = mcq_aligned.get(q_no, "")
            token = str(value or "").strip().upper()[:1]
            parts.append(token or "_")
        for item in defs["TF"]:
            q_no = int(item["q_no"])
            value = self._exact_answer_lookup(omr.true_false_answers, q_no)
            if value == "":
                value = tf_aligned.get(q_no, "")
            canonical = self._tf_to_canonical_string(value)[:4]
            if len(canonical) < 4:
                canonical = canonical + ("_" * (4 - len(canonical)))
            parts.append(canonical)
        for item in defs["NUMERIC"]:
            q_no = int(item["q_no"])
            width = int(item["width"])
            value = self._exact_answer_lookup(omr.numeric_answers, q_no)
            if value == "":
                value = numeric_aligned.get(q_no, "")
            student = self._normalize_numeric_text(value)
            if len(student) < width:
                student = student + ("_" * (width - len(student)))
            parts.append(student[:width])

        built = "".join(parts)
        omr.answer_string = built
        return built

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
            if parsed is None and ch not in {"G", "_"}:
                continue
            if ch in {"G", "_"}:
                chars.append(ch)
            else:
                chars.append("Đ" if parsed else "S")
        return "".join(chars)

    def describe_formula(self, subject_key: SubjectKey, subject_config: dict | None = None) -> str:
        mcq_points = self._question_score("MCQ", self._sorted_numeric_keys(subject_key.answers)[0], subject_key, subject_config) if self._sorted_numeric_keys(subject_key.answers) else 0.0
        num_points = self._question_score("NUMERIC", self._sorted_numeric_keys(subject_key.numeric_answers)[0], subject_key, subject_config) if self._sorted_numeric_keys(subject_key.numeric_answers) else 0.0
        tf_full = 0.0
        tf_keys = self._sorted_numeric_keys(subject_key.true_false_answers)
        if tf_keys:
            tf_full = self._question_score("TF", tf_keys[0], subject_key, subject_config)
        return (
            "Chấm theo đáp án đã cấu hình: "
            f"MCQ đúng = {mcq_points:g} điểm/câu; "
            f"TF = {tf_full:g} điểm/câu, chia đều 4 ký tự ({(tf_full/4.0 if tf_full else 0.0):g} điểm/ký tự); "
            f"NUMERIC đúng = {num_points:g} điểm/câu; đáp án 'G' = tự động đúng."
        )

    def score(self, omr: OMRResult, subject_key: SubjectKey, student_name: str = "", subject_config: dict | None = None) -> ScoreResult:
        correct = wrong = blank = 0
        score = 0.0
        mcq_correct = tf_correct = numeric_correct = 0
        bonus_full_credit_count = 0
        bonus_full_credit_points = 0.0
        mcq_compare_items: list[str] = []
        tf_compare_items: list[str] = []
        numeric_compare_items: list[str] = []

        answer_string = self._answer_string_for_scoring(omr, subject_key)
        defs = self._question_definitions(subject_key)
        cursor = 0
        full_credit_map = {
            "MCQ": {int(x) for x in (subject_key.full_credit_questions or {}).get("MCQ", []) if str(x).strip().lstrip("-").isdigit()},
            "TF": {int(x) for x in (subject_key.full_credit_questions or {}).get("TF", []) if str(x).strip().lstrip("-").isdigit()},
            "NUMERIC": {int(x) for x in (subject_key.full_credit_questions or {}).get("NUMERIC", []) if str(x).strip().lstrip("-").isdigit()},
        }

        for item in defs["MCQ"]:
            q_no = int(item["q_no"])
            key_display = str(item["display"] or "-")
            key_match = str(item["match"] or "").upper()
            width = int(item["width"])
            raw_student = self._slice_answer_string(answer_string, cursor, width)
            cursor += width
            student = raw_student.replace("_", "").strip().upper()
            mcq_compare_items.append(self._build_mcq_compare_text(key_display, student, q_no))
            q_points = self._question_score("MCQ", q_no, subject_key, subject_config)
            auto_full = bool(item["auto_full"]) or q_no in full_credit_map["MCQ"] or key_match == "G"
            if auto_full:
                correct += 1
                mcq_correct += 1
                bonus_full_credit_count += 1 if q_no in full_credit_map["MCQ"] else 0
                bonus_full_credit_points += q_points if q_no in full_credit_map["MCQ"] else 0.0
                score += q_points
            elif student == "":
                blank += 1
            elif key_match and student == key_match:
                correct += 1
                mcq_correct += 1
                score += q_points
            else:
                wrong += 1

        for item in defs["TF"]:
            q_no = int(item["q_no"])
            key_display = str(item["display"] or "-").upper()
            key_match = key_display
            width = int(item["width"])
            raw_student = self._slice_answer_string(answer_string, cursor, width)
            cursor += width
            student_tf = raw_student.upper()
            tf_compare_items.append(self._build_tf_compare_text(key_display, student_tf.replace("_", "-"), q_no))
            q_points = self._question_score("TF", q_no, subject_key, subject_config)
            auto_full = bool(item["auto_full"]) or q_no in full_credit_map["TF"] or key_match == "G"
            if auto_full:
                correct += 1
                tf_correct += 1
                bonus_full_credit_count += 1 if q_no in full_credit_map["TF"] else 0
                bonus_full_credit_points += q_points if q_no in full_credit_map["TF"] else 0.0
                score += q_points
                continue
            if raw_student == "_" * width:
                blank += 1
                continue
            char_points = q_points / float(width or 1)
            matched = 0
            for expected, actual in zip(key_match[:width], student_tf[:width]):
                if expected == actual or expected == "G":
                    matched += 1
            score += matched * char_points
            if matched == width:
                correct += 1
                tf_correct += 1
            else:
                wrong += 1

        for item in defs["NUMERIC"]:
            q_no = int(item["q_no"])
            key_display = str(item["display"] or "-")
            key_match = str(item["match"] or "")
            width = int(item["width"])
            raw_student = self._slice_answer_string(answer_string, cursor, width)
            cursor += width
            student = self._normalize_numeric_text(raw_student.replace("_", ""))
            numeric_compare_items.append(self._build_numeric_compare_text(key_display, student, q_no))
            q_points = self._question_score("NUMERIC", q_no, subject_key, subject_config)
            auto_full = bool(item["auto_full"]) or q_no in full_credit_map["NUMERIC"] or str(key_display).strip().upper() == "G"
            if auto_full:
                correct += 1
                numeric_correct += 1
                bonus_full_credit_count += 1 if q_no in full_credit_map["NUMERIC"] else 0
                bonus_full_credit_points += q_points if q_no in full_credit_map["NUMERIC"] else 0.0
                score += q_points
            elif raw_student == "_" * width:
                blank += 1
            elif key_match and student == key_match:
                correct += 1
                numeric_correct += 1
                score += q_points
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
