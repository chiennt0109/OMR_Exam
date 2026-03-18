from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
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
    def _full_credit_qset(subject_key: SubjectKey, section: str) -> set[int]:
        data = (subject_key.full_credit_questions or {}).get(section, []) if isinstance(subject_key.full_credit_questions, dict) else []
        invalid_map = (subject_key.invalid_answer_rows or {}).get(section, {}) if isinstance(subject_key.invalid_answer_rows, dict) else {}
        out: set[int] = set()
        for q in data or []:
            try:
                out.add(int(q))
            except Exception:
                continue
        for q in (invalid_map or {}).keys():
            try:
                out.add(int(q))
            except Exception:
                continue
        return out

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
        text = text.replace(" ", "").replace(",", ".")
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
    def _full_credit_compare_label(raw_text: str, points: float) -> str:
        base = str(raw_text or "").strip() or "[FULL]"
        return f"{base}[FULL+{points:g}]"

    @staticmethod
    def _invalid_key_text(subject_key: SubjectKey, section: str, q_no: int) -> str:
        raw = ((subject_key.invalid_answer_rows or {}).get(section, {}) or {}).get(q_no, "")
        text = str(raw or "").strip()
        return text or "[FULL]"

    @staticmethod
    def _aligned_marked_answers(key_answers: dict, marked_answers: object) -> dict[int, object]:
        if not isinstance(key_answers, dict) or not key_answers:
            return {}
        if not isinstance(marked_answers, dict) or not marked_answers:
            return {}

        key_qs: list[int] = []
        for k in key_answers.keys():
            try:
                key_qs.append(int(k))
            except Exception:
                continue
        key_qs.sort()
        if not key_qs:
            return {}

        raw_marked: dict[int, object] = {}
        for mk, mv in marked_answers.items():
            try:
                raw_marked[int(mk)] = mv
            except Exception:
                continue
        if not raw_marked:
            return {}

        aligned: dict[int, object] = {}
        used_marked: set[int] = set()

        # Prefer exact question-number matches first.
        for q in key_qs:
            if q in raw_marked:
                aligned[q] = raw_marked[q]
                used_marked.add(q)

        # Fallback by positional order for remaining questions (handles shifted numbering like key 13.. vs scan 1..).
        remaining_marked_qs = [q for q in sorted(raw_marked.keys()) if q not in used_marked]
        missing_key_qs = [q for q in key_qs if q not in aligned]
        for kq, mq in zip(missing_key_qs, remaining_marked_qs):
            aligned[kq] = raw_marked[mq]

        return aligned

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
            parsed = self._to_bool_mark(ch)
            if parsed is None:
                continue
            chars.append("Đ" if parsed else "S")
        return "".join(chars)

    def _score_profile(self, subject_key: SubjectKey, subject_config: dict | None) -> dict:
        cfg = subject_config if isinstance(subject_config, dict) else {}
        mode = str(cfg.get("score_mode", "") or "").strip() or "Điểm theo phần"
        section_scores = cfg.get("section_scores", {}) if isinstance(cfg.get("section_scores", {}), dict) else {}
        question_scores = cfg.get("question_scores", {}) if isinstance(cfg.get("question_scores", {}), dict) else {}

        mcq_qs = {
            int(q)
            for q, ans in (subject_key.answers or {}).items()
            if str(q).strip().lstrip("-").isdigit() and self._is_countable_mcq_key(ans)
        } | self._full_credit_qset(subject_key, "MCQ")
        num_qs = {
            int(q)
            for q, ans in (subject_key.numeric_answers or {}).items()
            if str(q).strip().lstrip("-").isdigit() and self._is_countable_numeric_key(ans)
        } | self._full_credit_qset(subject_key, "NUMERIC")

        mcq_count = len(mcq_qs)
        num_count = len(num_qs)

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
            "mode": mode,
        }

    def describe_formula(self, subject_key: SubjectKey, subject_config: dict | None = None) -> str:
        profile = self._score_profile(subject_key, subject_config)
        tf_map = profile.get("tf_points", {}) or {}
        tf_desc = ", ".join([f"{k} ý={v:g}" for k, v in sorted(tf_map.items()) if isinstance(k, int)])
        return (
            f"Công thức ({profile.get('mode', 'Điểm theo phần')}): "
            f"Điểm = (MCQ_đúng × {profile['mcq_per']:g}) + "
            f"(NUMERIC_đúng × {profile['num_per']:g}) + "
            f"Σ điểm_TF_theo_số_ý_đúng; TF: [{tf_desc}]"
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
        profile = self._score_profile(subject_key, subject_config)
        aligned_mcq_marked = self._aligned_marked_answers(subject_key.answers or {}, omr.mcq_answers or {})
        aligned_tf_marked = self._aligned_marked_answers(subject_key.true_false_answers or {}, omr.true_false_answers or {})
        aligned_numeric_marked = self._aligned_marked_answers(subject_key.numeric_answers or {}, omr.numeric_answers or {})
        mcq_full_credit = self._full_credit_qset(subject_key, "MCQ")
        tf_full_credit = self._full_credit_qset(subject_key, "TF")
        numeric_full_credit = self._full_credit_qset(subject_key, "NUMERIC")

        mcq_qs = sorted({int(q) for q in (subject_key.answers or {}).keys() if str(q).strip().lstrip("-").isdigit()} | mcq_full_credit)
        for q_no in mcq_qs:
            key_answer = (subject_key.answers or {}).get(q_no, "")
            print(f"[SCORING] Q{q_no} type=MCQ full={q_no in mcq_full_credit}")
            if q_no in mcq_full_credit:
                marked = aligned_mcq_marked.get(q_no)
                marked_mcq = str(marked or "").strip().upper()
                awarded = float(profile["mcq_per"])
                mcq_compare_items.append(self._build_mcq_compare_text(self._full_credit_compare_label(self._invalid_key_text(subject_key, "MCQ", q_no), awarded), marked_mcq, q_no))
                correct += 1
                mcq_correct += 1
                bonus_full_credit_count += 1
                bonus_full_credit_points += awarded
                score += awarded
                continue
            if not self._is_countable_mcq_key(key_answer):
                continue
            marked = aligned_mcq_marked.get(q_no)
            key_mcq = str(key_answer or "").strip().upper()
            marked_mcq = str(marked or "").strip().upper()
            mcq_compare_items.append(self._build_mcq_compare_text(key_mcq, marked_mcq, q_no))
            if not marked_mcq:
                blank += 1
            elif marked_mcq == key_mcq:
                correct += 1
                mcq_correct += 1
                score += float(profile["mcq_per"])
            else:
                wrong += 1

        tf_qs = sorted({int(q) for q in (subject_key.true_false_answers or {}).keys() if str(q).strip().lstrip("-").isdigit()} | tf_full_credit)
        for q_no in tf_qs:
            key_answer = (subject_key.true_false_answers or {}).get(q_no, {})
            print(f"[SCORING] Q{q_no} type=TF full={q_no in tf_full_credit}")
            if q_no in tf_full_credit:
                marked = aligned_tf_marked.get(q_no)
                marked_tf = self._tf_to_canonical_string(marked)
                awarded = float(profile["tf_points"].get(4, max(profile["tf_points"].values() or [0.0])))
                tf_compare_items.append(self._build_tf_compare_text(self._full_credit_compare_label(self._invalid_key_text(subject_key, "TF", q_no), awarded), marked_tf, q_no))
                correct += 1
                tf_correct += 1
                bonus_full_credit_count += 1
                bonus_full_credit_points += awarded
                score += awarded
                continue
            if not self._is_countable_tf_key(key_answer):
                continue
            marked = aligned_tf_marked.get(q_no)
            key_tf = self._tf_to_canonical_string(key_answer)
            marked_tf = self._tf_to_canonical_string(marked)
            if not key_tf:
                continue
            tf_compare_items.append(self._build_tf_compare_text(key_tf, marked_tf, q_no))
            if not marked_tf:
                blank += 1
            elif key_tf == marked_tf and len(key_tf) == len(marked_tf):
                correct += 1
                tf_correct += 1
                score += float(profile["tf_points"].get(len(key_tf), 1.0))
            else:
                wrong += 1

        numeric_qs = sorted({int(q) for q in (subject_key.numeric_answers or {}).keys() if str(q).strip().lstrip("-").isdigit()} | numeric_full_credit)
        for q_no in numeric_qs:
            key_answer = (subject_key.numeric_answers or {}).get(q_no, "")
            print(f"[SCORING] Q{q_no} type=NUMERIC full={q_no in numeric_full_credit}")
            if q_no in numeric_full_credit:
                marked = aligned_numeric_marked.get(q_no)
                norm_marked = self._normalize_numeric_text(marked)
                awarded = float(profile["num_per"])
                numeric_compare_items.append(self._build_numeric_compare_text(self._full_credit_compare_label(self._invalid_key_text(subject_key, "NUMERIC", q_no), awarded), norm_marked, q_no))
                correct += 1
                numeric_correct += 1
                bonus_full_credit_count += 1
                bonus_full_credit_points += awarded
                score += awarded
                continue
            if not self._is_countable_numeric_key(key_answer):
                continue
            marked = aligned_numeric_marked.get(q_no)
            norm_marked = self._normalize_numeric_text(marked)
            norm_key = self._normalize_numeric_text(key_answer)
            numeric_compare_items.append(self._build_numeric_compare_text(norm_key, norm_marked, q_no))
            if marked is None or str(marked).strip() == "":
                blank += 1
            elif norm_marked == norm_key:
                correct += 1
                numeric_correct += 1
                score += float(profile["num_per"])
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
