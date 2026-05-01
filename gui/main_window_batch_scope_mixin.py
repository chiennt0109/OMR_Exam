from __future__ import annotations

import copy
import csv
import gc
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import deque
from datetime import date, datetime
from pathlib import Path
import time
import uuid
from typing import TYPE_CHECKING

sys.dont_write_bytecode = True

from PySide6.QtCore import Qt, QEvent, QTimer, QSize
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPixmap, QTransform, QPainter, QPen, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QCompleter,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QInputDialog,
    QToolBar,
    QStyle,
    QGroupBox,
    QScrollArea,
)

from core.answer_key_importer import ImportedAnswerKey, ImportedAnswerKeyPackage, import_answer_key
from core.omr_engine import OMRProcessor, OMRResult, RecognitionContext
from core.scoring_engine import ScoringEngine
from models.answer_key import AnswerKeyRepository, SubjectKey
from models.database import OMRDatabase, bootstrap_application_db
from models.exam_session import ExamSession, Student
from models.template import Template, ZoneType
from models.template_repository import TemplateRepository
from gui.ui_branding import app_icon, load_theme, TOOLBAR, STATUS, logo_symbol, logo_main, apply_widget_branding, brand_button
from gui.batch_scan_flow import run_batch_scan_from_api_file

if TYPE_CHECKING:
    from editor.template_editor import TemplateEditorWindow

from gui.main_window_dialogs import PreviewImageWidget, SubjectConfigDialog, NewExamDialog, StudentListPreviewDialog


class MainWindowBatchScopeMixin:
    """Answer scope trimming, blank computation and recognition-content text formatting."""
    @staticmethod
    def _format_eta_text(seconds: float) -> str:
        sec = max(0, int(round(float(seconds or 0.0))))
        if sec <= 0:
            return "~0s"
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"~{h}h {m:02d}m {s:02d}s"
        if m > 0:
            return f"~{m}m {s:02d}s"
        return f"~{s}s"

    @staticmethod
    def _normalized_mcq_answer_map(raw_map: object) -> dict[int, str]:
        normalized: dict[int, str] = {}
        if not isinstance(raw_map, dict):
            return normalized
        for q_raw, ans in raw_map.items():
            try:
                q_no = int(q_raw)
            except Exception:
                continue
            normalized[q_no] = str(ans or "").strip().upper()
        return normalized

    @staticmethod
    def _normalized_tf_answer_map(raw_map: object) -> dict[int, dict[str, bool]]:
        normalized: dict[int, dict[str, bool]] = {}
        if not isinstance(raw_map, dict):
            return normalized
        for q_raw, flags_raw in raw_map.items():
            try:
                q_no = int(q_raw)
            except Exception:
                continue
            flags_src = flags_raw if isinstance(flags_raw, dict) else {}
            flags: dict[str, bool] = {}
            for key_raw, flag in flags_src.items():
                key = str(key_raw or "").strip().lower()
                if key in {"a", "b", "c", "d"}:
                    flags[key] = bool(flag)
            if flags:
                normalized[q_no] = flags
        return normalized

    @staticmethod
    def _normalized_numeric_answer_map(raw_map: object) -> dict[int, str]:
        normalized: dict[int, str] = {}
        if not isinstance(raw_map, dict):
            return normalized
        for q_raw, value in raw_map.items():
            try:
                q_no = int(q_raw)
            except Exception:
                continue
            normalized[q_no] = str(value or "").strip()
        return normalized

    def _scoped_result_copy(self, result):
        scoped = copy.deepcopy(result)
        self._trim_result_answers_to_expected_scope(scoped)
        return scoped

    def _match_answer_key_payload_by_exam_code(self, mapping: dict, exam_code: str):
        if not isinstance(mapping, dict) or not mapping:
            return None
        exam_text = str(exam_code or "").strip()
        normalized_exam_text = self._normalize_exam_code_text(exam_text)
        if exam_text and exam_text in mapping:
            return mapping.get(exam_text)
        if normalized_exam_text:
            for key_text, payload in mapping.items():
                if self._normalize_exam_code_text(str(key_text or "").strip()) == normalized_exam_text:
                    return payload
        return None

    def _question_scope_from_subject_counts(
        self,
        template_expected: dict[str, list[int]],
        subject_cfg: dict | None = None,
        subject_key: str = "",
    ) -> dict[str, list[int]]:
        counts = self._subject_section_question_counts(subject_key) if subject_key else {"MCQ": 0, "TF": 0, "NUMERIC": 0}
        if not any(int(counts.get(sec, 0) or 0) > 0 for sec in ["MCQ", "TF", "NUMERIC"]):
            raw_counts = (subject_cfg or {}).get("question_counts", {}) if isinstance(subject_cfg, dict) else {}
            if isinstance(raw_counts, dict):
                counts = {
                    "MCQ": int(raw_counts.get("MCQ", 0) or 0),
                    "TF": int(raw_counts.get("TF", 0) or 0),
                    "NUMERIC": int(raw_counts.get("NUMERIC", 0) or 0),
                }

        scoped = {"MCQ": [], "TF": [], "NUMERIC": []}
        for sec in ["MCQ", "TF", "NUMERIC"]:
            limit = max(0, int(counts.get(sec, 0) or 0))
            if limit <= 0:
                scoped[sec] = []
                continue
            template_section = [int(q) for q in (template_expected.get(sec, []) or []) if str(q).strip().lstrip("-").isdigit() and int(q) > 0]
            scoped[sec] = list(template_section[:limit]) if template_section else list(range(1, limit + 1))
        return scoped

    def _normalize_answer_map_to_expected_scope(self, data: dict, expected_questions: list[int], cast_value):
        normalized: dict[int, Any] = {}
        for q_raw, value in (data or {}).items():
            try:
                q_no = int(q_raw)
            except Exception:
                continue
            normalized[q_no] = cast_value(value)

        expected_set = sorted(
            {
                int(q)
                for q in (expected_questions or [])
                if str(q).strip().lstrip("-").isdigit() and int(q) > 0
            }
        )
        if not expected_set:
            return {}
        if not normalized:
            return {}

        actual_set = sorted(set(normalized.keys()))
        overlap = sorted(set(actual_set) & set(expected_set))
        if overlap:
            return {q: normalized[q] for q in expected_set if q in normalized}

        remapped: dict[int, Any] = {}
        for src_q, dst_q in zip(actual_set, expected_set):
            remapped[dst_q] = normalized[src_q]
        return remapped

    def _compute_blank_questions(self, result) -> dict[str, list]:
        expected_by_section = self._expected_questions_by_section(result)
        mcq_payload = {int(q): str(a) for q, a in (dict(getattr(result, "mcq_answers", {}) or {})).items()}
        tf_payload = {int(q): dict(v or {}) for q, v in (dict(getattr(result, "true_false_answers", {}) or {})).items()}
        numeric_payload = {int(q): str(v) for q, v in (dict(getattr(result, "numeric_answers", {}) or {})).items()}

        mcq_expected = sorted(set(int(q) for q in (expected_by_section.get("MCQ", []) or [])))
        tf_expected = sorted(set(int(q) for q in (expected_by_section.get("TF", []) or [])))
        numeric_expected = sorted(set(int(q) for q in (expected_by_section.get("NUMERIC", []) or [])))

        blanks: dict[str, list[int] | list[str]] = {}
        for sec in ["MCQ", "TF", "NUMERIC"]:
            if sec == "MCQ":
                blanks[sec] = [int(q) for q in mcq_expected if not str(mcq_payload.get(int(q), "") or "").strip()]
            elif sec == "NUMERIC":
                blanks[sec] = [int(q) for q in numeric_expected if not str(numeric_payload.get(int(q), "") or "").strip()]
            else:
                blanks[sec] = []
        tf_blank_detail: dict[int, int] = {}
        missing_tf_statements: list[str] = []
        for display_q in tf_expected:
            flags = tf_payload.get(int(display_q), {})
            flags = flags if isinstance(flags, dict) else {}
            missing_count = sum(1 for key in ["a", "b", "c", "d"] if key not in flags)
            if missing_count > 0:
                tf_blank_detail[int(display_q)] = int(missing_count)
                for key in ["a", "b", "c", "d"]:
                    if key not in flags:
                        missing_tf_statements.append(f"{int(display_q)}{key}")
        sec = "TF"
        blanks[sec] = missing_tf_statements

        setattr(result, "tf_blank_detail", tf_blank_detail)
        return {
            "MCQ": list(blanks.get("MCQ", []) or []),
            "TF": list(blanks.get("TF", []) or []),
            "NUMERIC": list(blanks.get("NUMERIC", []) or []),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    @staticmethod
    def _format_mcq_answers_with_expected(answers: dict[int, str], expected_questions: list[int]) -> str:
        expected = sorted({int(q) for q in (expected_questions or []) if int(q) > 0})
        if not expected:
            return MainWindowBatchScopeMixin._format_mcq_answers(answers)
        chunks: list[str] = []
        for q in expected:
            ans = str((answers or {}).get(int(q), "") or "").strip().upper()
            chunks.append(f"{int(q)}{ans if ans else '_'}")
        return "; ".join(chunks) if chunks else "-"

    @staticmethod
    def _format_tf_answers_with_expected(answers: dict[int, dict[str, bool]], expected_questions: list[int]) -> str:
        expected = sorted({int(q) for q in (expected_questions or []) if int(q) > 0})
        if not expected:
            return MainWindowBatchScopeMixin._format_tf_answers(answers)
        chunks: list[str] = []
        for q in expected:
            flags = dict((answers or {}).get(int(q), {}) or {})
            marks = "".join(("Đ" if bool(flags.get(k)) else "S") if k in flags else "_" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks) if chunks else "-"

    @staticmethod
    def _format_numeric_answers_with_expected(answers: dict[int, str], expected_questions: list[int]) -> str:
        expected = sorted({int(q) for q in (expected_questions or []) if int(q) > 0})
        if not expected:
            return MainWindowBatchScopeMixin._format_numeric_answers(answers)
        chunks: list[str] = []
        for q in expected:
            value = str((answers or {}).get(int(q), "") or "").strip()
            chunks.append(f"{int(q)}={value if value else '_'}")
        return "; ".join(chunks) if chunks else "-"

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        subject_cfg = self._selected_batch_subject_config() or self._resolve_subject_config_for_batch()
        current_subject_key = str(self._current_batch_subject_key() or self.active_batch_subject_key or "").strip()
        answer_subject_key = str(self._answer_key_subject_key(current_subject_key, subject_cfg) or "").strip()
        exam_code = str(getattr(result, "exam_code", "") or "").strip()
        normalized_exam_code = self._normalize_exam_code_text(exam_code)
        key_payload: Any = None
        has_configured_keys = False

        imported_answer_keys = self._subject_imported_answer_keys_for_main(subject_cfg or {})
        if imported_answer_keys:
            has_configured_keys = True
            key_payload = self._match_answer_key_payload_by_exam_code(imported_answer_keys, exam_code)

        if key_payload is None and current_subject_key:
            configured = self._fetch_answer_keys_for_subject_scoped(current_subject_key, subject_cfg) or {}
            if configured:
                has_configured_keys = True
                key_payload = self._match_answer_key_payload_by_exam_code(configured, exam_code)

        if key_payload is None and self.answer_keys is not None:
            subject_candidates: list[str] = []
            for candidate in [current_subject_key, answer_subject_key]:
                candidate_text = str(candidate or "").strip()
                if candidate_text and candidate_text not in subject_candidates:
                    subject_candidates.append(candidate_text)
            for candidate_subject in subject_candidates:
                rows = [
                    row for row in self.answer_keys.keys.values()
                    if str(getattr(row, "subject", "") or "").strip() == candidate_subject
                ]
                if rows:
                    has_configured_keys = True
                for row in rows:
                    row_exam_code = str(getattr(row, "exam_code", "") or "").strip()
                    if exam_code and row_exam_code == exam_code:
                        key_payload = row
                        break
                    if normalized_exam_code and self._normalize_exam_code_text(row_exam_code) == normalized_exam_code:
                        key_payload = row
                        break
                if key_payload is not None:
                    break

        scoped_by_counts = self._question_scope_from_subject_counts(template_expected, subject_cfg, current_subject_key)

        if key_payload is not None:
            full_credit = (key_payload.get("full_credit_questions", {}) if isinstance(key_payload, dict) else getattr(key_payload, "full_credit_questions", {})) or {}
            invalid_rows = (key_payload.get("invalid_answer_rows", {}) if isinstance(key_payload, dict) else getattr(key_payload, "invalid_answer_rows", {})) or {}

            def _extra_for_section(sec: str) -> set[int]:
                extra: set[int] = set()
                for q in list((full_credit.get(sec, []) or [])):
                    try:
                        extra.add(int(q))
                    except Exception:
                        continue
                for q in dict((invalid_rows.get(sec, {}) or {})).keys():
                    try:
                        extra.add(int(q))
                    except Exception:
                        continue
                return extra

            def _sorted_question_keys(payload: object) -> list[int]:
                out: set[int] = set()
                for q in dict(payload or {}).keys():
                    try:
                        out.add(int(q))
                    except Exception:
                        continue
                return sorted(out)

            mcq_map = (key_payload.get("mcq_answers", {}) if isinstance(key_payload, dict) else getattr(key_payload, "answers", {})) or {}
            tf_map = (key_payload.get("true_false_answers", {}) if isinstance(key_payload, dict) else getattr(key_payload, "true_false_answers", {})) or {}
            numeric_map = (key_payload.get("numeric_answers", {}) if isinstance(key_payload, dict) else getattr(key_payload, "numeric_answers", {})) or {}
            key_scope = {
                "MCQ": sorted(set(_sorted_question_keys(mcq_map)) | _extra_for_section("MCQ")),
                "TF": sorted(set(_sorted_question_keys(tf_map)) | _extra_for_section("TF")),
                "NUMERIC": sorted(set(_sorted_question_keys(numeric_map)) | _extra_for_section("NUMERIC")),
            }

            if any(scoped_by_counts.get(sec, []) for sec in ["MCQ", "TF", "NUMERIC"]):
                merged_scope: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    allowed = [int(q) for q in (scoped_by_counts.get(sec, []) or [])]
                    if not allowed:
                        merged_scope[sec] = []
                        continue
                    allowed_set = set(allowed)
                    filtered = [int(q) for q in (key_scope.get(sec, []) or []) if int(q) in allowed_set]
                    merged_scope[sec] = filtered or list(allowed)
                return merged_scope
            return key_scope

        if any(scoped_by_counts.get(sec, []) for sec in ["MCQ", "TF", "NUMERIC"]):
            return scoped_by_counts
        if has_configured_keys:
            return {"MCQ": [], "TF": [], "NUMERIC": []}
        return {sec: list(vals) for sec, vals in template_expected.items()}

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join(("Đ" if bool((flags or {}).get(k)) else "S") if k in (flags or {}) else "_" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, v in sorted(answers.items(), key=lambda x: int(x[0])):
            value = str(v).strip()
            chunks.append(f"{int(q)}={value if value else '_'}")
        return "; ".join(chunks)

    def _build_recognition_content_text(
        self,
        result,
        blank_map: dict[str, list[int]],
        expected_by_section: dict[str, list[int]] | None = None,
    ) -> str:
        expected = expected_by_section or self._expected_questions_by_section(result)
        lines: list[str] = []

        mcq_expected = set(int(q) for q in (expected.get("MCQ", []) or []) if str(q).strip())
        mcq_blank = [
            int(v)
            for v in list((blank_map or {}).get("MCQ", []) or [])
            if str(v).strip() and int(v) in mcq_expected
        ]
        if mcq_blank:
            lines.append(f"MCQ trống: {', '.join(str(v) for v in mcq_blank)}")

        tf_expected = sorted(int(q) for q in (expected.get("TF", []) or []) if str(q).strip())
        tf_blank_tokens = [str(v).strip().lower() for v in list((blank_map or {}).get("TF", []) or []) if str(v).strip()]
        tf_blank_set = {
            int(token[:-1])
            for token in tf_blank_tokens
            if len(token) >= 2 and token[:-1].lstrip("-").isdigit() and token[-1] in {"a", "b", "c", "d"}
        }
        tf_answers = {
            int(q): dict(v or {})
            for q, v in (dict(getattr(result, "true_false_answers", {}) or {})).items()
            if str(q).strip()
        }
        missing_tf_statements: list[str] = []
        for display_q in tf_expected:
            if display_q not in tf_blank_set:
                continue
            flags = tf_answers.get(int(display_q), {}) if isinstance(tf_answers.get(int(display_q), {}), dict) else {}
            for key in ["a", "b", "c", "d"]:
                if key not in flags:
                    missing_tf_statements.append(f"{int(display_q)}{key}")
        if missing_tf_statements:
            lines.append("TF trống: " + ", ".join(missing_tf_statements))

        numeric_expected = set(int(q) for q in (expected.get("NUMERIC", []) or []) if str(q).strip())
        numeric_blank = [
            int(v)
            for v in list((blank_map or {}).get("NUMERIC", []) or [])
            if str(v).strip() and int(v) in numeric_expected
        ]
        if numeric_blank:
            lines.append(f"Num trống: {', '.join(str(v) for v in numeric_blank)}")

        return "\n".join(lines)

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        if result is None:
            return

        code_text = str(getattr(result, "exam_code", "") or "").strip()
        avail_codes = self._available_exam_codes()
        code_valid = bool(code_text and "?" not in code_text and (not avail_codes or code_text in avail_codes or self._normalize_exam_code_text(code_text) in avail_codes))
        if not code_valid:
            return

        subject_key = str(self._current_batch_subject_key() or self.active_batch_subject_key or "").strip()
        if not subject_key:
            return
        if self.template is None:
            return

        expected = self._expected_questions_by_section(result)
        if not any(expected.get(sec, []) for sec in ["MCQ", "TF", "NUMERIC"]):
            return

        def _trim_map_by_expected(data: dict, expected_questions: list[int], cast_value):
            normalized = {int(q): cast_value(v) for q, v in (data or {}).items()}
            expected_set = sorted({int(q) for q in (expected_questions or []) if str(q).strip().lstrip('-').isdigit() and int(q) > 0})
            if not expected_set:
                return normalized
            return {q: normalized[q] for q in expected_set if q in normalized}

        result.mcq_answers = _trim_map_by_expected(result.mcq_answers or {}, list(expected.get("MCQ", []) or []), lambda v: str(v))
        result.true_false_answers = _trim_map_by_expected(result.true_false_answers or {}, list(expected.get("TF", []) or []), lambda v: dict(v or {}))
        result.numeric_answers = _trim_map_by_expected(result.numeric_answers or {}, list(expected.get("NUMERIC", []) or []), lambda v: str(v))

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _retrim_batch_results_to_answer_key_scope(self) -> None:
        if not self.scan_results:
            return
        for idx, res in enumerate(self.scan_results):
            scoped = self._scoped_result_copy(res)
            blank_map = self._compute_blank_questions(scoped)
            expected = self._expected_questions_by_section(scoped)
            self.scan_blank_summary[idx] = blank_map
            self.scan_blank_questions[idx] = blank_map.get("MCQ", [])
            if idx < self.scan_list.rowCount():
                self.scan_list.setItem(idx, self.SCAN_COL_CONTENT, QTableWidgetItem(self._build_recognition_content_text(scoped, blank_map, expected)))
                sid_item = self.scan_list.item(idx, self.SCAN_COL_STUDENT_ID)
                if sid_item:
                    sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(scoped))
        self._refresh_all_statuses()
        current = self.scan_list.currentRow() if hasattr(self, "scan_list") else -1
        if 0 <= current < len(self.scan_results):
            self._update_scan_preview(current)
            self._load_selected_result_for_correction()

    def _available_exam_codes(self, subject_key: str = "") -> set[str]:
        out: set[str] = set()
        subject = str(subject_key or self._current_batch_subject_key() or "").strip()
        if subject:
            cfg = self._subject_config_by_subject_key(subject) or {}
            imported_cfg = cfg.get("imported_answer_keys", {}) if isinstance(cfg.get("imported_answer_keys", {}), dict) else {}
            for raw in imported_cfg.keys():
                code_text = str(raw).strip()
                if not code_text:
                    continue
                out.add(code_text)
                out.add(self._normalize_exam_code_text(code_text))
            try:
                fetched = self._fetch_answer_keys_for_subject_scoped(subject)
            except Exception:
                fetched = {}
            for raw in (fetched or {}).keys():
                code_text = str(raw).strip()
                if not code_text:
                    continue
                out.add(code_text)
                out.add(self._normalize_exam_code_text(code_text))
            if self.answer_keys is not None:
                for key_obj in self.answer_keys.keys.values():
                    key_subject = str(getattr(key_obj, "subject", "") or "").strip()
                    code_text = str(getattr(key_obj, "exam_code", "") or "").strip()
                    if key_subject != subject or not code_text:
                        continue
                    out.add(code_text)
                    out.add(self._normalize_exam_code_text(code_text))
        if not out:
            for x in (self.imported_exam_codes or []):
                raw = str(x).strip()
                if not raw:
                    continue
                out.add(raw)
                out.add(self._normalize_exam_code_text(raw))
        return {v for v in out if v}

    def _subject_answer_key_for_result(self, result, subject_key: str = ""):
        subject = str(subject_key or self._current_batch_subject_key() or "").strip()
        exam_code = str(getattr(result, "exam_code", "") or "").strip()
        if not subject or not exam_code:
            return None
        if self.answer_keys is not None:
            key = self.answer_keys.get(subject, exam_code)
            if key is not None:
                return key
            normalized = self._normalize_exam_code_text(exam_code)
            for candidate in (self.imported_exam_codes or []):
                candidate_text = str(candidate).strip()
                if candidate_text and self._normalize_exam_code_text(candidate_text) == normalized:
                    key = self.answer_keys.get(subject, candidate_text)
                    if key is not None:
                        return key
        fetched = self._fetch_answer_keys_for_subject_scoped(subject)
        if exam_code in fetched:
            return fetched[exam_code]
        normalized = self._normalize_exam_code_text(exam_code)
        for candidate_text, candidate_key in fetched.items():
            if self._normalize_exam_code_text(candidate_text) == normalized:
                return candidate_key
        return None

    @staticmethod
    def _answer_string_from_maps(
        mcq_answers: dict[int, str],
        tf_answers: dict[int, dict[str, bool]],
        numeric_answers: dict[int, str],
        answer_key,
        *,
        use_semicolon: bool = True,
    ) -> str:
        if answer_key is None:
            return ""

        def _question_numbers(valid_map, invalid_map, fallback_map=None) -> list[int]:
            primary_nums = set()
            for src in [valid_map or {}, invalid_map or {}]:
                for key in src.keys():
                    if str(key).strip().lstrip("-").isdigit():
                        primary_nums.add(int(key))
            if primary_nums:
                return sorted(primary_nums)
            nums = set()
            for key in (fallback_map or {}).keys():
                if str(key).strip().lstrip("-").isdigit():
                    nums.add(int(key))
            return sorted(nums)

        invalid_rows = getattr(answer_key, "invalid_answer_rows", {}) or {}
        parts: list[str] = []
        for q_no in _question_numbers(
            getattr(answer_key, "answers", {}) or {},
            (invalid_rows.get("MCQ", {}) or {}),
            mcq_answers or {},
        ):
            value = str((mcq_answers or {}).get(q_no, "") or "").strip().upper()[:1]
            parts.append(value or "_")
        for q_no in _question_numbers(
            getattr(answer_key, "true_false_answers", {}) or {},
            (invalid_rows.get("TF", {}) or {}),
            tf_answers or {},
        ):
            flags = (tf_answers or {}).get(q_no, {}) or {}
            for key in ["a", "b", "c", "d"]:
                parts.append("Đ" if key in flags and bool(flags.get(key)) else ("S" if key in flags else "_"))
        numeric_valid = getattr(answer_key, "numeric_answers", {}) or {}
        numeric_invalid = (invalid_rows.get("NUMERIC", {}) or {})
        for q_no in _question_numbers(
            numeric_valid,
            numeric_invalid,
            numeric_answers or {},
        ):
            raw_key = str((getattr(answer_key, "numeric_answers", {}) or {}).get(q_no, ((invalid_rows.get("NUMERIC", {}) or {}).get(q_no, ""))) or "")
            student_text_full = str((numeric_answers or {}).get(q_no, "") or "").strip().replace(" ", "").lstrip("+").replace(".", ",")
            normalized_key = str(raw_key).strip().replace(" ", "").lstrip("+").replace(".", ",")
            if normalized_key:
                width = len(normalized_key)
                student_text = student_text_full[:width]
                if len(student_text) < width:
                    student_text = student_text + ("_" * (width - len(student_text)))
                parts.append(student_text)
            else:
                parts.append(student_text_full or "_")
        return ";".join(parts) if use_semicolon else "".join(parts)

    def _build_answer_string_for_result(self, result, subject_key: str = "") -> str:
        scoped_result = self._scoped_result_copy(result)
        key = self._subject_answer_key_for_result(scoped_result, subject_key)
        expected_by_section = self._expected_questions_by_section(scoped_result)
        if key is not None and any(expected_by_section.get(sec, []) for sec in ["MCQ", "TF", "NUMERIC"]):
            invalid_rows = getattr(key, "invalid_answer_rows", {}) or {}

            def _filter_question_map(payload: object, expected_questions: list[int]) -> dict[int, Any]:
                expected_set = {int(q) for q in (expected_questions or [])}
                filtered: dict[int, Any] = {}
                for q_raw, value in dict(payload or {}).items():
                    try:
                        q_no = int(q_raw)
                    except Exception:
                        continue
                    if q_no in expected_set:
                        filtered[q_no] = value
                return filtered

            scoped_key = type("ScopedAnswerKey", (), {})()
            setattr(scoped_key, "answers", _filter_question_map(getattr(key, "answers", {}) or {}, expected_by_section.get("MCQ", [])))
            setattr(scoped_key, "true_false_answers", _filter_question_map(getattr(key, "true_false_answers", {}) or {}, expected_by_section.get("TF", [])))
            setattr(scoped_key, "numeric_answers", _filter_question_map(getattr(key, "numeric_answers", {}) or {}, expected_by_section.get("NUMERIC", [])))
            setattr(
                scoped_key,
                "invalid_answer_rows",
                {
                    "MCQ": _filter_question_map((invalid_rows.get("MCQ", {}) or {}), expected_by_section.get("MCQ", [])),
                    "TF": _filter_question_map((invalid_rows.get("TF", {}) or {}), expected_by_section.get("TF", [])),
                    "NUMERIC": _filter_question_map((invalid_rows.get("NUMERIC", {}) or {}), expected_by_section.get("NUMERIC", [])),
                },
            )
            key = scoped_key
        use_semicolon = not bool(getattr(scoped_result, "answer_string_api_mode", False))
        return self._answer_string_from_maps(
            scoped_result.mcq_answers or {},
            scoped_result.true_false_answers or {},
            scoped_result.numeric_answers or {},
            key,
            use_semicolon=use_semicolon,
        )

    def _normalize_non_api_answer_string(self, result: OMRResult, subject_key: str = "") -> str:
        if result is None:
            return ""
        answer_text = str(getattr(result, "answer_string", "") or "").strip()
        if bool(getattr(result, "answer_string_api_mode", False)):
            return answer_text
        if answer_text:
            return answer_text
        subject = str(subject_key or self._current_batch_subject_key() or self.active_batch_subject_key or "").strip()
        rebuilt = str(self._build_answer_string_for_result(result, subject) or "").strip()
        if rebuilt:
            setattr(result, "answer_string", rebuilt)
        return rebuilt
