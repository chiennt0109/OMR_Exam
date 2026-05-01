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
    QStackedWidget,
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


class MainWindowBatchEditMixin:
    """Manual correction, visual answer editor, row status calculation and rerun/edit actions."""
    def action_apply_manual_correction(self) -> None:
        self.apply_manual_correction()

    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        btn_load_selected = QPushButton("Load Selected Scan Result")
        btn_load_selected.clicked.connect(self._load_selected_result_for_correction)
        btn_apply_correction = QPushButton("Apply Manual Correction")
        btn_apply_correction.clicked.connect(self.apply_manual_correction)

        left_layout.addWidget(QLabel("Detected Errors"))
        left_layout.addWidget(self.error_list)
        left_layout.addWidget(btn_load_selected)
        left_layout.addWidget(btn_apply_correction)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.preview_label)
        correction_form = QFormLayout()
        correction_form.addRow("Mã đề (DB)", self.exam_code_correction_combo)
        correction_form.addRow("Học sinh (DB)", self.student_correction_combo)
        right_layout.addLayout(correction_form)
        right_layout.addWidget(QLabel("Chỉnh sửa đáp án trực quan"))
        right_layout.addWidget(self.answer_editor_scroll, 4)
        right_layout.addWidget(self.result_preview)
        right_layout.addWidget(QLabel("Manual Edit"))
        right_layout.addWidget(self.manual_edit)

        splitter.addWidget(left)
        splitter.addWidget(right)

        layout = QHBoxLayout(w)
        layout.addWidget(splitter)
        return w

    def _correction_selected_result(self) -> tuple[int, OMRResult] | tuple[None, None]:
        idx = self.scan_list.currentRow() if hasattr(self, "scan_list") else -1
        if idx < 0 or idx >= len(self.scan_results):
            return None, None
        return idx, self.scan_results[idx]

    def _load_exam_code_correction_options(self, subject_key: str, current_code: str) -> None:
        self.exam_code_correction_combo.blockSignals(True)
        self.exam_code_correction_combo.clear()
        codes = set(self._fetch_answer_keys_for_subject_scoped(subject_key).keys())
        codes.update(str(x).strip() for x in (self.imported_exam_codes or []) if str(x).strip())
        if current_code:
            codes.add(str(current_code).strip())
        for code in sorted(codes):
            self.exam_code_correction_combo.addItem(code, code)
        if self.exam_code_correction_combo.count() == 0:
            self.exam_code_correction_combo.addItem(current_code or "-", current_code or "")
        match_index = max(0, self.exam_code_correction_combo.findData(current_code))
        self.exam_code_correction_combo.setCurrentIndex(match_index)
        self.exam_code_correction_combo.blockSignals(False)

    def _load_student_correction_options(self, current_student_id: str) -> None:
        cache_session_id = str(getattr(self, "_student_option_cache_session_id", "") or "")
        current_session_id = str(self.current_session_id or "")
        session_signature = ""
        if self.session and str(getattr(self.session, "session_id", "") or "") == current_session_id:
            sig_parts: list[str] = []
            for st in (self.session.students or []):
                sid = str(getattr(st, "student_id", "") or "").strip()
                if not sid:
                    continue
                extra = getattr(st, "extra", {}) if isinstance(getattr(st, "extra", {}), dict) else {}
                sig_parts.append(f"{sid}|{str(getattr(st, 'name', '') or '').strip()}|{str(extra.get('class_name', '') or '').strip()}")
            session_signature = "||".join(sorted(sig_parts))
        if cache_session_id != current_session_id:
            self._student_option_cache_session_id = current_session_id
            self._student_option_labels_cache = []
            self._student_option_sid_map = {}
            self._student_option_profile_map = {}
            self._student_option_sid_set = set()
            self._student_option_cache_signature = ""
        if session_signature and session_signature != str(getattr(self, "_student_option_cache_signature", "") or ""):
            self._student_option_labels_cache = []
            self._student_option_sid_map = {}
            self._student_option_profile_map = {}
            self._student_option_sid_set = set()
            self._student_option_cache_signature = session_signature

        if not isinstance(getattr(self, "_student_option_labels_cache", None), list) or not self._student_option_labels_cache:
            students: list[tuple[str, str, str]] = []
            if self.session and str(getattr(self.session, "session_id", "") or "") == current_session_id:
                for st in (self.session.students or []):
                    sid = str(getattr(st, "student_id", "") or "").strip()
                    if not sid:
                        continue
                    extra = getattr(st, "extra", {}) if isinstance(getattr(st, "extra", {}), dict) else {}
                    students.append((sid, str(getattr(st, "name", "") or "").strip(), str(extra.get("class_name", "") or "").strip()))
            elif current_session_id:
                payload = self.database.fetch_exam_session(current_session_id) or {}
                session_students = payload.get("students", []) if isinstance(payload.get("students", []), list) else []
                for item in session_students:
                    if not isinstance(item, dict):
                        continue
                    extra = item.get("extra", {}) if isinstance(item.get("extra", {}), dict) else {}
                    sid = str(item.get("student_id", "") or "").strip()
                    if not sid:
                        continue
                    students.append((sid, str(item.get("name", "") or "").strip(), str(extra.get("class_name", "") or "").strip()))
            labels: list[str] = []
            sid_map: dict[str, str] = {}
            profile_map: dict[str, dict[str, str]] = {}
            sid_set: set[str] = set()
            for sid, name, class_name in students:
                if sid in sid_set:
                    continue
                sid_set.add(sid)
                label = f"[{sid}] - {name or '-'} - {class_name or '-'}"
                labels.append(label)
                sid_map[label] = sid
                profile_map[sid] = {"name": name, "class_name": class_name}
            self._student_option_labels_cache = labels
            self._student_option_sid_map = sid_map
            self._student_option_profile_map = profile_map
            self._student_option_sid_set = sid_set
            if session_signature:
                self._student_option_cache_signature = session_signature

        self.student_correction_combo.blockSignals(True)
        self.student_correction_combo.clear()
        for label in list(self._student_option_labels_cache):
            sid = str(self._student_option_sid_map.get(label, "") or "")
            if sid:
                self.student_correction_combo.addItem(label, sid)
        if current_student_id and current_student_id not in set(self._student_option_sid_set):
            label = f"[{current_student_id}] - - -"
            self.student_correction_combo.addItem(label, current_student_id)
        idx = self.student_correction_combo.findData(current_student_id)
        self.student_correction_combo.setCurrentIndex(max(0, idx))
        completer = QCompleter(list(self._student_option_labels_cache), self.student_correction_combo)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.student_correction_combo.setCompleter(completer)
        self.student_correction_combo.blockSignals(False)

    def _build_visual_answer_editor(self, result: OMRResult) -> None:
        while self.answer_editor_layout.count():
            item = self.answer_editor_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        result = self._scoped_result_copy(result)
        expected = self._expected_questions_by_section(result)

        if expected.get("MCQ", []):
            mcq_box = QGroupBox("MCQ")
            mcq_layout = QVBoxLayout(mcq_box)
            for q_no in expected.get("MCQ", []):
                row = QHBoxLayout()
                row.addWidget(QLabel(f"Câu {q_no}"))
                group = QButtonGroup(mcq_box)
                current_value = str((result.mcq_answers or {}).get(q_no, "") or "")[:1]
                for choice in ["A", "B", "C", "D", "E"]:
                    radio = QRadioButton(choice)
                    if current_value == choice:
                        radio.setChecked(True)
                    radio.toggled.connect(lambda checked, q=q_no, v=choice: checked and self._handle_mcq_visual_change(q, v))
                    group.addButton(radio)
                    row.addWidget(radio)
                clear_btn = QPushButton("Clear")
                clear_btn.clicked.connect(lambda _=False, q=q_no: self._handle_mcq_visual_change(q, ""))
                row.addWidget(clear_btn)
                row.addStretch()
                mcq_layout.addLayout(row)
            self.answer_editor_layout.addWidget(mcq_box)

        if expected.get("TF", []):
            tf_box = QGroupBox("True/False")
            tf_layout = QVBoxLayout(tf_box)
            for q_no in expected.get("TF", []):
                row = QHBoxLayout()
                row.addWidget(QLabel(f"Câu {q_no}"))
                flags = (result.true_false_answers or {}).get(q_no, {}) or {}
                for key in ["a", "b", "c", "d"]:
                    cb = QCheckBox(key.upper())
                    cb.setChecked(bool(flags.get(key)))
                    cb.toggled.connect(lambda checked, q=q_no, k=key: self._handle_tf_visual_change(q, k, checked))
                    row.addWidget(cb)
                row.addStretch()
                tf_layout.addLayout(row)
            self.answer_editor_layout.addWidget(tf_box)

        if expected.get("NUMERIC", []):
            num_box = QGroupBox("Numeric")
            num_layout = QVBoxLayout(num_box)
            for q_no in expected.get("NUMERIC", []):
                row = QHBoxLayout()
                row.addWidget(QLabel(f"Câu {q_no}"))
                edit = QLineEdit(str((result.numeric_answers or {}).get(q_no, "") or ""))
                edit.editingFinished.connect(lambda q=q_no, w=edit: self._handle_numeric_visual_change(q, w.text()))
                row.addWidget(edit)
                num_layout.addLayout(row)
            self.answer_editor_layout.addWidget(num_box)
        self.answer_editor_layout.addStretch()

    def _ensure_correction_state(self) -> None:
        if not hasattr(self, "correction_ui_loading"):
            self.correction_ui_loading = False
        if not hasattr(self, "correction_pending_payload") or not isinstance(getattr(self, "correction_pending_payload", None), dict):
            self.correction_pending_payload = {}
        if not hasattr(self, "correction_save_timer") or not isinstance(getattr(self, "correction_save_timer", None), QTimer):
            self.correction_save_timer = QTimer(self)
            self.correction_save_timer.setSingleShot(True)
            self.correction_save_timer.timeout.connect(self._flush_pending_correction_updates)

    def _schedule_correction_update(self, field_name: str, old_value: object, new_value: object, apply_fn) -> None:
        self._ensure_correction_state()
        if self.correction_ui_loading or old_value == new_value:
            return
        apply_fn()
        self._refresh_all_statuses()
        self.correction_pending_payload[field_name] = {
            "old": old_value,
            "new": new_value,
        }
        self.correction_save_timer.start(150)

    def _flush_pending_correction_updates(self) -> None:
        self._ensure_correction_state()
        idx, result = self._correction_selected_result()
        if idx is None or result is None or not self.correction_pending_payload:
            return
        changes = [f"{field}: '{payload['old']}' -> '{payload['new']}'" for field, payload in self.correction_pending_payload.items()]
        self._mark_result_manually_edited(result, idx)
        self._refresh_student_profile_for_result(result, idx)
        scoped = self._scoped_result_copy(result)
        self.scan_blank_summary[idx] = self._compute_blank_questions(scoped)
        expected = self._expected_questions_by_section(scoped)
        self.scan_list.setItem(
            idx,
            self.SCAN_COL_CONTENT,
            QTableWidgetItem(self._build_recognition_content_text(result, self.scan_blank_summary[idx], expected)),
        )
        sid_item = self.scan_list.item(idx, self.SCAN_COL_STUDENT_ID)
        if sid_item:
            sid_item.setText((result.student_id or "").strip() or "-")
            sid_item.setData(Qt.UserRole + 1, result.exam_code or "")
            sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(scoped))
        self._record_adjustment(idx, changes, "visual_correction")
        self._persist_single_scan_result_to_db(result, note="visual_correction")
        self._sync_subject_batch_snapshot_after_inline_edit(persist_full_subject_rows=False)
        image_key = str(getattr(result, "image_path", "") or idx)
        for field_name, payload in self.correction_pending_payload.items():
            self.database.log_change("scan_results", image_key, field_name, payload["old"], payload["new"], "visual_correction")
        self.correction_pending_payload = {}
        self._refresh_all_statuses()
        self._update_scan_preview(idx)
        self.correction_ui_loading = True
        self._sync_correction_detail_panel(result, rebuild_editor=False)
        self.correction_ui_loading = False

    def _handle_exam_code_correction_changed(self, _index: int) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        new_code = str(self.exam_code_correction_combo.currentData() or "").strip()
        old_code = str(result.exam_code or "").strip()
        self._schedule_correction_update("exam_code", old_code, new_code, lambda: setattr(result, "exam_code", new_code))

    def _handle_student_correction_changed(self, _index: int) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        new_sid = str(self.student_correction_combo.currentData() or self.student_correction_combo.currentText() or "").strip()
        old_sid = str(result.student_id or "").strip()
        self._schedule_correction_update("student_id", old_sid, new_sid, lambda: setattr(result, "student_id", new_sid))

    def _handle_mcq_visual_change(self, question_no: int, answer_value: str) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        old_value = str((result.mcq_answers or {}).get(question_no, "") or "")[:1]

        def _apply() -> None:
            current = dict(result.mcq_answers or {})
            if answer_value:
                current[int(question_no)] = str(answer_value)[:1]
            else:
                current.pop(int(question_no), None)
            result.mcq_answers = current

        self._schedule_correction_update(f"mcq_answers[{question_no}]", old_value, str(answer_value)[:1], _apply)

    def _handle_tf_visual_change(self, question_no: int, key: str, checked: bool) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        old_flags = dict((result.true_false_answers or {}).get(question_no, {}) or {})
        new_flags = dict(old_flags)
        new_flags[key] = bool(checked)

        def _apply() -> None:
            current = dict(result.true_false_answers or {})
            current[int(question_no)] = dict(new_flags)
            result.true_false_answers = current

        self._schedule_correction_update(f"true_false_answers[{question_no}].{key}", old_flags.get(key), bool(checked), _apply)

    def _handle_numeric_visual_change(self, question_no: int, text: str) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        old_value = str((result.numeric_answers or {}).get(question_no, "") or "")
        new_value = str(text or "").strip()

        def _apply() -> None:
            current = dict(result.numeric_answers or {})
            if new_value:
                current[int(question_no)] = new_value
            else:
                current.pop(int(question_no), None)
            result.numeric_answers = current

        self._schedule_correction_update(f"numeric_answers[{question_no}]", old_value, new_value, _apply)

    def _analyze_scan_status(
        self,
        result,
        duplicate_count: int,
        subject_scope: tuple[set[str], set[str]] | None = None,
        available_exam_codes: set[str] | None = None,
        forced_status: str = "",
    ) -> dict:
        status_parts = self._status_parts_for_result(
            result,
            duplicate_count,
            subject_scope=subject_scope,
            available_exam_codes=available_exam_codes,
        )
        status_parts_text = ", ".join(status_parts) if status_parts else ""
        has_edit_history = bool([str(x) for x in (getattr(result, "edit_history", []) or []) if str(x or "").strip()])
        is_manual_edited = bool(getattr(result, "manually_edited", False)) or has_edit_history
        forced_status_text = str(forced_status or "").strip()
        has_duplicate = "Trùng SBD" in status_parts
        has_wrong_code = "Mã đề không hợp lệ" in status_parts
        has_error = bool(status_parts)
        if is_manual_edited:
            if status_parts_text:
                status_text = f"Đã sửa ({status_parts_text})"
            elif forced_status_text and forced_status_text not in {"Đã sửa", "OK"}:
                status_text = f"Đã sửa ({forced_status_text})"
            else:
                status_text = "Đã sửa"
            effective_forced_status = "Đã sửa"
        else:
            effective_forced_status = forced_status_text
            status_text = effective_forced_status or status_parts_text or "OK"
        return {
            "status_parts": list(status_parts),
            "status_parts_text": status_parts_text,
            "status_text": status_text,
            "forced_status_text": forced_status_text,
            "effective_forced_status": effective_forced_status,
            "has_edit_history": has_edit_history,
            "is_manual_edited": is_manual_edited,
            "has_error": has_error,
            "has_duplicate": has_duplicate,
            "has_wrong_code": has_wrong_code,
            "is_clean_ok": (not is_manual_edited) and (not has_error),
            "is_edited_clean": bool(is_manual_edited and (not has_error)),
            "is_edited_with_error": bool(is_manual_edited and has_error),
        }

    @staticmethod
    def _normalize_exam_code_text(code: str) -> str:
        c = str(code or "").strip()
        if not c:
            return ""
        if c.isdigit():
            # Treat purely numeric exam codes with/without leading zeros as equivalent.
            c2 = c.lstrip("0")
            return c2 if c2 else "0"
        return c

    def _status_parts_for_row(
        self,
        sid: str,
        exam_code_text: str,
        duplicate_count: int,
        subject_scope: tuple[set[str], set[str]] | None = None,
        available_exam_codes: set[str] | None = None,
        result: OMRResult | None = None,
    ) -> list[str]:
        sid_text = str(sid or "").strip()
        has_duplicate = False
        status_parts: list[str] = []

        if self._student_id_has_recognition_error(sid_text):
            status_parts.append("SBD không nhận dạng")
            has_duplicate = duplicate_count > 1
        else:
            has_duplicate = duplicate_count > 1
            all_sids, _room_sids = subject_scope if subject_scope is not None else self._subject_student_room_scope()
            sid_norm = self._normalized_student_id_for_match(sid_text)
            profile = self._resolve_student_profile_for_status(sid_text)
            cfg = self._selected_batch_subject_config() or {}

            if not self._has_valid_student_reference(sid_text):
                status_parts.append("SBD không có trong danh sách")
            elif all_sids and sid_norm not in all_sids:
                status_parts.append("SBD không có trong danh sách")
            else:
                room_text = self._subject_room_for_student_id(sid_text, cfg)
                if self._is_missing_room_for_status(room_text):
                    status_parts.append("Thiếu phòng thi")
                else:
                    mapping_by_room = self._normalized_exam_room_mapping_by_room(cfg)
                    if mapping_by_room:
                        resolved_norm = self._normalized_room_for_match(room_text)
                        resolved_candidates = [
                            room for room in mapping_by_room.keys()
                            if self._normalized_room_for_match(room) == resolved_norm
                        ]
                        if resolved_candidates:
                            resolved_room_sids: set[str] = set()
                            for room in resolved_candidates:
                                resolved_room_sids.update(mapping_by_room.get(room, set()))
                            if sid_norm not in resolved_room_sids:
                                status_parts.append("SBD không thuộc phòng thi môn")
                if self._is_missing_name_for_status(str(profile.get("name", "") or "")):
                    status_parts.append("Thiếu họ tên")

        if not self._is_valid_exam_code_for_subject(exam_code_text, available_exam_codes=available_exam_codes):
            status_parts.append("Mã đề không hợp lệ")
        if has_duplicate:
            status_parts.append("Trùng SBD")

        if result is not None:
            rec_errors = list(getattr(result, "recognition_errors", [])) or list(getattr(result, "errors", []))
            issue_codes = [str(getattr(issue, "code", "") or "").strip().upper() for issue in (getattr(result, "issues", []) or [])]
            rec_error_codes = [str(err or "").strip().upper() for err in rec_errors if str(err or "").strip()]
            if rec_errors or issue_codes:
                if any(code in {"ANCHOR_MISSING", "ANCHOR_FAIL", "SCANNER_LOCK_FAIL"} for code in issue_codes) or any("ANCHOR" in code or "SCANNER_LOCK_FAIL" in code for code in rec_error_codes):
                    status_parts.append("Lỗi nhận dạng anchor")
                elif any(code in {"POOR_IDENTIFIER_ZONE", "STUDENT_ID_FAST_FAIL", "EXAM_CODE_FAST_FAIL"} for code in issue_codes) or any(code in {"POOR_IDENTIFIER_ZONE", "STUDENT_ID_FAST_FAIL", "EXAM_CODE_FAST_FAIL"} or "HEADER" in code for code in rec_error_codes):
                    status_parts.append("Lỗi nhận dạng vùng header")
                else:
                    status_parts.append("Lỗi nhận dạng")
        return list(dict.fromkeys([x for x in status_parts if str(x or "").strip()]))

    @staticmethod
    def _student_id_has_recognition_error(student_id: str) -> bool:
        sid = str(student_id or "").strip()
        return sid in {"", "-"} or ("?" in sid)

    def _status_parts_for_result(
        self,
        result,
        duplicate_count: int,
        subject_scope: tuple[set[str], set[str]] | None = None,
        available_exam_codes: set[str] | None = None,
    ) -> list[str]:
        sid = str(getattr(result, "student_id", "") or "").strip()
        exam_code_text = str(getattr(result, "exam_code", "") or "").strip()
        return self._status_parts_for_row(
            sid,
            exam_code_text,
            duplicate_count,
            subject_scope=subject_scope,
            available_exam_codes=available_exam_codes,
            result=result,
        )

    def _status_text_for_row(
        self,
        idx: int,
        duplicate_count_map: dict[str, int] | None = None,
        subject_scope: tuple[set[str], set[str]] | None = None,
        available_exam_codes: set[str] | None = None,
    ) -> str:
        if idx < 0 or idx >= len(self.scan_results):
            return "OK"
        res = self.scan_results[idx]
        sid = (res.student_id or "").strip()
        if self._student_id_has_recognition_error(sid):
            dup = 0
        elif duplicate_count_map is not None:
            dup = int(duplicate_count_map.get(sid, 0) or 0)
        else:
            dup = sum(
                1
                for r in self.scan_results
                if not self._student_id_has_recognition_error((r.student_id or "").strip()) and (r.student_id or "").strip() == sid
            )
        status_parts = self._status_parts_for_result(
            res,
            dup,
            subject_scope=subject_scope,
            available_exam_codes=available_exam_codes,
        )
        return ", ".join(status_parts) if status_parts else "OK"

    def _current_batch_subject_key(self) -> str:
        cfg = self._selected_batch_subject_config()
        if cfg:
            return self._subject_key_from_cfg(cfg)
        return str(self.active_batch_subject_key or "").strip() or self._resolve_preferred_scoring_subject()

    @staticmethod
    def _ordered_unique_text_list(values) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for value in list(values or []):
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _record_adjustment(self, idx: int, details: list[str], source: str) -> None:
        image_key = self._row_image_key(idx)
        if not image_key or not details:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detail = "; ".join(str(change or "").strip() for change in details if str(change or "").strip())
        if not detail:
            return
        entry = f"[{timestamp}] {source}: {detail}"
        history = list(self.scan_edit_history.get(image_key, []))
        history.append(entry)
        self.scan_edit_history[image_key] = self._ordered_unique_text_list(history)
        self.scan_last_adjustment[image_key] = entry
        adjustments = list(self.scan_manual_adjustments.get(image_key, []))
        adjustments.extend(str(change or "").strip() for change in details if str(change or "").strip())
        self.scan_manual_adjustments[image_key] = self._ordered_unique_text_list(adjustments)
        target_result = self._result_by_image_path(image_key)
        if target_result is not None:
            setattr(target_result, "manually_edited", True)
            setattr(target_result, "edit_history", list(self.scan_edit_history.get(image_key, [])))
            setattr(target_result, "last_adjustment", entry)
            setattr(target_result, "manual_adjustments", list(self.scan_manual_adjustments.get(image_key, [])))
            setattr(target_result, "cached_forced_status", "Đã sửa")
            setattr(target_result, "cached_status", "Đã sửa")
            subject_key = str(self._current_batch_subject_key() or "").strip()
            if subject_key:
                self._persist_result_edit_registry(subject_key, target_result)
        self.scan_forced_status_by_index[image_key] = "Đã sửa"

    def _on_scan_cell_clicked(self, row: int, col: int) -> None:
        if row < 0:
            return
        # Show edit history from Status / Actions columns for quick access in all status scenarios.
        if col not in {self.SCAN_COL_STATUS, self.SCAN_COL_ACTIONS}:
            return
        image_key = self._row_image_key(row)
        history = list(self.scan_edit_history.get(image_key, []) or [])
        if (not history) and image_key:
            result = self._result_by_image_path(image_key)
            history = [str(x) for x in (getattr(result, "edit_history", []) or []) if str(x or "").strip()] if result is not None else []
            if history:
                self.scan_edit_history[image_key] = list(history)
                self.scan_last_adjustment[image_key] = str(getattr(result, "last_adjustment", "") or history[-1])
        if not history:
            sid_item = self.scan_list.item(row, self.SCAN_COL_STUDENT_ID)
            row_payload = dict(sid_item.data(Qt.UserRole + 12) or {}) if sid_item is not None else {}
            history = [str(x) for x in (row_payload.get("edit_history", []) or []) if str(x or "").strip()]
            if history and image_key:
                self.scan_edit_history[image_key] = list(history)
                self.scan_last_adjustment[image_key] = str(row_payload.get("last_adjustment", "") or history[-1])
        if not history:
            QMessageBox.information(self, "Lịch sử sửa", "Chưa có lịch sử điều chỉnh trong Status cho bài thi này.")
            return
        latest = self.scan_last_adjustment.get(image_key, history[-1])
        QMessageBox.information(
            self,
            "Lịch sử sửa bài",
            "Điều chỉnh gần nhất:\n"
            + latest
            + "\n\nToàn bộ lịch sử:\n"
            + "\n".join(history),
        )

    def _sync_correction_detail_panel(self, res: OMRResult, rebuild_editor: bool = False) -> None:
        subject_key = self._current_batch_subject_key()
        self._load_exam_code_correction_options(subject_key, str(res.exam_code or "").strip())
        self._load_student_correction_options(str(res.student_id or "").strip())
        scoped = self._scoped_result_copy(res)
        if rebuild_editor:
            self._build_visual_answer_editor(scoped)
        payload = {
            "student_id": res.student_id,
            "exam_code": res.exam_code,
            "answer_string": str(getattr(scoped, "answer_string", "") or self._build_answer_string_for_result(scoped, subject_key)),
            "mcq_answers": scoped.mcq_answers,
            "true_false_answers": scoped.true_false_answers,
            "numeric_answers": scoped.numeric_answers,
            "issues": [{"code": i.code, "message": i.message, "zone_id": i.zone_id} for i in res.issues],
            "recognition_errors": list(getattr(res, "recognition_errors", [])) or list(getattr(res, "errors", [])),
        }
        self.result_preview.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self.manual_edit.setPlainText(
            json.dumps(
                {
                    "student_id": res.student_id,
                    "exam_code": res.exam_code,
                    "answer_string": payload["answer_string"],
                    "mcq_answers": scoped.mcq_answers,
                    "true_false_answers": scoped.true_false_answers,
                    "numeric_answers": scoped.numeric_answers,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        self.correction_ui_loading = False

    def _load_selected_result_for_correction(self) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0:
            return
        res = self._restore_full_result_for_row(idx)
        if res is None:
            return
        self._trim_result_answers_to_expected_scope(res)
        self._set_scan_result_at_row(idx, res)
        self.correction_ui_loading = True
        self._sync_correction_detail_panel(res, rebuild_editor=True)
        self.correction_ui_loading = False

    def _open_edit_selected_scan(self, *_args) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "No selection", "Chọn bài thi cần sửa trước.")
            return
        restored_for_edit = self._restore_full_result_for_row(idx)
        if restored_for_edit is not None and idx >= len(self.scan_results):
            self._set_scan_result_at_row(idx, restored_for_edit)
        if idx >= len(self.scan_results):
            sid_item_existing = self.scan_list.item(idx, self.SCAN_COL_STUDENT_ID)
            sid = sid_item_existing.text() if sid_item_existing else "-"
            content = ""
            if restored_for_edit is not None:
                scoped_for_edit = self._scoped_result_copy(restored_for_edit)
                content = str(self._build_answer_string_for_result(scoped_for_edit, self._current_batch_subject_key()) or "").strip()
                if not content:
                    content = str(getattr(scoped_for_edit, "cached_recognized_short", "") or "").strip()
                if not content:
                    content = str(self._short_recognition_text_for_result(scoped_for_edit) or "").strip()
                if not content:
                    content = str(getattr(restored_for_edit, "manual_content_override", "") or "").strip()
            if not content:
                content = str(sid_item_existing.data(Qt.UserRole + 2) if sid_item_existing else "").strip()
            if not content:
                content = self.scan_list.item(idx, self.SCAN_COL_CONTENT).text() if self.scan_list.item(idx, self.SCAN_COL_CONTENT) else "-"
            exam_code = str(sid_item_existing.data(Qt.UserRole + 1) if sid_item_existing else "").strip()
            if not exam_code:
                for r in range(self.scan_result_preview.rowCount()):
                    k = self.scan_result_preview.item(r, 0)
                    v = self.scan_result_preview.item(r, 1)
                    if k and v and k.text().strip().lower() in {"exam code", "mã đề"}:
                        exam_code = v.text().strip()
                        break
            dlg = QDialog(self)
            dlg.setWindowTitle("Sửa bài thi đã lưu")
            lay = QVBoxLayout(dlg)
            form = QFormLayout()
            inp_sid = QLineEdit(sid)
            inp_code = QComboBox()
            inp_code.setEditable(True)
            inp_code.setInsertPolicy(QComboBox.NoInsert)
            subject_cfg = self._selected_batch_subject_config() or {}
            subject_key = self._current_batch_subject_key()
            valid_codes: set[str] = set()
            imported = subject_cfg.get("imported_answer_keys", {}) if isinstance(subject_cfg.get("imported_answer_keys", {}), dict) else {}
            valid_codes.update(str(x).strip() for x in imported.keys() if str(x).strip())
            try:
                valid_codes.update(str(x).strip() for x in self._fetch_answer_keys_for_subject_scoped(subject_key).keys() if str(x).strip())
            except Exception:
                pass
            if self.answer_keys:
                valid_codes.update(
                    str(item.exam_code).strip()
                    for item in self.answer_keys.keys.values()
                    if str(getattr(item, "subject", "") or "").strip() == str(subject_key or "").strip() and str(getattr(item, "exam_code", "") or "").strip()
                )
            if exam_code:
                valid_codes.add(exam_code)
            if valid_codes:
                for code in sorted(valid_codes):
                    inp_code.addItem(code, code)
            else:
                inp_code.addItem("-", "-")
            if exam_code:
                idx_code = inp_code.findData(exam_code)
                if idx_code >= 0:
                    inp_code.setCurrentIndex(idx_code)
                else:
                    inp_code.setEditText(exam_code)
            form.addRow("Student ID", inp_sid)
            form.addRow("Exam Code", inp_code)
            lay.addLayout(form)
            buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            lay.addWidget(buttons)
            if dlg.exec() != QDialog.Accepted:
                return
            old_item = self.scan_list.item(idx, self.SCAN_COL_STUDENT_ID)
            old_sid = old_item.text().strip() if old_item else ""
            old_img = str(old_item.data(Qt.UserRole) if old_item else "")
            old_exam_code = str(old_item.data(Qt.UserRole + 1) if old_item else "").strip()
            old_recognized_short = str(old_item.data(Qt.UserRole + 2) if old_item else "")
            new_sid_text = inp_sid.text().strip()
            new_exam_code = str(inp_code.currentData() or inp_code.currentText() or "").strip() or old_exam_code
            sid_item = QTableWidgetItem(new_sid_text or "-")
            sid_item.setData(Qt.UserRole, old_img)
            sid_item.setData(Qt.UserRole + 1, new_exam_code)
            sid_item.setData(Qt.UserRole + 2, old_recognized_short)
            self.scan_list.setItem(idx, self.SCAN_COL_STUDENT_ID, sid_item)
            self.scan_list.setItem(idx, self.SCAN_COL_EXAM_CODE, QTableWidgetItem(new_exam_code or "-"))
            changes: list[str] = []
            if new_sid_text != old_sid:
                changes.append(f"student_id: '{old_sid}' -> '{new_sid_text}'")
            if new_exam_code != old_exam_code:
                changes.append(f"exam_code: '{old_exam_code}' -> '{new_exam_code}'")
            rebuilt = self._build_result_from_saved_table_row(idx)
            if rebuilt is not None:
                setattr(rebuilt, "manual_content_override", "")
                if changes:
                    self._mark_result_manually_edited(rebuilt, idx)
                self._refresh_student_profile_for_result(rebuilt, idx)
                self._set_scan_result_at_row(idx, rebuilt)
                subject_key_now = self._current_batch_subject_key()
                if subject_key_now:
                    self.scan_results_by_subject[self._batch_result_subject_key(subject_key_now)] = list(self.scan_results)
                if changes:
                    self._record_adjustment(idx, changes, "saved_row_edit")
                self._update_scan_row_from_result(idx, rebuilt)
                if changes:
                    self._persist_single_scan_result_to_db(rebuilt, note="saved_row_edit")
                    self._sync_subject_batch_snapshot_after_inline_edit(persist_full_subject_rows=False)
                    self._refresh_all_statuses()
                else:
                    self._refresh_row_status(idx)
            else:
                self._refresh_row_status(idx)
            for r in range(self.scan_result_preview.rowCount()):
                k = self.scan_result_preview.item(r, 0)
                if k and k.text().strip().lower() in {"exam code", "mã đề"}:
                    self.scan_result_preview.setItem(r, 1, QTableWidgetItem(new_exam_code or "-"))
                    break
            if self.scan_list.currentRow() == idx:
                self._update_scan_preview_from_saved_row(idx)
            self.btn_save_batch_subject.setEnabled(False)
            invalidated = self._invalidate_scoring_for_student_ids(
                [old_sid, inp_sid.text().strip()],
                reason="saved_row_edit",
            )
            if invalidated > 0:
                QMessageBox.information(
                    self,
                    "Tính điểm",
                    f"Đã đánh dấu {invalidated} bản ghi cần chấm lại do sửa bài. Vui lòng chạy lại Tính điểm.",
                )
            return
        dialog_state = {"index": idx, "loading": False}
        dialog_requires_full_status_refresh = {"value": False}
        default_zoom_factor = 0.34

        dlg = QDialog(None)
        dlg.setWindowTitle(f"Sửa bài thi: {Path(self.scan_results[dialog_state['index']].image_path).name}")
        dlg.setModal(True)
        dlg.setWindowFlag(Qt.Window, True)
        screen_geom = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else self.geometry()
        dlg.setGeometry(screen_geom)
        dlg.setMinimumSize(screen_geom.size())
        dlg.setMaximumSize(screen_geom.size())
        QTimer.singleShot(0, lambda: (dlg.raise_(), dlg.showMaximized(), dlg.activateWindow()))
        lay = QHBoxLayout(dlg)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_lay = QVBoxLayout(left)
        form = QFormLayout()

        inp_sid = QLineEdit()

        inp_code = QComboBox()
        inp_code.setEditable(True)
        inp_code.setInsertPolicy(QComboBox.NoInsert)

        mcq_label = QLabel("MCQ")
        mcq_hint = QLabel("Nhập đáp án trực tiếp vào từng ô MCQ")
        tf_label = QLabel("True / False")
        numeric_label = QLabel("Numeric")
        mcq_host = QWidget()
        mcq_host_lay = QVBoxLayout(mcq_host)
        mcq_host_lay.setContentsMargins(0, 0, 0, 0)
        tf_host = QWidget()
        tf_host_lay = QVBoxLayout(tf_host)
        tf_host_lay.setContentsMargins(0, 0, 0, 0)
        numeric_host = QWidget()
        numeric_host_lay = QVBoxLayout(numeric_host)
        numeric_host_lay.setContentsMargins(0, 0, 0, 0)

        form.addRow("Student ID", inp_sid)
        form.addRow("Exam Code", inp_code)
        left_lay.addLayout(form)
        answer_grid = QTableWidget(0, 3)
        answer_grid.setHorizontalHeaderLabels(["Phần", "Câu", "Bài làm"])
        answer_grid.verticalHeader().setVisible(False)
        answer_grid.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        answer_grid.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        answer_grid.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        _orig_answer_grid_key_release = answer_grid.keyReleaseEvent

        def _answer_grid_key_release(event) -> None:
            _orig_answer_grid_key_release(event)
            if answer_grid.currentColumn() != 2:
                return
            sec_item = answer_grid.item(answer_grid.currentRow(), 0)
            section_text = str(sec_item.text() if sec_item else "").strip().upper()
            cur_item = answer_grid.currentItem()
            if cur_item is not None:
                txt = str(cur_item.text() or "")
                up_txt = txt.upper()
                if txt != up_txt:
                    answer_grid.blockSignals(True)
                    cur_item.setText(up_txt)
                    answer_grid.blockSignals(False)
            key_code = int(getattr(event, "key", lambda: 0)() or 0)
            move_next = False
            if section_text == "MCQ":
                move_next = (65 <= key_code <= 90) or (48 <= key_code <= 57)
            elif section_text in {"TF", "NUMERIC"}:
                move_next = key_code in {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab}
            if move_next:
                row_local = answer_grid.currentRow()
                if 0 <= row_local < answer_grid.rowCount() - 1:
                    answer_grid.setCurrentCell(row_local + 1, 2)
                elif row_local == answer_grid.rowCount() - 1:
                    answer_grid.setCurrentCell(row_local, 2)

        answer_grid.keyReleaseEvent = _answer_grid_key_release  # type: ignore[method-assign]

        def _normalize_answer_grid_cell_text(item: QTableWidgetItem) -> None:
            if item is None or item.column() != 2:
                return
            txt = str(item.text() or "")
            up_txt = txt.upper()
            if txt != up_txt:
                answer_grid.blockSignals(True)
                item.setText(up_txt)
                answer_grid.blockSignals(False)

        answer_grid.itemChanged.connect(_normalize_answer_grid_cell_text)
        left_lay.addWidget(answer_grid, 1)
        left_lay.addWidget(mcq_label)
        left_lay.addWidget(mcq_host)
        mcq_hint_row = QWidget()
        row_mcq = QHBoxLayout(mcq_hint_row)
        row_mcq.setContentsMargins(0, 0, 0, 0)
        row_mcq.addWidget(mcq_hint)
        row_mcq.addStretch()
        left_lay.addWidget(mcq_hint_row)
        left_lay.addWidget(tf_label)
        left_lay.addWidget(tf_host)
        tf_actions_row = QWidget()
        tf_actions = QHBoxLayout(tf_actions_row)
        tf_actions.setContentsMargins(0, 0, 0, 0)
        btn_add_tf = QPushButton("Thêm dòng TF")
        btn_del_tf = QPushButton("Xoá dòng chọn")
        tf_actions.addWidget(btn_add_tf)
        tf_actions.addWidget(btn_del_tf)
        tf_actions.addStretch()
        left_lay.addWidget(tf_actions_row)
        left_lay.addWidget(numeric_label)
        left_lay.addWidget(numeric_host)
        num_actions_row = QWidget()
        num_actions = QHBoxLayout(num_actions_row)
        num_actions.setContentsMargins(0, 0, 0, 0)
        btn_add_num = QPushButton("Thêm dòng Numeric")
        btn_del_num = QPushButton("Xoá dòng chọn")
        num_actions.addWidget(btn_add_num)
        num_actions.addWidget(btn_del_num)
        num_actions.addStretch()
        left_lay.addWidget(num_actions_row)

        button_row = QHBoxLayout()
        btn_prev = QPushButton("← Bài trước")
        btn_next = QPushButton("Bài tiếp →")
        btn_save = QPushButton("Save")
        btn_close = QPushButton("Close")
        button_row.addWidget(btn_prev)
        button_row.addWidget(btn_next)
        button_row.addStretch()
        button_row.addWidget(btn_save)
        button_row.addWidget(btn_close)
        left_lay.addLayout(button_row)

        splitter.addWidget(left)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.addWidget(QLabel("Ảnh bài làm"))
        zoom_row = QHBoxLayout()
        btn_prev_top = QPushButton("←")
        btn_zoom_out_dlg = QPushButton("-")
        btn_zoom_reset_dlg = QPushButton("34%")
        btn_zoom_in_dlg = QPushButton("+")
        btn_next_top = QPushButton("→")
        for btn in [btn_prev_top, btn_zoom_out_dlg, btn_zoom_reset_dlg, btn_zoom_in_dlg, btn_next_top]:
            btn.setMaximumWidth(60)
        zoom_row.addWidget(btn_prev_top)
        zoom_row.addWidget(btn_zoom_out_dlg)
        zoom_row.addWidget(btn_zoom_reset_dlg)
        zoom_row.addWidget(btn_zoom_in_dlg)
        zoom_row.addWidget(btn_next_top)
        zoom_row.addStretch()
        right_lay.addLayout(zoom_row)
        preview = QLabel()
        preview.setAlignment(Qt.AlignCenter)
        preview.setText("Đang tải ảnh bài làm...")
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(False)
        preview_scroll.setAlignment(Qt.AlignCenter)
        preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        preview_scroll.setFrameShape(QFrame.NoFrame)
        preview_scroll.setWidget(preview)
        right_lay.addWidget(preview_scroll, 1)
        splitter.addWidget(right)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        lay.addWidget(splitter)
        QTimer.singleShot(0, lambda s=splitter: s.setSizes([max(1, s.width() // 2), max(1, s.width() // 2)]))

        editor_refs: dict[str, object] = {"mcq_edits": {}, "table_tf": None, "table_num": None, "answer_grid": answer_grid}
        expected_questions_state: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        question_mapping_state: dict[str, dict[str, dict[int, int]]] = {
            "MCQ": {"display_to_actual": {}, "actual_to_display": {}},
            "TF": {"display_to_actual": {}, "actual_to_display": {}},
            "NUMERIC": {"display_to_actual": {}, "actual_to_display": {}},
        }
        preview_state: dict[str, object] = {"pix": QPixmap(), "image_name": "-", "zoom": default_zoom_factor}
        loaded_snapshots: dict[str, dict[str, object]] = {}
        dialog_saved_images: set[str] = set()

        def _question_numbers(values) -> list[int]:
            out = {int(q) for q in (values or {}).keys() if str(q).strip().lstrip('-').isdigit()}
            return sorted(out)

        def _current_result() -> OMRResult:
            return self.scan_results[dialog_state["index"]]

        def _current_result_image_key() -> str:
            return self._result_identity_key(getattr(_current_result(), "image_path", ""))

        def _snapshot_from_result(result: OMRResult) -> dict[str, object]:
            return {
                "image_path": self._result_identity_key(getattr(result, "image_path", "")),
                "student_id": str(result.student_id or "").strip(),
                "exam_code": str(result.exam_code or "").strip(),
                "mcq_answers": {int(q): str(v) for q, v in (result.mcq_answers or {}).items()},
                "true_false_answers": {int(q): dict(v or {}) for q, v in (result.true_false_answers or {}).items()},
                "numeric_answers": {int(q): str(v) for q, v in (result.numeric_answers or {}).items()},
            }

        def _set_combo_to_value(combo: QComboBox, value: str) -> None:
            idx_found = combo.findData(value)
            if idx_found < 0:
                idx_found = combo.findText(value)
            if idx_found >= 0:
                combo.setCurrentIndex(idx_found)
            else:
                combo.setEditText(value)

        student_label_to_sid = dict(getattr(self, "_student_option_sid_map", {}) or {})
        valid_student_id_set = set(getattr(self, "_student_option_sid_set", set()) or set())
        sid_pattern = re.compile(r"^\[([^\]]+)\]")

        def _normalize_student_id_input(raw_text: str) -> str:
            text = str(raw_text or "").strip()
            if not text:
                return ""
            if text in student_label_to_sid:
                return str(student_label_to_sid[text] or "").strip()
            match = sid_pattern.match(text)
            if match:
                return str(match.group(1) or "").strip()
            return text

        def _configured_exam_codes_for_subject_cfg(subject_cfg: dict | None) -> list[str]:
            codes: set[str] = set()
            cfg = subject_cfg if isinstance(subject_cfg, dict) else {}
            imported = cfg.get("imported_answer_keys", {}) if isinstance(cfg.get("imported_answer_keys", {}), dict) else {}
            codes.update(str(x).strip() for x in imported.keys() if str(x).strip())
            return sorted(codes)

        def _valid_exam_codes(subject_key: str, current_code: str = "", subject_cfg: dict | None = None) -> list[str]:
            codes: set[str] = set()
            subject = str(subject_key or "").strip()
            if not subject:
                return []
            codes.update(_configured_exam_codes_for_subject_cfg(subject_cfg))
            try:
                codes.update(str(x).strip() for x in self._fetch_answer_keys_for_subject_scoped(subject).keys() if str(x).strip())
            except Exception:
                pass
            if self.answer_keys:
                codes.update(
                    str(item.exam_code).strip()
                    for item in self.answer_keys.keys.values()
                    if str(getattr(item, "subject", "") or "").strip() == subject and str(getattr(item, "exam_code", "") or "").strip()
                )
            if str(current_code or "").strip():
                codes.add(str(current_code or "").strip())
            return sorted(codes)

        def _populate_exam_code_combo(combo: QComboBox, subject_key: str, current_code: str) -> None:
            combo.blockSignals(True)
            combo.clear()
            subject_cfg_local = self._selected_batch_subject_config() or {}
            valid_codes = _valid_exam_codes(subject_key, current_code, subject_cfg_local)
            selected_code = str(current_code or "").strip()
            selected_is_valid = bool(selected_code and selected_code in valid_codes)
            if not selected_is_valid:
                combo.addItem("-", "-")
            for code in valid_codes:
                combo.addItem(code, code)
            if combo.count() == 0:
                combo.addItem("-", "-")
            if selected_is_valid:
                _set_combo_to_value(combo, selected_code)
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

        def _answer_key_for_exam_code(exam_code_text: str):
            subject = str(self._current_batch_subject_key() or self.active_batch_subject_key or "").strip()
            code = str(exam_code_text or "").strip()
            if not subject or not self.answer_keys:
                return None
            if code and code != "-" and "?" not in code:
                key = self.answer_keys.get_flexible(subject, code)
                if key is not None:
                    return key
            return None

        def _expected_questions_for_dialog(result: OMRResult, exam_code_text: str, data_snapshot: dict[str, object]) -> dict[str, list[int]]:
            configured_counts = self._subject_section_question_counts(self._current_batch_subject_key())
            default_by_config = {
                sec: list(range(1, max(0, int(configured_counts.get(sec, 0) or 0)) + 1))
                for sec in ["MCQ", "TF", "NUMERIC"]
            }
            mapping_payload = {
                sec: {"display_to_actual": {}, "actual_to_display": {}}
                for sec in ["MCQ", "TF", "NUMERIC"]
            }
            key = _answer_key_for_exam_code(exam_code_text)
            if key is not None:
                full_credit = getattr(key, "full_credit_questions", {}) or {}
                invalid_rows = getattr(key, "invalid_answer_rows", {}) or {}

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

                expected = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys()) | _extra_for_section("MCQ")) or list(default_by_config["MCQ"]),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys()) | _extra_for_section("TF")) or list(default_by_config["TF"]),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys()) | _extra_for_section("NUMERIC")) or list(default_by_config["NUMERIC"]),
                }
            else:
                expected = {
                    "MCQ": list(default_by_config["MCQ"]),
                    "TF": list(default_by_config["TF"]),
                    "NUMERIC": list(default_by_config["NUMERIC"]),
                }
            for sec in ["MCQ", "TF", "NUMERIC"]:
                limit = max(0, int(configured_counts.get(sec, 0) or 0))
                if limit <= 0:
                    expected[sec] = []
                else:
                    actual_questions = sorted(expected.get(sec, []))[:limit]
                    if not actual_questions:
                        display_questions = list(range(1, limit + 1))
                        actual_questions = list(display_questions)
                    else:
                        contiguous_local = actual_questions == list(range(1, len(actual_questions) + 1))
                        display_questions = list(actual_questions) if contiguous_local else list(range(1, len(actual_questions) + 1))
                    expected[sec] = list(display_questions)
                    mapping_payload[sec]["display_to_actual"] = {
                        int(display_q): int(actual_q)
                        for display_q, actual_q in zip(display_questions, actual_questions)
                    }
                    mapping_payload[sec]["actual_to_display"] = {
                        int(actual_q): int(display_q)
                        for display_q, actual_q in zip(display_questions, actual_questions)
                    }
            for sec in ["MCQ", "TF", "NUMERIC"]:
                question_mapping_state[sec]["display_to_actual"] = dict(mapping_payload[sec]["display_to_actual"])
                question_mapping_state[sec]["actual_to_display"] = dict(mapping_payload[sec]["actual_to_display"])
            return expected

        def _clear_layout(layout_obj: QVBoxLayout) -> None:
            while layout_obj.count():
                item = layout_obj.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.deleteLater()
                elif child_layout is not None:
                    while child_layout.count():
                        sub = child_layout.takeAt(0)
                        if sub.widget() is not None:
                            sub.widget().deleteLater()

        def _build_pair_table(question_numbers: list[int], data: dict[int, str], value_placeholder: str = "") -> QTableWidget:
            table = QTableWidget(0, 2)
            table.setHorizontalHeaderLabels(["Câu", "Giá trị"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            rows = list(question_numbers or [])
            for q in rows:
                r = table.rowCount()
                table.insertRow(r)
                table.setItem(r, 0, QTableWidgetItem(str(int(q))))
                item_v = QTableWidgetItem(str((data or {}).get(int(q), "") or ""))
                if value_placeholder:
                    item_v.setToolTip(value_placeholder)
                table.setItem(r, 1, item_v)
            return table

        def _build_mcq_grid(question_numbers: list[int], data: dict[int, str]) -> tuple[QWidget, dict[int, QLineEdit]]:
            box = QWidget()
            grid = QGridLayout(box)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)
            edits: dict[int, QLineEdit] = {}
            questions = list(question_numbers or [])
            cols = 8
            for idx_q, q_no in enumerate(questions):
                row = (idx_q // cols) * 2
                col = idx_q % cols
                lbl = QLabel(str(q_no))
                lbl.setAlignment(Qt.AlignCenter)
                edit = QLineEdit(str((data or {}).get(int(q_no), "") or ""))
                edit.setMaxLength(1)
                edit.setMaximumWidth(52)
                edit.setAlignment(Qt.AlignCenter)
                edits[int(q_no)] = edit
                grid.addWidget(lbl, row, col)
                grid.addWidget(edit, row + 1, col)
            grid.setColumnStretch(cols, 1)
            return box, edits

        def _build_tf_table(question_numbers: list[int], data: dict[int, dict[str, bool]]) -> QTableWidget:
            table = QTableWidget(0, 5)
            labels = ["a", "b", "c", "d"]
            table.setHorizontalHeaderLabels(["Câu", *[s.upper() for s in labels]])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            for c in range(1, 5):
                table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
            rows = list(question_numbers or [])
            for q in rows:
                r = table.rowCount()
                table.insertRow(r)
                table.setItem(r, 0, QTableWidgetItem(str(int(q))))
                flags = dict((data or {}).get(int(q), {}) or {})
                for i, key in enumerate(labels, start=1):
                    cb = QComboBox()
                    cb.addItem("-", None)
                    cb.addItem("Đúng", True)
                    cb.addItem("Sai", False)
                    val = flags.get(key, None)
                    idx_found = 0
                    for k in range(cb.count()):
                        if cb.itemData(k) is val:
                            idx_found = k
                            break
                    cb.setCurrentIndex(idx_found)
                    table.setCellWidget(r, i, cb)
            return table

        def _collect_editor_snapshot(validate: bool = True) -> dict[str, object]:
            result = _current_result()
            student_id_text = _normalize_student_id_input(inp_sid.text())
            exam_code_text = str(inp_code.currentData() or inp_code.currentText() or "").strip()
            if exam_code_text == "-":
                exam_code_text = ""
            if validate:
                valid_exam_codes = _valid_exam_codes(self._current_batch_subject_key(), exam_code_text)
                if valid_student_id_set and student_id_text and student_id_text not in valid_student_id_set:
                    raise ValueError(f"Student ID '{student_id_text}' không có trong danh sách học sinh hợp lệ của ca thi.")
                if valid_exam_codes and exam_code_text not in {"", "-"} and exam_code_text not in valid_exam_codes:
                    raise ValueError(f"Exam code '{exam_code_text}' không có đáp án hợp lệ cho môn hiện tại.")
            snapshot = {
                "image_path": self._result_identity_key(getattr(result, "image_path", "")),
                "student_id": student_id_text,
                "exam_code": exam_code_text,
                "mcq_answers": {},
                "true_false_answers": {},
                "numeric_answers": {},
            }
            answer_grid_local = editor_refs.get("answer_grid")
            if isinstance(answer_grid_local, QTableWidget):
                for r in range(answer_grid_local.rowCount()):
                    sec_item = answer_grid_local.item(r, 0)
                    q_item = answer_grid_local.item(r, 1)
                    v_item = answer_grid_local.item(r, 2)
                    sec = str(sec_item.text() if sec_item else "").strip().upper()
                    q_text = str(q_item.text() if q_item else "").strip()
                    v_text = str(v_item.text() if v_item else "").strip()
                    if not q_text.lstrip('-').isdigit():
                        continue
                    display_q = int(q_text)
                    if sec == "MCQ":
                        actual_q = int(question_mapping_state.get("MCQ", {}).get("display_to_actual", {}).get(display_q, display_q))
                        if v_text:
                            snapshot["mcq_answers"][actual_q] = str(v_text).upper()[:1]
                    elif sec == "TF":
                        actual_q = int(question_mapping_state.get("TF", {}).get("display_to_actual", {}).get(display_q, display_q))
                        compact = "".join(ch for ch in str(v_text).upper() if ch in {"Đ", "D", "S", "_", "T", "F", "1", "0"})
                        flags: dict[str, bool] = {}
                        for i, ch in enumerate(compact[:4]):
                            key_flag = ["a", "b", "c", "d"][i]
                            if ch == "_":
                                continue
                            flags[key_flag] = ch in {"Đ", "D", "T", "1"}
                        if flags:
                            snapshot["true_false_answers"][actual_q] = flags
                    elif sec == "NUMERIC":
                        actual_q = int(question_mapping_state.get("NUMERIC", {}).get("display_to_actual", {}).get(display_q, display_q))
                        snapshot["numeric_answers"][actual_q] = v_text
                return snapshot
            mcq_edits = editor_refs.get("mcq_edits", {}) or {}
            for q_no, edit in mcq_edits.items():
                v_text = str(edit.text() if edit else "").strip().upper()[:1]
                if v_text:
                    actual_q = int(question_mapping_state.get("MCQ", {}).get("display_to_actual", {}).get(int(q_no), int(q_no)))
                    snapshot["mcq_answers"][actual_q] = v_text

            table_num = editor_refs.get("table_num")
            if isinstance(table_num, QTableWidget):
                for r in range(table_num.rowCount()):
                    q_item = table_num.item(r, 0)
                    v_item = table_num.item(r, 1)
                    q_text = str(q_item.text() if q_item else "").strip()
                    v_text = str(v_item.text() if v_item else "").strip()
                    if not q_text and not v_text:
                        continue
                    if validate and not q_text.lstrip('-').isdigit():
                        raise ValueError(f"Numeric dòng {r+1}: Câu phải là số nguyên.")
                    if not q_text.lstrip('-').isdigit() or not v_text:
                        continue
                    display_q = int(q_text)
                    actual_q = int(question_mapping_state.get("NUMERIC", {}).get("display_to_actual", {}).get(display_q, display_q))
                    snapshot["numeric_answers"][actual_q] = v_text

            table_tf = editor_refs.get("table_tf")
            if isinstance(table_tf, QTableWidget):
                labels = ["a", "b", "c", "d"]
                for r in range(table_tf.rowCount()):
                    q_item = table_tf.item(r, 0)
                    q_text = str(q_item.text() if q_item else "").strip()
                    if not q_text:
                        continue
                    if validate and not q_text.lstrip('-').isdigit():
                        raise ValueError(f"TF dòng {r+1}: Câu phải là số nguyên.")
                    if not q_text.lstrip('-').isdigit():
                        continue
                    display_q = int(q_text)
                    q = int(question_mapping_state.get("TF", {}).get("display_to_actual", {}).get(display_q, display_q))
                    flags: dict[str, bool] = {}
                    for i, key in enumerate(labels, start=1):
                        cb = table_tf.cellWidget(r, i)
                        if not isinstance(cb, QComboBox):
                            continue
                        val = cb.currentData()
                        if isinstance(val, bool):
                            flags[key] = val
                    if flags:
                        snapshot["true_false_answers"][q] = flags
            return snapshot

        def _remove_selected_row(table_key: str) -> None:
            table = editor_refs.get(table_key)
            if isinstance(table, QTableWidget):
                row = table.currentRow()
                if row >= 0:
                    table.removeRow(row)

        def _add_pair_row() -> None:
            table = editor_refs.get("table_num")
            if isinstance(table, QTableWidget):
                numeric_limit = len(expected_questions_state.get("NUMERIC", []) or [])
                if numeric_limit > 0 and table.rowCount() >= numeric_limit:
                    return
                r = table.rowCount()
                table.insertRow(r)
                table.setItem(r, 0, QTableWidgetItem(""))
                table.setItem(r, 1, QTableWidgetItem(""))

        def _add_tf_row() -> None:
            table = editor_refs.get("table_tf")
            if isinstance(table, QTableWidget):
                tf_limit = len(expected_questions_state.get("TF", []) or [])
                if tf_limit > 0 and table.rowCount() >= tf_limit:
                    return
                r = table.rowCount()
                table.insertRow(r)
                table.setItem(r, 0, QTableWidgetItem(""))
                for i in range(1, 5):
                    cb = QComboBox()
                    cb.addItem("-", None)
                    cb.addItem("Đúng", True)
                    cb.addItem("Sai", False)
                    table.setCellWidget(r, i, cb)

        btn_add_tf.clicked.connect(_add_tf_row)
        btn_del_tf.clicked.connect(lambda: _remove_selected_row("table_tf"))
        btn_add_num.clicked.connect(_add_pair_row)
        btn_del_num.clicked.connect(lambda: _remove_selected_row("table_num"))

        def _render_preview_for_result(result: OMRResult) -> None:
            image_path = str(getattr(result, "image_path", "") or "").strip()
            image_name = Path(image_path).name if image_path else "-"
            aligned_pix = self._aligned_image_to_qpixmap(getattr(result, "aligned_image", None))
            if not aligned_pix.isNull():
                pix = aligned_pix
            elif hasattr(self, "preview_source_pixmap") and not self.preview_source_pixmap.isNull() and self.scan_list.currentRow() == dialog_state["index"]:
                pix = self.preview_source_pixmap
            else:
                pix = QPixmap(image_path) if image_path else QPixmap()
            preview_state["pix"] = pix
            preview_state["image_name"] = image_name
            preview_state["zoom"] = default_zoom_factor
            _apply_preview_zoom()

        def _apply_preview_zoom() -> None:
            base_pix = preview_state.get("pix", QPixmap())
            image_name = str(preview_state.get("image_name", "-"))
            factor = float(preview_state.get("zoom", default_zoom_factor) or default_zoom_factor)
            if isinstance(base_pix, QPixmap) and not base_pix.isNull():
                scaled = base_pix.scaled(base_pix.size() * factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                preview.setText("")
                preview.setPixmap(scaled)
                preview.resize(scaled.size())
                preview.setMinimumSize(scaled.size())
                preview.setMaximumSize(scaled.size())
                preview.adjustSize()
                preview_scroll.ensureVisible(0, 0)
            else:
                preview.setPixmap(QPixmap())
                preview.setMinimumSize(preview_scroll.viewport().size())
                preview.setMaximumSize(16777215, 16777215)
                preview.setText(f"Không thể tải ảnh bài làm tương ứng.\n{image_name}")
                preview.resize(preview_scroll.viewport().size())
            btn_zoom_reset_dlg.setText(f"{int(round(factor * 100))}%")

        btn_zoom_out_dlg.clicked.connect(lambda: (preview_state.__setitem__("zoom", max(0.1, float(preview_state.get("zoom", default_zoom_factor)) / 1.2)), _apply_preview_zoom()))
        btn_zoom_in_dlg.clicked.connect(lambda: (preview_state.__setitem__("zoom", min(5.0, float(preview_state.get("zoom", default_zoom_factor)) * 1.2)), _apply_preview_zoom()))
        btn_zoom_reset_dlg.clicked.connect(lambda: (preview_state.__setitem__("zoom", default_zoom_factor), _apply_preview_zoom()))

        def _refresh_editor_widgets(data_snapshot: dict[str, object]) -> None:
            result = _current_result()
            expected = _expected_questions_for_dialog(result, str(data_snapshot.get("exam_code", "") or ""), data_snapshot)
            expected_questions_state["MCQ"] = list(expected.get("MCQ", []) or [])
            expected_questions_state["TF"] = list(expected.get("TF", []) or [])
            expected_questions_state["NUMERIC"] = list(expected.get("NUMERIC", []) or [])
            section_limits = self._subject_section_question_counts(self._current_batch_subject_key())
            mcq_map_actual_to_display = question_mapping_state.get("MCQ", {}).get("actual_to_display", {}) or {}
            mcq_data = data_snapshot.get("mcq_answers", {}) or {}
            mcq_limit = int(section_limits.get("MCQ", 0) or 0)
            if mcq_limit <= 0:
                mcq_questions = []
                mcq_data_display = {}
            else:
                mcq_data_display_keys = {
                    int(mcq_map_actual_to_display.get(int(q), int(q)))
                    for q in mcq_data.keys()
                }
                mcq_questions = sorted(set(expected.get("MCQ", []) or []).union(mcq_data_display_keys))
                mcq_questions = list(mcq_questions[:mcq_limit])
                allowed_mcq_questions = set(mcq_questions)
                mcq_data_display = {
                    int(mcq_map_actual_to_display.get(int(q), int(q))): str(v)
                    for q, v in mcq_data.items()
                    if int(mcq_map_actual_to_display.get(int(q), int(q))) in allowed_mcq_questions
                }

            tf_data = data_snapshot.get("true_false_answers", {}) or {}
            tf_map_actual_to_display = question_mapping_state.get("TF", {}).get("actual_to_display", {}) or {}
            tf_limit = int(section_limits.get("TF", 0) or 0)
            if tf_limit <= 0:
                tf_questions = []
                tf_data = {}
            else:
                tf_data_display_keys = {
                    int(tf_map_actual_to_display.get(int(q), int(q)))
                    for q in tf_data.keys()
                }
                tf_questions = sorted(set(expected.get("TF", []) or []).union(tf_data_display_keys))
                tf_questions = list(tf_questions[:tf_limit])
                allowed_tf_questions = set(tf_questions)
                tf_data = {
                    int(tf_map_actual_to_display.get(int(q), int(q))): dict(v or {})
                    for q, v in tf_data.items()
                    if int(tf_map_actual_to_display.get(int(q), int(q))) in allowed_tf_questions
                }

            numeric_data = data_snapshot.get("numeric_answers", {}) or {}
            numeric_map_actual_to_display = question_mapping_state.get("NUMERIC", {}).get("actual_to_display", {}) or {}
            numeric_limit = int(section_limits.get("NUMERIC", 0) or 0)
            if numeric_limit <= 0:
                numeric_questions = []
                numeric_data = {}
            else:
                numeric_data_display_keys = {
                    int(numeric_map_actual_to_display.get(int(q), int(q)))
                    for q in numeric_data.keys()
                }
                numeric_questions = sorted(set(expected.get("NUMERIC", []) or []).union(numeric_data_display_keys))
                numeric_questions = list(numeric_questions[:numeric_limit])
                allowed_numeric_questions = set(numeric_questions)
                numeric_data = {
                    int(numeric_map_actual_to_display.get(int(q), int(q))): str(v)
                    for q, v in numeric_data.items()
                    if int(numeric_map_actual_to_display.get(int(q), int(q))) in allowed_numeric_questions
                }
            grid_local = editor_refs.get("answer_grid")
            if isinstance(grid_local, QTableWidget):
                grid_local.blockSignals(True)
                grid_local.setRowCount(0)
                for q in mcq_questions:
                    r = grid_local.rowCount()
                    grid_local.insertRow(r)
                    grid_local.setItem(r, 0, QTableWidgetItem("MCQ"))
                    grid_local.setItem(r, 1, QTableWidgetItem(str(int(q))))
                    grid_local.setItem(r, 2, QTableWidgetItem(str(mcq_data_display.get(int(q), "") or "")))
                for q in tf_questions:
                    flags = dict(tf_data.get(int(q), {}) or {})
                    token = "".join(("Đ" if bool(flags.get(k)) else ("S" if k in flags else "_")) for k in ["a", "b", "c", "d"])
                    r = grid_local.rowCount()
                    grid_local.insertRow(r)
                    grid_local.setItem(r, 0, QTableWidgetItem("TF"))
                    grid_local.setItem(r, 1, QTableWidgetItem(str(int(q))))
                    grid_local.setItem(r, 2, QTableWidgetItem(token))
                for q in numeric_questions:
                    r = grid_local.rowCount()
                    grid_local.insertRow(r)
                    grid_local.setItem(r, 0, QTableWidgetItem("NUMERIC"))
                    grid_local.setItem(r, 1, QTableWidgetItem(str(int(q))))
                    grid_local.setItem(r, 2, QTableWidgetItem(str(numeric_data.get(int(q), "") or "")))
                for r in range(grid_local.rowCount()):
                    sec_item = grid_local.item(r, 0)
                    q_item = grid_local.item(r, 1)
                    if sec_item:
                        sec_item.setFlags(sec_item.flags() & ~Qt.ItemIsEditable)
                    if q_item:
                        q_item.setFlags(q_item.flags() & ~Qt.ItemIsEditable)
                grid_local.blockSignals(False)
            mcq_label.setVisible(False)
            mcq_host.setVisible(False)
            mcq_hint_row.setVisible(False)
            tf_label.setVisible(False)
            tf_host.setVisible(False)
            tf_actions_row.setVisible(False)
            numeric_label.setVisible(False)
            numeric_host.setVisible(False)
            num_actions_row.setVisible(False)

        def _apply_changes(save_feedback: bool = False) -> bool:
            idx_local = dialog_state["index"]
            current_ref = _current_result()
            if any(i != idx_local and self.scan_results[i] is current_ref for i in range(len(self.scan_results))):
                self.scan_results[idx_local] = self._lightweight_result_copy(current_ref)
            result = _current_result()
            try:
                snapshot = _collect_editor_snapshot(validate=True)
            except Exception as exc:
                QMessageBox.warning(self, "Dữ liệu không hợp lệ", str(exc))
                return False

            old_sid_for_score = str(result.student_id or "").strip()
            changes: list[str] = []
            new_sid = str(snapshot["student_id"] or "").strip()
            new_code = str(snapshot["exam_code"] or "").strip()
            new_mcq_answers = {int(q): str(v) for q, v in (snapshot.get("mcq_answers", {}) or {}).items()}
            new_tf_answers = {int(q): dict(v or {}) for q, v in (snapshot.get("true_false_answers", {}) or {}).items()}
            new_numeric_answers = {int(q): str(v) for q, v in (snapshot.get("numeric_answers", {}) or {}).items()}

            if new_sid != (result.student_id or ""):
                changes.append(f"student_id: '{result.student_id or ''}' -> '{new_sid}'")
                result.student_id = new_sid
                dialog_requires_full_status_refresh["value"] = True
            if new_code != (result.exam_code or ""):
                changes.append(f"exam_code: '{result.exam_code or ''}' -> '{new_code}'")
                result.exam_code = new_code
                dialog_requires_full_status_refresh["value"] = True
            if new_mcq_answers != (result.mcq_answers or {}):
                result.mcq_answers = new_mcq_answers
                changes.append("mcq_answers updated")
            if new_tf_answers != (result.true_false_answers or {}):
                result.true_false_answers = new_tf_answers
                changes.append("true_false_answers updated")
            if new_numeric_answers != (result.numeric_answers or {}):
                result.numeric_answers = new_numeric_answers
                changes.append("numeric_answers updated")

            if not changes:
                return True

            # Clear legacy manual-content text so Batch Scan "Nội dung" always reflects
            # the current structured answers edited in this dialog.
            setattr(result, "manual_content_override", "")
            self._mark_result_manually_edited(result, idx_local)
            self._refresh_student_profile_for_result(result)
            scoped = self._scoped_result_copy(result)
            self.scan_blank_summary[idx_local] = self._compute_blank_questions(scoped)
            self._record_adjustment(idx_local, changes, "dialog_edit")
            self._update_scan_row_from_result(idx_local, result)
            self._persist_single_scan_result_to_db(result, note="dialog_edit")
            self._sync_subject_batch_snapshot_after_inline_edit(persist_full_subject_rows=False)
            subject_key_now = self._current_batch_subject_key()
            if subject_key_now:
                self.scan_results_by_subject[self._batch_result_subject_key(subject_key_now)] = list(self.scan_results)
            dialog_saved_images.add(_current_result_image_key())
            self.btn_save_batch_subject.setEnabled(False)
            invalidated = self._invalidate_scoring_for_student_ids([old_sid_for_score, str(result.student_id or "").strip()], reason="dialog_edit")
            self._refresh_all_statuses()
            loaded_snapshots[_current_result_image_key()] = _snapshot_from_result(result)
            if save_feedback:
                notices = ["Đã lưu thay đổi cho bài hiện tại."]
                if invalidated > 0:
                    notices.append(f"Đã đánh dấu {invalidated} bản ghi cần chấm lại. Vui lòng chạy lại Tính điểm.")
                QMessageBox.information(self, "Sửa bài thi", "\n".join(notices))
            return True

        def _load_result_into_dialog(new_index: int, preserve_snapshot: dict[str, object] | None = None) -> None:
            if new_index < 0 or new_index >= len(self.scan_results):
                return
            dialog_state["loading"] = True
            dialog_state["index"] = new_index
            self.scan_list.setCurrentCell(new_index, self.SCAN_COL_STUDENT_ID)
            result = self.scan_results[new_index]
            image_key = self._result_identity_key(getattr(result, "image_path", ""))
            dlg.setWindowTitle(f"Sửa bài thi: {Path(result.image_path).name}")
            self._load_student_correction_options(str(result.student_id or "").strip())
            inp_sid.blockSignals(True)
            sid_value = str((preserve_snapshot or {}).get("student_id", result.student_id or "")).strip()
            inp_sid.setText(sid_value)
            sid_labels = list(getattr(self, "_student_option_labels_cache", []) or [])
            sid_completer = QCompleter(sid_labels, inp_sid)
            sid_completer.setCaseSensitivity(Qt.CaseInsensitive)
            sid_completer.setFilterMode(Qt.MatchContains)
            sid_completer.activated.connect(lambda text: inp_sid.setText(_normalize_student_id_input(str(text))))
            inp_sid.setCompleter(sid_completer)
            inp_sid.blockSignals(False)

            subject_key = self._current_batch_subject_key()
            current_code = str((preserve_snapshot or {}).get("exam_code", result.exam_code or ""))
            _populate_exam_code_combo(inp_code, subject_key, current_code)

            data_snapshot = preserve_snapshot or loaded_snapshots.get(image_key) or _snapshot_from_result(result)
            loaded_snapshots[image_key] = {
                "image_path": image_key,
                "student_id": str(data_snapshot.get("student_id", "") or "").strip(),
                "exam_code": str(data_snapshot.get("exam_code", "") or "").strip(),
                "mcq_answers": {int(q): str(v) for q, v in (data_snapshot.get("mcq_answers", {}) or {}).items()},
                "true_false_answers": {int(q): dict(v or {}) for q, v in (data_snapshot.get("true_false_answers", {}) or {}).items()},
                "numeric_answers": {int(q): str(v) for q, v in (data_snapshot.get("numeric_answers", {}) or {}).items()},
            }
            _refresh_editor_widgets(loaded_snapshots[image_key])
            _render_preview_for_result(result)
            btn_prev.setEnabled(new_index > 0)
            btn_prev_top.setEnabled(new_index > 0)
            btn_next.setEnabled(new_index < len(self.scan_results) - 1)
            btn_next_top.setEnabled(new_index < len(self.scan_results) - 1)
            dialog_state["loading"] = False

        def _rebuild_for_exam_code_change() -> None:
            if dialog_state["loading"]:
                return
            try:
                snapshot = _collect_editor_snapshot(validate=False)
            except Exception:
                snapshot = _snapshot_from_result(_current_result())
            image_key = _current_result_image_key()
            loaded_snapshots[image_key] = {
                "image_path": image_key,
                "student_id": str(snapshot.get("student_id", "") or "").strip(),
                "exam_code": str(snapshot.get("exam_code", "") or "").strip(),
                "mcq_answers": {int(q): str(v) for q, v in (snapshot.get("mcq_answers", {}) or {}).items()},
                "true_false_answers": {int(q): dict(v or {}) for q, v in (snapshot.get("true_false_answers", {}) or {}).items()},
                "numeric_answers": {int(q): str(v) for q, v in (snapshot.get("numeric_answers", {}) or {}).items()},
            }
            _refresh_editor_widgets(loaded_snapshots[image_key])

        def _navigate(offset: int) -> None:
            target = dialog_state["index"] + offset
            if target < 0 or target >= len(self.scan_results):
                return
            if not _apply_changes(save_feedback=False):
                return
            _load_result_into_dialog(target)

        def _has_unsaved_changes() -> bool:
            try:
                snapshot = _collect_editor_snapshot(validate=False)
            except Exception:
                return False
            baseline = loaded_snapshots.get(_current_result_image_key(), _snapshot_from_result(_current_result()))
            return snapshot != baseline

        def _request_close() -> None:
            if _has_unsaved_changes():
                msg = QMessageBox(self)
                msg.setWindowTitle("Sửa bài thi")
                msg.setText("Bài hiện tại có thay đổi chưa lưu.")
                msg.setInformativeText("Bạn muốn lưu trước khi đóng cửa sổ sửa bài?")
                msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
                msg.setDefaultButton(QMessageBox.Save)
                choice = msg.exec()
                if choice == QMessageBox.Cancel:
                    return
                if choice == QMessageBox.Save and not _apply_changes(save_feedback=False):
                    return
            if dialog_saved_images:
                for saved_image in sorted(dialog_saved_images):
                    saved_idx = self._row_index_by_image_path(saved_image)
                    if 0 <= saved_idx < len(self.scan_results):
                        saved_result = self.scan_results[saved_idx]
                        self._refresh_student_profile_for_result(saved_result, saved_idx)
                        scoped_saved = self._scoped_result_copy(saved_result)
                        self.scan_blank_summary[saved_idx] = self._compute_blank_questions(scoped_saved)
                        self._update_scan_row_from_result(saved_idx, saved_result)
                subject_key_now = self._current_batch_subject_key()
                if subject_key_now:
                    self.scan_results_by_subject[self._batch_result_subject_key(subject_key_now)] = list(self.scan_results)
                if dialog_requires_full_status_refresh["value"]:
                    self._refresh_all_statuses()
                else:
                    for saved_image in sorted(dialog_saved_images):
                        saved_idx = self._row_index_by_image_path(saved_image)
                        if 0 <= saved_idx < self.scan_list.rowCount():
                            self._refresh_row_status(saved_idx)
                self._update_batch_scan_bottom_status_text()
                current_idx = dialog_state["index"]
                if 0 <= current_idx < len(self.scan_results):
                    self.scan_list.setCurrentCell(current_idx, self.SCAN_COL_STUDENT_ID)
                    self._update_scan_preview(current_idx)
                    self._sync_correction_detail_panel(self.scan_results[current_idx], rebuild_editor=False)
            dlg.accept()

        inp_code.currentIndexChanged.connect(lambda _=0: _rebuild_for_exam_code_change())
        inp_code.editTextChanged.connect(lambda _text="": _rebuild_for_exam_code_change())
        btn_prev.clicked.connect(lambda: _navigate(-1))
        btn_prev_top.clicked.connect(lambda: _navigate(-1))
        btn_next.clicked.connect(lambda: _navigate(1))
        btn_next_top.clicked.connect(lambda: _navigate(1))
        btn_save.clicked.connect(lambda: _apply_changes(save_feedback=True))
        btn_close.clicked.connect(_request_close)
        dlg.reject = _request_close

        _load_result_into_dialog(dialog_state["index"])
        dlg.exec()

    def apply_manual_correction(self) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= len(self.scan_results):
            QMessageBox.warning(self, "No selection", "Select a scanned result first.")
            return
        txt = self.manual_edit.toPlainText().strip()
        if not txt:
            return
        try:
            patch = json.loads(txt)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid JSON", f"Cannot parse manual correction:\n{exc}")
            return

        res = self.scan_results[idx]
        if any(i != idx and self.scan_results[i] is res for i in range(len(self.scan_results))):
            res = self._lightweight_result_copy(res)
            self.scan_results[idx] = res
        old_sid_for_score = str(res.student_id or "").strip()
        changes: list[str] = []
        if "student_id" in patch:
            new_sid = str(patch["student_id"])
            if new_sid != (res.student_id or ""):
                changes.append(f"student_id: '{res.student_id or ''}' -> '{new_sid}'")
            res.student_id = new_sid
        if "exam_code" in patch:
            new_code = str(patch["exam_code"])
            if new_code != (res.exam_code or ""):
                changes.append(f"exam_code: '{res.exam_code or ''}' -> '{new_code}'")
            res.exam_code = new_code
        if isinstance(patch.get("mcq_answers"), dict):
            new_mcq_answers = {int(k): str(v) for k, v in patch["mcq_answers"].items()}
            if new_mcq_answers != (res.mcq_answers or {}):
                res.mcq_answers = new_mcq_answers
                changes.append("mcq_answers updated")
        if isinstance(patch.get("numeric_answers"), dict):
            new_numeric_answers = {int(k): str(v) for k, v in patch["numeric_answers"].items()}
            if new_numeric_answers != (res.numeric_answers or {}):
                res.numeric_answers = new_numeric_answers
                changes.append("numeric_answers updated")
        if isinstance(patch.get("true_false_answers"), dict):
            new_tf_answers = patch["true_false_answers"]
            if new_tf_answers != (res.true_false_answers or {}):
                res.true_false_answers = new_tf_answers
                changes.append("true_false_answers updated")

        sid = (res.student_id or "").strip() or "-"
        sid_item = QTableWidgetItem(sid)
        sid_item.setData(Qt.UserRole, str(res.image_path))
        sid_item.setData(Qt.UserRole + 1, res.exam_code or "")
        sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(res))
        self.scan_list.setItem(idx, self.SCAN_COL_STUDENT_ID, sid_item)
        if changes:
            self._mark_result_manually_edited(res, idx)
            self._refresh_student_profile_for_result(res)
            scoped = self._scoped_result_copy(res)
            self.scan_blank_summary[idx] = self._compute_blank_questions(scoped)
            self._update_scan_row_from_result(idx, res)
            self._record_adjustment(idx, changes, "manual_json")
            self._persist_single_scan_result_to_db(res, note="manual_json")
            self._sync_subject_batch_snapshot_after_inline_edit(persist_full_subject_rows=False)
            self.btn_save_batch_subject.setEnabled(False)
            invalidated = self._invalidate_scoring_for_student_ids(
                [old_sid_for_score, str(res.student_id or "").strip()],
                reason="manual_json",
            )
            if invalidated > 0:
                QMessageBox.information(
                    self,
                    "Tính điểm",
                    f"Đã đánh dấu {invalidated} bản ghi cần chấm lại do sửa bài. Vui lòng chạy lại Tính điểm.",
                )
        self._refresh_all_statuses()
        self._update_scan_preview(idx)
        self._load_selected_result_for_correction()
        QMessageBox.information(self, "Correction", "Manual correction applied to selected scan.")
