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


class MainWindowScoringMixin:
    """Scoring panel, cached score rows, score calculation, essay/direct-score flows."""
    def action_calculate_scores(self) -> None:
        # From Batch Scan -> Scoring: do not prompt save when no real batch edits.
        if self.stack.currentIndex() != 1 or bool(hasattr(self, "btn_save_batch_subject") and self.btn_save_batch_subject.isEnabled()):
            if not self._confirm_before_switching_work("màn hình Tính điểm"):
                return
        if self.stack.currentIndex() == 1 and hasattr(self, "batch_subject_combo") and self.batch_subject_combo.currentIndex() > 0:
            has_unsaved = bool(hasattr(self, "btn_save_batch_subject") and self.btn_save_batch_subject.isEnabled())
            if not has_unsaved:
                self._on_batch_subject_changed(self.batch_subject_combo.currentIndex(), force_reload=True)
        self._open_scoring_view()
        self._refresh_ribbon_action_states()

    def _has_scoring_unsaved_changes(self) -> bool:
        if hasattr(self, "btn_scoring_save") and self.btn_scoring_save.isEnabled():
            return True

        subject_key = ""
        try:
            subject_key = str(self._resolve_preferred_scoring_subject() or getattr(self, "_current_scoring_subject", "") or "").strip()
        except Exception:
            subject_key = str(getattr(self, "_current_scoring_subject", "") or "").strip()

        dirty_subjects = {
            str(s or "").strip()
            for s in (getattr(self, "_scoring_dirty_subjects", set()) or set())
            if str(s or "").strip()
        }
        if subject_key and subject_key in dirty_subjects:
            return True

        if subject_key:
            state = self._subject_scoring_state(subject_key) if hasattr(self, "_subject_scoring_state") else {}
            saved_flag = bool((state or {}).get("scoring_saved", False)) and subject_key not in dirty_subjects
            result_count = int((state or {}).get("scoring_result_count", 0) or 0)
            if result_count > 0 and not saved_flag:
                return True

        return bool(dirty_subjects)

    def _subject_scoring_state(self, subject_key: str) -> dict:
        cfg = self._subject_config_by_subject_key(subject_key) or {}
        return {
            "scoring_saved": bool(cfg.get("scoring_saved", False)),
            "scoring_saved_at": str(cfg.get("scoring_saved_at", "") or ""),
            "scoring_result_count": int(cfg.get("scoring_result_count", 0) or 0),
            "scoring_phase_last_note": str(cfg.get("scoring_phase_last_note", "") or ""),
            "scoring_last_mode": str(cfg.get("scoring_last_mode", "") or ""),
        }

    def _refresh_scoring_state_label(self, subject_key: str) -> None:
        if not hasattr(self, "scoring_state_label"):
            return
        state = self._subject_scoring_state(subject_key)
        saved_flag = bool(state.get("scoring_saved", False)) and subject_key not in self._scoring_dirty_subjects
        status_text = "Đã lưu" if saved_flag else "Chưa lưu"
        saved_at = str(state.get("scoring_saved_at", "") or "-")
        count = int(state.get("scoring_result_count", 0) or 0)
        mode = str(state.get("scoring_last_mode", "") or "-")
        note = str(state.get("scoring_phase_last_note", "") or "-")
        self.scoring_state_label.setText(
            f"Trạng thái: {status_text} | Lần lưu cuối: {saved_at} | Số bài đã chấm: {count} | Cơ chế: {mode} | Ghi chú: {note}"
        )
        if hasattr(self, "btn_scoring_save"):
            self.btn_scoring_save.setEnabled((subject_key in self._scoring_dirty_subjects) or (not saved_flag and count > 0))

    def _set_subject_scoring_metadata(self, subject_key: str, updates: dict) -> None:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not isinstance(cfg, dict):
            return
        cfg.update(dict(updates or {}))
        if self.session and isinstance(self.session.config, dict):
            self.session_dirty = True

    def _mark_subject_scoring_saved(self, subject_key: str, count: int, mode: str, note: str) -> None:
        # persisted scoring state
        now_text = datetime.now().isoformat(timespec="seconds")
        self._set_subject_scoring_metadata(subject_key, {
            "scoring_saved": True,
            "scoring_saved_at": now_text,
            "scoring_result_count": int(count or 0),
            "scoring_phase_last_note": str(note or ""),
            "scoring_last_mode": str(mode or ""),
        })
        self._scoring_dirty_subjects.discard(subject_key)
        self._refresh_scoring_state_label(subject_key)

    def _clear_subject_scoring_saved_state(self, subject_key: str, count: int = 0, mode: str = "", note: str = "") -> None:
        self._set_subject_scoring_metadata(subject_key, {
            "scoring_saved": False,
            "scoring_saved_at": "",
            "scoring_result_count": int(count or 0),
            "scoring_phase_last_note": str(note or ""),
            "scoring_last_mode": str(mode or ""),
        })
        self._scoring_dirty_subjects.add(subject_key)
        self._refresh_scoring_state_label(subject_key)

    def _persist_scoring_results_for_subject(self, subject_key: str, rows: list[dict], mode: str, note: str, *, mark_saved: bool) -> None:
        subject = str(subject_key or "").strip()
        if not subject:
            return
        packed: dict[str, dict] = {}
        for row in (rows or []):
            sid_key = str((row or {}).get("student_id", "") or "").strip()
            if sid_key:
                packed[sid_key] = dict(row or {})
        self.scoring_results_by_subject[subject] = packed
        try:
            self.database.conn.execute("DELETE FROM scores WHERE subject_key = ?", (subject,))
            for payload in packed.values():
                self.database.upsert_score_row(
                    subject,
                    str(payload.get("student_id", "") or ""),
                    str(payload.get("exam_code", "") or ""),
                    dict(payload),
                )
        except Exception:
            pass
        if mark_saved:
            self._mark_subject_scoring_saved(subject, len(packed), mode, note)

    def _persist_scoring_results(self, subject_key: str, phase: dict | None = None) -> None:
        subject = str(subject_key or "").strip()
        if not subject:
            return
        payloads = list((self.scoring_results_by_subject.get(subject, {}) or {}).values())
        phase = dict(phase or {})
        mode = str(phase.get("mode", "Tính lại toàn bộ") or "Tính lại toàn bộ")
        note = str(phase.get("note", "") or "")
        self._persist_scoring_results_for_subject(subject, payloads, mode, note, mark_saved=True)

    def _load_scoring_results_for_subject_from_storage(self, subject_key: str, *, force_db: bool = False) -> dict[str, dict]:
        # DB là nguồn chuẩn cho scoring; chỉ dùng runtime cache đã có nếu cùng subject trong phiên hiện tại.
        subject = str(subject_key or "").strip()
        if not subject:
            return {}
        cached = self.scoring_results_by_subject.get(subject)
        if (not force_db) and isinstance(cached, dict) and cached:
            return dict(cached)
        db_rows = self.database.fetch_scores_for_subject(subject) or []
        loaded: dict[str, dict] = {}
        for payload in db_rows:
            if not isinstance(payload, dict):
                continue
            sid_key = str(payload.get("student_id", "") or "").strip()
            if sid_key:
                loaded[sid_key] = dict(payload)
        if not loaded and self._subject_uses_direct_score_import(subject):
            loaded = self._materialize_direct_score_rows_for_subject(subject, persist=True)
        if loaded:
            self.scoring_results_by_subject[subject] = dict(loaded)
        return loaded

    def _load_cached_scoring_results_for_subject(self, subject_key: str) -> None:
        subject = str(subject_key or "").strip()
        if not subject or not hasattr(self, "score_preview_table"):
            return
        stored_rows = self._load_scoring_results_for_subject_from_storage(subject)
        loaded_rows = []
        for _, payload in (stored_rows or {}).items():
            if isinstance(payload, dict):
                loaded_rows.append(dict(payload))
        source_student_ids, source_row_count = self._scoring_source_student_ids(subject)
        if source_row_count > 0:
            filtered_rows: list[dict] = []
            for row in loaded_rows:
                sid = str((row or {}).get("student_id", "") or "").strip()
                status_text = str((row or {}).get("status", "") or (row or {}).get("note", "") or "").strip()
                if sid and sid in source_student_ids:
                    filtered_rows.append(row)
                    continue
                if not sid and status_text.startswith("Lỗi"):
                    filtered_rows.append(row)
            loaded_rows = filtered_rows
        loaded_rows.sort(key=lambda row: (
            0 if str((row or {}).get("status", "") or "").startswith("Lỗi") else 1,
            str((row or {}).get("student_id", "") or ""),
        ))

        success_count = 0
        error_count = 0
        edited_count = 0
        pending_count = 0
        self.score_preview_table.setSortingEnabled(False)
        self.score_preview_table.setUpdatesEnabled(False)
        try:
            self.score_preview_table.setRowCount(0)
            for row in loaded_rows:
                row = self._hydrate_scoring_row_with_student_profile(subject, row)
                r = self.score_preview_table.rowCount()
                self.score_preview_table.insertRow(r)
                status_text = str((row or {}).get("status", "") or (row or {}).get("note", "") or "OK")
                score_raw = (row or {}).get("score", "")
                score_display = str(score_raw) if score_raw not in {"", None} else ""
                if status_text == "OK" and score_display == "":
                    status_text = "Chưa có điểm"
                if status_text.startswith("Lỗi"):
                    error_count += 1
                else:
                    success_count += 1
                if status_text == "Đã sửa":
                    edited_count += 1
                if status_text in {"Chưa chấm", "Cần chấm lại", "Chưa có điểm"}:
                    pending_count += 1
                values = [
                    str((row or {}).get("student_id", "") or "-"),
                    str((row or {}).get("name", "") or "-"),
                    str((row or {}).get("class_name", "") or "-"),
                    str((row or {}).get("birth_date", "") or "-"),
                    str((row or {}).get("exam_code", "") or "-"),
                    str((row or {}).get("mcq_correct", 0)),
                    str(self._tf_statement_correct_count(str((row or {}).get("tf_compare", "") or ""))),
                    str((row or {}).get("numeric_correct", 0)),
                    str((row or {}).get("correct", 0)),
                    str((row or {}).get("wrong", 0)),
                    str((row or {}).get("blank", 0)),
                    str((row or {}).get("recheck_score", "") if (row or {}).get("recheck_score", "") not in {"", None} else score_display),
                    status_text,
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if col == 12 and status_text != "OK":
                        item.setForeground(QColor("red"))
                        category = "success"
                        if status_text.startswith("Lỗi"):
                            category = "error"
                        elif status_text == "Đã sửa":
                            category = "edited"
                        elif status_text in {"Chưa chấm", "Cần chấm lại", "Chưa có điểm"}:
                            category = "pending"
                        item.setData(Qt.UserRole, category)
                    self.score_preview_table.setItem(r, col, item)
                if status_text.startswith("Lỗi"):
                    for col in range(self.score_preview_table.columnCount()):
                        item = self.score_preview_table.item(r, col)
                        if item:
                            item.setBackground(QColor(255, 225, 225))
        finally:
            self.score_preview_table.setUpdatesEnabled(True)
            self.score_preview_table.viewport().update()
        self._current_scoring_subject = subject
        self._refresh_scoring_state_label(subject)
        self._update_scoring_status_bar(success_count, error_count, edited_count, pending_count)
        self._apply_scoring_filter()

    def _hydrate_scoring_row_with_student_profile(self, subject_key: str, row: dict) -> dict:
        payload = dict(row or {})
        sid = str(payload.get("student_id", "") or "").strip()
        if not sid:
            return payload
        cfg = self._subject_config_by_subject_key(subject_key) or {}
        profile = self._student_profile_by_id(sid)
        room_text = (
            self._subject_room_for_student_id(sid, cfg)
            or str(profile.get("exam_room", "") or "").strip()
            or str(payload.get("exam_room", "") or "").strip()
        )
        payload["name"] = str(profile.get("name", "") or payload.get("name", "") or "").strip()
        payload["class_name"] = str(profile.get("class_name", "") or payload.get("class_name", "") or "").strip()
        payload["birth_date"] = str(profile.get("birth_date", "") or payload.get("birth_date", "") or "").strip()
        payload["exam_room"] = room_text
        return payload

    @staticmethod
    def _scoring_filter_column_from_combo_index(combo_index: int) -> int | None:
        mapping = {
            0: None,  # Tất cả
            1: 0,     # Student ID
            2: 1,     # Name
            3: 2,     # Lớp
            4: 3,     # Ngày sinh
            5: 4,     # Exam Code
            6: 12,    # Trạng thái
        }
        return mapping.get(int(combo_index), None)

    def _apply_scoring_filter(self) -> None:
        if not hasattr(self, "score_preview_table"):
            return

        def _normalize(text: str) -> str:
            return " ".join(str(text or "").strip().lower().split())

        value = _normalize(self.scoring_search_value.text() if hasattr(self, "scoring_search_value") else "")
        col = self._scoring_filter_column_from_combo_index(self.scoring_filter_column.currentIndex()) if hasattr(self, "scoring_filter_column") else None
        for i in range(self.score_preview_table.rowCount()):
            status_item = self.score_preview_table.item(i, 12)
            status_category = str(status_item.data(Qt.UserRole) if status_item else "success")
            status_ok = self.scoring_status_filter_mode in {"all", "", status_category}
            if not value:
                self.score_preview_table.setRowHidden(i, not status_ok)
                continue
            if col is None:
                row_text = " | ".join(_normalize(self.score_preview_table.item(i, j).text() if self.score_preview_table.item(i, j) else "") for j in range(self.score_preview_table.columnCount()))
            else:
                item = self.score_preview_table.item(i, col)
                row_text = _normalize(item.text() if item else "")
            self.score_preview_table.setRowHidden(i, (value not in row_text) or (not status_ok))

    def _handle_scoring_status_filter_link(self, link: str) -> None:
        self.scoring_status_filter_mode = str(link or "all").strip() or "all"
        self._apply_scoring_filter()

    def _update_scoring_status_bar(self, success_count: int, error_count: int, edited_count: int, pending_count: int = 0) -> None:
        if not hasattr(self, "scoring_status_bar"):
            return
        self.scoring_status_bar.setText(
            f"Thống kê chấm điểm: "
            f"<a href='all'><b>Tất cả</b></a> | "
            f"<a href='success' style='color:#0b7a0b;font-weight:600'>Thành công {int(success_count)}</a> | "
            f"<a href='error' style='color:#c62828;font-weight:600'>Lỗi {int(error_count)}</a> | "
            f"<a href='edited' style='color:#1565c0;font-weight:600'>Đã sửa {int(edited_count)}</a> | "
            f"<a href='pending' style='color:#6d4c41;font-weight:600'>Chưa chấm/Cần chấm lại {int(pending_count)}</a>"
        )

    @staticmethod
    def _tf_statement_correct_count(tf_compare_text: str) -> int:
        total = 0
        for token in [x.strip() for x in str(tf_compare_text or "").split(";") if x.strip()]:
            _, _, pair = token.partition(":")
            key_txt, _, marked_txt = pair.partition("|")
            key_norm = "".join(ch for ch in str(key_txt or "").upper() if ch in {"T", "F", "Đ", "D", "S"})
            mark_norm = "".join(ch for ch in str(marked_txt or "").upper() if ch in {"T", "F", "Đ", "D", "S"})
            limit = min(len(key_norm), len(mark_norm))
            for i in range(limit):
                if key_norm[i] == mark_norm[i]:
                    total += 1
        return total

    def _find_scoring_scan_result(self, subject_key: str, student_id: str, exam_code: str = "") -> OMRResult | None:
        subject = str(subject_key or "").strip()
        sid = str(student_id or "").strip()
        code = str(exam_code or "").strip()
        if not subject or not sid:
            return None
        sid_norm = self._normalized_student_id_for_match(sid)
        rows = []
        cfg = self._subject_config_by_subject_key(subject)
        if cfg and hasattr(self, "_subject_scan_storage_key_candidates"):
            for candidate_key in self._subject_scan_storage_key_candidates(cfg):
                rows = self.database.fetch_scan_results_for_subject(candidate_key) or []
                if rows:
                    break
        if not rows:
            rows = self.database.fetch_scan_results_for_subject(self._batch_result_subject_key(subject)) or []
        best_match: OMRResult | None = None
        for payload in rows:
            res = self._deserialize_omr_result(payload)
            res_sid = str(getattr(res, "student_id", "") or "").strip()
            if res_sid != sid and self._normalized_student_id_for_match(res_sid) != sid_norm:
                continue
            if code and str(getattr(res, "exam_code", "") or "").strip() == code:
                return res
            if best_match is None:
                best_match = res
        return best_match

    def _open_scoring_review_editor_from_table(self, row: int, _col: int) -> None:
        if row < 0 or row >= self.score_preview_table.rowCount():
            return
        sid = str(self.score_preview_table.item(row, 0).text() if self.score_preview_table.item(row, 0) else "").strip()
        exam_code = str(self.score_preview_table.item(row, 4).text() if self.score_preview_table.item(row, 4) else "").strip()
        subject = str(self.scoring_subject_combo.currentData() or self.scoring_subject_combo.currentText() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        if not sid or not subject:
            return
        if self._subject_uses_direct_score_import(subject):
            self._open_essay_score_editor_from_table(row, subject)
            return
        result = self._find_scoring_scan_result(subject, sid, exam_code)
        if result is None:
            QMessageBox.warning(self, "Tính điểm", "Không tìm thấy bài scan gốc để mở màn hình sửa.")
            return
        self._open_scoring_review_editor(subject, result)

    def _open_scoring_review_editor(self, subject_key: str, result: OMRResult) -> None:
        from gui.main_window_scoring import open_scoring_review_editor_dialog

        return open_scoring_review_editor_dialog(self, subject_key, result)

    def _handle_scoring_subject_changed(self, _index: int) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        if subject_key:
            self._ensure_scoring_preview_current(subject_key, reason="auto_refresh_subject_change", force=False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "batch_scan_status_bottom"):
            self.batch_scan_status_bottom.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)
        self._current_route_name = "workspace_scoring"
        for action_name in ["ribbon_batch_execute_action", "ribbon_batch_save_action", "ribbon_batch_close_action"]:
            action = getattr(self, action_name, None)
            if action is not None:
                action.setVisible(False)
        self._refresh_ribbon_action_states()
        selected_subject = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        if selected_subject:
            self._ensure_scoring_preview_current(selected_subject, reason="auto_refresh_open_scoring", force=False)

    def _back_to_batch_scan(self) -> None:
        # workspace sub-route
        if not self._navigate_to(
            "workspace_batch_scan",
            context={"session_id": self.current_session_id, "origin": "workspace_scoring"},
            push_current=False,
            require_confirm=True,
            reason="back_to_batch",
        ):
            return

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = self._display_subject_label(cfg)
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self._ensure_current_session_loaded():
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return

        # Scoring là DB-first: nếu Batch Scan hiện tại còn thay đổi chưa lưu thì hỏi người dùng
        # lưu hay bỏ qua. Không dựng snapshot runtime để làm nguồn dữ liệu chấm.
        if self._has_batch_unsaved_changes():
            choice = self._prompt_save_changes_word_style(
                "Batch Scan chưa lưu",
                "Bạn có thay đổi chưa lưu cho môn hiện tại. Muốn lưu trước khi sang phần chấm điểm không?",
            )
            if choice == "cancel":
                return
            if choice == "save" and not self._save_batch_for_selected_subject():
                return

        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan hoặc đã import điểm trực tiếp trước khi tính điểm.")
            return

        self._navigate_to(
            "workspace_scoring",
            context={"session_id": self.current_session_id},
            push_current=True,
            require_confirm=False,
            reason="open_scoring",
        )
        selected_subject = self._resolve_preferred_scoring_subject()
        self._populate_scoring_subjects(selected_subject)
        self._refresh_scoring_phase_table()
        self._refresh_dashboard_summary_from_db(selected_subject)
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "score_phase_table"):
            return
        subjects = self._subject_configs_for_scoring()
        self.score_phase_table.setUpdatesEnabled(False)
        try:
            self.score_phase_table.setRowCount(0)
            for cfg in subjects:
                subject = self._subject_key_from_cfg(cfg)
                if not subject:
                    continue
                saved_at = str((cfg or {}).get("scoring_saved_at", "") or "").strip()
                mode = str((cfg or {}).get("scoring_last_mode", "") or "").strip()
                note = str((cfg or {}).get("scoring_phase_last_note", "") or "").strip()
                count = int((cfg or {}).get("scoring_result_count", 0) or 0)

                if (not saved_at and count <= 0) or (not mode and not note):
                    stored_rows = self._load_scoring_results_for_subject_from_storage(subject)
                    latest = None
                    latest_ts = ""
                    for payload in (stored_rows or {}).values():
                        if not isinstance(payload, dict):
                            continue
                        ts = str(payload.get("phase_timestamp", "") or "")
                        if ts >= latest_ts:
                            latest_ts = ts
                            latest = payload
                    if latest is not None:
                        saved_at = saved_at or latest_ts
                        mode = mode or str((latest or {}).get("phase_mode", "") or "")
                        note = note or str((latest or {}).get("note", "") or (latest or {}).get("status", "") or "")
                        count = count or len(stored_rows or {})

                r = self.score_phase_table.rowCount()
                self.score_phase_table.insertRow(r)
                self.score_phase_table.setItem(r, 0, QTableWidgetItem(subject or "-"))
                self.score_phase_table.setItem(r, 1, QTableWidgetItem(saved_at or "-"))
                self.score_phase_table.setItem(r, 2, QTableWidgetItem(mode or "-"))
                self.score_phase_table.setItem(r, 3, QTableWidgetItem(str(count)))
                self.score_phase_table.setItem(r, 4, QTableWidgetItem(note or "-"))
        finally:
            self.score_phase_table.setUpdatesEnabled(True)
            self.score_phase_table.viewport().update()

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)
        self._refresh_scoring_state_label(subject_key or self._resolve_preferred_scoring_subject())

    def _mark_scoring_stale_after_batch_save(self, subject_key: str, note: str = "Batch Scan đã thay đổi") -> None:
        subject = str(subject_key or "").strip()
        if not subject:
            return
        cfg = self._subject_config_by_subject_key(subject) or {}
        existing_count = int((cfg or {}).get("scoring_result_count", 0) or 0)
        if existing_count <= 0:
            existing_count = len(self.scoring_results_by_subject.get(subject, {}) or {})
        self._clear_subject_scoring_saved_state(
            subject,
            count=existing_count,
            mode=str((cfg or {}).get("scoring_last_mode", "") or ""),
            note=note,
        )

    def _ensure_scoring_preview_current(self, subject_key: str, reason: str = "", force: bool = False) -> None:
        subject = str(subject_key or "").strip()
        if not subject:
            return
        self._current_scoring_subject = subject
        self._apply_subject_section_visibility(subject)
        cached_rows = self._load_scoring_results_for_subject_from_storage(subject)
        is_dirty = subject in self._scoring_dirty_subjects
        if force or is_dirty or not cached_rows:
            self.calculate_scores(subject_key=subject, mode="Tính lại toàn bộ", note=reason)
            return
        self._load_cached_scoring_results_for_subject(subject)

    def _invalidate_scoring_for_student_ids(self, student_ids: list[str], subject_key: str = "", reason: str = "") -> int:
        subject = str(subject_key or self._current_batch_subject_key() or "").strip()
        if not subject:
            return 0
        subject_scores = dict(self.scoring_results_by_subject.get(subject, {}))
        if not subject_scores:
            return 0
        changed = 0
        normalized_ids = [str(sid or "").strip() for sid in (student_ids or []) if str(sid or "").strip()]
        if not normalized_ids:
            changed = len(subject_scores)
            subject_scores = {}
        else:
            for sid_key in normalized_ids:
                if sid_key in subject_scores:
                    subject_scores.pop(sid_key, None)
                    changed += 1
        if changed <= 0:
            return 0
        self.scoring_results_by_subject[subject] = subject_scores
        # Đồng bộ DB ngay khi invalidate để tránh resurrect dữ liệu chấm cũ
        # khi chuyển màn hình hoặc tải lại phiên.
        try:
            current_mode = str((self._subject_config_by_subject_key(subject) or {}).get("scoring_last_mode", "") or "Tính lại toàn bộ")
            self._persist_scoring_results_for_subject(
                subject,
                list(subject_scores.values()),
                current_mode,
                reason or "invalidate_scoring_records",
                mark_saved=False,
            )
        except Exception:
            pass
        try:
            self.database.log_change(
                "scoring_results",
                subject,
                "invalidate_scoring_records",
                "",
                f"removed={changed}; student_ids={','.join(sorted(set(normalized_ids)))}",
                reason or "invalidate_scoring_records",
            )
        except Exception:
            pass
        self._persist_runtime_session_state_quietly()
        return changed

    def _finalize_scoring_rows(
        self,
        subject: str,
        rows: list,
        prev_subject_scores: dict[str, dict],
        edited_student_ids: set[str],
        mode_text: str,
        note: str,
        missing: int,
        failed_scans: list[dict[str, str]],
        smart_scored_scans: list[dict[str, str]],
    ) -> list:
        self.score_rows = rows
        rows.sort(key=lambda r: (0 if str(getattr(r, "scoring_note", "") or "").startswith("Lỗi") else 1, str(getattr(r, "student_id", "") or "")))
        success_count = 0
        error_count = 0
        edited_count = 0
        pending_count = 0
        self.score_preview_table.setSortingEnabled(False)
        self.score_preview_table.setUpdatesEnabled(False)
        try:
            self.score_preview_table.setRowCount(0)
            for i, r in enumerate(rows):
                sid_key = str(r.student_id or "").strip()
                existing_payload = prev_subject_scores.get(sid_key, {}) if sid_key else {}
                recheck_score = existing_payload.get("recheck_score", "") if isinstance(existing_payload, dict) else ""
                final_score_display = recheck_score if recheck_score not in {"", None} else r.score
                class_name = str(getattr(r, "class_name", "") or existing_payload.get("class_name", "") or "-") if isinstance(existing_payload, dict) else str(getattr(r, "class_name", "") or "-")
                birth_date = str(getattr(r, "birth_date", "") or existing_payload.get("birth_date", "") or "-") if isinstance(existing_payload, dict) else str(getattr(r, "birth_date", "") or "-")
                status_text = str(getattr(r, "scoring_note", "") or "OK")
                if sid_key in edited_student_ids:
                    status_text = "Đã sửa"
                elif isinstance(existing_payload, dict) and existing_payload.get("status") == "Đã sửa":
                    status_text = "Đã sửa"

                self.score_preview_table.insertRow(i)
                self.score_preview_table.setItem(i, 0, QTableWidgetItem(r.student_id or "-"))
                self.score_preview_table.setItem(i, 1, QTableWidgetItem(r.name or "-"))
                self.score_preview_table.setItem(i, 2, QTableWidgetItem(class_name))
                self.score_preview_table.setItem(i, 3, QTableWidgetItem(birth_date))
                self.score_preview_table.setItem(i, 4, QTableWidgetItem(r.exam_code))
                self.score_preview_table.setItem(i, 5, QTableWidgetItem(str(getattr(r, "mcq_correct", 0))))
                tf_statement_count = self._tf_statement_correct_count(str(getattr(r, "tf_compare", "") or ""))
                self.score_preview_table.setItem(i, 6, QTableWidgetItem(str(tf_statement_count)))
                self.score_preview_table.setItem(i, 7, QTableWidgetItem(str(getattr(r, "numeric_correct", 0))))
                self.score_preview_table.setItem(i, 8, QTableWidgetItem(str(r.correct)))
                self.score_preview_table.setItem(i, 9, QTableWidgetItem(str(r.wrong)))
                self.score_preview_table.setItem(i, 10, QTableWidgetItem(str(r.blank)))
                self.score_preview_table.setItem(i, 11, QTableWidgetItem(str(final_score_display)))
                status_item = QTableWidgetItem(status_text)
                if status_text != "OK":
                    status_item.setForeground(QColor("red"))
                category = "success"
                if status_text.startswith("Lỗi"):
                    category = "error"
                elif status_text == "Đã sửa":
                    category = "edited"
                elif status_text in {"Chưa chấm", "Cần chấm lại"}:
                    category = "pending"
                status_item.setData(Qt.UserRole, category)
                self.score_preview_table.setItem(i, 12, status_item)
                if status_text.startswith("Lỗi"):
                    for col in range(self.score_preview_table.columnCount()):
                        item = self.score_preview_table.item(i, col)
                        if item:
                            item.setBackground(QColor(255, 225, 225))
                    error_count += 1
                else:
                    success_count += 1
                if status_text == "Đã sửa":
                    edited_count += 1
                if status_text in {"Chưa chấm", "Cần chấm lại"}:
                    pending_count += 1
        finally:
            self.score_preview_table.setUpdatesEnabled(True)
            self.score_preview_table.viewport().update()

        phase = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "subject": subject,
            "mode": mode_text,
            "count": len(rows),
            "missing": missing,
            "failed_count": len(failed_scans),
            "success_count": len(rows),
            "failed_scans": list(failed_scans),
            "smart_scoring_count": len(smart_scored_scans),
            "smart_scored_scans": list(smart_scored_scans),
            "note": note,
        }
        phase_marker = f"{phase['timestamp']}::{subject}::{mode_text}"

        subject_scores = dict(prev_subject_scores)
        for r in rows:
            sid_key = (r.student_id or "").strip()
            if sid_key:
                old_payload = (prev_subject_scores.get(sid_key, {}) or {})
                profile = self._student_profile_by_id(sid_key)
                row_name = str(profile.get("name", "") or getattr(r, "name", "") or old_payload.get("name", "") or "")
                row_class_name = str(profile.get("class_name", "") or getattr(r, "class_name", "") or old_payload.get("class_name", "") or "-")
                row_birth_date = str(profile.get("birth_date", "") or getattr(r, "birth_date", "") or old_payload.get("birth_date", "") or "-")
                row_exam_room = str(
                    self._subject_room_for_student_id(sid_key, self._subject_config_by_subject_key(subject) or {})
                    or getattr(r, "exam_room", "")
                    or old_payload.get("exam_room", "")
                    or profile.get("exam_room", "")
                    or ""
                )
                row_status = "Đã sửa" if (sid_key in edited_student_ids or old_payload.get("status") == "Đã sửa") else str(getattr(r, "scoring_note", "") or "OK")
                subject_scores[sid_key] = {
                    "student_id": r.student_id,
                    "name": row_name,
                    "subject": r.subject,
                    "class_name": row_class_name,
                    "birth_date": row_birth_date,
                    "exam_room": row_exam_room,
                    "exam_code": r.exam_code,
                    "mcq_correct": getattr(r, "mcq_correct", 0),
                    "tf_correct": getattr(r, "tf_correct", 0),
                    "numeric_correct": getattr(r, "numeric_correct", 0),
                    "tf_compare": getattr(r, "tf_compare", ""),
                    "numeric_compare": getattr(r, "numeric_compare", ""),
                    "correct": r.correct,
                    "wrong": r.wrong,
                    "blank": r.blank,
                    "score": r.score,
                    "recheck_score": (prev_subject_scores.get(sid_key, {}) or {}).get("recheck_score", ""),
                    "baithiphuctra": (prev_subject_scores.get(sid_key, {}) or {}).get("baithiphuctra", ""),
                    "phase": phase_marker,
                    "phase_timestamp": phase["timestamp"],
                    "phase_mode": mode_text,
                    "note": str(getattr(r, "scoring_note", "") or ""),
                    "status": row_status,
                }

        self.scoring_results_by_subject[subject] = subject_scores
        self._persist_scoring_results(subject, phase=phase)
        # DB is canonical source: reload full payload (especially after "Chỉ tính bài chưa có điểm")
        # so the grid always shows newest complete data from storage.
        self.scoring_results_by_subject.pop(subject, None)
        self._load_cached_scoring_results_for_subject(subject)
        refreshed_map = self._load_scoring_results_for_subject_from_storage(subject)
        self.score_rows = [self.scoring_engine.score_result_from_dict(dict(x)) for x in refreshed_map.values() if isinstance(x, dict)]
        self._update_scoring_status_bar(success_count, error_count, edited_count, pending_count)
        self._apply_scoring_filter()
        self._current_scoring_subject = subject
        self._scoring_dirty_subjects.discard(subject)
        self._persist_runtime_session_state_quietly()
        return rows

    def calculate_scores(self, subject_key: str = "", mode: str = "Tính lại toàn bộ", note: str = "") -> list:
        subject = (subject_key or self._resolve_preferred_scoring_subject() or "General").strip()
        subject_cfg = self._subject_config_by_subject_key(subject) or {}
        is_direct_score_subject = self._subject_uses_direct_score_import(subject_cfg)
        self._apply_subject_section_visibility(subject)

        mode_text = (mode or ("Nhập điểm trực tiếp" if is_direct_score_subject else "Tính lại toàn bộ")).strip()
        prev_subject_scores = self.scoring_results_by_subject.get(subject, {})
        rows = []
        missing = 0
        failed_scans: list[dict[str, str]] = []
        smart_scored_scans: list[dict[str, str]] = []
        edited_student_ids: set[str] = set()

        if is_direct_score_subject:
            imported_payloads = self._build_direct_score_payloads_for_subject(subject)
            if not imported_payloads:
                silent_missing_notes = {"auto_refresh_subject_change", "auto_refresh_open_scoring"}
                if str(note or "").strip() not in silent_missing_notes:
                    QMessageBox.warning(self, "Missing data", "Môn tự luận chưa có dữ liệu import điểm trực tiếp.")
                return []
            for payload in imported_payloads:
                sid_key = str(payload.get("student_id", "") or "").strip()
                if mode_text == "Chỉ tính bài chưa có điểm" and sid_key and sid_key in prev_subject_scores:
                    continue
                if sid_key and bool(payload.get("manually_edited", False)):
                    edited_student_ids.add(sid_key)
                row_obj = self.scoring_engine.score_result_from_dict(dict(payload))
                setattr(row_obj, "scoring_note", str(payload.get("note", "") or "Nhập điểm trực tiếp"))
                setattr(row_obj, "exam_room", str(payload.get("exam_room", "") or ""))
                rows.append(row_obj)
            return self._finalize_scoring_rows(
                subject,
                rows,
                prev_subject_scores,
                edited_student_ids,
                mode_text,
                note or "essay_direct_import",
                missing,
                failed_scans,
                smart_scored_scans,
            )

        # Chấm điểm chỉ đọc từ DB; không lấy fallback từ grid/config/runtime snapshot.
        subject_scans = list(self._refresh_scan_results_from_db(subject) or [])
        self.scan_results_by_subject[self._batch_result_subject_key(subject)] = list(subject_scans)

        if not subject_scans:
            silent_missing_notes = {"auto_refresh_subject_change", "auto_refresh_open_scoring"}
            if str(note or "").strip() not in silent_missing_notes:
                QMessageBox.warning(self, "Missing data", "Môn này chưa có dữ liệu Batch Scan để tính điểm.")
            return []

        self._ensure_answer_keys_for_subject(subject)
        if not self.answer_keys:
            QMessageBox.warning(self, "Missing data", "Không tìm thấy đáp án cho môn đã chọn. Vui lòng kiểm tra cấu hình môn.")
            return []

        all_subject_keys: list[SubjectKey] = []
        if self.answer_keys is not None:
            for key_obj in self.answer_keys.keys.values():
                if isinstance(key_obj, SubjectKey) and str(getattr(key_obj, "subject", "") or "").strip() == subject:
                    all_subject_keys.append(key_obj)
        fetched_keys = self._fetch_answer_keys_for_subject_scoped(subject)
        for exam_code, key_obj in (fetched_keys or {}).items():
            if not isinstance(key_obj, SubjectKey):
                continue
            if any(str(k.exam_code or "").strip() == str(exam_code or "").strip() for k in all_subject_keys):
                continue
            all_subject_keys.append(key_obj)

        for scan_row in (subject_scans or []):
            sid_key = str(getattr(scan_row, "student_id", "") or "").strip()
            if not sid_key:
                continue
            has_history = bool([str(x) for x in (getattr(scan_row, "edit_history", []) or []) if str(x or "").strip()])
            if bool(getattr(scan_row, "manually_edited", False)) or has_history:
                edited_student_ids.add(sid_key)

        for scan in subject_scans:
            sid = (scan.student_id or "").strip()
            profile = self._student_profile_by_id(sid)
            if profile.get("name") and not str(getattr(scan, "full_name", "") or "").strip():
                setattr(scan, "full_name", profile.get("name"))
            if profile.get("birth_date") and not str(getattr(scan, "birth_date", "") or "").strip():
                setattr(scan, "birth_date", profile.get("birth_date"))
            if profile.get("class_name") and not str(getattr(scan, "class_name", "") or "").strip():
                setattr(scan, "class_name", profile.get("class_name"))

            if mode_text == "Chỉ tính bài chưa có điểm" and sid and sid in prev_subject_scores:
                continue

            full_name = str(getattr(scan, "full_name", "") or profile.get("name", "") or "").strip()
            class_name = str(getattr(scan, "class_name", "") or profile.get("class_name", "") or "").strip()
            birth_date = str(getattr(scan, "birth_date", "") or profile.get("birth_date", "") or "").strip()

            if not sid or not full_name or not class_name or not birth_date:
                err_msg = "Lỗi thông tin thí sinh: thiếu SBD/Họ tên/Lớp/Ngày sinh"
                error_row = self.scoring_engine.score_result_from_dict({
                    "student_id": sid or "-",
                    "name": full_name or "-",
                    "subject": subject,
                    "exam_code": str(getattr(scan, "exam_code", "") or ""),
                    "mcq_correct": 0,
                    "tf_correct": 0,
                    "numeric_correct": 0,
                    "correct": 0,
                    "wrong": 0,
                    "blank": 0,
                    "score": 0.0,
                    "class_name": class_name or "-",
                    "birth_date": birth_date or "-",
                })
                setattr(error_row, "scoring_note", err_msg)
                rows.append(error_row)
                failed_scans.append({"file": str(getattr(scan, "image_path", "") or "-"), "reason": err_msg})
                continue

            key = self.answer_keys.get_flexible(subject, scan.exam_code)
            if not key:
                exam_code_text = str(getattr(scan, "exam_code", "") or "").strip()
                has_student_identity = bool(sid and (profile.get("name") or profile.get("class_name") or profile.get("birth_date")))
                allow_smart = (not exam_code_text) or ("?" in exam_code_text) or (bool(exam_code_text) and has_student_identity)
                best_row = None
                best_key_code = ""
                if allow_smart:
                    for candidate_key in all_subject_keys:
                        try:
                            cand_row = self.scoring_engine.score(
                                scan,
                                candidate_key,
                                student_name=str(getattr(scan, "full_name", "") or ""),
                                subject_config=subject_cfg,
                            )
                        except Exception:
                            continue
                        if best_row is None or float(getattr(cand_row, "score", 0.0) or 0.0) > float(getattr(best_row, "score", 0.0) or 0.0):
                            best_row = cand_row
                            best_key_code = str(getattr(candidate_key, "exam_code", "") or "").strip()
                if allow_smart and best_row is not None:
                    setattr(best_row, "scoring_note", "Chấm thông minh")
                    setattr(best_row, "class_name", str(getattr(scan, "class_name", "") or profile.get("class_name", "") or ""))
                    setattr(best_row, "birth_date", str(getattr(scan, "birth_date", "") or profile.get("birth_date", "") or ""))
                    if best_key_code:
                        try:
                            setattr(scan, "exam_code", best_key_code)
                            self.database.update_scan_result_payload(
                                self._batch_result_subject_key(subject),
                                str(getattr(scan, "image_path", "") or ""),
                                self._serialize_omr_result(scan),
                                note="smart_scoring_exam_code_pick",
                            )
                        except Exception:
                            pass
                    rows.append(best_row)
                    smart_scored_scans.append({
                        "file": str(getattr(scan, "image_path", "") or "-"),
                        "student_id": sid or "-",
                        "picked_exam_code": best_key_code or "-",
                    })
                    continue

                err_msg = f"Không tìm thấy đáp án phù hợp cho mã đề '{exam_code_text or '-'}'"
                missing += 1
                error_row = self.scoring_engine.score_result_from_dict({
                    "student_id": sid or "-",
                    "name": str(getattr(scan, "full_name", "") or profile.get("name", "") or "-"),
                    "subject": subject,
                    "exam_code": exam_code_text,
                    "mcq_correct": 0,
                    "tf_correct": 0,
                    "numeric_correct": 0,
                    "correct": 0,
                    "wrong": 0,
                    "blank": 0,
                    "score": 0.0,
                    "class_name": str(getattr(scan, "class_name", "") or profile.get("class_name", "") or "-"),
                    "birth_date": str(getattr(scan, "birth_date", "") or profile.get("birth_date", "") or "-"),
                })
                setattr(error_row, "scoring_note", err_msg)
                rows.append(error_row)
                failed_scans.append({
                    "file": str(getattr(scan, "image_path", "") or "-"),
                    "reason": err_msg,
                })
                continue

            try:
                scored = self.scoring_engine.score(
                    scan,
                    key,
                    student_name=str(getattr(scan, "full_name", "") or ""),
                    subject_config=subject_cfg,
                )
                setattr(scored, "class_name", str(getattr(scan, "class_name", "") or profile.get("class_name", "") or ""))
                setattr(scored, "birth_date", str(getattr(scan, "birth_date", "") or profile.get("birth_date", "") or ""))
                rows.append(scored)
            except Exception as exc:
                err_msg = f"Lỗi chấm điểm: {exc}"
                error_row = self.scoring_engine.score_result_from_dict({
                    "student_id": sid or "-",
                    "name": str(getattr(scan, "full_name", "") or profile.get("name", "") or "-"),
                    "subject": subject,
                    "exam_code": str(getattr(scan, "exam_code", "") or ""),
                    "mcq_correct": 0,
                    "tf_correct": 0,
                    "numeric_correct": 0,
                    "correct": 0,
                    "wrong": 0,
                    "blank": 0,
                    "score": 0.0,
                    "class_name": str(getattr(scan, "class_name", "") or profile.get("class_name", "") or "-"),
                    "birth_date": str(getattr(scan, "birth_date", "") or profile.get("birth_date", "") or "-"),
                })
                setattr(error_row, "scoring_note", err_msg)
                rows.append(error_row)
                failed_scans.append({
                    "file": str(getattr(scan, "image_path", "") or "-"),
                    "reason": err_msg,
                })

        return self._finalize_scoring_rows(
            subject,
            rows,
            prev_subject_scores,
            edited_student_ids,
            mode_text,
            note,
            missing,
            failed_scans,
            smart_scored_scans,
        )

    def action_open_recheck(self) -> None:
        from gui.main_window_recheck import open_recheck_dialog

        return open_recheck_dialog(self)
