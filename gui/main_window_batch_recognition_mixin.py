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


class MainWindowBatchRecognitionMixin:
    """Image discovery, template resolution, OMR recognition and full batch execution."""
    def _batch_context_session_path(self) -> Path | None:
        if self.current_session_id:
            return self._session_path_from_id(self.current_session_id)
        if self.current_session_path:
            return self.current_session_path
        if self.batch_editor_return_session_id:
            return self._session_path_from_id(self.batch_editor_return_session_id)
        return None

    def _recognized_image_paths_for_subject(self, subject_key: str) -> set[str]:
        key = str(subject_key or "").strip()
        scoped_key = self._batch_result_subject_key(key)
        rows = list(self.scan_results_by_subject.get(scoped_key) or [])
        if not rows and key:
            rows = list(self._refresh_scan_results_from_db(key) or [])
        recognized = {
            self._result_identity_key(str(getattr(item, "image_path", "") or ""))
            for item in rows
            if self._result_identity_key(str(getattr(item, "image_path", "") or ""))
        }
        recognized |= self._deleted_scan_images_for_subject(key)
        return recognized

    def _deleted_scan_images_for_subject(self, subject_key: str) -> set[str]:
        subject = str(subject_key or "").strip()
        if not subject:
            return set()
        scoped_key = self._batch_result_subject_key(subject)
        runtime_set = set(self.deleted_scan_images_by_subject.get(scoped_key, set()) or set())
        cfg = self._session_subject_config_ref_by_subject_key(subject)
        if not isinstance(cfg, dict):
            cfg = self._subject_config_by_subject_key(subject)
        cfg_list = set()
        if isinstance(cfg, dict):
            cfg_list = {self._result_identity_key(str(x)) for x in (cfg.get("deleted_scan_images", []) or []) if str(x).strip()}
        return {x for x in (runtime_set | cfg_list) if x}

    def _set_deleted_scan_images_for_subject(self, subject_key: str, images: set[str]) -> None:
        subject = str(subject_key or "").strip()
        if not subject:
            return
        normalized = {self._result_identity_key(x) for x in (images or set()) if self._result_identity_key(x)}
        scoped_key = self._batch_result_subject_key(subject)
        self.deleted_scan_images_by_subject[scoped_key] = set(normalized)
        cfg = self._session_subject_config_ref_by_subject_key(subject)
        if isinstance(cfg, dict):
            cfg["deleted_scan_images"] = sorted(normalized)
            self.session_dirty = True
            # Persist trực tiếp session.config vì đây mới là nguồn dữ liệu thật
            # mà auto-recognition đọc lại sau khi khởi động ứng dụng.
            self._persist_runtime_session_state_quietly()

    def _mark_deleted_scan_image(self, subject_key: str, image_path: str) -> None:
        subject = str(subject_key or "").strip()
        image_key = self._result_identity_key(image_path)
        if not subject or not image_key:
            return
        deleted_set = self._deleted_scan_images_for_subject(subject)
        deleted_set.add(image_key)
        self._set_deleted_scan_images_for_subject(subject, deleted_set)

    def _unmark_deleted_scan_images(self, subject_key: str, image_paths: list[str]) -> None:
        subject = str(subject_key or "").strip()
        if not subject:
            return
        deleted_set = self._deleted_scan_images_for_subject(subject)
        for path in image_paths or []:
            image_key = self._result_identity_key(path)
            if image_key:
                deleted_set.discard(image_key)
        self._set_deleted_scan_images_for_subject(subject, deleted_set)

    def _configured_scan_file_paths(self, cfg: dict | None) -> list[str]:
        if not cfg:
            return []
        scan_folder = str(cfg.get("scan_folder", "") or ((self.session.config or {}).get("scan_root", "") if self.session else "") or "").strip()
        if not scan_folder or scan_folder == "-":
            return []
        scan_dir = Path(scan_folder)
        if not scan_dir.exists() or not scan_dir.is_dir():
            return []
        scan_mode = str(cfg.get("scan_mode", "") or (self.session.config or {}).get("scan_mode", "") if self.session else "")
        image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        if self._is_subfolder_scan_mode(scan_mode):
            return [str(p) for p in sorted(scan_dir.rglob("*")) if p.is_file() and p.suffix.lower() in image_exts]
        return [str(p) for p in sorted(scan_dir.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]

    def _current_batch_file_scope_text(self) -> str:
        cfg = self._selected_batch_subject_config()
        if cfg:
            cfg = self._merge_saved_batch_snapshot(cfg)
        all_paths = self._configured_scan_file_paths(cfg)
        instance_key = self._subject_instance_key_from_cfg(cfg) if cfg else ""
        recognized = self._recognized_image_paths_for_subject(instance_key)
        deleted = self._deleted_scan_images_for_subject(instance_key)
        recognized_live = recognized - deleted
        recognized_count = sum(1 for path in all_paths if self._result_identity_key(str(path)) in recognized_live)
        deleted_count = sum(1 for path in all_paths if self._result_identity_key(str(path)) in deleted)
        pending_count = max(0, len(all_paths) - recognized_count - deleted_count)
        mode = str(self.batch_file_scope_combo.currentData() or "new_only") if hasattr(self, "batch_file_scope_combo") else "new_only"
        mode_label = "File mới" if mode == "new_only" else "Toàn bộ"
        return f"{mode_label} | Đã nhận diện: {recognized_count} | Đã xoá trong Batch Scan: {deleted_count} | Chưa nhận diện: {pending_count}"

    @staticmethod
    def _recommended_batch_timeout_sec(template: Template | None) -> float:
        if template is None:
            return 3.0
        zones = list(getattr(template, "zones", []) or [])
        answer_zone_count = sum(1 for z in zones if getattr(z, "zone_type", None) in {ZoneType.MCQ_BLOCK, ZoneType.TRUE_FALSE_BLOCK, ZoneType.NUMERIC_BLOCK})
        id_zone_count = sum(1 for z in zones if getattr(z, "zone_type", None) in {ZoneType.STUDENT_ID_BLOCK, ZoneType.EXAM_CODE_BLOCK})
        other_zone_count = max(0, len(zones) - answer_zone_count - id_zone_count)
        timeout = 2.5 + (0.25 * answer_zone_count) + (0.15 * id_zone_count) + (0.10 * other_zone_count)
        return float(min(6.0, max(2.5, timeout)))

    def _batch_subject_refresh_signature(self, cfg: dict | None) -> str:
        if not isinstance(cfg, dict):
            return ""
        subject_key = self._subject_key_from_cfg(cfg)
        imported_codes = sorted(str(k).strip() for k in (cfg.get("imported_answer_keys", {}) or {}).keys() if str(k).strip())
        fetched_codes = sorted(str(k).strip() for k in (self._fetch_answer_keys_for_subject_scoped(subject_key, cfg) or {}).keys() if str(k).strip())
        student_rows = []
        for st in (self.session.students or []) if self.session else []:
            extra = dict(getattr(st, "extra", {}) or {})
            student_rows.append(
                (
                    str(getattr(st, "student_id", "") or "").strip(),
                    str(getattr(st, "name", "") or "").strip(),
                    str(extra.get("birth_date", "") or "").strip(),
                    str(extra.get("class_name", "") or "").strip(),
                    str(extra.get("exam_room", "") or "").strip(),
                )
            )
        signature_payload = {
            "subject_key": str(subject_key or ""),
            "template_path": self._normalize_template_path(str(cfg.get("template_path", "") or "")),
            "scan_folder": str(cfg.get("scan_folder", "") or "").strip(),
            "answer_key_key": str(cfg.get("answer_key_key", "") or "").strip(),
            "imported_codes": imported_codes,
            "fetched_codes": fetched_codes,
            "students": student_rows,
        }
        return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)

    def _on_batch_subject_changed(self, _index: int, force_reload: bool = False) -> None:
        if self._switching_batch_subject:
            return
        self._switching_batch_subject = True
        try:
            previous_runtime_key = str(getattr(self, "active_batch_subject_key", "") or "").strip()
            cfg = self._selected_batch_subject_config()
            next_runtime_key = ""
            if cfg:
                cfg = self._merge_saved_batch_snapshot(cfg)
                next_runtime_key = self._batch_runtime_key(cfg)
            if previous_runtime_key and previous_runtime_key != next_runtime_key:
                pass
            if cfg:
                self.active_batch_subject_key = next_runtime_key
            else:
                self.active_batch_subject_key = None
            self._load_batch_subject_state(cfg, source_hint="subject_changed", force_reload=force_reload)
        finally:
            self._switching_batch_subject = False

    def _load_batch_subject_state(self, subject_cfg: dict | None, source_hint: str = "", force_reload: bool = False) -> bool:
        cfg = dict(subject_cfg or {}) if isinstance(subject_cfg, dict) else {}
        if cfg:
            cfg = self._merge_saved_batch_snapshot(cfg)

        if not cfg:
            self._reset_batch_subject_ui_state()
            wait_dlg = self._open_wait_progress("Đang tải dữ liệu môn, vui lòng chờ...", "Batch Scan")
            self.batch_template_value.setText("-")
            self.batch_template_path_value = "-"
            self.batch_answer_codes_value.setText("-")
            self.batch_student_id_value.setText("-")
            self.batch_scan_folder_value.setText("-")
            self.batch_scan_state_value.setText("-")
            if hasattr(self, "batch_context_value"):
                self.batch_context_value.setText("-")
            self._current_batch_data_source = "empty"
            self._batch_loaded_runtime_key = ""
            self._batch_loaded_subject_signature = ""
            self._close_wait_progress(wait_dlg)
            return False

        subject_key = self._subject_key_from_cfg(cfg)
        runtime_key = self._batch_runtime_key(subject_key)

        # Không clear/reload lại khi vẫn là đúng môn đang mở và không có yêu cầu ép reload.
        cfg_signature = self._batch_subject_refresh_signature(cfg)

        if (
            not force_reload
            and subject_key
            and runtime_key
            and str(getattr(self, "_batch_loaded_runtime_key", "") or "").strip() == runtime_key
            and str(getattr(self, "_batch_loaded_subject_signature", "") or "").strip() == cfg_signature
            and str(getattr(self, "_current_batch_data_source", "") or "").strip() == "database"
            and bool(self.scan_results or self.scan_list.rowCount() > 0)
        ):
            fetched_codes = sorted(str(k).strip() for k in (self._fetch_answer_keys_for_subject_scoped(subject_key, cfg) or {}).keys() if str(k).strip())
            imported_codes = sorted(str(k).strip() for k in (cfg.get("imported_answer_keys", {}) or {}).keys() if str(k).strip())
            codes = ", ".join(sorted(set(imported_codes) | set(fetched_codes))) or "-"
            self.batch_answer_codes_value.setText(codes)
            self._update_batch_scan_scope_summary()
            return True

        self._reset_batch_subject_ui_state()
        wait_dlg = self._open_wait_progress("Đang tải dữ liệu môn, vui lòng chờ...", "Batch Scan")

        if subject_key and subject_key not in self._answer_keys_ready_subjects:
            self._ensure_answer_keys_for_subject(subject_key)
            self._answer_keys_ready_subjects.add(subject_key)

        template_path = self._normalize_template_path(str(cfg.get("template_path", "") or "")) or self._normalize_template_path(str(self.session.template_path if self.session else "")) or "-"
        scan_folder = str(cfg.get("scan_folder", "") or ((self.session.config or {}).get("scan_root", "") if self.session else "") or "-")
        imported_codes = sorted(str(k).strip() for k in (cfg.get("imported_answer_keys", {}) or {}).keys() if str(k).strip())
        fetched_codes = sorted(str(k).strip() for k in (self._fetch_answer_keys_for_subject_scoped(subject_key, cfg) or {}).keys() if str(k).strip())
        codes = ", ".join(sorted(set(imported_codes) | set(fetched_codes))) or "-"
        self.batch_template_path_value = template_path
        template_display = Path(template_path).stem if template_path and template_path != "-" else "-"
        self.batch_template_value.setText(template_display)
        self.batch_answer_codes_value.setText(codes)
        self.batch_scan_folder_value.setText(scan_folder)

        tp = Path(template_path) if template_path and template_path != "-" else None
        template_cache_key = str(tp.resolve()) if tp and tp.exists() else ""
        tpl_for_view = self._template_cache_by_path.get(template_cache_key) if template_cache_key else None
        if tpl_for_view is None and tp and tp.exists():
            try:
                tpl_for_view = Template.load_json(tp)
                if template_cache_key:
                    self._template_cache_by_path[template_cache_key] = tpl_for_view
            except Exception:
                tpl_for_view = self.template
        if tpl_for_view is None:
            tpl_for_view = self.template
        has_sid = "Có" if (tpl_for_view and any(z.zone_type.value == "STUDENT_ID_BLOCK" for z in tpl_for_view.zones)) else "Không"
        self.batch_student_id_value.setText(has_sid)
        if tpl_for_view:
            self._apply_template_recognition_settings(tpl_for_view)
            self.template = tpl_for_view

        if self._subject_uses_direct_score_import(cfg):
            self.scan_results = []
            self.scan_files = []
            if hasattr(self, "scan_list"):
                self.scan_list.setRowCount(0)
            self.batch_scan_state_value.setText("Môn tự luận - dùng import điểm trực tiếp")
            self._current_batch_data_source = "essay_direct_import"
            self._batch_loaded_runtime_key = runtime_key
            self._batch_loaded_subject_signature = cfg_signature
            self._clear_batch_preview_panels()
            self._update_batch_scan_scope_summary()
            self._close_wait_progress(wait_dlg)
            return True

        loaded_results: list[OMRResult] = list(self._refresh_scan_results_from_db(subject_key) or [])
        scoped_subject = self._batch_result_subject_key(subject_key)
        if loaded_results:
            self.scan_results = list(loaded_results)
            self.scan_results_by_subject[scoped_subject] = list(self.scan_results)
            self._populate_scan_grid_from_results(self.scan_results, skip_expensive_checks=False)
            source = "database"
        else:
            self.scan_results = []
            self.scan_results_by_subject[scoped_subject] = []
            source = "empty"

        self._finalize_batch_scan_display(refresh_statuses=True)
        self.btn_save_batch_subject.setEnabled(False)
        self._current_batch_data_source = source
        self._batch_loaded_runtime_key = runtime_key if source != "empty" else ""
        self._batch_loaded_subject_signature = cfg_signature if source != "empty" else ""
        self._update_batch_scan_scope_summary()
        if self.scan_results:
            self._debug_scan_result_state("restore_subject_loaded_first_row", self.scan_results[0])
        self._close_wait_progress(wait_dlg)
        return source != "empty"

    def _hydrate_result_from_saved_row_payload(self, result: OMRResult | None, row_payload: dict | None) -> OMRResult | None:
        if result is None and not isinstance(row_payload, dict):
            return result
        row = dict(row_payload or {}) if isinstance(row_payload, dict) else {}
        if result is None:
            result = OMRResult(
                image_path=str(row.get("image_path", "") or ""),
                student_id=str(row.get("student_id", "") or ""),
                exam_code=str(row.get("exam_code", "") or ""),
            )

        payload_sources: list[dict] = []
        serialized_payload = dict(row.get("serialized_result", {}) or {}) if isinstance(row.get("serialized_result", {}), dict) else {}
        if serialized_payload:
            payload_sources.append(serialized_payload)
        if row:
            payload_sources.append(row)

        current_mcq = self._normalized_mcq_answer_map(getattr(result, "mcq_answers", {}) or {})
        current_tf = self._normalized_tf_answer_map(getattr(result, "true_false_answers", {}) or {})
        current_numeric = self._normalized_numeric_answer_map(getattr(result, "numeric_answers", {}) or {})

        for payload in payload_sources:
            payload_mcq = self._normalized_mcq_answer_map(payload.get("mcq_answers", {}) or {})
            if payload_mcq and (not current_mcq or len(current_mcq) < len(payload_mcq)):
                current_mcq = dict(payload_mcq)

            payload_tf = self._normalized_tf_answer_map(payload.get("true_false_answers", {}) or {})
            if payload_tf:
                merged_tf = dict(current_tf)
                for q_no, flags in payload_tf.items():
                    existing = dict(merged_tf.get(q_no, {}) or {})
                    if len(existing) < len(flags):
                        merged_tf[q_no] = dict(flags)
                    else:
                        for key, value in flags.items():
                            if key not in existing:
                                existing[key] = bool(value)
                        merged_tf[q_no] = existing
                current_tf = merged_tf

            payload_numeric = self._normalized_numeric_answer_map(payload.get("numeric_answers", {}) or {})
            if payload_numeric:
                merged_numeric = dict(current_numeric)
                for q_no, value in payload_numeric.items():
                    current_value = str(merged_numeric.get(q_no, "") or "").strip()
                    payload_value = str(value or "").strip()
                    if (q_no not in merged_numeric) or (not current_value and payload_value):
                        merged_numeric[q_no] = payload_value
                current_numeric = merged_numeric

            for attr_name in [
                "image_path",
                "student_id",
                "exam_code",
                "full_name",
                "birth_date",
                "exam_room",
                "class_name",
                "answer_string",
                "manual_content_override",
                "cached_content",
                "cached_status",
                "cached_recognized_short",
                "recognized_template_path",
                "recognized_alignment_profile",
                "last_adjustment",
            ]:
                current_value = str(getattr(result, attr_name, "") or "").strip()
                payload_value = str(payload.get(attr_name, "") or "").strip()
                if payload_value and not current_value:
                    setattr(result, attr_name, payload_value)

            current_errors = list(getattr(result, "recognition_errors", []) or [])
            payload_errors = [str(x) for x in (payload.get("recognition_errors", []) or []) if str(x or "").strip()]
            if payload_errors and not current_errors:
                setattr(result, "recognition_errors", payload_errors)
            current_edit_history = [str(x) for x in (getattr(result, "edit_history", []) or []) if str(x or "").strip()]
            payload_edit_history = [str(x) for x in (payload.get("edit_history", []) or []) if str(x or "").strip()]
            if payload_edit_history and not current_edit_history:
                setattr(result, "edit_history", payload_edit_history)
            current_manual_adjustments = [str(x) for x in (getattr(result, "manual_adjustments", []) or []) if str(x or "").strip()]
            payload_manual_adjustments = [str(x) for x in (payload.get("manual_adjustments", []) or []) if str(x or "").strip()]
            if payload_manual_adjustments and not current_manual_adjustments:
                setattr(result, "manual_adjustments", payload_manual_adjustments)
            forced_status_value = str(payload.get("cached_forced_status", payload.get("forced_status", "")) or "").strip()
            current_forced_status = str(getattr(result, "cached_forced_status", "") or "").strip()
            if forced_status_value and not current_forced_status:
                setattr(result, "cached_forced_status", forced_status_value)
            if bool(payload.get("manually_edited", False)) or bool(payload_edit_history) or forced_status_value == "Đã sửa":
                setattr(result, "manually_edited", True)
                if not str(getattr(result, "cached_forced_status", "") or "").strip():
                    setattr(result, "cached_forced_status", "Đã sửa")
                if not str(getattr(result, "cached_status", "") or "").strip():
                    setattr(result, "cached_status", "Đã sửa")

            for attr_name, default_value in [
                ("recognized_fill_threshold", 0.45),
                ("recognized_empty_threshold", 0.20),
                ("recognized_certainty_margin", 0.08),
            ]:
                current_value = getattr(result, attr_name, None)
                payload_value = payload.get(attr_name, None)
                if payload_value is not None and (current_value is None or float(current_value or default_value) == float(default_value)):
                    try:
                        setattr(result, attr_name, float(payload_value))
                    except Exception:
                        pass

            cached_blank_summary = dict(payload.get("cached_blank_summary", {}) or {}) if isinstance(payload.get("cached_blank_summary", {}), dict) else {}
            if cached_blank_summary and not dict(getattr(result, "cached_blank_summary", {}) or {}):
                setattr(result, "cached_blank_summary", cached_blank_summary)

        if current_mcq:
            result.mcq_answers = current_mcq
        if current_tf:
            result.true_false_answers = current_tf
        if current_numeric:
            result.numeric_answers = current_numeric
        result.sync_legacy_aliases()
        return result

    def _cache_working_batch_state(self, subject_key: str) -> None:
        # DB-only mode: không giữ working cache theo môn.
        return

    def _restore_cached_working_batch_state(self, subject_key: str) -> bool:
        # DB-only mode: không phục hồi từ cache tạm.
        return False

    def _batch_runtime_key(self, subject_key_or_cfg) -> str:
        # canonical runtime key for batch UI caches/scans.
        if isinstance(subject_key_or_cfg, dict):
            subject_key = self._subject_instance_key_from_cfg(subject_key_or_cfg)
        else:
            subject_key = str(subject_key_or_cfg or "").strip()
        return self._normalize_subject_runtime_key(subject_key)

    def _normalize_subject_runtime_key(self, key: str) -> str:
        key_text = str(key or "").strip()
        if not key_text:
            return ""
        scope_prefix = self._session_scope_prefix()
        if scope_prefix and key_text.startswith(f"{scope_prefix}::"):
            return key_text
        return self._batch_result_subject_key(key_text)

    def _reset_batch_subject_ui_state(self) -> None:
        # reset stale subject UI
        self.scan_results = []
        self.scan_files = []
        self.scan_blank_questions.clear()
        self.scan_blank_summary.clear()
        self.scan_manual_adjustments.clear()
        self.scan_edit_history.clear()
        self.scan_last_adjustment.clear()
        self.scan_forced_status_by_index.clear()
        self.preview_rotation_by_index.clear()
        self._batch_loaded_subject_signature = ""
        self.batch_status_filter_mode = "all"
        if hasattr(self, "scan_list"):
            self.scan_list.clearSelection()
            self.scan_list.setRowCount(0)
        self._clear_batch_preview_panels()
        if hasattr(self, "error_list"):
            self.error_list.clear()
        if hasattr(self, "progress"):
            self.progress.setValue(0)
        self.preview_source_pixmap = QPixmap()
        self.preview_zoom_factor = 0.3
        if hasattr(self, "btn_zoom_reset"):
            self.btn_zoom_reset.setText("30%")
        if hasattr(self, "btn_save_batch_subject"):
            self.btn_save_batch_subject.setEnabled(False)
        self._update_batch_scan_bottom_status_text()

    @staticmethod
    def _has_valid_identity(result) -> bool:
        sid = str(getattr(result, "student_id", "") or "").strip()
        code = str(getattr(result, "exam_code", "") or "").strip()
        has_id = bool(sid and "?" not in sid)
        has_code = bool(code and "?" not in code)
        return has_id or has_code

    @staticmethod
    def _result_has_meaningful_recognition(result) -> bool:
        has_identity = MainWindowBatchRecognitionMixin._has_valid_identity(result)
        has_answers = bool((result.mcq_answers or {}) or (result.true_false_answers or {}) or (result.numeric_answers or {}))
        return has_answers or has_identity

    @staticmethod
    def _should_force_image_error_status(result) -> bool:
        issues = list(getattr(result, "issues", []) or [])
        if any(str(getattr(issue, "code", "") or "").upper() == "FILE" for issue in issues):
            return True
        return not MainWindowBatchRecognitionMixin._result_has_meaningful_recognition(result)

    @staticmethod
    def _preferred_forced_status(result) -> str:
        issues = [str(getattr(issue, "code", "") or "").strip().upper() for issue in (getattr(result, "issues", []) or [])]
        if "FILE" in issues:
            return "Lỗi file ảnh"
        if "POOR_IDENTIFIER_ZONE" in issues:
            return "Không đủ chất lượng"
        if "POOR_IMAGE" in issues or "FAST_FAIL_POOR_SCAN" in issues:
            return "Ảnh xấu"
        if "IDENTIFIER_FAST_CAP" in issues or "STUDENT_ID_FAST_FAIL" in issues:
            return "Giới hạn SBD"
        if "SCANNER_LOCK_FAIL" in issues:
            return "Scanner lock fail"
        if "SAFE_FALLBACK_USED" in issues:
            return "Safe fallback used"
        if "IDENTIFIER_TIMEOUT" in issues:
            return "Timeout vùng SBD"
        if "TIMEOUT" in issues:
            return "Timeout nhận dạng"
        if not MainWindowBatchRecognitionMixin._result_has_meaningful_recognition(result):
            return "Lỗi file ảnh"
        return ""

    @staticmethod
    def _result_is_poor_image(result) -> bool:
        issues = [str(getattr(issue, "code", "") or "").strip().upper() for issue in (getattr(result, "issues", []) or [])]
        if "POOR_IMAGE" in issues or "FAST_FAIL_POOR_SCAN" in issues:
            return True
        alignment_debug = dict(getattr(result, "alignment_debug", {}) or {})
        return bool(alignment_debug.get("poor_image", False))

    @staticmethod
    def _recognition_quality_score(result) -> int:
        sid = str(getattr(result, "student_id", "") or "").strip()
        code = str(getattr(result, "exam_code", "") or "").strip()
        has_id = 1 if sid and "?" not in sid else 0
        has_code = 1 if code and "?" not in code else 0
        answers_count = len(result.mcq_answers or {}) + len(result.true_false_answers or {}) + len(result.numeric_answers or {})
        penalty = len(getattr(result, "issues", []) or []) + len(getattr(result, "recognition_errors", []) or getattr(result, "errors", []) or [])
        return has_id * 3 + has_code * 3 + answers_count - penalty

    def _apply_template_recognition_settings(self, template: Template, *, sync_mode_selector: bool = True) -> None:
        if not template:
            return
        md = template.metadata if isinstance(template.metadata, dict) else {}

        mode = str(md.get("alignment_profile", "") or "").strip().lower()
        if sync_mode_selector and mode in {"auto", "legacy", "border", "hybrid", "one_side"}:
            setattr(self.omr_processor, "alignment_profile", mode)
            if hasattr(self, "batch_recognition_mode_combo"):
                for i in range(self.batch_recognition_mode_combo.count()):
                    if str(self.batch_recognition_mode_combo.itemData(i) or "") == mode:
                        self.batch_recognition_mode_combo.blockSignals(True)
                        self.batch_recognition_mode_combo.setCurrentIndex(i)
                        self.batch_recognition_mode_combo.blockSignals(False)
                        break

        # Optional recognition thresholds can be embedded in template metadata.
        for field, default in (("fill_threshold", 0.45), ("empty_threshold", 0.20), ("certainty_margin", 0.08)):
            raw = md.get(field, None)
            if raw is None:
                continue
            try:
                value = float(raw)
            except Exception:
                continue
            setattr(self.omr_processor, field, value if value >= 0 else default)

    def _allow_batch_auto_rotate_retry(self) -> bool:
        template_md = (self.template.metadata if self.template and isinstance(self.template.metadata, dict) else {})
        raw = template_md.get("batch_auto_rotate_retry", False)
        if isinstance(raw, bool):
            return raw
        text = str(raw or "").strip().lower()
        return text in {"1", "true", "yes", "on"}

    def _skip_retry_for_poor_images(self) -> bool:
        template_md = (self.template.metadata if self.template and isinstance(self.template.metadata, dict) else {})
        raw = template_md.get("batch_skip_retry_for_poor_images", True)
        if isinstance(raw, bool):
            return raw
        return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}

    def _recognize_single_image(self, image_path: str, *, allow_retry: bool = False, context_tag: str = ""):
        cfg = self._selected_batch_subject_config() or self._resolve_subject_config_for_batch()
        template_path = self._resolve_template_path_for_subject(cfg)
        return self._recognize_image_with_exact_template(
            image_path,
            template_path,
            source_tag=context_tag or "single",
            allow_retry=allow_retry,
        )

    def _try_reprocess_result_rotated_180(self, result, template_path: str = "", source_tag: str = ""):
        image_path = str(getattr(result, "image_path", "") or "").strip()
        if not image_path or not Path(image_path).exists():
            return result, False
        pix = QPixmap(image_path)
        if pix.isNull():
            return result, False
        rotated = pix.transformed(QTransform().rotate(180.0), Qt.SmoothTransformation)
        temp_path = str(Path(image_path).with_name(f".{Path(image_path).stem}_tmp_auto180.png"))
        if not rotated.save(temp_path):
            return result, False
        try:
            chosen_template = template_path or self._resolve_template_path_for_subject(self._selected_batch_subject_config() or self._resolve_subject_config_for_batch())
            alt = self._recognize_image_with_exact_template(temp_path, chosen_template, source_tag=source_tag or "retry_180", allow_retry=False)
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
        alt.image_path = image_path
        if self._recognition_quality_score(alt) > self._recognition_quality_score(result):
            return alt, True
        return result, False

    def _resolve_template_path_for_subject(self, subject_cfg: dict | None = None) -> str:
        cfg = subject_cfg if isinstance(subject_cfg, dict) else (self._selected_batch_subject_config() or self._resolve_subject_config_for_batch() or {})
        template_path = self._normalize_template_path(str((cfg or {}).get("template_path", "") or ""))
        if not template_path and self.session:
            template_path = self._normalize_template_path(str(self.session.template_path or ""))
        return str(template_path or "").strip()

    def _recognize_image_with_exact_template(self, image_path: str, template_path: str, source_tag: str = "", allow_retry: bool = False):
        path_text = str(template_path or "").strip()
        if not path_text:
            raise RuntimeError("Chưa cấu hình template cho môn hiện tại.")
        pth = Path(path_text)
        if not pth.exists():
            raise RuntimeError(f"Không tìm thấy template: {path_text}")
        loaded_template = Template.load_json(pth)
        self.template = loaded_template
        setattr(self, "_active_template_path", str(pth.resolve()))
        self._apply_template_recognition_settings(self.template, sync_mode_selector=False)
        # Batch Scan must use the real production path, not Template Editor's diagnostic
        # test path. run_recognition_test() enables collect_diagnostics=True internally,
        # which is useful for visual debugging but too slow for hundreds of sheets.
        # Keep identifier recognition forced, but do not build diagnostic artifacts.
        fast_ctx = RecognitionContext(collect_diagnostics=False)
        fast_ctx.force_identifier_recognition = True
        fast_ctx.fast_production_test = True
        fast_ctx.debug_deep = False
        result = self.omr_processor.recognize_sheet_production_fast(
            image_path,
            self.template,
            fast_ctx,
        )
        result.sync_legacy_aliases()
        if allow_retry:
            retried, improved = self._try_reprocess_result_rotated_180(result, template_path=path_text, source_tag=f"{source_tag}_retry180")
            if improved:
                result = retried
        result.sync_legacy_aliases()
        subject_key = self._current_batch_subject_key() or self._resolve_preferred_scoring_subject()
        is_api_flow = "api" in str(source_tag or "").lower()
        setattr(result, "answer_string_api_mode", bool(is_api_flow))
        scoped = self._scoped_result_copy(result)
        result.answer_string = self._build_answer_string_for_result(scoped, subject_key)
        blank_map = self._compute_blank_questions(scoped)
        expected_by_section = self._expected_questions_by_section(scoped)
        cached_status = ", ".join(self._status_parts_for_result(result, 1)) or "OK"
        setattr(result, "cached_status", cached_status)
        setattr(result, "cached_content", self._build_recognition_content_text(scoped, blank_map, expected_by_section))
        setattr(result, "cached_recognized_short", self._short_recognition_text_for_result(scoped))
        if source_tag:
            setattr(result, "cached_forced_status", str(source_tag))
        setattr(result, "manual_content_override", "")
        md = self.template.metadata if isinstance(self.template.metadata, dict) else {}
        setattr(result, "recognized_template_path", path_text)
        setattr(result, "recognized_alignment_profile", str(getattr(self.omr_processor, "alignment_profile", md.get("alignment_profile", "")) or ""))
        setattr(result, "recognized_fill_threshold", float(getattr(self.omr_processor, "fill_threshold", md.get("fill_threshold", 0.45)) or 0.45))
        setattr(result, "recognized_empty_threshold", float(getattr(self.omr_processor, "empty_threshold", md.get("empty_threshold", 0.20)) or 0.20))
        setattr(result, "recognized_certainty_margin", float(getattr(self.omr_processor, "certainty_margin", md.get("certainty_margin", 0.08)) or 0.08))
        return result

    def run_batch_scan(self, auto_triggered: bool = False) -> None:
        if self._batch_scan_running:
            confirm_cancel = QMessageBox.question(
                self,
                "Huỷ Batch Scan",
                "Batch Scan đang chạy. Bạn có muốn huỷ không?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm_cancel == QMessageBox.Yes:
                self._batch_cancel_requested = True
            return
        self._batch_cancel_requested = False

        subject_cfg = self._selected_batch_subject_config() or self._resolve_subject_config_for_batch()
        if subject_cfg:
            subject_cfg = self._merge_saved_batch_snapshot(subject_cfg)
        elif self.session:
            cfgs = self._effective_subject_configs_for_batch()
            if cfgs:
                subject_cfg = self._merge_saved_batch_snapshot(cfgs[0])
        if self.session and not subject_cfg:
            QMessageBox.warning(self, "Batch Scan", "Không có môn nào để nhận dạng trong kỳ thi hiện tại.")
            return

        if subject_cfg and self._subject_uses_direct_score_import(subject_cfg):
            QMessageBox.information(
                self,
                "Batch Scan",
                "Môn tự luận không đi qua nhận dạng OMR. Hãy import điểm trực tiếp theo SBD trong cấu hình môn, sau đó mở Tính điểm / Export.",
            )
            return

        if hasattr(self, "batch_recognition_mode_combo"):
            mode = str(self.batch_recognition_mode_combo.currentData() or "auto")
            setattr(self.omr_processor, "alignment_profile", mode)
        file_scope_mode = str(self.batch_file_scope_combo.currentData() or "new_only") if hasattr(self, "batch_file_scope_combo") else "new_only"
        subject_key_for_reset = self._subject_key_from_cfg(subject_cfg) if isinstance(subject_cfg, dict) else ""
        if file_scope_mode == "all" and subject_key_for_reset:
            has_scoring = bool(self.scoring_results_by_subject.get(subject_key_for_reset, {}))
            try:
                has_scoring = has_scoring or bool(self.database.fetch_scores_for_subject(subject_key_for_reset))
            except Exception:
                pass
            if has_scoring:
                confirm_reset = QMessageBox.question(
                    self,
                    "Nhận dạng toàn bộ",
                    "Môn này đã có điểm. Nhận dạng toàn bộ sẽ xoá toàn bộ điểm đã tính trước đó và chấm lại từ đầu.\n\nBạn có muốn tiếp tục không?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if confirm_reset != QMessageBox.Yes:
                    return
                self._persist_scoring_results_for_subject(
                    subject_key_for_reset,
                    [],
                    mode="reset_by_batch_scan_all",
                    note="reset_before_batch_scan_all",
                    mark_saved=False,
                )
                self.scoring_phases = [p for p in (self.scoring_phases or []) if str((p or {}).get("subject", "") or "").strip() != subject_key_for_reset]
                if self.session:
                    self.session_dirty = True
                    self._persist_session_quietly()
                if hasattr(self, "score_preview_table"):
                    self.score_preview_table.setRowCount(0)
                self._update_scoring_status_bar(0, 0, 0, 0)
        api_file = str(getattr(self, "batch_api_file_value", QLineEdit("-")).text() if hasattr(self, "batch_api_file_value") else "").strip()
        if api_file and api_file != "-":
            self._run_batch_scan_from_api_file(subject_cfg or {}, file_scope_mode, api_file)
            return

        # Resolve template, scan folder and answer keys from selected subject config in session.
        subject_template_path = ""
        exam_template_path = self._normalize_template_path(str(self.session.template_path if self.session else ""))
        scan_folder = ""
        answer_key_key = None
        if subject_cfg:
            subject_template_path = self._normalize_template_path(str(subject_cfg.get("template_path", "") or ""))
            scan_folder = str(subject_cfg.get("scan_folder", "") or "")
            answer_key_key = str(subject_cfg.get("answer_key_key", "") or "")
            if not scan_folder and self.session:
                scan_folder = str((self.session.config or {}).get("scan_root", "") or "")

            imported_answer_keys_map = subject_cfg.get("imported_answer_keys", {}) if isinstance(subject_cfg.get("imported_answer_keys", {}), dict) else {}
            if answer_key_key and imported_answer_keys_map:
                repo = AnswerKeyRepository()
                for exam_code, kd in imported_answer_keys_map.items():
                    repo.upsert(SubjectKey(
                        subject=answer_key_key,
                        exam_code=str(exam_code),
                        answers={int(k): str(v) for k, v in (kd.get("mcq_answers", {}) or {}).items()},
                        true_false_answers={int(k): dict(v) for k, v in (kd.get("true_false_answers", {}) or {}).items()},
                        numeric_answers={int(k): str(v) for k, v in (kd.get("numeric_answers", {}) or {}).items()},
                        full_credit_questions={
                            str(sec): [int(x) for x in (vals or []) if str(x).strip().lstrip("-").isdigit()]
                            for sec, vals in (kd.get("full_credit_questions", {}) or {}).items()
                        },
                        invalid_answer_rows={
                            str(sec): {int(k): str(v) for k, v in (vals or {}).items()}
                            for sec, vals in (kd.get("invalid_answer_rows", {}) or {}).items()
                        },
                    ))
                self.answer_keys = repo
                self.imported_exam_codes = sorted(str(k) for k in imported_answer_keys_map.keys())
                self.active_batch_subject_key = answer_key_key
            else:
                # Fallback: load answer keys from subject path or exam/session path if available.
                answer_key_path = str(subject_cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
                self.active_batch_subject_key = answer_key_key or self.active_batch_subject_key
                if answer_key_path:
                    pth = Path(answer_key_path)
                    if pth.exists() and pth.suffix.lower() == ".json":
                        try:
                            self.answer_keys = AnswerKeyRepository.load_json(pth)
                            all_codes: set[str] = set()
                            for key_name in self.answer_keys.keys.keys():
                                parts = str(key_name).split("::", 1)
                                if len(parts) == 2 and (not answer_key_key or parts[0] == answer_key_key):
                                    all_codes.add(parts[1])
                            self.imported_exam_codes = sorted(all_codes)
                        except Exception:
                            pass

        if not subject_template_path:
            subject_template_path = self._normalize_template_path(str(getattr(self, "batch_template_path_value", "") or ""))
        if not subject_template_path and hasattr(self, "batch_template_value"):
            subject_template_path = self._normalize_template_path(self.batch_template_value.text().strip())
        if (not scan_folder or scan_folder == "-") and hasattr(self, "batch_scan_folder_value"):
            candidate_folder = str(self.batch_scan_folder_value.text() or "").strip()
            if candidate_folder and candidate_folder != "-":
                scan_folder = candidate_folder

        template_path = subject_template_path or exam_template_path
        if not template_path:
            QMessageBox.warning(self, "Batch Scan", "Chưa cấu hình mẫu giấy để nhận dạng.")
            return
        template_file = Path(template_path)
        if not template_file.exists():
            QMessageBox.warning(self, "Batch Scan", f"Không tìm thấy mẫu giấy\n{template_path}")
            return
        try:
            self.template = Template.load_json(template_file)
        except Exception as exc:
            QMessageBox.warning(self, "Batch Scan", f"Không thể tải mẫu giấy\n{exc}")
            return
        self.template.metadata["recognition_timeout_sec"] = self._recommended_batch_timeout_sec(self.template)

        scan_folder = str(scan_folder or "").strip()
        if not scan_folder:
            QMessageBox.warning(self, "Batch Scan", "Chưa chọn thư mục quét.")
            return
        scan_dir = Path(scan_folder)
        if not scan_dir.exists() or not scan_dir.is_dir():
            QMessageBox.warning(self, "Batch Scan", f"Không tìm thấy thư mục quét\n{scan_folder}")
            return

        file_paths = self._configured_scan_file_paths(subject_cfg or {})

        if not file_paths:
            QMessageBox.warning(self, "Batch Scan", "Không tìm thấy ảnh bài thi trong thư mục quét.")
            return

        subject_key_for_results = self._subject_key_from_cfg(subject_cfg) if subject_cfg else self._resolve_preferred_scoring_subject()
        subject_db_key = self._batch_result_subject_key(subject_key_for_results)
        existing_results = list(self._refresh_scan_results_from_db(subject_key_for_results) or [])
        recognized_paths = {
            self._result_identity_key(str(getattr(item, "image_path", "") or ""))
            for item in existing_results
            if self._result_identity_key(str(getattr(item, "image_path", "") or ""))
        }
        if file_scope_mode == "new_only":
            file_paths = [path for path in file_paths if self._result_identity_key(str(path)) not in recognized_paths]
            if auto_triggered and subject_key_for_results:
                deleted_set = self._deleted_scan_images_for_subject(subject_key_for_results)
                if deleted_set:
                    file_paths = [path for path in file_paths if self._result_identity_key(str(path)) not in deleted_set]
            if not file_paths:
                self.scan_results = list(existing_results)
                self._populate_scan_grid_from_results(self.scan_results)
                self._update_batch_scan_scope_summary()
                QMessageBox.information(self, "Batch Scan", "Không còn file mới cần nhận dạng trong phạm vi đã cấu hình.")
                return

        self.scan_list.setRowCount(0)
        self.error_list.clear()
        self.result_preview.clear()
        self.scan_result_preview.setRowCount(0)
        self.manual_edit.clear()
        self.scan_image_preview.setText("Chọn bài thi ở danh sách bên trái")
        self.scan_image_preview.clear_markers()
        self.btn_save_batch_subject.setEnabled(False)
        self.scan_files = [Path(p) for p in file_paths]
        self.scan_blank_questions.clear()
        self.scan_blank_summary.clear()
        self.scan_manual_adjustments.clear()
        self.scan_edit_history.clear()
        self.scan_last_adjustment.clear()
        self.preview_rotation_by_index.clear()
        self.scan_forced_status_by_index.clear()

        self._apply_template_recognition_settings(self.template, sync_mode_selector=False)
        batch_started = time.perf_counter()
        batch_progress_dialog = self._open_batch_progress_screen(len(file_paths), title="Batch Scan - Đang nhận dạng")

        def on_progress(current: int, total: int, image_path: str):
            if self._batch_cancel_requested:
                raise RuntimeError("BATCH_CANCELLED")
            self._update_batch_progress_screen(batch_progress_dialog, current, total, image_path, batch_started)

        self._batch_scan_running = True
        try:
            if file_scope_mode == "all":
                self.database.delete_scan_results_for_subject(subject_db_key)
                self.scan_results = []
            else:
                self.scan_results = list(existing_results)
            duplicate_ids: dict[str, int] = {}
            for existing in self.scan_results:
                sid_existing = (existing.student_id or "").strip()
                if not self._student_id_has_recognition_error(sid_existing):
                    duplicate_ids[sid_existing] = duplicate_ids.get(sid_existing, 0) + 1

            base_count = len(self.scan_results)
            new_results: list[OMRResult] = []
            total = len(file_paths)
            for offset, image_path in enumerate(file_paths):
                on_progress(offset + 1, total, image_path)
                result = self._recognize_image_with_exact_template(
                    str(image_path),
                    template_path,
                    source_tag="batch_scan",
                    allow_retry=False,
                )
                new_results.append(result)
                idx = base_count + offset
                forced_status = ""
                original_meaningful = self._result_has_meaningful_recognition(result)
                original_identity = self._has_valid_identity(result)

                # Keep batch behavior aligned with Template Editor by default (no auto-rotation retry).
                skip_retry_on_poor = self._skip_retry_for_poor_images() and self._result_is_poor_image(result)
                need_retry_180 = self._allow_batch_auto_rotate_retry() and (not skip_retry_on_poor) and ((not original_identity) or (not original_meaningful))
                if need_retry_180:
                    retried, improved = self._try_reprocess_result_rotated_180(result, template_path=template_path, source_tag="batch_scan_retry180")
                    # Accept 180° retry only when quality is strictly improved, otherwise keep original orientation.
                    if improved:
                        result = retried
                        self.preview_rotation_by_index[idx] = (int(self.preview_rotation_by_index.get(idx, 0) or 0) + 180) % 360

                preferred_status = self._preferred_forced_status(result)
                if preferred_status:
                    # Keep raw recognition data (answers + identifiers) and only override status text.
                    forced_status = preferred_status
                elif self._should_force_image_error_status(result):
                    forced_status = "Lỗi file ảnh"

                if forced_status:
                    image_key_for_result = self._result_identity_key(getattr(result, "image_path", ""))
                    if image_key_for_result:
                        self.scan_forced_status_by_index[image_key_for_result] = forced_status

                self._refresh_student_profile_for_result(result)
                result.answer_string = self._build_answer_string_for_result(result, subject_key_for_results)
                self.scan_results.append(self._strip_transient_scan_artifacts(result))
                sid_for_dup = (result.student_id or "").strip()
                if not self._student_id_has_recognition_error(sid_for_dup):
                    duplicate_ids[sid_for_dup] = duplicate_ids.get(sid_for_dup, 0) + 1

                # Keep UI responsive while filling table after recognition reaches 100%.
                if offset % 10 == 0:
                    QApplication.processEvents()
        except RuntimeError as exc:
            if str(exc) != "BATCH_CANCELLED":
                raise
            QMessageBox.information(self, "Batch Scan", "Đã huỷ Batch Scan.")
        finally:
            self._batch_scan_running = False
            self._batch_cancel_requested = False
            self._close_batch_progress_screen(batch_progress_dialog)

        self.scan_results_by_subject[subject_db_key] = list(self.scan_results)
        self._unmark_deleted_scan_images(subject_key_for_results, [str(getattr(x, "image_path", "") or "") for x in new_results])
        self.database.replace_scan_results_for_subject(
            subject_db_key,
            [self._serialize_omr_result(x) for x in self.scan_results],
        )

        forced_status_by_image = {
            str(result.image_path or ""): str(self.scan_forced_status_by_index.get(str(result.image_path or ""), "") or "")
            for result in self.scan_results
        }
        self._populate_scan_grid_from_results(self.scan_results, forced_status_by_image)
        self._finalize_batch_scan_display()
        self.btn_save_batch_subject.setEnabled(False)
        self._update_batch_scan_scope_summary()
        elapsed_sec = max(0.0, float(time.perf_counter() - batch_started))
        total_items = len(file_paths)
        timing_text = f"{elapsed_sec:.1f}s/{total_items} bài"

    def _pick_batch_api_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file API bài thi",
            "",
            "Data files (*.csv *.txt *.tsv *.xlsx);;All files (*.*)",
        )
        if not path:
            return
        if hasattr(self, "batch_api_file_value"):
            self.batch_api_file_value.setText(path)

    @staticmethod
    def _run_batch_scan_from_api_file(self, subject_cfg: dict, file_scope_mode: str, api_file: str) -> None:
        return run_batch_scan_from_api_file(self, subject_cfg, file_scope_mode, api_file)
