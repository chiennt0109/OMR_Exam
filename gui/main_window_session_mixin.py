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


class MainWindowSessionMixin:
    """Session persistence, dirty-state, save/open/close, registry and app close handling."""
    def _confirm(self, title: str, message: str) -> bool:
        return (
            QMessageBox.question(
                self,
                title,
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            == QMessageBox.Yes
        )

    def _canonicalize_subject_configs_for_session(self, subject_cfgs: list | None) -> list:
        """Return light subject configs with stable per-session storage keys.

        The subject table status, Batch Scan storage key and Scoring lookup must all
        use this single canonical key.  Without normalising the key here, a runtime
        copy may create a temporary key during Batch Scan; the editor then cannot
        find the saved DB rows until the whole session is reloaded.
        """
        normalized: list = []
        if not isinstance(subject_cfgs, list):
            return normalized
        heavy_keys = {
            "batch_saved",
            "batch_saved_at",
            "batch_result_count",
            "batch_saved_rows",
            "batch_saved_preview",
            "batch_saved_results",
            "scoring_rows",
            "scoring_results",
            "scoring_preview_rows",
            "scoring_phases",
        }
        for idx, item in enumerate(subject_cfgs):
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            cfg_item = dict(item)
            for key in heavy_keys:
                cfg_item.pop(key, None)
            self._ensure_subject_instance_key(cfg_item, idx)
            normalized.append(cfg_item)
        return normalized

    def _build_session_persistence_payload(self) -> dict:
        if not self.session:
            return {}
        current_payload = self.session.to_dict()
        current_cfg = dict(current_payload.get("config", {}) or {})

        # DB là nguồn dữ liệu nặng duy nhất. Session config chỉ giữ metadata nhẹ
        # để tránh lưu chồng chéo snapshot Batch/Scoring làm Save/Close chậm dần
        # theo số lượng bài đã nhận dạng.
        current_cfg.pop("scoring_phases", None)
        current_cfg.pop("scoring_results", None)
        subject_cfgs = current_cfg.get("subject_configs", [])
        if isinstance(subject_cfgs, list):
            current_cfg["subject_configs"] = self._canonicalize_subject_configs_for_session(subject_cfgs)

        current_payload["config"] = current_cfg
        return current_payload

    def _session_payload_signature(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    def _remember_current_session_snapshot(self) -> None:
        if not self.session:
            self._session_saved_signature = ""
            return
        try:
            self._session_saved_signature = self._session_payload_signature(self._build_session_persistence_payload())
        except Exception:
            self._session_saved_signature = ""

    def _has_pending_unsaved_work(self) -> bool:
        if self._has_batch_unsaved_changes():
            return True
        if self._has_scoring_unsaved_changes():
            return True
        if self.stack.currentIndex() == 5 and self._embedded_exam_has_real_changes():
            return True
        # Không dùng cờ session_dirty đơn thuần để hỏi lưu, vì nhiều thao tác runtime
        # đã được auto-persist. Chỉ hỏi khi payload thực sự khác snapshot đã lưu.
        return self._session_has_real_changes()

    def _embedded_exam_has_real_changes(self) -> bool:
        if not self.embedded_exam_dialog:
            return False
        try:
            current_payload = self.embedded_exam_dialog.payload()
        except Exception:
            return False
        return self._payload_changed(current_payload, self.embedded_exam_original_payload)

    def _session_has_real_changes(self) -> bool:
        if not self.session:
            return False

        if not self.current_session_id:
            return bool(getattr(self, "session_dirty", False))

        current_payload = self._build_session_persistence_payload()
        current_signature = self._session_payload_signature(current_payload)
        saved_signature = str(getattr(self, "_session_saved_signature", "") or "")
        if saved_signature:
            return current_signature != saved_signature

        saved_payload = self.database.fetch_exam_session(self.current_session_id)
        if not isinstance(saved_payload, dict):
            return bool(getattr(self, "session_dirty", False))

        return current_signature != self._session_payload_signature(saved_payload)

    def _confirm_before_switching_work(self, target_text: str) -> bool:
        if not self._has_pending_unsaved_work():
            return True
        return self._handle_pending_changes_before_switch(target_text)

    def _has_active_workflows(self) -> bool:
        return bool(
            self._batch_scan_running
            or self._auto_recognition_busy
            or self._auto_recognition_queue
            or self._auto_recognition_active_subject
        )

    def _interrupt_active_workflows(self) -> None:
        self._batch_cancel_requested = True
        self._auto_recognition_pause_requested = True
        self._auto_recognition_queue.clear()
        self._auto_recognition_enqueued.clear()
        self._auto_recognition_active_subject = ""
        self._update_auto_recognition_progress()

    def _confirm_interrupt_active_workflows(self, destination_text: str) -> bool:
        if not self._has_active_workflows():
            return True
        answer = QMessageBox.question(
            self,
            "Xác nhận chuyển màn hình",
            (
                f"Chuyển sang \"{destination_text}\" có thể ảnh hưởng đến luồng đang chạy.\n"
                "Bạn có đồng ý ngắt tất cả tiến trình nhận dạng đang chạy không?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False
        self._interrupt_active_workflows()
        return True

    def _prompt_save_changes_word_style(self, title: str, message: str) -> str:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setInformativeText("Bạn muốn lưu thay đổi trước khi tiếp tục không?")
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Save)
        choice = msg.exec()
        if choice == QMessageBox.Save:
            return "save"
        if choice == QMessageBox.Discard:
            return "discard"
        return "cancel"

    def _save_current_work(self) -> bool:
        saved_any = False

        if self._has_batch_unsaved_changes():
            if not self._save_batch_for_selected_subject(
                show_success_message=False,
                reload_after_save=False,
                refresh_exam_list=False,
            ):
                return False
            saved_any = True

        if self._has_scoring_unsaved_changes():
            subject_key = self._resolve_preferred_scoring_subject()
            if not subject_key:
                return False
            rows = self._collect_scoring_preview_rows(subject_key)
            cfg = self._subject_config_by_subject_key(subject_key) or {}
            mode = str((cfg or {}).get("scoring_last_mode", "Tính lại toàn bộ") or "Tính lại toàn bộ")
            note = str((cfg or {}).get("scoring_phase_last_note", "") or "")
            self._persist_scoring_results_for_subject(subject_key, rows, mode, note, mark_saved=True)
            saved_any = True

        if self._session_has_real_changes() or bool(getattr(self, "session_dirty", False)) or saved_any:
            return self._persist_runtime_session_state_quietly() if self.current_session_id else self._persist_session_quietly()
        return True

    def _handle_pending_changes_before_switch(self, target_text: str) -> bool:
        if not self._has_pending_unsaved_work():
            return True
        choice = self._prompt_save_changes_word_style(
            "Dữ liệu chưa lưu",
            f"Bạn đang có dữ liệu chưa lưu. Trước khi chuyển sang {target_text}, bạn có muốn lưu không?",
        )
        if choice == "cancel":
            return False
        if choice == "discard":
            return True
        return self._save_current_work()

    def _session_storage_dir(self) -> Path:
        d = Path.home() / ".omr_exam" / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _generate_session_id(self, seed: str = "") -> str:
        raw = f"{seed}-{datetime.now().isoformat()}"
        return str(abs(hash(raw)))

    def _session_path_from_id(self, session_id: str) -> Path:
        return self._session_storage_dir() / f"{session_id}.json"

    def _load_session_registry(self) -> list[dict[str, str | bool]]:
        try:
            rows = self.database.list_exam_sessions()
            return [dict(x) for x in rows]
        except Exception:
            return []

    def _save_session_registry(self) -> None:
        self.session_registry = self._load_session_registry()

    def _upsert_session_registry(self, session_id: str, name: str | None = None) -> None:
        payload = self.database.fetch_exam_session(session_id) or {}
        exam_name = str(name or payload.get("exam_name") or "Kỳ thi")
        if payload:
            self.database.save_exam_session(session_id, exam_name, payload)
        self.session_registry = self._load_session_registry()

    def _session_name_exists(self, exam_name: str, exclude_session_id: str = "") -> bool:
        name_norm = str(exam_name or "").strip().casefold()
        if not name_norm:
            return False
        for row in self.session_registry:
            sid = str(row.get("session_id", "") or "")
            if exclude_session_id and sid == exclude_session_id:
                continue
            if str(row.get("name", "") or "").strip().casefold() == name_norm:
                return True
        return False

    def _open_session_path(self, path: Path) -> None:
        try:
            self._release_batch_runtime_state()
            payload = self.database.fetch_exam_session(path.stem)
            if not payload:
                raise FileNotFoundError(f"Không tìm thấy session '{path.stem}' trong SQLite.")
            self.session = ExamSession.from_dict(payload)
            self.current_session_path = path
            self.current_session_id = path.stem
            if self.session.template_path:
                t = Path(self.session.template_path)
                if t.exists():
                    self.template = Template.load_json(t)
            cfg = dict(self.session.config or {})
            cfg.pop("scoring_phases", None)
            cfg.pop("scoring_results", None)
            self.session.config = cfg
            self.scoring_phases = []
            self.scoring_results_by_subject = {}
            self._scoring_dirty_subjects = set()
            # DB-only mode: không phục hồi Batch Scan từ subject config.
            # scan_results_by_subject chỉ được nạp lại khi người dùng mở môn, bằng truy vấn DB.
            self.scan_results_by_subject = {}
            self.batch_working_state_by_subject = {}
            self.subject_catalog = list(cfg.get("subject_catalog", self.subject_catalog)) or self.subject_catalog
            self.block_catalog = list(cfg.get("block_catalog", self.block_catalog)) or self.block_catalog
            if self.session.answer_key_path:
                p = Path(self.session.answer_key_path)
                if p.exists() and p.suffix.lower() == ".json":
                    self.answer_keys = AnswerKeyRepository.load_json(p)
                    self.imported_exam_codes = sorted({k.split("::", 1)[1] for k in self.answer_keys.keys.keys() if "::" in k})
            self._upsert_session_registry(path.stem, self.session.exam_name if self.session else None)
            self._save_session_registry()
            self._refresh_exam_list()
            self.session_dirty = False
            self._remember_current_session_snapshot()
            self.batch_editor_return_payload = None
            self.batch_editor_return_session_id = None
            self._refresh_session_info()
            self._refresh_batch_subject_controls()
            self._refresh_scoring_phase_table()
            cfg = self.session.config or {}
            editor_payload = {
                "exam_name": self.session.exam_name,
                "common_template": self.session.template_path,
                "scan_root": cfg.get("scan_root", ""),
                "student_list_path": cfg.get("student_list_path", ""),
                "students": [
                    {
                        "student_id": s.student_id,
                        "name": s.name,
                        "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                        "class_name": str((s.extra or {}).get("class_name", "") or ""),
                        "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
                    }
                    for s in (self.session.students or [])
                ],
                "scan_mode": cfg.get("scan_mode", "Ảnh trong thư mục gốc"),
                "paper_part_count": cfg.get("paper_part_count", 3),
                "subject_configs": cfg.get("subject_configs", []),
            }
            self._open_embedded_exam_editor(path.stem, self.session, editor_payload)
            self._refresh_ribbon_action_states()
            QMessageBox.information(self, "Open session", "Đã mở kỳ thi thành công.")
        except Exception as exc:
            QMessageBox.warning(self, "Open session", f"Không thể mở kỳ thi:\n{exc}")

    def open_session(self) -> None:
        if self.session and self.session_dirty:
            if not self._confirm("Dữ liệu chưa lưu", "Kỳ thi hiện tại có thay đổi chưa lưu. Vẫn mở kỳ thi khác?"):
                return
        row = self.exam_list_table.currentRow()
        if row >= 0:
            path = self._selected_registry_path()
            if path:
                self._open_session_path(path)
                return
        default_rows = [x for x in self.session_registry if bool(x.get("default"))]
        if default_rows:
            sid = str(default_rows[0].get("session_id", ""))
            if sid:
                self._open_session_path(self._session_path_from_id(sid))
                return
        QMessageBox.information(self, "Mở kỳ thi", "Vui lòng chọn kỳ thi trong danh sách để mở.")

    def save_session(self) -> None:
        if not self.session:
            self.create_session()
        if not self.current_session_id:
            self.current_session_id = self._generate_session_id(self.session.exam_name if self.session else "exam")
            self.current_session_path = self._session_path_from_id(self.current_session_id)
        try:
            if self.session:
                payload = self._build_session_persistence_payload()
                self.session.config = dict(payload.get("config", {}) or {})
                self.database.save_exam_session(self.current_session_id, self.session.exam_name, payload)
            self._upsert_session_registry(self.current_session_id, self.session.exam_name if self.session else None)
            self._save_session_registry()
            self._refresh_exam_list()
            self.session_dirty = False
            self._remember_current_session_snapshot()
            QMessageBox.information(self, "Save session", "Đã lưu kỳ thi vào kho hệ thống.")
        except Exception as exc:
            QMessageBox.warning(self, "Save session", f"Không thể lưu kỳ thi:\n{exc}")

    def _persist_session_quietly(self) -> bool:
        if not self.session:
            return False
        if not self.current_session_id:
            self.current_session_id = self._generate_session_id(self.session.exam_name if self.session else "exam")
            self.current_session_path = self._session_path_from_id(self.current_session_id)
        try:
            payload = self._build_session_persistence_payload()
            self.session.config = dict(payload.get("config", {}) or {})
            self.database.save_exam_session(self.current_session_id, self.session.exam_name, payload)
            self._upsert_session_registry(self.current_session_id, self.session.exam_name if self.session else None)
            self._save_session_registry()
            self._refresh_exam_list()
            self.session_dirty = False
            self._remember_current_session_snapshot()
            return True
        except Exception:
            self.session_dirty = True
            return False

    def save_session_as(self) -> None:
        if self.stack.currentIndex() != 0:
            QMessageBox.information(self, "Lưu dưới tên khác", "Chức năng này chỉ sử dụng trong màn hình danh sách kỳ thi.")
            return
        row = self.exam_list_table.currentRow() if hasattr(self, "exam_list_table") else -1
        source_session_id = self._session_id_for_row(row) if row >= 0 else ""
        if not source_session_id:
            QMessageBox.information(self, "Lưu dưới tên khác", "Vui lòng chọn kỳ thi nguồn trong danh sách.")
            return
        source_payload = self.database.fetch_exam_session(source_session_id)
        if not isinstance(source_payload, dict) or not source_payload:
            QMessageBox.warning(self, "Lưu dưới tên khác", "Không đọc được dữ liệu kỳ thi nguồn.")
            return

        # 1) Chỉ nhập và kiểm tra tên mới trong while; không làm gì khác.
        current_name = str(source_payload.get("exam_name", "") or "").strip() or "Kỳ thi"
        while True:
            new_name, ok = QInputDialog.getText(self, "Lưu dưới tên khác", "Nhập tên kỳ thi mới:", text=current_name)
            if not ok:
                return
            new_name = str(new_name or "").strip()
            if not new_name:
                QMessageBox.warning(self, "Lưu dưới tên khác", "Tên kỳ thi không được để trống.")
                continue
            if self._session_name_exists(new_name):
                QMessageBox.warning(self, "Lưu dưới tên khác", "Tên kỳ thi đã tồn tại. Vui lòng chọn tên khác.")
                continue
            break

        # 2) Chỉ sau khi tên hợp lệ mới thực hiện copy kỳ thi.
        try:
            new_session_id = self._generate_session_id(new_name)
            source_exam_name = str(source_payload.get("exam_name", "") or "").strip().lower()
            target_exam_name = str(new_name or "").strip().lower()
            source_prefix = f"{source_session_id}::{source_exam_name}" if source_session_id and source_exam_name else source_session_id
            target_prefix = f"{new_session_id}::{target_exam_name}" if new_session_id and target_exam_name else new_session_id

            payload = copy.deepcopy(source_payload)
            payload["exam_name"] = new_name
            cfg_root = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
            payload["config"] = cfg_root
            subject_cfgs = list(cfg_root.get("subject_configs", []) if isinstance(cfg_root.get("subject_configs", []), list) else [])

            subject_key_map: dict[str, str] = {}
            for subject_cfg in subject_cfgs:
                if not isinstance(subject_cfg, dict):
                    continue
                old_key = str(subject_cfg.get("subject_instance_key", "") or "").strip()
                logical = str(subject_cfg.get("logical_subject_key", "") or "").strip() or str(self._logical_subject_key_from_cfg(subject_cfg) or "General")
                new_uid = str(uuid.uuid4())
                new_key = f"{target_prefix}::{logical}::{new_uid}" if target_prefix else f"{logical}::{new_uid}"
                subject_cfg["subject_uid"] = new_uid
                subject_cfg["logical_subject_key"] = logical
                subject_cfg["subject_instance_key"] = new_key
                if old_key:
                    subject_key_map[old_key] = new_key

            self.database.save_exam_session(new_session_id, new_name, payload)

            for old_subject_key, new_subject_key in subject_key_map.items():
                source_scan_candidates = [old_subject_key]
                if source_prefix and not old_subject_key.startswith(f"{source_prefix}::"):
                    source_scan_candidates.append(f"{source_prefix}::{old_subject_key}")
                elif source_prefix:
                    source_scan_candidates.append(f"{source_prefix}::{old_subject_key}")

                source_rows = []
                for source_scan_key in dict.fromkeys(source_scan_candidates):
                    source_rows = list(self.database.fetch_scan_results_for_subject(source_scan_key) or [])
                    if source_rows:
                        break

                target_scan_key = new_subject_key if not target_prefix or new_subject_key.startswith(f"{target_prefix}::") else f"{target_prefix}::{new_subject_key}"
                self.database.replace_scan_results_for_subject(target_scan_key, source_rows)

                source_keys = self.database.fetch_answer_keys_for_subject(old_subject_key)
                self.database.replace_answer_keys_for_subject(new_subject_key, source_keys)

                source_scores = list(self.database.fetch_scores_for_subject(old_subject_key) or [])
                self.database.conn.execute("DELETE FROM scores WHERE subject_key = ?", (new_subject_key,))
                for score_row in source_scores:
                    self.database.upsert_score_row(
                        new_subject_key,
                        str((score_row or {}).get("student_id", "") or ""),
                        str((score_row or {}).get("exam_code", "") or ""),
                        dict(score_row or {}),
                    )

            source_histories = list(self.database.fetch_recheck_history(source_session_id) or [])
            for item in source_histories:
                old_hist_subject = str(item.get("subject_key", "") or "")
                new_hist_subject = subject_key_map.get(old_hist_subject, old_hist_subject)
                self.database.add_recheck_history(
                    session_id=new_session_id,
                    exam_name=new_name,
                    subject_key=new_hist_subject,
                    student_code=str(item.get("student_code", "") or ""),
                    exam_code=str(item.get("exam_code", "") or ""),
                    change_text=str(item.get("change_text", "") or ""),
                    old_score=float(item.get("old_score", 0.0) or 0.0),
                    new_score=float(item.get("new_score", 0.0) or 0.0),
                    payload=dict(item.get("payload", {}) or {}),
                )

            self._upsert_session_registry(new_session_id, new_name)
            self._save_session_registry()
            self._refresh_exam_list()
            QMessageBox.information(self, "Lưu dưới tên khác", f"Đã sao chép kỳ thi thành '{new_name}'.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu dưới tên khác", f"Không thể sao chép kỳ thi:\n{exc}")

    def close_current_session(self) -> None:
        self._release_batch_runtime_state()
        self._release_preview_resources()
        self._release_template_cache()
        self._release_editor_resources()
        self.session = None
        self.template = None
        self.answer_keys = None
        self.scan_results = []
        self.scan_results_by_subject = {}
        self.batch_working_state_by_subject = {}
        self.scoring_phases = []
        self.scoring_results_by_subject = {}
        self._scoring_dirty_subjects = set()
        self.scan_results_by_subject = {}
        self.current_session_path = None
        self.current_session_id = None
        self.session_dirty = False
        self._session_saved_signature = ""
        self.session_info.clear()
        self.exam_code_preview.setText("Mã đề trên phiếu trả lời mẫu: -")
        self.scan_list.setRowCount(0)
        self.score_preview_table.setRowCount(0)
        self.error_list.clear()
        self.result_preview.clear()
        self.scan_result_preview.setRowCount(0)
        self.manual_edit.clear()
        self._refresh_batch_subject_controls()
        self.stack.setCurrentIndex(0)

    def _release_batch_runtime_state(self) -> None:
        # resource cleanup / release: aggressively clear heavy per-subject runtime maps.
        self.scan_results = []
        self.scan_results_by_subject = {}
        self.batch_working_state_by_subject = {}
        self.scoring_results_by_subject = {}
        self.scoring_phases = []
        self._scoring_dirty_subjects = set()
        self.scan_files = []
        self.score_rows = []
        self.imported_exam_codes = []
        self.preview_rotation_by_index = {}
        self.scan_forced_status_by_index = {}
        self.deleted_scan_images_by_subject = {}
        self.scan_blank_questions = {}
        self.scan_blank_summary = {}
        self.scan_manual_adjustments = {}
        self.scan_edit_history = {}
        self.scan_last_adjustment = {}
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None
        self.active_batch_subject_key = None
        self._batch_loaded_runtime_key = ""
        self._batch_loaded_subject_signature = ""
        self._current_batch_data_source = "empty"
        self.batch_status_filter_mode = "all"
        self.scoring_status_filter_mode = "all"

    def _release_preview_resources(self) -> None:
        if hasattr(self, "scan_image_preview"):
            self.scan_image_preview.clear()
            if hasattr(self.scan_image_preview, "clear_markers"):
                self.scan_image_preview.clear_markers()
        if hasattr(self, "result_preview"):
            self.result_preview.clear()
        if hasattr(self, "preview_source_pixmap"):
            self.preview_source_pixmap = None

    def _release_editor_resources(self) -> None:
        for attr in ["template_editor_embedded", "embedded_exam_dialog"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.deleteLater()
                except Exception:
                    pass
                setattr(self, attr, None)
        for attr in ["embedded_exam_session", "embedded_exam_original_payload"]:
            if hasattr(self, attr):
                setattr(self, attr, None)

    def action_create_session(self) -> None:
        if not self._confirm_before_switching_work("kỳ thi mới"):
            return
        session_id = self._generate_session_id("new_exam")
        payload = {
            "exam_name": "",
            "common_template": "",
            "scan_root": "",
            "student_list_path": "",
            "students": [],
            "scan_mode": "Ảnh trong thư mục gốc",
            "paper_part_count": 3,
            "subject_configs": [],
        }
        draft = ExamSession(
            exam_name="Kỳ thi mới",
            exam_date=str(date.today()),
            subjects=[],
            template_path="",
            answer_key_path="",
            students=[],
            config={
                "scan_mode": "Ảnh trong thư mục gốc",
                "scan_root": "",
                "student_list_path": "",
                "paper_part_count": 3,
                "subject_configs": [],
                "subject_catalog": self.subject_catalog,
                "block_catalog": self.block_catalog,
            },
        )
        self._open_embedded_exam_editor(session_id, draft, payload, is_new=True)

    def action_open_session(self) -> None:
        if not self._confirm_interrupt_active_workflows("Danh sách kỳ thi"):
            return
        self._navigate_to("exam_list", context={}, push_current=True, require_confirm=False, reason="open_exam_list")

    def action_save_session(self) -> None:
        if self.stack.currentIndex() == 5 and self.embedded_exam_dialog:
            self._save_embedded_exam_editor()
            return
        if self.stack.currentIndex() == 4 and self.template_editor_embedded:
            self._save_current_template()
            return
        self.save_session()

    def action_save_session_as(self) -> None:
        self.save_session_as()

    def action_close_current_session(self) -> None:
        if not self._confirm_before_switching_work("đóng kỳ thi hiện tại"):
            return
        self.close_current_session()

    def create_session(self, payload: dict | None = None) -> None:
        payload = payload or {}
        self._register_templates_from_payload(payload)
        exam_name = str(payload.get("exam_name", "Untitled Exam"))
        common_template = str(payload.get("common_template", ""))
        subject_cfgs = payload.get("subject_configs", [])
        subjects = [
            f"{str(x.get('name', '')).strip()}_{str(x.get('block', '')).strip()}"
            for x in subject_cfgs
            if str(x.get("name", "")).strip()
        ]
        if not subjects:
            subjects = ["General"]

        self.session = ExamSession(
            exam_name=exam_name,
            exam_date=str(date.today()),
            subjects=subjects,
            template_path=common_template,
            answer_key_path="",
            students=self._students_from_editor_rows(
                payload.get("students", []) if isinstance(payload.get("students", []), list) else []
            ),
            config={
                "scan_mode": payload.get("scan_mode", "Ảnh trong thư mục gốc"),
                "scan_root": payload.get("scan_root", ""),
                "student_list_path": payload.get("student_list_path", ""),
                "paper_part_count": payload.get("paper_part_count", 3),
                "subject_configs": subject_cfgs,
                "subject_catalog": self.subject_catalog,
                "block_catalog": self.block_catalog,
            },
        )
        self.scoring_phases = []
        self.scoring_results_by_subject = {}
        self._scoring_dirty_subjects = set()
        self.batch_working_state_by_subject = {}
        if common_template and Path(common_template).exists():
            try:
                self.template = Template.load_json(common_template)
            except Exception:
                self.template = None
        self.current_session_path = None
        self.current_session_id = None
        self._session_saved_signature = ""
        self.session_dirty = True
        self._refresh_session_info()
        self._refresh_batch_subject_controls()

    def _refresh_session_info(self) -> None:
        if not self.session:
            return
        cfg = self.session.config or {}
        self.session_info.setPlainText(
            f"Exam: {self.session.exam_name}\nDate: {self.session.exam_date}\n"
            f"Subjects: {', '.join(self.session.subjects)}\n"
            f"Template: {self.session.template_path}\n"
            f"AnswerKey: {self.session.answer_key_path}\n"
            f"Scan mode: {cfg.get('scan_mode', '-')}\n"
            f"Scan root: {cfg.get('scan_root', '-')}\n"
            f"Paper parts: {cfg.get('paper_part_count', '-')}\n"
            f"Students: {len(self.session.students)}"
        )
        codes = ", ".join(self.imported_exam_codes) if self.imported_exam_codes else "-"
        self.exam_code_preview.setText(f"Mã đề trên phiếu trả lời mẫu: {codes}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._has_pending_unsaved_work():
            choice = self._prompt_save_changes_word_style(
                "Thoát ứng dụng",
                "Bạn muốn lưu thay đổi trước khi thoát không?",
            )
            if choice == "cancel":
                event.ignore()
                return
            if choice == "save" and not self._save_current_work():
                event.ignore()
                return
        event.accept()
