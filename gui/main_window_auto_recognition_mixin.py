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


class MainWindowAutoRecognitionMixin:
    """Auto-recognition timers, folder polling, queue and pause state."""
    def _reset_auto_recognition_state(self, *, pause: bool = False) -> None:
        self._auto_recognition_pause_requested = bool(pause)
        self._auto_recognition_busy = False
        self._auto_recognition_active_subject = ""
        self._auto_recognition_queue.clear()
        self._auto_recognition_enqueued.clear()
        self._auto_recognition_last_seen.clear()
        self._update_auto_recognition_progress()

    def _setup_auto_recognition_timer(self) -> None:
        self.auto_recognition_discovery_timer = QTimer(self)
        self.auto_recognition_discovery_timer.setInterval(4000)
        self.auto_recognition_discovery_timer.timeout.connect(self._poll_auto_recognition)
        self.auto_recognition_discovery_timer.start()

        self.auto_recognition_worker_timer = QTimer(self)
        self.auto_recognition_worker_timer.setInterval(600)
        self.auto_recognition_worker_timer.timeout.connect(self._process_auto_recognition_queue)
        self.auto_recognition_worker_timer.start()

    @staticmethod
    def _is_subfolder_scan_mode(mode_text: str) -> bool:
        mode = str(mode_text or "").strip().lower()
        if not mode:
            return False
        return any(token in mode for token in ["thư mục con", "folder con", "sub", "phòng thi", "room"])

    def _scan_folder_signature(self, cfg: dict | None) -> tuple[int, float, int]:
        if not isinstance(cfg, dict):
            return (0, 0.0, 0)
        scan_folder = str(cfg.get("scan_folder", "") or ((self.session.config or {}).get("scan_root", "") if self.session else "") or "").strip()
        if not scan_folder or scan_folder == "-":
            return (0, 0.0, 0)
        scan_dir = Path(scan_folder)
        if not scan_dir.exists() or not scan_dir.is_dir():
            return (0, 0.0, 0)
        mode = str(cfg.get("scan_mode", "") or ((self.session.config or {}).get("scan_mode", "") if self.session else "") or "")
        use_subfolders = self._is_subfolder_scan_mode(mode)
        image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        count = 0
        latest_mtime = 0.0
        fingerprint = 0
        iterator = scan_dir.rglob("*") if use_subfolders else scan_dir.iterdir()
        for p in iterator:
            if not p.is_file() or p.suffix.lower() not in image_exts:
                continue
            count += 1
            try:
                stat = p.stat()
                latest_mtime = max(latest_mtime, float(stat.st_mtime))
                # Include path + size + mtime into a rolling fingerprint so
                # replacing a whole folder with same file count still triggers.
                item_sig = hash((str(p.relative_to(scan_dir)).lower(), int(stat.st_size), int(stat.st_mtime_ns)))
                fingerprint ^= int(item_sig)
            except Exception:
                continue
        return (count, latest_mtime, fingerprint)

    @staticmethod
    def _scan_signature_has_files(signature: tuple[int, float, int]) -> bool:
        return int((signature or (0, 0.0, 0))[0] or 0) > 0

    def _pending_auto_recognition_paths_for_cfg(self, cfg: dict) -> list[str]:
        if not isinstance(cfg, dict):
            return []
        subject_key = self._subject_instance_key_from_cfg(cfg)
        if not subject_key:
            return []
        scan_paths = self._configured_scan_file_paths(cfg)
        if not scan_paths:
            return []
        recognized = self._recognized_image_paths_for_subject(subject_key)
        deleted = self._deleted_scan_images_for_subject(subject_key)
        return [
            path
            for path in scan_paths
            if (path_key := self._result_identity_key(str(path)))
            and path_key not in recognized
            and path_key not in deleted
        ]

    def _enqueue_auto_recognition_subject(self, subject_key: str) -> None:
        key = str(subject_key or "").strip()
        if not key or key in self._auto_recognition_enqueued:
            return
        self._auto_recognition_queue.append(key)
        self._auto_recognition_enqueued.add(key)
        self._update_auto_recognition_progress()

    def _pop_auto_recognition_subject(self) -> str:
        if not self._auto_recognition_queue:
            return ""
        key = self._auto_recognition_queue.popleft()
        self._auto_recognition_enqueued.discard(key)
        self._update_auto_recognition_progress()
        return key

    def _poll_auto_recognition(self) -> None:
        if not self.session or self._auto_recognition_pause_requested:
            return
        for cfg in self._effective_subject_configs_for_batch():
            if not isinstance(cfg, dict):
                continue
            if not bool(cfg.get("auto_recognize", False)):
                continue
            if self._subject_uses_direct_score_import(cfg):
                continue
            subject_key = self._subject_instance_key_from_cfg(cfg)
            if not subject_key:
                continue
            new_signature = self._scan_folder_signature(cfg)
            old_signature = self._auto_recognition_last_seen.get(subject_key)
            self._auto_recognition_last_seen[subject_key] = new_signature
            if old_signature is None:
                # Baseline only: when auto mode is newly enabled (or app just opened),
                # do not immediately re-run recognition for an entire existing folder.
                # Auto recognition should react to subsequent file changes.
                continue
            if new_signature == old_signature:
                continue
            if self._pending_auto_recognition_paths_for_cfg(cfg):
                self._enqueue_auto_recognition_subject(subject_key)

    def _process_auto_recognition_queue(self) -> None:
        if self._auto_recognition_pause_requested or self._auto_recognition_busy or self._batch_scan_running:
            return
        if not self._auto_recognition_queue:
            self._auto_recognition_active_subject = ""
            self._update_auto_recognition_progress()
            return
        subject_key = self._pop_auto_recognition_subject()
        if not subject_key:
            return
        self._auto_recognition_active_subject = subject_key
        self._auto_recognition_busy = True
        try:
            self._run_auto_recognition_for_subject(subject_key)
        finally:
            self._auto_recognition_busy = False
            self._auto_recognition_active_subject = ""
            gc.collect()
            self._update_auto_recognition_progress()

    def _schedule_auto_recognition_for_existing_files(self, subject_cfgs: list[dict] | None = None) -> None:
        """Queue subjects with auto mode enabled even when files already existed before config save."""
        cfgs = subject_cfgs if isinstance(subject_cfgs, list) else self._effective_subject_configs_for_batch()
        for cfg in cfgs or []:
            if not isinstance(cfg, dict):
                continue
            if not bool(cfg.get("auto_recognize", False)):
                continue
            if self._subject_uses_direct_score_import(cfg):
                continue
            subject_key = self._subject_instance_key_from_cfg(cfg)
            if not subject_key:
                continue
            signature = self._scan_folder_signature(cfg)
            self._auto_recognition_last_seen[subject_key] = signature
            if self._scan_signature_has_files(signature) and self._pending_auto_recognition_paths_for_cfg(cfg):
                self._enqueue_auto_recognition_subject(subject_key)

    def _run_auto_recognition_for_subject(self, subject_key: str) -> None:
        if not hasattr(self, "batch_subject_combo") or self.batch_subject_combo.count() <= 1:
            return
        scope_prefix = str(self._session_scope_prefix() or "").strip()
        if scope_prefix and not str(subject_key or "").strip().startswith(f"{scope_prefix}::"):
            # Hard session isolation: never allow auto-recognition jobs from another exam/session.
            return
        previous_subject_index = self.batch_subject_combo.currentIndex()
        previous_scope_index = self.batch_file_scope_combo.currentIndex() if hasattr(self, "batch_file_scope_combo") else -1

        target_index = -1
        for i in range(1, self.batch_subject_combo.count()):
            if str(self.batch_subject_combo.itemData(i) or "").strip() == str(subject_key or "").strip():
                target_index = i
                break
        if target_index <= 0:
            return

        try:
            self.batch_subject_combo.setCurrentIndex(target_index)
            cfg = self._selected_batch_subject_config()
            if not isinstance(cfg, dict):
                return

            scan_paths = self._configured_scan_file_paths(cfg)
            if not scan_paths:
                return
            pending_auto_paths = self._pending_auto_recognition_paths_for_cfg(cfg)
            if not pending_auto_paths:
                return

            if hasattr(self, "batch_file_scope_combo"):
                for i in range(self.batch_file_scope_combo.count()):
                    if str(self.batch_file_scope_combo.itemData(i) or "") == "new_only":
                        self.batch_file_scope_combo.setCurrentIndex(i)
                        break
            self._update_auto_recognition_progress()
            self.run_batch_scan(auto_triggered=True)
        finally:
            if previous_scope_index >= 0 and hasattr(self, "batch_file_scope_combo"):
                self.batch_file_scope_combo.setCurrentIndex(previous_scope_index)
            if previous_subject_index >= 0 and previous_subject_index < self.batch_subject_combo.count():
                self.batch_subject_combo.setCurrentIndex(previous_subject_index)

    def _update_auto_recognition_progress(self) -> None:
        active_key = str(self._auto_recognition_active_subject or "").strip()
        queue_count = len(self._auto_recognition_queue)
        has_work = bool(active_key or queue_count > 0)
        if not has_work:
            if hasattr(self, "auto_recognition_progress_dialog") and self.auto_recognition_progress_dialog is not None:
                self.auto_recognition_progress_dialog.hide()
            return

        if not hasattr(self, "auto_recognition_progress_dialog") or self.auto_recognition_progress_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Tiến trình nhận dạng tự động")
            dlg.setModal(False)
            lay = QVBoxLayout(dlg)
            self.auto_recognition_progress_label = QLabel("Đang khởi tạo cơ chế nhận dạng tự động...")
            self.auto_recognition_progress_label.setWordWrap(True)
            self.auto_recognition_progress_bar = QProgressBar()
            self.auto_recognition_progress_bar.setRange(0, 0)
            self.auto_recognition_progress_bar.setTextVisible(False)
            btn_pause = QPushButton("Tạm dừng")
            btn_pause.clicked.connect(self._toggle_auto_recognition_pause)
            lay.addWidget(self.auto_recognition_progress_label)
            lay.addWidget(self.auto_recognition_progress_bar)
            lay.addWidget(btn_pause)
            self.auto_recognition_progress_dialog = dlg
        else:
            dlg = self.auto_recognition_progress_dialog

        active_label = active_key or "-"
        paused = " (Tạm dừng)" if self._auto_recognition_pause_requested else ""
        text = (
            f"Cơ chế: Nhận dạng tự động{paused}\n"
            f"Đang xử lý: {active_label}\n"
            f"Hàng đợi còn: {queue_count} môn"
        )
        self.auto_recognition_progress_label.setText(text)
        if not dlg.isVisible():
            dlg.show()
            dlg.raise_()

    def _toggle_auto_recognition_pause(self) -> None:
        self._auto_recognition_pause_requested = not self._auto_recognition_pause_requested
        self._update_auto_recognition_progress()
