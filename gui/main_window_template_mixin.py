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


class MainWindowTemplateMixin:
    """Template repository page, embedded template editor and template save/register helpers."""
    def _build_template_management_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        split = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.template_library_table = QTableWidget(0, 2)
        self.template_library_table.setHorizontalHeaderLabels(["STT", "Tên mẫu"])
        self.template_library_table.verticalHeader().setVisible(False)
        self.template_library_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.template_library_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.template_library_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.template_library_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.template_library_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.template_library_table.itemSelectionChanged.connect(self._handle_template_library_selection)
        # double click navigation: open selected template directly.
        self.template_library_table.cellDoubleClicked.connect(self._handle_template_library_double_click)
        left_layout.addWidget(self.template_library_table)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.template_preview_title = QLabel("Chưa chọn mẫu giấy thi")
        self.template_preview_title.setWordWrap(True)
        self.template_preview_title.setContentsMargins(0, 0, 0, 4)
        self.template_preview_image = QLabel("Chọn mẫu giấy thi ở danh sách bên trái")
        self.template_preview_image.setAlignment(Qt.AlignCenter)
        self.template_preview_image.setMinimumHeight(420)
        self.template_preview_image.setStyleSheet("border: 1px solid #cfcfcf; background: #fafafa;")
        right_layout.addWidget(self.template_preview_title)
        right_layout.addWidget(self.template_preview_image, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)
        return w

    def _refresh_template_library(self) -> None:
        rows = self.template_repo.list_templates()
        self.template_library_table.setRowCount(len(rows))
        selected_row = 0 if rows else -1
        for idx, (name, path) in enumerate(rows):
            num_item = QTableWidgetItem(str(idx + 1))
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, path)
            self.template_library_table.setItem(idx, 0, num_item)
            self.template_library_table.setItem(idx, 1, name_item)
        if rows:
            self.template_library_table.selectRow(selected_row)
            self._update_template_preview_by_row(selected_row)
        else:
            self.template_preview_title.setText("Kho mẫu giấy thi đang trống")
            self.template_preview_image.setPixmap(QPixmap())
            self.template_preview_image.setText("Chưa có mẫu giấy thi trong kho")

    def _handle_template_library_selection(self) -> None:
        row = self.template_library_table.currentRow()
        self._update_template_preview_by_row(row)

    def _update_template_preview_by_row(self, row: int) -> None:
        if row < 0:
            self.template_preview_title.setText("Chưa chọn mẫu giấy thi")
            self.template_preview_image.setPixmap(QPixmap())
            self.template_preview_image.setText("Chọn mẫu giấy thi ở danh sách bên trái")
            return
        item = self.template_library_table.item(row, 1)
        template_path = str(item.data(Qt.UserRole) if item else "")
        if not template_path:
            return
        try:
            tpl = Template.load_json(template_path)
            img_path = Path(tpl.image_path)
            if not img_path.is_absolute():
                img_path = (Path(template_path).parent / img_path).resolve()
            pix = QPixmap(str(img_path))
            self.template_preview_title.setText(f"{row + 1}. {tpl.name}")
            if pix.isNull():
                self.template_preview_image.setPixmap(QPixmap())
                self.template_preview_image.setText("Không thể tải ảnh mẫu giấy thi")
                return
            scaled = pix.scaled(self.template_preview_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.template_preview_image.setPixmap(scaled)
            self.template_preview_image.setText("")
        except Exception:
            self.template_preview_title.setText(f"{row + 1}. Không thể đọc mẫu giấy thi")
            self.template_preview_image.setPixmap(QPixmap())
            self.template_preview_image.setText("Không thể đọc dữ liệu mẫu giấy thi")

    def _load_template_repository(self) -> TemplateRepository:
        payload = self.database.get_app_state("template_repository", {})
        if isinstance(payload, dict):
            try:
                return TemplateRepository.from_dict(payload)
            except Exception:
                pass
        return TemplateRepository()

    def _handle_template_library_double_click(self, row: int, col: int) -> None:
        # double click navigation: route row double click to template edit action.
        if row < 0:
            return
        self.template_library_table.selectRow(row)
        self._edit_selected_template()

    def _release_template_cache(self) -> None:
        if hasattr(self, "_template_cache_by_path"):
            self._template_cache_by_path = {}

    def action_manage_template(self) -> None:
        self.open_template_editor()

    def _rebuild_template_module_menu(self, *, library_mode: bool, editor_mode: bool) -> None:
        if not hasattr(self, "template_module_menu"):
            return
        self.template_module_menu.clear()
        if editor_mode and self.template_editor_embedded:
            editor = self.template_editor_embedded
            self.template_module_menu.addAction(editor.act_load_blank)
            self.template_module_menu.addAction(editor.act_open_template)
            self.template_module_menu.addSeparator()
            self.template_module_menu.addAction(editor.act_save)
            self.template_module_menu.addAction(editor.act_save_as)
            self.template_module_menu.addSeparator()
            self.template_module_menu.addAction(editor.act_preview)
            self.template_module_menu.addAction(editor.act_test_recognition)
            self.template_module_menu.addAction(editor.act_template_qc)
            self.template_module_menu.addAction(editor.act_snap_grid)
            self.template_module_menu.addSeparator()
            self.template_module_menu.addAction(editor.act_copy)
            self.template_module_menu.addAction(editor.act_paste)
            self.template_module_menu.addAction(editor.act_duplicate)
            self.template_module_menu.addAction(editor.act_delete)
            self.template_module_menu.addAction(editor.act_delete_anchor)
            self.template_module_menu.addSeparator()
            self.template_module_menu.addAction(editor.act_zoom_in)
            self.template_module_menu.addAction(editor.act_zoom_out)
            self.template_module_menu.addSeparator()
            self.template_module_menu.addAction(self.ribbon_close_template_action)
            return
        if library_mode:
            self.template_module_menu.addAction(self.ribbon_new_template_action)
            self.template_module_menu.addAction(self.ribbon_edit_template_action)
            self.template_module_menu.addAction(self.ribbon_delete_template_action)
            self.template_module_menu.addSeparator()
            self.template_module_menu.addAction(self.ribbon_close_template_action)

    def _selected_template_repository_entry(self) -> tuple[str, str] | None:
        row = self.template_library_table.currentRow() if hasattr(self, "template_library_table") else -1
        if row < 0:
            return None
        item = self.template_library_table.item(row, 1)
        if not item:
            return None
        return item.text(), str(item.data(Qt.UserRole) or "")

    def _register_single_template(self, template_path: str, display_name: str | None = None) -> None:
        self.template_repo.register(template_path, display_name=display_name)
        self._save_template_repository()
        self._refresh_template_library()

    def _open_embedded_template_editor(self, template_path: str = "") -> bool:
        if self.template_editor_embedded and not self._close_embedded_template_editor():
            return False
        while self.template_editor_layout.count():
            item = self.template_editor_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        from editor.template_editor import TemplateEditorWindow

        editor = TemplateEditorWindow(self, on_template_saved=lambda path, name: self._handle_template_saved(path, name))
        editor.setWindowFlags(Qt.Widget)
        editor.menuBar().setVisible(False)
        close_action = editor.template_toolbar.addAction(self.style().standardIcon(QStyle.SP_DialogCloseButton), "Close", self._close_template_module)
        close_action.setToolTip("Close")
        self.template_editor_embedded = editor
        self.template_editor_layout.addWidget(editor)
        self.template_editor_mode = "editor"
        self.stack.setCurrentIndex(4)
        if template_path:
            return bool(editor.load_template_from_path(template_path))
        return True

    def _handle_template_saved(self, template_path: str, display_name: str) -> None:
        self._register_single_template(template_path, display_name=display_name)

    def _create_new_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh mẫu giấy thi", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not path:
            return
        if not self._open_embedded_template_editor():
            return
        if not self.template_editor_embedded or not self.template_editor_embedded.load_image_from_path(path):
            return
        self.stack.setCurrentIndex(4)

    def _edit_selected_template(self) -> None:
        selected = self._selected_template_repository_entry()
        if not selected:
            return
        _, template_path = selected
        if not template_path:
            return
        if not self._open_embedded_template_editor(template_path):
            return
        self.stack.setCurrentIndex(4)

    def _delete_selected_template(self) -> None:
        selected = self._selected_template_repository_entry()
        if not selected:
            return
        name, template_path = selected
        if not self._confirm("Xoá mẫu giấy thi", f"Bạn có chắc muốn xoá mẫu giấy thi '{name}' khỏi kho?"):
            return
        self.template_repo.templates.pop(name, None)
        self._save_template_repository()
        if template_path and Path(template_path).exists():
            try:
                Path(template_path).unlink()
            except Exception:
                pass
        self._refresh_template_library()

    def _save_current_template(self) -> bool:
        if not self.template_editor_embedded:
            return False
        return bool(self.template_editor_embedded.save_template())

    def _save_current_template_as(self) -> bool:
        if not self.template_editor_embedded:
            return False
        return bool(self.template_editor_embedded.save_template_as())

    def _close_embedded_template_editor(self) -> bool:
        if self.template_editor_embedded and not self.template_editor_embedded._confirm_close():
            return False
        while self.template_editor_layout.count():
            item = self.template_editor_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.template_editor_embedded = None
        self.template_editor_mode = "library"
        self._refresh_template_library()
        self._navigate_to("template_library", context={"session_id": self.current_session_id}, push_current=False, require_confirm=False, reason="close_template_editor")
        return True

    def _close_template_module(self) -> bool:
        if self.template_editor_embedded:
            return self._close_embedded_template_editor()
        self._navigate_to("exam_list", context={}, push_current=False, require_confirm=False, reason="close_template_library")
        return True

    def _save_template_repository(self) -> None:
        try:
            self.database.set_app_state("template_repository", self.template_repo.to_dict())
        except Exception:
            pass

    def _register_templates_from_payload(self, payload: dict) -> None:
        common_template = str(payload.get("common_template", "") or "").strip()
        if common_template:
            self.template_repo.register(common_template)
        for cfg in payload.get("subject_configs", []) if isinstance(payload.get("subject_configs", []), list) else []:
            tp = str((cfg or {}).get("template_path", "") or "").strip()
            if tp:
                self.template_repo.register(tp)
        self._save_template_repository()

    def load_template(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Template", "", "JSON (*.json)")
        if not file_path:
            return
        self.template = Template.load_json(file_path)
        self.template_repo.register(file_path)
        self._save_template_repository()
        if self.session:
            self.session.template_path = file_path
        self.session_dirty = True
        self._refresh_session_info()
        self._refresh_batch_subject_controls()

    def open_template_editor(self) -> None:
        self._refresh_template_library()
        self.template_editor_mode = "library"
        self._navigate_to("template_library", context={"session_id": self.current_session_id}, push_current=True, require_confirm=False, reason="open_template_library")

    @staticmethod
    def _normalize_template_path(path_text: str) -> str:
        t = str(path_text or "").strip()
        if not t:
            return ""
        if t.lower() in {"[dùng mẫu chung]", "[dung mau chung]", "none", "null", "-"}:
            return ""
        return t
