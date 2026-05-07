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


class MainWindowWorkspaceMixin:
    """Exam list, embedded exam editor, route stack, menu/ribbon and generic action routing."""
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "scan_list"):
            idx = self.scan_list.currentRow()
            if 0 <= idx < len(self.scan_results):
                self._update_scan_preview(idx)
            elif idx >= 0:
                self._update_scan_preview_from_saved_row(idx)
        elif hasattr(self, "scan_image_scroll") and not self.preview_source_pixmap.isNull():
            self._render_preview_pixmap()

    def _build_exam_list_page(self) -> QWidget:
        """Build the single DB-backed exam list page.

        The table is intentionally simple: one row per exam session from SQLite,
        no cached registry file, no stacked sub-view, and one compact action cell
        using the shared software icon system.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        title_row = QHBoxLayout()
        title_icon = QLabel()
        try:
            pix = logo_symbol()
            if pix is not None and not pix.isNull():
                title_icon.setPixmap(pix.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass
        title = QLabel("Danh sách kỳ thi")
        title.setObjectName("ExamListTitle")
        title_row.addWidget(title_icon)
        title_row.addWidget(title)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.exam_list_table = QTableWidget(0, 7)
        self.exam_list_table.setHorizontalHeaderLabels([
            "STT", "Tên kỳ thi", "Số môn", "Thư mục quét", "Môn học", "Trạng thái", "Thao tác"
        ])
        self.exam_list_table.verticalHeader().setVisible(False)
        self.exam_list_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.exam_list_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.exam_list_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.exam_list_table.setAlternatingRowColors(True)
        hdr = self.exam_list_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.exam_list_table.cellDoubleClicked.connect(self._handle_exam_list_double_click)
        self.exam_list_table.itemSelectionChanged.connect(lambda: self._handle_stack_changed(self.stack.currentIndex()))
        layout.addWidget(self.exam_list_table, 1)
        return w

    def _build_subject_management_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Quản lý môn học và khối"))

        tables_row = QHBoxLayout()

        subject_group = QGroupBox("Môn học")
        subject_layout = QVBoxLayout(subject_group)
        self.subjects_table = QTableWidget(0, 2)
        self.subjects_table.setHorizontalHeaderLabels(["STT", "Tên môn"])
        self.subjects_table.verticalHeader().setVisible(False)
        self.subjects_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.subjects_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subjects_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.subjects_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.subjects_table.itemSelectionChanged.connect(lambda: self._handle_subject_management_selection("subjects"))
        subject_layout.addWidget(self.subjects_table)
        subject_order_row = QHBoxLayout()
        btn_subject_up = QPushButton("↑ Lên")
        btn_subject_up.clicked.connect(lambda: self._move_subject_management_row("subjects", -1))
        btn_subject_down = QPushButton("↓ Xuống")
        btn_subject_down.clicked.connect(lambda: self._move_subject_management_row("subjects", 1))
        subject_order_row.addWidget(btn_subject_up)
        subject_order_row.addWidget(btn_subject_down)
        subject_order_row.addStretch()
        subject_layout.addLayout(subject_order_row)

        grade_group = QGroupBox("Khối")
        grade_layout = QVBoxLayout(grade_group)
        self.grades_table = QTableWidget(0, 1)
        self.grades_table.setHorizontalHeaderLabels(["Tên khối"])
        self.grades_table.verticalHeader().setVisible(False)
        self.grades_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.grades_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.grades_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grades_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.grades_table.itemSelectionChanged.connect(lambda: self._handle_subject_management_selection("grades"))
        grade_layout.addWidget(self.grades_table)

        tables_row.addWidget(subject_group)
        tables_row.addWidget(grade_group)
        layout.addLayout(tables_row)

        form = QFormLayout()
        self.subject_management_label = QLabel("Tên môn")
        self.subject_management_editor = QLineEdit()
        self.subject_management_editor.setPlaceholderText("Nhập giá trị đang chọn")
        form.addRow(self.subject_management_label, self.subject_management_editor)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._subject_management_add)
        btn_edit = QPushButton("Edit")
        btn_edit.clicked.connect(self._subject_management_edit)
        btn_delete = QPushButton("Delete")
        btn_delete.clicked.connect(self._subject_management_delete)
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._save_subject_management)
        btn_back = QPushButton("Đóng")
        btn_back.clicked.connect(lambda: self._navigate_to("exam_list", context={}, push_current=False, require_confirm=True, reason="close_subject_management"))
        for btn in [btn_add, btn_edit, btn_delete, btn_save, btn_back]:
            btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        self._refresh_subject_management_tables()
        return w

    def _subject_management_values(self, mode: str) -> list[str]:
        return self.subjects if mode == "subjects" else self.grades

    def _subject_management_table(self, mode: str) -> QTableWidget:
        return self.subjects_table if mode == "subjects" else self.grades_table

    def _set_subject_management_mode(self, mode: str) -> None:
        self.subject_management_mode = mode
        self.subject_management_label.setText("Tên môn" if mode == "subjects" else "Tên khối")

    def _refresh_subject_management_tables(self, *, reset_from_catalog: bool = False) -> None:
        """Render the subject/block catalog editor without DB side effects."""
        if reset_from_catalog:
            self.subjects = list(self.subject_catalog)
            self.grades = list(self.block_catalog)
        for mode, values in (("subjects", self.subjects), ("grades", self.grades)):
            table = self._subject_management_table(mode)
            table.blockSignals(True)
            table.setRowCount(len(values))
            for row, value in enumerate(values):
                if mode == "subjects":
                    table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                    table.setItem(row, 1, QTableWidgetItem(value))
                else:
                    table.setItem(row, 0, QTableWidgetItem(value))
            table.clearSelection()
            table.blockSignals(False)
        self._subject_management_add()

    def _handle_subject_management_selection(self, mode: str) -> None:
        table = self._subject_management_table(mode)
        row = table.currentRow()
        if row < 0:
            return
        other_mode = "grades" if mode == "subjects" else "subjects"
        other_table = self._subject_management_table(other_mode)
        other_table.blockSignals(True)
        other_table.clearSelection()
        other_table.blockSignals(False)
        values = self._subject_management_values(mode)
        self._set_subject_management_mode(mode)
        self.subject_edit_index = row
        self.subject_management_editor.setText(values[row] if row < len(values) else "")

    def _subject_management_add(self) -> None:
        self.subject_edit_index = None
        self.subject_management_editor.clear()
        self._set_subject_management_mode(self.subject_management_mode or "subjects")
        self.subject_management_editor.setFocus()

    def _subject_management_edit(self) -> None:
        table = self._subject_management_table(self.subject_management_mode)
        row = table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Quản lý môn học", "Vui lòng chọn một dòng để chỉnh sửa.")
            return
        values = self._subject_management_values(self.subject_management_mode)
        self.subject_edit_index = row
        self.subject_management_editor.setText(values[row] if row < len(values) else "")
        self.subject_management_editor.setFocus()

    def _subject_management_delete(self) -> None:
        table = self._subject_management_table(self.subject_management_mode)
        row = table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Quản lý môn học", "Vui lòng chọn một dòng để xoá.")
            return
        values = self._subject_management_values(self.subject_management_mode)
        if row >= len(values):
            return
        del values[row]
        self._refresh_subject_management_tables()
        self._set_subject_management_mode(self.subject_management_mode)

    def _move_subject_management_row(self, mode: str, delta: int) -> None:
        table = self._subject_management_table(mode)
        row = table.currentRow()
        values = self._subject_management_values(mode)
        target = row + delta
        if row < 0 or row >= len(values) or target < 0 or target >= len(values):
            return
        values[row], values[target] = values[target], values[row]
        self._set_subject_management_mode(mode)
        self._refresh_subject_management_tables()
        table = self._subject_management_table(mode)
        table.selectRow(target)

    def _apply_subject_management_values(self) -> None:
        old_subjects = list(self.subject_catalog)
        old_blocks = list(self.block_catalog)
        self.subject_catalog = list(self.subjects)
        self.block_catalog = list(self.grades)
        self.database.replace_catalog("subjects", self.subject_catalog)
        self.database.replace_catalog("blocks", self.block_catalog)
        if old_subjects != self.subject_catalog:
            self.database.log_change("catalog", "subjects", "subject_catalog", old_subjects, self.subject_catalog, "subject_management")
        if old_blocks != self.block_catalog:
            self.database.log_change("catalog", "blocks", "block_catalog", old_blocks, self.block_catalog, "subject_management")

    def _sync_subject_configs_with_catalog(self) -> bool:
        """Compatibility no-op: keep exam subjects independent from the catalog.

        The catalog is a global pick-list for combo boxes. It is not a
        constraint on subjects already configured in an exam. Older code filtered
        session.config["subject_configs"] by the current catalog, which could
        silently remove a subject from the exam after a catalog edit or a
        partially loaded catalog.
        """
        return False

    def _save_subject_management(self) -> None:
        value = self.subject_management_editor.text().strip()
        values = self._subject_management_values(self.subject_management_mode)
        label = "môn học" if self.subject_management_mode == "subjects" else "khối"
        if value:
            if self.subject_edit_index is None:
                values.append(value)
            else:
                values[self.subject_edit_index] = value

        subject_values = [item.strip() for item in self.subjects if item.strip()]
        grade_values = [item.strip() for item in self.grades if item.strip()]
        if not subject_values:
            QMessageBox.warning(self, "Quản lý môn học", "Danh sách môn học không được để trống.")
            return
        if not grade_values:
            QMessageBox.warning(self, "Quản lý khối", "Danh sách khối không được để trống.")
            return

        normalized_current = [item.strip() for item in values if item.strip()]
        if len(normalized_current) != len(set(x.casefold() for x in normalized_current)):
            QMessageBox.warning(self, "Quản lý môn học", f"Danh sách {label} không được trùng lặp.")
            return
        if len(subject_values) != len(set(x.casefold() for x in subject_values)):
            QMessageBox.warning(self, "Quản lý môn học", "Danh sách môn học không được trùng lặp.")
            return
        if len(grade_values) != len(set(x.casefold() for x in grade_values)):
            QMessageBox.warning(self, "Quản lý khối", "Danh sách khối không được trùng lặp.")
            return

        self.subjects = subject_values
        self.grades = grade_values
        self._apply_subject_management_values()
        self.session_dirty = True

        if self.session:
            cfg = dict(self.session.config or {})
            cfg["subject_catalog"] = list(self.subject_catalog)
            cfg["block_catalog"] = list(self.block_catalog)
            self.session.config = cfg
            self._sync_subject_configs_with_catalog()

        QMessageBox.information(
            self,
            "Quản lý môn học",
            "Đã cập nhật danh sách môn và khối. Cấu hình môn trong kỳ thi hiện tại được giữ nguyên.",
        )
        self._refresh_subject_management_tables()
        self._set_subject_management_mode(self.subject_management_mode)

    def _build_workspace_page(self) -> QWidget:
        central = QWidget()
        root_layout = QVBoxLayout(central)

        # Keep only Batch Scan UI visible in workspace.
        group_scan = QGroupBox("Batch Scan")
        l2 = QVBoxLayout(group_scan); l2.addWidget(self._build_scan_tab())
        root_layout.addWidget(group_scan)

        # Initialize hidden widgets still used by existing logic.
        self._hidden_session_tab = self._build_session_tab()
        self._hidden_correction_tab = self._build_correction_tab()
        return central

    def _session_id_for_row(self, row_idx: int) -> str | None:
        item = self.exam_list_table.item(row_idx, 1)
        if not item:
            return None
        sid = item.data(Qt.UserRole)
        return str(sid) if sid else None

    def _handle_exam_list_double_click(self, row: int, col: int) -> None:
        # double click navigation: route row double click to exam editor action.
        sid = self._session_id_for_row(row)
        if not sid:
            return
        self.exam_list_table.selectRow(row)
        self._edit_registry_session_by_id(sid)

    def _handle_scan_list_double_click(self, row: int, col: int) -> None:
        # double click navigation: route row double click to scan edit action.
        if row < 0 or row >= self.scan_list.rowCount():
            return
        self.scan_list.selectRow(row)
        self._open_edit_selected_scan()

    def _make_row_icon_button(self, icon, tooltip: str, cb):
        """Create a compact text+shared-icon button for table action cells.

        The `icon` argument is accepted for compatibility with older call sites,
        but the rendered button uses the application's shared toolbar icon map.
        """
        short_text = {
            "Xem kỳ thi": "Xem",
            "Xoá kỳ thi": "Xoá",
            "Xóa kỳ thi": "Xoá",
            "Đặt mặc định": "Mặc định",
            "Sửa bài thi": "Sửa",
            "Xoá bài thi": "Xoá",
            "Xóa bài thi": "Xoá",
        }.get(str(tooltip or ""), str(tooltip or "...").replace(" kỳ thi", "").replace(" bài thi", ""))
        btn = QPushButton(short_text)
        btn.setToolTip(str(tooltip or short_text))
        icon_name = {
            "Xem kỳ thi": "preview",
            "Xoá kỳ thi": "delete",
            "Xóa kỳ thi": "delete",
            "Đặt mặc định": "save",
            "Sửa bài thi": "edit",
            "Xoá bài thi": "delete",
            "Xóa bài thi": "delete",
        }.get(str(tooltip or ""), "")
        if icon_name:
            try:
                ico = TOOLBAR.get(icon_name)
                if ico is not None and not ico.isNull():
                    btn.setIcon(ico)
            except Exception:
                pass
        btn.setFlat(False)
        btn.clicked.connect(cb)
        try:
            brand_button(btn)
        except Exception:
            pass
        return btn

    def _refresh_exam_list(self) -> None:
        """Reload and render the exam list strictly from SQLite.

        Flow after create/edit/delete/default-change:
            DB write -> list_exam_sessions() -> fetch each session payload -> render grid.
        No legacy registry cache and no per-row UI state are reused.
        """
        self.session_registry = self._load_session_registry()
        self.exam_list_table.setRowCount(0)
        self.exam_list_table.setRowCount(len(self.session_registry))
        for idx, row in enumerate(self.session_registry):
            sid = str(row.get("session_id", "") or "")
            name = str(row.get("name") or f"Kỳ thi {idx + 1}")
            subject_text = "-"
            subject_count = "0"
            scan_root = "-"
            status = "Mặc định" if bool(row.get("default")) else "Đã lưu"
            payload = self.database.fetch_exam_session(sid) if sid else None
            if sid and payload:
                try:
                    ses = ExamSession.from_dict(payload)
                    cfg = dict(ses.config or {})
                    subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
                    subject_count = str(len(subject_cfgs))
                    subject_items = [
                        f"{str(x.get('name', '?') or '?')}-{str(x.get('block', '?') or '?')}"
                        for x in subject_cfgs[:4]
                        if isinstance(x, dict)
                    ]
                    subject_text = ", ".join(subject_items) or "-"
                    if len(subject_cfgs) > 4:
                        subject_text += f" ...(+{len(subject_cfgs) - 4})"
                    scan_root = str(cfg.get("scan_root", "") or "-")
                    name = str(ses.exam_name or name)
                except Exception:
                    status = "Lỗi dữ liệu"
            elif sid:
                status = "Không tìm thấy"

            self.exam_list_table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, sid)
            self.exam_list_table.setItem(idx, 1, name_item)
            self.exam_list_table.setItem(idx, 2, QTableWidgetItem(subject_count))
            self.exam_list_table.setItem(idx, 3, QTableWidgetItem(scan_root))
            self.exam_list_table.setItem(idx, 4, QTableWidgetItem(subject_text))
            self.exam_list_table.setItem(idx, 5, QTableWidgetItem(status))

            action_wrap = QWidget()
            action_layout = QHBoxLayout(action_wrap)
            action_layout.setContentsMargins(4, 2, 4, 2)
            action_layout.setSpacing(6)
            for button in [
                self._make_row_icon_button(None, "Xem kỳ thi", lambda _=False, s=sid: self._edit_registry_session_by_id(s)),
                self._make_row_icon_button(None, "Xoá kỳ thi", lambda _=False, s=sid: self._delete_registry_session_by_id(s)),
                self._make_row_icon_button(None, "Đặt mặc định", lambda _=False, s=sid: self._set_default_registry_session_by_id(s)),
            ]:
                action_layout.addWidget(button)
            self.exam_list_table.setCellWidget(idx, 6, action_wrap)

        self.exam_list_table.resizeRowsToContents()

    def _selected_registry_path(self) -> Path | None:
        row = self.exam_list_table.currentRow()
        if row < 0:
            return None
        sid = self._session_id_for_row(row)
        if not sid:
            return None
        return self._session_path_from_id(sid)

    def _open_selected_registry_session(self) -> None:
        path = self._selected_registry_path()
        if not path:
            QMessageBox.warning(self, "Mở kỳ thi", "Chọn kỳ thi trong danh sách trước.")
            return
        self._open_session_path(path)

    def _edit_selected_registry_session(self) -> None:
        row = self.exam_list_table.currentRow()
        sid = self._session_id_for_row(row) if row >= 0 else None
        if not sid:
            QMessageBox.warning(self, "Sửa kỳ thi", "Chọn kỳ thi trong danh sách trước.")
            return
        self._edit_registry_session_by_id(sid)

    def _edit_registry_session_by_id(self, session_id: str) -> None:
        payload = self.database.fetch_exam_session(session_id)
        if not payload:
            QMessageBox.warning(self, "Sửa kỳ thi", "Không tìm thấy kỳ thi trong kho lưu trữ hệ thống.")
            return False        
        try:
            session = ExamSession.from_dict(payload)
            cfg = session.config or {}
            payload = {
                "exam_name": session.exam_name,
                "common_template": session.template_path,
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
                    for s in (session.students or [])
                ],
                "scan_mode": cfg.get("scan_mode", "Ảnh trong thư mục gốc"),
                "paper_part_count": cfg.get("paper_part_count", 3),
                "subject_configs": cfg.get("subject_configs", []),
            }
            self._open_embedded_exam_editor(session_id, session, payload)
        except Exception as exc:
            QMessageBox.warning(self, "Sửa kỳ thi", f"Không thể sửa kỳ thi\n{exc}")

    def _open_embedded_exam_editor(self, session_id: str, session: ExamSession, payload: dict, *, is_new: bool = False) -> None:
        prev_session_id = str(getattr(self, "current_session_id", "") or "").strip()
        next_session_id = str(session_id or "").strip()
        if prev_session_id and next_session_id and prev_session_id != next_session_id:
            # Hard isolation between exams: drop any pending auto-recognition jobs/signatures
            # before switching the active editor context to another session.
            self._reset_auto_recognition_state(pause=False)

        while self.exam_editor_layout.count():
            item = self.exam_editor_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if session_id:
            self.current_session_id = session_id
            self.current_session_path = self._session_path_from_id(session_id)
        if session is not None:
            session_cfg = dict(session.config or {})
            session_subjects = session_cfg.get("subject_configs", [])
            if isinstance(session_subjects, list):
                session_cfg["subject_configs"] = self._canonicalize_subject_configs_for_session(session_subjects)
                session.config = session_cfg
                payload = dict(payload or {})
                payload_subjects = payload.get("subject_configs", session_cfg["subject_configs"])
                if isinstance(payload_subjects, list):
                    payload["subject_configs"] = self._canonicalize_subject_configs_for_session(payload_subjects)
            self.session = session
            if not is_new:
                self.session_dirty = False
            self._refresh_session_info()
            self._refresh_batch_subject_controls()
            self._refresh_scoring_phase_table()
            self._refresh_ribbon_action_states()

        self.embedded_exam_session_id = session_id
        self.embedded_exam_session = session
        self.embedded_exam_original_payload = dict(payload)
        self.embedded_exam_is_new = bool(is_new)

        dlg = NewExamDialog(
            self.subject_catalog,
            self.block_catalog,
            data=payload,
            parent=self,
            on_batch_scan_subject=lambda x: self._handle_batch_request_from_editor(x),
            on_save_exam=self._save_embedded_exam_editor,
            stay_open_on_save=True,
            template_repo=self.template_repo,
        )
        dlg.setWindowFlags(Qt.Widget)
        dlg.rejected.connect(self._close_embedded_exam_editor)
        self.embedded_exam_dialog = dlg
        self.exam_editor_layout.addWidget(dlg)
        self._navigate_to("exam_editor", context={"session_id": session_id}, push_current=True, require_confirm=False, reason="open_exam_editor")

    def _save_embedded_exam_editor(self, *, show_message: bool = True, refresh_subject_grid: bool = True) -> bool:
        if not self.embedded_exam_dialog or not self.embedded_exam_session_id:
            return False
        edited = self.embedded_exam_dialog.payload()
        exam_name = str(edited.get("exam_name", "") or "").strip()
        if not exam_name:
            QMessageBox.warning(self, "Lưu kỳ thi", "Tên kỳ thi không được để trống.")
            return False
        session_id = self.embedded_exam_session_id
        if self._session_name_exists(exam_name, exclude_session_id=session_id):
            QMessageBox.warning(self, "Lưu kỳ thi", "Tên kỳ thi đã tồn tại. Vui lòng chọn tên khác.")
            return False
        self._register_templates_from_payload(edited)
        saved_payload = self.database.fetch_exam_session(session_id)
        if not saved_payload and not self.embedded_exam_is_new:
            QMessageBox.warning(self, "Sửa kỳ thi", "Không tìm thấy kỳ thi trong kho lưu trữ hệ thống.")
            return False
        try:
            if saved_payload:
                session = ExamSession.from_dict(saved_payload)
            else:
                session = ExamSession(
                    exam_name=str(edited.get("exam_name", "") or "").strip() or "Kỳ thi",
                    exam_date=str(date.today()),
                    subjects=[],
                    template_path=str(edited.get("common_template", "") or ""),
                    answer_key_path="",
                    students=[],
                    config={},
                )
            session.exam_name = edited.get("exam_name", session.exam_name)
            session.template_path = edited.get("common_template", session.template_path)
            session.subjects = [
                f"{str(x.get('name', '')).strip()}_{str(x.get('block', '')).strip()}"
                for x in edited.get("subject_configs", [])
                if str(x.get("name", "")).strip()
            ] or session.subjects
            incoming_students = edited.get("students", []) if isinstance(edited.get("students", []), list) else []
            session.students = self._students_from_editor_rows(incoming_students)
            session.config = {
                **(session.config or {}),
                "scan_mode": edited.get("scan_mode", "Ảnh trong thư mục gốc"),
                "scan_root": edited.get("scan_root", ""),
                "student_list_path": edited.get("student_list_path", ""),
                "paper_part_count": edited.get("paper_part_count", 3),
                "subject_configs": edited.get("subject_configs", []),
                "subject_catalog": self.subject_catalog,
                "block_catalog": self.block_catalog,
            }
            self.session = session
            self.embedded_exam_session = session
            self.embedded_exam_session_id = session_id
            self.current_session_id = session_id
            self.current_session_path = self._session_path_from_id(session_id)
            persistence_payload = self._build_session_persistence_payload()
            session.config = dict(persistence_payload.get("config", {}) or {})
            self.database.save_exam_session(session_id, session.exam_name, persistence_payload)
            if self.embedded_exam_dialog:
                self.embedded_exam_dialog.subject_configs = list(session.config.get("subject_configs", []) or [])
                if refresh_subject_grid:
                    self.embedded_exam_dialog._reload_subject_grid(reason="save_embedded_exam_editor")
                self.embedded_exam_original_payload = self.embedded_exam_dialog.payload()
            else:
                self.embedded_exam_original_payload = edited
            self.embedded_exam_is_new = False
            self.session_dirty = False
            self._refresh_exam_list()
            self._refresh_session_info()
            self._refresh_batch_subject_controls()
            self._schedule_auto_recognition_for_existing_files(
                list(session.config.get("subject_configs", []) or [])
            )
            self._refresh_ribbon_action_states()
            if hasattr(self, "scoring_subject_combo"):
                preferred_scoring = self._resolve_preferred_scoring_subject()
                self._populate_scoring_subjects(preferred_scoring)
                self._refresh_scoring_phase_table()
            if show_message:
                QMessageBox.information(self, "Xem kỳ thi", "Đã lưu thông số kỳ thi.")
            if self.embedded_exam_dialog:
                self.embedded_exam_dialog.show()
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Sửa kỳ thi", f"Không thể sửa kỳ thi\n{exc}")
            return False

    @staticmethod
    def _students_from_editor_rows(rows: list[dict] | None) -> list[Student]:
        """
        Convert student rows from exam-editor payload into canonical Student objects
        without dropping valid SID-only records.
        """
        normalized: dict[str, Student] = {}
        ordered_keys: list[str] = []
        for raw in rows or []:
            if not isinstance(raw, dict):
                continue
            sid = str(raw.get("student_id", "") or "").strip()
            if not sid:
                continue
            key = sid.casefold()
            name = str(raw.get("name", "") or "").strip()
            birth_date = str(raw.get("birth_date", "") or "")
            class_name = str(raw.get("class_name", "") or "")
            exam_room = str(raw.get("exam_room", "") or "")
            existing = normalized.get(key)
            if existing is None:
                normalized[key] = Student(
                    student_id=sid,
                    name=name,
                    extra={
                        "birth_date": birth_date,
                        "class_name": class_name,
                        "exam_room": exam_room,
                    },
                )
                ordered_keys.append(key)
                continue
            # Merge duplicates by SID: prefer non-empty values from the newest row.
            if name:
                existing.name = name
            merged_extra = dict(getattr(existing, "extra", {}) or {})
            if birth_date:
                merged_extra["birth_date"] = birth_date
            if class_name:
                merged_extra["class_name"] = class_name
            if exam_room:
                merged_extra["exam_room"] = exam_room
            existing.extra = merged_extra
        return [normalized[k] for k in ordered_keys if k in normalized]

    def _close_embedded_exam_editor(self) -> None:
        """Leave the active exam and return to the DB-backed exam list.

        The in-memory session is intentionally cleared here: opening an exam creates
        the runtime session; returning to the exam list destroys that runtime session.
        Persisted data remains in SQLite and the list is reloaded from DB.
        """
        self.close_current_session()

    @staticmethod
    def _payload_changed(a: dict | None, b: dict | None) -> bool:
        return json.dumps(a or {}, ensure_ascii=False, sort_keys=True) != json.dumps(b or {}, ensure_ascii=False, sort_keys=True)

    def _handle_batch_request_from_editor(self, batch_payload: dict) -> bool:
        if not self.embedded_exam_dialog or not self.embedded_exam_session or not self.embedded_exam_session_id:
            return False

        current_payload = self.embedded_exam_dialog.payload()
        if self._payload_changed(current_payload, self.embedded_exam_original_payload):
            msg = QMessageBox(self)
            msg.setWindowTitle("Xác nhận")
            msg.setText("Dữ liệu đã thay đổi. Bạn muốn lưu trước khi chuyển sang nhận dạng?")
            msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Save)
            ch = msg.exec()
            if ch == QMessageBox.Cancel:
                return False
            if ch == QMessageBox.Save and not self._save_embedded_exam_editor():
                return False

        session_id = self.embedded_exam_session_id
        base_session = self.embedded_exam_session
        self._close_embedded_exam_editor()
        if not session_id or not base_session:
            return False
        self._open_batch_scan_from_exam_editor(session_id, base_session, batch_payload)
        return True

    def _open_batch_scan_from_exam_editor(self, session_id: str, base_session: ExamSession, payload: dict) -> None:
        exam_name = str(payload.get("exam_name") or base_session.exam_name or "Kỳ thi")
        common_template = str(payload.get("common_template") or base_session.template_path or "")
        selected_subject_index = int(payload.get("selected_subject_index", 0) or 0)
        all_subjects = payload.get("subject_configs")
        if not isinstance(all_subjects, list) or not all_subjects:
            all_subjects = list((base_session.config or {}).get("subject_configs", []))
        all_subjects = self._canonicalize_subject_configs_for_session(list(all_subjects or []))
        if not all_subjects:
            subject_from_payload = dict(payload.get("subject_config") or {})
            if subject_from_payload:
                all_subjects = self._canonicalize_subject_configs_for_session([subject_from_payload])
                selected_subject_index = 0
        if not (0 <= selected_subject_index < len(all_subjects)):
            selected_subject_index = 0
        subject_cfg = dict(all_subjects[selected_subject_index]) if all_subjects else dict(payload.get("subject_config") or {})
        if not subject_cfg:
            QMessageBox.warning(self, "Batch Scan", "Không tìm thấy cấu hình môn để nhận dạng.")
            return

        self.batch_editor_return_payload = {
            "exam_name": exam_name,
            "common_template": common_template,
            "scan_root": str(payload.get("scan_root") or (base_session.config or {}).get("scan_root", "") or ""),
            "student_list_path": str(payload.get("student_list_path") or (base_session.config or {}).get("student_list_path", "") or ""),
            "students": list(payload.get("students", [])) if isinstance(payload.get("students", []), list) else [
                {
                    "student_id": s.student_id,
                    "name": s.name,
                    "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                    "class_name": str((s.extra or {}).get("class_name", "") or ""),
                    "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
                }
                for s in (base_session.students or [])
            ],
            "scan_mode": str(payload.get("scan_mode") or (base_session.config or {}).get("scan_mode", "Ảnh trong thư mục gốc")),
            "paper_part_count": int(payload.get("paper_part_count") or (base_session.config or {}).get("paper_part_count", 3) or 3),
            "subject_configs": all_subjects,
        }
        self.batch_editor_return_session_id = session_id
        scan_root = str(payload.get("scan_root") or (base_session.config or {}).get("scan_root", "") or "")
        scan_mode = str(payload.get("scan_mode") or (base_session.config or {}).get("scan_mode", "Ảnh trong thư mục gốc"))
        paper_part_count = int(payload.get("paper_part_count") or (base_session.config or {}).get("paper_part_count", 3) or 3)

        self.session = ExamSession(
            exam_name=exam_name,
            exam_date=str(date.today()),
            subjects=[f"{subject_cfg.get('name', '')}_{subject_cfg.get('block', '')}"],
            template_path=common_template,
            answer_key_path=str(base_session.answer_key_path or ""),
            students=list(base_session.students or []),
            config={
                "scan_mode": scan_mode,
                "scan_root": scan_root,
                "student_list_path": str(payload.get("student_list_path") or (base_session.config or {}).get("student_list_path", "") or ""),
                "paper_part_count": paper_part_count,
                "subject_configs": all_subjects,
                "subject_catalog": self.subject_catalog,
                "block_catalog": self.block_catalog,
            },
        )

        self.current_session_id = session_id
        self.current_session_path = self._session_path_from_id(session_id)
        # Persist only the lightweight canonical subject keys before Batch Scan.
        # Recognition rows will be saved under these same keys, so the subject
        # status column can refresh immediately without requiring a page switch.
        try:
            persistence_payload = self._build_session_persistence_payload()
            self.session.config = dict(persistence_payload.get("config", {}) or {})
            self.database.save_exam_session(session_id, self.session.exam_name, persistence_payload)
        except Exception:
            pass
        self.session_dirty = False
        self._refresh_session_info()
        self._refresh_batch_subject_controls()
        self.batch_subject_combo.setCurrentIndex(max(1, selected_subject_index + 1))
        selected_instance_key = self._subject_instance_key_from_cfg(subject_cfg)
        self._navigate_to(
            "workspace_batch_scan",
            context={
                "session_id": session_id,
                "origin": "exam_editor",
                "selected_subject_instance_key": selected_instance_key,
            },
            push_current=True,
            require_confirm=False,
            reason="open_batch_from_exam_editor",
        )

    def _delete_selected_registry_session(self) -> None:
        row = self.exam_list_table.currentRow()
        sid = self._session_id_for_row(row) if row >= 0 else None
        if not sid:
            QMessageBox.warning(self, "Xoá kỳ thi", "Chọn kỳ thi trong danh sách trước.")
            return
        self._delete_registry_session_by_id(sid)

    def _delete_registry_session_by_id(self, session_id: str) -> None:
        if not self._confirm("Xoá kỳ thi", "Bạn có chắc muốn xoá kỳ thi khỏi danh sách?"):
            return
        self.database.delete_exam_session(session_id)
        if str(getattr(self, "current_session_id", "") or "") == str(session_id):
            self.close_current_session()
        else:
            self._refresh_exam_list()

    def _set_default_selected_registry_session(self) -> None:
        row = self.exam_list_table.currentRow()
        sid = self._session_id_for_row(row) if row >= 0 else None
        if not sid:
            QMessageBox.warning(self, "Đặt mặc định", "Chọn kỳ thi trong danh sách trước.")
            return
        self._set_default_registry_session_by_id(sid)

    def _set_default_registry_session_by_id(self, session_id: str) -> None:
        if not self._confirm("Đặt mặc định", "Đặt kỳ thi này làm mặc định?"):
            return
        self.database.set_app_state("default_session_id", str(session_id))
        self._refresh_exam_list()

    def _create_branded_action(
        self,
        text: str,
        callback=None,
        *,
        icon_name: str = "",
        shortcut: str | QKeySequence | None = None,
        status_tip: str = "",
        parent=None,
    ) -> QAction:
        """Create one QAction and apply the common icon/shortcut/status policy."""
        action = QAction(str(text or ""), parent or self)
        if callback is not None:
            action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut) if isinstance(shortcut, str) else shortcut)
        if status_tip:
            action.setStatusTip(status_tip)
            action.setToolTip(status_tip)
        if icon_name:
            self._set_action_icon_from_branding(action, icon_name)
        return action

    def _add_menu_action(
        self,
        menu: QMenu,
        attr_name: str,
        text: str,
        callback,
        *,
        icon_name: str = "",
        shortcut: str | QKeySequence | None = None,
        status_tip: str = "",
    ) -> QAction:
        action = self._create_branded_action(
            text,
            callback,
            icon_name=icon_name,
            shortcut=shortcut,
            status_tip=status_tip,
        )
        setattr(self, attr_name, action)
        menu.addAction(action)
        return action

    def _build_menu(self) -> None:
        """Build a compact, workflow-oriented menu/ribbon.

        Legacy developer-only commands are intentionally not exposed here:
        Load Template JSON, Load Answer Keys JSON, Load Selected Scan Result and
        Apply Manual Correction. The corresponding methods are kept for backward
        compatibility, but the main UI now routes users through the DB-first
        workflow: configure -> recognize -> edit/recheck -> score -> export.
        """
        self.menuBar().clear()

        exam_menu = self.menuBar().addMenu("Kỳ thi")
        self.act_new_session = self._add_menu_action(
            exam_menu,
            "act_new_session",
            "Tạo kỳ thi mới",
            self.action_create_session,
            icon_name="exam",
            shortcut="Ctrl+N",
            status_tip="Tạo một kỳ thi mới.",
        )
        self.act_open_from_list = self._add_menu_action(
            exam_menu,
            "act_open_from_list",
            "Danh sách kỳ thi",
            self.action_open_session,
            icon_name="home",
            shortcut="Ctrl+O",
            status_tip="Quay về danh sách kỳ thi và mở kỳ thi đã có.",
        )
        exam_menu.addSeparator()
        self.act_save_session = self._add_menu_action(
            exam_menu,
            "act_save_session",
            "Lưu kỳ thi",
            self.action_save_session,
            icon_name="save",
            shortcut="Ctrl+S",
            status_tip="Lưu cấu hình kỳ thi hiện tại.",
        )
        self.act_save_as_subject = self._add_menu_action(
            exam_menu,
            "act_save_as_subject",
            "Nhân bản kỳ thi...",
            self.action_save_session_as,
            icon_name="export",
            status_tip="Tạo một bản sao từ kỳ thi đang chọn trong danh sách.",
        )
        self.act_close_current_session = self._add_menu_action(
            exam_menu,
            "act_close_current_session",
            "Đóng kỳ thi hiện tại",
            self.action_close_current_session,
            icon_name="close",
            status_tip="Đóng ngữ cảnh kỳ thi đang mở.",
        )
        exam_menu.addSeparator()
        self.act_exit = self._add_menu_action(
            exam_menu,
            "act_exit",
            "Thoát",
            self.action_exit,
            icon_name="close",
            status_tip="Đóng ứng dụng.",
        )

        config_menu = self.menuBar().addMenu("Cấu hình")
        self.act_current_exam_subjects = self._add_menu_action(
            config_menu,
            "act_current_exam_subjects",
            "Môn thi của kỳ thi",
            self.action_open_current_exam_subjects,
            icon_name="subject",
            status_tip="Mở danh sách môn thi thuộc kỳ thi hiện tại.",
        )
        self.act_manage_subject = self._add_menu_action(
            config_menu,
            "act_manage_subject",
            "Danh mục môn học / khối",
            self.action_manage_subjects,
            icon_name="subject",
            status_tip="Quản lý danh mục dùng chung cho cấu hình kỳ thi.",
        )
        self.act_manage_template = self._add_menu_action(
            config_menu,
            "act_manage_template",
            "Mẫu giấy thi",
            self.action_manage_template,
            icon_name="template",
            status_tip="Tạo, sửa và kiểm tra mẫu giấy thi.",
        )
        self.act_close_template_module = self._add_menu_action(
            config_menu,
            "act_close_template_module",
            "Đóng quản lý mẫu giấy thi",
            self._close_template_module,
            icon_name="close",
            status_tip="Đóng module mẫu giấy thi.",
        )
        self.act_close_template_module.setVisible(False)
        config_menu.addSeparator()
        self.act_import_answer_key = self._add_menu_action(
            config_menu,
            "act_import_answer_key",
            "Nhập đáp án...",
            self.action_import_answer_key,
            icon_name="import",
            status_tip="Nhập đáp án theo môn/mã đề vào CSDL.",
        )
        self.act_export_answer_key_sample = self._add_menu_action(
            config_menu,
            "act_export_answer_key_sample",
            "Xuất mẫu đáp án...",
            self.action_export_answer_key_sample,
            icon_name="export",
            status_tip="Xuất file mẫu để nhập đáp án.",
        )

        workflow_menu = self.menuBar().addMenu("Quy trình")
        self.act_batch_scan_menu = self._add_menu_action(
            workflow_menu,
            "act_batch_scan_menu",
            "Xử lý ảnh / Batch Scan",
            self.action_run_batch_scan,
            icon_name="scan",
            shortcut="Ctrl+B",
            status_tip="Mở màn hình nhận dạng bài thi.",
        )
        self.act_execute_batch_scan = self._add_menu_action(
            workflow_menu,
            "act_execute_batch_scan",
            "Nhận dạng môn đang chọn",
            self.action_execute_batch_scan,
            icon_name="scan",
            status_tip="Chạy nhận dạng cho môn đang chọn trong Batch Scan.",
        )
        self.act_edit_selected_scan = self._add_menu_action(
            workflow_menu,
            "act_edit_selected_scan",
            "Sửa bài thi đang chọn",
            self.action_edit_selected_scan,
            icon_name="edit",
            status_tip="Mở màn hình sửa bài thi đang chọn trong Batch Scan.",
        )
        workflow_menu.addSeparator()
        self.act_calculate_scores = self._add_menu_action(
            workflow_menu,
            "act_calculate_scores",
            "Tính điểm",
            self.action_calculate_scores,
            icon_name="scoring",
            shortcut="Ctrl+R",
            status_tip="Tính điểm theo dữ liệu nhận dạng đã chuẩn hóa theo môn.",
        )
        self.act_open_recheck = self._add_menu_action(
            workflow_menu,
            "act_open_recheck",
            "Phúc tra",
            self.action_open_recheck,
            icon_name="recheck",
            status_tip="Mở quy trình phúc tra / giải trình điểm.",
        )

        self.export_menu = self.menuBar().addMenu("Xuất báo cáo")
        self.act_export_reports_center = self._add_menu_action(
            self.export_menu,
            "act_export_reports_center",
            "Trung tâm báo cáo...",
            self.action_open_export_reports_center,
            icon_name="report",
            status_tip="Mở trung tâm báo cáo thống kê.",
        )
        self.export_menu.addSeparator()
        self.act_export_subject_scores = self._add_menu_action(
            self.export_menu,
            "act_export_subject_scores",
            "Xuất điểm môn...",
            self.action_export_subject_scores,
            icon_name="export",
        )
        self.act_export_subject_score_matrix = self._add_menu_action(
            self.export_menu,
            "act_export_subject_score_matrix",
            "Xuất điểm các môn...",
            self.action_export_subject_score_matrix,
            icon_name="export",
        )
        self.act_export_class_subject_scores = self._add_menu_action(
            self.export_menu,
            "act_export_class_subject_scores",
            "Xuất điểm lớp...",
            self.action_export_class_subject_scores,
            icon_name="export",
        )
        self.act_export_all_classes_subject_scores = self._add_menu_action(
            self.export_menu,
            "act_export_all_classes_subject_scores",
            "Xuất điểm các lớp...",
            self.action_export_all_classes_subject_scores,
            icon_name="export",
        )
        self.act_export_all_scores = self._add_menu_action(
            self.export_menu,
            "act_export_all_scores",
            "Xuất điểm chi tiết các môn...",
            self.action_export_all_subject_scores,
            icon_name="export",
        )
        self.act_export_return_by_class = self._add_menu_action(
            self.export_menu,
            "act_export_return_by_class",
            "Trả bài theo lớp...",
            self.action_export_return_by_class,
            icon_name="export",
        )
        self.act_export_recheck_by_subject = self._add_menu_action(
            self.export_menu,
            "act_export_recheck_by_subject",
            "Đóng gói bài phúc tra theo môn...",
            self.action_export_recheck_by_subject,
            icon_name="export",
        )
        self.act_export_recheck_by_class = self._add_menu_action(
            self.export_menu,
            "act_export_recheck_by_class",
            "Đóng gói bài phúc tra theo lớp...",
            self.action_export_recheck_by_class,
            icon_name="export",
        )
        self.export_menu.addSeparator()
        self.act_export_subject_api = self._add_menu_action(
            self.export_menu,
            "act_export_subject_api",
            "Xuất API bài làm theo môn (;)",
            self.action_export_subject_api_payload,
            icon_name="export",
        )
        self.export_menu.addSeparator()
        self.act_export_range_report = self._add_menu_action(
            self.export_menu,
            "act_export_range_report",
            "Báo cáo khoảng điểm...",
            self.action_export_score_range_report,
            icon_name="report",
        )
        self.act_export_class_report = self._add_menu_action(
            self.export_menu,
            "act_export_class_report",
            "Báo cáo theo lớp...",
            self.action_export_class_report,
            icon_name="report",
        )
        self.act_export_management_report = self._add_menu_action(
            self.export_menu,
            "act_export_management_report",
            "Báo cáo tổng hợp quản lý...",
            self.action_export_management_report,
            icon_name="report",
        )

        self.template_module_menu = self.menuBar().addMenu("Mẫu giấy thi")
        self.template_module_menu.menuAction().setVisible(False)

        toolbar = QToolBar("Quy trình")
        toolbar.setObjectName("mainWorkflowRibbon")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        self.main_ribbon = toolbar

        def _add_toolbar_action(attr_name: str, text: str, callback, icon_name: str, status_tip: str = "") -> QAction:
            action = self._create_branded_action(text, callback, icon_name=icon_name, status_tip=status_tip, parent=self)
            toolbar.addAction(action)
            setattr(self, attr_name, action)
            return action

        self.ribbon_new_exam_action = _add_toolbar_action("ribbon_new_exam_action", "Tạo kỳ thi", self.action_create_session, "add", "Tạo kỳ thi mới.")
        self.ribbon_view_exam_action = _add_toolbar_action("ribbon_view_exam_action", "Danh sách", self.action_open_session, "exam", "Danh sách kỳ thi.")
        self.ribbon_subject_list_action = _add_toolbar_action("ribbon_subject_list_action", "Môn thi", self.action_open_current_exam_subjects, "subject", "Danh sách môn thi của kỳ thi hiện tại.")
        self.ribbon_batch_scan_action = _add_toolbar_action("ribbon_batch_scan_action", "Xử lý ảnh", self.action_run_batch_scan, "scan", "Mở màn hình Batch Scan.")
        self.ribbon_scoring_action = _add_toolbar_action("ribbon_scoring_action", "Tính điểm", self.action_calculate_scores, "scoring", "Tính điểm bài thi.")
        self.ribbon_recheck_action = _add_toolbar_action("ribbon_recheck_action", "Phúc tra", self.action_open_recheck, "recheck", "Phúc tra / giải trình điểm.")

        self.ribbon_export_action = self._create_branded_action("Báo cáo", self.action_open_export_reports_center, icon_name="report", status_tip="Xuất điểm, API và báo cáo thống kê.", parent=self)
        self.ribbon_export_action.setMenu(self.export_menu)
        toolbar.addAction(self.ribbon_export_action)

        self.ribbon_context_separator = toolbar.addSeparator()

        self.ribbon_batch_execute_action = _add_toolbar_action("ribbon_batch_execute_action", "Nhận dạng", self.action_execute_batch_scan, "scan", "Chạy nhận dạng cho môn đang chọn.")
        self.ribbon_batch_save_action = _add_toolbar_action("ribbon_batch_save_action", "Lưu thay đổi", self._save_batch_for_selected_subject, "save", "Lưu các thay đổi phát sinh trong Batch Scan.")
        self.ribbon_batch_close_action = _add_toolbar_action("ribbon_batch_close_action", "Đóng Batch", self._close_batch_scan_view, "close", "Đóng màn hình Batch Scan.")

        self.ribbon_exam_editor_add_subject_action = _add_toolbar_action("ribbon_exam_editor_add_subject_action", "Thêm môn", self._exam_editor_add_subject, "add")
        self.ribbon_exam_editor_edit_subject_action = _add_toolbar_action("ribbon_exam_editor_edit_subject_action", "Sửa môn", self._exam_editor_edit_subject, "edit")
        self.ribbon_exam_editor_delete_subject_action = _add_toolbar_action("ribbon_exam_editor_delete_subject_action", "Xoá môn", self._exam_editor_delete_subject, "delete")
        self.ribbon_exam_editor_save_action = _add_toolbar_action("ribbon_exam_editor_save_action", "Lưu cấu hình", self._exam_editor_save, "save")
        self.ribbon_exam_editor_close_action = _add_toolbar_action("ribbon_exam_editor_close_action", "Đóng cấu hình", self._exam_editor_close, "close")

        self.ribbon_add_subject_action = _add_toolbar_action("ribbon_add_subject_action", "Thêm môn", self._subject_management_add, "add")
        self.ribbon_edit_subject_action = _add_toolbar_action("ribbon_edit_subject_action", "Sửa môn", self._subject_management_edit, "edit")
        self.ribbon_delete_subject_action = _add_toolbar_action("ribbon_delete_subject_action", "Xoá môn", self._subject_management_delete, "delete")
        self.ribbon_save_subject_action = _add_toolbar_action("ribbon_save_subject_action", "Lưu danh mục", self._save_subject_management, "save")

        self.ribbon_new_template_action = _add_toolbar_action("ribbon_new_template_action", "Tạo mẫu", self._create_new_template, "template")
        self.ribbon_edit_template_action = _add_toolbar_action("ribbon_edit_template_action", "Sửa mẫu", self._edit_selected_template, "edit")
        self.ribbon_delete_template_action = _add_toolbar_action("ribbon_delete_template_action", "Xoá mẫu", self._delete_selected_template, "delete")
        self.ribbon_close_template_action = _add_toolbar_action("ribbon_close_template_action", "Đóng mẫu", self._close_template_module, "close")

        self._apply_branding_to_ribbon_actions()

    def manage_subjects(self) -> None:
        if not self._confirm_interrupt_active_workflows("Danh sách môn thi"):
            return
        self._refresh_subject_management_tables()
        self._set_subject_management_mode("subjects")
        self._navigate_to("subject_management", context={"session_id": self.current_session_id}, push_current=True, require_confirm=False, reason="manage_subjects")

    def action_open_current_exam_subjects(self) -> None:
        if not self._confirm_interrupt_active_workflows("Danh sách môn thi"):
            return
        if self._open_current_session_in_exam_editor():
            return
        QMessageBox.information(
            self,
            "Danh sách môn thi",
            "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.",
        )

    def _set_actions_visible(self, actions: list[QAction | None], visible: bool) -> None:
        for action in actions:
            if action is not None:
                action.setVisible(bool(visible))

    def _handle_stack_changed(self, index: int) -> None:
        route_name = self._stack_index_to_route_name(index)
        self._current_route_name = route_name

        subject_management_visible = index == 2
        template_library_visible = index == 3
        template_editor_visible = index == 4
        exam_editor_visible = index == 5
        batch_scan_visible = route_name == "workspace_batch_scan"
        template_visible = template_library_visible or template_editor_visible

        general_visible = not subject_management_visible and not template_visible
        batch_context_visible = bool(batch_scan_visible and general_visible)
        exam_context_visible = bool(exam_editor_visible and general_visible)
        subject_context_visible = bool(subject_management_visible)
        template_context_visible = bool(template_library_visible)

        general_actions = [
            getattr(self, "ribbon_new_exam_action", None),
            getattr(self, "ribbon_view_exam_action", None),
            getattr(self, "ribbon_subject_list_action", None),
            getattr(self, "ribbon_batch_scan_action", None),
            getattr(self, "ribbon_scoring_action", None),
            getattr(self, "ribbon_recheck_action", None),
            getattr(self, "ribbon_export_action", None),
        ]
        batch_actions = [
            getattr(self, "ribbon_batch_execute_action", None),
            getattr(self, "ribbon_batch_save_action", None),
            getattr(self, "ribbon_batch_close_action", None),
        ]
        exam_editor_actions = [
            getattr(self, "ribbon_exam_editor_add_subject_action", None),
            getattr(self, "ribbon_exam_editor_edit_subject_action", None),
            getattr(self, "ribbon_exam_editor_delete_subject_action", None),
            getattr(self, "ribbon_exam_editor_save_action", None),
            getattr(self, "ribbon_exam_editor_close_action", None),
        ]
        subject_management_actions = [
            getattr(self, "ribbon_add_subject_action", None),
            getattr(self, "ribbon_edit_subject_action", None),
            getattr(self, "ribbon_delete_subject_action", None),
            getattr(self, "ribbon_save_subject_action", None),
        ]
        template_library_actions = [
            getattr(self, "ribbon_new_template_action", None),
            getattr(self, "ribbon_edit_template_action", None),
            getattr(self, "ribbon_delete_template_action", None),
            getattr(self, "ribbon_close_template_action", None),
        ]

        self._set_actions_visible(general_actions, general_visible)
        self._set_actions_visible(batch_actions, batch_context_visible)
        self._set_actions_visible(exam_editor_actions, exam_context_visible)
        self._set_actions_visible(subject_management_actions, subject_context_visible)
        self._set_actions_visible(template_library_actions, template_context_visible)

        context_visible = batch_context_visible or exam_context_visible or subject_context_visible or template_context_visible
        if getattr(self, "ribbon_context_separator", None) is not None:
            self.ribbon_context_separator.setVisible(context_visible and index != 4)

        if hasattr(self, "main_ribbon"):
            self.main_ribbon.setVisible(index != 4)

        if hasattr(self, "template_module_menu"):
            self.template_module_menu.menuAction().setVisible(template_visible)
            self._rebuild_template_module_menu(library_mode=template_library_visible, editor_mode=template_editor_visible)
        if hasattr(self, "act_close_template_module"):
            self.act_close_template_module.setVisible(template_visible)

        if hasattr(self, "act_save_as_subject"):
            row = self.exam_list_table.currentRow() if hasattr(self, "exam_list_table") else -1
            sid = self._session_id_for_row(row) if row >= 0 else ""
            can_save_as = (index == 0) and bool(str(sid or "").strip())
            self.act_save_as_subject.setVisible(True)
            self.act_save_as_subject.setEnabled(can_save_as)

        self._refresh_ribbon_action_states()

    def _refresh_ribbon_action_states(self) -> None:
        has_current_session = bool(str(getattr(self, "current_session_id", "") or "").strip())
        has_session = has_current_session
        has_subject_cfg = bool(self._effective_subject_configs_for_batch())
        route_name = getattr(self, "_current_route_name", "")
        batch_scan_visible = route_name == "workspace_batch_scan"

        def set_enabled(attr_name: str, enabled: bool) -> None:
            action = getattr(self, attr_name, None)
            if action is not None:
                action.setEnabled(bool(enabled))

        # Main exam actions
        set_enabled("ribbon_new_exam_action", True)
        set_enabled("ribbon_view_exam_action", True)
        set_enabled("act_new_session", True)
        set_enabled("act_open_from_list", True)
        set_enabled("act_save_session", has_current_session)
        set_enabled("act_close_current_session", has_current_session)
        set_enabled("act_current_exam_subjects", has_current_session)
        set_enabled("ribbon_subject_list_action", has_current_session)

        # Configuration actions
        set_enabled("act_manage_subject", True)
        set_enabled("act_manage_template", True)
        set_enabled("act_import_answer_key", has_session)
        set_enabled("act_export_answer_key_sample", has_session)

        # Workflow actions
        can_open_batch = has_session and has_subject_cfg
        set_enabled("ribbon_batch_scan_action", can_open_batch)
        set_enabled("act_batch_scan_menu", can_open_batch)
        set_enabled("ribbon_scoring_action", has_session)
        set_enabled("act_calculate_scores", has_session)
        set_enabled("ribbon_recheck_action", has_session)
        set_enabled("act_open_recheck", has_session)

        can_execute_batch = bool(has_session and batch_scan_visible and not getattr(self, "_batch_scan_running", False))
        set_enabled("ribbon_batch_execute_action", can_execute_batch)
        set_enabled("act_execute_batch_scan", can_execute_batch)
        set_enabled("act_edit_selected_scan", bool(has_session and batch_scan_visible))

        if getattr(self, "ribbon_batch_save_action", None) is not None:
            save_enabled = bool(hasattr(self, "btn_save_batch_subject") and self.btn_save_batch_subject.isEnabled())
            self.ribbon_batch_save_action.setEnabled(save_enabled)
        set_enabled("ribbon_batch_close_action", True)

        has_embedded_exam_editor = bool(self.embedded_exam_dialog is not None)
        for attr_name in [
            "ribbon_exam_editor_add_subject_action",
            "ribbon_exam_editor_edit_subject_action",
            "ribbon_exam_editor_delete_subject_action",
            "ribbon_exam_editor_save_action",
            "ribbon_exam_editor_close_action",
        ]:
            set_enabled(attr_name, has_embedded_exam_editor)

        for attr_name in [
            "ribbon_add_subject_action",
            "ribbon_edit_subject_action",
            "ribbon_delete_subject_action",
            "ribbon_save_subject_action",
            "ribbon_new_template_action",
            "ribbon_edit_template_action",
            "ribbon_delete_template_action",
            "ribbon_close_template_action",
        ]:
            set_enabled(attr_name, True)

        has_export_data = self._has_exportable_data()
        set_enabled("ribbon_export_action", has_session)
        self._refresh_export_action_states(has_session=has_session, has_export_data=has_export_data)

    def _exam_editor_add_subject(self) -> None:
        if self.embedded_exam_dialog is not None:
            self.embedded_exam_dialog._add_subject()
            self._refresh_ribbon_action_states()

    def _exam_editor_edit_subject(self) -> None:
        if self.embedded_exam_dialog is not None:
            self.embedded_exam_dialog._edit_subject()
            self._refresh_ribbon_action_states()

    def _exam_editor_delete_subject(self) -> None:
        if self.embedded_exam_dialog is not None:
            self.embedded_exam_dialog._delete_subject()
            self._refresh_ribbon_action_states()

    def _exam_editor_save(self) -> None:
        if self.embedded_exam_dialog is not None:
            self.embedded_exam_dialog._validate_and_accept()
            self._refresh_ribbon_action_states()

    def _exam_editor_close(self) -> None:
        if self.embedded_exam_dialog is not None:
            self.embedded_exam_dialog.reject()
            self._refresh_ribbon_action_states()

    def _route_to_stack_index(self, route_name: str) -> int:
        mapping = {
            "exam_list": 0,
            "workspace_batch_scan": 1,
            "workspace_scoring": 1,
            "subject_management": 2,
            "template_library": 3,
            "template_editor": 4,
            "exam_editor": 5,
        }
        return int(mapping.get(str(route_name or "").strip(), 0))

    def _stack_index_to_route_name(self, index: int) -> str:
        if int(index) == 1 and hasattr(self, "scoring_panel") and self.scoring_panel.isVisible():
            return "workspace_scoring"
        mapping = {0: "exam_list", 1: "workspace_batch_scan", 2: "subject_management", 3: "template_library", 4: "template_editor", 5: "exam_editor"}
        return str(mapping.get(int(index), "exam_list"))

    def _push_route_history(self, name: str, context: dict | None = None) -> None:
        if self._suspend_route_history_push:
            return
        entry = {"name": str(name or "exam_list"), "context": dict(context or {})}
        if self._route_history and self._route_history[-1] == entry:
            return
        self._route_history.append(entry)

    def _navigate_to(
        self,
        route_name: str,
        *,
        context: dict | None = None,
        push_current: bool = True,
        require_confirm: bool = True,
        reason: str = "",
    ) -> bool:
        # route-based navigation + confirm before destructive/switch action.
        target = str(route_name or "exam_list").strip() or "exam_list"
        if require_confirm and target != self._current_route_name:
            if not self._confirm_before_switching_work(reason or target):
                return False
        if target == "exam_list" and getattr(self, "current_session_id", None):
            self.close_current_session()
            return True

        if push_current:
            self._push_route_history(self._current_route_name, self._current_route_context)
        self._current_route_name = target
        self._current_route_context = dict(context or {})
        self.stack.setCurrentIndex(self._route_to_stack_index(target))
        if target == "workspace_batch_scan":
            self._show_batch_scan_panel()
        elif target == "workspace_scoring":
            self._show_scoring_panel()
        return True

    def _navigate_back(self, default_route: str = "exam_list") -> None:
        if self._route_history:
            target = self._route_history.pop()
        else:
            parent_map = {
                "subject_management": "exam_list",
                "template_library": "exam_list",
                "template_editor": "template_library",
                "exam_editor": "exam_list",
                "workspace_scoring": "workspace_batch_scan",
                "workspace_batch_scan": "exam_editor" if self.current_session_id else "exam_list",
            }
            fallback_name = parent_map.get(self._current_route_name, default_route)
            target = {"name": fallback_name, "context": {"session_id": self.current_session_id} if self.current_session_id else {}}
        self._navigate_to(
            str(target.get("name", default_route) or default_route),
            context=dict(target.get("context", {}) or {}),
            push_current=False,
            require_confirm=False,
            reason="back",
        )

    def action_manage_subjects(self) -> None:
        self.manage_subjects()

    def action_exit(self) -> None:
        self.close()
