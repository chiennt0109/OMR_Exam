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


class MainWindowBatchUiMixin:
    """Batch Scan panel widgets, progress UI, grid rendering, filters and preview panel."""
    def _begin_scan_grid_update(self) -> None:
        if not hasattr(self, "scan_list"):
            return
        self._scan_grid_loading = True
        self.scan_list.setUpdatesEnabled(False)
        self.scan_list.blockSignals(True)
        try:
            self.scan_list.setSortingEnabled(False)
        except Exception:
            pass

    def _end_scan_grid_update(self) -> None:
        if not hasattr(self, "scan_list"):
            self._scan_grid_loading = False
            return
        try:
            self.scan_list.setSortingEnabled(False)
        except Exception:
            pass
        self.scan_list.blockSignals(False)
        self.scan_list.setUpdatesEnabled(True)
        self._scan_grid_loading = False

    def _open_wait_progress(self, label_text: str, title: str = "Đang xử lý...") -> QProgressDialog:
        dlg = QProgressDialog(label_text, "", 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()
        return dlg

    @staticmethod
    def _close_wait_progress(dlg: QProgressDialog | None) -> None:
        if dlg is None:
            return
        try:
            dlg.close()
            dlg.deleteLater()
        except Exception:
            pass

    def _open_batch_progress_screen(self, total_items: int, title: str = "Đang nhận dạng Batch Scan") -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setModal(True)
        lay = QVBoxLayout(dlg)
        lbl_title = QLabel("Đang nhận dạng bài thi...")
        lbl_total = QLabel(f"Tổng số bài cần nhận dạng: {max(0, int(total_items or 0))}")
        lbl_current = QLabel("Đang nhận dạng bài thứ: 0/0")
        lbl_eta = QLabel("Thời gian còn lại ước tính: -")
        prog = QProgressBar(dlg)
        prog.setMinimum(0)
        prog.setMaximum(max(1, int(total_items or 0)))
        prog.setValue(0)
        prog.setFormat("%v/%m bài")
        lay.addWidget(lbl_title)
        lay.addWidget(lbl_total)
        lay.addWidget(lbl_current)
        lay.addWidget(lbl_eta)
        lay.addWidget(prog)
        setattr(dlg, "_batch_lbl_total", lbl_total)
        setattr(dlg, "_batch_lbl_current", lbl_current)
        setattr(dlg, "_batch_lbl_eta", lbl_eta)
        setattr(dlg, "_batch_progress", prog)
        dlg.resize(520, 180)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def _update_batch_progress_screen(self, dlg: QDialog | None, current: int, total: int, image_path: str, started_at: float) -> None:
        if dlg is None:
            return
        current_safe = max(0, int(current or 0))
        total_safe = max(1, int(total or 1))
        lbl_total = getattr(dlg, "_batch_lbl_total", None)
        lbl_current = getattr(dlg, "_batch_lbl_current", None)
        lbl_eta = getattr(dlg, "_batch_lbl_eta", None)
        prog = getattr(dlg, "_batch_progress", None)
        if isinstance(lbl_total, QLabel):
            lbl_total.setText(f"Tổng số bài cần nhận dạng: {total_safe}")
        if isinstance(lbl_current, QLabel):
            lbl_current.setText(f"Đang nhận dạng bài thứ: {min(current_safe, total_safe)}/{total_safe} - {Path(str(image_path or '')).name or '-'}")
        elapsed = max(0.0, float(time.perf_counter() - float(started_at or 0.0)))
        remain_items = max(0, total_safe - current_safe)
        eta_sec = (elapsed / current_safe) * remain_items if current_safe > 0 else 0.0
        if isinstance(lbl_eta, QLabel):
            lbl_eta.setText(f"Thời gian còn lại ước tính: {self._format_eta_text(eta_sec)}")
        if isinstance(prog, QProgressBar):
            prog.setMaximum(total_safe)
            prog.setValue(min(current_safe, total_safe))
        QApplication.processEvents()

    @staticmethod
    def _close_batch_progress_screen(dlg: QDialog | None) -> None:
        if dlg is None:
            return
        try:
            dlg.close()
            dlg.deleteLater()
        except Exception:
            pass

    def _set_scan_action_widget(self, row: int) -> None:
        if row < 0 or row >= self.scan_list.rowCount():
            return
        style = self.style()
        holder = QWidget()
        lay = QHBoxLayout(holder)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(2)
        btn_edit = self._make_row_icon_button(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Sửa bài thi", lambda _=False, r=row: self._edit_scan_row_by_index(r))
        btn_delete = self._make_row_icon_button(style.standardIcon(QStyle.SP_TrashIcon), "Xoá bài thi", lambda _=False, r=row: self._delete_scan_row_by_index(r))
        lay.addWidget(btn_edit)
        lay.addWidget(btn_delete)
        self.scan_list.setCellWidget(row, self.SCAN_COL_ACTIONS, holder)

    def _ensure_scan_action_widget(self, row: int) -> None:
        if row < 0 or row >= self.scan_list.rowCount():
            return
        if self.scan_list.cellWidget(row, self.SCAN_COL_ACTIONS) is not None:
            return
        self._set_scan_action_widget(row)

    def _open_scan_row_context_menu(self, pos) -> None:
        if not hasattr(self, "scan_list"):
            return
        row = self.scan_list.rowAt(pos.y())
        if row < 0 or row >= self.scan_list.rowCount():
            return
        self.scan_list.selectRow(row)
        menu = QMenu(self)
        act_edit = menu.addAction("Sửa bài thi")
        act_delete = menu.addAction("Xoá bài thi")
        act_rerecognize = menu.addAction("Nhận dạng lại bài thi")
        chosen = menu.exec(self.scan_list.viewport().mapToGlobal(pos))
        if chosen == act_edit:
            self._edit_scan_row_by_index(row)
            return
        if chosen == act_delete:
            self._delete_scan_row_by_index(row, require_confirm=True)
            return
        if chosen == act_rerecognize:
            self.scan_list.selectRow(row)
            self._on_scan_selected()
            self._rerecognize_selected_scan()

    def _edit_scan_row_by_index(self, row: int) -> None:
        if row < 0 or row >= self.scan_list.rowCount():
            return
        self.scan_list.selectRow(row)
        self._on_scan_selected()
        self._open_edit_selected_scan()

    def _delete_scan_row_by_index(self, row: int, require_confirm: bool = False) -> None:
        self._ensure_correction_state()
        if row < 0 or row >= self.scan_list.rowCount():
            return
        self.scan_list.selectRow(row)
        if self.correction_save_timer.isActive():
            self.correction_save_timer.stop()
            self._flush_pending_correction_updates()

        idx = self.scan_list.currentRow()
        sid_item = self.scan_list.item(idx, self.SCAN_COL_STUDENT_ID) if 0 <= idx < self.scan_list.rowCount() else None
        image_path = str(sid_item.data(Qt.UserRole) if sid_item else "").strip()
        sid_for_score = str(sid_item.text() if sid_item else "").strip()
        if not image_path:
            result = self.scan_results[idx] if 0 <= idx < len(self.scan_results) else self._build_result_from_saved_table_row(idx)
            if result is None:
                return
            image_path = str(getattr(result, "image_path", "") or "").strip()
            sid_for_score = str(getattr(result, "student_id", "") or sid_for_score).strip()
        if not image_path:
            return
        confirm_message = f"Bạn có chắc muốn xoá bản ghi này?\n\nẢnh: {Path(image_path).name}"
        if require_confirm:
            confirm_message = (
                "Bạn có chắc muốn xoá bài thi đã chọn?\n\n"
                f"Ảnh: {Path(image_path).name}"
            )
        confirm = QMessageBox.question(
            self,
            "Xoá bài thi",
            confirm_message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        wait_dlg = self._open_wait_progress("Đang xoá bản ghi và cập nhật danh sách...")
        subject_key = self._current_batch_subject_key()
        self._mark_deleted_scan_image(subject_key, image_path)
        if sid_for_score and sid_for_score != "-":
            self._invalidate_scoring_for_student_ids([sid_for_score], subject_key=subject_key, reason="delete_scan_row")
        scoped_subject = self._batch_result_subject_key(subject_key)
        candidate_subject_keys: list[str] = [scoped_subject]
        sid = str(self.current_session_id or "").strip()
        legacy_scoped = f"{sid}::{subject_key}" if sid and subject_key else ""
        if legacy_scoped and legacy_scoped not in candidate_subject_keys:
            candidate_subject_keys.append(legacy_scoped)
        if subject_key and subject_key not in candidate_subject_keys:
            candidate_subject_keys.append(subject_key)
        try:
            QApplication.processEvents()
            # Keep direct delete by current subject key for compatibility with legacy flows/tests.
            self.database.delete_scan_result(subject_key, image_path)
            for key in candidate_subject_keys:
                self.database.delete_scan_result(key, image_path)
            self.scan_results = [x for x in self._refresh_scan_results_from_db(subject_key) if str(getattr(x, "image_path", "") or "") != ""]
            self.scan_results_by_subject[scoped_subject] = list(self.scan_results)
            self._populate_scan_grid_from_results(self.scan_results)
            self._rebuild_error_list()
            self._refresh_all_statuses()
            self._update_batch_scan_bottom_status_text()
            if self.scan_list.rowCount() > 0:
                target_row = min(row, self.scan_list.rowCount() - 1)
                self.scan_list.selectRow(target_row)
                self._on_scan_selected()
            else:
                self.scan_result_preview.setRowCount(0)
                self.result_preview.clear()
                self.manual_edit.clear()
                self.scan_image_preview.setText("Chọn bài thi ở danh sách bên trái")
                self.scan_image_preview.clear_markers()
        finally:
            self._close_wait_progress(wait_dlg)

    def _start_batch_scan_from_ui(self) -> None:
        if not self._ensure_current_session_loaded():
            QMessageBox.warning(self, "Batch Scan", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        cfgs = self._effective_subject_configs_for_batch()
        if not cfgs:
            QMessageBox.warning(self, "Batch Scan", "Kỳ thi hiện tại chưa có môn thi để nhận dạng.")
            return
        if self.stack.currentIndex() != 1:
            self._navigate_to(
                "workspace_batch_scan",
                context={"session_id": self.current_session_id, "origin": self._current_route_name},
                push_current=True,
                require_confirm=False,
                reason="open_batch_scan",
            )
        self._refresh_batch_subject_controls()
        self._show_batch_scan_panel()
        if hasattr(self, "batch_subject_combo") and self.batch_subject_combo.currentIndex() <= 0 and self.batch_subject_combo.count() > 1:
            self.batch_subject_combo.setCurrentIndex(1)

    def action_run_batch_scan(self) -> None:
        self._start_batch_scan_from_ui()
        if hasattr(self, "batch_subject_combo") and self.batch_subject_combo.currentIndex() > 0:
            has_unsaved = bool(hasattr(self, "btn_save_batch_subject") and self.btn_save_batch_subject.isEnabled())
            if not has_unsaved:
                self._on_batch_subject_changed(self.batch_subject_combo.currentIndex(), force_reload=True)
        self._refresh_ribbon_action_states()

    def action_execute_batch_scan(self) -> None:
        if self.stack.currentIndex() != 1:
            self._start_batch_scan_from_ui()
            return
        self.run_batch_scan()

    def action_edit_selected_scan(self) -> None:
        self._open_edit_selected_scan()

    def action_load_selected_scan_result(self) -> None:
        self._load_selected_result_for_correction()

    def _build_session_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.session_info = QTextEdit()
        self.session_info.setReadOnly(True)
        self.exam_code_preview = QLabel("Mã đề trên phiếu trả lời mẫu: -")
        self.exam_code_preview.setWordWrap(True)

        layout.addWidget(self.session_info)
        layout.addWidget(self.exam_code_preview)
        return w

    def _build_scan_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        batch_group = QGroupBox("Nhận dạng theo môn đã cấu hình")
        batch_form = QFormLayout(batch_group)
        self.batch_subject_combo = QComboBox()
        self.batch_subject_combo.currentIndexChanged.connect(self._on_batch_subject_changed)
        self.batch_file_scope_combo = QComboBox()
        self.batch_file_scope_combo.addItem("Nhận dạng file mới", "new_only")
        self.batch_file_scope_combo.addItem("Nhận dạng toàn bộ", "all")
        self.batch_file_scope_combo.currentIndexChanged.connect(lambda _=0: self._update_batch_scan_scope_summary())
        self.batch_recognition_mode_combo = QComboBox()
        self.batch_recognition_mode_combo.addItem("Tự động (khuyến nghị)", "auto")
        self.batch_recognition_mode_combo.addItem("Mẫu cũ / Anchor chuẩn", "legacy")
        self.batch_recognition_mode_combo.addItem("Mẫu mới / Anchor sát biên", "border")
        self.batch_recognition_mode_combo.addItem("Anchor 1 phía (ruler theo dòng)", "one_side")
        self.batch_recognition_mode_combo.addItem("Lai (thử nhiều cơ chế)", "hybrid")
        self.batch_recognition_mode_combo.setCurrentIndex(0)
        self.batch_recognition_mode_combo.setVisible(False)
        self.batch_api_file_value = QLineEdit("-"); self.batch_api_file_value.setReadOnly(True)
        self.btn_pick_batch_api_file = QPushButton("Chọn file API")
        self.btn_pick_batch_api_file.clicked.connect(self._pick_batch_api_file)
        api_row = QHBoxLayout()
        api_row.addWidget(self.batch_api_file_value, 1)
        api_row.addWidget(self.btn_pick_batch_api_file)
        self.batch_template_value = QLineEdit("-"); self.batch_template_value.setReadOnly(True)
        self.batch_template_path_value = "-"
        self.batch_answer_codes_value = QLineEdit("-"); self.batch_answer_codes_value.setReadOnly(True)
        self.batch_student_id_value = QLineEdit("-"); self.batch_student_id_value.setReadOnly(True)
        self.batch_scan_folder_value = QLineEdit("-"); self.batch_scan_folder_value.setReadOnly(True)
        self.batch_scan_state_value = QLineEdit("-"); self.batch_scan_state_value.setReadOnly(True)
        self.batch_context_value = QLineEdit("-"); self.batch_context_value.setReadOnly(True)
        style = self.style()
        self.btn_batch_recognize = QPushButton("Nhận dạng")
        self.btn_batch_recognize.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.btn_batch_recognize.clicked.connect(self.action_execute_batch_scan)
        self.btn_save_batch_subject = QPushButton("Lưu")
        self.btn_save_batch_subject.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        self.btn_save_batch_subject.clicked.connect(self._save_batch_for_selected_subject)
        self.btn_save_batch_subject.setEnabled(False)
        self.btn_close_batch_view = QPushButton("Đóng")
        self.btn_close_batch_view.setIcon(style.standardIcon(QStyle.SP_DialogCloseButton))
        self.btn_close_batch_view.clicked.connect(self._close_batch_scan_view)
        for b in [self.btn_batch_recognize, self.btn_save_batch_subject, self.btn_close_batch_view]:
            b.setMaximumWidth(140)
            b.setVisible(False)

        batch_form.addRow("Môn", self.batch_subject_combo)
        batch_form.addRow("Phạm vi", self.batch_file_scope_combo)
        batch_form.addRow("API bài thi", api_row)
        batch_form.addRow("Mẫu giấy dùng", self.batch_template_value)
        batch_form.addRow("Mã đề", self.batch_answer_codes_value)
        batch_form.addRow("Thư mục quét", self.batch_scan_folder_value)

        self.filter_column = QComboBox()
        self.filter_column.addItems(["Tất cả", "STT", "STUDENT ID", "Phòng thi", "Mã đề", "Họ tên", "Ngày sinh", "Nội dung", "Status"])
        self.filter_column.currentTextChanged.connect(self._apply_scan_filter)
        self.search_value = QLineEdit()
        self.search_value.setPlaceholderText("Tìm trong cột đã chọn hoặc toàn bảng")
        self._scan_filter_debounce_timer = QTimer(self)
        self._scan_filter_debounce_timer.setSingleShot(True)
        self._scan_filter_debounce_timer.setInterval(180)
        self._scan_filter_debounce_timer.timeout.connect(self._apply_scan_filter)
        self.search_value.textChanged.connect(self._schedule_scan_filter)

        search_row = QHBoxLayout()
        search_row.addWidget(self.filter_column)
        search_row.addWidget(self.search_value)

        self.scan_list = QTableWidget(0, 9)
        self.scan_list.setHorizontalHeaderLabels(["STT", "STUDENT ID", "Phòng thi", "Mã đề", "Họ tên", "Ngày sinh", "Nội dung", "Status", "Chức năng"])
        self.scan_list.verticalHeader().setVisible(False)
        self.scan_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.scan_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scan_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.scan_list.customContextMenuRequested.connect(self._open_scan_row_context_menu)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_STT, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_STUDENT_ID, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_EXAM_ROOM, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_EXAM_CODE, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_FULL_NAME, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_BIRTH_DATE, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_CONTENT, QHeaderView.Stretch)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_STATUS, QHeaderView.Stretch)
        self.scan_list.horizontalHeader().setSectionResizeMode(self.SCAN_COL_ACTIONS, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().sectionClicked.connect(self._on_scan_header_clicked)
        self.scan_list.itemSelectionChanged.connect(self._on_scan_selected)
        # double click navigation: open selected scan directly.
        self.scan_list.cellDoubleClicked.connect(self._handle_scan_list_double_click)
        self.scan_list.cellClicked.connect(self._on_scan_cell_clicked)
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        self.scan_image_preview = PreviewImageWidget(); self.scan_image_preview.setText("Chọn bài thi ở danh sách bên trái")
        self.scan_image_scroll = QScrollArea()
        self.scan_image_scroll.setWidgetResizable(False)
        self.scan_image_scroll.setAlignment(Qt.AlignCenter)
        self.scan_image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scan_image_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scan_image_scroll.setWidget(self.scan_image_preview)
        self.scan_image_scroll.viewport().installEventFilter(self)

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setMaximumWidth(36)
        self.btn_zoom_out.clicked.connect(self._zoom_preview_out)
        self.btn_zoom_reset = QPushButton("30%")
        self.btn_zoom_reset.setMaximumWidth(60)
        self.btn_zoom_reset.clicked.connect(self._zoom_preview_reset)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setMaximumWidth(36)
        self.btn_zoom_in.clicked.connect(self._zoom_preview_in)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_reset)
        zoom_row.addWidget(self.btn_zoom_in)
        self.btn_rotate_left = QPushButton("⟲ 90°")
        self.btn_rotate_left.setToolTip("Xoay trái ảnh đang chọn")
        self.btn_rotate_left.clicked.connect(lambda: self._rotate_selected_scan(-90))
        self.btn_rotate_right = QPushButton("⟳ 90°")
        self.btn_rotate_right.setToolTip("Xoay phải ảnh đang chọn")
        self.btn_rotate_right.clicked.connect(lambda: self._rotate_selected_scan(90))
        self.btn_rerecognize_selected = QPushButton("Nhận dạng lại ảnh chọn")
        self.btn_rerecognize_selected.clicked.connect(self._rerecognize_selected_scan)
        zoom_row.addWidget(self.btn_rotate_left)
        zoom_row.addWidget(self.btn_rotate_right)
        zoom_row.addWidget(self.btn_rerecognize_selected)
        zoom_row.addStretch()

        self.scan_result_preview = QTableWidget(0, 2)
        self.scan_result_preview.setHorizontalHeaderLabels(["Mục nhận dạng", "Kết quả"])
        self.scan_result_preview.verticalHeader().setVisible(False)
        self.scan_result_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scan_result_preview.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.scan_result_preview.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.addWidget(batch_group)
        left_l.addLayout(search_row)
        left_l.addWidget(self.scan_list)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.addLayout(zoom_row)
        right_l.addWidget(self.scan_image_scroll, 7)
        right_l.addWidget(self.scan_result_preview, 3)

        self.scan_lr_split = QSplitter(Qt.Horizontal)
        self.scan_lr_split.addWidget(left)
        self.scan_lr_split.addWidget(right)
        self.scan_lr_split.setStretchFactor(0, 68)
        self.scan_lr_split.setStretchFactor(1, 32)
        self.scan_lr_split.setSizes([680, 320])
        self.batch_scan_status_bottom = QLabel("Trạng thái file: - | Lọc: 0/0")
        self.batch_scan_status_bottom.setWordWrap(False)
        self.batch_scan_status_bottom.setFixedHeight(22)
        self.batch_scan_status_bottom.setStyleSheet("QLabel { padding: 2px 8px; color: #444; }")
        self.batch_status_filter_mode = "all"
        self.batch_scan_status_bottom.setOpenExternalLinks(False)
        self.batch_scan_status_bottom.linkActivated.connect(self._handle_batch_status_filter_link)

        # Create scoring widgets with explicit parent to avoid lifecycle issues
        # on some PySide6 builds (preventing "Internal C++ object ... already deleted").
        self.scoring_panel = QWidget(w)

        self.score_preview_table = QTableWidget(0, 13, self.scoring_panel)
        self.score_preview_table.setHorizontalHeaderLabels([
            "Student ID", "Name", "Lớp", "Ngày sinh", "Exam Code", "MCQ đúng", "TF đúng", "NUM đúng", "Correct", "Wrong", "Blank", "Score", "Trạng thái"
        ])
        self.score_preview_table.verticalHeader().setVisible(False)
        self.score_preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.score_preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.score_preview_table.horizontalHeader().setSectionResizeMode(12, QHeaderView.Stretch)
        self.score_preview_table.cellDoubleClicked.connect(self._open_scoring_review_editor_from_table)

        self.scoring_subject_combo = QComboBox()
        self.scoring_subject_combo.currentIndexChanged.connect(self._handle_scoring_subject_changed)
        self.scoring_mode_combo = QComboBox()
        self.scoring_mode_combo.addItems(["Tính lại toàn bộ", "Chỉ tính bài chưa có điểm"])
        self.scoring_phase_note = QLineEdit()
        self.scoring_phase_note.setPlaceholderText("Ghi chú pha chấm điểm (tuỳ chọn)")
        self.scoring_phase_note.setVisible(False)
        self.scoring_filter_column = QComboBox()
        self.scoring_filter_column.addItems(["Tất cả", "Student ID", "Name", "Lớp", "Ngày sinh", "Exam Code", "Trạng thái"])
        self.scoring_filter_column.currentTextChanged.connect(self._apply_scoring_filter)
        self.scoring_search_value = QLineEdit()
        self.scoring_search_value.setPlaceholderText("Tìm kiếm trên lưới Tính điểm")
        self.scoring_search_value.textChanged.connect(self._apply_scoring_filter)
        self.btn_scoring_run = QPushButton("Chấm điểm")
        self.btn_scoring_run.clicked.connect(self._run_scoring_from_panel)
        self.btn_scoring_save = QPushButton("Lưu điểm")
        self.btn_scoring_save.clicked.connect(self._save_current_work)
        self.btn_scoring_save.setEnabled(False)
        self.btn_scoring_back = QPushButton("Quay lại Batch Scan")
        self.btn_scoring_back.clicked.connect(self._back_to_batch_scan)
        scoring_top = QHBoxLayout()
        scoring_top.addWidget(QLabel("Môn"))
        scoring_top.addWidget(self.scoring_subject_combo, 2)
        scoring_top.addWidget(QLabel("Cơ chế"))
        scoring_top.addWidget(self.scoring_mode_combo, 2)
        scoring_top.addWidget(self.scoring_filter_column, 2)
        scoring_top.addWidget(self.scoring_search_value, 3)
        scoring_top.addWidget(self.btn_scoring_run)
        scoring_top.addWidget(self.btn_scoring_save)
        scoring_top.addWidget(self.btn_scoring_back)

        self.scoring_phase_table = QTableWidget(0, 5, self.scoring_panel)
        self.scoring_phase_table.setHorizontalHeaderLabels(["Thời gian", "Môn", "Cơ chế", "Số bài", "Ghi chú"])
        self.scoring_phase_table.verticalHeader().setVisible(False)
        self.scoring_phase_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scoring_phase_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.scoring_phase_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.scoring_phase_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.scoring_phase_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.scoring_phase_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)

        scoring_panel_layout = QVBoxLayout(self.scoring_panel)
        scoring_panel_layout.setContentsMargins(0, 0, 0, 0)
        scoring_panel_layout.addLayout(scoring_top)
        scoring_panel_layout.addWidget(self.score_preview_table, 7)
        self.scoring_status_bar = QLabel("Thống kê chấm điểm: Thành công 0 | Lỗi 0 | Đã sửa 0")
        self.scoring_status_bar.setStyleSheet("QLabel { padding: 4px 8px; background: #f4f6f8; border: 1px solid #d0d7de; }")
        self.scoring_status_filter_mode = "all"
        self.scoring_status_bar.setTextFormat(Qt.RichText)
        self.scoring_status_bar.setOpenExternalLinks(False)
        self.scoring_status_bar.linkActivated.connect(self._handle_scoring_status_filter_link)
        scoring_panel_layout.addWidget(self.scoring_status_bar)

        layout.addWidget(self.progress)
        layout.addWidget(self.scan_lr_split)
        layout.addWidget(self.batch_scan_status_bottom)
        layout.addWidget(self.scoring_panel)
        self.scoring_panel.setVisible(False)
        return w

    def _close_batch_scan_view(self) -> None:
        if self._has_batch_unsaved_changes():
            choice = self._prompt_save_changes_word_style(
                "Batch Scan chưa lưu",
                "Bạn có thay đổi chưa lưu cho môn hiện tại. Bạn muốn lưu trước khi đóng không?",
            )
            if choice == "cancel":
                return
            if choice == "save" and not self._save_batch_for_selected_subject(
                show_success_message=False,
                reload_after_save=False,
                refresh_exam_list=False,
            ):
                return
        # On close, always route back to the current exam subject list view.
        self._return_to_current_exam_from_batch_scan()

    def _clear_batch_display_caches(self) -> None:
        for result in list(getattr(self, "scan_results", []) or []):
            for attr in ["cached_status", "cached_content", "cached_recognized_short", "cached_blank_summary", "cached_forced_status"]:
                try:
                    setattr(result, attr, "" if attr != "cached_blank_summary" else {})
                except Exception:
                    pass

    def _has_batch_unsaved_changes(self) -> bool:
        return bool(hasattr(self, "btn_save_batch_subject") and self.btn_save_batch_subject.isEnabled())

    def _open_current_session_in_exam_editor(self) -> bool:
        # return to current exam on batch close: open current runtime session, avoid stale batch return payload.
        if not self.current_session_id:
            return False
        if self.embedded_exam_dialog and str(self.embedded_exam_session_id or "").strip() == str(self.current_session_id or "").strip():
            self._navigate_to(
                "exam_editor",
                context={"session_id": self.current_session_id},
                push_current=False,
                require_confirm=False,
                reason="return_to_existing_exam_editor",
            )
            return True
        session = self.session if self.session is not None else None
        if session is None:
            payload = self.database.fetch_exam_session(self.current_session_id) or {}
            if not payload:
                return False
            try:
                session = ExamSession.from_dict(payload)
            except Exception:
                return False
        cfg = session.config or {}
        editor_payload = {
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
        self._open_embedded_exam_editor(self.current_session_id, session, editor_payload)
        return True

    def _return_to_current_exam_from_batch_scan(self) -> None:
        ctx = dict(self._current_route_context or {})
        origin = str(ctx.get("origin", "") or "")
        target = "exam_list"
        if self.embedded_exam_dialog and self.current_session_id:
            target = "exam_editor"
        elif self.current_session_id:
            target = "exam_editor"
        elif origin == "workspace_scoring":
            # keep deterministic fallback when context was opened from scoring but exam context is gone.
            target = "exam_list"
        if target == "exam_editor":
            if self.embedded_exam_dialog and str(self.embedded_exam_session_id or "").strip() == str(self.current_session_id or "").strip():
                self._navigate_to(
                    "exam_editor",
                    context={"session_id": self.current_session_id},
                    push_current=False,
                    require_confirm=False,
                    reason="return_existing_exam_editor",
                )
                return
            if self._open_current_session_in_exam_editor():
                return
        self._navigate_back(default_route="exam_list")

    def _strip_transient_scan_artifacts(self, result: OMRResult) -> OMRResult:
        for attr_name in [
            "aligned_image",
            "aligned_binary",
            "alignment_debug",
            "detected_anchors",
            "detected_digit_anchors",
            "bubble_states_by_zone",
            "digit_zone_debug",
        ]:
            if hasattr(result, attr_name):
                try:
                    delattr(result, attr_name)
                except Exception:
                    setattr(result, attr_name, None)
        return result

    def _lightweight_result_copy(self, result: OMRResult) -> OMRResult:
        return self._deserialize_omr_result(self._serialize_omr_result(result))

    def _refresh_dashboard_summary_from_db(self, subject_key: str) -> None:
        if not hasattr(self, "dashboard_summary_label"):
            return
        summary = self.database.dashboard_summary(subject_key)
        avg_score = float(summary.get("average_score", 0.0) or 0.0)
        distribution = summary.get("distribution", []) if isinstance(summary.get("distribution", []), list) else []
        top_students = summary.get("top_students", []) if isinstance(summary.get("top_students", []), list) else []
        dist_text = ", ".join(f"{item.get('bucket', 0)}: {item.get('count', 0)}" for item in distribution[:8]) or "-"
        top_text = ", ".join(f"{item.get('student_code', '-')}: {item.get('score', 0)}" for item in top_students[:5]) or "-"
        self.dashboard_summary_label.setText(
            f"Dashboard DB | Điểm TB: {avg_score:.2f} | Phổ điểm: {dist_text} | Top học sinh: {top_text}"
        )

    def _update_direct_score_import_row(self, subject_key: str, student_id: str, score_text: str, exam_room: str = "") -> bool:
        subject = str(subject_key or "").strip()
        sid = str(student_id or "").strip()
        if not subject or not sid:
            return False
        cfg = self._subject_config_by_subject_key(subject)
        if not isinstance(cfg, dict) or not self._subject_uses_direct_score_import(cfg):
            return False
        direct_payload = cfg.setdefault("direct_score_import", {})
        rows = direct_payload.setdefault("rows", [])
        if not isinstance(rows, list):
            rows = []
            direct_payload["rows"] = rows

        target_room = str(exam_room or "").strip()
        matched = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_sid = str(row.get("student_id", "") or "").strip()
            row_room = str(row.get("exam_room", "") or row.get("room", "") or "").strip()
            if row_sid != sid:
                continue
            if target_room and row_room and row_room != target_room:
                continue
            row["score"] = str(score_text or "").strip()
            row["manually_edited"] = True
            if target_room:
                row["exam_room"] = target_room
            matched = True
            break

        if not matched:
            rows.append({
                "student_id": sid,
                "score": str(score_text or "").strip(),
                "exam_room": target_room,
                "manually_edited": True,
            })

        cfg["direct_score_import"] = direct_payload
        subject_cfgs = list(self.subject_configs or [])
        idx = self._find_subject_config_index_for_batch_save(cfg, subject_cfgs)
        if idx >= 0:
            subject_cfgs[idx] = cfg
            self.subject_configs = subject_cfgs
        if self.session:
            self.session.config = {**(self.session.config or {}), "subject_configs": list(self.subject_configs or [])}
        self._persist_current_session_subject_configs(list(self.subject_configs or []))
        self._persist_runtime_session_state_quietly()
        return True

    def _open_essay_score_editor_from_table(self, row: int, subject_key: str) -> None:
        sid = str(self.score_preview_table.item(row, 0).text() if self.score_preview_table.item(row, 0) else "").strip()
        name = str(self.score_preview_table.item(row, 1).text() if self.score_preview_table.item(row, 1) else "").strip()
        exam_room = str(self.score_preview_table.item(row, 3).text() if self.score_preview_table.item(row, 3) else "").strip()
        current_score = str(self.score_preview_table.item(row, 6).text() if self.score_preview_table.item(row, 6) else "").strip()
        if not sid:
            return
        new_score, ok = QInputDialog.getText(
            self,
            "Sửa điểm tự luận",
            f"SBD: {sid}\nHọ tên: {name or '-'}\nPhòng thi: {exam_room or '-'}\n\nĐiểm:",
            text=current_score,
        )
        if not ok:
            return
        score_text = str(new_score or "").strip().replace(",", ".")
        if score_text == "":
            QMessageBox.warning(self, "Sửa điểm tự luận", "Điểm không được để trống.")
            return
        try:
            float(score_text)
        except Exception:
            QMessageBox.warning(self, "Sửa điểm tự luận", "Điểm không hợp lệ.")
            return
        if not self._update_direct_score_import_row(subject_key, sid, score_text, exam_room):
            QMessageBox.warning(self, "Sửa điểm tự luận", "Không cập nhật được dữ liệu điểm trực tiếp của môn.")
            return
        self.calculate_scores(subject_key, mode="Tính lại toàn bộ", note="essay_manual_edit")
        self._ensure_scoring_preview_current(subject_key, reason="essay_manual_edit", force=True)
        QMessageBox.information(self, "Sửa điểm tự luận", "Đã cập nhật điểm tự luận và đồng bộ lại bảng tính điểm.")

    def _sync_current_batch_subject_snapshot(self, persist_to_db: bool = True) -> tuple[str, list[OMRResult]]:
        # Helper này chỉ còn phục vụ thao tác explicit save của Batch Scan.
        # Khi persist_to_db=False, trả snapshot runtime để UI tham chiếu, không được ghi DB
        # và không được cập nhật nguồn dữ liệu lõi.
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg or not hasattr(self, "scan_list"):
            return "", []
        subject_key = self._subject_key_from_cfg(subject_cfg)
        if not subject_key:
            return "", []
        subject_db_key = self._batch_result_subject_key(subject_key)

        if self.scan_list.rowCount() <= 0:
            if persist_to_db:
                self.database.replace_scan_results_for_subject(subject_db_key, [])
                self.scan_results = []
                self.scan_results_by_subject[subject_db_key] = []
            return subject_key, []

        current_results = self._current_scan_results_snapshot()
        for result in current_results:
            result.answer_string = self._normalize_non_api_answer_string(result, subject_key)
            self._debug_scan_result_state("sync_snapshot", result)

        if not persist_to_db:
            return subject_key, list(current_results)

        self.scan_results = list(current_results)
        self.scan_results_by_subject[subject_db_key] = list(current_results)
        self.database.replace_scan_results_for_subject(
            subject_db_key,
            [self._serialize_omr_result(result) for result in current_results],
        )
        return subject_key, current_results

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "batch_scan_status_bottom"):
            self.batch_scan_status_bottom.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)
        self._current_route_name = "workspace_batch_scan"
        for action_name in ["ribbon_batch_execute_action", "ribbon_batch_save_action", "ribbon_batch_close_action"]:
            action = getattr(self, action_name, None)
            if action is not None:
                action.setVisible(True)
        self._refresh_ribbon_action_states()
        if hasattr(self, "batch_subject_combo") and self.batch_subject_combo.count() > 0:
            cfg = self._selected_batch_subject_config()
            if cfg:
                runtime_key = self._batch_runtime_key(cfg)
                need_reload = bool(self.scan_list.rowCount() <= 0 or runtime_key != self._batch_loaded_runtime_key)
                if need_reload:
                    self._load_batch_subject_state(cfg, source_hint="show_batch_panel", force_reload=False)
                else:
                    self._update_batch_scan_scope_summary()

    def _update_batch_scan_scope_summary(self) -> None:
        if not hasattr(self, "batch_scan_state_value"):
            return
        cfg = self._selected_batch_subject_config()
        if cfg:
            cfg = self._merge_saved_batch_snapshot(cfg)
        logical_key = self._logical_subject_key_from_cfg(cfg) if cfg else ""
        instance_key = self._subject_instance_key_from_cfg(cfg) if cfg else ""
        runtime_key = self._batch_runtime_key(instance_key) if instance_key else ""
        self.batch_scan_state_value.setText(self._current_batch_file_scope_text())
        self._update_batch_scan_bottom_status_text()
        if hasattr(self, "batch_context_value"):
            self.batch_context_value.setText(
                f"{self._display_subject_label(cfg)} | Logical: {logical_key or '-'} | "
                f"SubjectInstance: {instance_key or '-'} | Runtime: {runtime_key or '-'} | "
                f"Nguồn: {self._current_batch_data_source}"
            )

    def _current_batch_scan_status_metrics(self) -> dict[str, int]:
        metrics = {
            "total": 0,
            "visible": 0,
            "ok": 0,
            "error": 0,
            "duplicate": 0,
            "wrong_code": 0,
            "edited": 0,
            "edited_clean": 0,
            "edited_error": 0,
        }
        if hasattr(self, "scan_list"):
            total_rows = self.scan_list.rowCount()
            metrics["visible"] = sum(1 for row in range(total_rows) if not self.scan_list.isRowHidden(row))

        current_subject = str(self._current_batch_subject_key() or "").strip()
        subject_edit_registry = self._subject_edit_registry(current_subject) if current_subject else {}

        canonical_results: list[OMRResult] = []
        seen_images: set[str] = set()
        for res in list(self.scan_results or []):
            image_key = self._result_identity_key(getattr(res, "image_path", ""))
            if image_key and image_key in seen_images:
                continue
            if image_key:
                seen_images.add(image_key)
            if current_subject:
                self._apply_persisted_result_edit_metadata(current_subject, res)
            canonical_results.append(res)

        if not canonical_results and hasattr(self, "scan_list"):
            metrics["total"] = self.scan_list.rowCount()
            for row in range(self.scan_list.rowCount()):
                sid_item = self.scan_list.item(row, self.SCAN_COL_STUDENT_ID)
                payload = dict(sid_item.data(Qt.UserRole + 12) or {}) if sid_item is not None else {}
                flags = dict(payload.get("status_flags", {}) or {})
                status_text = str(payload.get("status", "") or "").strip().lower()
                image_key = self._row_image_key(row)
                row_history = self._ordered_unique_text_list(self.scan_edit_history.get(image_key, [])) if image_key else []
                if not row_history and image_key and current_subject:
                    row_history = self._ordered_unique_text_list((subject_edit_registry.get(image_key, {}) or {}).get("history", []))
                is_edited = bool(flags.get("edited", False)) or ("đã sửa" in status_text) or bool(row_history)
                has_error = bool(flags.get("has_error", False))
                has_duplicate = bool(flags.get("duplicate", False)) or ("trùng sbd" in status_text)
                has_wrong_code = bool(flags.get("wrong_code", False)) or ("mã đề không hợp lệ" in status_text)
                if is_edited:
                    metrics["edited"] += 1
                    if has_error:
                        metrics["edited_error"] += 1
                    else:
                        metrics["edited_clean"] += 1
                elif not has_error:
                    metrics["ok"] += 1
                if has_error:
                    metrics["error"] += 1
                if has_duplicate:
                    metrics["duplicate"] += 1
                if has_wrong_code:
                    metrics["wrong_code"] += 1
            return metrics

        duplicate_count_map: dict[str, int] = {}
        subject_scope = self._subject_student_room_scope()
        available_exam_codes = self._available_exam_codes()
        for res in canonical_results:
            sid = str(getattr(res, "student_id", "") or "").strip()
            if not self._student_id_has_recognition_error(sid):
                duplicate_count_map[sid] = duplicate_count_map.get(sid, 0) + 1

        metrics["total"] = len(canonical_results)
        for res in canonical_results:
            sid = str(getattr(res, "student_id", "") or "").strip()
            image_key = self._result_identity_key(getattr(res, "image_path", ""))
            duplicate_count = 0 if self._student_id_has_recognition_error(sid) else int(duplicate_count_map.get(sid, 0) or 0)
            analysis = self._analyze_scan_status(
                res,
                duplicate_count=duplicate_count,
                subject_scope=subject_scope,
                available_exam_codes=available_exam_codes,
                forced_status=str(getattr(res, "cached_forced_status", "") or ""),
            )
            row_history = self._ordered_unique_text_list(self.scan_edit_history.get(image_key, [])) if image_key else []
            if not row_history and image_key and current_subject:
                row_history = self._ordered_unique_text_list((subject_edit_registry.get(image_key, {}) or {}).get("history", []))
            is_edited = bool(analysis.get("is_manual_edited", False) or row_history)
            if is_edited:
                metrics["edited"] += 1
                if analysis["has_error"]:
                    metrics["edited_error"] += 1
                else:
                    metrics["edited_clean"] += 1
            elif analysis["is_clean_ok"]:
                metrics["ok"] += 1
            if analysis["has_error"]:
                metrics["error"] += 1
            if analysis["has_duplicate"]:
                metrics["duplicate"] += 1
            if analysis["has_wrong_code"]:
                metrics["wrong_code"] += 1
        return metrics

    def _update_batch_scan_bottom_status_text(self) -> None:
        if not hasattr(self, "batch_scan_status_bottom"):
            return
        file_status = str(self._current_batch_file_scope_text() or "-").strip() or "-"
        if hasattr(self, "batch_scan_state_value"):
            self.batch_scan_state_value.setText(file_status)
        metrics = self._current_batch_scan_status_metrics()
        total_rows = int(metrics.get("total", 0) or 0)
        visible_rows = int(metrics.get("visible", 0) or 0)
        ok_count = int(metrics.get("ok", 0) or 0)
        error_count = int(metrics.get("error", 0) or 0)
        duplicate_count = int(metrics.get("duplicate", 0) or 0)
        wrong_code_count = int(metrics.get("wrong_code", 0) or 0)
        edited_count = int(metrics.get("edited", 0) or 0)
        edited_clean_count = int(metrics.get("edited_clean", 0) or 0)
        edited_error_count = int(metrics.get("edited_error", 0) or 0)
        bar_text = f"Trạng thái file: {file_status} | Lọc: {visible_rows}/{total_rows}"
        bar_text += (
            " | "
            f"<a href='all'><b>Tất cả</b></a> | "
            f"<a href='error' style='color:#c62828;font-weight:600'>Lỗi hiện tại: {error_count}</a> | "
            f"<a href='duplicate' style='color:#6a1b9a;font-weight:600'>Trùng SBD: {duplicate_count}</a> | "
            f"<a href='wrong_code' style='color:#ef6c00;font-weight:600'>Sai mã đề: {wrong_code_count}</a> | "
            f"<a href='edited' style='color:#1565c0;font-weight:600'>Đã sửa: {edited_count}</a> "
            f"(hết lỗi: {edited_clean_count}, còn lỗi: {edited_error_count}) | "
            f"OK: {ok_count}"
        )
        self.batch_scan_status_bottom.setTextFormat(Qt.RichText)
        self.batch_scan_status_bottom.setText(bar_text)
        self.batch_scan_status_bottom.setToolTip(
            f"OK: {ok_count}\n"
            f"Lỗi hiện tại: {error_count}\n"
            f"Trùng SBD: {duplicate_count}\n"
            f"Sai mã đề: {wrong_code_count}\n"
            f"Đã sửa tổng: {edited_count}\n"
            f"- Đã sửa hết lỗi: {edited_clean_count}\n"
            f"- Đã sửa còn lỗi: {edited_error_count}"
        )

    def _clear_batch_preview_panels(self) -> None:
        if hasattr(self, "scan_result_preview"):
            self.scan_result_preview.setRowCount(0)
        if hasattr(self, "result_preview"):
            self.result_preview.clear()
        if hasattr(self, "manual_edit"):
            self.manual_edit.clear()
        if hasattr(self, "scan_image_preview"):
            self.scan_image_preview.clear()
            self.scan_image_preview.setText("Chọn bài thi ở danh sách bên trái")
            if hasattr(self.scan_image_preview, "clear_markers"):
                self.scan_image_preview.clear_markers()

    def _apply_scan_filter(self) -> None:
        def _normalize(text: str) -> str:
            return " ".join(str(text or "").strip().lower().split())

        value = _normalize(self.search_value.text())
        col = self._scan_filter_column_from_combo_index(self.filter_column.currentIndex())
        for i in range(self.scan_list.rowCount()):
            sid_item = self.scan_list.item(i, self.SCAN_COL_STUDENT_ID)
            payload = dict(sid_item.data(Qt.UserRole + 12) or {}) if sid_item is not None else {}
            flags = dict(payload.get("status_flags", {}) or {})
            status_text = str(payload.get("status", "") or (self.scan_list.item(i, self.SCAN_COL_STATUS).text() if self.scan_list.item(i, self.SCAN_COL_STATUS) else "")).strip().lower()
            is_edited = bool(flags.get("edited", False)) or ("đã sửa" in status_text)
            has_error = bool(flags.get("has_error", False))
            has_duplicate = bool(flags.get("duplicate", False)) or ("trùng sbd" in status_text or "duplicate" in status_text)
            has_wrong_code = bool(flags.get("wrong_code", False)) or ("mã đề không hợp lệ" in status_text)
            status_ok = True
            if self.batch_status_filter_mode == "error":
                status_ok = has_error
            elif self.batch_status_filter_mode == "duplicate":
                status_ok = has_duplicate
            elif self.batch_status_filter_mode == "wrong_code":
                status_ok = has_wrong_code
            elif self.batch_status_filter_mode == "edited":
                status_ok = is_edited
            if not value:
                self.scan_list.setRowHidden(i, not status_ok)
                continue
            if col is None:
                searchable = []
                for j in range(0, self.SCAN_COL_STATUS + 1):
                    item = self.scan_list.item(i, j)
                    searchable.append(_normalize(item.text() if item else ""))
                cell = " | ".join(searchable)
            else:
                item = self.scan_list.item(i, col)
                cell = _normalize(item.text() if item else "")
            self.scan_list.setRowHidden(i, (value not in cell) or (not status_ok))
        self._update_batch_scan_bottom_status_text()

    def _schedule_scan_filter(self, *_args) -> None:
        if hasattr(self, "_scan_filter_debounce_timer") and self._scan_filter_debounce_timer is not None:
            self._scan_filter_debounce_timer.start()
            return
        self._apply_scan_filter()

    def _handle_batch_status_filter_link(self, link: str) -> None:
        self.batch_status_filter_mode = str(link or "all").strip() or "all"
        self._apply_scan_filter()

    def _on_scan_header_clicked(self, section: int) -> None:
        combo_index = self._scan_filter_combo_index_from_header_section(section)
        if combo_index is None:
            return
        if 0 <= combo_index < self.filter_column.count():
            self.filter_column.setCurrentIndex(combo_index)
        self._apply_scan_filter()

    @staticmethod
    def _scan_filter_column_from_combo_index(combo_index: int) -> int | None:
        mapping = {
            0: None,  # Tất cả
            1: 0,     # STT
            2: 1,     # STUDENT ID
            3: 2,     # Phòng thi
            4: 3,     # Mã đề
            5: 4,     # Họ tên
            6: 5,     # Ngày sinh
            7: 6,     # Nội dung
            8: 7,     # Status
        }
        return mapping.get(int(combo_index), None)

    @staticmethod
    def _scan_filter_combo_index_from_header_section(section: int) -> int | None:
        mapping = {
            0: 1,  # STT
            1: 2,  # STUDENT ID
            2: 3,  # Phòng thi
            3: 4,  # Mã đề
            4: 5,  # Họ tên
            5: 6,  # Ngày sinh
            6: 7,  # Nội dung
            7: 8,  # Status
        }
        return mapping.get(int(section), None)

    def eventFilter(self, obj, event):
        if hasattr(self, "scan_image_scroll") and obj == self.scan_image_scroll.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self._zoom_preview_in()
                else:
                    self._zoom_preview_out()
                return True
        if hasattr(self, "scan_image_scroll") and obj == self.scan_image_scroll.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and not self.preview_source_pixmap.isNull():
                self.preview_drag_active = True
                self.preview_last_pos = event.position().toPoint()
                self.scan_image_scroll.viewport().setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self.preview_drag_active and self.preview_last_pos is not None:
                pos = event.position().toPoint()
                dx = pos.x() - self.preview_last_pos.x()
                dy = pos.y() - self.preview_last_pos.y()
                self.preview_last_pos = pos
                self.scan_image_scroll.horizontalScrollBar().setValue(
                    self.scan_image_scroll.horizontalScrollBar().value() - dx
                )
                self.scan_image_scroll.verticalScrollBar().setValue(
                    self.scan_image_scroll.verticalScrollBar().value() - dy
                )
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.preview_drag_active = False
                self.preview_last_pos = None
                self.scan_image_scroll.viewport().unsetCursor()
                return True
        return super().eventFilter(obj, event)

    def _selected_scan_row_index(self) -> int:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= self.scan_list.rowCount():
            return -1
        return idx

    def _rotate_selected_scan(self, degrees: int) -> None:
        row_idx = self._selected_scan_row_index()
        if row_idx < 0:
            return
        current = int(self.preview_rotation_by_index.get(row_idx, 0) or 0)
        self.preview_rotation_by_index[row_idx] = (current + int(degrees)) % 360
        if row_idx < len(self.scan_results):
            self._update_scan_preview(row_idx)
        else:
            self._update_scan_preview_from_saved_row(row_idx)

    def _rebuild_error_list(self) -> None:
        self.error_list.clear()
        for result in self.scan_results:
            rec_errors = list(getattr(result, "recognition_errors", [])) or list(getattr(result, "errors", []))
            for issue in result.issues:
                self.error_list.addItem(f"{Path(result.image_path).name}: {issue.code} - {issue.message}")
            for err in rec_errors:
                self.error_list.addItem(f"{Path(result.image_path).name}: RECOGNITION - {err}")

    @staticmethod
    def _student_sort_token(student_id: str) -> tuple[int, int, object]:
        sid = str(student_id or "").strip()
        if not sid or sid == "-":
            return (1, 0, "")
        if sid.isdigit():
            return (0, 0, int(sid))
        return (0, 1, sid.casefold())

    def _row_sort_bucket(self, status: str, blank_map: dict[str, list[int]]) -> int:
        status_text = str(status or "").strip()
        has_error = bool(status_text and status_text != "OK")
        has_blank = any(bool(blank_map.get(sec, [])) for sec in ["MCQ", "TF", "NUMERIC"])
        if has_error:
            return 0
        if has_blank:
            return 1
        return 2

    def _populate_scan_grid_from_results(
        self,
        results: list[OMRResult],
        forced_status_by_image: dict[str, str] | None = None,
        skip_expensive_checks: bool = False,
        preserve_selection_image_path: str = "",
        refresh_statuses: bool = False,
        rebuild_error_list: bool = False,
    ) -> None:
        # scan_list columns: 0 stt, 1 sid, 2 exam_room, 3 exam_code, 4 full_name, 5 birth_date, 6 content, 7 status, 8 actions
        forced_status_by_image = forced_status_by_image or {}
        duplicate_ids: dict[str, int] = {}
        subject_scope: tuple[set[str], set[str]] | None = None
        available_exam_codes: set[str] | None = None
        if not skip_expensive_checks:
            for res in results:
                sid = str(getattr(res, "student_id", "") or "").strip()
                if not self._student_id_has_recognition_error(sid):
                    duplicate_ids[sid] = duplicate_ids.get(sid, 0) + 1
            subject_scope = self._subject_student_room_scope()
            available_exam_codes = self._available_exam_codes()

        row_views: list[dict[str, object]] = []
        for result in results:
            self._refresh_student_profile_for_result(result)
            scoped = self._scoped_result_copy(result)
            sid = str(result.student_id or "").strip()
            exam_code_text = str(result.exam_code or "").strip()
            image_path = str(result.image_path or "")
            subject_cfg = self._selected_batch_subject_config() or {}
            room_text = ""
            if sid:
                room_text = str(self._subject_room_for_student_id(sid, subject_cfg) or "").strip()
            if not room_text:
                room_text = str(getattr(result, "exam_room", "") or "").strip()
            if not room_text:
                profile = self._student_profile_by_id(sid) if sid else {}
                room_text = str((profile or {}).get("exam_room", "") or "").strip()
            setattr(result, "exam_room", room_text)
            forced_status = str(forced_status_by_image.get(image_path, "") or "")
            status_override = ""
            if forced_status:
                status_override = forced_status
            elif skip_expensive_checks:
                status_override = str(getattr(result, "cached_status", "") or "OK")
            payload = self._build_scan_row_payload_from_result(
                result,
                row_idx=None,
                duplicate_count=duplicate_ids.get(sid, 0),
                subject_scope=None if skip_expensive_checks else subject_scope,
                available_exam_codes=None if skip_expensive_checks else available_exam_codes,
                forced_status=status_override,
            )
            setattr(result, "cached_forced_status", str(forced_status or getattr(result, "cached_forced_status", "") or ""))
            payload["forced_status"] = str(forced_status or payload.get("forced_status", "") or "")
            row_views.append(payload)

        row_views.sort(
            key=lambda item: (
                self._row_sort_bucket(str(item["status"]), item["blank_map"]),
                self._student_sort_token(str(item["student_id"])),
                str(item["exam_code"] or "").casefold(),
                Path(str(item["image_path"] or "")).name.casefold(),
            )
        )

        self.scan_results = [item["result"] for item in row_views]
        self.scan_edit_history = {}
        self.scan_last_adjustment = {}
        self.scan_manual_adjustments = {}
        for res in self.scan_results:
            image_key = self._result_identity_key(getattr(res, "image_path", ""))
            if not image_key:
                continue
            history = [str(x) for x in (getattr(res, "edit_history", []) or []) if str(x or "").strip()]
            if history:
                self.scan_edit_history[image_key] = list(history)
                self.scan_last_adjustment[image_key] = str(getattr(res, "last_adjustment", "") or history[-1])
                setattr(res, "manually_edited", True)
                if not str(getattr(res, "cached_forced_status", "") or "").strip():
                    setattr(res, "cached_forced_status", "Đã sửa")
                self.scan_forced_status_by_index[image_key] = "Đã sửa"
            manual_items = [str(x) for x in (getattr(res, "manual_adjustments", []) or []) if str(x or "").strip()]
            if manual_items:
                self.scan_manual_adjustments[image_key] = list(sorted(set(manual_items)))
        subject_key = self._current_batch_subject_key()
        if subject_key:
            self.scan_results_by_subject[self._batch_result_subject_key(subject_key)] = list(self.scan_results)
        self.scan_blank_questions = {idx: list(item["blank_map"].get("MCQ", [])) for idx, item in enumerate(row_views)}
        self.scan_blank_summary = {idx: dict(item["blank_map"]) for idx, item in enumerate(row_views)}
        self.scan_forced_status_by_index = {
            self._result_identity_key(str(item.get("image_path", "") or "")): str(item["forced_status"] or "")
            for item in row_views
            if self._result_identity_key(str(item.get("image_path", "") or "")) and str(item["forced_status"] or "")
        }

        scan_list = self.scan_list
        selected_image = str(preserve_selection_image_path or "").strip()
        if not selected_image and 0 <= scan_list.currentRow() < scan_list.rowCount():
            current_item = scan_list.item(scan_list.currentRow(), self.SCAN_COL_STUDENT_ID)
            selected_image = str(current_item.data(Qt.UserRole) if current_item else "").strip()

        self._begin_scan_grid_update()
        try:
            scan_list.setRowCount(len(row_views))
            for idx, item in enumerate(row_views):
                payload = dict(item)
                self._apply_scan_row_payload_to_grid(idx, payload, skip_actions=skip_expensive_checks)
            scan_list.resizeRowsToContents()
            for fit_col in [self.SCAN_COL_STT, self.SCAN_COL_STUDENT_ID, self.SCAN_COL_EXAM_ROOM, self.SCAN_COL_EXAM_CODE, self.SCAN_COL_FULL_NAME, self.SCAN_COL_BIRTH_DATE, self.SCAN_COL_ACTIONS]:
                scan_list.resizeColumnToContents(fit_col)
        finally:
            self._end_scan_grid_update()

        if refresh_statuses:
            self._refresh_all_statuses()
        if rebuild_error_list:
            self._rebuild_error_list()

        if selected_image:
            for row in range(scan_list.rowCount()):
                cell = scan_list.item(row, self.SCAN_COL_STUDENT_ID)
                if str(cell.data(Qt.UserRole) if cell else "").strip() == selected_image:
                    scan_list.setCurrentCell(row, self.SCAN_COL_STUDENT_ID)
                    scan_list.selectRow(row)
                    break

    def _finalize_batch_scan_display(self, refresh_statuses: bool = True) -> None:
        if not hasattr(self, "scan_list"):
            return
        if refresh_statuses:
            self._refresh_all_statuses()
        self._rebuild_error_list()
        self._apply_scan_filter()
        if self.scan_list.rowCount() <= 0:
            if hasattr(self, "scan_result_preview"):
                self.scan_result_preview.setRowCount(0)
            if hasattr(self, "result_preview"):
                self.result_preview.clear()
            if hasattr(self, "manual_edit"):
                self.manual_edit.clear()
            return
        target_row = -1
        for row in range(self.scan_list.rowCount()):
            if not self.scan_list.isRowHidden(row):
                target_row = row
                break
        if target_row < 0:
            target_row = 0
        self.scan_list.setCurrentCell(target_row, self.SCAN_COL_STUDENT_ID)
        self.scan_list.selectRow(target_row)
        self._on_scan_selected()
        self._refresh_ribbon_action_states()

    def _build_scan_row_payload_from_result(
        self,
        result,
        row_idx: int | None = None,
        duplicate_count: int = 1,
        subject_scope: tuple[set[str], set[str]] | None = None,
        available_exam_codes: set[str] | None = None,
        forced_status: str = "",
    ) -> dict:
        scoped_result = self._scoped_result_copy(result)
        blank_map = self._compute_blank_questions(scoped_result)
        expected_by_section = self._expected_questions_by_section(scoped_result)
        sid = str(getattr(result, "student_id", "") or "").strip()
        exam_code_text = str(getattr(result, "exam_code", "") or "").strip()
        room_text = str(self._subject_room_for_student_id(sid) or "").strip() if sid else ""
        if not room_text:
            room_text = str(getattr(result, "exam_room", "") or "").strip()
        if not room_text and sid:
            profile = self._student_profile_by_id(sid)
            room_text = str((profile or {}).get("exam_room", "") or "").strip()
        setattr(result, "exam_room", room_text)

        analysis = self._analyze_scan_status(
            result,
            duplicate_count=duplicate_count,
            subject_scope=subject_scope,
            available_exam_codes=available_exam_codes,
            forced_status=forced_status,
        )
        status_parts = list(analysis["status_parts"])
        has_edit_history = bool(analysis["has_edit_history"])
        is_manual_edited = bool(analysis["is_manual_edited"])
        forced_status_text = str(analysis["forced_status_text"])
        status_parts_text = str(analysis["status_parts_text"])
        status_text = str(analysis["status_text"])
        effective_forced_status = str(analysis["effective_forced_status"])
        manual_content_override = str(getattr(result, "manual_content_override", "") or "").strip()
        content_text = self._build_recognition_content_text(scoped_result, blank_map, expected_by_section)
        recognized_short = self._short_recognition_text_for_result(scoped_result)

        if row_idx is not None and row_idx >= 0:
            self.scan_blank_summary[row_idx] = dict(blank_map)
            self.scan_blank_questions[row_idx] = list(blank_map.get("MCQ", []))
            image_key_for_row = self._row_image_key(row_idx) or self._result_identity_key(getattr(result, "image_path", ""))
            if effective_forced_status:
                if image_key_for_row:
                    self.scan_forced_status_by_index[image_key_for_row] = effective_forced_status
            elif image_key_for_row and image_key_for_row in self.scan_forced_status_by_index:
                self.scan_forced_status_by_index.pop(image_key_for_row, None)

        setattr(result, "cached_status", status_text)
        setattr(result, "cached_content", content_text)
        setattr(result, "cached_recognized_short", recognized_short)
        setattr(result, "cached_blank_summary", dict(blank_map))
        setattr(result, "manual_content_override", manual_content_override)
        setattr(result, "cached_forced_status", effective_forced_status)

        return {
            "result": result,
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": sid,
            "exam_room": room_text,
            "exam_code": exam_code_text,
            "full_name": str(getattr(result, "full_name", "") or "-"),
            "birth_date": str(getattr(result, "birth_date", "") or "-"),
            "content": content_text,
            "manual_content_override": manual_content_override,
            "status": status_text,
            "recognized_short": recognized_short,
            "forced_status": effective_forced_status,
            "manually_edited": is_manual_edited,
            "edit_history": [str(x) for x in (getattr(result, "edit_history", []) or []) if str(x or "").strip()],
            "last_adjustment": str(getattr(result, "last_adjustment", "") or ""),
            "manual_adjustments": [str(x) for x in (getattr(result, "manual_adjustments", []) or []) if str(x or "").strip()],
            "blank_map": dict(blank_map),
            "expected_by_section": dict(expected_by_section),
            "status_flags": {
                "edited": bool(is_manual_edited),
                "has_error": bool(analysis["has_error"]),
                "duplicate": bool(analysis["has_duplicate"]),
                "wrong_code": bool(analysis["has_wrong_code"]),
                "ok": bool(analysis["is_clean_ok"]),
                "edited_clean": bool(analysis["is_edited_clean"]),
                "edited_error": bool(analysis["is_edited_with_error"]),
            },
            "serialized_result": self._serialize_omr_result(result),
            "recognized_template_path": str(getattr(result, "recognized_template_path", "") or ""),
            "recognized_alignment_profile": str(getattr(result, "recognized_alignment_profile", "") or ""),
            "recognized_fill_threshold": float(getattr(result, "recognized_fill_threshold", 0.45) or 0.45),
            "recognized_empty_threshold": float(getattr(result, "recognized_empty_threshold", 0.20) or 0.20),
            "recognized_certainty_margin": float(getattr(result, "recognized_certainty_margin", 0.08) or 0.08),
        }

    def _apply_scan_row_payload_to_grid(self, row_idx: int, payload: dict, *, skip_actions: bool = False) -> None:
        if row_idx < 0:
            return
        if row_idx >= self.scan_list.rowCount():
            self.scan_list.setRowCount(row_idx + 1)
        self.scan_list.setItem(row_idx, self.SCAN_COL_STT, QTableWidgetItem(str(row_idx + 1)))
        sid_item = QTableWidgetItem(str(payload.get("student_id", "") or "-"))
        sid_item.setData(Qt.UserRole, str(payload.get("image_path", "") or ""))
        sid_item.setData(Qt.UserRole + 1, str(payload.get("exam_code", "") or ""))
        sid_item.setData(Qt.UserRole + 2, str(payload.get("recognized_short", "") or ""))
        sid_item.setData(Qt.UserRole + 10, dict(payload.get("serialized_result", {}) or {}))
        sid_item.setData(Qt.UserRole + 11, str(payload.get("manual_content_override", "") or ""))
        sid_item.setData(Qt.UserRole + 12, dict(payload or {}))
        self.scan_list.setItem(row_idx, self.SCAN_COL_STUDENT_ID, sid_item)
        self.scan_list.setItem(row_idx, self.SCAN_COL_EXAM_ROOM, QTableWidgetItem(str(payload.get("exam_room", "") or "-")))
        self.scan_list.setItem(row_idx, self.SCAN_COL_EXAM_CODE, QTableWidgetItem(str(payload.get("exam_code", "") or "-")))
        self.scan_list.setItem(row_idx, self.SCAN_COL_FULL_NAME, QTableWidgetItem(str(payload.get("full_name", "") or "-")))
        self.scan_list.setItem(row_idx, self.SCAN_COL_BIRTH_DATE, QTableWidgetItem(str(payload.get("birth_date", "") or "-")))
        content_text = str(payload.get("content", "") or "")
        content_item = QTableWidgetItem(content_text)
        content_item.setToolTip(content_text)
        self.scan_list.setItem(row_idx, self.SCAN_COL_CONTENT, content_item)
        full_status = str(payload.get("status", "") or "OK")
        status_item = QTableWidgetItem(self._compact_status_text(full_status, max_len=150))
        status_item.setToolTip(full_status)
        if full_status != "OK":
            status_item.setForeground(Qt.red)
        self.scan_list.setItem(row_idx, self.SCAN_COL_STATUS, status_item)
        if skip_actions:
            self.scan_list.setItem(row_idx, self.SCAN_COL_ACTIONS, QTableWidgetItem("..."))
        else:
            self._set_scan_action_widget(row_idx)

    def _update_scan_row_from_result(self, idx: int, result) -> None:
        # scan_list columns: 0 stt, 1 sid, 2 exam_room, 3 exam_code, 4 full_name, 5 birth_date, 6 content, 7 status, 8 actions
        target_idx = int(idx)
        image_key = self._result_identity_key(getattr(result, "image_path", ""))
        if image_key:
            mapped_idx = self._row_index_by_image_path(image_key)
            if mapped_idx >= 0:
                target_idx = mapped_idx
        if target_idx < 0 or target_idx >= self.scan_list.rowCount():
            return
        self._refresh_student_profile_for_result(result)
        payload = self._build_scan_row_payload_from_result(result, row_idx=target_idx)
        self._apply_scan_row_payload_to_grid(target_idx, payload)

    def _ensure_template_for_selected_subject(self) -> bool:
        cfg = self._selected_batch_subject_config() or self._resolve_subject_config_for_batch()
        template_path = self._resolve_template_path_for_subject(cfg)
        if not template_path:
            return False
        pth = Path(template_path)
        if not pth.exists():
            return False
        active_template_path = str(getattr(self, "_active_template_path", "") or "").strip()
        desired_template_path = str(pth.resolve())
        try:
            if (self.template is None) or (active_template_path != desired_template_path):
                self.template = Template.load_json(pth)
                setattr(self, "_active_template_path", desired_template_path)
            self._apply_template_recognition_settings(self.template, sync_mode_selector=False)
            return True
        except Exception:
            return False

    @staticmethod
    def _aligned_image_to_qpixmap(image) -> QPixmap:
        try:
            if image is None:
                return QPixmap()
            h, w, ch = image.shape
            if h <= 0 or w <= 0 or ch < 3:
                return QPixmap()
            rgb = image[:, :, :3][:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())
        except Exception:
            return QPixmap()

    def _recognition_overlay_positions_for_result(self, result: OMRResult) -> list[dict[str, float]]:
        if self.template is None or self.preview_source_pixmap.isNull():
            return []
        states_by_zone = getattr(result, "bubble_states_by_zone", None)
        if not isinstance(states_by_zone, dict) or not states_by_zone:
            return []
        tpl_w = max(1.0, float(self.template.width))
        tpl_h = max(1.0, float(self.template.height))
        img_w = max(1.0, float(self.preview_source_pixmap.width()))
        img_h = max(1.0, float(self.preview_source_pixmap.height()))
        sx, sy = img_w / tpl_w, img_h / tpl_h
        markers: list[dict[str, float]] = []
        for z in self.template.zones:
            g = z.grid
            if not g or not g.bubble_positions:
                continue
            states = list(states_by_zone.get(z.id, []) or [])
            if not states:
                continue
            for i, pos in enumerate(g.bubble_positions):
                if i >= len(states) or not bool(states[i]):
                    continue
                bx, by = pos
                markers.append({"zone_id": z.id, "x": float(bx) * sx, "y": float(by) * sy})
        return markers

    def _rerecognize_selected_scan(self) -> None:
        idx = self._selected_scan_row_index()
        if idx < 0:
            QMessageBox.warning(self, "Nhận dạng lại", "Chọn một bài thi trong danh sách bên trái trước.")
            return
        if not self._ensure_template_for_selected_subject() or not self.template:
            QMessageBox.warning(self, "Nhận dạng lại", "Chưa có template khả dụng (theo môn hoặc theo kỳ thi).")
            return

        old_result = self.scan_results[idx] if idx < len(self.scan_results) else self._build_result_from_saved_table_row(idx)
        if old_result is None:
            QMessageBox.warning(self, "Nhận dạng lại", "Không tìm thấy dữ liệu dòng đang chọn để nhận dạng lại.")
            return
        subject_key = self._current_batch_subject_key()
        has_existing_score = bool(subject_key and self._scan_has_existing_score(subject_key, old_result))
        confirm_message = "Bạn có chắc muốn nhận dạng lại bài thi đã chọn?"
        if has_existing_score:
            confirm_message = (
                "Bài thi này đã có điểm. Nhận dạng lại ảnh đã chọn có thể làm thay đổi điểm đã tính.\n\n"
                "Bạn có muốn tiếp tục không?"
            )
        confirm_rerun = QMessageBox.question(
            self,
            "Nhận dạng lại ảnh chọn",
            confirm_message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm_rerun != QMessageBox.Yes:
            return
        image_path = str(old_result.image_path or "").strip()
        if not image_path or not Path(image_path).exists():
            QMessageBox.warning(self, "Nhận dạng lại", f"Không tìm thấy ảnh để nhận dạng lại:\n{image_path or '-'}")
            return

        process_path = image_path
        rotation = int(self.preview_rotation_by_index.get(idx, 0) or 0) % 360
        temp_rotated_path = None
        if rotation:
            pix = QPixmap(image_path)
            if pix.isNull():
                QMessageBox.warning(self, "Nhận dạng lại", "Không thể mở ảnh để xoay tạm thời trước khi nhận dạng lại.")
                return
            rotated = pix.transformed(QTransform().rotate(float(rotation)), Qt.SmoothTransformation)
            temp_rotated_path = str(Path(image_path).with_name(f".{Path(image_path).stem}_tmp_rerun_{rotation}.png"))
            if not rotated.save(temp_rotated_path):
                QMessageBox.warning(self, "Nhận dạng lại", "Không thể tạo ảnh xoay tạm thời để nhận dạng lại.")
                return
            process_path = temp_rotated_path

        recognized_template_path = str(getattr(old_result, "recognized_template_path", "") or "").strip()
        template_path = recognized_template_path or self._resolve_template_path_for_subject(self._selected_batch_subject_config() or self._resolve_subject_config_for_batch())
        wait_dlg = self._open_wait_progress("Đang nhận dạng lại ảnh đã chọn...")
        try:
            new_result = self._recognize_image_with_exact_template(process_path, template_path, source_tag="rerecognize_selected", allow_retry=False)
        finally:
            if temp_rotated_path:
                try:
                    Path(temp_rotated_path).unlink(missing_ok=True)
                except Exception:
                    pass
            self._close_wait_progress(wait_dlg)
        new_result.image_path = image_path
        setattr(new_result, "manually_edited", bool(getattr(old_result, "manually_edited", False)))
        setattr(new_result, "cached_forced_status", str(getattr(old_result, "cached_forced_status", "") or ""))
        image_key = self._result_identity_key(image_path)
        prior_history = [str(x) for x in (getattr(old_result, "edit_history", []) or []) if str(x or "").strip()]
        if image_key:
            prior_history = [str(x) for x in (self.scan_edit_history.get(image_key, []) or []) if str(x or "").strip()] or prior_history
        setattr(new_result, "edit_history", list(prior_history))
        setattr(new_result, "last_adjustment", str(getattr(old_result, "last_adjustment", "") or ""))
        setattr(new_result, "manual_adjustments", [str(x) for x in (getattr(old_result, "manual_adjustments", []) or []) if str(x or "").strip()])

        sid = (new_result.student_id or "").strip()
        profile = self._student_profile_by_id(sid)
        if profile.get("name"):
            setattr(new_result, "full_name", profile.get("name"))
        if profile.get("birth_date"):
            setattr(new_result, "birth_date", self._format_birth_date_mmddyyyy(profile.get("birth_date")))
        if profile.get("class_name"):
            setattr(new_result, "class_name", profile.get("class_name"))
        if profile.get("exam_room"):
            setattr(new_result, "exam_room", profile.get("exam_room"))
        scoped_new = self._scoped_result_copy(new_result)
        blank_map = self._compute_blank_questions(scoped_new)
        rec_errors = list(getattr(new_result, "recognition_errors", [])) or list(getattr(new_result, "errors", []))
        old_sid_for_score = str(getattr(old_result, "student_id", "") or "").strip()
        new_sid_for_score = str(getattr(new_result, "student_id", "") or "").strip()

        if rec_errors:
            # Failed/partial re-recognition must restart workflow like a newly recognized file:
            # clear manual-edit traces and reset forced-status cache.
            setattr(new_result, "manually_edited", False)
            setattr(new_result, "cached_forced_status", "")
            setattr(new_result, "cached_status", "")
            setattr(new_result, "edit_history", [])
            setattr(new_result, "manual_adjustments", [])
            setattr(new_result, "last_adjustment", "")
            if image_key:
                self.scan_edit_history.pop(image_key, None)
                self.scan_manual_adjustments.pop(image_key, None)
                self.scan_last_adjustment.pop(image_key, None)
                self.scan_forced_status_by_index.pop(image_key, None)
            # If re-recognition has errors, scoring must be reset and follow full scoring flow again.
            self._invalidate_scoring_for_student_ids(
                [sid for sid in [old_sid_for_score, new_sid_for_score] if sid],
                subject_key=subject_key,
                reason="rerecognize_with_errors_reset",
            )

        self._set_scan_result_at_row(idx, new_result)
        subject_key = self._current_batch_subject_key()
        if subject_key:
            self.scan_results_by_subject[self._batch_result_subject_key(subject_key)] = list(self.scan_results)
        self.scan_blank_questions[idx] = blank_map.get("MCQ", [])
        self.scan_blank_summary[idx] = blank_map
        self._update_scan_row_from_result(idx, new_result)
        self._refresh_all_statuses()
        self._rebuild_error_list()
        self._update_scan_preview(idx)
        self._sync_correction_detail_panel(new_result, rebuild_editor=False)
        if prior_history and not rec_errors:
            old_sid = str(getattr(old_result, "student_id", "") or "").strip() or "-"
            old_code = str(getattr(old_result, "exam_code", "") or "").strip() or "-"
            new_sid = str(getattr(new_result, "student_id", "") or "").strip() or "-"
            new_code = str(getattr(new_result, "exam_code", "") or "").strip() or "-"
            self._record_adjustment(
                idx,
                [f"Nhận dạng lại ảnh chọn: STUDENT ID {old_sid} -> {new_sid}; Mã đề {old_code} -> {new_code}"],
                "rerecognize_selected",
            )
            self._persist_single_scan_result_to_db(new_result, note="rerecognize_selected_with_history")
            self._refresh_all_statuses()
        self._persist_single_scan_result_to_db(new_result, note="rerecognize_selected_scan")
        self.btn_save_batch_subject.setEnabled(False)

        summary = (
            f"Ảnh: {Path(image_path).name}\n"
            f"STUDENT ID: {new_result.student_id or '-'}\n"
            f"Họ tên: {str(getattr(new_result, 'full_name', '') or '-')}\n"
            f"Ngày sinh: {str(getattr(new_result, 'birth_date', '') or '-')}\n"
            f"Mã đề: {new_result.exam_code or '-'}\n"
            f"Nhận dạng ngắn: {self._compact_value(self._short_recognition_text_for_result(scoped_new), 180)}\n"
            f"Số lỗi nhận dạng: {len(rec_errors)}"
        )
        QMessageBox.information(self, "Kết quả nhận dạng lại", summary)

    def _scan_has_existing_score(self, subject_key: str, result: OMRResult) -> bool:
        if not subject_key or result is None:
            return False
        sid = str(getattr(result, "student_id", "") or "").strip()
        exam_code = str(getattr(result, "exam_code", "") or "").strip()

        # DB-first: kiểm tra trực tiếp trong scores đã persist.
        try:
            db_rows = list(self.database.fetch_scores_for_subject(subject_key) or [])
        except Exception:
            db_rows = []
        for row in db_rows:
            row_sid = str((row or {}).get("student_id", "") or "").strip()
            row_exam = str((row or {}).get("exam_code", "") or "").strip()
            if sid and sid == row_sid:
                return True
            if exam_code and row_exam and exam_code == row_exam and ((not sid) or (not row_sid)):
                return True
        return False

    def _render_preview_pixmap(self) -> None:
        if self.preview_source_pixmap.isNull():
            self.scan_image_preview.setPixmap(QPixmap())
            return
        src_size = self.preview_source_pixmap.size()
        target_w = max(1, int(src_size.width() * self.preview_zoom_factor))
        target_h = max(1, int(src_size.height() * self.preview_zoom_factor))
        scaled = self.preview_source_pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.scan_image_preview.setPixmap(scaled)
        self.scan_image_preview.resize(scaled.size())
        self.scan_image_preview.adjustSize()

    def _zoom_preview_in(self) -> None:
        self.preview_zoom_factor = min(4.0, self.preview_zoom_factor + 0.1)
        self._render_preview_pixmap()
        self.btn_zoom_reset.setText(f"{int(self.preview_zoom_factor*100)}%")

    def _zoom_preview_out(self) -> None:
        self.preview_zoom_factor = max(0.3, self.preview_zoom_factor - 0.1)
        self._render_preview_pixmap()
        self.btn_zoom_reset.setText(f"{int(self.preview_zoom_factor*100)}%")

    def _zoom_preview_reset(self) -> None:
        self.preview_zoom_factor = 0.3
        self._render_preview_pixmap()
        self.btn_zoom_reset.setText("30%")

    @staticmethod
    def _compact_value(value, limit: int = 120) -> str:
        text = str(value)
        return text if len(text) <= limit else text[:limit] + "..."

    def _update_scan_preview_from_saved_row(self, row: int) -> None:
        sid = self.scan_list.item(row, self.SCAN_COL_STUDENT_ID).text() if self.scan_list.item(row, self.SCAN_COL_STUDENT_ID) else "-"
        exam_code_cell = self.scan_list.item(row, self.SCAN_COL_EXAM_CODE).text() if self.scan_list.item(row, self.SCAN_COL_EXAM_CODE) else "-"
        content = self.scan_list.item(row, self.SCAN_COL_CONTENT).text() if self.scan_list.item(row, self.SCAN_COL_CONTENT) else "-"
        status = self.scan_list.item(row, self.SCAN_COL_STATUS).text() if self.scan_list.item(row, self.SCAN_COL_STATUS) else "-"
        img_path = ""
        exam_code = ""
        item0 = self.scan_list.item(row, self.SCAN_COL_STUDENT_ID)
        if item0:
            img_path = str(item0.data(Qt.UserRole) or "")
            exam_code = str(item0.data(Qt.UserRole + 1) or "")

        pix = QPixmap(img_path) if img_path else QPixmap()
        if pix.isNull():
            self.preview_source_pixmap = QPixmap()
            self.scan_image_preview.setPixmap(QPixmap())
            self.scan_image_preview.setText("Không có ảnh tương ứng cho dòng đã lưu")
            self.scan_image_preview.clear_markers()
            self.btn_zoom_reset.setText("30%")
        else:
            rotation = int(self.preview_rotation_by_index.get(row, 0) or 0) % 360
            if rotation:
                pix = pix.transformed(QTransform().rotate(float(rotation)), Qt.SmoothTransformation)
            self.preview_source_pixmap = pix
            self._render_preview_pixmap()
            self.btn_zoom_reset.setText(f"{int(self.preview_zoom_factor*100)}%")

        tmp_result = self._restore_full_result_for_row(row)
        if tmp_result is not None:
            self.scan_image_preview.set_overlay_markers(self._recognition_overlay_positions_for_result(tmp_result))
        else:
            self.scan_image_preview.clear_markers()

        rows = [
            ("File ảnh", Path(str(img_path or "")).name or "-"),
            ("STUDENT ID", sid),
            ("Mã đề", exam_code or exam_code_cell or "-"),
            ("Nội dung", self._compact_value(content, 220)),
            ("Status", status),
        ]
        self.scan_result_preview.setRowCount(0)
        for r, (k, v) in enumerate(rows):
            self.scan_result_preview.insertRow(r)
            self.scan_result_preview.setItem(r, 0, QTableWidgetItem(str(k)))
            self.scan_result_preview.setItem(r, 1, QTableWidgetItem(str(v)))

    def _on_scan_selected(self) -> None:
        if self._scan_grid_loading:
            return
        index = self.scan_list.currentRow()
        if index < 0:
            return
        self._ensure_scan_action_widget(index)
        result = self._restore_full_result_for_row(index)
        if result is not None:
            if index >= len(self.scan_results):
                self._set_scan_result_at_row(index, result)
            self._update_scan_preview(index)
            self._load_selected_result_for_correction()
            return
        self._update_scan_preview_from_saved_row(index)

    def _short_recognition_text_for_result(self, result) -> str:
        scoped_result = self._scoped_result_copy(result)
        parts: list[str] = []
        mcq = self._format_mcq_answers(scoped_result.mcq_answers or {})
        tf = self._format_tf_answers(scoped_result.true_false_answers or {})
        num = self._format_numeric_answers(scoped_result.numeric_answers or {})
        if mcq and mcq != "-":
            parts.append(f"MCQ: {mcq}")
        if tf and tf != "-":
            parts.append(f"TF: {tf}")
        if num and num != "-":
            parts.append(f"NUM: {num}")
        return " | ".join(parts) if parts else "-"

    def _status_text_for_saved_table_row(self, row_idx: int) -> str:
        sid_item = self.scan_list.item(row_idx, self.SCAN_COL_STUDENT_ID)
        payload = dict(sid_item.data(Qt.UserRole + 12) or {}) if sid_item is not None else {}
        payload_status = str(payload.get("status", "") or "").strip()
        flags = dict(payload.get("status_flags", {}) or {})
        if payload_status:
            return payload_status
        payload_history = [str(x) for x in (payload.get("edit_history", []) or []) if str(x or "").strip()]
        payload_is_edited = bool(flags.get("edited", False)) or bool(payload.get("manually_edited", False)) or bool(payload_history)
        sid = (sid_item.text().strip() if sid_item else "")
        exam_code_text = str(sid_item.data(Qt.UserRole + 1) if sid_item else "").strip()
        dup = 0
        if not self._student_id_has_recognition_error(sid):
            unique_pairs: set[tuple[str, str]] = set()
            for result in (self._current_scan_results_snapshot() or []):
                sid_val = str(getattr(result, "student_id", "") or "").strip()
                img_val = self._result_identity_key(getattr(result, "image_path", ""))
                if sid_val and (not self._student_id_has_recognition_error(sid_val)):
                    unique_pairs.add((img_val, sid_val))
            dup = sum(1 for _img, sid_val in unique_pairs if sid_val == sid)
        status_parts = self._status_parts_for_row("" if self._student_id_has_recognition_error(sid) else sid, exam_code_text, dup)
        status_parts_text = ", ".join(status_parts) if status_parts else ""
        if payload_is_edited:
            if status_parts_text:
                return f"Đã sửa ({status_parts_text})"
            return "Đã sửa"
        return status_parts_text or "OK"

    @staticmethod
    def _compact_status_text(status_text: str, max_len: int = 150) -> str:
        text = str(status_text or "").strip()
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 3)].rstrip() + "..."

    def _refresh_row_status(
        self,
        idx: int,
        duplicate_count_map: dict[str, int] | None = None,
        subject_scope: tuple[set[str], set[str]] | None = None,
        available_exam_codes: set[str] | None = None,
    ) -> None:
        if idx < 0 or idx >= self.scan_list.rowCount():
            return
        image_key = self._row_image_key(idx)
        forced_status = self.scan_forced_status_by_index.get(image_key, "")
        row_result = self._result_by_image_path(image_key) if image_key else None
        if row_result is None and idx < len(self.scan_results):
            row_result = self.scan_results[idx]
        if not forced_status and row_result is not None:
            row_history = [str(x) for x in (getattr(row_result, "edit_history", []) or []) if str(x or "").strip()]
            if bool(getattr(row_result, "manually_edited", False)) or bool(row_history):
                forced_status = "Đã sửa"
                row_image = self._result_identity_key(getattr(row_result, "image_path", "")) or image_key
                if row_image:
                    self.scan_forced_status_by_index[row_image] = forced_status
        if row_result is not None:
            sid_text = str(getattr(row_result, "student_id", "") or "").strip()
            duplicate_count = 0
            if duplicate_count_map is not None and not self._student_id_has_recognition_error(sid_text):
                duplicate_count = int(duplicate_count_map.get(sid_text, 0) or 0)
            payload = self._build_scan_row_payload_from_result(
                row_result,
                row_idx=idx,
                duplicate_count=duplicate_count,
                subject_scope=subject_scope,
                available_exam_codes=available_exam_codes,
                forced_status=forced_status,
            )
            sid_item = self.scan_list.item(idx, self.SCAN_COL_STUDENT_ID)
            if sid_item is not None:
                sid_item.setData(Qt.UserRole + 1, str(payload.get("exam_code", "") or ""))
                sid_item.setData(Qt.UserRole + 2, str(payload.get("recognized_short", "") or ""))
                sid_item.setData(Qt.UserRole + 10, dict(payload.get("serialized_result", {}) or {}))
                sid_item.setData(Qt.UserRole + 11, str(payload.get("manual_content_override", "") or ""))
                sid_item.setData(Qt.UserRole + 12, dict(payload or {}))
            content_text = str(payload.get("content", "") or "")
            content_item = QTableWidgetItem(content_text)
            content_item.setToolTip(content_text)
            self.scan_list.setItem(idx, self.SCAN_COL_CONTENT, content_item)
            full_status = str(payload.get("status", "") or "OK")
        else:
            full_status = str(self._status_text_for_saved_table_row(idx) or "OK")
        display_status = self._compact_status_text(full_status, max_len=150)
        status_item = QTableWidgetItem(display_status)
        status_item.setToolTip(full_status)
        if full_status != "OK":
            status_item.setForeground(Qt.red)
        self.scan_list.setItem(idx, self.SCAN_COL_STATUS, status_item)

    def _update_scan_preview(self, index: int) -> None:
        if index < 0 or index >= len(self.scan_results):
            return
        result = self.scan_results[index]
        img_path = Path(result.image_path)
        aligned_pix = self._aligned_image_to_qpixmap(getattr(result, "aligned_image", None))
        pix = aligned_pix if not aligned_pix.isNull() else QPixmap(str(img_path))
        if pix.isNull():
            self.preview_source_pixmap = QPixmap()
            self.scan_image_preview.setText(f"Cannot load image: {img_path.name}")
            self.scan_image_preview.clear_markers()
        else:
            rotation = int(self.preview_rotation_by_index.get(index, 0) or 0) % 360
            if rotation:
                pix = pix.transformed(QTransform().rotate(float(rotation)), Qt.SmoothTransformation)
            self.preview_source_pixmap = pix
            self._render_preview_pixmap()
            self.scan_image_preview.set_overlay_markers(self._recognition_overlay_positions_for_result(result))

        preview_result = self._scoped_result_copy(self._lightweight_result_copy(result))
        section_counts = self._subject_section_question_counts(self._current_batch_subject_key())
        expected_actual = self._expected_questions_by_section(preview_result)

        def _build_display_mapping(actual_questions: list[int], limit: int) -> tuple[list[int], dict[int, int], dict[int, int]]:
            actual_sorted = sorted(set(int(q) for q in (actual_questions or [])))
            if limit > 0:
                actual_sorted = actual_sorted[:limit]
                if not actual_sorted:
                    actual_sorted = list(range(1, limit + 1))
            display_questions = list(range(1, len(actual_sorted) + 1))
            actual_to_display = {int(a): int(d) for d, a in zip(display_questions, actual_sorted)}
            display_to_actual = {int(d): int(a) for d, a in zip(display_questions, actual_sorted)}
            return display_questions, actual_to_display, display_to_actual

        expected_display: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        actual_to_display: dict[str, dict[int, int]] = {"MCQ": {}, "TF": {}, "NUMERIC": {}}
        for sec in ["MCQ", "TF", "NUMERIC"]:
            sec_limit = max(0, int(section_counts.get(sec, 0) or 0))
            display_qs, map_actual_to_display, _map_display_to_actual = _build_display_mapping(
                list(expected_actual.get(sec, []) or []),
                sec_limit,
            )
            expected_display[sec] = list(display_qs)
            actual_to_display[sec] = dict(map_actual_to_display)

        mcq_display = {
            int(actual_to_display["MCQ"].get(int(q), int(q))): str(v)
            for q, v in (preview_result.mcq_answers or {}).items()
        }
        tf_display = {
            int(actual_to_display["TF"].get(int(q), int(q))): dict(v or {})
            for q, v in (preview_result.true_false_answers or {}).items()
        }
        num_display = {
            int(actual_to_display["NUMERIC"].get(int(q), int(q))): str(v)
            for q, v in (preview_result.numeric_answers or {}).items()
        }
        blank_map = self.scan_blank_summary.get(index) or self._compute_blank_questions(self._lightweight_result_copy(result))

        mcq_text = self._compact_value(self._format_mcq_answers_with_expected(mcq_display, expected_display.get("MCQ", [])), 220)
        tf_text = self._compact_value(self._format_tf_answers_with_expected(tf_display, expected_display.get("TF", [])), 220)
        num_text = self._compact_value(self._format_numeric_answers_with_expected(num_display, expected_display.get("NUMERIC", [])), 220)
        rows = [
            ("File ảnh", img_path.name),
            ("STUDENT ID", result.student_id or "-"),
            ("Exam code", result.exam_code or "-"),
        ]
        if section_counts.get("MCQ", 0) > 0:
            rows.extend([
                ("MCQ", mcq_text),
                ("MCQ không tô", ", ".join(str(x) for x in blank_map.get("MCQ", [])) or "-"),
            ])
        if section_counts.get("TF", 0) > 0:
            rows.extend([
                ("TF", tf_text),
                ("TF không tô", ", ".join(str(x) for x in blank_map.get("TF", [])) or "-"),
            ])
        if section_counts.get("NUMERIC", 0) > 0:
            rows.extend([
                ("NUM", num_text),
                ("NUMERIC không tô", ", ".join(str(x) for x in blank_map.get("NUMERIC", [])) or "-"),
            ])
        self.scan_result_preview.setRowCount(0)
        for r, (k, v) in enumerate(rows):
            self.scan_result_preview.insertRow(r)
            self.scan_result_preview.setItem(r, 0, QTableWidgetItem(str(k)))
            self.scan_result_preview.setItem(r, 1, QTableWidgetItem(str(v)))
