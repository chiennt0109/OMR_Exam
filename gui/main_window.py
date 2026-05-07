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
from gui.main_window_branding_mixin import MainWindowBrandingMixin
from gui.main_window_auto_recognition_mixin import MainWindowAutoRecognitionMixin
from gui.main_window_session_mixin import MainWindowSessionMixin
from gui.main_window_workspace_mixin import MainWindowWorkspaceMixin
from gui.main_window_template_mixin import MainWindowTemplateMixin
from gui.main_window_import_mixin import MainWindowImportMixin
from gui.main_window_batch_subject_mixin import MainWindowBatchSubjectMixin
from gui.main_window_batch_recognition_mixin import MainWindowBatchRecognitionMixin
from gui.main_window_batch_scope_mixin import MainWindowBatchScopeMixin
from gui.main_window_batch_storage_mixin import MainWindowBatchStorageMixin
from gui.main_window_batch_ui_mixin import MainWindowBatchUiMixin
from gui.main_window_batch_edit_mixin import MainWindowBatchEditMixin
from gui.main_window_scoring_mixin import MainWindowScoringMixin
from gui.main_window_export_mixin import MainWindowExportMixin
from gui.main_window_misc_mixin import MainWindowMiscMixin



class _SignalHook:
    """Tiny signal adapter used by _SimplePageHost.

    This keeps old `currentChanged.connect(...)` call sites working without using
    a Qt stacked page widget.
    """
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        if callable(callback) and callback not in self._callbacks:
            self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            try:
                callback(*args)
            except TypeError:
                callback()


class _SimplePageHost(QWidget):
    """Single-visible-page host that replaces the old stacked-page widget.

    Pages are plain widgets in one VBoxLayout. Only the active page is visible.
    The class intentionally implements the small subset of methods used by the
    existing modules (`addWidget`, `setCurrentIndex`, `currentIndex`) so the rest
    of the application remains stable while the heavy stacked widget dependency
    is removed.
    """
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.currentChanged = _SignalHook()
        self._pages: list[QWidget] = []
        self._current_index = -1
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def addWidget(self, widget: QWidget) -> int:
        index = len(self._pages)
        self._pages.append(widget)
        self._layout.addWidget(widget)
        widget.setVisible(False)
        if self._current_index < 0:
            self.setCurrentIndex(index)
        return index

    def currentIndex(self) -> int:
        return int(self._current_index)

    def setCurrentIndex(self, index: int) -> None:
        try:
            index = int(index)
        except Exception:
            index = 0
        if not (0 <= index < len(self._pages)):
            index = 0 if self._pages else -1
        if index == self._current_index:
            return
        if 0 <= self._current_index < len(self._pages):
            self._pages[self._current_index].setVisible(False)
        self._current_index = index
        if 0 <= index < len(self._pages):
            self._pages[index].setVisible(True)
        self.currentChanged.emit(index)
class MainWindow(MainWindowBrandingMixin, MainWindowAutoRecognitionMixin, MainWindowSessionMixin, MainWindowWorkspaceMixin, MainWindowTemplateMixin, MainWindowImportMixin, MainWindowBatchSubjectMixin, MainWindowBatchRecognitionMixin, MainWindowBatchScopeMixin, MainWindowBatchStorageMixin, MainWindowBatchUiMixin, MainWindowBatchEditMixin, MainWindowScoringMixin, MainWindowExportMixin, MainWindowMiscMixin, QMainWindow):
    SCAN_COL_STT = 0
    SCAN_COL_STUDENT_ID = 1
    SCAN_COL_EXAM_ROOM = 2
    SCAN_COL_EXAM_CODE = 3
    SCAN_COL_FULL_NAME = 4
    SCAN_COL_BIRTH_DATE = 5
    SCAN_COL_CONTENT = 6
    SCAN_COL_STATUS = 7
    SCAN_COL_ACTIONS = 8

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Exam Grading System")
        self.resize(1200, 800)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        try:
            _ico = app_icon()
            if _ico is not None and not _ico.isNull():
                self.setWindowIcon(_ico)
        except Exception:
            pass

        self.session: ExamSession | None = None
        self.template: Template | None = None
        self.answer_keys: AnswerKeyRepository | None = None
        self.scan_results = []
        self.scan_results_by_subject: dict[str, list] = {}
        self.batch_working_state_by_subject: dict[str, dict] = {}
        self.scan_files: list[Path] = []
        self.scan_blank_questions: dict[int, list[int]] = {}
        self.scan_blank_summary: dict[int, dict[str, list[int]]] = {}
        self.scan_manual_adjustments: dict[str, list[str]] = {}
        self.scan_edit_history: dict[str, list[str]] = {}
        self.scan_last_adjustment: dict[str, str] = {}
        self.score_rows = []
        self.scoring_results_by_subject: dict[str, dict[str, dict]] = {}
        self.scoring_phases: list[dict] = []
        self._scoring_dirty_subjects: set[str] = set()
        self.imported_exam_codes: list[str] = []
        self.active_batch_subject_key: str | None = None
        self.subject_catalog: list[str] = ["Toán", "Ngữ văn", "Tiếng Anh", "Vật lý", "Hóa học", "Sinh học"]
        self.block_catalog: list[str] = ["10", "11", "12"]
        self.subjects: list[str] = list(self.subject_catalog)
        self.grades: list[str] = list(self.block_catalog)
        self.subject_configs: list[dict] = []
        self.subject_management_mode = "subjects"
        self.subject_edit_index: int | None = None
        self.batch_editor_return_payload: dict | None = None
        self.batch_editor_return_session_id: str | None = None
        self._current_batch_data_source: str = "empty"
        self._route_history: list[dict] = []
        self._current_route_name: str = "exam_list"
        self._current_route_context: dict = {}
        self._suspend_route_history_push: bool = False

        self.omr_processor = OMRProcessor()
        self.scoring_engine = ScoringEngine()
        self.current_session_path: Path | None = None
        self.current_session_id: str | None = None
        self.session_dirty = False
        self._session_saved_signature = ""

        self.database = OMRDatabase.default()
        self.session_registry: list[dict[str, str | bool]] = self._load_session_registry()
        self.template_repo = self._load_template_repository()
        self.template_editor_embedded: TemplateEditorWindow | None = None
        self.template_editor_mode = "library"

        self.stack = _SimplePageHost(self)  # Compatibility alias; not the old Qt stacked widget.
        self.stack.addWidget(self._build_exam_list_page())
        self.stack.addWidget(self._build_workspace_page())
        self.stack.addWidget(self._build_subject_management_page())
        self.stack.addWidget(self._build_template_management_page())
        self.template_editor_page = QWidget()
        self.template_editor_layout = QVBoxLayout(self.template_editor_page)
        self.template_editor_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.template_editor_page)
        self.exam_editor_page = QWidget()
        self.exam_editor_layout = QVBoxLayout(self.exam_editor_page)
        self.exam_editor_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.exam_editor_page)
        self.embedded_exam_dialog: NewExamDialog | None = None
        self.embedded_exam_session_id: str | None = None
        self.embedded_exam_session: ExamSession | None = None
        self.embedded_exam_original_payload: dict | None = None
        self.embedded_exam_is_new: bool = False
        self.preview_zoom_factor = 0.3
        self.preview_source_pixmap = QPixmap()
        self.preview_rotation_by_index: dict[int, int] = {}
        self.scan_forced_status_by_index: dict[str, str] = {}
        self.deleted_scan_images_by_subject: dict[str, set[str]] = {}
        self._student_option_cache_session_id: str = ""
        self._student_option_labels_cache: list[str] = []
        self._student_option_sid_map: dict[str, str] = {}
        self._student_option_profile_map: dict[str, dict[str, str]] = {}
        self._student_option_sid_set: set[str] = set()
        self._student_option_cache_signature: str = ""
        self._scan_grid_loading = False
        self._switching_batch_subject = False
        self._template_cache_by_path: dict[str, Template] = {}
        self._answer_keys_ready_subjects: set[str] = set()
        self._batch_scan_running = False
        self._batch_cancel_requested = False
        self._batch_loaded_runtime_key: str = ""
        self._batch_loaded_subject_signature: str = ""
        self._auto_recognition_busy = False
        self._auto_recognition_queue: deque[str] = deque()
        self._auto_recognition_enqueued: set[str] = set()
        self._auto_recognition_last_seen: dict[str, tuple[int, float, int]] = {}
        self._auto_recognition_pause_requested = False
        self._auto_recognition_active_subject: str = ""
        self.preview_drag_active = False
        self.preview_last_pos = None
        self.setCentralWidget(self.stack)

        self._build_menu()
        self._apply_application_branding()
        self.stack.currentChanged.connect(self._handle_stack_changed)
        db_subjects = self.database.fetch_catalog("subjects")
        db_blocks = self.database.fetch_catalog("blocks")
        default_subject_catalog = list(self.subject_catalog)
        default_block_catalog = list(self.block_catalog)
        if db_subjects:
            merged_subjects = list(db_subjects)
            seen_subjects = {str(x).strip().casefold() for x in merged_subjects if str(x).strip()}
            for item in default_subject_catalog:
                key = str(item).strip().casefold()
                if key and key not in seen_subjects:
                    merged_subjects.append(item)
                    seen_subjects.add(key)
            self.subject_catalog = merged_subjects
            self.subjects = list(merged_subjects)
            if merged_subjects != db_subjects:
                self.database.replace_catalog("subjects", self.subject_catalog)
        else:
            self.database.replace_catalog("subjects", self.subject_catalog)
        if db_blocks:
            merged_blocks = list(db_blocks)
            seen_blocks = {str(x).strip().casefold() for x in merged_blocks if str(x).strip()}
            for item in default_block_catalog:
                key = str(item).strip().casefold()
                if key and key not in seen_blocks:
                    merged_blocks.append(item)
                    seen_blocks.add(key)
            self.block_catalog = merged_blocks
            self.grades = list(merged_blocks)
            if merged_blocks != db_blocks:
                self.database.replace_catalog("blocks", self.block_catalog)
        else:
            self.database.replace_catalog("blocks", self.block_catalog)
        self._refresh_exam_list()
        self._refresh_batch_subject_controls()
        self._handle_stack_changed(self.stack.currentIndex())
        self.stack.setCurrentIndex(0)
        self._setup_auto_recognition_timer()


def run() -> None:
    bootstrap_application_db()
    app = QApplication([])
    window = MainWindow()
    window.showMaximized()
    app.exec()


if __name__ == "__main__":
    run()