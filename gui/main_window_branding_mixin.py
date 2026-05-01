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


class MainWindowBrandingMixin:
    """Branding, icon, theme, toolbar logo and ribbon icon setup."""
    def _set_action_icon_from_branding(self, action, icon_name: str) -> None:
        if action is None:
            return
        try:
            branded_icon = TOOLBAR.get(icon_name)
            if branded_icon is not None and not branded_icon.isNull():
                action.setIcon(branded_icon)
        except Exception:
            pass

    def _apply_branding_to_ribbon_actions(self) -> None:
        mapping = [
            # Main workflow ribbon
            ("ribbon_new_exam_action", "exam"),
            ("ribbon_view_exam_action", "home"),
            ("ribbon_subject_list_action", "subject"),
            ("ribbon_batch_scan_action", "scan"),
            ("ribbon_scoring_action", "scoring"),
            ("ribbon_recheck_action", "recheck"),
            ("ribbon_export_action", "export"),
            # Batch-scan context
            ("ribbon_batch_execute_action", "scan"),
            ("ribbon_batch_save_action", "save"),
            ("ribbon_batch_close_action", "close"),
            # Embedded exam editor context
            ("ribbon_exam_editor_add_subject_action", "add"),
            ("ribbon_exam_editor_edit_subject_action", "edit"),
            ("ribbon_exam_editor_delete_subject_action", "delete"),
            ("ribbon_exam_editor_save_action", "save"),
            ("ribbon_exam_editor_close_action", "close"),
            # Catalog subject management context
            ("ribbon_add_subject_action", "add"),
            ("ribbon_edit_subject_action", "edit"),
            ("ribbon_delete_subject_action", "delete"),
            ("ribbon_save_subject_action", "save"),
            # Template-library context
            ("ribbon_new_template_action", "template"),
            ("ribbon_edit_template_action", "edit"),
            ("ribbon_delete_template_action", "delete"),
            ("ribbon_close_template_action", "close"),
            # Main menus
            ("act_new_session", "exam"),
            ("act_open_from_list", "home"),
            ("act_save_session", "save"),
            ("act_save_as_subject", "export"),
            ("act_close_current_session", "close"),
            ("act_manage_template", "template"),
            ("act_manage_subject", "subject"),
            ("act_current_exam_subjects", "subject"),
            ("act_import_answer_key", "import"),
            ("act_export_answer_key_sample", "export"),
            ("act_batch_scan_menu", "scan"),
            ("act_execute_batch_scan", "scan"),
            ("act_edit_selected_scan", "edit"),
            ("act_calculate_scores", "scoring"),
            ("act_open_recheck", "recheck"),
            ("act_export_subject_scores", "export"),
            ("act_export_subject_score_matrix", "export"),
            ("act_export_class_subject_scores", "export"),
            ("act_export_all_classes_subject_scores", "export"),
            ("act_export_all_scores", "export"),
            ("act_export_return_by_class", "export"),
            ("act_export_recheck_by_subject", "export"),
            ("act_export_recheck_by_class", "export"),
            ("act_export_subject_api", "export"),
            ("act_export_reports_center", "report"),
            ("act_export_range_report", "report"),
            ("act_export_class_report", "report"),
            ("act_export_management_report", "report"),
        ]
        for attr_name, icon_name in mapping:
            self._set_action_icon_from_branding(getattr(self, attr_name, None), icon_name)

    def _ensure_toolbar_logo(self) -> None:
        if not hasattr(self, "main_ribbon") or getattr(self, "_toolbar_logo_added", False):
            return
        try:
            label = QLabel(self)
            pix = logo_symbol()
            if pix is None or pix.isNull():
                return
            label.setPixmap(pix.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            label.setToolTip("OMR Exam Grading System")
            label.setContentsMargins(8, 0, 8, 0)
            actions = self.main_ribbon.actions()
            self.main_ribbon.insertWidget(actions[0] if actions else None, label)
            self._toolbar_logo_added = True
        except Exception:
            pass

    def _apply_application_branding(self) -> None:
        try:
            qss = load_theme("light")
            if qss:
                self.setStyleSheet(qss)
        except Exception:
            pass
        try:
            _ico = app_icon()
            if _ico is not None and not _ico.isNull():
                self.setWindowIcon(_ico)
        except Exception:
            pass
        self._apply_branding_to_ribbon_actions()
        self._ensure_toolbar_logo()
        try:
            if hasattr(self, "main_ribbon"):
                self.main_ribbon.setIconSize(QSize(24, 24))
        except Exception:
            pass
        try:
            apply_widget_branding(self)
        except Exception:
            pass
