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


class MainWindowImportMixin:
    """Answer-key import, answer-key sample export and source file loading entry points."""
    def action_load_answer_keys(self) -> None:
        self.load_answer_keys()

    def action_import_answer_key(self) -> None:
        self.import_answer_key_file()

    def import_answer_key_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Answer Key",
            "",
            "Answer key files (*.xlsx *.csv)",
        )
        if not file_path:
            return
        try:
            imported_package = import_answer_key(file_path)
        except Exception as exc:
            message = (
                f"Cannot import answer key:\n{exc}\n\n"
                "Continue importing only valid rows?\n"
                "- Yes: Continue and AWARD FULL SCORE for invalid-answer questions.\n"
                "- No: Continue but SKIP invalid-answer questions.\n"
                "- Cancel: Stop import."
            )
            choose = QMessageBox.question(
                self,
                "Import failed",
                message,
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if choose == QMessageBox.Cancel:
                return
            try:
                imported_package = import_answer_key(
                    file_path,
                    strict=False,
                    award_full_credit_for_invalid=(choose == QMessageBox.Yes),
                )
            except Exception as inner_exc:
                QMessageBox.warning(self, "Import failed", f"Cannot import answer key:\n{inner_exc}")
                return

        if imported_package.warnings:
            QMessageBox.information(
                self,
                "Import warnings",
                "Imported with warnings:\n- " + "\n- ".join(imported_package.warnings[:20])
                + ("\n..." if len(imported_package.warnings) > 20 else ""),
            )

        from gui.import_answer_key_dialog import ImportAnswerKeyDialog

        dlg = ImportAnswerKeyDialog(imported_package, self)
        if dlg.exec() != QDialog.Accepted:
            return
        edited_package = dlg.result_answer_key()

        if not self.session:
            self.create_session()
        subject = self.session.subjects[0] if self.session and self.session.subjects else "General"

        if self.answer_keys is None:
            self.answer_keys = AnswerKeyRepository()

        imported_count = 0
        self.imported_exam_codes = sorted(set(edited_package.exam_keys.keys()))
        for exam_code, edited in edited_package.exam_keys.items():
            code = exam_code.strip() or "DEFAULT"
            self.answer_keys.upsert(
                SubjectKey(
                    subject=subject,
                    exam_code=code,
                    answers=edited.mcq_answers,
                    true_false_answers=edited.true_false_answers,
                    numeric_answers=edited.numeric_answers,
                    full_credit_questions=edited.full_credit_questions,
                    invalid_answer_rows=edited.invalid_answer_rows,
                )
            )
            imported_count += 1

        if self.session:
            self.session.answer_key_path = file_path
        self.active_batch_subject_key = subject
        self.session_dirty = True
        self._refresh_session_info()
        self._refresh_batch_subject_controls()
        self._refresh_batch_subject_controls()
        self._retrim_batch_results_to_answer_key_scope()
        QMessageBox.information(self, "Import successful", f"Imported {imported_count} exam code(s) into current session.")

    def load_answer_keys(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Answer Keys", "", "JSON (*.json)")
        if not file_path:
            return
        self.answer_keys = AnswerKeyRepository.load_json(file_path)
        self.imported_exam_codes = sorted({k.split("::", 1)[1] for k in self.answer_keys.keys.keys() if "::" in k})
        if self.session:
            self.session.answer_key_path = file_path
        self.session_dirty = True
        self._refresh_session_info()
