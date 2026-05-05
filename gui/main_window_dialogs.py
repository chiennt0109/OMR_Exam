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
    QListWidgetItem,
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

# Extracted dialogs/widgets from main_window.py.

class PreviewImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(260)
        self._overlay_markers: list[dict[str, float]] = []

    def clear_markers(self) -> None:
        self._overlay_markers = []
        self.update()

    def set_overlay_markers(self, markers: list[dict[str, float]]) -> None:
        self._overlay_markers = [dict(m) for m in markers]
        self.update()

    def paintEvent(self, event):  # type: ignore[override]
        super().paintEvent(event)
        if not self._overlay_markers:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self._overlay_markers:
            painter.setPen(QPen(QColor(40, 130, 255), 2))
            for m in self._overlay_markers:
                x = float(m.get("x", 0.0))
                y = float(m.get("y", 0.0))
                r = 5
                painter.drawLine(int(x - r), int(y - r), int(x + r), int(y + r))
                painter.drawLine(int(x - r), int(y + r), int(x + r), int(y - r))

class SubjectConfigDialog(QDialog):
    @staticmethod
    def default_section_scores() -> dict:
        return {
            "MCQ": {"total_points": 3.0, "distribution": "auto_by_question_count"},
            "TF": {
                "total_points": 2.0,
                "rule_per_question": {"1": 0.1, "2": 0.25, "3": 0.5, "4": 1.0},
            },
            "NUMERIC": {"total_points": 2.0, "distribution": "auto_by_question_count"},
        }

    @staticmethod
    def default_question_scores() -> dict:
        return {
            "MCQ": {"per_question": 0.25},
            "TF": {"1": 0.1, "2": 0.25, "3": 0.5, "4": 1.0},
            "NUMERIC": {"per_question": 1.0},
        }

    @staticmethod
    def _to_float(text: str, fallback: float = 0.0) -> float:
        try:
            return float((text or "").strip().replace(",", "."))
        except Exception:
            return fallback

    @staticmethod
    def _template_question_counts(template_path: str) -> dict[str, int]:
        counts = {"MCQ": 0, "TF": 0, "NUMERIC": 0}
        if not template_path:
            return counts
        path = Path(template_path)
        if not path.exists():
            return counts
        try:
            tpl = Template.load_json(path)
        except Exception:
            return counts
        for z in tpl.zones:
            if not z.grid:
                continue
            if z.zone_type.value == "MCQ_BLOCK":
                c = int(z.grid.question_count or z.grid.rows or 0)
                counts["MCQ"] += max(0, c)
            elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                # TF grid usually has rows = questions * statements_per_question.
                # `grid.question_count` can be stale/legacy in some templates, so derive from rows first.
                spq = max(1, int(z.metadata.get("statements_per_question", 4) or 4))
                from_rows = int((z.grid.rows or 0) // spq)
                from_meta = int(z.metadata.get("questions_per_block", 0) or 0)
                from_grid = int(z.grid.question_count or 0)
                candidates = [x for x in [from_rows, from_meta, from_grid] if x > 0]
                c = min(candidates) if candidates else 0
                counts["TF"] += max(0, c)
            elif z.zone_type.value == "NUMERIC_BLOCK":
                # Numeric grid usually has cols = questions * digits_per_answer.
                dpa = max(1, int(z.metadata.get("digits_per_answer", 3) or 3))
                from_cols = int((z.grid.cols or 0) // dpa)
                from_meta = int(z.metadata.get("questions_per_block", z.metadata.get("total_questions", 0)) or 0)
                from_grid = int(z.grid.question_count or 0)
                candidates = [x for x in [from_cols, from_meta, from_grid] if x > 0]
                c = min(candidates) if candidates else 0
                counts["NUMERIC"] += max(0, c)
        return counts

    @staticmethod
    def _answer_key_question_counts(answer_key_data: dict) -> dict[str, int]:
        counts = {"MCQ": 0, "TF": 0, "NUMERIC": 0}
        if not isinstance(answer_key_data, dict) or not answer_key_data:
            return counts
        # Total score is defined per one exam code; use the first code as representative.
        first_code = sorted(answer_key_data.keys())[0]
        key_data = answer_key_data.get(first_code, {}) or {}
        for sec, bucket_name in [("MCQ", "mcq_answers"), ("TF", "true_false_answers"), ("NUMERIC", "numeric_answers")]:
            valid_qs = {
                int(q) for q in (key_data.get(bucket_name, {}) or {}).keys()
                if str(q).strip().lstrip("-").isdigit()
            }
            full_qs = {
                int(q) for q in ((key_data.get("full_credit_questions", {}) or {}).get(sec, []) or [])
                if str(q).strip().lstrip("-").isdigit()
            }
            invalid_qs = {
                int(q) for q in ((key_data.get("invalid_answer_rows", {}) or {}).get(sec, {}) or {}).keys()
                if str(q).strip().lstrip("-").isdigit()
            }
            counts[sec] = len(valid_qs | full_qs | invalid_qs)
        return counts

    @staticmethod
    def _template_part_count(template_path: str, fallback: int = 3) -> int:
        counts = SubjectConfigDialog._template_question_counts(template_path)
        parts = sum(1 for k in counts if counts[k] > 0)
        return parts if parts > 0 else fallback

    def __init__(
        self,
        data: dict | None = None,
        subject_options: list[str] | None = None,
        block_options: list[str] | None = None,
        paper_part_count: int = 3,
        common_template_path: str = "",
        template_repo: TemplateRepository | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Cấu hình môn học")
        self.resize(980, 760)
        self.setMinimumSize(880, 680)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        try:
            _ico = app_icon()
            if _ico is not None and not _ico.isNull():
                self.setWindowIcon(_ico)
        except Exception:
            pass
        data = data or {}
        subject_options = subject_options or []
        block_options = block_options or ["10", "11", "12"]

        self.common_template_path = common_template_path
        self.template_repo = template_repo or TemplateRepository()
        self.paper_part_count_default = paper_part_count
        self.answer_key_data: dict = dict(data.get("imported_answer_keys", {}))
        if not self.answer_key_data:
            db = getattr(parent, "database", None)
            subject_key_seed = str(data.get("answer_key_key", "") or "").strip()
            if not subject_key_seed:
                seed_name = str(data.get("name", "") or "").strip()
                seed_block = str(data.get("block", "") or "").strip()
                subject_key_seed = f"{seed_name}_{seed_block}" if seed_name and seed_block else ""
            if db is not None and subject_key_seed:
                try:
                    scoped_seed = subject_key_seed
                    if parent is not None and hasattr(parent, "_answer_key_scope_key"):
                        try:
                            scoped_seed = parent._answer_key_scope_key(subject_key_seed, str(data.get("block", "") or ""))
                        except Exception:
                            scoped_seed = subject_key_seed
                    fetched = db.fetch_answer_keys_for_subject(scoped_seed)
                    if not fetched and scoped_seed != subject_key_seed:
                        fetched = db.fetch_answer_keys_for_subject(subject_key_seed)
                    if fetched:
                        self.answer_key_data = fetched
                except Exception:
                    pass

        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.subject_name = QComboBox(); self.subject_name.setEditable(True); self.subject_name.addItems(subject_options)
        if str(data.get("name", "")).strip():
            self.subject_name.setCurrentText(str(data.get("name", "")).strip())

        self.block_name = QComboBox(); self.block_name.setEditable(True); self.block_name.addItems(block_options)
        self.block_name.setCurrentText(str(data.get("block", block_options[0] if block_options else "10")))
        self.block_name.setMinimumContentsLength(16)
        self.block_name.setMinimumWidth(220)
        self.is_essay_subject = QCheckBox("Môn tự luận")
        self.is_essay_subject.setChecked(bool(data.get("is_essay_subject", False)))
        self.auto_recognize = QCheckBox("Tự động nhận dạng")
        self.auto_recognize.setChecked(bool(data.get("auto_recognize", False)))

        self.template_path = QLineEdit(str(data.get("template_path", "")))
        self.scan_folder = QLineEdit(str(data.get("scan_folder", "")))
        self.answer_key = QLineEdit(str(data.get("answer_key_path", "")))
        self.answer_key_key = QLineEdit(str(data.get("answer_key_key", ""))); self.answer_key_key.setReadOnly(True)
        self._fit_line_edit_to_text(self.template_path, min_chars=42, max_chars=78, padding=30)
        self._fit_line_edit_to_text(self.scan_folder, min_chars=42, max_chars=78, padding=30)
        self._fit_line_edit_to_text(self.answer_key, min_chars=42, max_chars=78, padding=30)
        self._fit_line_edit_to_text(self.answer_key_key, min_chars=20, max_chars=36, padding=28)
        self.template_path.setClearButtonEnabled(True)
        self.scan_folder.setClearButtonEnabled(True)
        self.answer_key.setClearButtonEnabled(True)
        self.template_path.textChanged.connect(lambda _text: self._fit_line_edit_to_text(self.template_path, min_chars=42, max_chars=78, padding=30))
        self.scan_folder.textChanged.connect(lambda _text: self._fit_line_edit_to_text(self.scan_folder, min_chars=42, max_chars=78, padding=30))
        self.answer_key.textChanged.connect(lambda _text: self._fit_line_edit_to_text(self.answer_key, min_chars=42, max_chars=78, padding=30))
        self.answer_key_key.textChanged.connect(lambda _text: self._fit_line_edit_to_text(self.answer_key_key, min_chars=20, max_chars=36, padding=28))
        self._exam_room_mapping_cache: dict[str, list[str]] = {}
        self.exam_room_name = QComboBox(); self.exam_room_name.setEditable(True)
        seed_room = str(data.get("exam_room_name", "") or "").strip()
        if seed_room:
            self.exam_room_name.addItem(seed_room)
            self.exam_room_name.setCurrentText(seed_room)
        self.exam_room_mapping_selector = QComboBox()
        self.exam_room_mapping_selector.setMinimumWidth(260)
        self.exam_room_mapping_selector.currentIndexChanged.connect(self._on_exam_room_mapping_selected)
        self.btn_import_exam_room_mapping = QPushButton("Import...")
        self.btn_import_exam_room_mapping.clicked.connect(self._import_exam_room_mapping_from_file)
        self.btn_delete_exam_room_mapping = QPushButton("Xóa phòng")
        self.btn_delete_exam_room_mapping.clicked.connect(self._remove_selected_exam_room_mapping)
        self.btn_delete_multi_exam_room_mapping = QPushButton("Xóa nhiều...")
        self.btn_delete_multi_exam_room_mapping.clicked.connect(self._remove_multiple_exam_room_mappings)
        self.exam_room_mapping_hint = QLabel("Chưa nạp mapping SBD/phòng.")
        self.exam_room_mapping_hint.setWordWrap(False)
        room_map_wrap = QWidget()
        room_map_lay = QHBoxLayout(room_map_wrap)
        room_map_lay.setContentsMargins(0, 0, 0, 0)
        room_map_lay.setSpacing(8)
        room_map_lay.addWidget(self.exam_room_mapping_selector, 0)
        room_map_lay.addWidget(self.btn_import_exam_room_mapping, 0)
        room_map_lay.addWidget(self.btn_delete_exam_room_mapping, 0)
        room_map_lay.addWidget(self.btn_delete_multi_exam_room_mapping, 0)
        room_map_lay.addWidget(self.exam_room_mapping_hint, 1)
        self._restore_exam_room_mapping_from_data(data, seed_room)
        self.answer_codes = QLineEdit(", ".join(sorted((data.get("imported_answer_keys") or {}).keys()))); self.answer_codes.setReadOnly(True)
        self._fit_line_edit_to_text(self.answer_codes, min_chars=24, max_chars=56, padding=28)
        self.answer_codes.textChanged.connect(lambda _text: self._fit_line_edit_to_text(self.answer_codes, min_chars=24, max_chars=56, padding=28))
        self.answer_summary = QTextEdit()
        self.answer_summary.setReadOnly(True)
        self.answer_summary.setFixedSize(640, 170)
        self.answer_summary.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.answer_summary.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.direct_score_import_data: dict = copy.deepcopy(data.get("direct_score_import", {}) or {})
        self.direct_score_import_path = QLineEdit(str(self.direct_score_import_data.get("path", "") or ""))
        self.direct_score_import_path.setReadOnly(True)
        self._fit_line_edit_to_text(self.direct_score_import_path, min_chars=42, max_chars=78, padding=30)
        self.direct_score_import_path.textChanged.connect(lambda _text: self._fit_line_edit_to_text(self.direct_score_import_path, min_chars=42, max_chars=78, padding=30))
        self.btn_import_direct_score = QPushButton("Import file điểm...")
        self.btn_import_direct_score.clicked.connect(self._import_direct_score_file)
        self.direct_score_import_summary = QTextEdit()
        self.direct_score_import_summary.setReadOnly(True)
        self.direct_score_import_summary.setFixedSize(640, 160)
        self.direct_score_import_summary.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.direct_score_import_summary.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._refresh_direct_score_import_summary()

        self.paper_part_label = QLabel(str(paper_part_count))

        self.score_mode = QComboBox(); self.score_mode.addItems(["Điểm theo phần", "Điểm theo câu"])
        self.score_mode.setCurrentText(str(data.get("score_mode", "Điểm theo phần")))

        sec = data.get("section_scores", self.default_section_scores())
        self.sec_mcq_total = QLineEdit(str((sec.get("MCQ") or {}).get("total_points", 3.0)))
        self.sec_tf_total = QLineEdit(str((sec.get("TF") or {}).get("total_points", 2.0)))
        self.sec_tf_1 = QLineEdit(str(((sec.get("TF") or {}).get("rule_per_question") or {}).get("1", 0.1)))
        self.sec_tf_2 = QLineEdit(str(((sec.get("TF") or {}).get("rule_per_question") or {}).get("2", 0.25)))
        self.sec_tf_3 = QLineEdit(str(((sec.get("TF") or {}).get("rule_per_question") or {}).get("3", 0.5)))
        self.sec_tf_4 = QLineEdit(str(((sec.get("TF") or {}).get("rule_per_question") or {}).get("4", 1.0)))
        self.sec_numeric_total = QLineEdit(str((sec.get("NUMERIC") or {}).get("total_points", 2.0)))

        qsc = data.get("question_scores", self.default_question_scores())
        self.q_mcq = QLineEdit(str((qsc.get("MCQ") or {}).get("per_question", 0.25)))
        self.q_tf_1 = QLineEdit(str((qsc.get("TF") or {}).get("1", 0.1)))
        self.q_tf_2 = QLineEdit(str((qsc.get("TF") or {}).get("2", 0.25)))
        self.q_tf_3 = QLineEdit(str((qsc.get("TF") or {}).get("3", 0.5)))
        self.q_tf_4 = QLineEdit(str((qsc.get("TF") or {}).get("4", 1.0)))
        self.q_numeric = QLineEdit(str((qsc.get("NUMERIC") or {}).get("per_question", 1.0)))

        self.total_score = QLineEdit(); self.total_score.setReadOnly(True)
        self.total_score.setMaximumWidth(120)

        self._score_line_edits = [
            self.sec_mcq_total, self.sec_tf_total, self.sec_tf_1, self.sec_tf_2, self.sec_tf_3, self.sec_tf_4, self.sec_numeric_total,
            self.q_mcq, self.q_tf_1, self.q_tf_2, self.q_tf_3, self.q_tf_4, self.q_numeric, self.total_score,
        ]
        for _score_edit in self._score_line_edits:
            _score_edit.setAlignment(Qt.AlignRight)
            self._fit_line_edit_to_text(_score_edit)
            _score_edit.textChanged.connect(lambda _text, widget=_score_edit: self._fit_line_edit_to_text(widget))

        row_tpl = QHBoxLayout(); row_tpl.setContentsMargins(0, 0, 0, 0); row_tpl.addWidget(self.template_path); b_tpl = QPushButton("..."); row_tpl.addWidget(b_tpl); b_tpl_repo = QPushButton("Kho mẫu..."); row_tpl.addWidget(b_tpl_repo); row_tpl.addStretch(1)
        b_tpl.clicked.connect(self._browse_template)
        b_tpl_repo.clicked.connect(self._pick_template_from_repo)
        row_scan = QHBoxLayout(); row_scan.setContentsMargins(0, 0, 0, 0); row_scan.addWidget(self.scan_folder); b_scan = QPushButton("..."); row_scan.addWidget(b_scan); row_scan.addStretch(1)
        b_scan.clicked.connect(self._browse_scan_folder)
        row_key = QHBoxLayout(); row_key.setContentsMargins(0, 0, 0, 0); row_key.addWidget(self.answer_key); b_key = QPushButton("..."); row_key.addWidget(b_key)
        b_key_view = QPushButton("Xem/Sửa đáp án...")
        row_key.addWidget(b_key_view)
        row_key.addStretch(1)
        b_key.clicked.connect(self._browse_answer_key)
        b_key_view.clicked.connect(self._edit_current_answer_keys)

        subject_row_wrap = QWidget()
        subject_row_lay = QHBoxLayout(subject_row_wrap)
        subject_row_lay.setContentsMargins(0, 0, 0, 0)
        subject_row_lay.setSpacing(10)
        subject_row_lay.addWidget(self.subject_name, 1)
        subject_row_lay.addWidget(self.is_essay_subject, 0, Qt.AlignRight)

        top_form = QFormLayout()
        top_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        top_form.addRow("Tên môn", subject_row_wrap)
        top_form.addRow("Khối", self.block_name)
        lay.addLayout(top_form)

        self.standard_config_widget = QWidget()
        standard_lay = QVBoxLayout(self.standard_config_widget)
        standard_lay.setContentsMargins(0, 0, 0, 0)
        standard_lay.setSpacing(10)
        standard_form = QFormLayout()
        standard_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        standard_form.addRow("Giấy thi riêng (tùy chọn)", row_tpl)
        standard_form.addRow("Thư mục bài thi môn", row_scan)
        standard_form.addRow("", self.auto_recognize)
        standard_form.addRow("Đáp án môn", row_key)
        standard_form.addRow("Mã đáp án môn_khối", self.answer_key_key)
        standard_form.addRow("Danh sách SBD phòng thi", room_map_wrap)
        standard_form.addRow("Các mã đề của môn", self.answer_codes)
        standard_form.addRow("Tóm tắt đáp án", self.answer_summary)
        standard_form.addRow("Số phần giấy thi", self.paper_part_label)
        standard_form.addRow("Cách nhập điểm", self.score_mode)

        self.section_group = QGroupBox("Điểm theo phần")
        sec_form = QFormLayout(self.section_group)
        sec_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        sec_form.addRow("MCQ tổng điểm", self.sec_mcq_total)
        sec_form.addRow("TF tổng điểm", self.sec_tf_total)
        sec_form.addRow("TF đúng 1 ý", self.sec_tf_1)
        sec_form.addRow("TF đúng 2 ý", self.sec_tf_2)
        sec_form.addRow("TF đúng 3 ý", self.sec_tf_3)
        sec_form.addRow("TF đúng 4 ý", self.sec_tf_4)
        sec_form.addRow("NUMERIC tổng điểm", self.sec_numeric_total)

        self.question_group = QGroupBox("Điểm theo câu")
        q_form = QFormLayout(self.question_group)
        q_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        q_form.addRow("MCQ điểm/câu", self.q_mcq)
        q_form.addRow("TF đúng 1 ý", self.q_tf_1)
        q_form.addRow("TF đúng 2 ý", self.q_tf_2)
        q_form.addRow("TF đúng 3 ý", self.q_tf_3)
        q_form.addRow("TF đúng 4 ý", self.q_tf_4)
        q_form.addRow("NUMERIC điểm/câu", self.q_numeric)

        standard_form.addRow("Tổng điểm bài thi", self.total_score)
        standard_lay.addLayout(standard_form)
        standard_lay.addWidget(self.section_group)
        standard_lay.addWidget(self.question_group)
        lay.addWidget(self.standard_config_widget)

        self.essay_import_group = QGroupBox("Import điểm trực tiếp theo SBD")
        essay_form = QFormLayout(self.essay_import_group)
        essay_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        direct_file_row = QHBoxLayout()
        direct_file_row.setContentsMargins(0, 0, 0, 0)
        direct_file_row.addWidget(self.direct_score_import_path)
        direct_file_row.addWidget(self.btn_import_direct_score)
        direct_file_row.addStretch(1)
        essay_form.addRow("File điểm", direct_file_row)
        direct_note = QLabel("Bắt buộc chọn cột SBD và cột Điểm. Cột Phòng thi là tùy chọn; nếu không có, hệ thống sẽ dùng phòng thi chung của kỳ thi.")
        direct_note.setWordWrap(True)
        essay_form.addRow("Quy tắc import", direct_note)
        essay_form.addRow("Tóm tắt import", self.direct_score_import_summary)
        lay.addWidget(self.essay_import_group)

        self.subject_name.currentTextChanged.connect(self._update_answer_key_key)
        self.block_name.currentTextChanged.connect(self._update_answer_key_key)
        self.is_essay_subject.toggled.connect(self._refresh_subject_mode_ui)
        self.score_mode.currentTextChanged.connect(self._refresh_score_mode_ui)
        self.template_path.textChanged.connect(self._update_paper_parts)

        for w in [self.sec_mcq_total, self.sec_tf_total, self.sec_tf_1, self.sec_tf_2, self.sec_tf_3, self.sec_tf_4, self.sec_numeric_total,
                  self.q_mcq, self.q_tf_1, self.q_tf_2, self.q_tf_3, self.q_tf_4, self.q_numeric]:
            w.textChanged.connect(self._update_total_score)

        self._update_answer_key_key()
        self._refresh_answer_key_summary()
        self._update_paper_parts()
        self._refresh_score_mode_ui()
        self._refresh_subject_mode_ui()

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @staticmethod
    def _normalize_exam_room_sid_values(raw_values: object) -> list[str]:
        tokens: list[str] = []
        if isinstance(raw_values, str):
            tokens = re.split(r"[,\n;]+", raw_values)
        elif isinstance(raw_values, (list, tuple, set)):
            tokens = [str(x) for x in raw_values]
        else:
            return []
        clean = [str(x or "").strip() for x in tokens if str(x or "").strip()]
        return sorted(set(clean))

    def _restore_exam_room_mapping_from_data(self, data: dict, seed_room: str = "") -> None:
        rebuilt: dict[str, list[str]] = {}
        mapping_by_room_seed = data.get("exam_room_sbd_mapping_by_room", {})
        if isinstance(mapping_by_room_seed, dict) and mapping_by_room_seed:
            for room_key, values in mapping_by_room_seed.items():
                room_text = str(room_key or "").strip()
                if not room_text:
                    continue
                normalized_values = self._normalize_exam_room_sid_values(values)
                if normalized_values:
                    rebuilt[room_text] = normalized_values

        if not rebuilt:
            mapping_seed = str(data.get("exam_room_sbd_mapping", "") or "").strip()
            normalized_values = self._normalize_exam_room_sid_values(mapping_seed)
            if normalized_values:
                room_key = str(seed_room or "").strip() or "[Không rõ phòng]"
                rebuilt[room_key] = normalized_values

        self._exam_room_mapping_cache = dict(rebuilt)
        self._refresh_exam_room_mapping_selector()

        preferred_room = str(seed_room or "").strip()
        if preferred_room and preferred_room in self._exam_room_mapping_cache:
            self.exam_room_name.setCurrentText(preferred_room)
            idx = self.exam_room_mapping_selector.findData(preferred_room)
            if idx >= 0:
                self.exam_room_mapping_selector.setCurrentIndex(idx)
        elif self._exam_room_mapping_cache:
            if not self.exam_room_name.currentText().strip():
                first_room = sorted(self._exam_room_mapping_cache.keys())[0]
                self.exam_room_name.setCurrentText(first_room)
            current_room = self.exam_room_name.currentText().strip()
            idx = self.exam_room_mapping_selector.findData(current_room)
            if idx >= 0:
                self.exam_room_mapping_selector.setCurrentIndex(idx)
        if not self._exam_room_mapping_cache:
            self.exam_room_mapping_hint.setText("Chưa nạp mapping SBD/phòng.")

    def _update_paper_parts(self) -> None:
        tpl = self.template_path.text().strip() or self.common_template_path
        part_count = self._template_part_count(tpl, self.paper_part_count_default)
        self.paper_part_label.setText(str(part_count))
        self._update_total_score()

    def _question_counts(self) -> dict[str, int]:
        key_counts = self._answer_key_question_counts(self.answer_key_data)
        if any(key_counts.values()):
            return key_counts
        tpl = self.template_path.text().strip() or self.common_template_path
        return self._template_question_counts(tpl)

    @staticmethod
    def _fit_line_edit_to_text(widget: QLineEdit, min_chars: int = 5, max_chars: int = 14, padding: int = 24) -> None:
        text = str(widget.text() or "").strip()
        metrics = widget.fontMetrics()
        min_width = metrics.horizontalAdvance("0" * max(1, min_chars)) + padding
        sample = text if text else ("0" * max(1, min_chars))
        width = metrics.horizontalAdvance(sample[:max_chars]) + padding
        width = max(min_width, width)
        width = min(width, metrics.horizontalAdvance("0" * max_chars) + padding)
        widget.setMinimumWidth(width)
        widget.setMaximumWidth(16777215)
        widget.setToolTip(widget.text() or widget.placeholderText() or "")

    def _refresh_score_mode_ui(self) -> None:
        keep_size = self.size()
        section_mode = self.score_mode.currentText() == "Điểm theo phần"
        self.section_group.setVisible(section_mode)
        self.question_group.setVisible(not section_mode)
        self.layout().activate()
        if keep_size.isValid() and not self.isMaximized():
            self.resize(keep_size)
        self._update_total_score()

    def _update_total_score(self) -> None:
        if self.score_mode.currentText() == "Điểm theo phần":
            total = (
                self._to_float(self.sec_mcq_total.text())
                + self._to_float(self.sec_tf_total.text())
                + self._to_float(self.sec_numeric_total.text())
            )
        else:
            counts = self._question_counts()
            tf_max = max(
                self._to_float(self.q_tf_1.text()),
                self._to_float(self.q_tf_2.text()),
                self._to_float(self.q_tf_3.text()),
                self._to_float(self.q_tf_4.text()),
            )
            total = (
                self._to_float(self.q_mcq.text()) * counts.get("MCQ", 0)
                + tf_max * counts.get("TF", 0)
                + self._to_float(self.q_numeric.text()) * counts.get("NUMERIC", 0)
            )
        self.total_score.setText(f"{round(total, 4)}")

    def _update_answer_key_key(self) -> None:
        subject = self.subject_name.currentText().strip()
        block = self.block_name.currentText().strip()
        self.answer_key_key.setText(f"{subject}_{block}" if subject and block else "")

    def _browse_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn giấy thi", "", "JSON (*.json)")
        if path:
            self.template_repo.register(path)
            self.template_path.setText(path)


    def _pick_template_from_repo(self) -> None:
        items = [f"{name} | {path}" for name, path in self.template_repo.list_templates()]
        if not items:
            QMessageBox.information(self, "Kho mẫu giấy thi", "Kho mẫu đang trống. Hãy thêm mẫu bằng nút ...")
            return
        chosen, ok = QInputDialog.getItem(self, "Kho mẫu giấy thi", "Chọn mẫu:", items, 0, False)
        if ok and chosen:
            self.template_path.setText(chosen.split(" | ", 1)[1])

    def _browse_scan_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục bài thi môn")
        if path:
            self.scan_folder.setText(path)

    def _browse_answer_key(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn đáp án môn", "", "Answer files (*.json *.xlsx *.csv)")
        if not path:
            return
        try:
            imported_package = import_answer_key(path)
        except Exception as exc:
            message = (
                f"Không thể import đáp án:\n{exc}\n\n"
                "Bạn có muốn tiếp tục import các câu hợp lệ không?\n"
                "- Yes: Vẫn import và CHO ĐIỂM TỐI ĐA cho câu đáp án không đúng chuẩn.\n"
                "- No: Vẫn import nhưng BỎ QUA câu đáp án không đúng chuẩn (không chấm câu đó).\n"
                "- Cancel: Hủy import."
            )
            choose = QMessageBox.question(
                self,
                "Import đáp án",
                message,
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if choose == QMessageBox.Cancel:
                return
            try:
                imported_package = import_answer_key(
                    path,
                    strict=False,
                    award_full_credit_for_invalid=(choose == QMessageBox.Yes),
                )
            except Exception as inner_exc:
                QMessageBox.warning(self, "Import đáp án", f"Không thể import đáp án:\n{inner_exc}")
                return

        if imported_package.warnings:
            QMessageBox.information(
                self,
                "Import đáp án",
                "Import hoàn tất với cảnh báo:\n- " + "\n- ".join(imported_package.warnings[:20])
                + ("\n..." if len(imported_package.warnings) > 20 else ""),
            )

        from gui.import_answer_key_dialog import ImportAnswerKeyDialog

        dlg = ImportAnswerKeyDialog(imported_package, self)
        if dlg.exec() != QDialog.Accepted:
            return
        edited_package = dlg.result_answer_key()
        if not edited_package.exam_keys:
            QMessageBox.warning(self, "Import đáp án", "Không có mã đề nào trong file đáp án.")
            return

        # One subject-block can have multiple exam codes.
        self.answer_key_data = {}
        for code, key in edited_package.exam_keys.items():
            self.answer_key_data[code] = {
                "mcq_answers": key.mcq_answers,
                "true_false_answers": key.true_false_answers,
                "numeric_answers": key.numeric_answers,
                "full_credit_questions": key.full_credit_questions,
                "invalid_answer_rows": key.invalid_answer_rows,
            }
        self.answer_codes.setText(", ".join(sorted(self.answer_key_data.keys())))
        self.answer_key.setText(path)
        self._refresh_answer_key_summary()
        self._update_total_score()
        QMessageBox.information(self, "Import đáp án", "Đã gắn toàn bộ mã đề của file đáp án cho môn đang cấu hình.")

    @staticmethod
    def _build_imported_package_from_answer_data(answer_key_data: dict) -> ImportedAnswerKeyPackage:
        package = ImportedAnswerKeyPackage()
        for exam_code, payload in sorted((answer_key_data or {}).items()):
            if not isinstance(payload, dict):
                continue
            key = ImportedAnswerKey()
            key.mcq_answers = {
                int(k): str(v)
                for k, v in (payload.get("mcq_answers", {}) or {}).items()
                if str(k).strip().lstrip("-").isdigit()
            }
            key.true_false_answers = {
                int(k): {str(sub): bool(flag) for sub, flag in (flags or {}).items()}
                for k, flags in (payload.get("true_false_answers", {}) or {}).items()
                if str(k).strip().lstrip("-").isdigit()
            }
            key.numeric_answers = {
                int(k): str(v)
                for k, v in (payload.get("numeric_answers", {}) or {}).items()
                if str(k).strip().lstrip("-").isdigit()
            }
            key.full_credit_questions = {
                str(sec): [int(x) for x in (vals or []) if str(x).strip().lstrip("-").isdigit()]
                for sec, vals in (payload.get("full_credit_questions", {}) or {}).items()
            }
            key.invalid_answer_rows = {
                str(sec): {int(k): str(v) for k, v in (vals or {}).items()}
                for sec, vals in (payload.get("invalid_answer_rows", {}) or {}).items()
            }
            package.exam_keys[str(exam_code)] = key
        return package

    @staticmethod
    def _answer_payload_from_package(package: ImportedAnswerKeyPackage) -> dict[str, dict]:
        payload: dict[str, dict] = {}
        for code, key in (package.exam_keys or {}).items():
            payload[str(code)] = {
                "mcq_answers": {int(k): str(v) for k, v in (key.mcq_answers or {}).items()},
                "true_false_answers": {
                    int(k): {str(sub): bool(flag) for sub, flag in (flags or {}).items()}
                    for k, flags in (key.true_false_answers or {}).items()
                },
                "numeric_answers": {int(k): str(v) for k, v in (key.numeric_answers or {}).items()},
                "full_credit_questions": {
                    str(sec): [int(x) for x in (vals or []) if str(x).strip().lstrip("-").isdigit()]
                    for sec, vals in (key.full_credit_questions or {}).items()
                },
                "invalid_answer_rows": {
                    str(sec): {int(k): str(v) for k, v in (vals or {}).items()}
                    for sec, vals in (key.invalid_answer_rows or {}).items()
                },
            }
        return payload

    @staticmethod
    def _describe_answer_key_data(answer_key_data: dict) -> str:
        if not isinstance(answer_key_data, dict) or not answer_key_data:
            return "Chưa có đáp án cho môn này."
        lines: list[str] = []
        for exam_code, payload in sorted(answer_key_data.items()):
            if not isinstance(payload, dict):
                continue
            lines.append(f"Mã đề {exam_code}:")
            mcq = ", ".join(
                f"C{int(q)}:{str(a)}" for q, a in sorted((payload.get('mcq_answers', {}) or {}).items(), key=lambda item: int(item[0]))
            ) or "-"
            tf = ", ".join(
                f"C{int(q)}:{''.join('Đ' if bool((flags or {}).get(ch)) else 'S' for ch in ['a','b','c','d'])}"
                for q, flags in sorted((payload.get("true_false_answers", {}) or {}).items(), key=lambda item: int(item[0]))
            ) or "-"
            numeric = ", ".join(
                f"C{int(q)}:{str(a)}" for q, a in sorted((payload.get('numeric_answers', {}) or {}).items(), key=lambda item: int(item[0]))
            ) or "-"
            invalid_descriptions: list[str] = []
            for sec, invalid_rows in sorted((payload.get("invalid_answer_rows", {}) or {}).items()):
                if not invalid_rows:
                    continue
                mode = "cho điểm tối đa/giữ mô tả nhập sai"
                vals = ", ".join(f"C{int(q)}:{str(v)}" for q, v in sorted(invalid_rows.items(), key=lambda item: int(item[0])))
                invalid_descriptions.append(f"{sec} [{mode}] {vals}")
            lines.append(f"  - MCQ: {mcq}")
            lines.append(f"  - TF: {tf}")
            lines.append(f"  - NUMERIC: {numeric}")
            if invalid_descriptions:
                lines.append("  - Dòng nhập sai vẫn giữ để chấm:")
                lines.extend(f"    * {item}" for item in invalid_descriptions)
        return "\n".join(lines)

    def _refresh_answer_key_summary(self) -> None:
        self.answer_codes.setText(", ".join(sorted(self.answer_key_data.keys())))
        self.answer_summary.setPlainText(self._describe_answer_key_data(self.answer_key_data))

    def _refresh_subject_mode_ui(self) -> None:
        is_essay = self.is_essay_subject.isChecked()
        self.standard_config_widget.setVisible(not is_essay)
        self.essay_import_group.setVisible(is_essay)

    def _refresh_direct_score_import_summary(self) -> None:
        payload = self.direct_score_import_data if isinstance(self.direct_score_import_data, dict) else {}
        rows = payload.get("rows", []) or []
        path_text = str(payload.get("path", "") or "")
        mapping = payload.get("mapping", {}) or {}
        self.direct_score_import_path.setText(path_text)
        if not rows:
            self.direct_score_import_summary.setPlainText(
                "Chưa nạp file điểm trực tiếp.\n\n"
                "Yêu cầu tối thiểu:\n"
                "- Có cột SBD\n"
                "- Có cột Điểm\n"
                "- Cột Phòng thi là tùy chọn; nếu không có sẽ dùng phòng thi chung của kỳ thi."
            )
            return
        preview_lines = []
        for rec in rows[:8]:
            sid = str((rec or {}).get("sid", "") or "").strip()
            score = str((rec or {}).get("score", "") or "").strip()
            room = str((rec or {}).get("room", "") or "").strip() or "[dùng phòng chung của kỳ thi]"
            preview_lines.append(f"- SBD {sid}: điểm {score} | phòng {room}")
        if len(rows) > 8:
            preview_lines.append(f"- ... còn {len(rows) - 8} dòng")
        summary = [
            f"Số dòng hợp lệ: {len(rows)}",
            f"Cột SBD: {mapping.get('student_id_column', '-')}",
            f"Cột Điểm: {mapping.get('score_column', '-')}",
            f"Cột Phòng thi: {mapping.get('room_column', '[không dùng]')}",
            "",
            "Xem trước:",
            *preview_lines,
        ]
        self.direct_score_import_summary.setPlainText("\n".join(summary))

    @staticmethod
    def _find_header_index(keys: set[str], combo: QComboBox) -> int:
        normalized_keys = {str(k or "").strip().lower().replace(" ", "").replace("_", "") for k in keys}
        for i in range(combo.count()):
            text = str(combo.itemText(i) or "").strip().lower().replace(" ", "").replace("_", "")
            if text in normalized_keys:
                return i
        return 0

    def _import_direct_score_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import điểm trực tiếp theo SBD",
            "",
            "Data files (*.xlsx *.csv *.txt *.tsv);;All files (*.*)",
        )
        if not path:
            return
        try:
            headers, raw_rows = self._load_exam_room_mapping_rows(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Import điểm trực tiếp", f"Không đọc được file:\n{exc}")
            return
        if not headers or not raw_rows:
            QMessageBox.warning(self, "Import điểm trực tiếp", "Không tìm thấy dữ liệu hợp lệ trong file điểm.")
            return
        pick = QDialog(self)
        pick.setWindowTitle("Chọn cột import điểm trực tiếp")
        pick_l = QFormLayout(pick)
        sid_col = QComboBox(); sid_col.addItems(headers)
        score_col = QComboBox(); score_col.addItems(headers)
        room_col = QComboBox(); room_col.addItem("[Không dùng]", ""); room_col.addItems(headers)
        sid_col.setCurrentIndex(self._find_header_index({"sbd", "studentid", "student_id", "sobaodanh"}, sid_col))
        score_col.setCurrentIndex(self._find_header_index({"diem", "score", "mark", "point"}, score_col))
        room_col.setCurrentIndex(self._find_header_index({"phongthi", "examroom", "exam_room", "room"}, room_col))
        pick_l.addRow("Cột SBD *", sid_col)
        pick_l.addRow("Cột Điểm *", score_col)
        pick_l.addRow("Cột Phòng thi", room_col)
        pick_note = QLabel("Nếu không chọn cột Phòng thi, hệ thống sẽ dùng phòng thi chung của kỳ thi khi import.")
        pick_note.setWordWrap(True)
        pick_l.addRow("", pick_note)
        pick_btn = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        pick_btn.accepted.connect(pick.accept)
        pick_btn.rejected.connect(pick.reject)
        pick_l.addRow(pick_btn)
        if pick.exec() != QDialog.Accepted:
            return
        sid_key = sid_col.currentText().strip()
        score_key = score_col.currentText().strip()
        room_key = str(room_col.currentData() or room_col.currentText() or "").strip()
        rows: list[dict[str, str]] = []
        skipped = 0
        for rec in raw_rows:
            sid = str(rec.get(sid_key, "") or "").strip()
            score_text = str(rec.get(score_key, "") or "").strip()
            room = str(rec.get(room_key, "") or "").strip() if room_key and room_key != "[Không dùng]" else ""
            if not sid or not score_text:
                skipped += 1
                continue
            try:
                score_value = str(float(score_text.replace(",", "."))).rstrip("0").rstrip(".")
            except Exception:
                skipped += 1
                continue
            rows.append({"sid": sid, "score": score_value, "room": room})
        if not rows:
            QMessageBox.warning(self, "Import điểm trực tiếp", "Không có dòng hợp lệ sau khi chọn cột SBD và Điểm.")
            return
        self.direct_score_import_data = {
            "path": path,
            "mapping": {
                "student_id_column": sid_key,
                "score_column": score_key,
                "room_column": room_key if room_key and room_key != "[Không dùng]" else "",
                "fallback_room_policy": "use_exam_common_room_if_missing",
            },
            "rows": rows,
            "skipped_rows": skipped,
        }
        self._refresh_direct_score_import_summary()
        QMessageBox.information(self, "Import điểm trực tiếp", f"Đã nạp {len(rows)} dòng hợp lệ cho môn tự luận.")

    def _edit_current_answer_keys(self) -> None:
        if not self.answer_key_data:
            QMessageBox.information(self, "Đáp án môn", "Môn này chưa có đáp án. Hãy import file hoặc thêm đáp án trước.")
            return
        from gui.import_answer_key_dialog import ImportAnswerKeyDialog

        dlg = ImportAnswerKeyDialog(self._build_imported_package_from_answer_data(self.answer_key_data), self)
        if dlg.exec() != QDialog.Accepted:
            return
        edited_package = dlg.result_answer_key()
        self.answer_key_data = self._answer_payload_from_package(edited_package)
        self._refresh_answer_key_summary()
        self._update_total_score()
        QMessageBox.information(self, "Đáp án môn", "Đã cập nhật đáp án hiện tại. Bạn có thể tiếp tục sửa hoặc thay đáp án khác.")

    def _import_exam_room_mapping_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import mapping SBD và phòng thi",
            "",
            "Data files (*.xlsx *.csv *.txt *.tsv);;All files (*.*)",
        )
        if not path:
            return
        try:
            headers, raw_rows = self._load_exam_room_mapping_rows(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Import mapping phòng thi", f"Không đọc được file:\n{exc}")
            return
        if not raw_rows:
            QMessageBox.warning(self, "Import mapping phòng thi", "Không tìm thấy dữ liệu hợp lệ (cần cột SBD).")
            return
        pick = QDialog(self)
        pick.setWindowTitle("Chọn cột mapping SBD/phòng")
        pick_l = QFormLayout(pick)
        sid_col = QComboBox(); sid_col.addItems(headers)
        room_col = QComboBox(); room_col.addItem("[Không dùng]", ""); room_col.addItems(headers)
        def _find_idx(keys: set[str], combo: QComboBox) -> int:
            for i in range(combo.count()):
                text = str(combo.itemText(i) or "").strip().lower().replace(" ", "").replace("_", "")
                if text in keys:
                    return i
            return 0
        sid_col.setCurrentIndex(_find_idx({"sbd", "studentid", "student_id", "sobaodanh"}, sid_col))
        room_col.setCurrentIndex(_find_idx({"phongthi", "examroom", "exam_room", "room"}, room_col))
        pick_l.addRow("Cột SBD", sid_col)
        pick_l.addRow("Cột phòng thi", room_col)
        pick_btn = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        pick_btn.accepted.connect(pick.accept)
        pick_btn.rejected.connect(pick.reject)
        pick_l.addRow(pick_btn)
        if pick.exec() != QDialog.Accepted:
            return
        sid_key = sid_col.currentText().strip()
        room_key = str(room_col.currentData() or room_col.currentText() or "").strip()
        rows: list[dict[str, str]] = []
        for rec in raw_rows:
            sid = str(rec.get(sid_key, "")).strip()
            room = str(rec.get(room_key, "")).strip() if room_key and room_key != "[Không dùng]" else ""
            if sid:
                rows.append({"sid": sid, "room": room})
        if not rows:
            QMessageBox.warning(self, "Import mapping phòng thi", "Không có dữ liệu hợp lệ sau khi chọn cột.")
            return
        grouped: dict[str, set[str]] = {}
        for r in rows:
            sid = str(r.get("sid", "")).strip()
            room = str(r.get("room", "")).strip() or "[Không rõ phòng]"
            if not sid:
                continue
            grouped.setdefault(room, set()).add(sid)
        self._exam_room_mapping_cache = {k: sorted(v) for k, v in grouped.items()}
        self._refresh_exam_room_mapping_selector()
        QMessageBox.information(self, "Import mapping phòng thi", f"Đã nạp mapping cho {len(self._exam_room_mapping_cache)} phòng thi.")

    def _refresh_exam_room_mapping_selector(self) -> None:
        self.exam_room_mapping_selector.clear()
        if not self._exam_room_mapping_cache:
            self.exam_room_mapping_selector.addItem("[Chưa có danh sách]", "")
            self.exam_room_mapping_hint.setText("Chưa nạp mapping SBD/phòng.")
            self.btn_delete_exam_room_mapping.setEnabled(False)
            self.btn_delete_multi_exam_room_mapping.setEnabled(False)
            return
        for room in sorted(self._exam_room_mapping_cache.keys()):
            cnt = len(self._exam_room_mapping_cache.get(room, []))
            self.exam_room_mapping_selector.addItem(f"{room} ({cnt} SBD)", room)
        current_room = self.exam_room_name.currentText().strip()
        if current_room:
            idx = self.exam_room_mapping_selector.findData(current_room)
            if idx >= 0:
                self.exam_room_mapping_selector.setCurrentIndex(idx)
        self.btn_delete_exam_room_mapping.setEnabled(self.exam_room_mapping_selector.count() > 0)
        self.btn_delete_multi_exam_room_mapping.setEnabled(self.exam_room_mapping_selector.count() > 1)
        self._on_exam_room_mapping_selected(self.exam_room_mapping_selector.currentIndex())

    def _on_exam_room_mapping_selected(self, _index: int) -> None:
        room = str(self.exam_room_mapping_selector.currentData() or "").strip()
        if room:
            self.exam_room_name.setCurrentText(room)
            count = len(self._exam_room_mapping_cache.get(room, []))
            self.exam_room_mapping_hint.setText(f"{room}: {count} SBD")
            self.btn_delete_exam_room_mapping.setEnabled(True)
            self.btn_delete_multi_exam_room_mapping.setEnabled(len(self._exam_room_mapping_cache) > 1)
        else:
            self.exam_room_mapping_hint.setText("Chưa nạp mapping SBD/phòng.")
            self.btn_delete_exam_room_mapping.setEnabled(False)
            self.btn_delete_multi_exam_room_mapping.setEnabled(False)

    def _remove_selected_exam_room_mapping(self) -> None:
        room = str(self.exam_room_mapping_selector.currentData() or "").strip()
        if not room:
            return
        self._exam_room_mapping_cache.pop(room, None)
        if self.exam_room_name.currentText().strip() == room:
            self.exam_room_name.setCurrentText("")
        if self._exam_room_mapping_cache and not self.exam_room_name.currentText().strip():
            self.exam_room_name.setCurrentText(sorted(self._exam_room_mapping_cache.keys())[0])
        self._refresh_exam_room_mapping_selector()

    def _remove_multiple_exam_room_mappings(self) -> None:
        rooms = sorted(self._exam_room_mapping_cache.keys())
        if len(rooms) <= 1:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Xóa nhiều phòng")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Chọn các phòng cần xóa khỏi danh sách SBD phòng thi:"))
        room_list = QListWidget()
        room_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for room in rooms:
            count = len(self._exam_room_mapping_cache.get(room, []))
            item = QListWidgetItem(f"{room} ({count} SBD)")
            item.setData(Qt.UserRole, room)
            room_list.addItem(item)
        lay.addWidget(room_list)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        if dlg.exec() != QDialog.Accepted:
            return
        selected_rooms = [str(i.data(Qt.UserRole) or "").strip() for i in room_list.selectedItems()]
        selected_rooms = [r for r in selected_rooms if r]
        if not selected_rooms:
            QMessageBox.information(self, "Xóa nhiều phòng", "Chưa chọn phòng nào để xóa.")
            return
        for room in selected_rooms:
            self._exam_room_mapping_cache.pop(room, None)
        if self.exam_room_name.currentText().strip() in selected_rooms:
            self.exam_room_name.setCurrentText("")
        if self._exam_room_mapping_cache and not self.exam_room_name.currentText().strip():
            self.exam_room_name.setCurrentText(sorted(self._exam_room_mapping_cache.keys())[0])
        self._refresh_exam_room_mapping_selector()

    @staticmethod
    def _load_exam_room_mapping_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        ext = path.suffix.lower()
        rows: list[dict[str, str]] = []
        headers: list[str] = []
        if ext in {".csv", ".txt", ".tsv"}:
            raw = path.read_text(encoding="utf-8-sig", errors="ignore")
            dialect = csv.Sniffer().sniff(raw[:2048]) if raw.strip() else csv.excel
            reader = csv.DictReader(raw.splitlines(), dialect=dialect)
            headers = [str(h or "") for h in (reader.fieldnames or [])]
            for row in reader:
                rec = {str(k): str(v or "") for k, v in (row or {}).items()}
                if rec:
                    rows.append(rec)
            return headers, rows
        if ext == ".xlsx":
            from openpyxl import load_workbook  # type: ignore
            wb = load_workbook(path, read_only=True, data_only=True)
            ws = wb.active
            values = list(ws.values)
            if not values:
                return headers, rows
            headers = [str(x or "") for x in values[0]]
            for data in values[1:]:
                rec: dict[str, str] = {}
                for idx, key in enumerate(headers):
                    rec[str(key)] = str(data[idx] if idx < len(data) and data[idx] is not None else "")
                if rec:
                    rows.append(rec)
            return headers, rows
        raise RuntimeError("Chỉ hỗ trợ .xlsx/.csv/.txt/.tsv")

    def payload(self) -> dict:
        def f(v: str, label: str) -> float:
            try:
                return float(v.strip().replace(",", "."))
            except Exception as exc:
                raise ImportError(f"Giá trị điểm '{label}' không hợp lệ: {v}") from exc

        section_scores = {
            "MCQ": {"total_points": f(self.sec_mcq_total.text(), "MCQ tổng điểm"), "distribution": "auto_by_question_count"},
            "TF": {
                "total_points": f(self.sec_tf_total.text(), "TF tổng điểm"),
                "rule_per_question": {
                    "1": f(self.sec_tf_1.text(), "TF đúng 1 ý"),
                    "2": f(self.sec_tf_2.text(), "TF đúng 2 ý"),
                    "3": f(self.sec_tf_3.text(), "TF đúng 3 ý"),
                    "4": f(self.sec_tf_4.text(), "TF đúng 4 ý"),
                },
            },
            "NUMERIC": {"total_points": f(self.sec_numeric_total.text(), "NUMERIC tổng điểm"), "distribution": "auto_by_question_count"},
        }
        question_scores = {
            "MCQ": {"per_question": f(self.q_mcq.text(), "MCQ điểm/câu")},
            "TF": {
                "1": f(self.q_tf_1.text(), "TF đúng 1 ý"),
                "2": f(self.q_tf_2.text(), "TF đúng 2 ý"),
                "3": f(self.q_tf_3.text(), "TF đúng 3 ý"),
                "4": f(self.q_tf_4.text(), "TF đúng 4 ý"),
            },
            "NUMERIC": {"per_question": f(self.q_numeric.text(), "NUMERIC điểm/câu")},
        }

        return {
            "name": self.subject_name.currentText().strip(),
            "block": self.block_name.currentText().strip(),
            "is_essay_subject": bool(self.is_essay_subject.isChecked()),
            "template_path": self.template_path.text().strip(),
            "scan_folder": self.scan_folder.text().strip(),
            "auto_recognize": bool(self.auto_recognize.isChecked()),
            "answer_key_path": self.answer_key.text().strip(),
            "answer_key_key": self.answer_key_key.text().strip(),
            "exam_room_name": str(self.exam_room_mapping_selector.currentData() or self.exam_room_name.currentText() or "").strip(),
            "exam_room_sbd_mapping": ",".join(self._exam_room_mapping_cache.get(str(self.exam_room_mapping_selector.currentData() or self.exam_room_name.currentText() or "").strip(), [])),
            "exam_room_sbd_mapping_by_room": {str(k): list(v) for k, v in (self._exam_room_mapping_cache or {}).items()},
            "imported_answer_keys": self.answer_key_data,
            "direct_score_import": copy.deepcopy(self.direct_score_import_data),
            "score_mode": self.score_mode.currentText(),
            "section_scores": section_scores,
            "question_scores": question_scores,
            "question_counts": self._question_counts(),
            "total_exam_points": self._to_float(self.total_score.text()),
            "paper_part_count": int(self.paper_part_label.text() or self.paper_part_count_default),
        }

class NewExamDialog(QDialog):
    def __init__(
        self,
        subject_options: list[str],
        block_options: list[str],
        data: dict | None = None,
        parent=None,
        on_batch_scan_subject=None,
        on_save_exam=None,
        stay_open_on_save: bool = False,
        template_repo: TemplateRepository | None = None,
    ):
        super().__init__(parent)
        data = data or {}
        self.setWindowTitle("Sửa kỳ thi" if data else "Tạo kỳ thi mới")
        self.resize(860, 640)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        try:
            _ico = app_icon()
            if _ico is not None and not _ico.isNull():
                self.setWindowIcon(_ico)
        except Exception:
            pass
        self.subject_configs: list[dict] = list(data.get("subject_configs", []))
        self.student_list_path_value = str(data.get("student_list_path", "") or "")
        self.student_rows: list[dict] = list(data.get("students", [])) if isinstance(data.get("students", []), list) else []
        self.on_batch_scan_subject = on_batch_scan_subject
        self.on_save_exam = on_save_exam
        self.stay_open_on_save = bool(stay_open_on_save)
        self.subject_options = subject_options
        self.block_options = block_options
        self.template_repo = template_repo or TemplateRepository()
        self.database = getattr(parent, "database", None)

        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.exam_name = QLineEdit(str(data.get("exam_name", "")))
        self.common_template = QLineEdit(str(data.get("common_template", "")))
        self.scan_root = QLineEdit(str(data.get("scan_root", "")))
        self.student_list_path = QLineEdit(self.student_list_path_value)
        self.student_list_path.setReadOnly(True)
        self.student_count_label = QLabel(f"{len(self.student_rows)} học sinh")
        self.scan_mode = QComboBox(); self.scan_mode.addItems(["Ảnh trong thư mục gốc", "Ảnh theo phòng thi (thư mục con)"])
        self.scan_mode.setCurrentText(str(data.get("scan_mode", "Ảnh trong thư mục gốc")))
        self.paper_part_count = QComboBox(); self.paper_part_count.addItems(["1", "2", "3", "4", "5"]); self.paper_part_count.setCurrentText(str(data.get("paper_part_count", "3")))

        row_tpl = QHBoxLayout(); row_tpl.addWidget(self.common_template); btn_tpl = QPushButton("..."); row_tpl.addWidget(btn_tpl); btn_tpl_repo = QPushButton("Kho mẫu..."); row_tpl.addWidget(btn_tpl_repo)
        btn_tpl.clicked.connect(self._browse_common_template)
        btn_tpl_repo.clicked.connect(self._pick_common_template_from_repo)
        row_scan = QHBoxLayout(); row_scan.addWidget(self.scan_root); btn_scan = QPushButton("..."); row_scan.addWidget(btn_scan)
        btn_scan.clicked.connect(self._browse_scan_root)
        row_students = QHBoxLayout(); row_students.addWidget(self.student_list_path)
        btn_students = QPushButton("Import Excel...")
        self.btn_view_students = QPushButton("Xem danh sách")
        row_students.addWidget(btn_students)
        row_students.addWidget(self.btn_view_students)
        row_students.addWidget(self.student_count_label)
        btn_students.clicked.connect(self._import_student_list)
        self.btn_view_students.clicked.connect(self._open_student_list_preview)
        self.btn_view_students.setEnabled(bool(self.student_rows))

        form.addRow("Tên kỳ thi", self.exam_name)
        form.addRow("Giấy thi dùng chung", row_tpl)
        form.addRow("Thư mục gốc bài thi", row_scan)
        form.addRow("Danh sách học sinh", row_students)
        form.addRow("Cơ chế thư mục bài thi", self.scan_mode)
        form.addRow("Số phần trên giấy thi", self.paper_part_count)
        lay.addLayout(form)

        lay.addWidget(QLabel("Các môn trong kỳ thi"))
        self.subject_table = QTableWidget(0, 11)
        self.subject_table.setHorizontalHeaderLabels(["STT", "Môn", "Khối", "Key", "Mã đề", "Chế độ điểm", "Tổng điểm", "Template", "Cơ chế", "Trạng thái", "Thao tác"])
        self.subject_table.verticalHeader().setVisible(False)
        self.subject_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subject_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.subject_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.subject_table.setShowGrid(True)
        self.subject_table.setGridStyle(Qt.SolidLine)
        self.subject_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.subject_table.customContextMenuRequested.connect(self._open_subject_row_context_menu)
        # double click navigation: open subject editor directly from row.
        self.subject_table.cellDoubleClicked.connect(self._handle_subject_table_double_click)
        hdr = self.subject_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.Stretch)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(9, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(10, QHeaderView.ResizeToContents)
        lay.addWidget(self.subject_table)

        self._refresh_subject_list()

    def _answer_key_scope_key(self, subject_key: str, block: str = "") -> str:
        base = str(subject_key or "").strip()
        if not base:
            return ""
        if "::" in base:
            return base
        session_id = str(getattr(self.parent(), "current_session_id", "") or "").strip()
        exam_name = str(self.exam_name.text() if hasattr(self, "exam_name") else "").strip().lower()
        block_text = str(block or "").strip()
        if not block_text and "_" in base:
            block_text = str(base.rsplit("_", 1)[-1]).strip()
        scope_prefix = f"{session_id}::{exam_name}" if session_id and exam_name else session_id
        if scope_prefix and block_text:
            return f"{scope_prefix}::{base}::{block_text}"
        if scope_prefix:
            return f"{scope_prefix}::{base}"
        return base

    @staticmethod
    def _normalized_student_id_for_match(student_id: str) -> str:
        sid = str(student_id or "").strip()
        if not sid:
            return ""
        compact = sid.replace(" ", "")
        if compact.endswith(".0"):
            prefix = compact[:-2]
            if prefix.isdigit():
                compact = prefix
        if compact.isdigit():
            compact = compact.lstrip("0") or "0"
        return compact.upper()

    @staticmethod
    def _normalized_room_for_match(room_text: str) -> str:
        room = str(room_text or "").strip().casefold()
        if room.isdigit():
            room = room.lstrip("0") or "0"
        return room

    def _browse_common_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn giấy thi dùng chung", "", "JSON (*.json)")
        if path:
            self.template_repo.register(path)
            self.common_template.setText(path)


    def _pick_common_template_from_repo(self) -> None:
        items = [f"{name} | {path}" for name, path in self.template_repo.list_templates()]
        if not items:
            QMessageBox.information(self, "Kho mẫu giấy thi", "Kho mẫu đang trống. Hãy thêm mẫu bằng nút ...")
            return
        chosen, ok = QInputDialog.getItem(self, "Kho mẫu giấy thi", "Chọn mẫu dùng chung:", items, 0, False)
        if ok and chosen:
            self.common_template.setText(chosen.split(" | ", 1)[1])

    def _browse_scan_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục gốc bài thi")
        if path:
            self.scan_root.setText(path)

    def _import_student_list(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import danh sách học sinh", "", "Excel/CSV (*.xlsx *.xls *.csv)")
        if not path:
            return
        try:
            import pandas as pd
            if Path(path).suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path, dtype=str)
            else:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as exc:
            QMessageBox.warning(self, "Danh sách học sinh", f"Không đọc được file học sinh:\n{exc}")
            return
        if df.empty:
            QMessageBox.warning(self, "Danh sách học sinh", "File học sinh rỗng.")
            return

        columns = [str(c) for c in df.columns]
        dlg = QDialog(self)
        dlg.setWindowTitle("Mapping cột danh sách học sinh")
        lay = QVBoxLayout(dlg)
        frm = QFormLayout()
        c_sid = QComboBox(); c_sid.addItems(columns)
        c_name = QComboBox(); c_name.addItems(columns)
        c_birth = QComboBox(); c_birth.addItems(["[Không dùng]"] + columns)
        c_class = QComboBox(); c_class.addItems(["[Không dùng]"] + columns)
        c_room = QComboBox(); c_room.addItems(["[Không dùng]"] + columns)
        # best-effort default picks
        lower_cols = {x.lower(): x for x in columns}
        for key, cb in [
            ("studentid", c_sid), ("sobaodanh", c_sid), ("student_id", c_sid),
            ("name", c_name), ("hoten", c_name), ("họ tên", c_name),
        ]:
            if key in lower_cols:
                cb.setCurrentText(lower_cols[key])
        frm.addRow("Số báo danh (bắt buộc)", c_sid)
        frm.addRow("Họ tên (bắt buộc)", c_name)
        frm.addRow("Ngày sinh", c_birth)
        frm.addRow("Lớp", c_class)
        frm.addRow("Phòng thi", c_room)
        lay.addLayout(frm)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return

        sid_col = c_sid.currentText().strip()
        name_col = c_name.currentText().strip()
        if not sid_col or not name_col:
            QMessageBox.warning(self, "Danh sách học sinh", "Bắt buộc map cột Số báo danh và Họ tên.")
            return

        def _col_value(row_obj, col_name: str) -> str:
            if not col_name or col_name == "[Không dùng]":
                return ""
            v = row_obj.get(col_name, "")
            return "" if v is None else str(v).strip()

        out: list[dict] = []
        for _, row_obj in df.iterrows():
            sid = _col_value(row_obj, sid_col)
            name = _col_value(row_obj, name_col)
            if not sid or not name:
                continue
            out.append(
                {
                    "student_id": sid,
                    "name": name,
                    "birth_date": _col_value(row_obj, c_birth.currentText()),
                    "class_name": _col_value(row_obj, c_class.currentText()),
                    "exam_room": _col_value(row_obj, c_room.currentText()),
                }
            )

        if not out:
            QMessageBox.warning(self, "Danh sách học sinh", "Không có dòng hợp lệ (thiếu Số báo danh/Họ tên).")
            return

        action_text = "thay thế"
        if self.student_rows:
            msg = QMessageBox(self)
            msg.setWindowTitle("Danh sách học sinh")
            msg.setText(
                f"Đã import được {len(out)} học sinh từ file mới.\n"
                "Bạn muốn thêm vào danh sách hiện tại hay thay thế toàn bộ?"
            )
            btn_append = msg.addButton("Thêm vào", QMessageBox.AcceptRole)
            btn_replace = msg.addButton("Thay thế", QMessageBox.DestructiveRole)
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_append:
                action_text = "thêm vào"
                merged: dict[str, dict] = {}
                for row in self.student_rows:
                    sid_key = self._normalized_student_id_for_match(str(row.get("student_id", "") or ""))
                    merged[sid_key or str(row.get("student_id", "") or "")] = row
                for row in out:
                    sid_key = self._normalized_student_id_for_match(str(row.get("student_id", "") or ""))
                    merged[sid_key or str(row.get("student_id", "") or "")] = row
                self.student_rows = list(merged.values())
            elif clicked == btn_replace:
                self.student_rows = out
            else:
                return
        else:
            self.student_rows = out
        duplicate_sids = self._duplicate_student_ids(self.student_rows)
        self.student_list_path_value = path
        self.student_list_path.setText(path)
        self.student_count_label.setText(f"{len(self.student_rows)} học sinh")
        self.btn_view_students.setEnabled(bool(self.student_rows))
        QMessageBox.information(
            self,
            "Danh sách học sinh",
            f"Đã {action_text} danh sách học sinh. Tổng hiện tại: {len(self.student_rows)} học sinh.",
        )
        if duplicate_sids:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Danh sách học sinh")
            msg.setText(
                f"Phát hiện {len(duplicate_sids)} SBD bị trùng trong danh sách vừa import.\n"
                "Bạn có muốn mở màn hình \"Xem danh sách\" để kiểm tra và xoá bản ghi không?"
            )
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.Yes)
            if msg.exec() == QMessageBox.Yes:
                self._open_student_list_preview()

    @staticmethod
    def _duplicate_student_ids(rows: list[dict]) -> set[str]:
        counts: dict[str, int] = {}
        for row in rows or []:
            sid = str((row or {}).get("student_id", "") or "").strip()
            if sid:
                counts[sid] = counts.get(sid, 0) + 1
        return {sid for sid, count in counts.items() if count > 1}

    def _open_student_list_preview(self) -> None:
        if not self.student_rows:
            QMessageBox.information(self, "Danh sách học sinh", "Chưa có dữ liệu học sinh để xem.")
            return
        dlg = StudentListPreviewDialog(self.student_rows, self)
        if dlg.exec() == QDialog.Accepted:
            self.student_rows = dlg.rows()
            self.student_count_label.setText(f"{len(self.student_rows)} học sinh")
            self.btn_view_students.setEnabled(bool(self.student_rows))


    def _refresh_subject_list(self) -> None:
        """Compatibility wrapper. Every subject-grid refresh must go through _reload_subject_grid."""
        self._reload_subject_grid(reason="legacy_refresh_call")

    def _reload_subject_grid(self, *, reason: str = "") -> None:
        """Single source of truth for the subject table in the exam editor.

        This method owns all rendering of the columns Template and Trạng thái.
        Add/edit/delete/save flows must call this method instead of writing rows directly.
        """
        rows = self._build_subject_grid_rows()
        table = self.subject_table
        table.setUpdatesEnabled(False)
        table.blockSignals(True)
        try:
            table.clearContents()
            table.setRowCount(len(rows))
            style = self.style()
            for row_idx, row in enumerate(rows):
                self._render_subject_grid_row(row_idx, row, style)
            table.resizeRowsToContents()
        finally:
            table.blockSignals(False)
            table.setUpdatesEnabled(True)

    def _build_subject_grid_rows(self) -> list[dict]:
        status_map = self._subject_status_snapshot()
        rows: list[dict] = []
        for row_idx, cfg in enumerate(self.subject_configs):
            cfg = dict(cfg or {})
            imported_keys = cfg.get("imported_answer_keys", {}) or {}
            codes = ",".join(sorted(imported_keys.keys())) if isinstance(imported_keys, dict) else ""
            rows.append(
                {
                    "index": row_idx,
                    "stt": str(row_idx + 1),
                    "name": str(cfg.get("name", "") or "-"),
                    "block": str(cfg.get("block", "") or "-"),
                    "key": str(cfg.get("answer_key_key", "") or "-"),
                    "codes": codes or "-",
                    "score_mode": str(cfg.get("score_mode", "Điểm theo phần") or "-"),
                    "total": str(cfg.get("total_exam_points", "-") or "-"),
                    "template": self._subject_template_display_text(cfg),
                    "template_tooltip": self._subject_template_tooltip(cfg),
                    "auto_mode": "Nhận dạng tự động" if bool(cfg.get("auto_recognize", False)) else "Thủ công",
                    "status": str(status_map.get(row_idx, "-") or "-"),
                }
            )
        return rows

    def _render_subject_grid_row(self, row_idx: int, row: dict, style) -> None:
        values = [
            row.get("stt", ""),
            row.get("name", "-"),
            row.get("block", "-"),
            row.get("key", "-"),
            row.get("codes", "-"),
            row.get("score_mode", "-"),
            row.get("total", "-"),
            row.get("template", "-"),
            row.get("auto_mode", "-"),
            row.get("status", "-"),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if col == 7:
                item.setToolTip(str(row.get("template_tooltip", "") or str(value)))
            if col == 9:
                item.setToolTip("Trạng thái được tính trực tiếp từ DB/cache nhận dạng của môn hiện tại.")
            self.subject_table.setItem(row_idx, col, item)

        btn_batch_scan = QPushButton("Nhận dạng")
        btn_batch_scan.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        btn_batch_scan.setToolTip("Batch Scan theo môn")
        btn_batch_scan.setEnabled(callable(self.on_batch_scan_subject))
        btn_batch_scan.clicked.connect(lambda _=False, i=row_idx: self._trigger_subject_batch_scan(i))
        wrap = QWidget()
        wrap_l = QHBoxLayout(wrap)
        wrap_l.setContentsMargins(0, 0, 0, 0)
        wrap_l.addWidget(btn_batch_scan)
        self.subject_table.setCellWidget(row_idx, 10, wrap)

    def _subject_template_display_text(self, cfg: dict) -> str:
        subject_template = self._normalize_template_path(str((cfg or {}).get("template_path", "") or ""))
        if subject_template:
            return Path(subject_template).name or subject_template
        common_template = self._normalize_template_path(self.common_template.text().strip() if hasattr(self, "common_template") else "")
        if common_template:
            return f"Mẫu chung: {Path(common_template).name or common_template}"
        return "[chưa chọn mẫu]"

    def _subject_template_tooltip(self, cfg: dict) -> str:
        subject_template = self._normalize_template_path(str((cfg or {}).get("template_path", "") or ""))
        if subject_template:
            return subject_template
        common_template = self._normalize_template_path(self.common_template.text().strip() if hasattr(self, "common_template") else "")
        if common_template:
            return f"Dùng mẫu giấy thi chung:\n{common_template}"
        return "Chưa chọn mẫu riêng hoặc mẫu chung."

    def _subject_status_snapshot(self) -> dict[int, str]:
        parent = self.parent()
        session_id = (
            getattr(parent, "embedded_exam_session_id", None)
            or getattr(parent, "current_session_id", None)
        ) if parent is not None else None
        session_obj = (
            getattr(parent, "embedded_exam_session", None)
            or getattr(parent, "session", None)
        ) if parent is not None else None
        if parent is not None and hasattr(parent, "_subject_display_status_map"):
            try:
                return dict(parent._subject_display_status_map(self.subject_configs, session_id=session_id, session=session_obj) or {})
            except Exception:
                pass
        return {idx: self._subject_status_text(cfg, idx) for idx, cfg in enumerate(self.subject_configs)}

    def _subject_status_text(self, cfg: dict, row_index: int | None = None) -> str:
        parent = self.parent()
        if parent is not None and hasattr(parent, "_subject_display_status_text"):
            try:
                if isinstance(cfg, dict) and hasattr(parent, "_ensure_subject_instance_key"):
                    try:
                        parent._ensure_subject_instance_key(cfg, row_index)
                    except Exception:
                        pass
                session_id = (
                    getattr(parent, "embedded_exam_session_id", None)
                    or getattr(parent, "current_session_id", None)
                )
                session_obj = (
                    getattr(parent, "embedded_exam_session", None)
                    or getattr(parent, "session", None)
                )
                return str(parent._subject_display_status_text(cfg, session_id=session_id, session=session_obj) or "-")
            except TypeError:
                try:
                    return str(parent._subject_display_status_text(cfg) or "-")
                except Exception:
                    return "-"
            except Exception:
                return "-"
        return "-"

    def _current_subject_index(self) -> int:
        idx = self.subject_table.currentRow()
        return idx if 0 <= idx < len(self.subject_configs) else -1

    def _handle_subject_table_double_click(self, row: int, col: int) -> None:
        if row < 0 or row >= len(self.subject_configs):
            return
        self.subject_table.selectRow(row)
        self._edit_subject()

    def _open_subject_row_context_menu(self, pos) -> None:
        row = self.subject_table.rowAt(pos.y())
        if row < 0 or row >= len(self.subject_configs):
            return
        self.subject_table.selectRow(row)
        menu = QMenu(self)
        act_scan = menu.addAction("Nhận dạng")
        act_edit = menu.addAction("Sửa cấu hình môn")
        act_delete = menu.addAction("Xoá")
        chosen = menu.exec(self.subject_table.viewport().mapToGlobal(pos))
        if chosen == act_scan:
            self._trigger_subject_batch_scan(row)
            return
        if chosen == act_edit:
            self._edit_subject()
            return
        if chosen == act_delete:
            self._delete_subject()

    def _trigger_subject_batch_scan(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.subject_configs):
            return
        if not callable(self.on_batch_scan_subject):
            QMessageBox.information(self, "Batch Scan", "Vui lòng lưu kỳ thi trước khi chạy Batch Scan theo từng môn.")
            return
        cfg = dict(self.subject_configs[idx])
        cfg["template_path"] = self._normalize_template_path(str(cfg.get("template_path", "")))
        cfg["scan_folder"] = str(cfg.get("scan_folder", "") or self.scan_root.text().strip())
        proceed = self.on_batch_scan_subject(
            {
                "exam_name": self.exam_name.text().strip(),
                "common_template": self.common_template.text().strip(),
                "scan_root": self.scan_root.text().strip(),
                "scan_mode": self.scan_mode.currentText(),
                "paper_part_count": int(self.paper_part_count.currentText()),
                "subject_configs": list(self.subject_configs),
                "selected_subject_index": idx,
                "subject_config": cfg,
            }
        )
        if proceed is False:
            return
        if proceed is True:
            return
        self.reject()

    @staticmethod
    def _normalize_template_path(path_text: str) -> str:
        t = str(path_text or "").strip()
        if not t:
            return ""
        if t.lower() in {"[dùng mẫu chung]", "[dung mau chung]", "none", "null", "-"}:
            return ""
        return t

    @staticmethod
    def _subject_imported_answer_keys(subject_cfg: dict) -> dict:
        if not isinstance(subject_cfg, dict):
            return {}
        raw = subject_cfg.get("imported_answer_keys", {})
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _subject_identity_changed(old_cfg: dict, new_cfg: dict) -> bool:
        def _norm(v: object) -> str:
            return str(v or "").strip().lower()
        watched = ["template_path", "scan_folder", "name", "block", "answer_key_key"]
        return any(_norm(old_cfg.get(k)) != _norm(new_cfg.get(k)) for k in watched)

    @staticmethod
    def _copy_noncritical_subject_updates(old_cfg: dict, new_cfg: dict) -> dict:
        merged = dict(old_cfg)
        merged.update(new_cfg)
        # Preserve batch/scoring artifacts when identity is unchanged.
        for k in [
            "batch_saved",
            "batch_saved_at",
            "batch_result_count",
            "batch_saved_rows",
            "batch_saved_preview",
            "batch_saved_results",
        ]:
            if k in old_cfg:
                merged[k] = old_cfg.get(k)
        return merged

    def _confirm_subject_identity_change(self, old_cfg: dict, new_cfg: dict) -> str:
        parent = self.parent()
        has_existing_batch = False
        if parent is not None and hasattr(parent, "_subject_has_recognition_data"):
            try:
                has_existing_batch = bool(parent._subject_has_recognition_data(old_cfg))
            except Exception:
                has_existing_batch = False
        else:
            has_existing_batch = bool(old_cfg.get("batch_saved")) or bool(old_cfg.get("batch_result_count"))
        if not has_existing_batch:
            return "apply"

        msg = QMessageBox(self)
        msg.setWindowTitle("Môn đã có dữ liệu nhận dạng")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(
            "Bạn đã thay đổi thông tin ảnh hưởng đến nhận dạng/chấm điểm (mẫu giấy, thư mục quét, tên môn, khối hoặc mã đáp án môn_khối)."
        )
        msg.setInformativeText(
            "Giữ dữ liệu Batch cũ có thể làm sai kết quả.\n"
            "Yes: Xóa trạng thái/dữ liệu nhận dạng của môn này và áp dụng thay đổi.\n"
            "No: Vẫn giữ dữ liệu nhận dạng cũ và áp dụng thay đổi.\n"
            "Cancel: Hủy chỉnh sửa."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        ch = msg.exec()
        if ch == QMessageBox.Cancel:
            return "cancel"
        if ch == QMessageBox.Yes:
            return "reset"
        return "apply"

    def _call_save_exam(self, *, show_message: bool = True, refresh_subject_grid: bool = True) -> bool:
        if not callable(self.on_save_exam):
            return True
        try:
            ok = self.on_save_exam(show_message=show_message, refresh_subject_grid=refresh_subject_grid)
        except TypeError:
            ok = self.on_save_exam()
        return ok is not False

    def _persist_subject_grid_after_change(self, *, reason: str) -> bool:
        """Persist the current exam payload, then reload the subject grid through one renderer."""
        ok = self._call_save_exam(show_message=False, refresh_subject_grid=False)
        if not ok:
            QMessageBox.warning(self, "Môn thi", "Không thể lưu thay đổi cấu hình môn vào CSDL.")
            return False
        parent = self.parent()
        if parent is not None:
            session_obj = getattr(parent, "embedded_exam_session", None) or getattr(parent, "session", None)
            cfg = getattr(session_obj, "config", {}) if session_obj is not None else {}
            if isinstance(cfg, dict):
                saved_cfgs = cfg.get("subject_configs", [])
                if isinstance(saved_cfgs, list) and saved_cfgs:
                    self.subject_configs = list(saved_cfgs)
        self._reload_subject_grid(reason=reason)
        return True

    def _db(self):
        if self.database is not None:
            return self.database
        parent = self.parent()
        return getattr(parent, "database", None) if parent is not None else None

    def _persist_subject_answer_keys(self, old_cfg: dict | None, new_cfg: dict | None, *, action: str) -> None:
        db = self._db()
        if db is None or not isinstance(new_cfg, dict):
            return
        try:
            old_cfg = old_cfg if isinstance(old_cfg, dict) else {}
            old_subject_key = str(old_cfg.get("answer_key_key", "") or "")
            new_subject_key = str(new_cfg.get("answer_key_key", "") or "")
            old_scoped_key = self._answer_key_scope_key(old_subject_key, str(old_cfg.get("block", "") or "")) if old_subject_key else ""
            new_scoped_key = self._answer_key_scope_key(new_subject_key, str(new_cfg.get("block", "") or "")) if new_subject_key else ""
            if old_scoped_key and old_scoped_key != new_scoped_key:
                db.replace_answer_keys_for_subject(old_scoped_key, {})
            if new_scoped_key:
                new_keys = new_cfg.get("imported_answer_keys", {}) or {}
                db.replace_answer_keys_for_subject(new_scoped_key, new_keys)
                if old_cfg.get("imported_answer_keys", {}) != new_keys:
                    db.log_change(
                        "answer_keys",
                        new_scoped_key,
                        "imported_answer_keys",
                        old_cfg.get("imported_answer_keys", {}) or {},
                        new_keys,
                        action,
                    )
        except Exception:
            pass

    def _delete_subject_answer_keys(self, cfg: dict | None) -> None:
        db = self._db()
        if db is None or not isinstance(cfg, dict):
            return
        try:
            subject_key = str(cfg.get("answer_key_key", "") or "")
            scoped_key = self._answer_key_scope_key(subject_key, str(cfg.get("block", "") or "")) if subject_key else ""
            if scoped_key:
                db.replace_answer_keys_for_subject(scoped_key, {})
                db.log_change("answer_keys", scoped_key, "imported_answer_keys", cfg.get("imported_answer_keys", {}) or {}, {}, "delete_subject")
        except Exception:
            pass

    def _add_subject(self) -> None:
        dlg = SubjectConfigDialog(
            subject_options=self.subject_options,
            block_options=self.block_options,
            paper_part_count=int(self.paper_part_count.currentText()),
            common_template_path=self.common_template.text().strip(),
            template_repo=self.template_repo,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        payload = dlg.payload()
        before = copy.deepcopy(self.subject_configs)
        self.subject_configs.append(payload)
        if not self._persist_subject_grid_after_change(reason="add_subject"):
            self.subject_configs = before
            self._reload_subject_grid(reason="add_subject_rollback")
            return
        self._persist_subject_answer_keys(None, payload, action="add_subject")

    def _edit_subject(self) -> None:
        idx = self._current_subject_index()
        if idx < 0:
            return
        dlg = SubjectConfigDialog(
            self.subject_configs[idx],
            subject_options=self.subject_options,
            block_options=self.block_options,
            paper_part_count=int(self.paper_part_count.currentText()),
            common_template_path=self.common_template.text().strip(),
            template_repo=self.template_repo,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        before = copy.deepcopy(self.subject_configs)
        old_cfg = dict(self.subject_configs[idx])
        edited = dlg.payload()

        if not self._subject_identity_changed(old_cfg, edited):
            self.subject_configs[idx] = self._copy_noncritical_subject_updates(old_cfg, edited)
            if not self._persist_subject_grid_after_change(reason="edit_subject_noncritical"):
                self.subject_configs = before
                self._reload_subject_grid(reason="edit_subject_rollback")
                return
            self._persist_subject_answer_keys(old_cfg, self.subject_configs[idx], action="edit_subject")
            return

        decision = self._confirm_subject_identity_change(old_cfg, edited)
        if decision == "cancel":
            return
        reset_recognition_data = decision == "reset"
        if decision == "reset":
            updated = dict(edited)
            # Keep subject runtime identity stable across config edits so status lookup,
            # DB storage key resolution and auto-recognition all read the same stream.
            for stable_key in ("subject_instance_key", "subject_uid", "logical_subject_key", "legacy_subject_instance_keys"):
                if stable_key in old_cfg and stable_key not in updated:
                    updated[stable_key] = copy.deepcopy(old_cfg.get(stable_key))
            updated["batch_saved"] = False
            updated["batch_saved_at"] = "-"
            updated["batch_result_count"] = 0
            updated["batch_saved_rows"] = []
            updated["batch_saved_preview"] = []
            updated["batch_saved_results"] = []
            self.subject_configs[idx] = updated
        else:
            self.subject_configs[idx] = dict(old_cfg) | dict(edited)

        if not self._persist_subject_grid_after_change(reason="edit_subject"):
            self.subject_configs = before
            self._reload_subject_grid(reason="edit_subject_rollback")
            return
        if reset_recognition_data:
            parent = self.parent()
            if parent is not None and hasattr(parent, "_delete_subject_recognition_data"):
                try:
                    parent._delete_subject_recognition_data(old_cfg)
                except Exception:
                    pass
            self._reload_subject_grid(reason="edit_subject_reset_recognition")
        self._persist_subject_answer_keys(old_cfg, self.subject_configs[idx], action="edit_subject")

    def _delete_subject(self) -> None:
        idx = self._current_subject_index()
        if idx < 0:
            return
        old_cfg = dict(self.subject_configs[idx] or {})
        name = str(old_cfg.get("name", "") or "").strip() or "môn đã chọn"
        if QMessageBox.question(
            self,
            "Xoá môn",
            f"Bạn có chắc muốn xoá {name} khỏi kỳ thi?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        before = copy.deepcopy(self.subject_configs)
        del self.subject_configs[idx]
        if not self._persist_subject_grid_after_change(reason="delete_subject"):
            self.subject_configs = before
            self._reload_subject_grid(reason="delete_subject_rollback")
            return
        self._delete_subject_answer_keys(old_cfg)

    def _validate_and_accept(self) -> None:
        if not self.exam_name.text().strip():
            QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng nhập tên kỳ thi.")
            return
        if not self.subject_configs:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng thêm ít nhất 1 môn học.")
            return
        for cfg in self.subject_configs:
            if not cfg.get("name"):
                QMessageBox.warning(self, "Thiếu dữ liệu", "Mỗi môn phải có tên.")
                return
            if not cfg.get("block"):
                QMessageBox.warning(self, "Thiếu dữ liệu", "Mỗi môn phải có khối.")
                return
        if self.stay_open_on_save:
            ok = self._call_save_exam(show_message=True, refresh_subject_grid=True)
            if not ok:
                return
            return
        self.accept()

    def payload(self) -> dict:
        return {
            "exam_name": self.exam_name.text().strip(),
            "common_template": self.common_template.text().strip(),
            "scan_root": self.scan_root.text().strip(),
            "student_list_path": self.student_list_path_value,
            "students": self.student_rows,
            "scan_mode": self.scan_mode.currentText(),
            "paper_part_count": int(self.paper_part_count.currentText()),
            "subject_configs": self.subject_configs,
        }

class StudentListPreviewDialog(QDialog):
    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xem danh sách học sinh")
        self.resize(980, 640)
        self._rows: list[dict] = [dict(r or {}) for r in (rows or [])]

        lay = QVBoxLayout(self)
        note = QLabel("Các dòng trùng SBD được tô đỏ. Có thể chọn nhiều dòng và bấm Xoá.")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["STT", "SBD", "Họ tên", "Ngày sinh", "Lớp", "Phòng thi"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        lay.addWidget(self.table)

        action_row = QHBoxLayout()
        self.delete_btn = QPushButton("Xoá bản ghi đã chọn")
        self.delete_btn.clicked.connect(self._delete_selected_rows)
        action_row.addWidget(self.delete_btn)
        action_row.addStretch(1)
        lay.addLayout(action_row)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Save).setText("Lưu danh sách")
        bb.button(QDialogButtonBox.Cancel).setText("Đóng")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self._reload_table()

    def rows(self) -> list[dict]:
        return [dict(r or {}) for r in self._rows]

    def _reload_table(self) -> None:
        duplicate_sids = NewExamDialog._duplicate_student_ids(self._rows)
        self.table.setRowCount(len(self._rows))
        for idx, row in enumerate(self._rows):
            sid = str((row or {}).get("student_id", "") or "").strip()
            values = [
                str(idx + 1),
                sid,
                str((row or {}).get("name", "") or ""),
                str((row or {}).get("birth_date", "") or ""),
                str((row or {}).get("class_name", "") or ""),
                str((row or {}).get("exam_room", "") or ""),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if sid and sid in duplicate_sids:
                    item.setBackground(QColor(255, 220, 220))
                    item.setForeground(QColor(180, 0, 0))
                self.table.setItem(idx, col, item)
        self.table.resizeRowsToContents()

    def _delete_selected_rows(self) -> None:
        selected_indexes = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not selected_indexes:
            QMessageBox.information(self, "Xem danh sách học sinh", "Vui lòng chọn bản ghi cần xoá.")
            return
        target_rows = sorted({int(index.row()) for index in selected_indexes}, reverse=True)
        for row_idx in target_rows:
            if 0 <= row_idx < len(self._rows):
                self._rows.pop(row_idx)
        self._reload_table()