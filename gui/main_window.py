from __future__ import annotations

import copy
import json
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, QPointF, QTimer
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPixmap, QTransform, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QCompleter,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
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
from core.omr_engine import OMRProcessor, OMRResult
from core.scoring_engine import ScoringEngine
from editor.template_editor import TemplateEditorWindow
from gui.import_answer_key_dialog import ImportAnswerKeyDialog
from models.answer_key import AnswerKeyRepository, SubjectKey
from models.database import OMRDatabase, bootstrap_application_db
from models.exam_session import ExamSession, Student
from models.template import Template
from models.template_repository import TemplateRepository


class PreviewImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(260)
        self._markers: list[dict[str, float]] = []
        self._overlay_markers: list[dict[str, float]] = []
        self._drag_index: int = -1

    def set_markers(self, markers: list[dict[str, float]]) -> None:
        self._markers = [dict(m) for m in markers]
        self.update()

    def clear_markers(self) -> None:
        self._markers = []
        self._overlay_markers = []
        self.update()

    def set_overlay_markers(self, markers: list[dict[str, float]]) -> None:
        self._overlay_markers = [dict(m) for m in markers]
        self.update()

    def markers(self) -> list[dict[str, float]]:
        return [dict(m) for m in self._markers]

    def has_markers(self) -> bool:
        return bool(self._markers)

    def paintEvent(self, event):  # type: ignore[override]
        super().paintEvent(event)
        if not self._markers and not self._overlay_markers:
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

        if self._markers:
            painter.setPen(QPen(Qt.green, 2))
            for m in self._markers:
                x = float(m.get("x", 0.0))
                y = float(m.get("y", 0.0))
                r = 6
                painter.drawLine(int(x - r), int(y - r), int(x + r), int(y + r))
                painter.drawLine(int(x - r), int(y + r), int(x + r), int(y - r))

    def _pick_marker_index(self, pos: QPointF) -> int:
        px, py = float(pos.x()), float(pos.y())
        best = -1
        best_d2 = 1e9
        for i, m in enumerate(self._markers):
            dx = px - float(m.get("x", 0.0))
            dy = py - float(m.get("y", 0.0))
            d2 = dx * dx + dy * dy
            if d2 < best_d2 and d2 <= 14.0 * 14.0:
                best_d2 = d2
                best = i
        return best

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._markers:
            self._drag_index = self._pick_marker_index(event.position())
            if self._drag_index >= 0:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._drag_index >= 0 and self._drag_index < len(self._markers):
            self._markers[self._drag_index]["x"] = max(0.0, min(float(self.width() - 1), float(event.position().x())))
            self._markers[self._drag_index]["y"] = max(0.0, min(float(self.height() - 1), float(event.position().y())))
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        self._drag_index = -1
        super().mouseReleaseEvent(event)


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
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        self.resize(1180, 820)
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
                    fetched = db.fetch_answer_keys_for_subject(subject_key_seed)
                    if fetched:
                        self.answer_key_data = fetched
                except Exception:
                    pass

        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.subject_name = QComboBox(); self.subject_name.setEditable(True); self.subject_name.addItems(subject_options)
        if str(data.get("name", "")).strip():
            self.subject_name.setCurrentText(str(data.get("name", "")).strip())

        self.block_name = QComboBox(); self.block_name.setEditable(True); self.block_name.addItems(block_options)
        self.block_name.setCurrentText(str(data.get("block", block_options[0] if block_options else "10")))

        self.template_path = QLineEdit(str(data.get("template_path", "")))
        self.scan_folder = QLineEdit(str(data.get("scan_folder", "")))
        self.answer_key = QLineEdit(str(data.get("answer_key_path", "")))
        self.answer_key_key = QLineEdit(str(data.get("answer_key_key", ""))); self.answer_key_key.setReadOnly(True)
        self.answer_codes = QLineEdit(", ".join(sorted((data.get("imported_answer_keys") or {}).keys()))); self.answer_codes.setReadOnly(True)
        self.answer_summary = QTextEdit()
        self.answer_summary.setReadOnly(True)
        self.answer_summary.setMinimumHeight(110)
        self.answer_summary.setMaximumHeight(180)

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

        row_tpl = QHBoxLayout(); row_tpl.addWidget(self.template_path); b_tpl = QPushButton("..."); row_tpl.addWidget(b_tpl); b_tpl_repo = QPushButton("Kho mẫu..."); row_tpl.addWidget(b_tpl_repo)
        b_tpl.clicked.connect(self._browse_template)
        b_tpl_repo.clicked.connect(self._pick_template_from_repo)
        row_scan = QHBoxLayout(); row_scan.addWidget(self.scan_folder); b_scan = QPushButton("..."); row_scan.addWidget(b_scan)
        b_scan.clicked.connect(self._browse_scan_folder)
        row_key = QHBoxLayout(); row_key.addWidget(self.answer_key); b_key = QPushButton("..."); row_key.addWidget(b_key)
        b_key_view = QPushButton("Xem/Sửa đáp án...")
        row_key.addWidget(b_key_view)
        b_key.clicked.connect(self._browse_answer_key)
        b_key_view.clicked.connect(self._edit_current_answer_keys)

        form.addRow("Tên môn", self.subject_name)
        form.addRow("Khối", self.block_name)
        form.addRow("Giấy thi riêng (tùy chọn)", row_tpl)
        form.addRow("Thư mục bài thi môn", row_scan)
        form.addRow("Đáp án môn", row_key)
        form.addRow("Mã đáp án môn_khối", self.answer_key_key)
        form.addRow("Các mã đề của môn", self.answer_codes)
        form.addRow("Tóm tắt đáp án", self.answer_summary)
        form.addRow("Số phần giấy thi", self.paper_part_label)
        form.addRow("Cách nhập điểm", self.score_mode)

        self.section_group = QGroupBox("Điểm theo phần")
        sec_form = QFormLayout(self.section_group)
        sec_form.addRow("MCQ tổng điểm", self.sec_mcq_total)
        sec_form.addRow("TF tổng điểm", self.sec_tf_total)
        sec_form.addRow("TF đúng 1 ý", self.sec_tf_1)
        sec_form.addRow("TF đúng 2 ý", self.sec_tf_2)
        sec_form.addRow("TF đúng 3 ý", self.sec_tf_3)
        sec_form.addRow("TF đúng 4 ý", self.sec_tf_4)
        sec_form.addRow("NUMERIC tổng điểm", self.sec_numeric_total)

        self.question_group = QGroupBox("Điểm theo câu")
        q_form = QFormLayout(self.question_group)
        q_form.addRow("MCQ điểm/câu", self.q_mcq)
        q_form.addRow("TF đúng 1 ý", self.q_tf_1)
        q_form.addRow("TF đúng 2 ý", self.q_tf_2)
        q_form.addRow("TF đúng 3 ý", self.q_tf_3)
        q_form.addRow("TF đúng 4 ý", self.q_tf_4)
        q_form.addRow("NUMERIC điểm/câu", self.q_numeric)

        form.addRow("Tổng điểm bài thi", self.total_score)

        left_left_lay.addLayout(form)
        lay.addWidget(self.section_group)
        lay.addWidget(self.question_group)

        self.subject_name.currentTextChanged.connect(self._update_answer_key_key)
        self.block_name.currentTextChanged.connect(self._update_answer_key_key)
        self.score_mode.currentTextChanged.connect(self._refresh_score_mode_ui)
        self.template_path.textChanged.connect(self._update_paper_parts)

        for w in [self.sec_mcq_total, self.sec_tf_total, self.sec_tf_1, self.sec_tf_2, self.sec_tf_3, self.sec_tf_4, self.sec_numeric_total,
                  self.q_mcq, self.q_tf_1, self.q_tf_2, self.q_tf_3, self.q_tf_4, self.q_numeric]:
            w.textChanged.connect(self._update_total_score)

        self._update_answer_key_key()
        self._refresh_answer_key_summary()
        self._update_paper_parts()
        self._refresh_score_mode_ui()

        self.section_group.setVisible(True)
        self.question_group.setVisible(True)
        self._refresh_score_mode_ui()

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

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

    def _refresh_score_mode_ui(self) -> None:
        section_mode = self.score_mode.currentText() == "Điểm theo phần"
        self.section_group.setVisible(section_mode)
        self.question_group.setVisible(not section_mode)
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

    def _edit_current_answer_keys(self) -> None:
        if not self.answer_key_data:
            QMessageBox.information(self, "Đáp án môn", "Môn này chưa có đáp án. Hãy import file hoặc thêm đáp án trước.")
            return
        dlg = ImportAnswerKeyDialog(self._build_imported_package_from_answer_data(self.answer_key_data), self)
        if dlg.exec() != QDialog.Accepted:
            return
        edited_package = dlg.result_answer_key()
        self.answer_key_data = self._answer_payload_from_package(edited_package)
        self._refresh_answer_key_summary()
        self._update_total_score()
        QMessageBox.information(self, "Đáp án môn", "Đã cập nhật đáp án hiện tại. Bạn có thể tiếp tục sửa hoặc thay đáp án khác.")

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
            "template_path": self.template_path.text().strip(),
            "scan_folder": self.scan_folder.text().strip(),
            "answer_key_path": self.answer_key.text().strip(),
            "answer_key_key": self.answer_key_key.text().strip(),
            "imported_answer_keys": self.answer_key_data,
            "score_mode": self.score_mode.currentText(),
            "section_scores": section_scores,
            "question_scores": question_scores,
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
        self.subject_configs: list[dict] = list(data.get("subject_configs", []))
        self.student_list_path_value = str(data.get("student_list_path", "") or "")
        self.student_rows: list[dict] = list(data.get("students", [])) if isinstance(data.get("students", []), list) else []
        self.on_batch_scan_subject = on_batch_scan_subject
        self.on_save_exam = on_save_exam
        self.stay_open_on_save = bool(stay_open_on_save)
        self.subject_options = subject_options
        self.block_options = block_options
        self.template_repo = template_repo or TemplateRepository()

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
        row_students.addWidget(btn_students)
        row_students.addWidget(self.student_count_label)
        btn_students.clicked.connect(self._import_student_list)

        form.addRow("Tên kỳ thi", self.exam_name)
        form.addRow("Giấy thi dùng chung", row_tpl)
        form.addRow("Thư mục gốc bài thi", row_scan)
        form.addRow("Danh sách học sinh", row_students)
        form.addRow("Cơ chế thư mục bài thi", self.scan_mode)
        form.addRow("Số phần trên giấy thi", self.paper_part_count)
        lay.addLayout(form)

        lay.addWidget(QLabel("Các môn trong kỳ thi"))
        self.subject_table = QTableWidget(0, 9)
        self.subject_table.setHorizontalHeaderLabels(["Môn", "Khối", "Key", "Mã đề", "Chế độ điểm", "Tổng điểm", "Template", "Trạng thái", "Thao tác"])
        self.subject_table.verticalHeader().setVisible(False)
        self.subject_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subject_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.subject_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.subject_table.setShowGrid(True)
        self.subject_table.setGridStyle(Qt.SolidLine)
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
        lay.addWidget(self.subject_table)

        row = QHBoxLayout()
        b_add = QPushButton("Thêm môn")
        b_edit = QPushButton("Sửa môn")
        b_del = QPushButton("Xoá môn")
        b_add.clicked.connect(self._add_subject)
        b_edit.clicked.connect(self._edit_subject)
        b_del.clicked.connect(self._delete_subject)
        row.addWidget(b_add); row.addWidget(b_edit); row.addWidget(b_del)
        lay.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._validate_and_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self._refresh_subject_list()

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
        self.student_list_path_value = path
        self.student_list_path.setText(path)
        self.student_count_label.setText(f"{len(self.student_rows)} học sinh")
        QMessageBox.information(
            self,
            "Danh sách học sinh",
            f"Đã {action_text} danh sách học sinh. Tổng hiện tại: {len(self.student_rows)} học sinh.",
        )

    def _refresh_subject_list(self) -> None:
        self.subject_table.setRowCount(len(self.subject_configs))
        style = self.style()
        for row_idx, cfg in enumerate(self.subject_configs):
            tpl = cfg.get("template_path") or "[dùng mẫu chung]"
            key = cfg.get("answer_key_key", "")
            mode = cfg.get("score_mode", "Điểm theo phần")
            total = cfg.get("total_exam_points", "-")
            codes = ",".join(sorted((cfg.get("imported_answer_keys") or {}).keys()))
            self.subject_table.setItem(row_idx, 0, QTableWidgetItem(str(cfg.get("name", "") or "-")))
            self.subject_table.setItem(row_idx, 1, QTableWidgetItem(str(cfg.get("block", "") or "-")))
            self.subject_table.setItem(row_idx, 2, QTableWidgetItem(str(key or "-")))
            self.subject_table.setItem(row_idx, 3, QTableWidgetItem(codes or "-"))
            self.subject_table.setItem(row_idx, 4, QTableWidgetItem(str(mode or "-")))
            self.subject_table.setItem(row_idx, 5, QTableWidgetItem(str(total or "-")))
            self.subject_table.setItem(row_idx, 6, QTableWidgetItem(str(tpl or "-")))
            status_text = "Đã nhận dạng" if bool(cfg.get("batch_saved")) else "-"
            self.subject_table.setItem(row_idx, 7, QTableWidgetItem(status_text))

            btn_batch_scan = QPushButton("Nhận dạng")
            btn_batch_scan.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
            btn_batch_scan.setToolTip("Batch Scan theo môn")
            btn_batch_scan.setEnabled(callable(self.on_batch_scan_subject))
            btn_batch_scan.clicked.connect(lambda _=False, i=row_idx: self._trigger_subject_batch_scan(i))
            wrap = QWidget()
            wrap_l = QHBoxLayout(wrap)
            wrap_l.setContentsMargins(0, 0, 0, 0)
            wrap_l.addWidget(btn_batch_scan)
            self.subject_table.setCellWidget(row_idx, 8, wrap)
        self.subject_table.resizeRowsToContents()

    def _current_subject_index(self) -> int:
        idx = self.subject_table.currentRow()
        return idx if 0 <= idx < len(self.subject_configs) else -1

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
        has_existing_batch = bool(old_cfg.get("batch_saved")) or bool(old_cfg.get("batch_saved_rows")) or bool(old_cfg.get("batch_saved_results"))
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
        self.subject_configs.append(payload)
        try:
            self.database.replace_answer_keys_for_subject(str(payload.get("answer_key_key", "") or ""), payload.get("imported_answer_keys", {}) or {})
            self.database.log_change(
                "answer_keys",
                str(payload.get("answer_key_key", "") or ""),
                "imported_answer_keys",
                "",
                payload.get("imported_answer_keys", {}) or {},
                "add_subject",
            )
        except Exception:
            pass
        self._refresh_subject_list()

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
        old_cfg = dict(self.subject_configs[idx])
        edited = dlg.payload()
        try:
            old_subject_key = str(old_cfg.get("answer_key_key", "") or "")
            new_subject_key = str(edited.get("answer_key_key", "") or "")
            if old_subject_key and old_subject_key != new_subject_key:
                self.database.replace_answer_keys_for_subject(old_subject_key, {})
            self.database.replace_answer_keys_for_subject(new_subject_key, edited.get("imported_answer_keys", {}) or {})
            if old_cfg.get("imported_answer_keys", {}) != edited.get("imported_answer_keys", {}):
                self.database.log_change(
                    "answer_keys",
                    new_subject_key,
                    "imported_answer_keys",
                    old_cfg.get("imported_answer_keys", {}) or {},
                    edited.get("imported_answer_keys", {}) or {},
                    "edit_subject",
                )
        except Exception:
            pass

        if not self._subject_identity_changed(old_cfg, edited):
            self.subject_configs[idx] = self._copy_noncritical_subject_updates(old_cfg, edited)
            self._refresh_subject_list()
            return

        decision = self._confirm_subject_identity_change(old_cfg, edited)
        if decision == "cancel":
            return
        if decision == "reset":
            updated = dict(edited)
            updated["batch_saved"] = False
            updated["batch_saved_at"] = "-"
            updated["batch_result_count"] = 0
            updated["batch_saved_rows"] = []
            updated["batch_saved_preview"] = []
            updated["batch_saved_results"] = []
            self.subject_configs[idx] = updated
        else:
            self.subject_configs[idx] = dict(old_cfg) | dict(edited)
        self._refresh_subject_list()

    def _delete_subject(self) -> None:
        idx = self._current_subject_index()
        if idx < 0:
            return
        del self.subject_configs[idx]
        self._refresh_subject_list()

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
            if callable(self.on_save_exam):
                ok = self.on_save_exam()
                if ok is False:
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Exam Grading System")
        self.resize(1200, 800)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.session: ExamSession | None = None
        self.template: Template | None = None
        self.answer_keys: AnswerKeyRepository | None = None
        self.scan_results = []
        self.scan_results_by_subject: dict[str, list] = {}
        self.batch_working_state_by_subject: dict[str, dict] = {}
        self.scan_files: list[Path] = []
        self.scan_blank_questions: dict[int, list[int]] = {}
        self.scan_blank_summary: dict[int, dict[str, list[int]]] = {}
        self.scan_manual_adjustments: dict[int, list[str]] = {}
        self.scan_edit_history: dict[int, list[str]] = {}
        self.scan_last_adjustment: dict[int, str] = {}
        self.score_rows = []
        self.scoring_results_by_subject: dict[str, dict[str, dict]] = {}
        self.scoring_phases: list[dict] = []
        self.imported_exam_codes: list[str] = []
        self.active_batch_subject_key: str | None = None
        self.subject_catalog: list[str] = ["Toán", "Ngữ văn", "Tiếng Anh", "Vật lý", "Hóa học", "Sinh học"]
        self.block_catalog: list[str] = ["10", "11", "12"]
        self.subjects: list[str] = list(self.subject_catalog)
        self.grades: list[str] = list(self.block_catalog)
        self.subject_management_mode = "subjects"
        self.subject_edit_index: int | None = None
        self.batch_editor_return_payload: dict | None = None
        self.batch_editor_return_session_id: str | None = None

        self.omr_processor = OMRProcessor()
        self.scoring_engine = ScoringEngine()
        self.current_session_path: Path | None = None
        self.current_session_id: str | None = None
        self.session_dirty = False

        self.database = OMRDatabase.default()
        self.session_registry: list[dict[str, str | bool]] = self._load_session_registry()
        self.template_repo = self._load_template_repository()
        self.template_editor_embedded: TemplateEditorWindow | None = None
        self.template_editor_mode = "library"

        self.stack = QStackedWidget()
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
        self.preview_zoom_factor = 1.0
        self.preview_source_pixmap = QPixmap()
        self.preview_rotation_by_index: dict[int, int] = {}
        self.preview_markers_by_index: dict[int, list[dict[str, float]]] = {}
        self.scan_forced_status_by_index: dict[int, str] = {}
        self.preview_drag_active = False
        self.preview_last_pos = None
        self.setCentralWidget(self.stack)

        self._build_menu()
        self.stack.currentChanged.connect(self._handle_stack_changed)
        db_subjects = self.database.fetch_catalog("subjects")
        db_blocks = self.database.fetch_catalog("blocks")
        if db_subjects:
            self.subject_catalog = db_subjects
            self.subjects = list(db_subjects)
        else:
            self.database.replace_catalog("subjects", self.subject_catalog)
        if db_blocks:
            self.block_catalog = db_blocks
            self.grades = list(db_blocks)
        else:
            self.database.replace_catalog("blocks", self.block_catalog)
        self._refresh_exam_list()
        self._refresh_batch_subject_controls()
        self._handle_stack_changed(self.stack.currentIndex())
        self.stack.setCurrentIndex(0)

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

    def _has_pending_unsaved_work(self) -> bool:
        if self._session_has_real_changes():
            return True
        if hasattr(self, "btn_save_batch_subject") and self.btn_save_batch_subject.isEnabled():
            return True
        if self.stack.currentIndex() == 5 and self.embedded_exam_dialog:
            return True
        return False

    def _session_has_real_changes(self) -> bool:
        if not self.session:
            return False

        if not self.current_session_id:
            return bool(getattr(self, "session_dirty", False))

        current_payload = self.session.to_dict()
        current_cfg = dict(current_payload.get("config", {}) or {})
        current_cfg["scoring_phases"] = list(self.scoring_phases)
        current_cfg["scoring_results"] = dict(self.scoring_results_by_subject)
        current_payload["config"] = current_cfg

        saved_payload = self.database.fetch_exam_session(self.current_session_id)
        if not isinstance(saved_payload, dict):
            return bool(getattr(self, "session_dirty", False))

        return json.dumps(current_payload, ensure_ascii=False, sort_keys=True, default=str) != json.dumps(
            saved_payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _confirm_before_switching_work(self, target_text: str) -> bool:
        if not self._has_pending_unsaved_work():
            return True
        return self._handle_pending_changes_before_switch(target_text)

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
        if self.stack.currentIndex() == 5 and self.embedded_exam_dialog:
            return bool(self._save_embedded_exam_editor())

        if hasattr(self, "btn_save_batch_subject") and self.btn_save_batch_subject.isEnabled():
            self._save_batch_for_selected_subject()
            if self.btn_save_batch_subject.isEnabled():
                return False

        if self.session and self._session_has_real_changes():
            self.save_session()
            return not self.session_dirty
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

    def _load_template_repository(self) -> TemplateRepository:
        payload = self.database.get_app_state("template_repository", {})
        if isinstance(payload, dict):
            try:
                return TemplateRepository.from_dict(payload)
            except Exception:
                pass
        return TemplateRepository()

    def _save_template_repository(self) -> None:
        try:
            self.database.set_app_state("template_repository", self.template_repo.to_dict())
        except Exception:
            pass

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

    def _build_exam_list_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Danh sách các kỳ thi"))

        self.exam_list_table = QTableWidget(0, 9)
        self.exam_list_table.setHorizontalHeaderLabels(["STT", "Tên kỳ thi", "Số môn", "Thư mục quét", "Môn học", "Trạng thái", "Xem", "Xoá", "Mặc định"])
        self.exam_list_table.verticalHeader().setVisible(False)
        self.exam_list_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.exam_list_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.exam_list_table.setSelectionMode(QAbstractItemView.SingleSelection)
        hdr = self.exam_list_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        layout.addWidget(self.exam_list_table)
        return w

    def _build_subject_management_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Quản lý môn học và khối"))

        tables_row = QHBoxLayout()

        self.subjects_table = QTableWidget(0, 1)
        self.subjects_table.setHorizontalHeaderLabels(["Subject Name"])
        self.subjects_table.verticalHeader().setVisible(False)
        self.subjects_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.subjects_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subjects_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.subjects_table.itemSelectionChanged.connect(lambda: self._handle_subject_management_selection("subjects"))

        self.grades_table = QTableWidget(0, 1)
        self.grades_table.setHorizontalHeaderLabels(["Grade"])
        self.grades_table.verticalHeader().setVisible(False)
        self.grades_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.grades_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.grades_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grades_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.grades_table.itemSelectionChanged.connect(lambda: self._handle_subject_management_selection("grades"))

        tables_row.addWidget(self.subjects_table)
        tables_row.addWidget(self.grades_table)
        layout.addLayout(tables_row)

        form = QFormLayout()
        self.subject_management_label = QLabel("Subject Name")
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
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
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
        self.subject_management_label.setText("Subject Name" if mode == "subjects" else "Grade")

    def _refresh_subject_management_tables(self) -> None:
        self.subjects = list(self.subject_catalog)
        self.grades = list(self.block_catalog)
        for mode, values in (("subjects", self.subjects), ("grades", self.grades)):
            table = self._subject_management_table(mode)
            table.blockSignals(True)
            table.setRowCount(len(values))
            for row, value in enumerate(values):
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
        self._apply_subject_management_values()
        self._refresh_subject_management_tables()
        self._set_subject_management_mode(self.subject_management_mode)

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
        if not self.session:
            return False
        cfg = dict(self.session.config or {})
        subject_configs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
        allowed = {x.casefold() for x in self.subject_catalog}
        filtered = [x for x in subject_configs if str(x.get("name", "")).strip().casefold() in allowed]
        changed = len(filtered) != len(subject_configs)
        if changed:
            cfg["subject_configs"] = filtered
            self.session.config = cfg
            self.session.subjects = [
                f"{str(x.get('name', '')).strip()}_{str(x.get('block', '')).strip()}"
                for x in filtered
                if str(x.get("name", "")).strip()
            ]
            self.session_dirty = True
            self._refresh_batch_subject_controls()
            self._refresh_session_info()
        return changed

    def _save_subject_management(self) -> None:
        value = self.subject_management_editor.text().strip()
        values = self._subject_management_values(self.subject_management_mode)
        label = "môn học" if self.subject_management_mode == "subjects" else "khối"
        if value:
            if self.subject_edit_index is None:
                values.append(value)
            else:
                values[self.subject_edit_index] = value
        if not self.subjects:
            QMessageBox.warning(self, "Quản lý môn học", "Danh sách môn học không được để trống.")
            return
        if not self.grades:
            QMessageBox.warning(self, "Quản lý khối", "Danh sách khối không được để trống.")
            return

        normalized = [item.strip() for item in values if item.strip()]
        if len(normalized) != len(set(x.casefold() for x in normalized)):
            QMessageBox.warning(self, "Quản lý môn học", f"Danh sách {label} không được trùng lặp.")
            return
        if self.subject_management_mode == "subjects":
            self.subjects = normalized
        else:
            self.grades = normalized
        self._apply_subject_management_values()
        self.session_dirty = True

        if self.session:
            cfg = dict(self.session.config or {})
            cfg["subject_catalog"] = list(self.subject_catalog)
            cfg["block_catalog"] = list(self.block_catalog)
            self.session.config = cfg
            removed = self._sync_subject_configs_with_catalog()
            if removed:
                QMessageBox.information(self, "Quản lý môn học", "Đã cập nhật danh sách môn/khối và đồng bộ các môn trong kỳ thi hiện tại.")
            else:
                QMessageBox.information(self, "Quản lý môn học", "Đã cập nhật danh sách môn và khối.")
        else:
            QMessageBox.information(self, "Quản lý môn học", "Đã cập nhật danh sách môn và khối.")
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

    def _make_row_icon_button(self, icon, tooltip: str, cb):
        btn = QPushButton()
        btn.setIcon(icon)
        btn.setToolTip(tooltip)
        btn.setFlat(True)
        btn.clicked.connect(cb)
        return btn

    def _set_scan_action_widget(self, row: int) -> None:
        if row < 0 or row >= self.scan_list.rowCount():
            return
        style = self.style()
        holder = QWidget()
        lay = QHBoxLayout(holder)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(2)
        btn_edit = self._make_row_icon_button(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Sửa bài thi", lambda _=False, r=row: self._edit_scan_row_by_index(r))
        btn_save = self._make_row_icon_button(style.standardIcon(QStyle.SP_DialogSaveButton), "Lưu bài thi", lambda _=False, r=row: self._save_scan_row_by_index(r))
        lay.addWidget(btn_edit)
        lay.addWidget(btn_save)
        self.scan_list.setCellWidget(row, 6, holder)

    def _edit_scan_row_by_index(self, row: int) -> None:
        if row < 0 or row >= self.scan_list.rowCount():
            return
        self.scan_list.selectRow(row)
        self._on_scan_selected()
        self._open_edit_selected_scan()

    def _save_scan_row_by_index(self, row: int) -> None:
        self._ensure_correction_state()
        if row < 0 or row >= self.scan_list.rowCount():
            return
        self.scan_list.selectRow(row)
        self._on_scan_selected()
        if self.correction_save_timer.isActive():
            self.correction_save_timer.stop()
            self._flush_pending_correction_updates()
        idx = self.scan_list.currentRow()
        if 0 <= idx < len(self.scan_results):
            res = self.scan_results[idx]
            self._persist_single_scan_result_to_db(res, note="row_action_save")
            self._refresh_all_statuses()
            self._update_scan_preview(idx)

    def _refresh_exam_list(self) -> None:
        self.exam_list_table.setRowCount(len(self.session_registry))
        style = self.style()
        for idx, row in enumerate(self.session_registry):
            sid = str(row.get("session_id", ""))
            name = str(row.get("name") or f"Kỳ thi {idx+1}")
            subject_text = "-"
            subject_count = "0"
            scan_root = "-"
            payload = self.database.fetch_exam_session(sid) if sid else None
            if sid and payload:
                try:
                    ses = ExamSession.from_dict(payload)
                    cfg = ses.config or {}
                    subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
                    subject_count = str(len(subject_cfgs))
                    subject_text = ", ".join(f"{x.get('name','?')}-{x.get('block','?')}" for x in subject_cfgs[:4])
                    if len(subject_cfgs) > 4:
                        subject_text += f" ...(+{len(subject_cfgs)-4})"
                    scan_root = str(cfg.get("scan_root", "") or "-")
                except Exception:
                    pass
            status = "Mặc định" if bool(row.get("default")) else "Thường"
            if sid and not payload:
                status = "Không tìm thấy"

            self.exam_list_table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, sid)
            self.exam_list_table.setItem(idx, 1, name_item)
            self.exam_list_table.setItem(idx, 2, QTableWidgetItem(subject_count))
            self.exam_list_table.setItem(idx, 3, QTableWidgetItem(scan_root))
            self.exam_list_table.setItem(idx, 4, QTableWidgetItem(subject_text or "-"))
            self.exam_list_table.setItem(idx, 5, QTableWidgetItem(status))

            b_edit = self._make_row_icon_button(style.standardIcon(QStyle.SP_DialogOpenButton), "Xem kỳ thi", lambda _=False, s=sid: self._edit_registry_session_by_id(s))
            b_del = self._make_row_icon_button(style.standardIcon(QStyle.SP_TrashIcon), "Xoá kỳ thi", lambda _=False, s=sid: self._delete_registry_session_by_id(s))
            b_def = self._make_row_icon_button(style.standardIcon(QStyle.SP_DialogApplyButton), "Đặt mặc định", lambda _=False, s=sid: self._set_default_registry_session_by_id(s))
            edit_wrap = QWidget(); e_l = QHBoxLayout(edit_wrap); e_l.setContentsMargins(0, 0, 0, 0); e_l.addWidget(b_edit)
            del_wrap = QWidget(); d_l = QHBoxLayout(del_wrap); d_l.setContentsMargins(0, 0, 0, 0); d_l.addWidget(b_del)
            def_wrap = QWidget(); f_l = QHBoxLayout(def_wrap); f_l.setContentsMargins(0, 0, 0, 0); f_l.addWidget(b_def)
            self.exam_list_table.setCellWidget(idx, 6, edit_wrap)
            self.exam_list_table.setCellWidget(idx, 7, del_wrap)
            self.exam_list_table.setCellWidget(idx, 8, def_wrap)

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
        if not self._confirm("Mở kỳ thi", "Bạn có chắc muốn mở kỳ thi này?"):
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
        if not self._confirm("Xem kỳ thi", "Bạn có chắc muốn xem kỳ thi này?"):
            return
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

    def _open_embedded_exam_editor(self, session_id: str, session: ExamSession, payload: dict) -> None:
        while self.exam_editor_layout.count():
            item = self.exam_editor_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.embedded_exam_session_id = session_id
        self.embedded_exam_session = session
        self.embedded_exam_original_payload = dict(payload)

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
        self.stack.setCurrentIndex(5)

    def _save_embedded_exam_editor(self) -> bool:
        if not self.embedded_exam_dialog or not self.embedded_exam_session_id:
            return False
        edited = self.embedded_exam_dialog.payload()
        self._register_templates_from_payload(edited)
        session_id = self.embedded_exam_session_id
        saved_payload = self.database.fetch_exam_session(session_id)
        if not saved_payload:
            QMessageBox.warning(self, "Sửa kỳ thi", "Không tìm thấy kỳ thi trong kho lưu trữ hệ thống.")
            return False
        try:
            session = ExamSession.from_dict(saved_payload)
            session.exam_name = edited.get("exam_name", session.exam_name)
            session.template_path = edited.get("common_template", session.template_path)
            session.subjects = [
                f"{str(x.get('name', '')).strip()}_{str(x.get('block', '')).strip()}"
                for x in edited.get("subject_configs", [])
                if str(x.get("name", "")).strip()
            ] or session.subjects
            incoming_students = edited.get("students", []) if isinstance(edited.get("students", []), list) else []
            session.students = [
                Student(
                    student_id=str(x.get("student_id", "") or "").strip(),
                    name=str(x.get("name", "") or "").strip(),
                    extra={
                        "birth_date": str(x.get("birth_date", "") or ""),
                        "class_name": str(x.get("class_name", "") or ""),
                        "exam_room": str(x.get("exam_room", "") or ""),
                    },
                )
                for x in incoming_students
                if str(x.get("student_id", "") or "").strip() and str(x.get("name", "") or "").strip()
            ]
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
            self.database.save_exam_session(session_id, session.exam_name, session.to_dict())
            self.embedded_exam_original_payload = edited
            self.session = session
            self.current_session_id = session_id
            self.current_session_path = self._session_path_from_id(session_id)
            self.session_dirty = False
            self._upsert_session_registry(session_id, session.exam_name)
            self._save_session_registry()
            self._refresh_exam_list()
            self._refresh_session_info()
            self._refresh_batch_subject_controls()
            QMessageBox.information(self, "Xem kỳ thi", "Đã lưu thông số kỳ thi.")
            if self.embedded_exam_dialog:
                self.embedded_exam_dialog.show()
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Sửa kỳ thi", f"Không thể sửa kỳ thi\n{exc}")
            return False

    def _close_embedded_exam_editor(self) -> None:
        self.embedded_exam_dialog = None
        self.embedded_exam_session = None
        self.embedded_exam_session_id = None
        self.embedded_exam_original_payload = None
        self.stack.setCurrentIndex(0)

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
        subject_cfg = dict(payload.get("subject_config") or {})
        if not subject_cfg:
            QMessageBox.warning(self, "Batch Scan", "Không tìm thấy cấu hình môn để nhận dạng.")
            return

        exam_name = str(payload.get("exam_name") or base_session.exam_name or "Kỳ thi")
        common_template = str(payload.get("common_template") or base_session.template_path or "")
        all_subjects = payload.get("subject_configs")
        if not isinstance(all_subjects, list) or not all_subjects:
            all_subjects = list((base_session.config or {}).get("subject_configs", []))
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
        selected_subject_index = int(payload.get("selected_subject_index", 0) or 0)
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
        self.session_dirty = True
        self._refresh_session_info()
        self._refresh_batch_subject_controls()
        self.batch_subject_combo.setCurrentIndex(max(1, selected_subject_index + 1))
        self.stack.setCurrentIndex(1)

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
        self.session_registry = self._load_session_registry()
        self._save_session_registry()
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
        self.session_registry = self._load_session_registry()
        self._save_session_registry()
        self._refresh_exam_list()

    def _open_session_path(self, path: Path) -> None:
        try:
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
            cfg = self.session.config or {}
            self.scoring_phases = list(cfg.get("scoring_phases", [])) if isinstance(cfg.get("scoring_phases", []), list) else []
            self.scoring_results_by_subject = dict(cfg.get("scoring_results", {})) if isinstance(cfg.get("scoring_results", {}), dict) else {}
            self.scan_results_by_subject = {}
            self.batch_working_state_by_subject = {}
            for sc in (cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []):
                if not isinstance(sc, dict):
                    continue
                key = self._subject_key_from_cfg(sc)
                rows = sc.get("batch_saved_results", [])
                if not isinstance(rows, list) or not rows:
                    continue
                restored: list[OMRResult] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    try:
                        restored.append(self._deserialize_omr_result(row))
                    except Exception:
                        continue
                if restored:
                    self.scan_results_by_subject[key] = restored
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
            self.batch_editor_return_payload = None
            self.batch_editor_return_session_id = None
            self._refresh_session_info()
            self._refresh_batch_subject_controls()
            self._refresh_scoring_phase_table()
            self.stack.setCurrentIndex(1)
            QMessageBox.information(self, "Open session", "Đã mở kỳ thi thành công.")
        except Exception as exc:
            QMessageBox.warning(self, "Open session", f"Không thể mở kỳ thi:\n{exc}")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        act_new = file_menu.addAction("Tạo kỳ thi mới")
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self.action_create_session)

        act_open = file_menu.addAction("Mở từ danh sách")
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self.action_open_session)

        act_save = file_menu.addAction("Lưu kỳ thi")
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self.action_save_session)

        act_save_as = file_menu.addAction("Lưu dưới tên khác")
        act_save_as.triggered.connect(self.action_save_session_as)

        act_close_current = file_menu.addAction("Đóng kỳ thi hiện tại")
        act_close_current.triggered.connect(self.action_close_current_session)

        file_menu.addSeparator()
        self.act_manage_template = file_menu.addAction("Quản lý mẫu giấy thi")
        self.act_manage_template.triggered.connect(self.action_manage_template)
        self.act_close_template_module = file_menu.addAction("Đóng quản lý mẫu giấy thi")
        self.act_close_template_module.triggered.connect(self._close_template_module)

        act_manage_subject = file_menu.addAction("Quản lý môn học")
        act_manage_subject.triggered.connect(self.action_manage_subjects)

        file_menu.addSeparator()
        act_exit = file_menu.addAction("Thoát")
        act_exit.triggered.connect(self.action_exit)

        exam_menu = self.menuBar().addMenu("Exam")
        exam_menu.addAction("Load Template JSON", self.action_load_template)
        exam_menu.addAction("Load Answer Keys JSON", self.action_load_answer_keys)
        exam_menu.addAction("Import Answer Key", self.action_import_answer_key)
        exam_menu.addAction("Export Answer Key Sample", self.action_export_answer_key_sample)
        exam_menu.addAction("Batch Scan Images", self.action_run_batch_scan)
        exam_menu.addAction("Sửa bài thi được chọn", self.action_edit_selected_scan)
        exam_menu.addAction("Load Selected Scan Result", self.action_load_selected_scan_result)
        exam_menu.addAction("Apply Manual Correction", self.action_apply_manual_correction)

        scoring_menu = self.menuBar().addMenu("Scoring")
        scoring_menu.addAction("Calculate & Preview Scores", self.action_calculate_scores)
        scoring_menu.addAction("Export Results", self.action_export_results)

        self.template_module_menu = self.menuBar().addMenu("Template Editor")
        self.template_module_menu.addAction("Tạo mới", self._create_new_template)
        self.template_module_menu.addAction("Sửa", self._edit_selected_template)
        self.template_module_menu.addAction("Xoá", self._delete_selected_template)
        self.template_module_menu.addSeparator()
        self.template_module_menu.addAction("Save", self._save_current_template)
        self.template_module_menu.addAction("Save As", self._save_current_template_as)
        self.template_module_menu.addSeparator()
        self.template_module_menu.addAction("Close", self._close_template_module)

        toolbar = QToolBar("Ribbon")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(toolbar)
        self.main_ribbon = toolbar

        style = self.style()
        # Session actions
        self.ribbon_new_exam_action = toolbar.addAction(style.standardIcon(QStyle.SP_FileIcon), "Tạo mới", self.action_create_session)
        self.ribbon_view_exam_action = toolbar.addAction(style.standardIcon(QStyle.SP_DialogOpenButton), "Xem", self._edit_selected_registry_session)
        self.ribbon_delete_exam_action = toolbar.addAction(style.standardIcon(QStyle.SP_TrashIcon), "Xoá", self._delete_selected_registry_session)
        toolbar.addSeparator()
        # Workflow actions
        self.ribbon_batch_scan_action = toolbar.addAction(style.standardIcon(QStyle.SP_MediaPlay), "Nhận dạng", self.action_run_batch_scan)
        self.ribbon_scoring_action = toolbar.addAction(style.standardIcon(QStyle.SP_CommandLink), "Tính điểm", self.action_calculate_scores)
        self.ribbon_export_action = toolbar.addAction(style.standardIcon(QStyle.SP_DriveNetIcon), "Xuất KQ", self.action_export_results)
        toolbar.addSeparator()
        self.ribbon_add_subject_action = toolbar.addAction(style.standardIcon(QStyle.SP_FileDialogNewFolder), "Add Subject", self._subject_management_add)
        self.ribbon_edit_subject_action = toolbar.addAction(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Edit", self._subject_management_edit)
        self.ribbon_delete_subject_action = toolbar.addAction(style.standardIcon(QStyle.SP_TrashIcon), "Delete Subject", self._subject_management_delete)
        self.ribbon_save_subject_action = toolbar.addAction(style.standardIcon(QStyle.SP_DialogSaveButton), "Save", self._save_subject_management)
        toolbar.addSeparator()
        self.ribbon_new_template_action = QAction(style.standardIcon(QStyle.SP_FileIcon), "Tạo mới", self)
        self.ribbon_new_template_action.triggered.connect(self._create_new_template)
        toolbar.addAction(self.ribbon_new_template_action)
        self.ribbon_edit_template_action = QAction(style.standardIcon(QStyle.SP_DialogOpenButton), "Sửa", self)
        self.ribbon_edit_template_action.triggered.connect(self._edit_selected_template)
        toolbar.addAction(self.ribbon_edit_template_action)
        self.ribbon_delete_template_action = QAction(style.standardIcon(QStyle.SP_TrashIcon), "Xoá", self)
        self.ribbon_delete_template_action.triggered.connect(self._delete_selected_template)
        toolbar.addAction(self.ribbon_delete_template_action)
        self.ribbon_save_template_action = QAction(style.standardIcon(QStyle.SP_DialogSaveButton), "Save", self)
        self.ribbon_save_template_action.triggered.connect(self._save_current_template)
        toolbar.addAction(self.ribbon_save_template_action)
        self.ribbon_save_template_as_action = QAction(style.standardIcon(QStyle.SP_DriveFDIcon), "Save As", self)
        self.ribbon_save_template_as_action.triggered.connect(self._save_current_template_as)
        toolbar.addAction(self.ribbon_save_template_as_action)
        self.ribbon_close_template_action = QAction(style.standardIcon(QStyle.SP_DialogCloseButton), "Close", self)
        self.ribbon_close_template_action.triggered.connect(self._close_template_module)
        toolbar.addAction(self.ribbon_close_template_action)

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
                cfg = dict(self.session.config or {})
                cfg["scoring_phases"] = list(self.scoring_phases)
                cfg["scoring_results"] = dict(self.scoring_results_by_subject)
                self.session.config = cfg
                self.database.save_exam_session(self.current_session_id, self.session.exam_name, self.session.to_dict())
            self._upsert_session_registry(self.current_session_id, self.session.exam_name if self.session else None)
            self._save_session_registry()
            self._refresh_exam_list()
            self.session_dirty = False
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
            self.database.save_exam_session(self.current_session_id, self.session.exam_name, self.session.to_dict())
            self._upsert_session_registry(self.current_session_id, self.session.exam_name if self.session else None)
            self._save_session_registry()
            self._refresh_exam_list()
            self.session_dirty = False
            return True
        except Exception:
            self.session_dirty = True
            return False

    def save_session_as(self) -> None:
        if not self.session:
            self.create_session()
        if not self.session:
            return

        current_name = str(self.session.exam_name or "").strip() or "Kỳ thi"
        while True:
            new_name, ok = QInputDialog.getText(
                self,
                "Lưu dưới tên khác",
                "Nhập tên kỳ thi mới:",
                text=current_name,
            )
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

        old_session_id = self.current_session_id
        old_session_path = self.current_session_path
        old_session_name = self.session.exam_name
        self.session.exam_name = new_name
        self.current_session_id = self._generate_session_id(new_name)
        self.current_session_path = self._session_path_from_id(self.current_session_id)

        try:
            cfg = dict(self.session.config or {})
            cfg["scoring_phases"] = list(self.scoring_phases)
            cfg["scoring_results"] = dict(self.scoring_results_by_subject)
            self.session.config = cfg
            self.database.save_exam_session(self.current_session_id, self.session.exam_name, self.session.to_dict())
            self._upsert_session_registry(self.current_session_id, self.session.exam_name)
            self._save_session_registry()
            self._refresh_exam_list()
            self.session_dirty = False
            QMessageBox.information(self, "Lưu dưới tên khác", "Đã lưu thành kỳ thi mới trong kho hệ thống.")
        except Exception as exc:
            self.current_session_id = old_session_id
            self.current_session_path = old_session_path
            self.session.exam_name = old_session_name
            QMessageBox.warning(self, "Lưu dưới tên khác", f"Không thể lưu kỳ thi:\n{exc}")

    def close_current_session(self) -> None:
        if self.session_dirty:
            if not self._confirm("Dữ liệu chưa lưu", "Kỳ thi hiện tại có thay đổi chưa lưu. Vẫn đóng?"):
                return
        self.session = None
        self.template = None
        self.answer_keys = None
        self.scan_results = []
        self.scan_results_by_subject = {}
        self.batch_working_state_by_subject = {}
        self.scoring_phases = []
        self.scoring_results_by_subject = {}
        self.scan_results_by_subject = {}
        self.current_session_path = None
        self.current_session_id = None
        self.session_dirty = False
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

    def manage_subjects(self) -> None:
        self._refresh_subject_management_tables()
        self._set_subject_management_mode("subjects")
        self.stack.setCurrentIndex(2)

    def action_create_session(self) -> None:
        if not self._confirm_before_switching_work("kỳ thi mới"):
            return
        if not self._confirm("Tạo kỳ thi mới", "Bạn có chắc muốn tạo kỳ thi mới?"):
            return
        dlg = NewExamDialog(self.subject_catalog, self.block_catalog, parent=self, template_repo=self.template_repo)
        if dlg.exec() != QDialog.Accepted:
            return
        self.create_session(dlg.payload())
        # Persist immediately so the new exam appears in the list right after Save.
        self.save_session()
        # Saving/creating an exam should not force users into Batch Scan immediately.
        # Keep them on the exam list screen after save.
        self.stack.setCurrentIndex(0)

    def action_open_session(self) -> None:
        if not self._confirm_before_switching_work("kỳ thi khác"):
            return
        if self._confirm("Mở kỳ thi", "Bạn có chắc muốn mở kỳ thi?"):
            self.open_session()

    def action_save_session(self) -> None:
        if self._confirm("Lưu kỳ thi", "Bạn có chắc muốn lưu kỳ thi?"):
            if self.stack.currentIndex() == 5 and self.embedded_exam_dialog:
                self._save_embedded_exam_editor()
                return
            if self.stack.currentIndex() == 4 and self.template_editor_embedded:
                self._save_current_template()
                return
            self.save_session()

    def action_save_session_as(self) -> None:
        if self._confirm("Lưu dưới tên khác", "Bạn có chắc muốn lưu kỳ thi dưới tên khác?"):
            self.save_session_as()

    def action_close_current_session(self) -> None:
        if not self._confirm_before_switching_work("đóng kỳ thi hiện tại"):
            return
        if self._confirm("Đóng kỳ thi", "Bạn có chắc muốn đóng kỳ thi hiện tại?"):
            self.close_current_session()

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

    def _handle_stack_changed(self, index: int) -> None:
        subject_management_visible = index == 2
        template_library_visible = index == 3
        template_editor_visible = index == 4
        template_visible = template_library_visible or template_editor_visible
        for action in [
            getattr(self, "ribbon_new_exam_action", None),
            getattr(self, "ribbon_view_exam_action", None),
            getattr(self, "ribbon_delete_exam_action", None),
            getattr(self, "ribbon_batch_scan_action", None),
            getattr(self, "ribbon_scoring_action", None),
            getattr(self, "ribbon_export_action", None),
        ]:
            if action is not None:
                action.setVisible(not subject_management_visible and not template_visible)
        for action in [
            getattr(self, "ribbon_add_subject_action", None),
            getattr(self, "ribbon_edit_subject_action", None),
            getattr(self, "ribbon_delete_subject_action", None),
            getattr(self, "ribbon_save_subject_action", None),
        ]:
            if action is not None:
                action.setVisible(subject_management_visible)
        template_library_actions = [
            getattr(self, "ribbon_new_template_action", None),
            getattr(self, "ribbon_edit_template_action", None),
            getattr(self, "ribbon_delete_template_action", None),
        ]
        template_editor_actions = [
            getattr(self, "ribbon_save_template_action", None),
            getattr(self, "ribbon_save_template_as_action", None),
        ]
        for action in template_library_actions:
            if action is not None:
                action.setVisible(template_library_visible)
        for action in template_editor_actions:
            if action is not None:
                action.setVisible(False)
        if getattr(self, "ribbon_close_template_action", None) is not None:
            self.ribbon_close_template_action.setVisible(template_library_visible)

        if hasattr(self, "main_ribbon"):
            self.main_ribbon.setVisible(index != 4)
        if hasattr(self, "template_module_menu"):
            self.template_module_menu.menuAction().setVisible(template_visible)
            self._rebuild_template_module_menu(library_mode=template_library_visible, editor_mode=template_editor_visible)
        if hasattr(self, "act_close_template_module"):
            self.act_close_template_module.setVisible(template_visible)

    def action_manage_subjects(self) -> None:
        self.manage_subjects()

    def action_exit(self) -> None:
        if not self._confirm_before_switching_work("thoát ứng dụng"):
            return
        if self._confirm("Thoát", "Bạn có chắc muốn thoát ứng dụng?"):
            self.close()

    def action_load_template(self) -> None:
        if self._confirm("Load Template", "Bạn có chắc muốn tải Template JSON?"):
            self.load_template()

    def action_load_answer_keys(self) -> None:
        if self._confirm("Load Answer Keys", "Bạn có chắc muốn tải Answer Keys JSON?"):
            self.load_answer_keys()

    def action_import_answer_key(self) -> None:
        if self._confirm("Import Answer Key", "Bạn có chắc muốn import Answer Key?"):
            self.import_answer_key_file()

    def action_export_answer_key_sample(self) -> None:
        if self._confirm("Export Sample", "Bạn có chắc muốn export file mẫu?"):
            self.export_answer_key_sample()

    def _start_batch_scan_from_ui(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Batch Scan", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        cfgs = self._effective_subject_configs_for_batch()
        if not cfgs:
            QMessageBox.warning(self, "Batch Scan", "Kỳ thi hiện tại chưa có môn thi để nhận dạng.")
            return
        if self.stack.currentIndex() != 1:
            self.stack.setCurrentIndex(1)
        self._refresh_batch_subject_controls()
        self._show_batch_scan_panel()
        if hasattr(self, "batch_subject_combo") and self.batch_subject_combo.currentIndex() <= 0 and self.batch_subject_combo.count() > 1:
            self.batch_subject_combo.setCurrentIndex(1)

    def action_run_batch_scan(self) -> None:
        self._start_batch_scan_from_ui()

    def action_execute_batch_scan(self) -> None:
        if self.stack.currentIndex() != 1:
            self._start_batch_scan_from_ui()
            return
        self.run_batch_scan()

    def action_edit_selected_scan(self) -> None:
        if self._confirm("Sửa bài thi", "Bạn có chắc muốn sửa bài thi được chọn?"):
            self._open_edit_selected_scan()

    def action_load_selected_scan_result(self) -> None:
        if self._confirm("Load Selected", "Bạn có chắc muốn load kết quả bài thi đang chọn?"):
            self._load_selected_result_for_correction()

    def action_apply_manual_correction(self) -> None:
        if self._confirm("Apply Correction", "Bạn có chắc muốn áp dụng manual correction?"):
            self.apply_manual_correction()

    def action_calculate_scores(self) -> None:
        if not self._confirm_before_switching_work("màn hình Tính điểm"):
            return
        self._open_scoring_view()

    def action_export_results(self) -> None:
        if self._confirm("Export Results", "Bạn có chắc muốn export kết quả?"):
            self.export_results()

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

    def export_answer_key_sample(self) -> None:
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Sample Answer Key",
            "answer_key_sample.xlsx",
            "Excel (*.xlsx);;CSV (*.csv)",
        )
        if not save_path:
            return

        import pandas as pd

        data = {
            "Question": list(range(1, 19)) + [1, 2, 3, 4] + [1, 2, 3, 4, 5, 6],
            "0101": ["C", "C", "C", "B", "D", "D", "D", "B", "B", "B", "B", "B", "D", "C", "D", "B", "C", "C", "ĐSĐĐ", "ĐĐĐS", "ĐSĐS", "ĐDDS", "5", "69", "0,61", "58,3", "49,6", "2"],
            "0102": ["B", "C", "C", "C", "D", "D", "B", "D", "B", "B", "B", "C", "C", "B", "D", "C", "D", "B", "ĐĐSĐ", "ĐĐĐS", "ĐĐSĐ", "ĐSDS", "2", "69", "58,3", "5", "49,6", "0,61"],
            "0103": ["C", "C", "C", "B", "C", "B", "D", "D", "C", "B", "B", "C", "B", "D", "C", "D", "B", "B", "SĐĐĐ", "ĐĐĐS", "ĐĐSĐ", "SĐDS", "2", "0,61", "58,3", "49,6", "5", "69"],
            "0104": ["C", "C", "C", "B", "C", "B", "D", "D", "C", "B", "B", "C", "B", "D", "C", "D", "B", "B", "ĐĐĐS", "SĐSĐ", "ĐĐSĐ", "SĐDD", "5", "2", "49,6", "0,61", "58,3", "69"],
        }
        df = pd.DataFrame(data)
        path = Path(save_path)
        if path.suffix.lower() == ".csv" or "CSV" in selected_filter:
            if path.suffix.lower() != ".csv":
                path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
        else:
            if path.suffix.lower() != ".xlsx":
                path = path.with_suffix(".xlsx")
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, startrow=1)
                ws = writer.sheets[next(iter(writer.sheets))]
                ws["A1"] = "Câu hỏi"
                ws["B1"] = "Mã đề thi"
                ws.merge_cells(start_row=1, start_column=2, end_row=1, end_column=5)
                ws["B2"] = "0101"
                ws["C2"] = "0102"
                ws["D2"] = "0103"
                ws["E2"] = "0104"
        QMessageBox.information(self, "Sample exported", f"Saved sample answer key file to:\n{path}")

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
        self.batch_recognition_mode_combo = QComboBox()
        self.batch_recognition_mode_combo.addItem("Tự động (khuyến nghị)", "auto")
        self.batch_recognition_mode_combo.addItem("Mẫu cũ / Anchor chuẩn", "legacy")
        self.batch_recognition_mode_combo.addItem("Mẫu mới / Anchor sát biên", "border")
        self.batch_recognition_mode_combo.addItem("Anchor 1 phía (ruler theo dòng)", "one_side")
        self.batch_recognition_mode_combo.addItem("Lai (thử nhiều cơ chế)", "hybrid")
        self.batch_template_value = QLineEdit("-"); self.batch_template_value.setReadOnly(True)
        self.batch_answer_codes_value = QLineEdit("-"); self.batch_answer_codes_value.setReadOnly(True)
        self.batch_student_id_value = QLineEdit("-"); self.batch_student_id_value.setReadOnly(True)
        self.batch_scan_folder_value = QLineEdit("-"); self.batch_scan_folder_value.setReadOnly(True)
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

        action_row = QHBoxLayout()
        action_row.addWidget(self.btn_batch_recognize)
        action_row.addWidget(self.btn_save_batch_subject)
        action_row.addWidget(self.btn_close_batch_view)
        action_row.addStretch()

        batch_form.addRow("Môn", self.batch_subject_combo)
        batch_form.addRow("Cơ chế nhận dạng", self.batch_recognition_mode_combo)
        batch_form.addRow("Mẫu giấy dùng", self.batch_template_value)
        batch_form.addRow("Mã đề", self.batch_answer_codes_value)
        batch_form.addRow("Vùng STUDENT ID", self.batch_student_id_value)
        batch_form.addRow("Thư mục quét", self.batch_scan_folder_value)
        batch_form.addRow("", action_row)

        self.filter_column = QComboBox()
        self.filter_column.addItems(["STUDENT ID", "Họ tên", "Ngày sinh", "Nội dung", "Status"])
        self.filter_column.currentTextChanged.connect(self._apply_scan_filter)
        self.search_value = QLineEdit()
        self.search_value.setPlaceholderText("Filter theo tiêu đề bảng đang chọn")
        self.search_value.textChanged.connect(self._apply_scan_filter)

        search_row = QHBoxLayout()
        search_row.addWidget(self.filter_column)
        search_row.addWidget(self.search_value)

        self.scan_list = QTableWidget(0, 7)
        self.scan_list.setHorizontalHeaderLabels(["STUDENT ID", "Mã đề", "Họ tên", "Ngày sinh", "Nội dung", "Status", "Chức năng"])
        self.scan_list.verticalHeader().setVisible(False)
        self.scan_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.scan_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scan_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.scan_list.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().sectionClicked.connect(self._on_scan_header_clicked)
        self.scan_list.itemSelectionChanged.connect(self._on_scan_selected)
        self.scan_list.cellDoubleClicked.connect(self._open_edit_selected_scan)
        self.scan_list.cellClicked.connect(self._on_scan_cell_clicked)
        self.progress = QProgressBar()

        self.scan_image_preview = PreviewImageWidget(); self.scan_image_preview.setText("Chọn bài thi ở danh sách bên trái")
        self.scan_image_scroll = QScrollArea()
        self.scan_image_scroll.setWidgetResizable(True)
        self.scan_image_scroll.setAlignment(Qt.AlignCenter)
        self.scan_image_scroll.setWidget(self.scan_image_preview)
        self.scan_image_scroll.viewport().installEventFilter(self)

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setMaximumWidth(36)
        self.btn_zoom_out.clicked.connect(self._zoom_preview_out)
        self.btn_zoom_reset = QPushButton("100%")
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
        self.scan_lr_split.setStretchFactor(0, 6)
        self.scan_lr_split.setStretchFactor(1, 4)
        self.scan_lr_split.setSizes([720, 480])

        # Create scoring widgets with explicit parent to avoid lifecycle issues
        # on some PySide6 builds (preventing "Internal C++ object ... already deleted").
        self.scoring_panel = QWidget(w)

        self.score_preview_table = QTableWidget(0, 16, self.scoring_panel)
        self.score_preview_table.setHorizontalHeaderLabels([
            "Student ID", "Name", "Subject", "Exam Code", "MCQ đúng", "TF đúng", "NUM đúng", "Correct", "Wrong", "Blank", "Số câu FULL", "Điểm FULL", "Score", "MCQ đáp án|bài làm", "TF đáp án|bài làm", "NUM đáp án|bài làm"
        ])
        self.score_preview_table.verticalHeader().setVisible(False)
        self.score_preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.score_preview_table.horizontalHeader().setSectionResizeMode(13, QHeaderView.Stretch)
        self.score_preview_table.horizontalHeader().setSectionResizeMode(14, QHeaderView.Stretch)
        self.score_preview_table.horizontalHeader().setSectionResizeMode(15, QHeaderView.Stretch)
        self.score_preview_table.setColumnWidth(13, 280)
        self.score_preview_table.setColumnWidth(14, 280)
        self.score_preview_table.setColumnWidth(15, 320)

        self.scoring_subject_combo = QComboBox()
        self.scoring_subject_combo.currentIndexChanged.connect(self._handle_scoring_subject_changed)
        self.scoring_mode_combo = QComboBox()
        self.scoring_mode_combo.addItems(["Tính lại toàn bộ", "Chỉ tính bài chưa có điểm"])
        self.scoring_phase_note = QLineEdit()
        self.scoring_phase_note.setPlaceholderText("Ghi chú pha chấm điểm (tuỳ chọn)")
        self.btn_scoring_run = QPushButton("Chấm điểm")
        self.btn_scoring_run.clicked.connect(self._run_scoring_from_panel)
        self.btn_scoring_save = QPushButton("Lưu điểm")
        self.btn_scoring_save.clicked.connect(self._save_current_work)
        self.btn_scoring_back = QPushButton("Quay lại Batch Scan")
        self.btn_scoring_back.clicked.connect(self._back_to_batch_scan)
        scoring_top = QHBoxLayout()
        scoring_top.addWidget(QLabel("Môn"))
        scoring_top.addWidget(self.scoring_subject_combo, 2)
        scoring_top.addWidget(QLabel("Cơ chế"))
        scoring_top.addWidget(self.scoring_mode_combo, 2)
        scoring_top.addWidget(self.scoring_phase_note, 3)
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

        self.dashboard_summary_label = QLabel("Dashboard DB: chưa có dữ liệu.")
        self.dashboard_summary_label.setWordWrap(True)

        scoring_panel_layout = QVBoxLayout(self.scoring_panel)
        scoring_panel_layout.setContentsMargins(0, 0, 0, 0)
        scoring_panel_layout.addLayout(scoring_top)
        scoring_panel_layout.addWidget(self.dashboard_summary_label)
        scoring_panel_layout.addWidget(self.score_preview_table, 7)
        scoring_panel_layout.addWidget(QLabel("Lịch sử pha chấm điểm"))
        scoring_panel_layout.addWidget(self.scoring_phase_table, 3)

        layout.addWidget(self.progress)
        layout.addWidget(self.scan_lr_split)
        layout.addWidget(self.scoring_panel)
        self.scoring_panel.setVisible(False)
        return w

    def _close_batch_scan_view(self) -> None:
        if self.batch_editor_return_payload is None:
            self.stack.setCurrentIndex(0)
            return

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "answer_string": str(getattr(result, "answer_string", "") or ""),
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            answer_string=str(payload.get("answer_string", "") or ""),
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _collect_current_subject_results_for_save(self, subject_key: str) -> list[OMRResult]:
        key = str(subject_key or "").strip()
        base_results = list(self.scan_results_by_subject.get(key) or self.scan_results or [])
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            return base_results

        by_image: dict[str, OMRResult] = {}
        for res in base_results:
            img = str(getattr(res, "image_path", "") or "").strip()
            if img and img not in by_image:
                by_image[img] = res

        out: list[OMRResult] = []
        for r in range(row_count):
            sid_item = self.scan_list.item(r, 0)
            image_path = str(sid_item.data(Qt.UserRole) if sid_item else "")
            exam_code = str(sid_item.data(Qt.UserRole + 1) if sid_item else "")
            sid_text = str(sid_item.text() if sid_item else "-")
            base = by_image.get(image_path) if image_path else None
            if base is not None:
                res = copy.deepcopy(base)
            else:
                res = OMRResult(image_path=image_path)
            res.student_id = "" if sid_text == "-" else sid_text
            res.exam_code = exam_code
            if hasattr(self, "scan_list"):
                res.full_name = str(self.scan_list.item(r, 1).text() if self.scan_list.item(r, 1) else "")
                res.birth_date = str(self.scan_list.item(r, 2).text() if self.scan_list.item(r, 2) else "")
            res.sync_legacy_aliases()
            out.append(res)

        return out


    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        db_rows = self.database.fetch_scan_results_for_subject(subject_key)
        if db_rows:
            out: list[OMRResult] = []
            for item in db_rows:
                try:
                    out.append(self._deserialize_omr_result(item))
                except Exception:
                    continue
            if out:
                return out
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _refresh_scan_results_from_db(self, subject_key: str) -> list[OMRResult]:
        rows = self.database.fetch_scan_results_for_subject(subject_key)
        refreshed: list[OMRResult] = []
        for item in rows:
            try:
                refreshed.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        if refreshed:
            self.scan_results_by_subject[subject_key] = list(refreshed)
        return refreshed

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

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        selected_subject = self._resolve_preferred_scoring_subject()
        self._populate_scoring_subjects(selected_subject)
        self._refresh_scoring_phase_table()
        self._refresh_dashboard_summary_from_db(selected_subject)
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _handle_scoring_subject_changed(self, _index: int) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        if subject_key:
            self._refresh_dashboard_summary_from_db(subject_key)

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            disk_cfg = ses.config or {}
            mem_cfg = self.session.config if self.session else {}
            cfg = {**disk_cfg, **(mem_cfg or {})}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._collect_current_subject_results_for_save(subject_key)
            self.scan_results_by_subject[subject_key] = list(current_results)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "preview_rotation": int(self.preview_rotation_by_index.get(r, 0) or 0),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "preview_rotation": int(self.preview_rotation_by_index.get(r, 0) or 0),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                merged_session_cfg = {**(self.session.config or {}), "subject_configs": subject_cfgs}
                if self.scoring_phases:
                    merged_session_cfg["scoring_phases"] = list(self.scoring_phases)
                if self.scoring_results_by_subject:
                    merged_session_cfg["scoring_results"] = dict(self.scoring_results_by_subject)
                self.session.config = merged_session_cfg
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.correction_ui_loading = False
        self.correction_pending_payload: dict[str, object] = {}
        self.correction_save_timer = QTimer(self)
        self.correction_save_timer.setSingleShot(True)
        self.correction_save_timer.timeout.connect(self._flush_pending_correction_updates)

        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)

        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_layout.setContentsMargins(4, 4, 4, 4)
        self.answer_editor_layout.setSpacing(8)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)

        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs

            # Keep selected subject and show its just-saved content right away.
            self._refresh_batch_subject_controls()
            if hasattr(self, "batch_subject_combo"):
                for i in range(1, self.batch_subject_combo.count()):
                    cfg_i = self.batch_subject_combo.itemData(i)
                    if isinstance(cfg_i, dict) and self._subject_key_from_cfg(cfg_i) == subject_key:
                        self.batch_subject_combo.setCurrentIndex(i)
                        self._on_batch_subject_changed(i)
                        break

            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs

            # Keep selected subject in combo and refresh the grid with that subject's saved rows.
            self._refresh_batch_subject_controls()
            if hasattr(self, "batch_subject_combo"):
                for i in range(1, self.batch_subject_combo.count()):
                    cfg_i = self.batch_subject_combo.itemData(i)
                    if isinstance(cfg_i, dict) and self._subject_key_from_cfg(cfg_i) == subject_key:
                        self.batch_subject_combo.setCurrentIndex(i)
                        self._on_batch_subject_changed(i)
                        break

            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        btn_load_selected = QPushButton("Load Selected Scan Result")
        btn_load_selected.clicked.connect(self._load_selected_result_for_correction)
        btn_apply_correction = QPushButton("Apply Manual Correction")
        btn_apply_correction.clicked.connect(self.apply_manual_correction)

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        btn_load_selected = QPushButton("Load Selected Scan Result")
        btn_load_selected.clicked.connect(self._load_selected_result_for_correction)
        btn_apply_correction = QPushButton("Apply Manual Correction")
        btn_apply_correction.clicked.connect(self.apply_manual_correction)

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            self._refresh_batch_subject_controls()
            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        btn_load_selected = QPushButton("Load Selected Scan Result")
        btn_load_selected.clicked.connect(self._load_selected_result_for_correction)
        btn_apply_correction = QPushButton("Apply Manual Correction")
        btn_apply_correction.clicked.connect(self.apply_manual_correction)

        payload = dict(self.batch_editor_return_payload)
        session_id = self.batch_editor_return_session_id
        self.batch_editor_return_payload = None
        self.batch_editor_return_session_id = None

        if not session_id:
            self.stack.setCurrentIndex(0)
            return
        p = self._session_path_from_id(session_id)
        if not p.exists():
            self.stack.setCurrentIndex(0)
            return
        try:
            session = ExamSession.load_json(p)
        except Exception:
            self.stack.setCurrentIndex(0)
            return
        cfg = session.config or {}
        payload["subject_configs"] = cfg.get("subject_configs", payload.get("subject_configs", []))
        payload["scan_root"] = cfg.get("scan_root", payload.get("scan_root", ""))
        payload["student_list_path"] = cfg.get("student_list_path", payload.get("student_list_path", ""))
        payload["students"] = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "birth_date": str((s.extra or {}).get("birth_date", "") or ""),
                "class_name": str((s.extra or {}).get("class_name", "") or ""),
                "exam_room": str((s.extra or {}).get("exam_room", "") or ""),
            }
            for s in (session.students or [])
        ]
        payload["scan_mode"] = cfg.get("scan_mode", payload.get("scan_mode", "Ảnh trong thư mục gốc"))
        payload["paper_part_count"] = cfg.get("paper_part_count", payload.get("paper_part_count", 3))
        self._open_embedded_exam_editor(session_id, session, payload)

    def _show_batch_scan_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(True)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(False)

    def _show_scoring_panel(self) -> None:
        if hasattr(self, "scan_lr_split"):
            self.scan_lr_split.setVisible(False)
        if hasattr(self, "scoring_panel"):
            self.scoring_panel.setVisible(True)

    def _back_to_batch_scan(self) -> None:
        if not self._confirm_before_switching_work("Batch Scan"):
            return
        self._show_batch_scan_panel()

    def _subject_configs_for_scoring(self) -> list[dict]:
        return self._effective_subject_configs_for_batch()

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        for cfg in self._subject_configs_for_scoring():
            if self._subject_key_from_cfg(cfg) == key_norm:
                return cfg
        return None

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        key = self._subject_key_from_cfg(cfg)
        if self.scan_results_by_subject.get(key):
            return True
        merged = self._merge_saved_batch_snapshot(cfg)
        if bool(merged.get("batch_saved")):
            return True
        if isinstance(merged.get("batch_saved_rows", []), list) and merged.get("batch_saved_rows"):
            return True
        if isinstance(merged.get("batch_saved_results", []), list) and merged.get("batch_saved_results"):
            return True
        return False

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return []
        merged = self._merge_saved_batch_snapshot(cfg)
        cached = merged.get("batch_saved_results", [])
        if not isinstance(cached, list) or not cached:
            return []
        out: list[OMRResult] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        return out

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
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
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _resolve_preferred_scoring_subject(self) -> str:
        if self.stack.currentIndex() == 1:
            cfg = self._selected_batch_subject_config()
            if cfg:
                return self._subject_key_from_cfg(cfg)
        cfgs = self._subject_configs_for_scoring()
        if cfgs:
            return self._subject_key_from_cfg(cfgs[0])
        if self.session and self.session.subjects:
            return str(self.session.subjects[0])
        return "General"

    def _eligible_scoring_subject_keys(self) -> list[str]:
        out: list[str] = []
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if self._is_subject_marked_batched(cfg):
                out.append(key)
        return out

    def _populate_scoring_subjects(self, preferred_key: str = "") -> None:
        if not hasattr(self, "scoring_subject_combo"):
            return
        self.scoring_subject_combo.blockSignals(True)
        self.scoring_subject_combo.clear()
        eligible = set(self._eligible_scoring_subject_keys())
        for cfg in self._subject_configs_for_scoring():
            key = self._subject_key_from_cfg(cfg)
            if eligible and key not in eligible:
                continue
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.scoring_subject_combo.addItem(label, key)
        if self.scoring_subject_combo.count() == 0:
            fallback = self._resolve_preferred_scoring_subject()
            self.scoring_subject_combo.addItem(fallback, fallback)
        pick = preferred_key or self._resolve_preferred_scoring_subject()
        for i in range(self.scoring_subject_combo.count()):
            if str(self.scoring_subject_combo.itemData(i)) == pick:
                self.scoring_subject_combo.setCurrentIndex(i)
                break
        self.scoring_subject_combo.blockSignals(False)

    def _open_scoring_view(self) -> None:
        if not self.session:
            QMessageBox.warning(self, "Tính điểm", "Chưa có kỳ thi hiện tại. Vui lòng mở hoặc tạo kỳ thi trước.")
            return
        if not self._eligible_scoring_subject_keys():
            QMessageBox.warning(self, "Tính điểm", "Cần có ít nhất 1 môn đã Batch Scan trước khi tính điểm.")
            return
        self.stack.setCurrentIndex(1)
        self._populate_scoring_subjects(self._resolve_preferred_scoring_subject())
        self._refresh_scoring_phase_table()
        self._show_scoring_panel()

    def _refresh_scoring_phase_table(self) -> None:
        if not hasattr(self, "scoring_phase_table"):
            return
        self.scoring_phase_table.setRowCount(0)
        for i, p in enumerate(self.scoring_phases[-100:]):
            self.scoring_phase_table.insertRow(i)
            self.scoring_phase_table.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", "-"))))
            self.scoring_phase_table.setItem(i, 1, QTableWidgetItem(str(p.get("subject", "-"))))
            self.scoring_phase_table.setItem(i, 2, QTableWidgetItem(str(p.get("mode", "-"))))
            self.scoring_phase_table.setItem(i, 3, QTableWidgetItem(str(p.get("count", 0))))
            self.scoring_phase_table.setItem(i, 4, QTableWidgetItem(str(p.get("note", ""))))

    def _run_scoring_from_panel(self) -> None:
        subject_key = str(self.scoring_subject_combo.currentData() or "").strip() if hasattr(self, "scoring_subject_combo") else ""
        mode = self.scoring_mode_combo.currentText() if hasattr(self, "scoring_mode_combo") else "Tính lại toàn bộ"
        note = self.scoring_phase_note.text().strip() if hasattr(self, "scoring_phase_note") else ""
        self.calculate_scores(subject_key=subject_key or self._resolve_preferred_scoring_subject(), mode=mode, note=note)



    def _save_batch_for_selected_subject(self) -> None:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return

        session_path = self.current_session_path
        if not session_path and self.batch_editor_return_session_id:
            p = self._session_path_from_id(self.batch_editor_return_session_id)
            if p.exists():
                session_path = p
        if not session_path:
            QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy kỳ thi để lưu trạng thái Batch theo môn.")
            return

        try:
            self._refresh_all_statuses()
            ses = ExamSession.load_json(session_path)
            cfg = ses.config or {}
            subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
            subject_key = self._subject_key_from_cfg(subject_cfg)
            current_results = self._current_scan_results_snapshot()
            self.scan_results = list(current_results)
            for result in current_results:
                result.answer_string = self._build_answer_string_for_result(result, subject_key)
            saved_results = [self._serialize_omr_result(x) for x in current_results]
            timestamp = datetime.now().isoformat(timespec="seconds")
            updated = False
            for item in subject_cfgs:
                if not isinstance(item, dict):
                    continue
                same_name_block = str(item.get("name", "")).strip() == str(subject_cfg.get("name", "")).strip() and str(item.get("block", "")).strip() == str(subject_cfg.get("block", "")).strip()
                same_key = str(self._subject_key_from_cfg(item)).strip() == str(subject_key).strip()
                if same_name_block or same_key:
                    item["batch_saved"] = True
                    item["batch_saved_at"] = timestamp
                    item["batch_result_count"] = row_count
                    item["batch_saved_rows"] = [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ]
                    item["batch_saved_preview"] = [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ]
                    item["batch_saved_results"] = saved_results
                    updated = True
                    break
            if not updated:
                QMessageBox.warning(self, "Lưu Batch", "Không tìm thấy môn tương ứng trong kỳ thi để cập nhật.")
                return
            ses.config = {**cfg, "subject_configs": subject_cfgs}
            ses.save_json(session_path)

            # Write sidecar cache to ensure grids can be restored even if subject config is trimmed elsewhere.
            try:
                cache_path = session_path.with_suffix(".batch_cache.json")
                cache_data = {}
                if cache_path.exists():
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache_data = raw
                cache_key = f"{str(subject_cfg.get('name','')).strip().lower()}::{str(subject_cfg.get('block','')).strip().lower()}::{str(subject_cfg.get('answer_key_key','')).strip().lower()}"
                cache_data[cache_key] = {
                    "batch_saved": True,
                    "batch_saved_at": timestamp,
                    "batch_result_count": row_count,
                    "batch_saved_rows": [
                        {
                            "student_id": self.scan_list.item(r, 0).text() if self.scan_list.item(r, 0) else "-",
                            "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                            "exam_code": str(self.scan_list.item(r, 0).data(Qt.UserRole + 1) if self.scan_list.item(r, 0) else ""),
                            "recognized_short": str(self.scan_list.item(r, 0).data(Qt.UserRole + 2) if self.scan_list.item(r, 0) else ""),
                            "image_path": str(self.scan_list.item(r, 0).data(Qt.UserRole) if self.scan_list.item(r, 0) else ""),
                            "forced_status": str(self.scan_forced_status_by_index.get(r, "") or ""),
                        }
                        for r in range(self.scan_list.rowCount())
                    ],
                    "batch_saved_preview": [
                        {
                            "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                            "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                        }
                        for r in range(self.scan_result_preview.rowCount())
                    ],
                    "batch_saved_results": saved_results,
                }
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if self.session:
                self.session.config = {**(self.session.config or {}), "subject_configs": subject_cfgs}
            self.scan_results_by_subject[subject_key] = list(current_results)
            if isinstance(self.batch_editor_return_payload, dict):
                self.batch_editor_return_payload["subject_configs"] = subject_cfgs
            # Keep selected subject in combo and refresh the grid with that subject's saved rows.
            current_row = self.scan_list.currentRow() if hasattr(self, "scan_list") else -1
            current_image = ""
            if hasattr(self, "scan_list") and 0 <= current_row < self.scan_list.rowCount():
                it = self.scan_list.item(current_row, 0)
                current_image = str(it.data(Qt.UserRole) if it else "")

            self._refresh_batch_subject_controls()
            if hasattr(self, "batch_subject_combo"):
                for i in range(1, self.batch_subject_combo.count()):
                    cfg_i = self.batch_subject_combo.itemData(i)
                    if isinstance(cfg_i, dict) and self._subject_key_from_cfg(cfg_i) == subject_key:
                        self.batch_subject_combo.setCurrentIndex(i)
                        self._on_batch_subject_changed(i)
                        break

            # Try to keep the edited record position/selection after save.
            if hasattr(self, "scan_list") and self.scan_list.rowCount() > 0:
                target_row = -1
                if current_image:
                    for r in range(self.scan_list.rowCount()):
                        it = self.scan_list.item(r, 0)
                        if str(it.data(Qt.UserRole) if it else "") == current_image:
                            target_row = r
                            break
                if target_row < 0 and current_row >= 0:
                    target_row = min(current_row, self.scan_list.rowCount() - 1)
                if target_row >= 0:
                    self.scan_list.selectRow(target_row)
                    self._on_scan_selected()

            self.btn_save_batch_subject.setEnabled(False)
            QMessageBox.information(self, "Lưu Batch", "Đã lưu trạng thái Batch Scan cho môn đã chọn.")
        except Exception as exc:
            QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu trạng thái Batch\n{exc}")


    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
        self.exam_code_correction_combo = QComboBox()
        self.exam_code_correction_combo.currentIndexChanged.connect(self._handle_exam_code_correction_changed)
        self.student_correction_combo = QComboBox()
        self.student_correction_combo.setEditable(True)
        self.student_correction_combo.setInsertPolicy(QComboBox.NoInsert)
        self.student_correction_combo.currentIndexChanged.connect(self._handle_student_correction_changed)
        self.answer_editor_scroll = QScrollArea()
        self.answer_editor_scroll.setWidgetResizable(True)
        self.answer_editor_container = QWidget()
        self.answer_editor_layout = QVBoxLayout(self.answer_editor_container)
        self.answer_editor_scroll.setWidget(self.answer_editor_container)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.error_list = QListWidget()
        self.preview_label = QLabel("Image preview / bubble overlay placeholder")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.manual_edit = QTextEdit()
        self.manual_edit.setPlaceholderText("Manual corrections JSON (e.g. {'student_id': '1001', 'answers': {'1':'A'}})")
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText("Recognized result for selected scan")

        btn_load_selected = QPushButton("Load Selected Scan Result")
        btn_load_selected.clicked.connect(self._load_selected_result_for_correction)
        btn_apply_correction = QPushButton("Apply Manual Correction")
        btn_apply_correction.clicked.connect(self.apply_manual_correction)

        left_layout.addWidget(QLabel("Detected Errors"))
        left_layout.addWidget(self.error_list)
        left_layout.addWidget(btn_load_selected)
        left_layout.addWidget(btn_apply_correction)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.preview_label)
        correction_form = QFormLayout()
        correction_form.addRow("Mã đề (DB)", self.exam_code_correction_combo)
        correction_form.addRow("Học sinh (DB)", self.student_correction_combo)
        right_layout.addLayout(correction_form)
        right_layout.addWidget(QLabel("Chỉnh sửa đáp án trực quan"))
        right_layout.addWidget(self.answer_editor_scroll, 4)
        right_layout.addWidget(self.result_preview)
        right_layout.addWidget(QLabel("Manual Edit"))
        right_layout.addWidget(self.manual_edit)

        splitter.addWidget(left)
        splitter.addWidget(right)

        layout = QHBoxLayout(w)
        layout.addWidget(splitter)
        return w

    def _correction_selected_result(self) -> tuple[int, OMRResult] | tuple[None, None]:
        idx = self.scan_list.currentRow() if hasattr(self, "scan_list") else -1
        if idx < 0 or idx >= len(self.scan_results):
            return None, None
        return idx, self.scan_results[idx]

    def _load_exam_code_correction_options(self, subject_key: str, current_code: str) -> None:
        self.exam_code_correction_combo.blockSignals(True)
        self.exam_code_correction_combo.clear()
        codes = set(self.database.fetch_answer_keys_for_subject(subject_key).keys())
        codes.update(str(x).strip() for x in (self.imported_exam_codes or []) if str(x).strip())
        if current_code:
            codes.add(str(current_code).strip())
        for code in sorted(codes):
            self.exam_code_correction_combo.addItem(code, code)
        if self.exam_code_correction_combo.count() == 0:
            self.exam_code_correction_combo.addItem(current_code or "-", current_code or "")
        match_index = max(0, self.exam_code_correction_combo.findData(current_code))
        self.exam_code_correction_combo.setCurrentIndex(match_index)
        self.exam_code_correction_combo.blockSignals(False)

    def _load_student_correction_options(self, current_student_id: str) -> None:
        self.student_correction_combo.blockSignals(True)
        self.student_correction_combo.clear()
        students: list[tuple[str, str, str]] = []
        if self.current_session_id:
            payload = self.database.fetch_exam_session(self.current_session_id) or {}
            session_students = payload.get("students", []) if isinstance(payload.get("students", []), list) else []
            for item in session_students:
                if not isinstance(item, dict):
                    continue
                extra = item.get("extra", {}) if isinstance(item.get("extra", {}), dict) else {}
                sid = str(item.get("student_id", "") or "").strip()
                if not sid:
                    continue
                students.append((sid, str(item.get("name", "") or "").strip(), str(extra.get("class_name", "") or "").strip()))
        labels: list[str] = []
        seen: set[str] = set()
        for sid, name, class_name in students:
            if sid in seen:
                continue
            seen.add(sid)
            label = f"[{sid}] - {name or '-'} - {class_name or '-'}"
            labels.append(label)
            self.student_correction_combo.addItem(label, sid)
        if current_student_id and current_student_id not in seen:
            label = f"[{current_student_id}] - - -"
            labels.append(label)
            self.student_correction_combo.addItem(label, current_student_id)
        idx = self.student_correction_combo.findData(current_student_id)
        self.student_correction_combo.setCurrentIndex(max(0, idx))
        completer = QCompleter(labels, self.student_correction_combo)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.student_correction_combo.setCompleter(completer)
        self.student_correction_combo.blockSignals(False)

    def _build_visual_answer_editor(self, result: OMRResult) -> None:
        while self.answer_editor_layout.count():
            item = self.answer_editor_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        expected = self._expected_questions_by_section(result)

        mcq_box = QGroupBox("MCQ")
        mcq_layout = QVBoxLayout(mcq_box)
        for q_no in expected.get("MCQ", []):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Câu {q_no}"))
            group = QButtonGroup(mcq_box)
            current_value = str((result.mcq_answers or {}).get(q_no, "") or "")[:1]
            for choice in ["A", "B", "C", "D", "E"]:
                radio = QRadioButton(choice)
                if current_value == choice:
                    radio.setChecked(True)
                radio.toggled.connect(lambda checked, q=q_no, v=choice: checked and self._handle_mcq_visual_change(q, v))
                group.addButton(radio)
                row.addWidget(radio)
            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda _=False, q=q_no: self._handle_mcq_visual_change(q, ""))
            row.addWidget(clear_btn)
            row.addStretch()
            mcq_layout.addLayout(row)
        self.answer_editor_layout.addWidget(mcq_box)

        tf_box = QGroupBox("True/False")
        tf_layout = QVBoxLayout(tf_box)
        for q_no in expected.get("TF", []):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Câu {q_no}"))
            flags = (result.true_false_answers or {}).get(q_no, {}) or {}
            for key in ["a", "b", "c", "d"]:
                cb = QCheckBox(key.upper())
                cb.setChecked(bool(flags.get(key)))
                cb.toggled.connect(lambda checked, q=q_no, k=key: self._handle_tf_visual_change(q, k, checked))
                row.addWidget(cb)
            row.addStretch()
            tf_layout.addLayout(row)
        self.answer_editor_layout.addWidget(tf_box)

        num_box = QGroupBox("Numeric")
        num_layout = QVBoxLayout(num_box)
        for q_no in expected.get("NUMERIC", []):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Câu {q_no}"))
            edit = QLineEdit(str((result.numeric_answers or {}).get(q_no, "") or ""))
            edit.editingFinished.connect(lambda q=q_no, w=edit: self._handle_numeric_visual_change(q, w.text()))
            row.addWidget(edit)
            num_layout.addLayout(row)
        self.answer_editor_layout.addWidget(num_box)
        self.answer_editor_layout.addStretch()

    def _ensure_correction_state(self) -> None:
        if not hasattr(self, "correction_ui_loading"):
            self.correction_ui_loading = False
        if not hasattr(self, "correction_pending_payload") or not isinstance(getattr(self, "correction_pending_payload", None), dict):
            self.correction_pending_payload = {}
        if not hasattr(self, "correction_save_timer") or not isinstance(getattr(self, "correction_save_timer", None), QTimer):
            self.correction_save_timer = QTimer(self)
            self.correction_save_timer.setSingleShot(True)
            self.correction_save_timer.timeout.connect(self._flush_pending_correction_updates)

    def _schedule_correction_update(self, field_name: str, old_value: object, new_value: object, apply_fn) -> None:
        self._ensure_correction_state()
        if self.correction_ui_loading or old_value == new_value:
            return
        apply_fn()
        self.correction_pending_payload[field_name] = {
            "old": old_value,
            "new": new_value,
        }
        self.correction_save_timer.start(150)

    def _flush_pending_correction_updates(self) -> None:
        self._ensure_correction_state()
        idx, result = self._correction_selected_result()
        if idx is None or result is None or not self.correction_pending_payload:
            return
        changes = [f"{field}: '{payload['old']}' -> '{payload['new']}'" for field, payload in self.correction_pending_payload.items()]
        self._refresh_student_profile_for_result(result, idx)
        scoped = self._scoped_result_copy(result)
        self.scan_blank_summary[idx] = self._compute_blank_questions(scoped)
        self.scan_list.setItem(idx, 3, QTableWidgetItem(self._build_recognition_content_text(result, self.scan_blank_summary[idx])))
        sid_item = self.scan_list.item(idx, 0)
        if sid_item:
            sid_item.setText((result.student_id or "").strip() or "-")
            sid_item.setData(Qt.UserRole + 1, result.exam_code or "")
            sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(result))
        self._record_adjustment(idx, changes, "visual_correction")
        self._persist_single_scan_result_to_db(result, note="visual_correction")
        image_key = str(getattr(result, "image_path", "") or idx)
        for field_name, payload in self.correction_pending_payload.items():
            self.database.log_change("scan_results", image_key, field_name, payload["old"], payload["new"], "visual_correction")
        self.correction_pending_payload = {}
        self._refresh_all_statuses()
        self._update_scan_preview(idx)
        self.correction_ui_loading = True
        self._sync_correction_detail_panel(result, rebuild_editor=False)
        self.correction_ui_loading = False

    def _handle_exam_code_correction_changed(self, _index: int) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        new_code = str(self.exam_code_correction_combo.currentData() or "").strip()
        old_code = str(result.exam_code or "").strip()
        self._schedule_correction_update("exam_code", old_code, new_code, lambda: setattr(result, "exam_code", new_code))

    def _handle_student_correction_changed(self, _index: int) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        new_sid = str(self.student_correction_combo.currentData() or self.student_correction_combo.currentText() or "").strip()
        old_sid = str(result.student_id or "").strip()
        self._schedule_correction_update("student_id", old_sid, new_sid, lambda: setattr(result, "student_id", new_sid))

    def _handle_mcq_visual_change(self, question_no: int, answer_value: str) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        old_value = str((result.mcq_answers or {}).get(question_no, "") or "")[:1]

        def _apply() -> None:
            current = dict(result.mcq_answers or {})
            if answer_value:
                current[int(question_no)] = str(answer_value)[:1]
            else:
                current.pop(int(question_no), None)
            result.mcq_answers = current

        self._schedule_correction_update(f"mcq_answers[{question_no}]", old_value, str(answer_value)[:1], _apply)

    def _handle_tf_visual_change(self, question_no: int, key: str, checked: bool) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        old_flags = dict((result.true_false_answers or {}).get(question_no, {}) or {})
        new_flags = dict(old_flags)
        new_flags[key] = bool(checked)

        def _apply() -> None:
            current = dict(result.true_false_answers or {})
            current[int(question_no)] = dict(new_flags)
            result.true_false_answers = current

        self._schedule_correction_update(f"true_false_answers[{question_no}].{key}", old_flags.get(key), bool(checked), _apply)

    def _handle_numeric_visual_change(self, question_no: int, text: str) -> None:
        idx, result = self._correction_selected_result()
        if idx is None or result is None:
            return
        old_value = str((result.numeric_answers or {}).get(question_no, "") or "")
        new_value = str(text or "").strip()

        def _apply() -> None:
            current = dict(result.numeric_answers or {})
            if new_value:
                current[int(question_no)] = new_value
            else:
                current.pop(int(question_no), None)
            result.numeric_answers = current

        self._schedule_correction_update(f"numeric_answers[{question_no}]", old_value, new_value, _apply)


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
        self.stack.setCurrentIndex(3)
        return True

    def _close_template_module(self) -> bool:
        if self.template_editor_embedded:
            return self._close_embedded_template_editor()
        self.stack.setCurrentIndex(0)
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
            students=[
                Student(
                    student_id=str(x.get("student_id", "") or "").strip(),
                    name=str(x.get("name", "") or "").strip(),
                    extra={
                        "birth_date": str(x.get("birth_date", "") or ""),
                        "class_name": str(x.get("class_name", "") or ""),
                        "exam_room": str(x.get("exam_room", "") or ""),
                    },
                )
                for x in (payload.get("students", []) if isinstance(payload.get("students", []), list) else [])
                if str(x.get("student_id", "") or "").strip() and str(x.get("name", "") or "").strip()
            ],
            config={
                "scan_mode": payload.get("scan_mode", "Ảnh trong thư mục gốc"),
                "scan_root": payload.get("scan_root", ""),
                "student_list_path": payload.get("student_list_path", ""),
                "paper_part_count": payload.get("paper_part_count", 3),
                "subject_configs": subject_cfgs,
                "subject_catalog": self.subject_catalog,
                "block_catalog": self.block_catalog,
                "scoring_phases": [],
                "scoring_results": {},
            },
        )
        self.scoring_phases = []
        self.scoring_results_by_subject = {}
        self.batch_working_state_by_subject = {}
        if common_template and Path(common_template).exists():
            try:
                self.template = Template.load_json(common_template)
            except Exception:
                self.template = None
        self.current_session_path = None
        self.current_session_id = None
        self.session_dirty = True
        self._refresh_session_info()
        self._refresh_batch_subject_controls()

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

    def open_template_editor(self) -> None:
        self._refresh_template_library()
        self.template_editor_mode = "library"
        self.stack.setCurrentIndex(3)

    def _subject_configs_in_session(self) -> list[dict]:
        if not self.session:
            return []
        cfg = self.session.config or {}
        raw = cfg.get("subject_configs", [])
        return raw if isinstance(raw, list) else []

    def _student_profile_by_id(self, student_id: str) -> dict:
        sid = str(student_id or "").strip()
        if not sid or not self.session:
            return {}
        normalized_sid = self._normalized_student_id_for_match(sid)
        for s in (self.session.students or []):
            candidate = str(getattr(s, "student_id", "") or "").strip()
            if candidate == sid:
                return {
                    "name": str(getattr(s, "name", "") or ""),
                    "birth_date": str((getattr(s, "extra", {}) or {}).get("birth_date", "") or ""),
                    "class_name": str((getattr(s, "extra", {}) or {}).get("class_name", "") or ""),
                    "exam_room": str((getattr(s, "extra", {}) or {}).get("exam_room", "") or ""),
                }
            if normalized_sid and self._normalized_student_id_for_match(candidate) == normalized_sid:
                return {
                    "name": str(getattr(s, "name", "") or ""),
                    "birth_date": str((getattr(s, "extra", {}) or {}).get("birth_date", "") or ""),
                    "class_name": str((getattr(s, "extra", {}) or {}).get("class_name", "") or ""),
                    "exam_room": str((getattr(s, "extra", {}) or {}).get("exam_room", "") or ""),
                }
        return {}

    def _refresh_student_profile_for_result(self, result, row_idx: int | None = None) -> None:
        sid = str(getattr(result, "student_id", "") or "").strip()
        profile = self._student_profile_by_id(sid)
        setattr(result, "full_name", str(profile.get("name", "") or ""))
        setattr(result, "birth_date", str(profile.get("birth_date", "") or ""))
        setattr(result, "class_name", str(profile.get("class_name", "") or ""))
        setattr(result, "exam_room", str(profile.get("exam_room", "") or ""))
        if row_idx is not None and 0 <= row_idx < self.scan_list.rowCount():
            self.scan_list.setItem(row_idx, 1, QTableWidgetItem(str(getattr(result, "full_name", "") or "-")))
            self.scan_list.setItem(row_idx, 2, QTableWidgetItem(str(getattr(result, "birth_date", "") or "-")))

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
    def _normalize_template_path(path_text: str) -> str:
        t = str(path_text or "").strip()
        if not t:
            return ""
        if t.lower() in {"[dùng mẫu chung]", "[dung mau chung]", "none", "null", "-"}:
            return ""
        return t

    @staticmethod
    def _subject_imported_answer_keys_for_main(subject_cfg: dict) -> dict:
        if not isinstance(subject_cfg, dict):
            return {}
        raw = subject_cfg.get("imported_answer_keys", {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _effective_subject_configs_for_batch(self) -> list[dict]:
        cfgs = self._subject_configs_in_session()
        common_template = self._normalize_template_path(str(self.session.template_path if self.session else ""))
        if cfgs:
            # Default to exam common template, but keep subject-specific template if explicitly configured.
            out_cfgs: list[dict] = []
            for cfg in cfgs:
                if not isinstance(cfg, dict):
                    continue
                item = dict(cfg)
                subject_template = self._normalize_template_path(str(item.get("template_path", "") or ""))
                item["template_path"] = subject_template or common_template
                out_cfgs.append(item)
            return out_cfgs
        # Fallback for older sessions without subject_configs.
        if not self.session:
            return []
        scan_root = str((self.session.config or {}).get("scan_root", "") or "")
        out: list[dict] = []
        for raw in (self.session.subjects or ["General"]):
            subject = str(raw)
            name, block = (subject.split("_", 1) + [""])[:2] if "_" in subject else (subject, "")
            out.append({
                "name": name,
                "block": block,
                "template_path": common_template or str(self.session.template_path or ""),
                "scan_folder": scan_root,
                "answer_key_key": subject,
                "imported_answer_keys": {},
            })
        return out

    def _resolve_subject_config_for_batch(self) -> dict | None:
        cfg = self._selected_batch_subject_config()
        if cfg:
            return cfg
        cfgs = self._effective_subject_configs_for_batch()
        if not cfgs:
            return None
        # Do not show extra chooser dialog: Batch panel already has subject combo.
        # Fallback to first configured subject when none is currently selected.
        return cfgs[0]

    def _refresh_batch_subject_controls(self) -> None:
        if not hasattr(self, "batch_subject_combo"):
            return
        self.batch_subject_combo.blockSignals(True)
        self.batch_subject_combo.clear()
        self.batch_subject_combo.addItem("[Chọn môn]")
        for cfg in self._effective_subject_configs_for_batch():
            label = f"{cfg.get('name', '-')}-Khối {cfg.get('block', '-')}"
            self.batch_subject_combo.addItem(label, cfg)
        self.batch_subject_combo.blockSignals(False)
        self._on_batch_subject_changed(self.batch_subject_combo.currentIndex())

    def _selected_batch_subject_config(self) -> dict | None:
        if not hasattr(self, "batch_subject_combo"):
            return None
        idx = self.batch_subject_combo.currentIndex()
        if idx <= 0:
            return None
        cfg = self.batch_subject_combo.itemData(idx)
        return cfg if isinstance(cfg, dict) else None

    def _batch_context_session_path(self) -> Path | None:
        if self.current_session_id:
            return self._session_path_from_id(self.current_session_id)
        if self.current_session_path:
            return self.current_session_path
        if self.batch_editor_return_session_id:
            return self._session_path_from_id(self.batch_editor_return_session_id)
        return None

    def _merge_saved_batch_snapshot(self, cfg: dict) -> dict:
        merged = dict(cfg)
        if merged.get("batch_saved_rows") or merged.get("batch_saved_preview"):
            return merged

        session_path = self._batch_context_session_path()
        if not session_path:
            return merged
        try:
            ses = ExamSession.load_json(session_path)
        except Exception:
            return merged

        raw_cfgs = (ses.config or {}).get("subject_configs", [])
        if not isinstance(raw_cfgs, list):
            return merged

        def _norm(v: str) -> str:
            return str(v or "").strip().lower()

        name = _norm(merged.get("name", ""))
        block = _norm(merged.get("block", ""))
        key = _norm(merged.get("answer_key_key", ""))
        found: dict | None = None
        for item in raw_cfgs:
            if not isinstance(item, dict):
                continue
            same_name_block = _norm(item.get("name", "")) == name and _norm(item.get("block", "")) == block
            same_key = key and _norm(item.get("answer_key_key", "")) == key
            if same_name_block or same_key:
                found = item
                break

        if found:
            for k in ["batch_saved", "batch_saved_at", "batch_result_count", "batch_saved_rows", "batch_saved_preview", "batch_saved_results"]:
                if k in found:
                    merged[k] = found.get(k)

        # Sidecar fallback for large snapshots or older sessions.
        sidecar = session_path.with_suffix(".batch_cache.json")
        if sidecar.exists() and not (merged.get("batch_saved_rows") or merged.get("batch_saved_preview")):
            try:
                raw = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    cache_key = f"{_norm(merged.get('name', ''))}::{_norm(merged.get('block', ''))}::{_norm(merged.get('answer_key_key', ''))}"
                    payload = raw.get(cache_key)
                    if isinstance(payload, dict):
                        merged["batch_saved_rows"] = payload.get("batch_saved_rows", [])
                        merged["batch_saved_preview"] = payload.get("batch_saved_preview", [])
                        merged["batch_saved"] = bool(payload.get("batch_saved", True))
                        merged["batch_saved_at"] = payload.get("batch_saved_at", merged.get("batch_saved_at", "-"))
                        merged["batch_result_count"] = payload.get("batch_result_count", merged.get("batch_result_count", "-"))
                        merged["batch_saved_results"] = payload.get("batch_saved_results", merged.get("batch_saved_results", []))
            except Exception:
                pass
        return merged

    def _on_batch_subject_changed(self, _index: int) -> None:
        cfg = self._selected_batch_subject_config()
        if cfg:
            cfg = self._merge_saved_batch_snapshot(cfg)
            self.active_batch_subject_key = self._subject_key_from_cfg(cfg)
        else:
            self.active_batch_subject_key = None

        # Refresh recognition grids below when switching subject to avoid stale cross-subject data.
        self.scan_results = []
        self.scan_files = []
        self.scan_blank_questions.clear()
        self.scan_blank_summary.clear()
        self.scan_manual_adjustments.clear()
        self.scan_edit_history.clear()
        self.scan_last_adjustment.clear()
        self.scan_forced_status_by_index.clear()
        if hasattr(self, "scan_list"):
            self.scan_list.setRowCount(0)
        if hasattr(self, "scan_result_preview"):
            self.scan_result_preview.setRowCount(0)
        if hasattr(self, "error_list"):
            self.error_list.clear()
        if hasattr(self, "result_preview"):
            self.result_preview.clear()
        if hasattr(self, "manual_edit"):
            self.manual_edit.clear()
        if hasattr(self, "progress"):
            self.progress.setValue(0)
        if hasattr(self, "scan_image_preview"):
            self.preview_source_pixmap = QPixmap()
            self.scan_image_preview.setPixmap(QPixmap())
            self.scan_image_preview.setText("Chọn bài thi ở danh sách bên trái")
            self.scan_image_preview.clear_markers()
            if hasattr(self, "btn_zoom_reset"):
                self.preview_zoom_factor = 1.0
                self.btn_zoom_reset.setText("100%")
        if hasattr(self, "btn_save_batch_subject"):
            self.btn_save_batch_subject.setEnabled(False)

        if not cfg:
            self.batch_template_value.setText("-")
            self.batch_answer_codes_value.setText("-")
            self.batch_student_id_value.setText("-")
            self.batch_scan_folder_value.setText("-")
            return

        template_path = self._normalize_template_path(str(cfg.get("template_path", "") or "")) or self._normalize_template_path(str(self.session.template_path if self.session else "")) or "-"
        scan_folder = str(cfg.get("scan_folder", "") or ((self.session.config or {}).get("scan_root", "") if self.session else "") or "-")
        codes = ", ".join(sorted((cfg.get("imported_answer_keys") or {}).keys())) or "-"
        self.batch_template_value.setText(template_path)
        self.batch_answer_codes_value.setText(codes)
        self.batch_scan_folder_value.setText(scan_folder)
        tpl_for_view = None
        tp = Path(template_path) if template_path and template_path != "-" else None
        if tp and tp.exists():
            try:
                tpl_for_view = Template.load_json(tp)
            except Exception:
                tpl_for_view = self.template
        else:
            tpl_for_view = self.template
        has_sid = "Có" if (tpl_for_view and any(z.zone_type.value == "STUDENT_ID_BLOCK" for z in tpl_for_view.zones)) else "Không"
        self.batch_student_id_value.setText(has_sid)
        if tpl_for_view:
            self._apply_template_recognition_settings(tpl_for_view)

        saved_results_payload = cfg.get("batch_saved_results", []) if isinstance(cfg.get("batch_saved_results", []), list) else []
        self.scan_results = []
        for payload in saved_results_payload:
            if not isinstance(payload, dict):
                continue
            try:
                self.scan_results.append(self._deserialize_omr_result(payload))
            except Exception:
                continue
        subject_key = self._subject_key_from_cfg(cfg)
        if self.scan_results:
            self.scan_results_by_subject[subject_key] = list(self.scan_results)

        saved_rows = cfg.get("batch_saved_rows", []) if isinstance(cfg.get("batch_saved_rows", []), list) else []
        for row in saved_rows:
            if not isinstance(row, dict):
                continue
            r = self.scan_list.rowCount()
            self.scan_list.insertRow(r)
            sid_item = QTableWidgetItem(str(row.get("student_id", "-")))
            sid_item.setData(Qt.UserRole, str(row.get("image_path", "") or ""))
            sid_item.setData(Qt.UserRole + 1, str(row.get("exam_code", "") or ""))
            sid_item.setData(Qt.UserRole + 2, str(row.get("recognized_short", "") or ""))
            self.scan_list.setItem(r, 0, sid_item)
            self.scan_list.setItem(r, 1, QTableWidgetItem(str(row.get("exam_code", "-") or "-")))
            self.scan_list.setItem(r, 2, QTableWidgetItem(str(row.get("full_name", "-"))))
            self.scan_list.setItem(r, 3, QTableWidgetItem(str(row.get("birth_date", "-"))))
            self.scan_list.setItem(r, 4, QTableWidgetItem(str(row.get("content", "-"))))
            forced_status = str(row.get("forced_status", "") or "")
            if forced_status:
                self.scan_forced_status_by_index[r] = forced_status
            item_status = QTableWidgetItem(str(row.get("status", "-")))
            if item_status.text() != "OK":
                item_status.setForeground(Qt.red)
            self.scan_list.setItem(r, 5, item_status)
            self._set_scan_action_widget(r)

        saved_preview = cfg.get("batch_saved_preview", []) if isinstance(cfg.get("batch_saved_preview", []), list) else []
        for row in saved_preview:
            if not isinstance(row, dict):
                continue
            r = self.scan_result_preview.rowCount()
            self.scan_result_preview.insertRow(r)
            self.scan_result_preview.setItem(r, 0, QTableWidgetItem(str(row.get("label", ""))))
            self.scan_result_preview.setItem(r, 1, QTableWidgetItem(str(row.get("value", ""))))

        if saved_rows:
            self.scan_image_preview.setText("Đã nạp nội dung Batch đã lưu cho môn này")
        elif bool(cfg.get("batch_saved")):
            self.scan_image_preview.setText(
                f"Môn này đã lưu Batch ({cfg.get('batch_saved_at', '-')}) - Số bài: {cfg.get('batch_result_count', '-')}."
            )

    def _cache_working_batch_state(self, subject_key: str) -> None:
        key = str(subject_key or "").strip()
        if not key or not hasattr(self, "scan_list"):
            return

        rows: list[dict] = []
        for r in range(self.scan_list.rowCount()):
            sid_item = self.scan_list.item(r, 0)
            rows.append(
                {
                    "student_id": sid_item.text() if sid_item else "-",
                    "image_path": str(sid_item.data(Qt.UserRole) if sid_item else ""),
                    "exam_code": str(sid_item.data(Qt.UserRole + 1) if sid_item else ""),
                    "recognized_short": str(sid_item.data(Qt.UserRole + 2) if sid_item else ""),
                    "full_name": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                    "birth_date": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                    "content": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                    "status": self.scan_list.item(r, 5).text() if self.scan_list.item(r, 5) else "-",
                }
            )

        preview_rows: list[dict] = []
        if hasattr(self, "scan_result_preview"):
            for r in range(self.scan_result_preview.rowCount()):
                preview_rows.append(
                    {
                        "label": self.scan_result_preview.item(r, 0).text() if self.scan_result_preview.item(r, 0) else "",
                        "value": self.scan_result_preview.item(r, 1).text() if self.scan_result_preview.item(r, 1) else "",
                    }
                )

        self.batch_working_state_by_subject[key] = {
            "scan_results": list(self.scan_results_by_subject.get(key, self.scan_results or [])),
            "rows": rows,
            "preview": preview_rows,
        }

    def _restore_cached_working_batch_state(self, subject_key: str) -> bool:
        key = str(subject_key or "").strip()
        cached = self.batch_working_state_by_subject.get(key)
        if not isinstance(cached, dict):
            return False

        self.scan_results = list(cached.get("scan_results", []))
        self.scan_results_by_subject[key] = list(self.scan_results)

        for row in (cached.get("rows", []) if isinstance(cached.get("rows", []), list) else []):
            if not isinstance(row, dict):
                continue
            r = self.scan_list.rowCount()
            self.scan_list.insertRow(r)
            sid_item = QTableWidgetItem(str(row.get("student_id", "-")))
            sid_item.setData(Qt.UserRole, str(row.get("image_path", "") or ""))
            sid_item.setData(Qt.UserRole + 1, str(row.get("exam_code", "") or ""))
            sid_item.setData(Qt.UserRole + 2, str(row.get("recognized_short", "") or ""))
            self.scan_list.setItem(r, 0, sid_item)
            self.scan_list.setItem(r, 1, QTableWidgetItem(str(row.get("exam_code", "-") or "-")))
            self.scan_list.setItem(r, 2, QTableWidgetItem(str(row.get("full_name", "-"))))
            self.scan_list.setItem(r, 3, QTableWidgetItem(str(row.get("birth_date", "-"))))
            self.scan_list.setItem(r, 4, QTableWidgetItem(str(row.get("content", "-"))))
            st = QTableWidgetItem(str(row.get("status", "-")))
            if st.text() != "OK":
                st.setForeground(Qt.red)
            self.scan_list.setItem(r, 5, st)
            self._set_scan_action_widget(r)

        for row in (cached.get("preview", []) if isinstance(cached.get("preview", []), list) else []):
            if not isinstance(row, dict):
                continue
            r = self.scan_result_preview.rowCount()
            self.scan_result_preview.insertRow(r)
            self.scan_result_preview.setItem(r, 0, QTableWidgetItem(str(row.get("label", ""))))
            self.scan_result_preview.setItem(r, 1, QTableWidgetItem(str(row.get("value", ""))))

        if hasattr(self, "btn_save_batch_subject") and self.scan_list.rowCount() > 0:
            self.btn_save_batch_subject.setEnabled(True)
        return self.scan_list.rowCount() > 0

    @staticmethod
    def _has_valid_identity(result) -> bool:
        sid = str(getattr(result, "student_id", "") or "").strip()
        code = str(getattr(result, "exam_code", "") or "").strip()
        has_id = bool(sid and "?" not in sid)
        has_code = bool(code and "?" not in code)
        return has_id or has_code

    @staticmethod
    def _result_has_meaningful_recognition(result) -> bool:
        has_identity = MainWindow._has_valid_identity(result)
        has_answers = bool((result.mcq_answers or {}) or (result.true_false_answers or {}) or (result.numeric_answers or {}))
        return has_answers or has_identity

    @staticmethod
    def _should_force_image_error_status(result) -> bool:
        issues = list(getattr(result, "issues", []) or [])
        if any(str(getattr(issue, "code", "") or "").upper() == "FILE" for issue in issues):
            return True
        return not MainWindow._result_has_meaningful_recognition(result)

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

    def _try_reprocess_result_rotated_180(self, result):
        image_path = str(getattr(result, "image_path", "") or "").strip()
        if not image_path or not Path(image_path).exists() or not self.template:
            return result, False
        pix = QPixmap(image_path)
        if pix.isNull():
            return result, False
        rotated = pix.transformed(QTransform().rotate(180.0), Qt.SmoothTransformation)
        temp_path = str(Path(image_path).with_name(f".{Path(image_path).stem}_tmp_auto180.png"))
        if not rotated.save(temp_path):
            return result, False
        try:
            alt = self.omr_processor.run_recognition_test(temp_path, self.template)
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
        alt.image_path = image_path
        if self._recognition_quality_score(alt) > self._recognition_quality_score(result):
            return alt, True
        return result, False

    def run_batch_scan(self) -> None:
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

        if hasattr(self, "batch_recognition_mode_combo"):
            mode = str(self.batch_recognition_mode_combo.currentData() or "auto")
            setattr(self.omr_processor, "alignment_profile", mode)

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

        scan_folder = str(scan_folder or "").strip()
        if not scan_folder:
            QMessageBox.warning(self, "Batch Scan", "Chưa chọn thư mục quét.")
            return
        scan_dir = Path(scan_folder)
        if not scan_dir.exists() or not scan_dir.is_dir():
            QMessageBox.warning(self, "Batch Scan", f"Không tìm thấy thư mục quét\n{scan_folder}")
            return

        scan_mode = str((subject_cfg or {}).get("scan_mode", "") or (self.session.config or {}).get("scan_mode", "") if self.session else "")
        image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        if "thư mục con" in scan_mode.lower() or "sub" in scan_mode.lower():
            file_paths = [str(p) for p in sorted(scan_dir.rglob("*")) if p.is_file() and p.suffix.lower() in image_exts]
        else:
            file_paths = [str(p) for p in sorted(scan_dir.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]

        if not file_paths:
            QMessageBox.warning(self, "Batch Scan", "Không tìm thấy ảnh bài thi trong thư mục quét.")
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

        def on_progress(current: int, total: int, image_path: str):
            self.progress.setMaximum(total)
            self.progress.setValue(current)
            QApplication.processEvents()

        self.scan_results = self.omr_processor.process_batch(file_paths, self.template, on_progress)
        subject_key_for_results = self._subject_key_from_cfg(subject_cfg) if subject_cfg else self._resolve_preferred_scoring_subject()
        self.scan_list.setRowCount(0)
        duplicate_ids: dict[str, int] = {}
        for res in self.scan_results:
            sid = (res.student_id or "").strip()
            if sid:
                duplicate_ids[sid] = duplicate_ids.get(sid, 0) + 1

        for idx, result in enumerate(self.scan_results):
            forced_status = ""
            original_meaningful = self._result_has_meaningful_recognition(result)
            original_identity = self._has_valid_identity(result)

            # Keep batch behavior aligned with Template Editor by default (no auto-rotation retry).
            need_retry_180 = self._allow_batch_auto_rotate_retry() and ((not original_identity) or (not original_meaningful))
            if need_retry_180:
                retried, improved = self._try_reprocess_result_rotated_180(result)
                # Accept 180° retry only when quality is strictly improved, otherwise keep original orientation.
                if improved:
                    result = retried
                    self.scan_results[idx] = result
                    self.preview_rotation_by_index[idx] = (int(self.preview_rotation_by_index.get(idx, 0) or 0) + 180) % 360

            if self._should_force_image_error_status(result):
                # Keep raw recognition data for consistency with single-image re-recognition,
                # but only mark image-file error when recognition is truly unusable/file-loading failed.
                forced_status = "Lỗi file ảnh"

            if forced_status:
                self.scan_forced_status_by_index[idx] = forced_status

            rec_errors = list(getattr(result, "recognition_errors", [])) or list(getattr(result, "errors", []))
            sid = (result.student_id or "").strip()
            self._refresh_student_profile_for_result(result)
            full_name = str(getattr(result, "full_name", "") or "-")
            birth_date = str(getattr(result, "birth_date", "") or "-")
            scoped = self._scoped_result_copy(result)
            blank_map = self._compute_blank_questions(scoped)
            blank_questions = blank_map.get("MCQ", [])
            self.scan_blank_questions[idx] = blank_questions
            self.scan_blank_summary[idx] = blank_map
            exam_code_text = (result.exam_code or "").strip()
            forced_status = self.scan_forced_status_by_index.get(idx, "")
            if forced_status:
                status = forced_status
            else:
                status_parts = self._status_parts_for_row(sid, exam_code_text, duplicate_ids.get(sid, 0))
                status = ", ".join(status_parts) if status_parts else "OK"
            content_text = self._build_recognition_content_text(result, blank_map)

            self.scan_list.insertRow(idx)
            sid_item = QTableWidgetItem(sid or "-")
            sid_item.setData(Qt.UserRole, str(result.image_path))
            sid_item.setData(Qt.UserRole + 1, exam_code_text)
            sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(result))
            self.scan_list.setItem(idx, 0, sid_item)
            self.scan_list.setItem(idx, 1, QTableWidgetItem(exam_code_text or "-"))
            self.scan_list.setItem(idx, 2, QTableWidgetItem(full_name))
            self.scan_list.setItem(idx, 3, QTableWidgetItem(birth_date))
            self.scan_list.setItem(idx, 4, QTableWidgetItem(content_text))
            status_item = QTableWidgetItem(status)
            if status != "OK":
                status_item.setForeground(Qt.red)
            self.scan_list.setItem(idx, 5, status_item)
            self._set_scan_action_widget(idx)
            for issue in result.issues:
                self.error_list.addItem(f"{Path(result.image_path).name}: {issue.code} - {issue.message}")
            for err in rec_errors:
                self.error_list.addItem(f"{Path(result.image_path).name}: RECOGNITION - {err}")

            # Keep UI responsive while filling table after recognition reaches 100%.
            if idx % 10 == 0:
                QApplication.processEvents()

        self.scan_results_by_subject[subject_key_for_results] = list(self.scan_results)

        self.btn_save_batch_subject.setEnabled(True)
        self._apply_scan_filter()

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _scoped_result_copy(self, result):
        scoped = copy.deepcopy(result)
        self._trim_result_answers_to_expected_scope(scoped)
        return scoped

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

        has_exam_code_zone = any(z.zone_type.value == "EXAM_CODE_BLOCK" for z in self.template.zones)
        has_student_id_zone = any(z.zone_type.value == "STUDENT_ID_BLOCK" for z in self.template.zones)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

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

        def on_progress(current: int, total: int, image_path: str):
            self.progress.setMaximum(total)
            self.progress.setValue(current)
            QApplication.processEvents()

        self.scan_results = self.omr_processor.process_batch(file_paths, self.template, on_progress)
        subject_key_for_results = self._subject_key_from_cfg(subject_cfg) if subject_cfg else self._resolve_preferred_scoring_subject()
        self.scan_list.setRowCount(0)
        duplicate_ids: dict[str, int] = {}
        for res in self.scan_results:
            sid = (res.student_id or "").strip()
            if sid:
                duplicate_ids[sid] = duplicate_ids.get(sid, 0) + 1

        for idx, result in enumerate(self.scan_results):
            rec_errors = list(getattr(result, "recognition_errors", [])) or list(getattr(result, "errors", []))
            total_errors = len(rec_errors) + len(result.issues)
            sid = (result.student_id or "").strip()
            self._refresh_student_profile_for_result(result)
            full_name = str(getattr(result, "full_name", "") or "-")
            birth_date = str(getattr(result, "birth_date", "") or "-")
            self._trim_result_answers_to_expected_scope(result)
            blank_map = self._compute_blank_questions(result)
            blank_questions = blank_map.get("MCQ", [])
            self.scan_blank_questions[idx] = blank_questions
            self.scan_blank_summary[idx] = blank_map
            exam_code_text = (result.exam_code or "").strip()
            status_parts = self._status_parts_for_row(sid, exam_code_text, duplicate_ids.get(sid, 0))
            status = ", ".join(status_parts) if status_parts else "OK"
            content_text = self._build_recognition_content_text(result, blank_map)

            self.scan_list.insertRow(idx)
            sid_item = QTableWidgetItem(sid or "-")
            sid_item.setData(Qt.UserRole, str(result.image_path))
            sid_item.setData(Qt.UserRole + 1, exam_code_text)
            sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(result))
            self.scan_list.setItem(idx, 0, sid_item)
            self.scan_list.setItem(idx, 1, QTableWidgetItem(exam_code_text or "-"))
            self.scan_list.setItem(idx, 2, QTableWidgetItem(full_name))
            self.scan_list.setItem(idx, 3, QTableWidgetItem(birth_date))
            self.scan_list.setItem(idx, 4, QTableWidgetItem(content_text))
            status_item = QTableWidgetItem(status)
            if status != "OK":
                status_item.setForeground(Qt.red)
            self.scan_list.setItem(idx, 5, status_item)
            self._set_scan_action_widget(idx)
            for issue in result.issues:
                self.error_list.addItem(f"{Path(result.image_path).name}: {issue.code} - {issue.message}")
            for err in rec_errors:
                self.error_list.addItem(f"{Path(result.image_path).name}: RECOGNITION - {err}")

        self.btn_save_batch_subject.setEnabled(True)
        self._apply_scan_filter()

    def _compute_blank_questions(self, result) -> dict[str, list[int]]:
        expected_by_section = self._expected_questions_by_section(result)
        return {
            "MCQ": [q for q in sorted(set(expected_by_section["MCQ"])) if q not in set((result.mcq_answers or {}).keys())],
            "TF": [q for q in sorted(set(expected_by_section["TF"])) if q not in set((result.true_false_answers or {}).keys())],
            "NUMERIC": [q for q in sorted(set(expected_by_section["NUMERIC"])) if q not in set((result.numeric_answers or {}).keys())],
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _expected_questions_by_section(self, result) -> dict[str, list[int]]:
        template_expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if self.template:
            for z in self.template.zones:
                if not z.grid:
                    continue
                count = int(z.grid.question_count or z.grid.rows or 0)
                start = int(z.grid.question_start)
                rng = list(range(start, start + max(0, count)))
                if z.zone_type.value == "MCQ_BLOCK":
                    template_expected["MCQ"].extend(rng)
                elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                    template_expected["TF"].extend(rng)
                elif z.zone_type.value == "NUMERIC_BLOCK":
                    template_expected["NUMERIC"].extend(rng)
            template_expected = {sec: sorted(set(vals)) for sec, vals in template_expected.items()}

        expected_by_section = dict(template_expected)
        subject_key_name = self.active_batch_subject_key
        if not subject_key_name and self.session and self.session.subjects:
            subject_key_name = self.session.subjects[0]
        if self.answer_keys and subject_key_name:
            key = self.answer_keys.get(subject_key_name, (result.exam_code or "").strip())
            if key:
                key_sections = {
                    "MCQ": sorted(set(int(q) for q in (key.answers or {}).keys())),
                    "TF": sorted(set(int(q) for q in (key.true_false_answers or {}).keys())),
                    "NUMERIC": sorted(set(int(q) for q in (key.numeric_answers or {}).keys())),
                }
                for sec in ["MCQ", "TF", "NUMERIC"]:
                    if not key_sections[sec]:
                        continue
                    template_set = set(template_expected.get(sec, []))
                    key_set = set(key_sections[sec])
                    if template_set:
                        overlap = sorted(template_set & key_set)
                        if overlap:
                            expected_by_section[sec] = overlap
                        else:
                            # When answer-key numbering does not align with template numbering,
                            # prioritize answer-key section size and trim by template order.
                            count = len(key_sections[sec])
                            expected_by_section[sec] = sorted(template_set)[: max(0, count)]
                    else:
                        expected_by_section[sec] = key_sections[sec]
        return expected_by_section

    @staticmethod
    def _format_mcq_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}{str(a).strip()}" for q, a in sorted(answers.items(), key=lambda x: int(x[0])))

    @staticmethod
    def _format_tf_answers(answers: dict[int, dict[str, bool]]) -> str:
        if not answers:
            return "-"
        chunks: list[str] = []
        for q, flags in sorted(answers.items(), key=lambda x: int(x[0])):
            marks = "".join("Đ" if bool((flags or {}).get(k)) else "S" for k in ["a", "b", "c", "d"])
            chunks.append(f"{int(q)}{marks}")
        return "; ".join(chunks)

    @staticmethod
    def _format_numeric_answers(answers: dict[int, str]) -> str:
        if not answers:
            return "-"
        return "; ".join(f"{int(q)}={str(v).strip()}" for q, v in sorted(answers.items(), key=lambda x: int(x[0])))

    def _build_recognition_content_text(self, result, blank_map: dict[str, list[int]]) -> str:
        blank_parts = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            vals = blank_map.get(sec, [])
            if vals:
                blank_parts.append(f"{sec} trống: {','.join(str(v) for v in vals)}")
        return " | ".join(blank_parts) if blank_parts else ""

    def _trim_result_answers_to_expected_scope(self, result) -> None:
        expected = self._expected_questions_by_section(result)
        if expected.get("MCQ"):
            allow = set(expected["MCQ"])
            result.mcq_answers = {int(q): str(a) for q, a in (result.mcq_answers or {}).items() if int(q) in allow}
        if expected.get("TF"):
            allow = set(expected["TF"])
            result.true_false_answers = {int(q): dict(a) for q, a in (result.true_false_answers or {}).items() if int(q) in allow}
        if expected.get("NUMERIC"):
            allow = set(expected["NUMERIC"])
            result.numeric_answers = {int(q): str(a) for q, a in (result.numeric_answers or {}).items() if int(q) in allow}

    def _count_mismatch_status_parts(self, result) -> list[str]:
        expected = self._expected_questions_by_section(result)
        actual_map = {
            "MCQ": set((result.mcq_answers or {}).keys()),
            "TF": set((result.true_false_answers or {}).keys()),
            "NUMERIC": set((result.numeric_answers or {}).keys()),
        }
        messages: list[str] = []
        for sec in ["MCQ", "TF", "NUMERIC"]:
            expected_set = set(expected.get(sec, []))
            if not expected_set:
                continue
            actual_set = {int(q) for q in actual_map.get(sec, set())}
            missing = sorted(expected_set - actual_set)
            if missing:
                messages.append(
                    f"thiếu {sec} ({len(expected_set)-len(missing)}/{len(expected_set)}): {','.join(str(v) for v in missing)}"
                )
        return messages

    def _retrim_batch_results_to_answer_key_scope(self) -> None:
        if not self.scan_results:
            return
        for idx, res in enumerate(self.scan_results):
            scoped = self._scoped_result_copy(res)
            blank_map = self._compute_blank_questions(scoped)
            self.scan_blank_summary[idx] = blank_map
            self.scan_blank_questions[idx] = blank_map.get("MCQ", [])
            if idx < self.scan_list.rowCount():
                self.scan_list.setItem(idx, 3, QTableWidgetItem(self._build_recognition_content_text(res, blank_map)))
                sid_item = self.scan_list.item(idx, 0)
                if sid_item:
                    sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(res))
        self._refresh_all_statuses()
        current = self.scan_list.currentRow() if hasattr(self, "scan_list") else -1
        if 0 <= current < len(self.scan_results):
            self._update_scan_preview(current)
            self._load_selected_result_for_correction()

    def _apply_scan_filter(self) -> None:
        value = self.search_value.text().strip().lower()
        col = self.filter_column.currentIndex()
        for i in range(self.scan_list.rowCount()):
            item = self.scan_list.item(i, col)
            cell = (item.text() if item else "").lower()
            show = value in cell if value else True
            self.scan_list.setRowHidden(i, not show)

    def _on_scan_header_clicked(self, section: int) -> None:
        if 0 <= section < self.filter_column.count():
            self.filter_column.setCurrentIndex(section)
            self._apply_scan_filter()

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
        self.btn_save_batch_subject.setEnabled(True)


    def _rebuild_error_list(self) -> None:
        self.error_list.clear()
        for result in self.scan_results:
            rec_errors = list(getattr(result, "recognition_errors", [])) or list(getattr(result, "errors", []))
            for issue in result.issues:
                self.error_list.addItem(f"{Path(result.image_path).name}: {issue.code} - {issue.message}")
            for err in rec_errors:
                self.error_list.addItem(f"{Path(result.image_path).name}: RECOGNITION - {err}")

    def _update_scan_row_from_result(self, idx: int, result) -> None:
        if idx < 0 or idx >= self.scan_list.rowCount():
            return
        sid = (result.student_id or "").strip()
        exam_code_text = (result.exam_code or "").strip()
        sid_item = QTableWidgetItem(sid or "-")
        sid_item.setData(Qt.UserRole, str(result.image_path))
        sid_item.setData(Qt.UserRole + 1, exam_code_text)
        sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(result))
        self.scan_list.setItem(idx, 0, sid_item)
        self.scan_list.setItem(idx, 1, QTableWidgetItem(str(getattr(result, "full_name", "") or "-")))
        self.scan_list.setItem(idx, 2, QTableWidgetItem(str(getattr(result, "birth_date", "") or "-")))
        self.scan_list.setItem(
            idx,
            3,
            QTableWidgetItem(self._build_recognition_content_text(result, self.scan_blank_summary.get(idx, {"MCQ": [], "TF": [], "NUMERIC": []}))),
        )

    def _current_scan_results_snapshot(self) -> list[OMRResult]:
        base = list(self.scan_results or [])
        if not hasattr(self, "scan_list"):
            return base
        row_count = self.scan_list.rowCount()
        if row_count <= 0:
            return base

        out: list[OMRResult] = []
        for idx in range(row_count):
            if idx < len(base):
                result = copy.deepcopy(base[idx])
            else:
                fallback = self._build_result_from_saved_table_row(idx)
                result = fallback if fallback is not None else OMRResult(image_path="")

            sid_item = self.scan_list.item(idx, 0)
            sid_text = str(sid_item.text() if sid_item else "").strip()
            result.student_id = "" if sid_text in {"", "-"} else sid_text
            result.exam_code = str(sid_item.data(Qt.UserRole + 1) if sid_item else "").strip()
            image_path = str(sid_item.data(Qt.UserRole) if sid_item else "").strip()
            if image_path:
                result.image_path = image_path
            setattr(result, "full_name", str(self.scan_list.item(idx, 3).text() if self.scan_list.item(idx, 3) else ""))
            setattr(result, "birth_date", str(self.scan_list.item(idx, 3).text() if self.scan_list.item(idx, 3) else ""))
            out.append(result)
        return out

    def _build_result_from_saved_table_row(self, idx: int) -> OMRResult | None:
        if idx < 0 or idx >= self.scan_list.rowCount():
            return None
        sid_item = self.scan_list.item(idx, 0)
        image_path = str(sid_item.data(Qt.UserRole) if sid_item else "")
        if not image_path:
            return None
        student_id = str(sid_item.text() if sid_item else "").strip()
        if student_id == "-":
            student_id = ""
        exam_code = str(sid_item.data(Qt.UserRole + 1) if sid_item else "").strip()
        result = OMRResult(image_path=image_path, student_id=student_id, exam_code=exam_code)
        result.full_name = str(self.scan_list.item(idx, 3).text() if self.scan_list.item(idx, 3) else "")
        result.birth_date = str(self.scan_list.item(idx, 3).text() if self.scan_list.item(idx, 3) else "")
        result.sync_legacy_aliases()
        return result

    def _ensure_template_for_selected_subject(self) -> bool:
        cfg = self._selected_batch_subject_config() or self._resolve_subject_config_for_batch()
        template_path = ""
        if cfg:
            template_path = self._normalize_template_path(str(cfg.get("template_path", "") or ""))
        if not template_path and self.session:
            template_path = self._normalize_template_path(str(self.session.template_path or ""))
        if not template_path:
            return self.template is not None
        pth = Path(template_path)
        if not pth.exists():
            return self.template is not None
        try:
            self.template = Template.load_json(pth)
            return True
        except Exception:
            return self.template is not None

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

    def _marker_positions_for_result(self, result: OMRResult) -> list[dict[str, float]]:
        if self.template is None or self.preview_source_pixmap.isNull():
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
            if z.zone_type.value == "MCQ_BLOCK":
                options = list(g.options or ["A", "B", "C", "D"])
                cols = max(1, int(g.cols or len(options) or 1))
                qcount = int(g.question_count or g.rows or 0)
                for i in range(qcount):
                    qno = int(g.question_start) + i
                    ans = str((result.mcq_answers or {}).get(qno, "") or "").strip().upper()
                    if not ans:
                        continue
                    for ch in ans:
                        if ch not in options:
                            continue
                        c = options.index(ch)
                        idx = i * cols + c
                        if 0 <= idx < len(g.bubble_positions):
                            bx, by = g.bubble_positions[idx]
                            markers.append({"zone_id": z.id, "section": "MCQ", "qno": float(qno), "choice": float(c), "x": float(bx) * sx, "y": float(by) * sy})
            elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                qpb = int(z.metadata.get("questions_per_block", 2))
                spq = int(z.metadata.get("statements_per_question", 4))
                cps = int(z.metadata.get("choices_per_statement", 2))
                cols = max(1, int(g.cols or cps))
                labels = [chr(ord("a") + i) for i in range(spq)]
                for q in range(qpb):
                    qno = int(g.question_start) + q
                    flags = (result.true_false_answers or {}).get(qno, {}) or {}
                    for sidx, label in enumerate(labels):
                        if label not in flags:
                            continue
                        c = 0 if bool(flags.get(label)) else 1
                        row = q * spq + sidx
                        idx = row * cols + c
                        if 0 <= idx < len(g.bubble_positions):
                            bx, by = g.bubble_positions[idx]
                            markers.append({"zone_id": z.id, "section": "TF", "qno": float(qno), "stmt": float(sidx), "choice": float(c), "x": float(bx) * sx, "y": float(by) * sy})
        return markers

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

    def _apply_adjusted_markers_to_result(self, idx: int, result: OMRResult) -> bool:
        if self.template is None:
            return False
        markers = self.scan_image_preview.markers() if hasattr(self, "scan_image_preview") else []
        if not markers or self.preview_source_pixmap.isNull():
            return False
        tpl_w = max(1.0, float(self.template.width))
        tpl_h = max(1.0, float(self.template.height))
        img_w = max(1.0, float(self.preview_source_pixmap.width()))
        img_h = max(1.0, float(self.preview_source_pixmap.height()))
        sx, sy = tpl_w / img_w, tpl_h / img_h

        by_zone: dict[str, list[dict[str, float]]] = {}
        for m in markers:
            zid = str(m.get("zone_id", ""))
            if zid:
                by_zone.setdefault(zid, []).append(m)

        updated = False
        mcq_new = dict(result.mcq_answers or {})
        tf_new = dict(result.true_false_answers or {})
        for z in self.template.zones:
            g = z.grid
            if not g or not g.bubble_positions:
                continue
            zmarks = by_zone.get(z.id, [])
            if not zmarks:
                continue
            pts = [(float(x), float(y)) for x, y in g.bubble_positions]
            if z.zone_type.value == "MCQ_BLOCK":
                options = list(g.options or ["A", "B", "C", "D"])
                cols = max(1, int(g.cols or len(options) or 1))
                qcount = int(g.question_count or g.rows or 0)
                for qidx in range(qcount):
                    qno = int(g.question_start) + qidx
                    picks = [m for m in zmarks if int(round(float(m.get("qno", -1)))) == qno]
                    if not picks:
                        continue
                    letters: list[str] = []
                    for m in picks:
                        tx, ty = float(m.get("x", 0.0)) * sx, float(m.get("y", 0.0)) * sy
                        best = min(range(len(pts)), key=lambda k: (pts[k][0]-tx)**2 + (pts[k][1]-ty)**2)
                        row = best // cols
                        col = best % cols
                        if row != qidx:
                            continue
                        if 0 <= col < len(options):
                            letters.append(options[col])
                    if letters:
                        mcq_new[qno] = "".join(sorted(set(letters), key=letters.index))
                        updated = True
            elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                qpb = int(z.metadata.get("questions_per_block", 2))
                spq = int(z.metadata.get("statements_per_question", 4))
                cps = int(z.metadata.get("choices_per_statement", 2))
                cols = max(1, int(g.cols or cps))
                labels = [chr(ord("a") + i) for i in range(spq)]
                for q in range(qpb):
                    qno = int(g.question_start) + q
                    flags = dict(tf_new.get(qno, {}) or {})
                    for sidx, label in enumerate(labels):
                        picks = [m for m in zmarks if int(round(float(m.get("qno", -1)))) == qno and int(round(float(m.get("stmt", -1)))) == sidx]
                        if not picks:
                            continue
                        m = picks[-1]
                        tx, ty = float(m.get("x", 0.0)) * sx, float(m.get("y", 0.0)) * sy
                        best = min(range(len(pts)), key=lambda k: (pts[k][0]-tx)**2 + (pts[k][1]-ty)**2)
                        row = best // cols
                        col = best % cols
                        expected_row = q * spq + sidx
                        if row != expected_row:
                            continue
                        flags[label] = (col == 0)
                        updated = True
                    if flags:
                        tf_new[qno] = flags

        if not updated:
            return False
        result.mcq_answers = {int(k): str(v) for k, v in mcq_new.items()}
        result.true_false_answers = {int(k): dict(v) for k, v in tf_new.items()}
        self.scan_blank_summary[idx] = self._compute_blank_questions(result)
        return True

    def _set_scan_result_at_row(self, idx: int, result: OMRResult) -> None:
        if idx < 0:
            return
        while len(self.scan_results) <= idx:
            placeholder = self._build_result_from_saved_table_row(len(self.scan_results))
            if placeholder is None:
                placeholder = OMRResult(image_path="")
            self.scan_results.append(placeholder)
        self.scan_results[idx] = result

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
        image_path = str(old_result.image_path or "").strip()
        if not image_path or not Path(image_path).exists():
            QMessageBox.warning(self, "Nhận dạng lại", f"Không tìm thấy ảnh để nhận dạng lại:\n{image_path or '-'}")
            return

        if self.scan_image_preview.has_markers():
            choose = QMessageBox.question(
                self,
                "Nhận dạng lại",
                "Bạn muốn áp dụng vị trí dấu X đã chỉnh để cập nhật kết quả (không thay đổi template)?\nChọn No để chạy nhận dạng ảnh lại như bình thường.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if choose == QMessageBox.Cancel:
                return
            if choose == QMessageBox.Yes:
                manual_result = copy.deepcopy(old_result)
                if self._apply_adjusted_markers_to_result(idx, manual_result):
                    self._set_scan_result_at_row(idx, manual_result)
                    scoped_manual = self._scoped_result_copy(manual_result)
                    self.scan_blank_questions[idx] = self._compute_blank_questions(scoped_manual).get("MCQ", [])
                    self.scan_blank_summary[idx] = self._compute_blank_questions(scoped_manual)
                    self._update_scan_row_from_result(idx, manual_result)
                    self._refresh_all_statuses()
                    self._update_scan_preview(idx)
                    self.btn_save_batch_subject.setEnabled(True)
                    return
                QMessageBox.information(self, "Nhận dạng lại", "Không áp dụng được dấu X đã chỉnh. Hệ thống sẽ chạy nhận dạng ảnh lại.")

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

        new_result = self.omr_processor.process_image(process_path, self.template)
        if temp_rotated_path:
            try:
                Path(temp_rotated_path).unlink(missing_ok=True)
            except Exception:
                pass
        new_result.image_path = image_path

        sid = (new_result.student_id or "").strip()
        profile = self._student_profile_by_id(sid)
        if profile.get("name"):
            setattr(new_result, "full_name", profile.get("name"))
        if profile.get("birth_date"):
            setattr(new_result, "birth_date", profile.get("birth_date"))
        if profile.get("class_name"):
            setattr(new_result, "class_name", profile.get("class_name"))
        if profile.get("exam_room"):
            setattr(new_result, "exam_room", profile.get("exam_room"))
        scoped_new = self._scoped_result_copy(new_result)
        blank_map = self._compute_blank_questions(scoped_new)

        rec_errors = list(getattr(new_result, "recognition_errors", [])) or list(getattr(new_result, "errors", []))
        message = (
            f"Ảnh: {Path(image_path).name}\n"
            f"STUDENT ID mới: {new_result.student_id or '-'}\n"
            f"Mã đề mới: {new_result.exam_code or '-'}\n"
            f"Nhận dạng ngắn: {self._compact_value(self._short_recognition_text_for_result(new_result), 180)}\n"
            f"Số lỗi nhận dạng: {len(rec_errors)}\n\n"
            "Lưu nhận dạng hay không?\n"
            "Yes: Cập nhật lưới và lưu batch ngay.\n"
            "No: Chỉ cập nhật lưới (chưa lưu batch).\n"
            "Cancel: Không cập nhật."
        )
        decision = QMessageBox.question(
            self,
            "Xác nhận nhận dạng lại",
            message,
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if decision == QMessageBox.Cancel:
            return

        self._set_scan_result_at_row(idx, new_result)
        self.scan_blank_questions[idx] = blank_map.get("MCQ", [])
        self.scan_blank_summary[idx] = blank_map
        self._update_scan_row_from_result(idx, new_result)
        self._refresh_all_statuses()
        self._rebuild_error_list()
        self._update_scan_preview(idx)
        self._load_selected_result_for_correction()
        self.btn_save_batch_subject.setEnabled(True)

        if decision == QMessageBox.Yes:
            self._save_batch_for_selected_subject()

    def _render_preview_pixmap(self) -> None:
        if self.preview_source_pixmap.isNull():
            return
        base_w = max(1, self.scan_image_scroll.viewport().width() - 8)
        w = max(1, int(base_w * self.preview_zoom_factor))
        scaled = self.preview_source_pixmap.scaledToWidth(w, Qt.SmoothTransformation)
        self.scan_image_preview.setPixmap(scaled)

    def _zoom_preview_in(self) -> None:
        self.preview_zoom_factor = min(4.0, self.preview_zoom_factor + 0.1)
        self._render_preview_pixmap()
        self.btn_zoom_reset.setText(f"{int(self.preview_zoom_factor*100)}%")

    def _zoom_preview_out(self) -> None:
        self.preview_zoom_factor = max(0.3, self.preview_zoom_factor - 0.1)
        self._render_preview_pixmap()
        self.btn_zoom_reset.setText(f"{int(self.preview_zoom_factor*100)}%")

    def _zoom_preview_reset(self) -> None:
        self.preview_zoom_factor = 1.0
        self._render_preview_pixmap()
        self.btn_zoom_reset.setText("100%")

    @staticmethod
    def _compact_value(value, limit: int = 120) -> str:
        text = str(value)
        return text if len(text) <= limit else text[:limit] + "..."

    def _update_scan_preview_from_saved_row(self, row: int) -> None:
        sid = self.scan_list.item(row, 0).text() if self.scan_list.item(row, 0) else "-"
        exam_code_cell = self.scan_list.item(row, 1).text() if self.scan_list.item(row, 1) else "-"
        full_name = self.scan_list.item(row, 2).text() if self.scan_list.item(row, 2) else "-"
        birth = self.scan_list.item(row, 3).text() if self.scan_list.item(row, 3) else "-"
        content = self.scan_list.item(row, 4).text() if self.scan_list.item(row, 4) else "-"
        status = self.scan_list.item(row, 5).text() if self.scan_list.item(row, 5) else "-"
        img_path = ""
        exam_code = ""
        recognized_short = ""
        item0 = self.scan_list.item(row, 0)
        if item0:
            img_path = str(item0.data(Qt.UserRole) or "")
            exam_code = str(item0.data(Qt.UserRole + 1) or "")
            recognized_short = str(item0.data(Qt.UserRole + 2) or "")

        pix = QPixmap(img_path) if img_path else QPixmap()
        if pix.isNull():
            self.preview_source_pixmap = QPixmap()
            self.scan_image_preview.setPixmap(QPixmap())
            self.scan_image_preview.setText("Không có ảnh tương ứng cho dòng đã lưu")
            self.scan_image_preview.clear_markers()
            self.btn_zoom_reset.setText("100%")
        else:
            rotation = int(self.preview_rotation_by_index.get(row, 0) or 0) % 360
            if rotation:
                pix = pix.transformed(QTransform().rotate(float(rotation)), Qt.SmoothTransformation)
            self.preview_source_pixmap = pix
            self._render_preview_pixmap()
            self.btn_zoom_reset.setText(f"{int(self.preview_zoom_factor*100)}%")

        tmp_result = self._build_result_from_saved_table_row(row)
        if tmp_result is not None:
            self.scan_image_preview.set_overlay_markers(self._recognition_overlay_positions_for_result(tmp_result))
            self.scan_image_preview.set_markers(self._marker_positions_for_result(tmp_result))
        else:
            self.scan_image_preview.clear_markers()

        rows = [
            ("STUDENT ID", sid),
            ("Họ tên", full_name),
            ("Ngày sinh", birth),
            ("Mã đề", exam_code or exam_code_cell or "-"),
            ("Xoay tạm", f"{int(self.preview_rotation_by_index.get(row, 0) or 0)%360}°"),
            ("Nhận dạng ngắn", self._compact_value(recognized_short or "-", 220)),
            ("Nội dung", self._compact_value(content, 220)),
            ("Status", status),
            ("Ảnh", img_path or "-"),
        ]
        self.scan_result_preview.setRowCount(0)
        for r, (k, v) in enumerate(rows):
            self.scan_result_preview.insertRow(r)
            self.scan_result_preview.setItem(r, 0, QTableWidgetItem(str(k)))
            self.scan_result_preview.setItem(r, 1, QTableWidgetItem(str(v)))

    def _on_scan_selected(self) -> None:
        index = self.scan_list.currentRow()
        if index < 0:
            return
        if 0 <= index < len(self.scan_results):
            self._update_scan_preview(index)
            self._load_selected_result_for_correction()
            return
        self._update_scan_preview_from_saved_row(index)

    @staticmethod
    def _normalize_exam_code_text(code: str) -> str:
        c = str(code or "").strip()
        if not c:
            return ""
        if c.isdigit():
            # Treat purely numeric exam codes with/without leading zeros as equivalent.
            c2 = c.lstrip("0")
            return c2 if c2 else "0"
        return c

    def _available_exam_codes(self) -> set[str]:
        out: set[str] = set()
        for x in (self.imported_exam_codes or []):
            raw = str(x).strip()
            if not raw:
                continue
            out.add(raw)
            out.add(self._normalize_exam_code_text(raw))
        return {v for v in out if v}

    def _subject_answer_key_for_result(self, result, subject_key: str = ""):
        subject = str(subject_key or self._current_batch_subject_key() or "").strip()
        exam_code = str(getattr(result, "exam_code", "") or "").strip()
        if not subject or not exam_code:
            return None
        if self.answer_keys is not None:
            key = self.answer_keys.get(subject, exam_code)
            if key is not None:
                return key
            normalized = self._normalize_exam_code_text(exam_code)
            for candidate in (self.imported_exam_codes or []):
                candidate_text = str(candidate).strip()
                if candidate_text and self._normalize_exam_code_text(candidate_text) == normalized:
                    key = self.answer_keys.get(subject, candidate_text)
                    if key is not None:
                        return key
        fetched = self.database.fetch_answer_keys_for_subject(subject)
        if exam_code in fetched:
            return fetched[exam_code]
        normalized = self._normalize_exam_code_text(exam_code)
        for candidate_text, candidate_key in fetched.items():
            if self._normalize_exam_code_text(candidate_text) == normalized:
                return candidate_key
        return None

    @staticmethod
    def _answer_string_from_maps(mcq_answers: dict[int, str], tf_answers: dict[int, dict[str, bool]], numeric_answers: dict[int, str], answer_key) -> str:
        if answer_key is None:
            return ""

        def _question_numbers(valid_map, invalid_map) -> list[int]:
            nums = set()
            for src in [valid_map or {}, invalid_map or {}]:
                for key in src.keys():
                    if str(key).strip().lstrip("-").isdigit():
                        nums.add(int(key))
            return sorted(nums)

        invalid_rows = getattr(answer_key, "invalid_answer_rows", {}) or {}
        parts: list[str] = []
        for q_no in _question_numbers(getattr(answer_key, "answers", {}) or {}, (invalid_rows.get("MCQ", {}) or {})):
            value = str((mcq_answers or {}).get(q_no, "") or "").strip().upper()[:1]
            parts.append(value or "_")
        for q_no in _question_numbers(getattr(answer_key, "true_false_answers", {}) or {}, (invalid_rows.get("TF", {}) or {})):
            flags = (tf_answers or {}).get(q_no, {}) or {}
            for key in ["a", "b", "c", "d"]:
                parts.append("Đ" if key in flags and bool(flags.get(key)) else ("S" if key in flags else "_"))
        for q_no in _question_numbers(getattr(answer_key, "numeric_answers", {}) or {}, (invalid_rows.get("NUMERIC", {}) or {})):
            raw_key = str((getattr(answer_key, "numeric_answers", {}) or {}).get(q_no, ((invalid_rows.get("NUMERIC", {}) or {}).get(q_no, ""))) or "")
            normalized_key = str(raw_key).strip().replace(" ", "").lstrip("+").replace(".", ",")
            width = len(normalized_key) if normalized_key else max(1, len(raw_key.strip()))
            student_text = str((numeric_answers or {}).get(q_no, "") or "").strip().replace(" ", "").lstrip("+").replace(".", ",")[:width]
            if len(student_text) < width:
                student_text = student_text + ("_" * (width - len(student_text)))
            parts.append(student_text)
        return "".join(parts)

    def _build_answer_string_for_result(self, result, subject_key: str = "") -> str:
        key = self._subject_answer_key_for_result(result, subject_key)
        return self._answer_string_from_maps(result.mcq_answers or {}, result.true_false_answers or {}, result.numeric_answers or {}, key)

    def _short_recognition_text_for_result(self, result) -> str:
        parts: list[str] = []
        mcq = self._format_mcq_answers(result.mcq_answers or {})
        tf = self._format_tf_answers(result.true_false_answers or {})
        num = self._format_numeric_answers(result.numeric_answers or {})
        if mcq and mcq != "-":
            parts.append(f"MCQ: {mcq}")
        if tf and tf != "-":
            parts.append(f"TF: {tf}")
        if num and num != "-":
            parts.append(f"NUM: {num}")
        return " | ".join(parts) if parts else "-"

    def _status_parts_for_row(self, sid: str, exam_code_text: str, duplicate_count: int) -> list[str]:
        parts: list[str] = []
        if sid and duplicate_count > 1:
            parts.append("Trùng SBD")
        avail_codes = self._available_exam_codes()
        code = (exam_code_text or "").strip()
        norm_code = self._normalize_exam_code_text(code)
        if not code or "?" in code or (avail_codes and (code not in avail_codes and norm_code not in avail_codes)):
            parts.append("Lỗi mã đề")
        return parts

    @staticmethod
    def _name_missing(name_text: str) -> bool:
        name = str(name_text or "").strip()
        return name in {"", "-"}

    def _status_text_for_row(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.scan_results):
            return "OK"
        res = self.scan_results[idx]
        sid = (res.student_id or "").strip()
        dup = sum(1 for r in self.scan_results if (r.student_id or "").strip() == sid) if sid else 0
        exam_code_text = (res.exam_code or "").strip()
        status_parts = self._status_parts_for_row(sid, exam_code_text, dup)
        full_name = str(getattr(res, "full_name", "") or "")
        if sid and self._name_missing(full_name):
            profile = self._student_profile_by_id(sid)
            if not str(profile.get("name", "") or "").strip():
                status_parts.append("Lỗi SBD")
        return ", ".join(status_parts) if status_parts else "OK"

    def _current_batch_subject_key(self) -> str:
        cfg = self._selected_batch_subject_config()
        if cfg:
            return self._subject_key_from_cfg(cfg)
        return str(self.active_batch_subject_key or "").strip() or self._resolve_preferred_scoring_subject()

    def _invalidate_scoring_for_student_ids(self, student_ids: list[str], subject_key: str = "", reason: str = "") -> int:
        subject = str(subject_key or self._current_batch_subject_key() or "").strip()
        if not subject:
            return 0
        subject_scores = dict(self.scoring_results_by_subject.get(subject, {}))
        if not subject_scores:
            return 0
        changed = 0
        for sid in student_ids:
            sid_key = str(sid or "").strip()
            if not sid_key:
                continue
            if sid_key in subject_scores:
                subject_scores.pop(sid_key, None)
                changed += 1
        if changed <= 0:
            return 0
        self.scoring_results_by_subject[subject] = subject_scores
        for row in rows:
            self.database.upsert_score_row(subject, row.student_id, row.exam_code, {
                "student_id": row.student_id,
                "name": row.name,
                "subject": row.subject,
                "exam_code": row.exam_code,
                "score": row.score,
                "correct": row.correct,
                "wrong": row.wrong,
                "blank": row.blank,
            })
        phase = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "subject": subject,
            "mode": "Cập nhật sau sửa bài",
            "count": changed,
            "missing": 0,
            "note": reason or "invalidate_scoring_records",
        }
        self.scoring_phases.append(phase)
        if len(self.scoring_phases) > 500:
            self.scoring_phases = self.scoring_phases[-500:]
        if self.session:
            cfg = dict(self.session.config or {})
            cfg["scoring_phases"] = list(self.scoring_phases)
            cfg["scoring_results"] = dict(self.scoring_results_by_subject)
            cfg["last_scoring_phase"] = dict(phase)
            self.session.config = cfg
            self.session_dirty = True
            self._persist_session_quietly()
        self._refresh_scoring_phase_table()
        return changed

    def _record_adjustment(self, idx: int, details: list[str], source: str) -> None:
        if not details:
            return
        message = f"({source}) " + "; ".join(details)
        self.scan_edit_history.setdefault(idx, []).append(message)
        self.scan_last_adjustment[idx] = message
        self.scan_manual_adjustments[idx] = sorted(set(self.scan_manual_adjustments.get(idx, []) + details))
        if 0 <= idx < len(self.scan_results):
            res = self.scan_results[idx]
            self.database.log_change("scan_results", str(getattr(res, "image_path", "") or idx), source, "", message, source)

    def _persist_scan_results_to_db(self, subject_key: str) -> None:
        source_rows = list(self.scan_results_by_subject.get(subject_key, self.scan_results) or [])
        for result in source_rows:
            result.answer_string = self._build_answer_string_for_result(result, subject_key)
        rows = [self._serialize_omr_result(x) for x in source_rows]
        self.database.replace_scan_results_for_subject(subject_key, rows)
        self.database.log_change("scan_results", subject_key, "replace_subject_rows", "", f"{len(rows)} rows", "batch_save")
        self._refresh_scan_results_from_db(subject_key)

    def _persist_single_scan_result_to_db(self, result: OMRResult, note: str = "") -> None:
        subject_key = self._current_batch_subject_key()
        result.answer_string = self._build_answer_string_for_result(result, subject_key)
        self.database.update_scan_result_payload(str(getattr(result, "image_path", "") or ""), self._serialize_omr_result(result), note=note)

    def _refresh_all_statuses(self) -> None:
        for row_idx in range(self.scan_list.rowCount()):
            self._refresh_row_status(row_idx)

    def _on_scan_cell_clicked(self, row: int, col: int) -> None:
        if row < 0:
            return
        # Only show edit history when clicking the Status column.
        if col != 4:
            return
        history = self.scan_edit_history.get(row, [])
        if not history:
            QMessageBox.information(self, "Lịch sử sửa", "Chưa có lịch sử điều chỉnh trong Status cho bài thi này.")
            return
        latest = self.scan_last_adjustment.get(row, history[-1])
        QMessageBox.information(
            self,
            "Lịch sử sửa bài",
            "Điều chỉnh gần nhất:\n"
            + latest
            + "\n\nToàn bộ lịch sử:\n"
            + "\n".join(history),
        )

    def _status_text_for_saved_table_row(self, row_idx: int) -> str:
        sid_item = self.scan_list.item(row_idx, 0)
        sid = (sid_item.text().strip() if sid_item else "")
        exam_code_text = str(sid_item.data(Qt.UserRole + 1) if sid_item else "").strip()
        dup = 0
        if sid and sid != "-":
            for r in range(self.scan_list.rowCount()):
                it = self.scan_list.item(r, 0)
                v = (it.text().strip() if it else "")
                if v and v != "-" and v == sid:
                    dup += 1
        status_parts = self._status_parts_for_row(sid if sid != "-" else "", exam_code_text, dup)
        name_item = self.scan_list.item(row_idx, 2)
        name_text = name_item.text().strip() if name_item else ""
        if sid and sid != "-" and self._name_missing(name_text):
            status_parts.append("Lỗi SBD")
        return ", ".join(status_parts) if status_parts else "OK"

    def _refresh_row_status(self, idx: int) -> None:
        if idx < 0 or idx >= self.scan_list.rowCount():
            return
        forced_status = self.scan_forced_status_by_index.get(idx, "")
        status = forced_status or (self._status_text_for_row(idx) if idx < len(self.scan_results) else self._status_text_for_saved_table_row(idx))
        item = QTableWidgetItem(status)
        if status != "OK":
            item.setForeground(Qt.red)
        self.scan_list.setItem(idx, 5, item)

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
            self.scan_image_preview.set_markers(self._marker_positions_for_result(result))

        rec_errors = list(getattr(result, "recognition_errors", [])) or list(getattr(result, "errors", []))
        blank_map = self.scan_blank_summary.get(index, {"MCQ": [], "TF": [], "NUMERIC": []})
        rows = [
            ("STUDENT ID", result.student_id or "-"),
            ("Họ tên", str(getattr(result, "full_name", "") or "-")),
            ("Ngày sinh", str(getattr(result, "birth_date", "") or "-")),
            ("Exam code", result.exam_code or "-"),
            ("Xoay tạm", f"{int(self.preview_rotation_by_index.get(index, 0) or 0)%360}°"),
            ("Nhận dạng ngắn", self._compact_value(self._short_recognition_text_for_result(result), 220)),
            ("MCQ", self._compact_value(self._format_mcq_answers(result.mcq_answers or {}), 220)),
            ("TF", self._compact_value(self._format_tf_answers(result.true_false_answers or {}), 220)),
            ("NUM", self._compact_value(self._format_numeric_answers(result.numeric_answers or {}), 220)),
            ("MCQ không tô", ", ".join(str(x) for x in blank_map.get("MCQ", [])) or "-"),
            ("TF không tô", ", ".join(str(x) for x in blank_map.get("TF", [])) or "-"),
            ("NUMERIC không tô", ", ".join(str(x) for x in blank_map.get("NUMERIC", [])) or "-"),
            ("Issues", "; ".join(f"{i.code}:{i.message}" for i in result.issues) or "-"),
            ("Recognition errors", "; ".join(rec_errors) or "-"),
        ]
        self.scan_result_preview.setRowCount(0)
        for r, (k, v) in enumerate(rows):
            self.scan_result_preview.insertRow(r)
            self.scan_result_preview.setItem(r, 0, QTableWidgetItem(str(k)))
            self.scan_result_preview.setItem(r, 1, QTableWidgetItem(str(v)))

    def _sync_correction_detail_panel(self, res: OMRResult, rebuild_editor: bool = False) -> None:
        subject_key = self._current_batch_subject_key()
        self._load_exam_code_correction_options(subject_key, str(res.exam_code or "").strip())
        self._load_student_correction_options(str(res.student_id or "").strip())
        if rebuild_editor:
            self._build_visual_answer_editor(res)
        payload = {
            "student_id": res.student_id,
            "exam_code": res.exam_code,
            "answer_string": str(getattr(res, "answer_string", "") or self._build_answer_string_for_result(res, subject_key)),
            "mcq_answers": res.mcq_answers,
            "true_false_answers": res.true_false_answers,
            "numeric_answers": res.numeric_answers,
            "issues": [{"code": i.code, "message": i.message, "zone_id": i.zone_id} for i in res.issues],
            "recognition_errors": list(getattr(res, "recognition_errors", [])) or list(getattr(res, "errors", [])),
        }
        self.result_preview.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self.manual_edit.setPlainText(
            json.dumps(
                {
                    "student_id": res.student_id,
                    "exam_code": res.exam_code,
                    "answer_string": payload["answer_string"],
                    "mcq_answers": res.mcq_answers,
                    "true_false_answers": res.true_false_answers,
                    "numeric_answers": res.numeric_answers,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        self.correction_ui_loading = False

    def _load_selected_result_for_correction(self) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= len(self.scan_results):
            return
        res = self.scan_results[idx]
        self.correction_ui_loading = True
        self._sync_correction_detail_panel(res, rebuild_editor=True)
        self.correction_ui_loading = False

    def _load_selected_result_for_correction(self) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= len(self.scan_results):
            return
        res = self.scan_results[idx]
        self.correction_ui_loading = True
        self._sync_correction_detail_panel(res, rebuild_editor=True)
        self.correction_ui_loading = False

    def _load_selected_result_for_correction(self) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= len(self.scan_results):
            return
        res = self.scan_results[idx]
        self.correction_ui_loading = True
        self._sync_correction_detail_panel(res, rebuild_editor=True)
        self.correction_ui_loading = False

    def _load_selected_result_for_correction(self) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= len(self.scan_results):
            return
        res = self.scan_results[idx]
        self.correction_ui_loading = True
        self._sync_correction_detail_panel(res, rebuild_editor=True)
        self.correction_ui_loading = False

    def _open_edit_selected_scan(self, *_args) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "No selection", "Chọn bài thi cần sửa trước.")
            return
        if idx >= len(self.scan_results):
            sid_item_existing = self.scan_list.item(idx, 0)
            sid = sid_item_existing.text() if sid_item_existing else "-"
            content = self.scan_list.item(idx, 4).text() if self.scan_list.item(idx, 4) else "-"
            exam_code = str(sid_item_existing.data(Qt.UserRole + 1) if sid_item_existing else "").strip()
            if not exam_code:
                for r in range(self.scan_result_preview.rowCount()):
                    k = self.scan_result_preview.item(r, 0)
                    v = self.scan_result_preview.item(r, 1)
                    if k and v and k.text().strip().lower() in {"exam code", "mã đề"}:
                        exam_code = v.text().strip()
                        break
            dlg = QDialog(self)
            dlg.setWindowTitle("Sửa bài thi đã lưu")
            lay = QVBoxLayout(dlg)
            form = QFormLayout()
            inp_sid = QLineEdit(sid)
            inp_code = QLineEdit(exam_code)
            txt_content = QTextEdit(content)
            form.addRow("Student ID", inp_sid)
            form.addRow("Exam Code", inp_code)
            lay.addLayout(form)
            lay.addWidget(QLabel("Nội dung"))
            lay.addWidget(txt_content)
            buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            left_left_lay.addWidget(buttons)
            if dlg.exec() != QDialog.Accepted:
                return
            old_item = self.scan_list.item(idx, 0)
            old_sid = old_item.text().strip() if old_item else ""
            old_img = str(old_item.data(Qt.UserRole) if old_item else "")
            old_exam_code = str(old_item.data(Qt.UserRole + 1) if old_item else "").strip()
            old_recognized_short = str(old_item.data(Qt.UserRole + 2) if old_item else "")
            new_exam_code = inp_code.text().strip() or old_exam_code
            sid_item = QTableWidgetItem(inp_sid.text().strip() or "-")
            sid_item.setData(Qt.UserRole, old_img)
            sid_item.setData(Qt.UserRole + 1, new_exam_code)
            sid_item.setData(Qt.UserRole + 2, old_recognized_short)
            self.scan_list.setItem(idx, 0, sid_item)
            self.scan_list.setItem(idx, 4, QTableWidgetItem(txt_content.toPlainText().strip() or "-"))
            self._refresh_row_status(idx)
            for r in range(self.scan_result_preview.rowCount()):
                k = self.scan_result_preview.item(r, 0)
                if k and k.text().strip().lower() in {"exam code", "mã đề"}:
                    self.scan_result_preview.setItem(r, 1, QTableWidgetItem(new_exam_code or "-"))
                    break
            self.btn_save_batch_subject.setEnabled(True)
            invalidated = self._invalidate_scoring_for_student_ids(
                [old_sid, inp_sid.text().strip()],
                reason="saved_row_edit",
            )
            if invalidated > 0:
                QMessageBox.information(
                    self,
                    "Tính điểm",
                    f"Đã đánh dấu {invalidated} bản ghi cần chấm lại do sửa bài. Vui lòng chạy lại Tính điểm.",
                )
            return
        res = self.scan_results[idx]
        old_sid_for_score = str(res.student_id or "").strip()

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Sửa bài thi: {Path(res.image_path).name}")
        dlg.setWindowState(Qt.WindowMaximized)
        lay = QHBoxLayout(dlg)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_lay = QVBoxLayout(left)
        form = QFormLayout()

        inp_sid = QComboBox()
        inp_sid.setEditable(True)
        inp_sid.setInsertPolicy(QComboBox.NoInsert)
        self._load_student_correction_options(str(res.student_id or "").strip())
        for i in range(self.student_correction_combo.count()):
            inp_sid.addItem(self.student_correction_combo.itemText(i), self.student_correction_combo.itemData(i))
        idx_sid = inp_sid.findData(str(res.student_id or "").strip())
        inp_sid.setCurrentIndex(max(0, idx_sid))
        inp_sid.setCompleter(self.student_correction_combo.completer())

        inp_code = QComboBox()
        subject_key = self._current_batch_subject_key()
        self._load_exam_code_correction_options(subject_key, str(res.exam_code or "").strip())
        for i in range(self.exam_code_correction_combo.count()):
            inp_code.addItem(self.exam_code_correction_combo.itemText(i), self.exam_code_correction_combo.itemData(i))
        idx_code = inp_code.findData(str(res.exam_code or "").strip())
        inp_code.setCurrentIndex(max(0, idx_code))
        expected = self._expected_questions_by_section(res)

        def _build_pair_table(data: dict[int, str], value_placeholder: str = "") -> QTableWidget:
            table = QTableWidget(0, 2)
            table.setHorizontalHeaderLabels(["Câu", "Giá trị"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            for q, v in sorted((data or {}).items(), key=lambda x: int(x[0])):
                r = table.rowCount()
                table.insertRow(r)
                table.setItem(r, 0, QTableWidgetItem(str(int(q))))
                item_v = QTableWidgetItem(str(v))
                if value_placeholder:
                    item_v.setToolTip(value_placeholder)
                table.setItem(r, 1, item_v)
            return table

        def _build_mcq_grid(data: dict[int, str]) -> tuple[QWidget, dict[int, QLineEdit]]:
            box = QWidget()
            grid = QGridLayout(box)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)
            edits: dict[int, QLineEdit] = {}
            questions = list(expected.get("MCQ", [])) or sorted(int(q) for q in (data or {}).keys())
            if not questions:
                questions = []
            cols = 8
            for idx_q, q_no in enumerate(questions):
                row = (idx_q // cols) * 2
                col = idx_q % cols
                lbl = QLabel(str(q_no))
                lbl.setAlignment(Qt.AlignCenter)
                edit = QLineEdit(str((data or {}).get(q_no, "") or ""))
                edit.setMaxLength(1)
                edit.setMaximumWidth(52)
                edit.setAlignment(Qt.AlignCenter)
                edits[int(q_no)] = edit
                grid.addWidget(lbl, row, col)
                grid.addWidget(edit, row + 1, col)
            grid.setColumnStretch(cols, 1)
            return box, edits

        def _build_tf_table(data: dict[int, dict[str, bool]]) -> QTableWidget:
            table = QTableWidget(0, 5)
            labels = ["a", "b", "c", "d"]
            table.setHorizontalHeaderLabels(["Câu", *[s.upper() for s in labels]])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            for c in range(1, 5):
                table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
            for q, flags in sorted((data or {}).items(), key=lambda x: int(x[0])):
                r = table.rowCount()
                table.insertRow(r)
                table.setItem(r, 0, QTableWidgetItem(str(int(q))))
                flags = dict(flags or {})
                for i, key in enumerate(labels, start=1):
                    cb = QComboBox()
                    cb.addItem("-", None)
                    cb.addItem("Đúng", True)
                    cb.addItem("Sai", False)
                    val = flags.get(key, None)
                    idx_found = 0
                    for k in range(cb.count()):
                        if cb.itemData(k) is val:
                            idx_found = k
                            break
                    cb.setCurrentIndex(idx_found)
                    table.setCellWidget(r, i, cb)
            return table

        mcq_widget, mcq_edits = _build_mcq_grid(res.mcq_answers)
        table_num = _build_pair_table(res.numeric_answers, "Ví dụ: -12.5")
        table_tf = _build_tf_table(res.true_false_answers)

        def _add_pair_row(table: QTableWidget) -> None:
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(""))
            table.setItem(r, 1, QTableWidgetItem(""))

        def _add_tf_row() -> None:
            r = table_tf.rowCount()
            table_tf.insertRow(r)
            table_tf.setItem(r, 0, QTableWidgetItem(""))
            for i in range(1, 5):
                cb = QComboBox()
                cb.addItem("-", None)
                cb.addItem("Đúng", True)
                cb.addItem("Sai", False)
                table_tf.setCellWidget(r, i, cb)

        def _remove_selected_row(table: QTableWidget) -> None:
            row = table.currentRow()
            if row >= 0:
                table.removeRow(row)

        form.addRow("Student ID", inp_sid)
        form.addRow("Exam Code", inp_code)
        splitter.addWidget(left)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.addWidget(QLabel("Ảnh bài làm"))
        zoom_row = QHBoxLayout()
        btn_zoom_out_dlg = QPushButton("-")
        btn_zoom_reset_dlg = QPushButton("100%")
        btn_zoom_in_dlg = QPushButton("+")
        for btn in [btn_zoom_out_dlg, btn_zoom_reset_dlg, btn_zoom_in_dlg]:
            btn.setMaximumWidth(52)
        zoom_row.addWidget(btn_zoom_out_dlg)
        zoom_row.addWidget(btn_zoom_reset_dlg)
        zoom_row.addWidget(btn_zoom_in_dlg)
        zoom_row.addStretch()
        right_lay.addLayout(zoom_row)
        preview = QLabel()
        preview.setAlignment(Qt.AlignCenter)
        pix = self.preview_source_pixmap if hasattr(self, "preview_source_pixmap") and not self.preview_source_pixmap.isNull() else QPixmap(str(Path(res.image_path)))
        base_pix = pix
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(False)
        preview_scroll.setAlignment(Qt.AlignCenter)
        preview_scroll.setWidget(preview)
        zoom_state = {"factor": 1.0}
        def _fit_preview_to_viewport() -> None:
            if base_pix.isNull():
                return
            viewport = preview_scroll.viewport().size()
            if viewport.width() <= 0 or viewport.height() <= 0:
                return
            scale_w = viewport.width() / max(1, base_pix.width())
            scale_h = viewport.height() / max(1, base_pix.height())
            zoom_state["factor"] = max(0.2, min(1.0, min(scale_w, scale_h)))
            _apply_preview_zoom()

        def _apply_preview_zoom() -> None:
            if base_pix.isNull():
                preview.setPixmap(QPixmap())
                return
            scaled = base_pix.scaled(base_pix.size() * zoom_state["factor"], Qt.KeepAspectRatio, Qt.SmoothTransformation)
            preview.setPixmap(scaled)
            btn_zoom_reset_dlg.setText(f"{int(zoom_state['factor'] * 100)}%")
        btn_zoom_out_dlg.clicked.connect(lambda: (zoom_state.__setitem__("factor", max(0.2, zoom_state["factor"] / 1.2)), _apply_preview_zoom()))
        btn_zoom_in_dlg.clicked.connect(lambda: (zoom_state.__setitem__("factor", min(5.0, zoom_state["factor"] * 1.2)), _apply_preview_zoom()))
        btn_zoom_reset_dlg.clicked.connect(lambda: (zoom_state.__setitem__("factor", 1.0), _apply_preview_zoom()))
        right_lay.addWidget(preview_scroll, 1)
        splitter.addWidget(right)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        lay.addWidget(splitter)
        QTimer.singleShot(0, lambda s=splitter: s.setSizes([max(1, s.width() // 2), max(1, s.width() // 2)]))
        QTimer.singleShot(0, _fit_preview_to_viewport)
        left_lay.addLayout(form)
        left_lay.addWidget(QLabel("MCQ"))
        left_lay.addWidget(mcq_widget)
        row_mcq = QHBoxLayout()
        row_mcq.addWidget(QLabel("Nhập đáp án trực tiếp vào từng ô MCQ"))
        row_mcq.addStretch()
        left_lay.addLayout(row_mcq)

        left_lay.addWidget(QLabel("True / False"))
        left_lay.addWidget(table_tf)
        row_tf = QHBoxLayout()
        btn_add_tf = QPushButton("Thêm dòng TF")
        btn_del_tf = QPushButton("Xoá dòng chọn")
        btn_add_tf.clicked.connect(_add_tf_row)
        btn_del_tf.clicked.connect(lambda: _remove_selected_row(table_tf))
        row_tf.addWidget(btn_add_tf)
        row_tf.addWidget(btn_del_tf)
        row_tf.addStretch()
        left_lay.addLayout(row_tf)

        left_lay.addWidget(QLabel("Numeric"))
        left_lay.addWidget(table_num)
        row_num = QHBoxLayout()
        btn_add_num = QPushButton("Thêm dòng Numeric")
        btn_del_num = QPushButton("Xoá dòng chọn")
        btn_add_num.clicked.connect(lambda: _add_pair_row(table_num))
        btn_del_num.clicked.connect(lambda: _remove_selected_row(table_num))
        row_num.addWidget(btn_add_num)
        row_num.addWidget(btn_del_num)
        row_num.addStretch()
        left_lay.addLayout(row_num)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        left_lay.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        changes: list[str] = []
        new_sid = str(inp_sid.currentData() or inp_sid.currentText() or "").strip()
        new_code = str(inp_code.currentData() or inp_code.currentText() or "").strip()
        if new_sid != (res.student_id or ""):
            old_sid = res.student_id or ""
            res.student_id = new_sid
            sid_item = QTableWidgetItem(new_sid or "-")
            sid_item.setData(Qt.UserRole, str(res.image_path))
            sid_item.setData(Qt.UserRole + 1, new_code)
            sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(res))
            self.scan_list.setItem(idx, 0, sid_item)
            changes.append(f"student_id: '{old_sid}' -> '{new_sid}'")
        if new_code != (res.exam_code or ""):
            old_code = res.exam_code or ""
            res.exam_code = new_code
            changes.append(f"exam_code: '{old_code}' -> '{new_code}'")

        try:
            new_mcq_answers: dict[int, str] = {}
            for q_no, edit in mcq_edits.items():
                v_text = str(edit.text() if edit else "").strip().upper()[:1]
                if v_text:
                    new_mcq_answers[int(q_no)] = v_text

            new_numeric_answers: dict[int, str] = {}
            for r in range(table_num.rowCount()):
                q_item = table_num.item(r, 0)
                v_item = table_num.item(r, 1)
                q_text = str(q_item.text() if q_item else "").strip()
                v_text = str(v_item.text() if v_item else "").strip()
                if not q_text and not v_text:
                    continue
                if not q_text.lstrip("-").isdigit():
                    raise ValueError(f"Numeric dòng {r+1}: Câu phải là số nguyên.")
                if not v_text:
                    continue
                new_numeric_answers[int(q_text)] = v_text

            new_tf_answers: dict[int, dict[str, bool]] = {}
            labels = ["a", "b", "c", "d"]
            for r in range(table_tf.rowCount()):
                q_item = table_tf.item(r, 0)
                q_text = str(q_item.text() if q_item else "").strip()
                if not q_text:
                    continue
                if not q_text.lstrip("-").isdigit():
                    raise ValueError(f"TF dòng {r+1}: Câu phải là số nguyên.")
                q = int(q_text)
                flags: dict[str, bool] = {}
                for i, key in enumerate(labels, start=1):
                    cb = table_tf.cellWidget(r, i)
                    if not isinstance(cb, QComboBox):
                        continue
                    val = cb.currentData()
                    if isinstance(val, bool):
                        flags[key] = val
                if flags:
                    new_tf_answers[q] = flags

            if new_mcq_answers != (res.mcq_answers or {}):
                res.mcq_answers = new_mcq_answers
                changes.append("mcq_answers updated")
            if new_tf_answers != (res.true_false_answers or {}):
                res.true_false_answers = new_tf_answers
                changes.append("true_false_answers updated")
            if new_numeric_answers != (res.numeric_answers or {}):
                res.numeric_answers = new_numeric_answers
                changes.append("numeric_answers updated")
        except Exception as exc:
            QMessageBox.warning(self, "Dữ liệu không hợp lệ", str(exc))
            return

        if changes:
            self._refresh_student_profile_for_result(res, idx)
            scoped = self._scoped_result_copy(res)
            self.scan_blank_summary[idx] = self._compute_blank_questions(scoped)
            self.scan_list.setItem(idx, 3, QTableWidgetItem(self._build_recognition_content_text(res, self.scan_blank_summary[idx])))
            sid_item = self.scan_list.item(idx, 0)
            if sid_item:
                sid_item.setData(Qt.UserRole + 1, res.exam_code or "")
                sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(res))
            self._record_adjustment(idx, changes, "dialog_edit")
            self._persist_single_scan_result_to_db(res, note="dialog_edit")
            self._refresh_all_statuses()
            self._update_scan_preview(idx)
            self._load_selected_result_for_correction()
            self.btn_save_batch_subject.setEnabled(True)
            invalidated = self._invalidate_scoring_for_student_ids(
                [old_sid_for_score, str(res.student_id or "").strip()],
                reason="dialog_edit",
            )
            if invalidated > 0:
                QMessageBox.information(
                    self,
                    "Tính điểm",
                    f"Đã đánh dấu {invalidated} bản ghi cần chấm lại do sửa bài. Vui lòng chạy lại Tính điểm.",
                )

    def apply_manual_correction(self) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= len(self.scan_results):
            QMessageBox.warning(self, "No selection", "Select a scanned result first.")
            return
        txt = self.manual_edit.toPlainText().strip()
        if not txt:
            return
        try:
            patch = json.loads(txt)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid JSON", f"Cannot parse manual correction:\n{exc}")
            return

        res = self.scan_results[idx]
        old_sid_for_score = str(res.student_id or "").strip()
        changes: list[str] = []
        if "student_id" in patch:
            new_sid = str(patch["student_id"])
            if new_sid != (res.student_id or ""):
                changes.append(f"student_id: '{res.student_id or ''}' -> '{new_sid}'")
            res.student_id = new_sid
        if "exam_code" in patch:
            new_code = str(patch["exam_code"])
            if new_code != (res.exam_code or ""):
                changes.append(f"exam_code: '{res.exam_code or ''}' -> '{new_code}'")
            res.exam_code = new_code
        if isinstance(patch.get("mcq_answers"), dict):
            new_mcq_answers = {int(k): str(v) for k, v in patch["mcq_answers"].items()}
            if new_mcq_answers != (res.mcq_answers or {}):
                res.mcq_answers = new_mcq_answers
                changes.append("mcq_answers updated")
        if isinstance(patch.get("numeric_answers"), dict):
            new_numeric_answers = {int(k): str(v) for k, v in patch["numeric_answers"].items()}
            if new_numeric_answers != (res.numeric_answers or {}):
                res.numeric_answers = new_numeric_answers
                changes.append("numeric_answers updated")
        if isinstance(patch.get("true_false_answers"), dict):
            new_tf_answers = patch["true_false_answers"]
            if new_tf_answers != (res.true_false_answers or {}):
                res.true_false_answers = new_tf_answers
                changes.append("true_false_answers updated")

        sid = (res.student_id or "").strip() or "-"
        sid_item = QTableWidgetItem(sid)
        sid_item.setData(Qt.UserRole, str(res.image_path))
        sid_item.setData(Qt.UserRole + 1, res.exam_code or "")
        sid_item.setData(Qt.UserRole + 2, self._short_recognition_text_for_result(res))
        self.scan_list.setItem(idx, 0, sid_item)
        if changes:
            self._refresh_student_profile_for_result(res, idx)
            scoped = self._scoped_result_copy(res)
            self.scan_blank_summary[idx] = self._compute_blank_questions(scoped)
            self.scan_list.setItem(idx, 3, QTableWidgetItem(self._build_recognition_content_text(res, self.scan_blank_summary[idx])))
            self._record_adjustment(idx, changes, "manual_json")
            self._persist_single_scan_result_to_db(res, note="manual_json")
            self.btn_save_batch_subject.setEnabled(False)
            invalidated = self._invalidate_scoring_for_student_ids(
                [old_sid_for_score, str(res.student_id or "").strip()],
                reason="manual_json",
            )
            if invalidated > 0:
                QMessageBox.information(
                    self,
                    "Tính điểm",
                    f"Đã đánh dấu {invalidated} bản ghi cần chấm lại do sửa bài. Vui lòng chạy lại Tính điểm.",
                )
        self._refresh_all_statuses()
        self._update_scan_preview(idx)
        self._load_selected_result_for_correction()
        QMessageBox.information(self, "Correction", "Manual correction applied to selected scan.")

    def calculate_scores(self, subject_key: str = "", mode: str = "Tính lại toàn bộ", note: str = "") -> list:
        subject = (subject_key or self._resolve_preferred_scoring_subject() or "General").strip()
        subject_scans = self._refresh_scan_results_from_db(subject) or self.scan_results_by_subject.get(subject, [])
        if not subject_scans:
            subject_scans = self._cached_subject_scans_from_config(subject)
            if subject_scans:
                self.scan_results_by_subject[subject] = list(subject_scans)
        cfg = self._selected_batch_subject_config()
        current_key = self._subject_key_from_cfg(cfg) if cfg else ""
        if current_key == subject and hasattr(self, "scan_list") and self.scan_list.rowCount() > 0:
            subject_scans = self._current_scan_results_snapshot()
            self.scan_results = list(subject_scans)
            self.scan_results_by_subject[subject] = list(subject_scans)
        elif not subject_scans and self.scan_results:
            if current_key == subject:
                subject_scans = list(self.scan_results)
                self.scan_results_by_subject[subject] = list(subject_scans)
        if not subject_scans:
            QMessageBox.warning(self, "Missing data", "Môn này chưa có dữ liệu Batch Scan để tính điểm.")
            return []

        self._ensure_answer_keys_for_subject(subject)
        if not self.answer_keys:
            QMessageBox.warning(self, "Missing data", "Không tìm thấy đáp án cho môn đã chọn. Vui lòng kiểm tra cấu hình môn.")
            return []
        subject_cfg = self._subject_config_by_subject_key(subject) or {}
        mode_text = (mode or "Tính lại toàn bộ").strip()
        prev_subject_scores = self.scoring_results_by_subject.get(subject, {})
        rows = []
        missing = 0
        failed_scans: list[dict[str, str]] = []
        for scan in subject_scans:
            sid = (scan.student_id or "").strip()
            profile = self._student_profile_by_id(sid)
            if profile.get("name") and not str(getattr(scan, "full_name", "") or "").strip():
                setattr(scan, "full_name", profile.get("name"))
            if profile.get("birth_date") and not str(getattr(scan, "birth_date", "") or "").strip():
                setattr(scan, "birth_date", profile.get("birth_date"))
            if mode_text == "Chỉ tính bài chưa có điểm" and sid and sid in prev_subject_scores:
                continue
            key = self.answer_keys.get_flexible(subject, scan.exam_code)
            if not key:
                missing += 1
                failed_scans.append({
                    "file": str(getattr(scan, "image_path", "") or "-"),
                    "reason": f"Thiếu đáp án cho mã đề '{str(getattr(scan, 'exam_code', '') or '').strip() or '-'}'",
                })
                continue
            try:
                rows.append(
                    self.scoring_engine.score(
                        scan,
                        key,
                        student_name=str(getattr(scan, "full_name", "") or ""),
                        subject_config=subject_cfg,
                    )
                )
            except Exception as exc:
                failed_scans.append({
                    "file": str(getattr(scan, "image_path", "") or "-"),
                    "reason": f"Lỗi chấm điểm: {exc}",
                })

        self.score_rows = rows
        self.score_preview_table.setRowCount(0)
        for i, r in enumerate(rows):
            self.score_preview_table.insertRow(i)
            self.score_preview_table.setItem(i, 0, QTableWidgetItem(r.student_id or "-"))
            self.score_preview_table.setItem(i, 1, QTableWidgetItem(r.name or "-"))
            self.score_preview_table.setItem(i, 2, QTableWidgetItem(r.subject))
            self.score_preview_table.setItem(i, 3, QTableWidgetItem(r.exam_code))
            self.score_preview_table.setItem(i, 4, QTableWidgetItem(str(getattr(r, "mcq_correct", 0))))
            self.score_preview_table.setItem(i, 5, QTableWidgetItem(str(getattr(r, "tf_correct", 0))))
            self.score_preview_table.setItem(i, 6, QTableWidgetItem(str(getattr(r, "numeric_correct", 0))))
            self.score_preview_table.setItem(i, 7, QTableWidgetItem(str(r.correct)))
            self.score_preview_table.setItem(i, 8, QTableWidgetItem(str(r.wrong)))
            self.score_preview_table.setItem(i, 9, QTableWidgetItem(str(r.blank)))
            self.score_preview_table.setItem(i, 10, QTableWidgetItem(str(getattr(r, "bonus_full_credit_count", 0))))
            self.score_preview_table.setItem(i, 11, QTableWidgetItem(str(getattr(r, "bonus_full_credit_points", 0.0))))
            self.score_preview_table.setItem(i, 12, QTableWidgetItem(str(r.score)))
            self.score_preview_table.setItem(i, 13, QTableWidgetItem(str(getattr(r, "mcq_compare", ""))))
            self.score_preview_table.setItem(i, 14, QTableWidgetItem(str(getattr(r, "tf_compare", ""))))
            self.score_preview_table.setItem(i, 15, QTableWidgetItem(str(getattr(r, "numeric_compare", ""))))

        phase = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "subject": subject,
            "mode": mode_text,
            "count": len(rows),
            "missing": missing,
            "failed_count": len(failed_scans),
            "success_count": len(rows),
            "failed_scans": list(failed_scans),
            "note": note,
        }
        phase_marker = f"{phase['timestamp']}::{subject}::{mode_text}"

        subject_scores = dict(prev_subject_scores)
        for r in rows:
            sid_key = (r.student_id or "").strip()
            if sid_key:
                subject_scores[sid_key] = {
                    "student_id": r.student_id,
                    "name": r.name,
                    "subject": r.subject,
                    "exam_code": r.exam_code,
                    "mcq_correct": getattr(r, "mcq_correct", 0),
                    "tf_correct": getattr(r, "tf_correct", 0),
                    "numeric_correct": getattr(r, "numeric_correct", 0),
                    "tf_compare": getattr(r, "tf_compare", ""),
                    "numeric_compare": getattr(r, "numeric_compare", ""),
                    "correct": r.correct,
                    "wrong": r.wrong,
                    "blank": r.blank,
                    "score": r.score,
                    "phase": phase_marker,
                    "phase_timestamp": phase["timestamp"],
                    "phase_mode": mode_text,
                }
        self.scoring_results_by_subject[subject] = subject_scores
        phase["phase_marker"] = phase_marker
        self.scoring_phases.append(phase)
        if len(self.scoring_phases) > 500:
            self.scoring_phases = self.scoring_phases[-500:]
        if self.session:
            cfg = dict(self.session.config or {})
            cfg["scoring_phases"] = list(self.scoring_phases)
            cfg["scoring_results"] = dict(self.scoring_results_by_subject)
            cfg["last_scoring_phase"] = dict(phase)
            self.session.config = cfg
            self.session_dirty = True
            if not self._persist_session_quietly():
                QMessageBox.warning(self, "Scoring", "Không thể tự động lưu kết quả chấm điểm. Vui lòng dùng nút Lưu kỳ thi.")
        self._refresh_scoring_phase_table()
        self._refresh_dashboard_summary_from_db(subject)

        formula_text = ""
        if rows:
            first_scan = subject_scans[0] if subject_scans else None
            first_key = self.answer_keys.get_flexible(subject, first_scan.exam_code) if first_scan and self.answer_keys else None
            if first_key:
                formula_text = self.scoring_engine.describe_formula(first_key, subject_cfg)
        total_scans = len(subject_scans)
        fail_count = len(failed_scans)
        success_count = len(rows)
        base_msg = (
            f"Đã chấm xong môn '{subject}'.\n"
            f"Tổng file: {total_scans} | Thành công: {success_count} | Thất bại: {fail_count}."
        )
        if formula_text:
            base_msg = f"{base_msg}\n\n{formula_text}"
        if failed_scans:
            details = []
            for item in failed_scans[:30]:
                details.append(f"- {Path(str(item.get('file', '-') or '-')).name}: {str(item.get('reason', '') or '-')}")
            if len(failed_scans) > 30:
                details.append(f"... và {len(failed_scans)-30} file khác")
            base_msg = f"{base_msg}\n\nFile thất bại:\n" + "\n".join(details)
        QMessageBox.information(self, "Scoring preview", base_msg)
        return rows

    def _export_student_subject_matrix_excel(self, output_path: Path) -> None:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "scores_by_student"

        subjects = sorted(set(str(k or "").strip() for k in (self.scoring_results_by_subject or {}).keys() if str(k or "").strip()))
        header = ["Student ID", "Name"] + subjects
        ws.append(header)

        base_students: list[tuple[str, str]] = []
        seen: set[str] = set()
        if self.session:
            for st in (self.session.students or []):
                sid = str(getattr(st, "student_id", "") or "").strip()
                name = str(getattr(st, "name", "") or "").strip()
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                base_students.append((sid, name))

        for subject in subjects:
            for sid, row in (self.scoring_results_by_subject.get(subject, {}) or {}).items():
                sid_text = str(sid or "").strip()
                if not sid_text or sid_text in seen:
                    continue
                seen.add(sid_text)
                base_students.append((sid_text, str((row or {}).get("name", "") or "")))

        for sid, name in base_students:
            vals = [sid, name]
            for subject in subjects:
                row = (self.scoring_results_by_subject.get(subject, {}) or {}).get(sid, {}) or {}
                vals.append(row.get("score", ""))
            ws.append(vals)

        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                max_len = max(max_len, len(str(cell.value or "")))
            ws.column_dimensions[col_letter].width = min(40, max(12, max_len + 2))

        wb.save(output_path)

    def export_results(self) -> None:
        rows = self.score_rows or self.calculate_scores()
        if not rows:
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not output_dir:
            return
        self.scoring_engine.export_csv(rows, Path(output_dir) / "results.csv")
        self.scoring_engine.export_json(rows, Path(output_dir) / "results.json")
        self.scoring_engine.export_xml(rows, Path(output_dir) / "results.xml")
        self.scoring_engine.export_excel(rows, Path(output_dir) / "results.xlsx")
        try:
            self._export_student_subject_matrix_excel(Path(output_dir) / "scores_by_student.xlsx")
            QMessageBox.information(self, "Export", "Exported CSV, JSON, XML, XLSX, scores_by_student.xlsx.")
        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Đã export CSV/JSON/XML/XLSX nhưng không tạo được scores_by_student.xlsx:\n{exc}")

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



def run() -> None:
    bootstrap_application_db()
    app = QApplication([])
    window = MainWindow()
    window.showMaximized()
    app.exec()


if __name__ == "__main__":
    run()
