from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
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
)

from core.answer_key_importer import import_answer_key
from core.omr_engine import OMRProcessor
from core.scoring_engine import ScoringEngine
from editor.template_editor import TemplateEditorWindow
from gui.import_answer_key_dialog import ImportAnswerKeyDialog
from models.answer_key import AnswerKeyRepository, SubjectKey
from models.exam_session import ExamSession
from models.template import Template


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
                c = int(z.metadata.get("questions_per_block", 0) or z.grid.question_count or (z.grid.rows // max(1, int(z.metadata.get("statements_per_question", 4)))) or 0)
                counts["TF"] += max(0, c)
            elif z.zone_type.value == "NUMERIC_BLOCK":
                c = int(z.metadata.get("questions_per_block", z.metadata.get("total_questions", 0)) or z.grid.question_count or 0)
                counts["NUMERIC"] += max(0, c)
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
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Cấu hình môn học")
        data = data or {}
        subject_options = subject_options or []
        block_options = block_options or ["10", "11", "12"]

        self.common_template_path = common_template_path
        self.paper_part_count_default = paper_part_count
        self.answer_key_data: dict = dict(data.get("imported_answer_keys", {}))

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

        row_tpl = QHBoxLayout(); row_tpl.addWidget(self.template_path); b_tpl = QPushButton("..."); row_tpl.addWidget(b_tpl)
        b_tpl.clicked.connect(self._browse_template)
        row_scan = QHBoxLayout(); row_scan.addWidget(self.scan_folder); b_scan = QPushButton("..."); row_scan.addWidget(b_scan)
        b_scan.clicked.connect(self._browse_scan_folder)
        row_key = QHBoxLayout(); row_key.addWidget(self.answer_key); b_key = QPushButton("..."); row_key.addWidget(b_key)
        b_key.clicked.connect(self._browse_answer_key)

        form.addRow("Tên môn", self.subject_name)
        form.addRow("Khối", self.block_name)
        form.addRow("Giấy thi riêng (tùy chọn)", row_tpl)
        form.addRow("Thư mục bài thi môn", row_scan)
        form.addRow("Đáp án môn", row_key)
        form.addRow("Mã đáp án môn_khối", self.answer_key_key)
        form.addRow("Các mã đề của môn", self.answer_codes)
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

        lay.addLayout(form)
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
        self._update_paper_parts()
        self._refresh_score_mode_ui()

        # Lock window size to avoid resize jumps when toggling score mode.
        self.section_group.setVisible(True)
        self.question_group.setVisible(True)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())
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
            self.template_path.setText(path)

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
            QMessageBox.warning(self, "Import đáp án", f"Không thể import đáp án:\n{exc}")
            return

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
            }
        self.answer_codes.setText(", ".join(sorted(self.answer_key_data.keys())))
        self.answer_key.setText(path)
        QMessageBox.information(self, "Import đáp án", "Đã gắn toàn bộ mã đề của file đáp án cho môn đang cấu hình.")

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
    def __init__(self, subject_options: list[str], block_options: list[str], data: dict | None = None, parent=None):
        super().__init__(parent)
        data = data or {}
        self.setWindowTitle("Sửa kỳ thi" if data else "Tạo kỳ thi mới")
        self.resize(860, 640)
        self.subject_configs: list[dict] = list(data.get("subject_configs", []))
        self.subject_options = subject_options
        self.block_options = block_options

        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.exam_name = QLineEdit(str(data.get("exam_name", "")))
        self.common_template = QLineEdit(str(data.get("common_template", "")))
        self.scan_root = QLineEdit(str(data.get("scan_root", "")))
        self.scan_mode = QComboBox(); self.scan_mode.addItems(["Ảnh trong thư mục gốc", "Ảnh theo phòng thi (thư mục con)"])
        self.scan_mode.setCurrentText(str(data.get("scan_mode", "Ảnh trong thư mục gốc")))
        self.paper_part_count = QComboBox(); self.paper_part_count.addItems(["1", "2", "3", "4", "5"]); self.paper_part_count.setCurrentText(str(data.get("paper_part_count", "3")))

        row_tpl = QHBoxLayout(); row_tpl.addWidget(self.common_template); btn_tpl = QPushButton("..."); row_tpl.addWidget(btn_tpl)
        btn_tpl.clicked.connect(self._browse_common_template)
        row_scan = QHBoxLayout(); row_scan.addWidget(self.scan_root); btn_scan = QPushButton("..."); row_scan.addWidget(btn_scan)
        btn_scan.clicked.connect(self._browse_scan_root)

        form.addRow("Tên kỳ thi", self.exam_name)
        form.addRow("Giấy thi dùng chung", row_tpl)
        form.addRow("Thư mục gốc bài thi", row_scan)
        form.addRow("Cơ chế thư mục bài thi", self.scan_mode)
        form.addRow("Số phần trên giấy thi", self.paper_part_count)
        lay.addLayout(form)

        lay.addWidget(QLabel("Các môn trong kỳ thi"))
        self.subject_list = QListWidget()
        lay.addWidget(self.subject_list)

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

    def _browse_common_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn giấy thi dùng chung", "", "JSON (*.json)")
        if path:
            self.common_template.setText(path)

    def _browse_scan_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục gốc bài thi")
        if path:
            self.scan_root.setText(path)

    def _refresh_subject_list(self) -> None:
        self.subject_list.clear()
        for cfg in self.subject_configs:
            tpl = cfg.get("template_path") or "[dùng mẫu chung]"
            key = cfg.get("answer_key_key", "")
            mode = cfg.get("score_mode", "Điểm theo phần")
            total = cfg.get("total_exam_points", "-")
            codes = ",".join(sorted((cfg.get("imported_answer_keys") or {}).keys()))
            self.subject_list.addItem(f"{cfg.get('name','')} - Khối {cfg.get('block','')} — Key: {key} — Mã đề: {codes or '-'} — {mode} — Tổng:{total} — Template: {tpl}")

    def _add_subject(self) -> None:
        dlg = SubjectConfigDialog(
            subject_options=self.subject_options,
            block_options=self.block_options,
            paper_part_count=int(self.paper_part_count.currentText()),
            common_template_path=self.common_template.text().strip(),
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self.subject_configs.append(dlg.payload())
        self._refresh_subject_list()

    def _edit_subject(self) -> None:
        idx = self.subject_list.currentRow()
        if idx < 0 or idx >= len(self.subject_configs):
            return
        dlg = SubjectConfigDialog(
            self.subject_configs[idx],
            subject_options=self.subject_options,
            block_options=self.block_options,
            paper_part_count=int(self.paper_part_count.currentText()),
            common_template_path=self.common_template.text().strip(),
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self.subject_configs[idx] = dlg.payload()
        self._refresh_subject_list()

    def _delete_subject(self) -> None:
        idx = self.subject_list.currentRow()
        if idx < 0 or idx >= len(self.subject_configs):
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
        self.accept()

    def payload(self) -> dict:
        return {
            "exam_name": self.exam_name.text().strip(),
            "common_template": self.common_template.text().strip(),
            "scan_root": self.scan_root.text().strip(),
            "scan_mode": self.scan_mode.currentText(),
            "paper_part_count": int(self.paper_part_count.currentText()),
            "subject_configs": self.subject_configs,
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Exam Grading System")
        self.resize(1200, 800)

        self.session: ExamSession | None = None
        self.template: Template | None = None
        self.answer_keys: AnswerKeyRepository | None = None
        self.scan_results = []
        self.scan_files: list[Path] = []
        self.scan_blank_questions: dict[int, list[int]] = {}
        self.scan_blank_summary: dict[int, dict[str, list[int]]] = {}
        self.scan_manual_adjustments: dict[int, list[str]] = {}
        self.scan_edit_history: dict[int, list[str]] = {}
        self.scan_last_adjustment: dict[int, str] = {}
        self.score_rows = []
        self.imported_exam_codes: list[str] = []
        self.subject_catalog: list[str] = ["Toán", "Ngữ văn", "Tiếng Anh", "Vật lý", "Hóa học", "Sinh học"]
        self.block_catalog: list[str] = ["10", "11", "12"]

        self.omr_processor = OMRProcessor()
        self.scoring_engine = ScoringEngine()
        self.current_session_path: Path | None = None
        self.session_dirty = False

        self.session_registry_path = Path.home() / ".omr_exam_sessions.json"
        self.session_registry: list[dict[str, str | bool]] = self._load_session_registry()

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_exam_list_page())
        self.stack.addWidget(self._build_workspace_page())
        self.setCentralWidget(self.stack)

        self._build_menu()
        self._refresh_exam_list()
        self.stack.setCurrentIndex(0)

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

    def _load_session_registry(self) -> list[dict[str, str | bool]]:
        if not self.session_registry_path.exists():
            return []
        try:
            data = json.loads(self.session_registry_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict) and x.get("path")]
        except Exception:
            return []
        return []

    def _save_session_registry(self) -> None:
        self.session_registry_path.write_text(json.dumps(self.session_registry, ensure_ascii=False, indent=2), encoding="utf-8")

    def _upsert_session_registry(self, path: Path, name: str | None = None) -> None:
        p = str(path)
        for row in self.session_registry:
            if row.get("path") == p:
                row["name"] = name or row.get("name") or path.stem
                return
        self.session_registry.append({"name": name or path.stem, "path": p, "default": False})

    def _build_exam_list_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Danh sách các kỳ thi"))

        self.exam_list_table = QTableWidget(0, 6)
        self.exam_list_table.setHorizontalHeaderLabels(["STT", "Tên kỳ thi", "Số môn", "Thư mục quét", "Môn học", "Đường dẫn"])
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
        hdr.setSectionResizeMode(5, QHeaderView.Stretch)
        layout.addWidget(self.exam_list_table)

        row = QHBoxLayout()
        btn_open = QPushButton("Mở")
        btn_open.clicked.connect(self._open_selected_registry_session)
        btn_edit = QPushButton("Sửa")
        btn_edit.clicked.connect(self._edit_selected_registry_session)
        btn_delete = QPushButton("Xoá")
        btn_delete.clicked.connect(self._delete_selected_registry_session)
        btn_default = QPushButton("Đặt mặc định")
        btn_default.clicked.connect(self._set_default_selected_registry_session)
        btn_new = QPushButton("Tạo kỳ thi mới")
        btn_new.clicked.connect(self.action_create_session)
        for b in [btn_open, btn_edit, btn_delete, btn_default, btn_new]:
            row.addWidget(b)
        layout.addLayout(row)
        return w

    def _build_workspace_page(self) -> QWidget:
        central = QWidget()
        root_layout = QVBoxLayout(central)

        group_session = QGroupBox("Exam Session")
        l1 = QVBoxLayout(group_session); l1.addWidget(self._build_session_tab())
        group_scan = QGroupBox("OMR Scan")
        l2 = QVBoxLayout(group_scan); l2.addWidget(self._build_scan_tab())
        group_correction = QGroupBox("Error Correction")
        l3 = QVBoxLayout(group_correction); l3.addWidget(self._build_correction_tab())

        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(group_session)
        main_split.addWidget(group_scan)
        main_split.addWidget(group_correction)
        main_split.setSizes([220, 420, 260])

        root_layout.addWidget(main_split)
        return central

    def _refresh_exam_list(self) -> None:
        self.exam_list_table.setRowCount(len(self.session_registry))
        for idx, row in enumerate(self.session_registry):
            name = str(row.get("name") or Path(str(row.get("path"))).stem)
            suffix = " [MẶC ĐỊNH]" if bool(row.get("default")) else ""
            path = Path(str(row.get("path", "")))
            subject_text = "-"
            subject_count = "0"
            scan_root = "-"
            if path.exists():
                try:
                    ses = ExamSession.load_json(path)
                    cfg = ses.config or {}
                    subject_cfgs = cfg.get("subject_configs", []) if isinstance(cfg.get("subject_configs", []), list) else []
                    subject_count = str(len(subject_cfgs))
                    subject_text = ", ".join(f"{x.get('name','?')}-{x.get('block','?')}" for x in subject_cfgs[:4])
                    if len(subject_cfgs) > 4:
                        subject_text += f" ...(+{len(subject_cfgs)-4})"
                    scan_root = str(cfg.get("scan_root", "") or "-")
                except Exception:
                    pass
            self.exam_list_table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            path_item = QTableWidgetItem(f"{name}{suffix}")
            path_item.setData(Qt.UserRole, row.get("path"))
            self.exam_list_table.setItem(idx, 1, path_item)
            self.exam_list_table.setItem(idx, 2, QTableWidgetItem(subject_count))
            self.exam_list_table.setItem(idx, 3, QTableWidgetItem(scan_root))
            self.exam_list_table.setItem(idx, 4, QTableWidgetItem(subject_text or "-"))
            self.exam_list_table.setItem(idx, 5, QTableWidgetItem(str(row.get("path", ""))))
        self.exam_list_table.resizeRowsToContents()

    def _selected_registry_path(self) -> Path | None:
        row = self.exam_list_table.currentRow()
        if row < 0:
            return None
        item = self.exam_list_table.item(row, 1)
        if not item:
            return None
        path = item.data(Qt.UserRole)
        return Path(path) if path else None

    def _open_selected_registry_session(self) -> None:
        path = self._selected_registry_path()
        if not path:
            QMessageBox.warning(self, "Mở kỳ thi", "Chọn kỳ thi trong danh sách trước.")
            return
        if not self._confirm("Mở kỳ thi", f"Bạn có chắc muốn mở kỳ thi này?\n{path}"):
            return
        self._open_session_path(path)

    def _edit_selected_registry_session(self) -> None:
        path = self._selected_registry_path()
        if not path:
            QMessageBox.warning(self, "Sửa kỳ thi", "Chọn kỳ thi trong danh sách trước.")
            return
        if not self._confirm("Sửa kỳ thi", f"Bạn có chắc muốn sửa kỳ thi này?\n{path}"):
            return
        try:
            session = ExamSession.load_json(path)
            cfg = session.config or {}
            payload = {
                "exam_name": session.exam_name,
                "common_template": session.template_path,
                "scan_root": cfg.get("scan_root", ""),
                "scan_mode": cfg.get("scan_mode", "Ảnh trong thư mục gốc"),
                "paper_part_count": cfg.get("paper_part_count", 3),
                "subject_configs": cfg.get("subject_configs", []),
            }
            dlg = NewExamDialog(self.subject_catalog, self.block_catalog, data=payload, parent=self)
            if dlg.exec() != QDialog.Accepted:
                return
            edited = dlg.payload()
            session.exam_name = edited.get("exam_name", session.exam_name)
            session.template_path = edited.get("common_template", session.template_path)
            session.subjects = [
                f"{str(x.get('name', '')).strip()}_{str(x.get('block', '')).strip()}"
                for x in edited.get("subject_configs", [])
                if str(x.get("name", "")).strip()
            ] or session.subjects
            session.config = {
                **(session.config or {}),
                "scan_mode": edited.get("scan_mode", "Ảnh trong thư mục gốc"),
                "scan_root": edited.get("scan_root", ""),
                "paper_part_count": edited.get("paper_part_count", 3),
                "subject_configs": edited.get("subject_configs", []),
                "subject_catalog": self.subject_catalog,
                "block_catalog": self.block_catalog,
            }
            session.save_json(path)
            self._upsert_session_registry(path, session.exam_name)
            self._save_session_registry()
            self._refresh_exam_list()
            if self.current_session_path and self.current_session_path == path:
                self.session = session
                self._refresh_session_info()
            QMessageBox.information(self, "Sửa kỳ thi", "Đã cập nhật thông số kỳ thi.")
        except Exception as exc:
            QMessageBox.warning(self, "Sửa kỳ thi", f"Không thể sửa kỳ thi:\n{exc}")

    def _delete_selected_registry_session(self) -> None:
        path = self._selected_registry_path()
        if not path:
            QMessageBox.warning(self, "Xoá kỳ thi", "Chọn kỳ thi trong danh sách trước.")
            return
        if not self._confirm("Xoá kỳ thi", f"Bạn có chắc muốn xoá kỳ thi khỏi danh sách?\n{path}"):
            return
        self.session_registry = [x for x in self.session_registry if x.get("path") != str(path)]
        self._save_session_registry()
        self._refresh_exam_list()

    def _set_default_selected_registry_session(self) -> None:
        path = self._selected_registry_path()
        if not path:
            QMessageBox.warning(self, "Đặt mặc định", "Chọn kỳ thi trong danh sách trước.")
            return
        if not self._confirm("Đặt mặc định", f"Đặt kỳ thi này làm mặc định?\n{path}"):
            return
        for row in self.session_registry:
            row["default"] = row.get("path") == str(path)
        self._save_session_registry()
        self._refresh_exam_list()

    def _open_session_path(self, path: Path) -> None:
        try:
            self.session = ExamSession.load_json(path)
            self.current_session_path = path
            if self.session.template_path:
                t = Path(self.session.template_path)
                if t.exists():
                    self.template = Template.load_json(t)
            cfg = self.session.config or {}
            self.subject_catalog = list(cfg.get("subject_catalog", self.subject_catalog)) or self.subject_catalog
            self.block_catalog = list(cfg.get("block_catalog", self.block_catalog)) or self.block_catalog
            if self.session.answer_key_path:
                p = Path(self.session.answer_key_path)
                if p.exists() and p.suffix.lower() == ".json":
                    self.answer_keys = AnswerKeyRepository.load_json(p)
                    self.imported_exam_codes = sorted({k.split("::", 1)[1] for k in self.answer_keys.keys.keys() if "::" in k})
            self._upsert_session_registry(path, self.session.exam_name if self.session else None)
            self._save_session_registry()
            self._refresh_exam_list()
            self.session_dirty = False
            self._refresh_session_info()
            self.stack.setCurrentIndex(1)
            QMessageBox.information(self, "Open session", "Đã mở kỳ thi thành công.")
        except Exception as exc:
            QMessageBox.warning(self, "Open session", f"Không thể mở kỳ thi:\n{exc}")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        act_new = file_menu.addAction("Tạo kỳ thi mới")
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self.action_create_session)

        act_open = file_menu.addAction("Mở kỳ thi cũ")
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
        act_manage_template = file_menu.addAction("Quản lý mẫu giấy thi")
        act_manage_template.triggered.connect(self.action_manage_template)

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

        toolbar = QToolBar("Ribbon")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        style = self.style()
        toolbar.addAction(style.standardIcon(QStyle.SP_FileIcon), "New", self.action_create_session)
        toolbar.addAction(style.standardIcon(QStyle.SP_DialogOpenButton), "Open", self.action_open_session)
        toolbar.addAction(style.standardIcon(QStyle.SP_DialogSaveButton), "Save", self.action_save_session)
        toolbar.addSeparator()
        toolbar.addAction(style.standardIcon(QStyle.SP_DialogOpenButton), "Load Template", self.action_load_template)
        toolbar.addAction(style.standardIcon(QStyle.SP_DialogOpenButton), "Load Keys", self.action_load_answer_keys)
        toolbar.addAction(style.standardIcon(QStyle.SP_DialogOpenButton), "Import Key", self.action_import_answer_key)
        toolbar.addAction(style.standardIcon(QStyle.SP_DriveFDIcon), "Export Sample", self.action_export_answer_key_sample)
        toolbar.addSeparator()
        toolbar.addAction(style.standardIcon(QStyle.SP_MediaPlay), "Batch Scan", self.action_run_batch_scan)
        toolbar.addAction(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Edit Selected", self.action_edit_selected_scan)
        toolbar.addAction(style.standardIcon(QStyle.SP_DialogApplyButton), "Apply Correction", self.action_apply_manual_correction)
        toolbar.addSeparator()
        toolbar.addAction(style.standardIcon(QStyle.SP_DialogApplyButton), "Calculate Scores", self.action_calculate_scores)
        toolbar.addAction(style.standardIcon(QStyle.SP_DialogSaveButton), "Export", self.action_export_results)

    def open_session(self) -> None:
        if self.session and self.session_dirty:
            if not self._confirm("Dữ liệu chưa lưu", "Kỳ thi hiện tại có thay đổi chưa lưu. Vẫn mở kỳ thi khác?"):
                return
        file_path, _ = QFileDialog.getOpenFileName(self, "Mở kỳ thi cũ", "", "Exam Session JSON (*.json)")
        if not file_path:
            return
        self._open_session_path(Path(file_path))

    def save_session(self) -> None:
        if not self.session:
            self.create_session()
        if not self.current_session_path:
            self.save_session_as()
            return
        try:
            self.session.save_json(self.current_session_path)
            self._upsert_session_registry(self.current_session_path, self.session.exam_name if self.session else None)
            self._save_session_registry()
            self._refresh_exam_list()
            self.session_dirty = False
            QMessageBox.information(self, "Save session", f"Đã lưu kỳ thi:\n{self.current_session_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save session", f"Không thể lưu kỳ thi:\n{exc}")

    def save_session_as(self) -> None:
        if not self.session:
            self.create_session()
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu kỳ thi", "exam_session.json", "Exam Session JSON (*.json)")
        if not file_path:
            return
        self.current_session_path = Path(file_path)
        if self.current_session_path.suffix.lower() != ".json":
            self.current_session_path = self.current_session_path.with_suffix(".json")
        self.save_session()

    def close_current_session(self) -> None:
        if self.session_dirty:
            if not self._confirm("Dữ liệu chưa lưu", "Kỳ thi hiện tại có thay đổi chưa lưu. Vẫn đóng?"):
                return
        self.session = None
        self.template = None
        self.answer_keys = None
        self.scan_results = []
        self.current_session_path = None
        self.session_dirty = False
        self.session_info.clear()
        self.exam_code_preview.setText("Mã đề trên phiếu trả lời mẫu: -")
        self.scan_list.setRowCount(0)
        self.score_preview_table.setRowCount(0)
        self.error_list.clear()
        self.result_preview.clear()
        self.scan_result_preview.setRowCount(0)
        self.manual_edit.clear()
        self.stack.setCurrentIndex(0)

    def manage_subjects(self) -> None:
        subjects_text = ", ".join(self.subject_catalog)
        blocks_text = ", ".join(self.block_catalog)
        text, ok = QInputDialog.getText(
            self,
            "Quản lý môn học",
            "Nhập danh sách môn (phân tách bằng dấu phẩy):",
            text=subjects_text,
        )
        if not ok:
            return
        block_input, ok2 = QInputDialog.getText(
            self,
            "Quản lý khối",
            "Nhập danh sách khối (phân tách bằng dấu phẩy):",
            text=blocks_text,
        )
        if not ok2:
            return

        subjects = [x.strip() for x in text.split(",") if x.strip()]
        blocks = [x.strip() for x in block_input.split(",") if x.strip()]
        if not subjects:
            QMessageBox.warning(self, "Quản lý môn học", "Danh sách môn học không được để trống.")
            return
        if not blocks:
            QMessageBox.warning(self, "Quản lý khối", "Danh sách khối không được để trống.")
            return

        self.subject_catalog = subjects
        self.block_catalog = blocks
        self.session_dirty = True
        QMessageBox.information(self, "Quản lý môn/khối", "Đã cập nhật danh sách môn và khối.")
        self._refresh_session_info()

    def action_create_session(self) -> None:
        if not self._confirm("Tạo kỳ thi mới", "Bạn có chắc muốn tạo kỳ thi mới?"):
            return
        dlg = NewExamDialog(self.subject_catalog, self.block_catalog, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.create_session(dlg.payload())
        self.stack.setCurrentIndex(1)

    def action_open_session(self) -> None:
        if self._confirm("Mở kỳ thi", "Bạn có chắc muốn mở kỳ thi?"):
            self.open_session()

    def action_save_session(self) -> None:
        if self._confirm("Lưu kỳ thi", "Bạn có chắc muốn lưu kỳ thi?"):
            self.save_session()

    def action_save_session_as(self) -> None:
        if self._confirm("Lưu dưới tên khác", "Bạn có chắc muốn lưu kỳ thi dưới tên khác?"):
            self.save_session_as()

    def action_close_current_session(self) -> None:
        if self._confirm("Đóng kỳ thi", "Bạn có chắc muốn đóng kỳ thi hiện tại?"):
            self.close_current_session()

    def action_manage_template(self) -> None:
        if self._confirm("Quản lý mẫu giấy thi", "Mở trình quản lý mẫu giấy thi?"):
            self.open_template_editor()

    def action_manage_subjects(self) -> None:
        if self._confirm("Quản lý môn học", "Mở quản lý môn học?"):
            self.manage_subjects()

    def action_exit(self) -> None:
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

    def action_run_batch_scan(self) -> None:
        if self._confirm("Batch Scan", "Bạn có chắc muốn chạy Batch Scan?"):
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
        if self._confirm("Calculate Scores", "Bạn có chắc muốn chấm điểm?"):
            self.calculate_scores()

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
        except ImportError as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", f"Cannot import answer key:\n{exc}")
            return

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
                )
            )
            imported_count += 1

        if self.session:
            self.session.answer_key_path = file_path
        self.session_dirty = True
        self._refresh_session_info()
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

        self.filter_column = QComboBox()
        self.filter_column.addItems(["STUDENT ID", "Họ tên", "Ngày sinh", "Nội dung", "Status"])
        self.filter_column.currentTextChanged.connect(self._apply_scan_filter)
        self.search_value = QLineEdit()
        self.search_value.setPlaceholderText("Filter theo tiêu đề bảng đang chọn")
        self.search_value.textChanged.connect(self._apply_scan_filter)

        search_row = QHBoxLayout()
        search_row.addWidget(self.filter_column)
        search_row.addWidget(self.search_value)

        self.scan_list = QTableWidget(0, 5)
        self.scan_list.setHorizontalHeaderLabels(["STUDENT ID", "Họ tên", "Ngày sinh", "Nội dung", "Status"])
        self.scan_list.verticalHeader().setVisible(False)
        self.scan_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.scan_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scan_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.scan_list.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.scan_list.horizontalHeader().sectionClicked.connect(self._on_scan_header_clicked)
        self.scan_list.itemSelectionChanged.connect(self._on_scan_selected)
        self.scan_list.cellDoubleClicked.connect(self._open_edit_selected_scan)
        self.scan_list.cellClicked.connect(self._on_scan_cell_clicked)
        self.progress = QProgressBar()

        self.scan_image_preview = QLabel("Chọn bài thi ở danh sách bên trái")
        self.scan_image_preview.setAlignment(Qt.AlignCenter)
        self.scan_image_preview.setMinimumHeight(260)
        self.scan_result_preview = QTableWidget(0, 2)
        self.scan_result_preview.setHorizontalHeaderLabels(["Mục nhận dạng", "Kết quả"])
        self.scan_result_preview.verticalHeader().setVisible(False)
        self.scan_result_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scan_result_preview.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.scan_result_preview.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        right_split = QSplitter(Qt.Vertical)
        right_top = QWidget(); right_top_l = QVBoxLayout(right_top); right_top_l.addWidget(self.scan_image_preview)
        right_bottom = QWidget(); right_bottom_l = QVBoxLayout(right_bottom); right_bottom_l.addWidget(self.scan_result_preview)
        right_split.addWidget(right_top)
        right_split.addWidget(right_bottom)
        right_split.setSizes([360, 260])

        lr_split = QSplitter(Qt.Horizontal)
        left = QWidget(); left_l = QVBoxLayout(left); left_l.addLayout(search_row); left_l.addWidget(self.scan_list)
        lr_split.addWidget(left)
        lr_split.addWidget(right_split)
        lr_split.setSizes([450, 750])

        self.score_preview_table = QTableWidget(0, 8)
        self.score_preview_table.setHorizontalHeaderLabels(["Student ID", "Name", "Subject", "Exam Code", "Correct", "Wrong", "Blank", "Score"])
        self.score_preview_table.verticalHeader().setVisible(False)
        self.score_preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        layout.addWidget(self.progress)
        layout.addWidget(lr_split)
        layout.addWidget(QLabel("Bảng điểm (xem trước trước khi export)"))
        layout.addWidget(self.score_preview_table)
        return w

    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        splitter = QSplitter(Qt.Horizontal)
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
        right_layout.addWidget(self.preview_label)
        right_layout.addWidget(self.result_preview)
        right_layout.addWidget(QLabel("Manual Edit"))
        right_layout.addWidget(self.manual_edit)

        splitter.addWidget(left)
        splitter.addWidget(right)

        layout = QHBoxLayout(w)
        layout.addWidget(splitter)
        return w

    def create_session(self, payload: dict | None = None) -> None:
        payload = payload or {}
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
            config={
                "scan_mode": payload.get("scan_mode", "Ảnh trong thư mục gốc"),
                "scan_root": payload.get("scan_root", ""),
                "paper_part_count": payload.get("paper_part_count", 3),
                "subject_configs": subject_cfgs,
                "subject_catalog": self.subject_catalog,
                "block_catalog": self.block_catalog,
            },
        )
        if common_template and Path(common_template).exists():
            try:
                self.template = Template.load_json(common_template)
            except Exception:
                self.template = None
        self.current_session_path = None
        self.session_dirty = True
        self._refresh_session_info()

    def load_template(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Template", "", "JSON (*.json)")
        if not file_path:
            return
        self.template = Template.load_json(file_path)
        if self.session:
            self.session.template_path = file_path
        self.session_dirty = True
        self._refresh_session_info()

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
        self.editor = TemplateEditorWindow()
        self.editor.show()

    def run_batch_scan(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "Missing template", "Load template first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing scanned sheets")
        if not folder:
            return
        directory = Path(folder)
        file_paths = sorted(
            [
                str(p)
                for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        if not file_paths:
            QMessageBox.information(self, "No images", "Selected folder has no PNG/JPG images.")
            return

        self.scan_list.setRowCount(0)
        self.error_list.clear()
        self.result_preview.clear()
        self.scan_result_preview.setRowCount(0)
        self.manual_edit.clear()
        self.scan_image_preview.setText("Chọn bài thi ở danh sách bên trái")
        self.scan_files = [Path(p) for p in file_paths]
        self.scan_blank_questions.clear()
        self.scan_blank_summary.clear()
        self.scan_manual_adjustments.clear()
        self.scan_edit_history.clear()
        self.scan_last_adjustment.clear()

        def on_progress(current: int, total: int, image_path: str):
            self.progress.setMaximum(total)
            self.progress.setValue(current)

        self.scan_results = self.omr_processor.process_batch(file_paths, self.template, on_progress)
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
            full_name = str(getattr(result, "full_name", "") or "-")
            birth_date = str(getattr(result, "birth_date", "") or "-")
            blank_map = self._compute_blank_questions(result)
            blank_questions = blank_map.get("MCQ", [])
            self.scan_blank_questions[idx] = blank_questions
            self.scan_blank_summary[idx] = blank_map
            status_parts: list[str] = []
            if sid and duplicate_ids.get(sid, 0) > 1:
                status_parts.append("trùng STUDENT ID")
            if not (result.exam_code or "").strip() or "?" in (result.exam_code or ""):
                status_parts.append("không tô exam code")
            status = ", ".join(status_parts) if status_parts else "OK"
            content_parts = []
            for sec in ["MCQ", "TF", "NUMERIC"]:
                vals = blank_map.get(sec, [])
                if vals:
                    content_parts.append(f"{sec}:{','.join(str(v) for v in vals)}")
            content_text = " | ".join(content_parts) if content_parts else "-"

            self.scan_list.insertRow(idx)
            self.scan_list.setItem(idx, 0, QTableWidgetItem(sid or "-"))
            self.scan_list.setItem(idx, 1, QTableWidgetItem(full_name))
            self.scan_list.setItem(idx, 2, QTableWidgetItem(birth_date))
            self.scan_list.setItem(idx, 3, QTableWidgetItem(content_text))
            status_item = QTableWidgetItem(status)
            if status != "OK":
                status_item.setForeground(Qt.red)
            self.scan_list.setItem(idx, 4, status_item)
            for issue in result.issues:
                self.error_list.addItem(f"{Path(result.image_path).name}: {issue.code} - {issue.message}")
            for err in rec_errors:
                self.error_list.addItem(f"{Path(result.image_path).name}: RECOGNITION - {err}")

        self._apply_scan_filter()

    def _compute_blank_questions(self, result) -> dict[str, list[int]]:
        expected_by_section: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if not self.template:
            return expected_by_section
        for z in self.template.zones:
            if not z.grid:
                continue
            count = int(z.grid.question_count or z.grid.rows or 0)
            start = int(z.grid.question_start)
            rng = list(range(start, start + max(0, count)))
            if z.zone_type.value == "MCQ_BLOCK":
                expected_by_section["MCQ"].extend(rng)
            elif z.zone_type.value == "TRUE_FALSE_BLOCK":
                expected_by_section["TF"].extend(rng)
            elif z.zone_type.value == "NUMERIC_BLOCK":
                expected_by_section["NUMERIC"].extend(rng)

        if self.answer_keys and self.session and self.session.subjects:
            key = self.answer_keys.get(self.session.subjects[0], result.exam_code or "")
            if key:
                key_questions = set(key.answers.keys())
                for sec in expected_by_section:
                    expected_by_section[sec] = [q for q in expected_by_section[sec] if q in key_questions]

        return {
            "MCQ": [q for q in sorted(set(expected_by_section["MCQ"])) if q not in set((result.mcq_answers or {}).keys())],
            "TF": [q for q in sorted(set(expected_by_section["TF"])) if q not in set((result.true_false_answers or {}).keys())],
            "NUMERIC": [q for q in sorted(set(expected_by_section["NUMERIC"])) if q not in set((result.numeric_answers or {}).keys())],
        }

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

    def _on_scan_selected(self) -> None:
        index = self.scan_list.currentRow()
        if index < 0 or index >= len(self.scan_results):
            return
        self._update_scan_preview(index)
        self._load_selected_result_for_correction()

    def _status_text_for_row(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.scan_results):
            return "OK"
        res = self.scan_results[idx]
        sid = (res.student_id or "").strip()
        status_parts: list[str] = []
        if sid:
            dup = sum(1 for r in self.scan_results if (r.student_id or "").strip() == sid)
            if dup > 1:
                status_parts.append("trùng STUDENT ID")
        if not (res.exam_code or "").strip() or "?" in (res.exam_code or ""):
            status_parts.append("không tô exam code")
        edits = self.scan_manual_adjustments.get(idx, [])
        if edits:
            status_parts.append("đã chỉnh sửa: " + "; ".join(edits))
        return ", ".join(status_parts) if status_parts else "OK"

    def _record_adjustment(self, idx: int, details: list[str], source: str) -> None:
        if not details:
            return
        message = f"({source}) " + "; ".join(details)
        self.scan_edit_history.setdefault(idx, []).append(message)
        self.scan_last_adjustment[idx] = message
        self.scan_manual_adjustments[idx] = sorted(set(self.scan_manual_adjustments.get(idx, []) + details))

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

    def _refresh_row_status(self, idx: int) -> None:
        status = self._status_text_for_row(idx)
        item = QTableWidgetItem(status)
        if status != "OK":
            item.setForeground(Qt.red)
        self.scan_list.setItem(idx, 4, item)

    def _update_scan_preview(self, index: int) -> None:
        if index < 0 or index >= len(self.scan_results):
            return
        result = self.scan_results[index]
        img_path = Path(result.image_path)
        pix = QPixmap(str(img_path))
        if pix.isNull():
            self.scan_image_preview.setText(f"Cannot load image: {img_path.name}")
        else:
            self.scan_image_preview.setPixmap(
                pix.scaled(
                    self.scan_image_preview.width(),
                    self.scan_image_preview.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        rec_errors = list(getattr(result, "recognition_errors", [])) or list(getattr(result, "errors", []))
        blank_map = self.scan_blank_summary.get(index, {"MCQ": [], "TF": [], "NUMERIC": []})
        rows = [
            ("STUDENT ID", result.student_id or "-"),
            ("Họ tên", str(getattr(result, "full_name", "") or "-")),
            ("Ngày sinh", str(getattr(result, "birth_date", "") or "-")),
            ("Exam code", result.exam_code or "-"),
            ("MCQ nhận dạng", json.dumps(result.mcq_answers, ensure_ascii=False)),
            ("TF nhận dạng", json.dumps(result.true_false_answers, ensure_ascii=False)),
            ("NUMERIC nhận dạng", json.dumps(result.numeric_answers, ensure_ascii=False)),
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

    def _load_selected_result_for_correction(self) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= len(self.scan_results):
            return
        res = self.scan_results[idx]
        payload = {
            "student_id": res.student_id,
            "exam_code": res.exam_code,
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
                    "mcq_answers": res.mcq_answers,
                    "numeric_answers": res.numeric_answers,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    def _open_edit_selected_scan(self, *_args) -> None:
        idx = self.scan_list.currentRow()
        if idx < 0 or idx >= len(self.scan_results):
            QMessageBox.warning(self, "No selection", "Chọn bài thi cần sửa trước.")
            return
        res = self.scan_results[idx]

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Sửa bài thi: {Path(res.image_path).name}")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()

        inp_sid = QLineEdit(res.student_id)
        inp_code = QLineEdit(res.exam_code)
        txt = QTextEdit()
        txt.setPlainText(
            json.dumps(
                {
                    "mcq_answers": res.mcq_answers,
                    "true_false_answers": res.true_false_answers,
                    "numeric_answers": res.numeric_answers,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        form.addRow("Student ID", inp_sid)
        form.addRow("Exam Code", inp_code)
        lay.addLayout(form)
        lay.addWidget(QLabel("Sửa nhận dạng (JSON)"))
        lay.addWidget(txt)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        changes: list[str] = []
        new_sid = inp_sid.text().strip()
        new_code = inp_code.text().strip()
        if new_sid != (res.student_id or ""):
            old_sid = res.student_id or ""
            res.student_id = new_sid
            self.scan_list.setItem(idx, 0, QTableWidgetItem(new_sid or "-"))
            changes.append(f"student_id: '{old_sid}' -> '{new_sid}'")
        if new_code != (res.exam_code or ""):
            old_code = res.exam_code or ""
            res.exam_code = new_code
            changes.append(f"exam_code: '{old_code}' -> '{new_code}'")

        try:
            payload = json.loads(txt.toPlainText().strip() or "{}")
            if isinstance(payload.get("mcq_answers"), dict):
                new_mcq_answers = {int(k): str(v) for k, v in payload["mcq_answers"].items()}
                if new_mcq_answers != (res.mcq_answers or {}):
                    res.mcq_answers = new_mcq_answers
                    changes.append("mcq_answers updated")
            if isinstance(payload.get("true_false_answers"), dict):
                new_tf_answers = payload["true_false_answers"]
                if new_tf_answers != (res.true_false_answers or {}):
                    res.true_false_answers = new_tf_answers
                    changes.append("true_false_answers updated")
            if isinstance(payload.get("numeric_answers"), dict):
                new_numeric_answers = {int(k): str(v) for k, v in payload["numeric_answers"].items()}
                if new_numeric_answers != (res.numeric_answers or {}):
                    res.numeric_answers = new_numeric_answers
                    changes.append("numeric_answers updated")
        except Exception as exc:
            QMessageBox.warning(self, "Invalid JSON", f"Dữ liệu nhận dạng không hợp lệ:\n{exc}")
            return

        if changes:
            self._record_adjustment(idx, changes, "dialog_edit")
            self._refresh_all_statuses()
            self._update_scan_preview(idx)
            self._load_selected_result_for_correction()

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
        self.scan_list.setItem(idx, 0, QTableWidgetItem(sid))
        if changes:
            self._record_adjustment(idx, changes, "manual_json")
        self._refresh_all_statuses()
        self._update_scan_preview(idx)
        self._load_selected_result_for_correction()
        QMessageBox.information(self, "Correction", "Manual correction applied to selected scan.")

    def calculate_scores(self) -> list:
        if not self.scan_results or not self.answer_keys:
            QMessageBox.warning(self, "Missing data", "Run scans and load/import answer keys first.")
            return []

        subject = self.session.subjects[0] if self.session else "General"
        rows = []
        missing = 0
        for scan in self.scan_results:
            key = self.answer_keys.get(subject, scan.exam_code)
            if not key:
                missing += 1
                continue
            rows.append(self.scoring_engine.score(scan, key))

        self.score_rows = rows
        self.score_preview_table.setRowCount(0)
        for i, r in enumerate(rows):
            self.score_preview_table.insertRow(i)
            self.score_preview_table.setItem(i, 0, QTableWidgetItem(r.student_id or "-"))
            self.score_preview_table.setItem(i, 1, QTableWidgetItem(r.name or "-"))
            self.score_preview_table.setItem(i, 2, QTableWidgetItem(r.subject))
            self.score_preview_table.setItem(i, 3, QTableWidgetItem(r.exam_code))
            self.score_preview_table.setItem(i, 4, QTableWidgetItem(str(r.correct)))
            self.score_preview_table.setItem(i, 5, QTableWidgetItem(str(r.wrong)))
            self.score_preview_table.setItem(i, 6, QTableWidgetItem(str(r.blank)))
            self.score_preview_table.setItem(i, 7, QTableWidgetItem(str(r.score)))

        if missing:
            QMessageBox.information(self, "Scoring preview", f"Calculated {len(rows)} result(s). Missing key for {missing} scan(s).")
        else:
            QMessageBox.information(self, "Scoring preview", f"Calculated {len(rows)} result(s).")
        return rows

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
        QMessageBox.information(self, "Export", "Exported CSV, JSON, XML, XLSX.")

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
        if self.session_dirty:
            if not self._confirm("Thoát", "Kỳ thi hiện tại có thay đổi chưa lưu. Bạn vẫn muốn thoát?"):
                event.ignore()
                return
        event.accept()



def run() -> None:
    app = QApplication([])
    window = MainWindow()
    window.showMaximized()
    app.exec()


if __name__ == "__main__":
    run()
