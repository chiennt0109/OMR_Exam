from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.omr_engine import OMRProcessor
from core.scoring_engine import ScoringEngine
from editor.template_editor import TemplateEditorWindow
from models.answer_key import AnswerKeyRepository
from models.exam_session import ExamSession
from models.template import Template


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Exam Grading System")
        self.resize(1200, 800)

        self.session: ExamSession | None = None
        self.template: Template | None = None
        self.answer_keys: AnswerKeyRepository | None = None
        self.scan_results = []

        self.omr_processor = OMRProcessor()
        self.scoring_engine = ScoringEngine()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(self._build_session_tab(), "Exam Session")
        self.tabs.addTab(self._build_scan_tab(), "OMR Scan")
        self.tabs.addTab(self._build_correction_tab(), "Error Correction")

    def _build_session_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        btn_new_session = QPushButton("Create New Session")
        btn_new_session.clicked.connect(self.create_session)

        btn_load_template = QPushButton("Load Template JSON")
        btn_load_template.clicked.connect(self.load_template)

        btn_load_answers = QPushButton("Load Answer Keys JSON")
        btn_load_answers.clicked.connect(self.load_answer_keys)

        btn_template_editor = QPushButton("Open Template Editor")
        btn_template_editor.clicked.connect(self.open_template_editor)

        self.session_info = QTextEdit()
        self.session_info.setReadOnly(True)

        for btn in [btn_new_session, btn_load_template, btn_load_answers, btn_template_editor]:
            layout.addWidget(btn)
        layout.addWidget(self.session_info)
        return w

    def _build_scan_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.scan_list = QListWidget()
        self.progress = QProgressBar()

        btn_run_scan = QPushButton("Batch Scan Images")
        btn_run_scan.clicked.connect(self.run_batch_scan)

        btn_export = QPushButton("Export Scoring Results")
        btn_export.clicked.connect(self.export_results)

        layout.addWidget(btn_run_scan)
        layout.addWidget(btn_export)
        layout.addWidget(self.progress)
        layout.addWidget(self.scan_list)
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

        left_layout.addWidget(QLabel("Detected Errors"))
        left_layout.addWidget(self.error_list)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.preview_label)
        right_layout.addWidget(QLabel("Manual Edit"))
        right_layout.addWidget(self.manual_edit)

        splitter.addWidget(left)
        splitter.addWidget(right)

        layout = QHBoxLayout(w)
        layout.addWidget(splitter)
        return w

    def create_session(self) -> None:
        self.session = ExamSession(
            exam_name="Untitled Exam",
            exam_date=str(date.today()),
            subjects=["General"],
            template_path="",
            answer_key_path="",
        )
        self.session_info.setPlainText("Session created. Load template and answer key.")

    def load_template(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Template", "", "JSON (*.json)")
        if not file_path:
            return
        self.template = Template.load_json(file_path)
        if self.session:
            self.session.template_path = file_path
        self._refresh_session_info()

    def load_answer_keys(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Answer Keys", "", "JSON (*.json)")
        if not file_path:
            return
        self.answer_keys = AnswerKeyRepository.load_json(file_path)
        if self.session:
            self.session.answer_key_path = file_path
        self._refresh_session_info()

    def open_template_editor(self) -> None:
        self.editor = TemplateEditorWindow()
        self.editor.show()

    def run_batch_scan(self) -> None:
        if not self.template:
            QMessageBox.warning(self, "Missing template", "Load template first.")
            return
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select scanned sheets", "", "Images (*.png *.jpg *.jpeg)")
        if not file_paths:
            return

        self.scan_list.clear()
        self.error_list.clear()

        def on_progress(current: int, total: int, image_path: str):
            self.progress.setMaximum(total)
            self.progress.setValue(current)
            self.scan_list.addItem(f"{current}/{total}: {Path(image_path).name}")

        self.scan_results = self.omr_processor.process_batch(file_paths, self.template, on_progress)
        for result in self.scan_results:
            for issue in result.issues:
                self.error_list.addItem(f"{Path(result.image_path).name}: {issue.code} - {issue.message}")

    def export_results(self) -> None:
        if not self.scan_results or not self.answer_keys:
            QMessageBox.warning(self, "Missing data", "Run scans and load answer keys first.")
            return

        subject = self.session.subjects[0] if self.session else "General"
        rows = []
        for scan in self.scan_results:
            key = self.answer_keys.get(subject, scan.exam_code)
            if not key:
                continue
            rows.append(self.scoring_engine.score(scan, key))

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
        self.session_info.setPlainText(
            f"Exam: {self.session.exam_name}\nDate: {self.session.exam_date}\n"
            f"Subjects: {', '.join(self.session.subjects)}\n"
            f"Template: {self.session.template_path}\n"
            f"AnswerKey: {self.session.answer_key_path}\n"
            f"Students: {len(self.session.students)}"
        )


def run() -> None:
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    run()
