from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
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
        self.scan_files: list[Path] = []
        self.scan_blank_questions: dict[int, list[int]] = {}

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

        self.search_student_id = QLineEdit()
        self.search_student_id.setPlaceholderText("Search STUDENT ID")
        self.search_student_id.textChanged.connect(self._apply_scan_filter)

        self.search_name = QLineEdit()
        self.search_name.setPlaceholderText("Search Họ tên")
        self.search_name.textChanged.connect(self._apply_scan_filter)

        search_row = QHBoxLayout()
        search_row.addWidget(self.search_student_id)
        search_row.addWidget(self.search_name)

        self.scan_list = QListWidget()
        self.scan_list.currentRowChanged.connect(self._on_scan_selected)
        self.progress = QProgressBar()

        self.scan_image_preview = QLabel("Chọn bài thi ở danh sách bên trái")
        self.scan_image_preview.setAlignment(Qt.AlignCenter)
        self.scan_image_preview.setMinimumHeight(260)
        self.scan_result_preview = QTextEdit()
        self.scan_result_preview.setReadOnly(True)
        self.scan_result_preview.setPlaceholderText("Kết quả nhận dạng")

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

        btn_run_scan = QPushButton("Batch Scan Images")
        btn_run_scan.clicked.connect(self.run_batch_scan)

        btn_export = QPushButton("Export Scoring Results")
        btn_export.clicked.connect(self.export_results)

        layout.addWidget(btn_run_scan)
        layout.addWidget(btn_export)
        layout.addWidget(self.progress)
        layout.addWidget(lr_split)
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

        self.scan_list.clear()
        self.error_list.clear()
        self.result_preview.clear()
        self.scan_result_preview.clear()
        self.manual_edit.clear()
        self.scan_image_preview.setText("Chọn bài thi ở danh sách bên trái")
        self.scan_files = [Path(p) for p in file_paths]
        self.scan_blank_questions.clear()

        def on_progress(current: int, total: int, image_path: str):
            self.progress.setMaximum(total)
            self.progress.setValue(current)
            self.scan_list.addItem(f"{current}/{total}: {Path(image_path).name}")

        self.scan_results = self.omr_processor.process_batch(file_paths, self.template, on_progress)
        self.scan_list.clear()
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
            blank_questions = self._compute_blank_mcq_questions(result)
            self.scan_blank_questions[idx] = blank_questions
            status_parts: list[str] = []
            if sid and duplicate_ids.get(sid, 0) > 1:
                status_parts.append("trùng STUDENT ID")
            if not (result.exam_code or "").strip() or "?" in (result.exam_code or ""):
                status_parts.append("không tô exam code")
            status = ", ".join(status_parts) if status_parts else "OK"
            blank_txt = ",".join(str(q) for q in blank_questions) if blank_questions else "-"

            text = (
                f"{idx+1}. STUDENT ID: {sid or '-'} | Họ tên: {full_name} | Ngày sinh: {birth_date} | "
                f"Nội dung: Câu không tô [{blank_txt}] | Status: {status}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, {"student_id": sid, "full_name": full_name})
            self.scan_list.addItem(item)
            for issue in result.issues:
                self.error_list.addItem(f"{Path(result.image_path).name}: {issue.code} - {issue.message}")
            for err in rec_errors:
                self.error_list.addItem(f"{Path(result.image_path).name}: RECOGNITION - {err}")

        self._apply_scan_filter()

    def _compute_blank_mcq_questions(self, result) -> list[int]:
        expected: list[int] = []
        if not self.template:
            return expected
        for z in self.template.zones:
            if z.zone_type.value != "MCQ_BLOCK" or not z.grid:
                continue
            count = int(z.grid.question_count or z.grid.rows or 0)
            start = int(z.grid.question_start)
            expected.extend(range(start, start + max(0, count)))
        if not expected:
            return []
        marked = set((result.mcq_answers or {}).keys())
        return [q for q in sorted(set(expected)) if q not in marked]

    def _apply_scan_filter(self) -> None:
        sid_q = self.search_student_id.text().strip().lower()
        name_q = self.search_name.text().strip().lower()
        for i in range(self.scan_list.count()):
            item = self.scan_list.item(i)
            data = item.data(Qt.UserRole) or {}
            sid = str(data.get("student_id", "")).lower()
            full_name = str(data.get("full_name", "")).lower()
            show = (sid_q in sid if sid_q else True) and (name_q in full_name if name_q else True)
            item.setHidden(not show)

    def _on_scan_selected(self, index: int) -> None:
        if index < 0 or index >= len(self.scan_results):
            return
        self._update_scan_preview(index)
        self._load_selected_result_for_correction()

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
        payload = {
            "student_id": result.student_id,
            "exam_code": result.exam_code,
            "mcq_answers": result.mcq_answers,
            "true_false_answers": result.true_false_answers,
            "numeric_answers": result.numeric_answers,
            "blank_mcq_questions": self.scan_blank_questions.get(index, []),
            "issues": [{"code": i.code, "message": i.message, "zone_id": i.zone_id} for i in result.issues],
            "recognition_errors": rec_errors,
        }
        self.scan_result_preview.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

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
        if "student_id" in patch:
            res.student_id = str(patch["student_id"])
        if "exam_code" in patch:
            res.exam_code = str(patch["exam_code"])
        if isinstance(patch.get("mcq_answers"), dict):
            res.mcq_answers = {int(k): str(v) for k, v in patch["mcq_answers"].items()}
        if isinstance(patch.get("numeric_answers"), dict):
            res.numeric_answers = {int(k): str(v) for k, v in patch["numeric_answers"].items()}
        if isinstance(patch.get("true_false_answers"), dict):
            res.true_false_answers = patch["true_false_answers"]

        self.scan_list.item(idx).setText(
            f"{idx+1}. {Path(res.image_path).name} | ID:{res.student_id or '-'} | Code:{res.exam_code or '-'} | Errors:{len(res.issues) + len(getattr(res, 'recognition_errors', []) or getattr(res, 'errors', []))}"
        )
        self._load_selected_result_for_correction()
        QMessageBox.information(self, "Correction", "Manual correction applied to selected scan.")

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
