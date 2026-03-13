from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
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
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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
        self.scan_blank_summary: dict[int, dict[str, list[int]]] = {}
        self.scan_manual_adjustments: dict[int, list[str]] = {}
        self.scan_edit_history: dict[int, list[str]] = {}
        self.scan_last_adjustment: dict[int, str] = {}

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

        btn_run_scan = QPushButton("Batch Scan Images")
        btn_run_scan.clicked.connect(self.run_batch_scan)
        btn_edit_scan = QPushButton("Sửa bài thi được chọn")
        btn_edit_scan.clicked.connect(self._open_edit_selected_scan)

        btn_export = QPushButton("Export Scoring Results")
        btn_export.clicked.connect(self.export_results)

        layout.addWidget(btn_run_scan)
        layout.addWidget(btn_edit_scan)
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
        latest = self.scan_last_adjustment.get(idx, "")
        if latest:
            status_parts.append("điều chỉnh gần nhất: " + latest)
        elif edits:
            status_parts.append("đã chỉnh sửa")
        return ", ".join(status_parts) if status_parts else "OK"

    def _record_adjustment(self, idx: int, details: list[str], source: str) -> None:
        if not details:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{stamp}] ({source}) " + "; ".join(details)
        self.scan_edit_history.setdefault(idx, []).append(message)
        self.scan_last_adjustment[idx] = message
        self.scan_manual_adjustments[idx] = sorted(set(self.scan_manual_adjustments.get(idx, []) + details))

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
                res.mcq_answers = {int(k): str(v) for k, v in payload["mcq_answers"].items()}
                changes.append("mcq_answers updated")
            if isinstance(payload.get("true_false_answers"), dict):
                res.true_false_answers = payload["true_false_answers"]
                changes.append("true_false_answers updated")
            if isinstance(payload.get("numeric_answers"), dict):
                res.numeric_answers = {int(k): str(v) for k, v in payload["numeric_answers"].items()}
                changes.append("numeric_answers updated")
        except Exception as exc:
            QMessageBox.warning(self, "Invalid JSON", f"Dữ liệu nhận dạng không hợp lệ:\n{exc}")
            return

        if changes:
            self._record_adjustment(idx, changes, "dialog_edit")
            self._refresh_row_status(idx)
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
            res.mcq_answers = {int(k): str(v) for k, v in patch["mcq_answers"].items()}
            changes.append("mcq_answers updated")
        if isinstance(patch.get("numeric_answers"), dict):
            res.numeric_answers = {int(k): str(v) for k, v in patch["numeric_answers"].items()}
            changes.append("numeric_answers updated")
        if isinstance(patch.get("true_false_answers"), dict):
            res.true_false_answers = patch["true_false_answers"]
            changes.append("true_false_answers updated")

        sid = (res.student_id or "").strip() or "-"
        self.scan_list.setItem(idx, 0, QTableWidgetItem(sid))
        if changes:
            self._record_adjustment(idx, changes, "manual_json")
        self._refresh_row_status(idx)
        self._update_scan_preview(idx)
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
