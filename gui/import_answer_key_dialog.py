from __future__ import annotations

from dataclasses import dataclass
import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.answer_key_importer import ImportedAnswerKey, ImportedAnswerKeyPackage
from models.answer_key import SubjectKey


@dataclass
class ImportedAnswerRow:
    section: str
    question_no: int
    answer: str


class ImportAnswerKeyDialog(QDialog):
    def __init__(self, parent_window, rows: list[ImportedAnswerRow] | None = None, subject_key: str = "") -> None:
        self._legacy_mode = isinstance(parent_window, ImportedAnswerKeyPackage)
        if self._legacy_mode:
            imported_package = copy.deepcopy(parent_window)
            real_parent = rows
            super().__init__(real_parent)
            self.main_window = real_parent
            self.rows = []
            self.subject_key = str(subject_key or "").strip()
            self._legacy_package: ImportedAnswerKeyPackage = imported_package
            self._legacy_rows_by_exam_code = self._rows_by_exam_code_from_package(imported_package)
        else:
            super().__init__(parent_window)
            self.main_window = parent_window
            self.rows = rows or []
            self.subject_key = str(subject_key or "").strip()
            self._legacy_package = ImportedAnswerKeyPackage()
            self._legacy_rows_by_exam_code: dict[str, list[ImportedAnswerRow]] = {}
        self._is_refreshing = False

        self.setWindowTitle("Kiểm tra và nhập đáp án")
        self.resize(960, 760)

        self.exam_code_combo = QComboBox(self)
        self.exam_code_combo.setEditable(True)
        self._load_exam_codes()

        self.mcq_count_spin = QSpinBox(self)
        self.mcq_count_spin.setRange(0, 500)
        self.tf_count_spin = QSpinBox(self)
        self.tf_count_spin.setRange(0, 500)
        self.numeric_count_spin = QSpinBox(self)
        self.numeric_count_spin.setRange(0, 500)
        self.lbl_total = QLabel(self)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Phần", "Câu", "Đáp án"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)

        self.btn_add = QPushButton("Thêm dòng", self)
        self.btn_delete = QPushButton("Xóa dòng chọn", self)
        self.btn_apply_mapping = QPushButton("Áp mapping phần", self)

        form = QFormLayout()
        form.addRow("Mã đề", self.exam_code_combo)
        form.addRow("Số câu MCQ", self.mcq_count_spin)
        form.addRow("Số câu TF", self.tf_count_spin)
        form.addRow("Số câu NUMERIC", self.numeric_count_spin)
        form.addRow("Tổng số dòng", self.lbl_total)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_add)
        actions.addWidget(self.btn_delete)
        actions.addStretch(1)
        actions.addWidget(self.btn_apply_mapping)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.table, 1)
        layout.addLayout(actions)
        layout.addWidget(buttons)

        self.btn_add.clicked.connect(self._append_row)
        self.btn_delete.clicked.connect(self._delete_selected_rows)
        self.btn_apply_mapping.clicked.connect(self._apply_mapping)
        self.table.itemChanged.connect(self._handle_table_changed)
        self.mcq_count_spin.valueChanged.connect(self._refresh_total_label)
        self.tf_count_spin.valueChanged.connect(self._refresh_total_label)
        self.numeric_count_spin.valueChanged.connect(self._refresh_total_label)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        self._load_rows()
        self.exam_code_combo.currentTextChanged.connect(self._handle_exam_code_changed)
        self._guess_section_counts()
        self._refresh_total_label()

    def _load_exam_codes(self) -> None:
        if self._legacy_mode:
            codes = sorted(str(code or "").strip() for code in (self._legacy_rows_by_exam_code or {}).keys() if str(code or "").strip())
        else:
            existing = self.main_window._fetch_answer_keys_for_subject_scoped(self.subject_key) or {}
            codes = sorted(str(code or "").strip() for code in existing.keys() if str(code or "").strip())
        if not codes:
            codes = ["0000"]
        self.exam_code_combo.clear()
        self.exam_code_combo.addItems(codes)
        if self.exam_code_combo.findText("0000") < 0:
            self.exam_code_combo.addItem("0000")
        self.exam_code_combo.setCurrentText(codes[0] if codes else "0000")

    def _load_rows(self) -> None:
        if self._legacy_mode:
            exam_code = str(self.exam_code_combo.currentText() or "").strip() or "0000"
            self.rows = list(self._legacy_rows_by_exam_code.get(exam_code, []))
        self._is_refreshing = True
        self.table.setRowCount(0)
        for row in self.rows:
            self._append_row(row.section, row.question_no, row.answer)
        self._is_refreshing = False
        self._refresh_total_label()

    def _append_row(self, section: str = "MCQ", question_no: int = 1, answer: str = "") -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        section_combo = QComboBox(self.table)
        section_combo.addItems(["MCQ", "TF", "NUMERIC"])
        target_section = str(section or "MCQ").strip().upper() or "MCQ"
        section_combo.setCurrentText(target_section if target_section in {"MCQ", "TF", "NUMERIC"} else "MCQ")
        section_combo.currentTextChanged.connect(self._refresh_total_label)
        self.table.setCellWidget(row, 0, section_combo)

        q_item = QTableWidgetItem(str(question_no))
        q_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 1, q_item)

        answer_item = QTableWidgetItem(str(answer))
        self.table.setItem(row, 2, answer_item)

    def _delete_selected_rows(self) -> None:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        if not rows:
            return
        self._is_refreshing = True
        for row in rows:
            self.table.removeRow(row)
        self._is_refreshing = False
        self._refresh_total_label()

    def _apply_mapping(self) -> None:
        total = self.table.rowCount()
        mcq = int(self.mcq_count_spin.value())
        tf = int(self.tf_count_spin.value())
        numeric = int(self.numeric_count_spin.value())
        if mcq + tf + numeric != total:
            QMessageBox.warning(
                self,
                "Mapping phần",
                "Tổng số câu MCQ + TF + NUMERIC phải đúng bằng tổng số dòng hiện có để tránh gán nhầm phần.",
            )
            return
        self._is_refreshing = True
        for row in range(total):
            combo = self.table.cellWidget(row, 0)
            if not isinstance(combo, QComboBox):
                continue
            if row < mcq:
                combo.setCurrentText("MCQ")
            elif row < mcq + tf:
                combo.setCurrentText("TF")
            else:
                combo.setCurrentText("NUMERIC")
        self._is_refreshing = False
        self._refresh_total_label()

    def _guess_section_counts(self) -> None:
        mcq = 0
        tf = 0
        numeric = 0
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            text = combo.currentText().strip().upper() if isinstance(combo, QComboBox) else "MCQ"
            if text == "MCQ":
                mcq += 1
            elif text == "TF":
                tf += 1
            else:
                numeric += 1
        self.mcq_count_spin.setValue(mcq)
        self.tf_count_spin.setValue(tf)
        self.numeric_count_spin.setValue(numeric)

    def _refresh_total_label(self) -> None:
        total = self.table.rowCount()
        typed_total = int(self.mcq_count_spin.value()) + int(self.tf_count_spin.value()) + int(self.numeric_count_spin.value())
        suffix = ""
        if typed_total != total:
            suffix = f" | Mapping hiện tại: {typed_total}"
        self.lbl_total.setText(f"{total}{suffix}")

    def _handle_table_changed(self, _item) -> None:
        if self._is_refreshing:
            return
        self._refresh_total_label()

    def _handle_exam_code_changed(self, _text: str) -> None:
        if self._is_refreshing:
            return
        if self._legacy_mode:
            self._store_current_rows_to_legacy_cache()
        self._load_rows()
        self._guess_section_counts()
        self._refresh_total_label()

    def _rows_from_table(self) -> list[ImportedAnswerRow]:
        out: list[ImportedAnswerRow] = []
        seen_pairs: set[tuple[str, int]] = set()
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            section = combo.currentText().strip().upper() if isinstance(combo, QComboBox) else "MCQ"
            q_text = str(self.table.item(row, 1).text() if self.table.item(row, 1) else "").strip()
            answer = str(self.table.item(row, 2).text() if self.table.item(row, 2) else "").strip()
            if not q_text:
                raise ValueError(f"Dòng {row + 1}: thiếu số câu.")
            try:
                question_no = int(q_text)
            except Exception as exc:
                raise ValueError(f"Dòng {row + 1}: số câu không hợp lệ.") from exc
            if question_no <= 0:
                raise ValueError(f"Dòng {row + 1}: số câu phải lớn hơn 0.")
            pair = (section, question_no)
            if pair in seen_pairs:
                raise ValueError(f"Dòng {row + 1}: trùng cặp phần/câu {section} - {question_no}.")
            seen_pairs.add(pair)
            out.append(ImportedAnswerRow(section=section, question_no=question_no, answer=answer))
        return out

    @staticmethod
    def _rows_by_exam_code_from_package(package: ImportedAnswerKeyPackage) -> dict[str, list[ImportedAnswerRow]]:
        out: dict[str, list[ImportedAnswerRow]] = {}
        for exam_code, key in (package.exam_keys or {}).items():
            rows: list[ImportedAnswerRow] = []
            for q_no, ans in sorted((key.mcq_answers or {}).items(), key=lambda x: int(x[0])):
                rows.append(ImportedAnswerRow(section="MCQ", question_no=int(q_no), answer=str(ans)))
            for q_no, flags in sorted((key.true_false_answers or {}).items(), key=lambda x: int(x[0])):
                token = "".join("Đ" if bool((flags or {}).get(ch)) else "S" for ch in ["a", "b", "c", "d"])
                rows.append(ImportedAnswerRow(section="TF", question_no=int(q_no), answer=token))
            for q_no, ans in sorted((key.numeric_answers or {}).items(), key=lambda x: int(x[0])):
                rows.append(ImportedAnswerRow(section="NUMERIC", question_no=int(q_no), answer=str(ans)))
            out[str(exam_code or "0000").strip() or "0000"] = rows
        return out

    def _store_current_rows_to_legacy_cache(self) -> None:
        if not self._legacy_mode:
            return
        exam_code = str(self.exam_code_combo.currentText() or "").strip() or "0000"
        try:
            rows = self._rows_from_table()
        except Exception:
            rows = list(self.rows or [])
        self._legacy_rows_by_exam_code[exam_code] = rows

    @staticmethod
    def _package_from_rows_by_exam_code(rows_by_exam_code: dict[str, list[ImportedAnswerRow]]) -> ImportedAnswerKeyPackage:
        package = ImportedAnswerKeyPackage()
        for exam_code, rows in (rows_by_exam_code or {}).items():
            key = ImportedAnswerKey()
            for row in rows:
                section = str(row.section or "").strip().upper()
                if section == "MCQ":
                    key.mcq_answers[int(row.question_no)] = str(row.answer or "").strip().upper()
                elif section == "TF":
                    text = str(row.answer or "").strip().upper()
                    key.true_false_answers[int(row.question_no)] = {
                        flag: (text[idx] in {"Đ", "D", "T", "1"})
                        for idx, flag in enumerate(["a", "b", "c", "d"])
                        if idx < len(text) and text[idx] in {"Đ", "D", "T", "1", "S", "F", "0"}
                    }
                elif section == "NUMERIC":
                    key.numeric_answers[int(row.question_no)] = str(row.answer or "").strip()
            package.exam_keys[str(exam_code or "0000").strip() or "0000"] = key
        return package

    def result_answer_key(self) -> ImportedAnswerKeyPackage:
        if self._legacy_mode:
            return copy.deepcopy(self._legacy_package)
        return self._package_from_rows_by_exam_code(
            {str(self.exam_code_combo.currentText() or "0000").strip() or "0000": self._rows_from_table()}
        )

    def _on_accept(self) -> None:
        exam_code = str(self.exam_code_combo.currentText() or "").strip()
        if not exam_code:
            QMessageBox.warning(self, "Nhập đáp án", "Vui lòng nhập mã đề.")
            return
        try:
            rows = self._rows_from_table()
        except ValueError as exc:
            QMessageBox.warning(self, "Nhập đáp án", str(exc))
            return

        answers: dict[int, str] = {}
        tf_answers: dict[int, dict[str, bool]] = {}
        numeric_answers: dict[int, str] = {}

        for row in rows:
            if row.section == "MCQ":
                answers[int(row.question_no)] = str(row.answer or "").strip().upper()
            elif row.section == "TF":
                text = str(row.answer or "").strip().upper()
                tf_answers[int(row.question_no)] = {
                    key: (text[idx] in {"Đ", "D", "T", "1"})
                    for idx, key in enumerate(["a", "b", "c", "d"])
                    if idx < len(text) and text[idx] in {"Đ", "D", "T", "1", "S", "F", "0"}
                }
            else:
                numeric_answers[int(row.question_no)] = str(row.answer or "").strip()

        if self._legacy_mode:
            self._legacy_rows_by_exam_code[exam_code] = rows
            self._legacy_package = self._package_from_rows_by_exam_code(self._legacy_rows_by_exam_code)
        else:
            existing = self.main_window._fetch_answer_keys_for_subject_scoped(self.subject_key) or {}
            merged = dict(existing)
            merged[exam_code] = {
                "mcq_answers": answers,
                "true_false_answers": tf_answers,
                "numeric_answers": numeric_answers,
            }
            self.main_window._save_answer_keys_for_subject_scoped(self.subject_key, merged)
        self.accept()


__all__ = ["ImportAnswerKeyDialog", "ImportedAnswerRow"]
