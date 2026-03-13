from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.answer_key_importer import ImportedAnswerKey, ImportedAnswerKeyPackage


@dataclass
class PreviewRow:
    question: int
    exam_code: str
    answer_type: str
    answer_value: str


class ImportAnswerKeyDialog(QDialog):
    def __init__(self, imported: ImportedAnswerKeyPackage, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Imported Answer Keys")
        self.resize(920, 600)
        self.imported = imported

        first_exam = next(iter(imported.exam_keys.values()), ImportedAnswerKey())
        self.exam_id_edit = QLineEdit(str(first_exam.exam_id))
        self.exam_codes_label = QLabel()

        self.mcq_count_edit = QLineEdit("0")
        self.tf_count_edit = QLineEdit("0")
        self.numeric_count_edit = QLineEdit("0")

        top = QHBoxLayout()
        top.addWidget(QLabel("Exam ID:"))
        top.addWidget(self.exam_id_edit)
        top.addWidget(QLabel("Mã đề:"))
        top.addWidget(self.exam_codes_label)

        mapping = QGridLayout()
        mapping.addWidget(QLabel("Mapping sections (theo thứ tự dòng):"), 0, 0, 1, 4)
        mapping.addWidget(QLabel("MCQ số câu"), 1, 0)
        mapping.addWidget(self.mcq_count_edit, 1, 1)
        mapping.addWidget(QLabel("TF số câu"), 1, 2)
        mapping.addWidget(self.tf_count_edit, 1, 3)
        mapping.addWidget(QLabel("NUMERIC số câu"), 2, 0)
        mapping.addWidget(self.numeric_count_edit, 2, 1)
        btn_apply_mapping = QPushButton("Apply Mapping")
        btn_apply_mapping.clicked.connect(self._apply_mapping)
        mapping.addWidget(btn_apply_mapping, 2, 3)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Question", "Exam Code", "Type", "Answer"])

        btn_add = QPushButton("Add Row")
        btn_add.clicked.connect(self._add_empty_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(mapping)
        layout.addWidget(self.table)
        layout.addWidget(btn_add)
        layout.addWidget(btn_box)

        self._load_rows()

    def _load_rows(self) -> None:
        rows: list[PreviewRow] = []
        for exam_code, key in sorted(self.imported.exam_keys.items()):
            for q, ans in sorted(key.mcq_answers.items()):
                rows.append(PreviewRow(q, exam_code, "MCQ", ans))
            for q, ans in sorted(key.true_false_answers.items()):
                text = "".join("T" if ans.get(ch, False) else "F" for ch in ["a", "b", "c", "d"])
                rows.append(PreviewRow(q, exam_code, "TF", text))
            for q, ans in sorted(key.numeric_answers.items()):
                rows.append(PreviewRow(q, exam_code, "NUMERIC", ans))

        self.table.setRowCount(0)
        for row in rows:
            self._append_row(row)

        self._refresh_exam_codes_label()
        self._update_section_counts_from_rows()

    def _refresh_exam_codes_label(self) -> None:
        codes = sorted(set((self.table.item(r, 1).text().strip() if self.table.item(r, 1) else "") for r in range(self.table.rowCount())))
        codes = [c for c in codes if c]
        self.exam_codes_label.setText(", ".join(codes) if codes else "-")

    def _update_section_counts_from_rows(self) -> None:
        counts = {"MCQ": 0, "TF": 0, "NUMERIC": 0}
        for r in range(self.table.rowCount()):
            widget = self.table.cellWidget(r, 2)
            if widget:
                counts[widget.currentText()] += 1
        self.mcq_count_edit.setText(str(counts["MCQ"]))
        self.tf_count_edit.setText(str(counts["TF"]))
        self.numeric_count_edit.setText(str(counts["NUMERIC"]))

    def _append_row(self, row: PreviewRow) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(row.question)))
        self.table.setItem(r, 1, QTableWidgetItem(row.exam_code))

        type_combo = QComboBox()
        type_combo.addItems(["MCQ", "TF", "NUMERIC"])
        type_combo.setCurrentText(row.answer_type)
        self.table.setCellWidget(r, 2, type_combo)
        self.table.setItem(r, 3, QTableWidgetItem(row.answer_value))

    def _add_empty_row(self) -> None:
        default_exam = next(iter(self.imported.exam_keys.keys()), "0101")
        self._append_row(PreviewRow(question=self.table.rowCount() + 1, exam_code=default_exam, answer_type="MCQ", answer_value="A"))
        self._refresh_exam_codes_label()
        self._update_section_counts_from_rows()

    def _apply_mapping(self) -> None:
        try:
            mcq_count = int(self.mcq_count_edit.text().strip() or "0")
            tf_count = int(self.tf_count_edit.text().strip() or "0")
            numeric_count = int(self.numeric_count_edit.text().strip() or "0")
        except ValueError:
            QMessageBox.warning(self, "Invalid mapping", "Section counts must be integer values.")
            return

        total = self.table.rowCount()
        if mcq_count + tf_count + numeric_count > total:
            QMessageBox.warning(self, "Invalid mapping", "Sum of section counts exceeds total rows.")
            return

        for r in range(total):
            widget = self.table.cellWidget(r, 2)
            if not widget:
                continue
            if r < mcq_count:
                widget.setCurrentText("MCQ")
            elif r < mcq_count + tf_count:
                widget.setCurrentText("TF")
            else:
                widget.setCurrentText("NUMERIC")

    def _on_accept(self) -> None:
        try:
            exam_id = int(self.exam_id_edit.text().strip())
        except Exception:
            QMessageBox.warning(self, "Invalid exam id", "Exam ID must be an integer.")
            return

        package = ImportedAnswerKeyPackage()
        try:
            for row_idx in range(self.table.rowCount()):
                q_item = self.table.item(row_idx, 0)
                code_item = self.table.item(row_idx, 1)
                a_item = self.table.item(row_idx, 3)
                if not q_item or not code_item or not a_item:
                    continue
                q = int((q_item.text() or "").strip())
                exam_code = (code_item.text() or "").strip() or "DEFAULT"
                answer_type = self.table.cellWidget(row_idx, 2).currentText()
                answer_text = (a_item.text() or "").strip()
                key = package.exam_keys.setdefault(exam_code, ImportedAnswerKey(exam_id=exam_id))

                if answer_type == "MCQ":
                    val = answer_text.upper()
                    if val not in {"A", "B", "C", "D", "E"}:
                        raise ImportError(
                            f"Row {row_idx + 1}: invalid MCQ value '{answer_text}'. Expected A/B/C/D/E."
                        )
                    key.mcq_answers[q] = val
                elif answer_type == "NUMERIC":
                    token = answer_text.replace(" ", "").replace(",", ".")
                    if token.startswith(("+", "-")):
                        token = token[1:]
                    if not token or token.count(".") > 1 or not token.replace(".", "").isdigit():
                        raise ImportError(
                            f"Row {row_idx + 1}: invalid numeric value '{answer_text}'. Expected numeric (e.g. 69, -105, 0,61)."
                        )
                    key.numeric_answers[q] = answer_text
                else:
                    token = answer_text.replace(" ", "").upper()
                    if len(token) != 4 or any(ch not in {"T", "F", "D", "S", "Đ"} for ch in token):
                        raise ImportError(
                            f"Row {row_idx + 1}: invalid TF value '{answer_text}'. Expected 4 chars (T/F or Đ/S)."
                        )
                    key.true_false_answers[q] = {
                        "a": token[0] in {"T", "D", "Đ"},
                        "b": token[1] in {"T", "D", "Đ"},
                        "c": token[2] in {"T", "D", "Đ"},
                        "d": token[3] in {"T", "D", "Đ"},
                    }
        except ImportError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Validation error", f"Invalid edited data: {exc}")
            return

        self.imported = package
        self._refresh_exam_codes_label()
        self.accept()

    def result_answer_key(self) -> ImportedAnswerKeyPackage:
        return self.imported
