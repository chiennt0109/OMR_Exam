from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
        self.resize(860, 560)
        self.imported = imported

        first_exam = next(iter(imported.exam_keys.values()), ImportedAnswerKey())
        self.exam_id_edit = QLineEdit(str(first_exam.exam_id))
        top = QHBoxLayout()
        top.addWidget(QLabel("Exam ID:"))
        top.addWidget(self.exam_id_edit)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Question", "Exam Code", "Type", "Answer"])

        btn_add = QPushButton("Add Row")
        btn_add.clicked.connect(self._add_empty_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table)
        layout.addWidget(btn_add)
        layout.addWidget(btn_box)

        self._load_rows()

    def _load_rows(self) -> None:
        rows: list[PreviewRow] = []
        for exam_code, key in sorted(self.imported.exam_keys.items()):
            for q, ans in sorted(key.mcq_answers.items()):
                rows.append(PreviewRow(q, exam_code, "MCQ", ans))
            for q, ans in sorted(key.numeric_answers.items()):
                rows.append(PreviewRow(q, exam_code, "NUMERIC", ans))
            for q, ans in sorted(key.true_false_answers.items()):
                text = "".join("T" if ans.get(ch, False) else "F" for ch in ["a", "b", "c", "d"])
                rows.append(PreviewRow(q, exam_code, "TF", text))

        self.table.setRowCount(0)
        for row in rows:
            self._append_row(row)

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
                    if not answer_text.lstrip("-").isdigit():
                        raise ImportError(
                            f"Row {row_idx + 1}: invalid numeric value '{answer_text}'. Expected digits only."
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
        self.accept()

    def result_answer_key(self) -> ImportedAnswerKeyPackage:
        return self.imported
