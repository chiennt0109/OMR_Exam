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

from core.answer_key_importer import ImportedAnswerKey


@dataclass
class PreviewRow:
    question: int
    answer_type: str
    answer_value: str


class ImportAnswerKeyDialog(QDialog):
    def __init__(self, imported: ImportedAnswerKey, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Imported Answer Key")
        self.resize(720, 520)
        self.imported = imported

        self.exam_id_edit = QLineEdit(str(imported.exam_id))
        top = QHBoxLayout()
        top.addWidget(QLabel("Exam ID:"))
        top.addWidget(self.exam_id_edit)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Question", "Type", "Answer"])

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
        for q, ans in sorted(self.imported.mcq_answers.items()):
            rows.append(PreviewRow(question=q, answer_type="MCQ", answer_value=ans))
        for q, ans in sorted(self.imported.numeric_answers.items()):
            rows.append(PreviewRow(question=q, answer_type="NUMERIC", answer_value=ans))
        for q, ans in sorted(self.imported.true_false_answers.items()):
            text = ",".join(f"{k}:{'T' if v else 'F'}" for k, v in sorted(ans.items()))
            rows.append(PreviewRow(question=q, answer_type="TF", answer_value=text))

        self.table.setRowCount(0)
        for row in rows:
            self._append_row(row)

    def _append_row(self, row: PreviewRow) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(row.question)))

        type_combo = QComboBox()
        type_combo.addItems(["MCQ", "TF", "NUMERIC"])
        type_combo.setCurrentText(row.answer_type)
        self.table.setCellWidget(r, 1, type_combo)
        self.table.setItem(r, 2, QTableWidgetItem(row.answer_value))

    def _add_empty_row(self) -> None:
        self._append_row(PreviewRow(question=self.table.rowCount() + 1, answer_type="MCQ", answer_value="A"))

    def _on_accept(self) -> None:
        try:
            exam_id = int(self.exam_id_edit.text().strip())
        except Exception:
            QMessageBox.warning(self, "Invalid exam id", "Exam ID must be an integer.")
            return

        parsed = ImportedAnswerKey(exam_id=exam_id)
        try:
            for row_idx in range(self.table.rowCount()):
                q_item = self.table.item(row_idx, 0)
                a_item = self.table.item(row_idx, 2)
                if q_item is None or a_item is None:
                    continue
                q = int((q_item.text() or "").strip())
                answer_type = self.table.cellWidget(row_idx, 1).currentText()
                answer_text = (a_item.text() or "").strip()

                if answer_type == "MCQ":
                    val = answer_text.upper()
                    if val not in {"A", "B", "C", "D", "E"}:
                        raise ImportError(
                            f"Row {row_idx + 1}: invalid MCQ value '{answer_text}'. Expected A/B/C/D/E."
                        )
                    parsed.mcq_answers[q] = val
                elif answer_type == "NUMERIC":
                    if not answer_text.isdigit():
                        raise ImportError(
                            f"Row {row_idx + 1}: invalid numeric value '{answer_text}'. Expected digits only."
                        )
                    parsed.numeric_answers[q] = answer_text
                else:
                    tf_payload: dict[str, bool] = {}
                    parts = [p.strip() for p in answer_text.split(",") if p.strip()]
                    if not parts:
                        raise ImportError(
                            f"Row {row_idx + 1}: invalid TF value '{answer_text}'. "
                            "Expected format a:T,b:F,c:T,d:F"
                        )
                    for part in parts:
                        if ":" not in part:
                            raise ImportError(
                                f"Row {row_idx + 1}: invalid TF entry '{part}'. Expected key:value."
                            )
                        key, raw_val = [x.strip() for x in part.split(":", 1)]
                        val = raw_val.upper()
                        if val in {"T", "TRUE", "D", "Đ", "1"}:
                            tf_payload[key.lower()] = True
                        elif val in {"F", "FALSE", "S", "0"}:
                            tf_payload[key.lower()] = False
                        else:
                            raise ImportError(
                                f"Row {row_idx + 1}: invalid TF value '{raw_val}'. Expected T/F or D/S."
                            )
                    parsed.true_false_answers[q] = tf_payload
        except ImportError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Validation error", f"Invalid edited data: {exc}")
            return

        self.imported = parsed
        self.accept()

    def result_answer_key(self) -> ImportedAnswerKey:
        return self.imported
