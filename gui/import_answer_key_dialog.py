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
    QTabWidget,
    QVBoxLayout,
)

from core.answer_key_importer import ImportedAnswerKey, ImportedAnswerKeyPackage


@dataclass
class PreviewRow:
    question: int
    answer_type: str
    answer_value: str


class ImportAnswerKeyDialog(QDialog):
    def __init__(self, imported: ImportedAnswerKeyPackage, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Imported Answer Keys")
        self.resize(980, 620)
        self.imported = imported
        self.exam_tables: dict[str, QTableWidget] = {}

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
        mapping.addWidget(QLabel("Mapping sections (theo thứ tự dòng tab hiện tại):"), 0, 0, 1, 4)
        mapping.addWidget(QLabel("MCQ số câu"), 1, 0)
        mapping.addWidget(self.mcq_count_edit, 1, 1)
        mapping.addWidget(QLabel("TF số câu"), 1, 2)
        mapping.addWidget(self.tf_count_edit, 1, 3)
        mapping.addWidget(QLabel("NUMERIC số câu"), 2, 0)
        mapping.addWidget(self.numeric_count_edit, 2, 1)
        btn_apply_mapping = QPushButton("Apply Mapping")
        btn_apply_mapping.clicked.connect(self._apply_mapping)
        mapping.addWidget(btn_apply_mapping, 2, 3)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        btn_add = QPushButton("Add Row")
        btn_add.clicked.connect(self._add_empty_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(mapping)
        layout.addWidget(self.tabs)
        layout.addWidget(btn_add)
        layout.addWidget(btn_box)

        self._load_rows()

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Question", "Type", "Answer"])
        return table

    def _append_row(self, table: QTableWidget, row: PreviewRow) -> None:
        r = table.rowCount()
        table.insertRow(r)
        table.setItem(r, 0, QTableWidgetItem(str(row.question)))

        type_combo = QComboBox()
        type_combo.addItems(["MCQ", "TF", "NUMERIC"])
        type_combo.setCurrentText(row.answer_type)
        table.setCellWidget(r, 1, type_combo)
        table.setItem(r, 2, QTableWidgetItem(row.answer_value))

    def _load_rows(self) -> None:
        self.tabs.clear()
        self.exam_tables.clear()
        for exam_code, key in sorted(self.imported.exam_keys.items()):
            table = self._build_table()
            self.exam_tables[exam_code] = table
            self.tabs.addTab(table, exam_code)

            rows: list[PreviewRow] = []
            for q, ans in sorted(key.mcq_answers.items()):
                rows.append(PreviewRow(q, "MCQ", ans))
            for q, ans in sorted(key.true_false_answers.items()):
                text = "".join("T" if ans.get(ch, False) else "F" for ch in ["a", "b", "c", "d"])
                rows.append(PreviewRow(q, "TF", text))
            for q, ans in sorted(key.numeric_answers.items()):
                rows.append(PreviewRow(q, "NUMERIC", ans))
            for sec, answer_type in [("MCQ", "MCQ"), ("TF", "TF"), ("NUMERIC", "NUMERIC")]:
                invalid_map = (key.invalid_answer_rows or {}).get(sec, {}) if isinstance(getattr(key, "invalid_answer_rows", {}), dict) else {}
                for q, ans in sorted((invalid_map or {}).items()):
                    rows.append(PreviewRow(int(q), answer_type, str(ans)))

            rows.sort(key=lambda x: int(x.question))
            for row in rows:
                self._append_row(table, row)

        self._refresh_exam_codes_label()
        self._update_section_counts_from_current_tab()

    def _refresh_exam_codes_label(self) -> None:
        codes = sorted(self.exam_tables.keys())
        self.exam_codes_label.setText(", ".join(codes) if codes else "-")

    def _current_table(self) -> QTableWidget | None:
        w = self.tabs.currentWidget()
        return w if isinstance(w, QTableWidget) else None

    def _on_tab_changed(self, _idx: int) -> None:
        self._update_section_counts_from_current_tab()

    def _update_section_counts_from_current_tab(self) -> None:
        table = self._current_table()
        counts = {"MCQ": 0, "TF": 0, "NUMERIC": 0}
        if table:
            for r in range(table.rowCount()):
                widget = table.cellWidget(r, 1)
                if widget:
                    counts[widget.currentText()] += 1
        self.mcq_count_edit.setText(str(counts["MCQ"]))
        self.tf_count_edit.setText(str(counts["TF"]))
        self.numeric_count_edit.setText(str(counts["NUMERIC"]))

    def _add_empty_row(self) -> None:
        table = self._current_table()
        if not table:
            return
        self._append_row(
            table,
            PreviewRow(question=table.rowCount() + 1, answer_type="MCQ", answer_value="A"),
        )
        self._update_section_counts_from_current_tab()

    def _apply_mapping(self) -> None:
        table = self._current_table()
        if not table:
            return
        try:
            mcq_count = int(self.mcq_count_edit.text().strip() or "0")
            tf_count = int(self.tf_count_edit.text().strip() or "0")
            numeric_count = int(self.numeric_count_edit.text().strip() or "0")
        except ValueError:
            QMessageBox.warning(self, "Invalid mapping", "Section counts must be integer values.")
            return

        total = table.rowCount()
        if mcq_count + tf_count + numeric_count > total:
            QMessageBox.warning(self, "Invalid mapping", "Sum of section counts exceeds total rows in current tab.")
            return

        for r in range(total):
            widget = table.cellWidget(r, 1)
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
            for exam_code, table in self.exam_tables.items():
                key = package.exam_keys.setdefault(exam_code, ImportedAnswerKey(exam_id=exam_id))
                src = self.imported.exam_keys.get(exam_code)
                if src is not None:
                    key.full_credit_questions = {
                        str(sec): [int(x) for x in (vals or []) if str(x).strip().lstrip("-").isdigit()]
                        for sec, vals in (src.full_credit_questions or {}).items()
                    }
                    key.invalid_answer_rows = {
                        str(sec): {int(k): str(v) for k, v in (vals or {}).items()}
                        for sec, vals in (src.invalid_answer_rows or {}).items()
                    }
                for row_idx in range(table.rowCount()):
                    q_item = table.item(row_idx, 0)
                    a_item = table.item(row_idx, 2)
                    if not q_item or not a_item:
                        continue
                    q = int((q_item.text() or "").strip())
                    answer_type = table.cellWidget(row_idx, 1).currentText()
                    answer_text = (a_item.text() or "").strip()

                    if answer_type == "MCQ":
                        val = answer_text.upper()
                        if val not in {"A", "B", "C", "D", "E"}:
                            bucket = key.full_credit_questions.setdefault("MCQ", [])
                            if q not in bucket:
                                bucket.append(q)
                            key.invalid_answer_rows.setdefault("MCQ", {})[q] = answer_text
                        else:
                            key.mcq_answers[q] = val
                    elif answer_type == "NUMERIC":
                        token = answer_text.replace(" ", "").replace(",", ".")
                        if token.startswith(("+", "-")):
                            token = token[1:]
                        if not token or token.count(".") > 1 or not token.replace(".", "").isdigit():
                            bucket = key.full_credit_questions.setdefault("NUMERIC", [])
                            if q not in bucket:
                                bucket.append(q)
                            key.invalid_answer_rows.setdefault("NUMERIC", {})[q] = answer_text
                        else:
                            key.numeric_answers[q] = answer_text
                    else:
                        token = answer_text.replace(" ", "").upper()
                        if len(token) != 4 or any(ch not in {"T", "F", "D", "S", "Đ"} for ch in token):
                            bucket = key.full_credit_questions.setdefault("TF", [])
                            if q not in bucket:
                                bucket.append(q)
                            key.invalid_answer_rows.setdefault("TF", {})[q] = answer_text
                        else:
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
