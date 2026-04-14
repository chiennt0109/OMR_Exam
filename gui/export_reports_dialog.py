from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtGui import QPageSize
from PySide6.QtGui import QPdfWriter
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass
class ReportTable:
    headers: list[str]
    rows: list[list[object]]
    grouped_rows: dict[str, list[list[object]]] | None = None


class ExportReportsDialog(QDialog):
    REPORT_SUBJECT_DIST = "Phổ điểm theo môn"
    REPORT_SUBJECT_STATS = "Chỉ số theo môn"
    REPORT_COMBO_RANK = "Bảng điểm theo tổ hợp"
    REPORT_COMBO_DIST = "Phổ điểm tổ hợp"
    REPORT_CLASS_SUMMARY = "Tổng hợp theo lớp"

    def __init__(self, parent_window) -> None:
        super().__init__(parent_window)
        self.main_window = parent_window
        self.setWindowTitle("Báo cáo thống kê")
        self.resize(1200, 720)

        self.report_list = QListWidget()
        for name in [
            self.REPORT_SUBJECT_DIST,
            self.REPORT_SUBJECT_STATS,
            self.REPORT_COMBO_RANK,
            self.REPORT_COMBO_DIST,
            self.REPORT_CLASS_SUMMARY,
        ]:
            self.report_list.addItem(name)

        self.title_label = QLabel(self.REPORT_SUBJECT_DIST)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.exam_combo = QComboBox()
        self.exam_combo.addItems(self._collect_exam_options())
        self.grade_combo = QComboBox()
        self.grade_combo.addItems(self._collect_grade_options())
        self.class_combo = QComboBox()
        self.class_combo.addItems(self._collect_class_options())
        self.subject_combo = QComboBox()
        self.subject_combo.addItems(self._collect_subject_options())
        self.combo_text = QTextEdit()
        self.combo_text.setPlaceholderText("VD: A00: TOAN,LY,HOA\nB00: TOAN,HOA,SINH")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Excel", "PDF"])

        filter_widget = QWidget()
        filter_form = QFormLayout(filter_widget)
        filter_form.addRow("Kỳ thi", self.exam_combo)
        filter_form.addRow("Khối", self.grade_combo)
        filter_form.addRow("Lớp", self.class_combo)
        filter_form.addRow("Môn", self.subject_combo)
        filter_form.addRow("Tổ hợp", self.combo_text)
        filter_form.addRow("Định dạng", self.format_combo)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(self.title_label)
        center_layout.addWidget(self.preview_table)

        btn_preview = QPushButton("Xem trước")
        btn_export_excel = QPushButton("Xuất Excel")
        btn_export_pdf = QPushButton("Xuất PDF")
        btn_close = QPushButton("Đóng")

        btn_bar = QHBoxLayout()
        btn_bar.addStretch(1)
        btn_bar.addWidget(btn_preview)
        btn_bar.addWidget(btn_export_excel)
        btn_bar.addWidget(btn_export_pdf)
        btn_bar.addWidget(btn_close)

        layout = QGridLayout(self)
        layout.addWidget(self.report_list, 0, 0)
        layout.addWidget(center, 0, 1)
        layout.addWidget(filter_widget, 0, 2)
        layout.addLayout(btn_bar, 1, 0, 1, 3)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 6)
        layout.setColumnStretch(2, 3)

        self.report_list.currentTextChanged.connect(self._on_report_changed)
        btn_preview.clicked.connect(self.preview_report)
        btn_export_excel.clicked.connect(self.export_excel)
        btn_export_pdf.clicked.connect(self.export_pdf)
        btn_close.clicked.connect(self.close)

        self._last_report: ReportTable | None = None
        self.report_list.setCurrentRow(0)
        self.preview_report()

    def _collect_subject_pairs(self) -> list[tuple[str, str]]:
        return list(self.main_window._iter_export_subjects())

    def _collect_subject_options(self) -> list[str]:
        labels = ["Tất cả"]
        labels.extend([label for label, _ in self._collect_subject_pairs()])
        return labels

    def _collect_class_options(self) -> list[str]:
        out = ["Tất cả"]
        classes = self.main_window._class_options_for_export() if hasattr(self.main_window, "_class_options_for_export") else []
        out.extend(classes)
        return out

    def _collect_exam_options(self) -> list[str]:
        return ["Tất cả"]

    def _collect_grade_options(self) -> list[str]:
        return ["Tất cả"]

    def _on_report_changed(self, text: str) -> None:
        self.title_label.setText(text or "Báo cáo thống kê")

    def _subject_score_by_sid(self, subject_key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in self.main_window._score_rows_for_subject(subject_key):
            sid = str(row.get("student_id", "") or "").strip()
            if not sid:
                continue
            try:
                out[sid] = float(row.get("score", "") or 0)
            except Exception:
                continue
        return out

    def _parse_combos(self) -> list[tuple[str, list[str]]]:
        parsed: list[tuple[str, list[str]]] = []
        for line in self.combo_text.toPlainText().splitlines():
            txt = line.strip()
            if not txt:
                continue
            if ":" in txt:
                name, rhs = txt.split(":", 1)
                subjects = [x.strip() for x in rhs.split(",") if x.strip()]
                if len(subjects) == 3:
                    parsed.append((name.strip() or "Tổ hợp", subjects))
        if parsed:
            return parsed
        pairs = self._collect_subject_pairs()
        if len(pairs) >= 3:
            parsed.append(("Mặc định", [pairs[0][0], pairs[1][0], pairs[2][0]]))
        return parsed

    def build_subject_distribution_report(self) -> ReportTable:
        headers = ["Mã môn", "Tổng bài", "0–<=1", "1–<2", "2–<3", "3–<4", "4–<5", "5–<6", "6–<7", "7–<8", "8–<9", "9–<10", "=10"]
        rows: list[list[object]] = []
        bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        for label, key in self._collect_subject_pairs():
            scores = []
            for row in self.main_window._score_rows_for_subject(key):
                try:
                    scores.append(float(row.get("score", "") or 0))
                except Exception:
                    pass
            counts = [0] * len(bins)
            perfect = 0
            for s in scores:
                if s == 10:
                    perfect += 1
                    continue
                for idx, (lo, hi) in enumerate(bins):
                    if idx == 0 and lo <= s <= hi:
                        counts[idx] += 1
                        break
                    if lo <= s < hi:
                        counts[idx] += 1
                        break
            rows.append([key or label, len(scores), *counts, perfect])
        return ReportTable(headers=headers, rows=rows)

    def build_subject_stats_report(self) -> ReportTable:
        headers = ["Mã/Tên môn", "Điểm trung bình", "Điểm trung vị", "Điểm cao nhất", "Điểm thấp nhất"]
        rows: list[list[object]] = []
        for label, key in self._collect_subject_pairs():
            scores: list[float] = []
            for row in self.main_window._score_rows_for_subject(key):
                try:
                    scores.append(float(row.get("score", "") or 0))
                except Exception:
                    pass
            if scores:
                rows.append([label, round(mean(scores), 4), round(median(scores), 4), max(scores), min(scores)])
            else:
                rows.append([label, "", "", "", ""])
        return ReportTable(headers=headers, rows=rows)

    def build_combo_ranking_report(self) -> ReportTable:
        combos = self._parse_combos()
        headers = ["STT", "Họ đệm", "Tên", "Ngày sinh", "Mã lớp", "Môn 1", "Môn 2", "Môn 3", "Tổng điểm", "Xếp hạng"]
        if not combos:
            return ReportTable(headers=headers, rows=[])
        _combo_name, combo_subjects = combos[0]
        label_to_key = {label: key for label, key in self._collect_subject_pairs()}
        score_maps = [self._subject_score_by_sid(label_to_key.get(name, "")) for name in combo_subjects]
        rows: list[list[object]] = []
        students = list(self.main_window.session.students or []) if self.main_window.session else []
        for st in students:
            sid = str(getattr(st, "student_id", "") or "").strip()
            if not sid:
                continue
            vals = [score_maps[i].get(sid, "") for i in range(3)]
            total = ""
            if all(isinstance(v, (int, float)) for v in vals):
                total = round(float(vals[0]) + float(vals[1]) + float(vals[2]), 4)
            full_name = str(getattr(st, "name", "") or "").strip()
            parts = full_name.split()
            ten = parts[-1] if parts else ""
            ho_dem = " ".join(parts[:-1]) if len(parts) > 1 else ""
            rows.append([0, ho_dem, ten, str(getattr(st, "birth_date", "") or ""), str(getattr(st, "class_name", "") or ""), *vals, total, ""])
        sortable = [(idx, r) for idx, r in enumerate(rows) if isinstance(r[8], (int, float))]
        sortable.sort(key=lambda x: float(x[1][8]), reverse=True)
        rank_by_idx = {idx: rank + 1 for rank, (idx, _row) in enumerate(sortable)}
        for idx, row in enumerate(rows, start=1):
            row[0] = idx
            row[9] = rank_by_idx.get(idx - 1, "")
        return ReportTable(headers=headers, rows=rows)

    def build_combo_distribution_report(self) -> ReportTable:
        combos = self._parse_combos()
        headers = ["Khoảng điểm", "Tổng cộng"] + [name for name, _ in combos]
        rows: list[list[object]] = []
        if not combos:
            return ReportTable(headers=headers, rows=rows)
        bins = []
        cur = 27.0
        while cur < 30.0:
            nxt = round(cur + 0.25, 2)
            bins.append((round(cur, 2), nxt))
            cur = nxt
        bins.append((30.0, 30.0))

        label_to_key = {label: key for label, key in self._collect_subject_pairs()}
        combo_totals: list[list[float]] = []
        students = list(self.main_window.session.students or []) if self.main_window.session else []
        for _name, subjects in combos:
            maps = [self._subject_score_by_sid(label_to_key.get(sub, "")) for sub in subjects]
            totals: list[float] = []
            for st in students:
                sid = str(getattr(st, "student_id", "") or "").strip()
                vals = [maps[i].get(sid, None) for i in range(3)]
                if all(isinstance(v, (int, float)) for v in vals):
                    totals.append(float(vals[0]) + float(vals[1]) + float(vals[2]))
            combo_totals.append(totals)

        for lo, hi in bins:
            label = f"{lo:.2f} - {hi:.2f}" if lo != hi else "=30.00"
            counts = []
            for totals in combo_totals:
                c = 0
                for t in totals:
                    if lo == hi and abs(t - 30.0) < 1e-9:
                        c += 1
                    elif lo <= t < hi:
                        c += 1
                counts.append(c)
            rows.append([label, sum(counts), *counts])
        return ReportTable(headers=headers, rows=rows)

    def build_class_summary_report(self) -> ReportTable:
        subjects = self._collect_subject_pairs()
        score_by_subject = {label: self._subject_score_by_sid(key) for label, key in subjects}
        combo_defs = self._parse_combos()
        headers = ["STT", "SBD", "Họ tên", "Ngày sinh", "Mã lớp"] + [label for label, _ in subjects] + [name for name, _subs in combo_defs]
        grouped: dict[str, list[list[object]]] = {}
        students = list(self.main_window.session.students or []) if self.main_window.session else []
        for st in students:
            sid = str(getattr(st, "student_id", "") or "").strip()
            if not sid:
                continue
            cls = str(getattr(st, "class_name", "") or "").strip() or "(Không lớp)"
            row = [0, sid, str(getattr(st, "name", "") or ""), str(getattr(st, "birth_date", "") or ""), cls]
            subj_vals = []
            for label, _ in subjects:
                val = score_by_subject.get(label, {}).get(sid, "")
                subj_vals.append(val)
            row.extend(subj_vals)
            label_to_key = {label: key for label, key in subjects}
            for _name, subs in combo_defs:
                combo_vals = [self._subject_score_by_sid(label_to_key.get(s, "")).get(sid, None) for s in subs]
                if all(isinstance(v, (int, float)) for v in combo_vals):
                    row.append(round(sum(float(v) for v in combo_vals), 4))
                else:
                    row.append("")
            grouped.setdefault(cls, []).append(row)

        all_rows: list[list[object]] = []
        for cls in sorted(grouped):
            rows = grouped[cls]
            rows.sort(key=lambda x: str(x[1]))
            for idx, row in enumerate(rows, start=1):
                row[0] = idx
            if rows:
                avg_row = ["", "", "", "", "Trung bình lớp"]
                for col_idx in range(5, len(headers)):
                    vals = [float(r[col_idx]) for r in rows if isinstance(r[col_idx], (int, float))]
                    avg_row.append(round(mean(vals), 4) if vals else "")
                rows.append(avg_row)
            all_rows.extend(rows)
        selected_class = self.class_combo.currentText().strip()
        if selected_class and selected_class != "Tất cả":
            target_rows = grouped.get(selected_class, [])
            return ReportTable(headers=headers, rows=target_rows, grouped_rows={selected_class: target_rows})
        return ReportTable(headers=headers, rows=all_rows, grouped_rows=grouped)

    def _build_report(self) -> ReportTable:
        name = self.report_list.currentItem().text() if self.report_list.currentItem() else self.REPORT_SUBJECT_DIST
        if name == self.REPORT_SUBJECT_DIST:
            return self.build_subject_distribution_report()
        if name == self.REPORT_SUBJECT_STATS:
            return self.build_subject_stats_report()
        if name == self.REPORT_COMBO_RANK:
            return self.build_combo_ranking_report()
        if name == self.REPORT_COMBO_DIST:
            return self.build_combo_distribution_report()
        return self.build_class_summary_report()

    def _render_table(self, table: ReportTable) -> None:
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(table.headers))
        self.preview_table.setHorizontalHeaderLabels(table.headers)
        self.preview_table.setRowCount(len(table.rows))
        for r_idx, row in enumerate(table.rows):
            for c_idx, value in enumerate(row):
                item = QTableWidgetItem("" if value is None else str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.preview_table.setItem(r_idx, c_idx, item)
        self.preview_table.resizeColumnsToContents()

    def preview_report(self) -> None:
        table = self._build_report()
        self._last_report = table
        self._render_table(table)

    def export_excel(self) -> None:
        if self._last_report is None:
            self.preview_report()
        if self._last_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Xuất Excel", "report.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        from openpyxl import Workbook
        wb = Workbook()
        report_name = self.report_list.currentItem().text() if self.report_list.currentItem() else "report"
        if report_name == self.REPORT_CLASS_SUMMARY and self._last_report.grouped_rows:
            if wb.active:
                wb.remove(wb.active)
            for cls, rows in self._last_report.grouped_rows.items():
                ws = wb.create_sheet(self.main_window._safe_sheet_name(cls, fallback="class"))
                ws.append(self._last_report.headers)
                for row in rows:
                    ws.append(row)
        else:
            ws = wb.active
            ws.title = "report"
            ws.append(self._last_report.headers)
            for row in self._last_report.rows:
                ws.append(row)
        wb.save(Path(path))
        QMessageBox.information(self, "Báo cáo", f"Đã xuất Excel:\n{path}")

    def export_pdf(self) -> None:
        if self._last_report is None:
            self.preview_report()
        if self._last_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Xuất PDF", "report.pdf", "PDF (*.pdf)")
        if not path:
            return
        writer = QPdfWriter(path)
        writer.setPageSize(QPageSize(QPageSize.A4))
        painter = QPainter(writer)
        x = 40
        y = 60
        line_h = 22
        header_line = " | ".join(self._last_report.headers)
        painter.drawText(x, y, header_line)
        y += line_h
        for row in self._last_report.rows:
            painter.drawText(x, y, " | ".join("" if v is None else str(v) for v in row))
            y += line_h
            if y > writer.height() - 60:
                writer.newPage()
                y = 60
        painter.end()
        QMessageBox.information(self, "Báo cáo", f"Đã xuất PDF:\n{path}")
