from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPageSize, QPainter, QPdfWriter
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
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
        self.setWindowFlag(Qt.Window, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)

        self.combo_defs: list[tuple[str, list[str]]] = []
        self._subject_pairs_cache: list[tuple[str, str]] | None = None
        self._student_profile_cache: dict[str, dict[str, str]] | None = None
        self._score_rows_cache: dict[str, list[dict]] = {}
        self._subject_score_cache: dict[str, dict[str, float]] = {}

        self.report_list = QListWidget()
        for name in [
            self.REPORT_SUBJECT_DIST,
            self.REPORT_SUBJECT_STATS,
            self.REPORT_COMBO_RANK,
            self.REPORT_COMBO_DIST,
            self.REPORT_CLASS_SUMMARY,
        ]:
            self.report_list.addItem(name)

        self.exam_label = QLabel(self._current_exam_text())
        self.title_label = QLabel(self.REPORT_SUBJECT_DIST)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.class_combo = QComboBox()
        self.class_combo.addItems(self._collect_class_options())

        self.combo_name_edit = QLineEdit()
        self.combo_name_edit.setPlaceholderText("Tên tổ hợp (VD: A00)")
        subject_options = [label for label, _ in self._collect_subject_pairs()]
        self.combo_sub1 = QComboBox(); self.combo_sub1.addItems(subject_options)
        self.combo_sub2 = QComboBox(); self.combo_sub2.addItems(subject_options)
        self.combo_sub3 = QComboBox(); self.combo_sub3.addItems(subject_options)
        self.combo_list = QListWidget()
        self.btn_add_combo = QPushButton("Thêm tổ hợp")
        self.btn_remove_combo = QPushButton("Xóa tổ hợp")

        self.right_widget = QWidget()
        self.right_form = QFormLayout(self.right_widget)
        self.row_class = QWidget()
        row_class_layout = QVBoxLayout(self.row_class)
        row_class_layout.setContentsMargins(0, 0, 0, 0)
        row_class_layout.addWidget(self.class_combo)
        self.row_combo = QWidget()
        combo_layout = QVBoxLayout(self.row_combo)
        combo_layout.setContentsMargins(0, 0, 0, 0)
        combo_layout.addWidget(self.combo_name_edit)
        combo_layout.addWidget(self.combo_sub1)
        combo_layout.addWidget(self.combo_sub2)
        combo_layout.addWidget(self.combo_sub3)
        row_btns = QHBoxLayout()
        row_btns.addWidget(self.btn_add_combo)
        row_btns.addWidget(self.btn_remove_combo)
        combo_layout.addLayout(row_btns)
        combo_layout.addWidget(self.combo_list)

        self.right_form.addRow("Kỳ thi hiện tại", self.exam_label)
        self.right_form.addRow("Lớp", self.row_class)
        self.right_form.addRow("Chọn tổ hợp", self.row_combo)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(self.title_label)
        center_layout.addWidget(self.preview_table)

        self.btn_preview = QPushButton("Xem trước")
        self.btn_export_excel = QPushButton("Xuất Excel")
        self.btn_export_pdf = QPushButton("Xuất PDF")
        self.btn_close = QPushButton("Đóng")
        bottom_ribbon = QHBoxLayout()
        bottom_ribbon.addStretch(1)
        bottom_ribbon.addWidget(self.btn_preview)
        bottom_ribbon.addWidget(self.btn_export_excel)
        bottom_ribbon.addWidget(self.btn_export_pdf)
        bottom_ribbon.addWidget(self.btn_close)

        layout = QGridLayout(self)
        layout.addWidget(self.report_list, 0, 0)
        layout.addWidget(center, 0, 1)
        layout.addWidget(self.right_widget, 0, 2)
        layout.addLayout(bottom_ribbon, 1, 0, 1, 3)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 7)
        layout.setColumnStretch(2, 3)

        self.report_list.currentTextChanged.connect(self._on_report_changed)
        self.btn_add_combo.clicked.connect(self._add_combo)
        self.btn_remove_combo.clicked.connect(self._remove_combo)
        self.btn_preview.clicked.connect(self.preview_report)
        self.btn_export_excel.clicked.connect(self.export_excel)
        self.btn_export_pdf.clicked.connect(self.export_pdf)
        self.btn_close.clicked.connect(self.close)

        self._last_report: ReportTable | None = None
        self.report_list.setCurrentRow(0)
        self._ensure_default_combo()
        self.preview_report()
        QTimer.singleShot(0, self.showMaximized)

    def _current_exam_text(self) -> str:
        session_id = str(getattr(self.main_window, "current_session_id", "") or "").strip()
        return session_id or "Chưa có kỳ thi"

    def _collect_subject_pairs(self) -> list[tuple[str, str]]:
        if self._subject_pairs_cache is not None:
            return list(self._subject_pairs_cache)
        out: list[tuple[str, str]] = []
        for _label, key in self.main_window._iter_export_subjects():
            cfg = self.main_window._subject_config_by_subject_key(key) or {}
            raw_name = str(cfg.get("name", "") or key).strip()
            clean = raw_name
            clean = clean.replace("Môn:", "").strip()
            clean = clean.split("- Kỳ thi", 1)[0].strip()
            clean = clean.split("- Khối", 1)[0].strip()
            out.append((clean or str(key), key))
        seen: set[str] = set()
        dedup: list[tuple[str, str]] = []
        for label, key in out:
            final = label
            if final in seen:
                final = f"{label} ({key})"
            seen.add(final)
            dedup.append((final, key))
        self._subject_pairs_cache = dedup
        return list(dedup)

    def _score_rows_for_subject_cached(self, subject_key: str) -> list[dict]:
        key = str(subject_key or "").strip()
        if not key:
            return []
        if key not in self._score_rows_cache:
            self._score_rows_cache[key] = list(self.main_window._ensure_export_score_rows_for_subject(key) or [])
        return list(self._score_rows_cache.get(key, []))

    def _collect_class_options(self) -> list[str]:
        vals = ["Tất cả"]
        classes = sorted(
            set(
                str(profile.get("class_name", "") or "").strip()
                for profile in self._student_profile_map().values()
                if str(profile.get("class_name", "") or "").strip()
            )
        )
        vals.extend(classes)
        return vals

    def _student_profile_map(self) -> dict[str, dict[str, str]]:
        if self._student_profile_cache is not None:
            return dict(self._student_profile_cache)
        out: dict[str, dict[str, str]] = {}
        if self.main_window.session:
            for st in (self.main_window.session.students or []):
                sid = str(getattr(st, "student_id", "") or "").strip()
                if not sid:
                    continue
                out[sid] = {
                    "name": str(getattr(st, "name", "") or ""),
                    "birth_date": str(getattr(st, "birth_date", "") or ""),
                    "class_name": str(getattr(st, "class_name", "") or ""),
                    "exam_room": str(getattr(st, "exam_room", "") or ""),
                }
        for _label, key in self._collect_subject_pairs():
            for row in self._score_rows_for_subject_cached(key):
                sid = str(row.get("student_id", "") or "").strip()
                if not sid:
                    continue
                rec = out.setdefault(
                    sid,
                    {
                        "name": str(row.get("name", "") or ""),
                        "birth_date": str(row.get("birth_date", "") or ""),
                        "class_name": str(row.get("class_name", "") or ""),
                        "exam_room": str(row.get("exam_room", "") or ""),
                    },
                )
                if not rec.get("name"):
                    rec["name"] = str(row.get("name", "") or "")
                if not rec.get("birth_date"):
                    rec["birth_date"] = str(row.get("birth_date", "") or "")
                if not rec.get("class_name"):
                    rec["class_name"] = str(row.get("class_name", "") or "")
                if not rec.get("exam_room"):
                    rec["exam_room"] = str(row.get("exam_room", "") or "")
        self._student_profile_cache = out
        return dict(out)

    def _ensure_default_combo(self) -> None:
        if self.combo_defs:
            return
        subjects = [label for label, _ in self._collect_subject_pairs()]
        if len(subjects) >= 3:
            self.combo_defs.append(("Mặc định", subjects[:3]))
            self._refresh_combo_list(select_index=0)

    def _selected_combo(self) -> tuple[str, list[str]] | None:
        if not self.combo_defs:
            return None
        row = self.combo_list.currentRow()
        if 0 <= row < len(self.combo_defs):
            return self.combo_defs[row]
        return self.combo_defs[0]

    def _add_combo(self) -> None:
        name = self.combo_name_edit.text().strip() or f"Tổ hợp {len(self.combo_defs) + 1}"
        subjects = [
            self.combo_sub1.currentText().strip(),
            self.combo_sub2.currentText().strip(),
            self.combo_sub3.currentText().strip(),
        ]
        if len(set(subjects)) < 3:
            QMessageBox.warning(self, "Tổ hợp", "3 môn trong tổ hợp phải khác nhau.")
            return
        self.combo_defs.append((name, subjects))
        self._refresh_combo_list(select_index=len(self.combo_defs) - 1)
        self._last_report = None

    def _remove_combo(self) -> None:
        row = self.combo_list.currentRow()
        if row < 0 or row >= len(self.combo_defs):
            return
        self.combo_defs.pop(row)
        next_row = min(row, len(self.combo_defs) - 1)
        self._refresh_combo_list(select_index=next_row)
        self._last_report = None

    def _refresh_combo_list(self, select_index: int | None = None) -> None:
        old_row = self.combo_list.currentRow()
        self.combo_list.clear()
        for name, subjects in self.combo_defs:
            self.combo_list.addItem(f"{name}: {subjects[0]}, {subjects[1]}, {subjects[2]}")
        if self.combo_defs:
            target = old_row if select_index is None else select_index
            target = max(0, min(target, len(self.combo_defs) - 1))
            self.combo_list.setCurrentRow(target)

    def _on_report_changed(self, text: str) -> None:
        self.title_label.setText(text or "Báo cáo thống kê")
        needs_filters = text in {self.REPORT_COMBO_RANK, self.REPORT_COMBO_DIST, self.REPORT_CLASS_SUMMARY}
        self.right_widget.setVisible(needs_filters)
        self.row_class.setVisible(text == self.REPORT_CLASS_SUMMARY)
        self.row_combo.setVisible(text in {self.REPORT_COMBO_RANK, self.REPORT_COMBO_DIST, self.REPORT_CLASS_SUMMARY})

    def _subject_score_by_sid(self, subject_key: str) -> dict[str, float]:
        key = str(subject_key or "").strip()
        if not key:
            return {}
        if key in self._subject_score_cache:
            return dict(self._subject_score_cache[key])
        out: dict[str, float] = {}
        for row in self._score_rows_for_subject_cached(key):
            sid = str(row.get("student_id", "") or "").strip()
            if not sid:
                continue
            score_value = self.main_window._score_value_for_statistics(row)
            if score_value is None:
                continue
            out[sid] = score_value
        self._subject_score_cache[key] = out
        return dict(out)

    def build_subject_distribution_report(self) -> ReportTable:
        headers = ["Mã môn", "Tổng bài", "0–<=1", "1–<2", "2–<3", "3–<4", "4–<5", "5–<6", "6–<7", "7–<8", "8–<9", "9–<10", "=10"]
        rows: list[list[object]] = []
        bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        for label, key in self._collect_subject_pairs():
            subject_rows = self._score_rows_for_subject_cached(key)
            scores: list[float] = []
            for row in subject_rows:
                score_value = self.main_window._score_value_for_statistics(row)
                if score_value is not None:
                    scores.append(score_value)
            counts = [0] * len(bins)
            perfect = 0
            for score in scores:
                if abs(score - 10.0) < 1e-9:
                    perfect += 1
                    continue
                for idx, (lo, hi) in enumerate(bins):
                    if idx == 0 and lo <= score <= hi:
                        counts[idx] += 1
                        break
                    if lo <= score < hi:
                        counts[idx] += 1
                        break
            rows.append([label, len(scores), *counts, perfect])
        return ReportTable(headers, rows)

    def build_subject_stats_report(self) -> ReportTable:
        headers = ["Tên môn", "Điểm trung bình", "Điểm trung vị", "Điểm cao nhất", "Điểm thấp nhất"]
        rows: list[list[object]] = []
        for label, key in self._collect_subject_pairs():
            scores: list[float] = []
            for row in self._score_rows_for_subject_cached(key):
                score_value = self.main_window._score_value_for_statistics(row)
                if score_value is not None:
                    scores.append(score_value)
            if scores:
                rows.append([label, round(mean(scores), 4), round(median(scores), 4), max(scores), min(scores)])
            else:
                rows.append([label, "", "", "", ""])
        return ReportTable(headers, rows)

    def _combo_score_maps(self, subjects: list[str]) -> list[dict[str, float]]:
        label_to_key = {label: key for label, key in self._collect_subject_pairs()}
        return [self._subject_score_by_sid(label_to_key.get(sub, "")) for sub in subjects]

    def build_combo_ranking_report(self) -> ReportTable:
        headers = ["STT", "Họ đệm", "Tên", "Ngày sinh", "Mã lớp", "Phòng thi", "Môn 1", "Môn 2", "Môn 3", "Tổng điểm", "Xếp hạng"]
        selected = self._selected_combo()
        if not selected:
            return ReportTable(headers, [])
        _name, subjects = selected
        score_maps = self._combo_score_maps(subjects)
        rows: list[list[object]] = []
        profiles = self._student_profile_map()
        for sid, profile in profiles.items():
            if not sid:
                continue
            vals = [score_maps[i].get(sid, "") for i in range(3)]
            total = round(sum(float(v) for v in vals), 4) if all(isinstance(v, (int, float)) for v in vals) else ""
            full_name = str(profile.get("name", "") or "").strip()
            parts = full_name.split()
            ho_dem = " ".join(parts[:-1]) if len(parts) > 1 else ""
            ten = parts[-1] if parts else ""
            rows.append([
                0,
                ho_dem,
                ten,
                str(profile.get("birth_date", "") or ""),
                str(profile.get("class_name", "") or ""),
                str(profile.get("exam_room", "") or ""),
                *vals,
                total,
                "",
            ])
        sortable = [(idx, row) for idx, row in enumerate(rows) if isinstance(row[9], (int, float))]
        sortable.sort(key=lambda item: float(item[1][9]), reverse=True)
        rank_by_idx = {idx: rank + 1 for rank, (idx, _row) in enumerate(sortable)}
        for idx, row in enumerate(rows, start=1):
            row[0] = idx
            row[10] = rank_by_idx.get(idx - 1, "")
        return ReportTable(headers, rows)

    def build_combo_distribution_report(self) -> ReportTable:
        headers = ["Khoảng điểm", "Tổng cộng"] + [name for name, _ in self.combo_defs]
        if not self.combo_defs:
            return ReportTable(headers, [])
        bins: list[tuple[float, float]] = []
        cur = 27.0
        while cur < 30.0:
            nxt = round(cur + 0.25, 2)
            bins.append((round(cur, 2), nxt))
            cur = nxt
        bins.append((30.0, 30.0))

        profiles = self._student_profile_map()
        combo_totals: list[list[float]] = []
        for _name, subjects in self.combo_defs:
            maps = self._combo_score_maps(subjects)
            totals: list[float] = []
            for sid in profiles.keys():
                vals = [maps[i].get(sid, None) for i in range(3)]
                if all(isinstance(v, (int, float)) for v in vals):
                    totals.append(float(vals[0]) + float(vals[1]) + float(vals[2]))
            combo_totals.append(totals)

        rows: list[list[object]] = []
        for lo, hi in bins:
            label = f"{lo:.2f}-{hi:.2f}" if lo != hi else "=30.00"
            counts: list[int] = []
            for totals in combo_totals:
                count = 0
                for total in totals:
                    if lo == hi and abs(total - 30.0) < 1e-9:
                        count += 1
                    elif lo <= total < hi:
                        count += 1
                counts.append(count)
            rows.append([label, sum(counts), *counts])
        return ReportTable(headers, rows)

    def build_class_summary_report(self) -> ReportTable:
        subjects = self._collect_subject_pairs()
        score_by_subject = {label: self._subject_score_by_sid(key) for label, key in subjects}
        combo_score_maps = {
            combo_name: self._combo_score_maps(combo_subjects)
            for combo_name, combo_subjects in self.combo_defs
        }
        headers = ["STT", "SBD", "Họ tên", "Ngày sinh", "Mã lớp", "Phòng thi"] + [label for label, _ in subjects] + [name for name, _ in self.combo_defs]
        grouped: dict[str, list[list[object]]] = {}
        profiles = self._student_profile_map()
        for sid, profile in profiles.items():
            if not sid:
                continue
            cls = str(profile.get("class_name", "") or "").strip() or "(Không lớp)"
            row = [0, sid, str(profile.get("name", "") or ""), str(profile.get("birth_date", "") or ""), cls, str(profile.get("exam_room", "") or "")]
            for label, _ in subjects:
                row.append(score_by_subject.get(label, {}).get(sid, ""))
            for combo_name, _combo_subjects in self.combo_defs:
                maps = combo_score_maps.get(combo_name, [])
                vals = [maps[i].get(sid, None) for i in range(3)] if len(maps) >= 3 else [None, None, None]
                row.append(round(sum(float(v) for v in vals), 4) if all(isinstance(v, (int, float)) for v in vals) else "")
            grouped.setdefault(cls, []).append(row)

        selected_class = self.class_combo.currentText().strip()
        if selected_class and selected_class != "Tất cả":
            grouped = {selected_class: grouped.get(selected_class, [])}

        all_rows: list[list[object]] = []
        for cls in sorted(grouped):
            rows = grouped[cls]
            rows.sort(key=lambda item: str(item[1]))
            for idx, row in enumerate(rows, start=1):
                row[0] = idx
            if rows:
                avg_row = ["", "", "", "", "Trung bình lớp", ""]
                for col_idx in range(6, len(headers)):
                    vals = [float(r[col_idx]) for r in rows if isinstance(r[col_idx], (int, float))]
                    avg_row.append(round(mean(vals), 4) if vals else "")
                rows.append(avg_row)
            all_rows.extend(rows)
        return ReportTable(headers, all_rows, grouped)

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
        if not str(getattr(self.main_window, "current_session_id", "") or "").strip():
            QMessageBox.information(self, "Báo cáo thống kê", "Vui lòng chọn kỳ thi hiện tại trước khi xem báo cáo.")
            return
        report = self._build_report()
        self._last_report = report
        self._render_table(report)

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
                painter.drawText(x, y, header_line)
                y += line_h
        painter.end()
        QMessageBox.information(self, "Báo cáo", f"Đã xuất PDF:\n{path}")