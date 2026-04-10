from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.omr_engine import OMRResult


def open_scoring_review_editor_dialog(self, subject_key: str, result: OMRResult) -> None:
    self._ensure_answer_keys_for_subject(subject_key)
    key = self.answer_keys.get_flexible(subject_key, str(getattr(result, "exam_code", "") or "").strip()) if self.answer_keys else None
    if key is None:
        QMessageBox.warning(self, "Tính điểm", "Không tìm thấy đáp án cho mã đề hiện tại để mở màn hình giải trình.")
        return
    dlg = QDialog(self)
    dlg.setWindowTitle(f"Giải trình điểm - {str(getattr(result, 'student_id', '') or '-')}")
    dlg.resize(1320, 760)
    lay = QVBoxLayout(dlg)
    top_form = QFormLayout()
    inp_sid = QComboBox(dlg)
    inp_sid.setEditable(True)
    sid_current = str(getattr(result, "student_id", "") or "-").strip()
    sid_display_map: dict[str, str] = {}
    sid_reverse_map: dict[str, str] = {}
    def _sid_label(sid_text: str) -> str:
        profile = self._student_profile_by_id(sid_text)
        label = f"[{sid_text}] - {str(profile.get('name', '') or '-')} - {str(profile.get('class_name', '') or '-')}"
        sid_display_map[sid_text] = label
        sid_reverse_map[label] = sid_text
        return label

    inp_sid.addItem(_sid_label(sid_current), sid_current)
    if self.session:
        for st in (self.session.students or []):
            sid_val = str(getattr(st, "student_id", "") or "").strip()
            if sid_val and inp_sid.findData(sid_val) < 0:
                inp_sid.addItem(_sid_label(sid_val), sid_val)
    inp_sid.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    inp_sid.setMinimumContentsLength(32)
    current_sid_idx = inp_sid.findData(sid_current)
    if current_sid_idx >= 0:
        inp_sid.setCurrentIndex(current_sid_idx)
    inp_code = QComboBox(dlg)
    inp_code.setEditable(True)
    code_current = str(getattr(result, "exam_code", "") or "-").strip()
    if code_current:
        inp_code.addItem(code_current, code_current)
    code_candidates: set[str] = set()
    code_candidates.update(str(x).strip() for x in (self._fetch_answer_keys_for_subject_scoped(subject_key) or {}).keys() if str(x).strip())
    if self.answer_keys:
        code_candidates.update(
            str(item.exam_code).strip()
            for item in self.answer_keys.keys.values()
            if str(getattr(item, "subject", "") or "").strip() == str(subject_key or "").strip() and str(getattr(item, "exam_code", "") or "").strip()
        )
    for code in sorted(code_candidates):
        if inp_code.findData(code) < 0 and inp_code.findText(code) < 0:
            inp_code.addItem(code, code)
    inp_code.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    inp_code.setMinimumContentsLength(16)

    def _fit_combo_popup_width(combo: QComboBox) -> None:
        try:
            fm = combo.fontMetrics()
            longest = max((len(combo.itemText(i)) for i in range(combo.count())), default=12)
            combo.view().setMinimumWidth(max(combo.width(), fm.averageCharWidth() * max(12, longest) + 40))
        except Exception:
            pass
    _fit_combo_popup_width(inp_sid)
    _fit_combo_popup_width(inp_code)
    def _attach_combo_filter(combo: QComboBox) -> None:
        if combo.lineEdit() is None:
            return
        completer = QCompleter([combo.itemText(i) for i in range(combo.count())], combo)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        combo.setCompleter(completer)
    _attach_combo_filter(inp_sid)
    _attach_combo_filter(inp_code)
    top_form.addRow("SBD", inp_sid)
    top_form.addRow("Mã đề", inp_code)
    lay.addLayout(top_form)

    splitter = QSplitter(Qt.Horizontal, dlg)
    left = QWidget(splitter)
    right = QWidget(splitter)
    left_lay = QVBoxLayout(left)
    right_lay = QVBoxLayout(right)

    grid = QTableWidget(0, 5, left)
    grid.setHorizontalHeaderLabels(["Phần", "Câu", "Đáp án", "Bài làm", "Điểm"])
    grid.verticalHeader().setVisible(False)
    grid.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
    grid.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    grid.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
    grid.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
    grid.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
    left_lay.addWidget(grid, 1)
    _orig_key_release = grid.keyReleaseEvent

    def _grid_key_release(event) -> None:
        _orig_key_release(event)
        if grid.currentColumn() != 3:
            return
        sec_item = grid.item(grid.currentRow(), 0)
        section_text = str(sec_item.text() if sec_item else "").strip().upper()
        cur_item = grid.currentItem()
        if cur_item is not None:
            txt = str(cur_item.text() or "")
            up_txt = txt.upper()
            if txt != up_txt:
                grid.blockSignals(True)
                cur_item.setText(up_txt)
                grid.blockSignals(False)
        key_code = int(getattr(event, "key", lambda: 0)() or 0)
        move_next = False
        if section_text == "MCQ":
            move_next = (65 <= key_code <= 90) or (48 <= key_code <= 57)
        elif section_text in {"TF", "NUMERIC"}:
            move_next = key_code in {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab}
        if move_next:
            r = grid.currentRow()
            if 0 <= r < grid.rowCount() - 1:
                grid.setCurrentCell(r + 1, 3)
            elif r == grid.rowCount() - 1:
                grid.setCurrentCell(r, 3)

    grid.keyReleaseEvent = _grid_key_release  # type: ignore[method-assign]

    def _normalize_grid_cell_text(item: QTableWidgetItem) -> None:
        if item is None or item.column() != 3:
            return
        txt = str(item.text() or "")
        up_txt = txt.upper()
        if txt != up_txt:
            grid.blockSignals(True)
            item.setText(up_txt)
            grid.blockSignals(False)
        row_idx = item.row()
        sec = str(grid.item(row_idx, 0).text() if grid.item(row_idx, 0) else "").strip().upper()
        q_item = grid.item(row_idx, 1)
        q_no = int(q_item.data(Qt.UserRole) if q_item and q_item.data(Qt.UserRole) is not None else 0)
        answer_text = str(grid.item(row_idx, 2).text() if grid.item(row_idx, 2) else "").strip()
        student_text = str(grid.item(row_idx, 3).text() if grid.item(row_idx, 3) else "").strip()
        points_item = grid.item(row_idx, 4)
        if q_no > 0 and sec in {"MCQ", "TF", "NUMERIC"} and points_item is not None:
            points_item.setText(f"{_points_for_row(sec, q_no, answer_text, student_text):g}")

    grid.itemChanged.connect(_normalize_grid_cell_text)

    score_info = QLabel("")
    score_info.setWordWrap(True)
    left_lay.addWidget(score_info)

    right_lay.addWidget(QLabel("Ảnh bài làm"))
    preview = QLabel("Không tải được ảnh bài làm")
    preview.setAlignment(Qt.AlignCenter)
    preview_scroll = QScrollArea(right)
    preview_scroll.setWidgetResizable(True)
    preview_scroll.setWidget(preview)
    right_lay.addWidget(preview_scroll, 1)

    def _tf_compact(value: object) -> str:
        if isinstance(value, dict):
            out = []
            for key_flag in ["a", "b", "c", "d"]:
                if key_flag in value:
                    out.append("Đ" if bool(value.get(key_flag)) else "S")
            return "".join(out)
        text = str(value or "").strip().upper()
        if not text:
            return ""
        if ":" in text or "," in text:
            out = []
            for token in [x.strip() for x in text.split(",") if x.strip()]:
                _, _, val = token.partition(":")
                marker = str(val or token).strip().upper()
                out.append("Đ" if marker in {"T", "TRUE", "1", "Đ", "D", "ĐÚNG", "DUNG"} else "S")
            return "".join(out)
        return "".join("Đ" if ch in {"T", "Đ", "D", "1"} else "S" for ch in text if ch in {"T", "F", "Đ", "D", "S", "1", "0"})

    def _value_by_actual_or_display(answer_map: dict, q_actual: int, q_display: int) -> object:
        if q_actual in answer_map:
            return answer_map.get(q_actual)
        if q_display in answer_map:
            return answer_map.get(q_display)
        if str(q_actual) in answer_map:
            return answer_map.get(str(q_actual))
        if str(q_display) in answer_map:
            return answer_map.get(str(q_display))
        return ""

    def _points_for_row(section: str, q_actual: int, answer: str, student: str) -> float:
        subject_cfg_local = self._subject_config_by_subject_key(subject_key) or {}
        answer_text = str(answer or "").strip().upper()
        student_text = str(student or "").strip().upper()
        if answer_text == "G":
            return float(self.scoring_engine._question_score(section, q_actual, key, subject_cfg_local))
        if section == "MCQ":
            if student_text and student_text == answer_text:
                return float(self.scoring_engine._question_score(section, q_actual, key, subject_cfg_local))
            return 0.0
        if section == "NUMERIC":
            answer_norm = self.scoring_engine._normalize_numeric_text(answer_text)
            student_norm = self.scoring_engine._normalize_numeric_text(student_text)
            if answer_norm and student_norm and answer_norm == student_norm:
                return float(self.scoring_engine._question_score(section, q_actual, key, subject_cfg_local))
            return 0.0
        if section == "TF":
            mode = self.scoring_engine._score_mode(subject_cfg_local)
            width = max(1, min(len(answer_text), 4))
            matched = 0
            for expected, actual in zip(answer_text[:width], student_text[:width]):
                if expected == actual or expected == "G":
                    matched += 1
            if mode == "Điểm theo phần":
                sec_scores = (subject_cfg_local.get("section_scores", {}) or {}) if isinstance(subject_cfg_local, dict) else {}
                tf_rule_cfg = ((sec_scores.get("TF") or {}).get("rule_per_question") or {})
                tf_full_points = self.scoring_engine._to_float(((sec_scores.get("TF") or {}).get("total_points")), 0.0)
            else:
                tf_rule_cfg = ((subject_cfg_local.get("question_scores", {}) or {}).get("TF", {}) if isinstance(subject_cfg_local, dict) else {}) or {}
                tf_full_points = self.scoring_engine._question_score(section, q_actual, key, subject_cfg_local)
            tf_rule_points = {
                0: 0.0,
                1: max(0.0, self.scoring_engine._to_float(tf_rule_cfg.get("1"), 0.1)),
                2: max(0.0, self.scoring_engine._to_float(tf_rule_cfg.get("2"), 0.25)),
                3: max(0.0, self.scoring_engine._to_float(tf_rule_cfg.get("3"), 0.5)),
                4: max(0.0, self.scoring_engine._to_float(tf_rule_cfg.get("4"), tf_full_points)),
            }
            return float(tf_rule_points.get(matched, 0.0))
        return 0.0

    def _append_row(section: str, q_display: int, q_actual: int, answer: str, student: str) -> None:
        r = grid.rowCount()
        grid.insertRow(r)
        grid.setItem(r, 0, QTableWidgetItem(section))
        q_item = QTableWidgetItem(str(q_display))
        q_item.setData(Qt.UserRole, int(q_actual))
        grid.setItem(r, 1, q_item)
        answer_item = QTableWidgetItem(str(answer))
        if str(answer).strip().upper() == "G":
            answer_item.setBackground(QColor(255, 244, 179))
        grid.setItem(r, 2, answer_item)
        edit_item = QTableWidgetItem(str(student))
        if str(answer).strip().upper() != str(student).strip().upper():
            edit_item.setBackground(QColor(255, 225, 225))
        grid.setItem(r, 3, edit_item)
        points_item = QTableWidgetItem(f"{_points_for_row(section, q_actual, answer, student):g}")
        points_item.setFlags(points_item.flags() & ~Qt.ItemIsEditable)
        grid.setItem(r, 4, points_item)

    section_limits = self._subject_section_question_counts(subject_key)
    invalid_rows = dict(getattr(key, "invalid_answer_rows", {}) or {})
    mcq_qs = sorted(
        {
            int(x)
            for x in list((key.answers or {}).keys()) + list((invalid_rows.get("MCQ", {}) or {}).keys())
        }
    )
    tf_qs = sorted(
        {
            int(x)
            for x in list((key.true_false_answers or {}).keys()) + list((invalid_rows.get("TF", {}) or {}).keys())
        }
    )
    numeric_qs = sorted(
        {
            int(x)
            for x in list((key.numeric_answers or {}).keys()) + list((invalid_rows.get("NUMERIC", {}) or {}).keys())
        }
    )
    mcq_limit = int(section_limits.get("MCQ", 0) or 0)
    tf_limit = int(section_limits.get("TF", 0) or 0)
    numeric_limit = int(section_limits.get("NUMERIC", 0) or 0)
    if mcq_limit > 0:
        mcq_qs = mcq_qs[:mcq_limit]
    if tf_limit > 0:
        tf_qs = tf_qs[:tf_limit]
    if numeric_limit > 0:
        numeric_qs = numeric_qs[:numeric_limit]
    for q_display, q_actual in enumerate(mcq_qs, start=1):
        student_val = _value_by_actual_or_display((result.mcq_answers or {}), q_actual, q_display)
        answer_val = str(
            (key.answers or {}).get(
                q_actual,
                (invalid_rows.get("MCQ", {}) or {}).get(q_actual, ""),
            )
            or ""
        )
        _append_row("MCQ", q_display, q_actual, answer_val, str(student_val or ""))
    for q_display, q_actual in enumerate(tf_qs, start=1):
        answer_flags = (key.true_false_answers or {}).get(q_actual, {}) or {}
        student_flags = _value_by_actual_or_display((result.true_false_answers or {}), q_actual, q_display)
        answer_text = _tf_compact(answer_flags)
        if not answer_text:
            answer_text = str((invalid_rows.get("TF", {}) or {}).get(q_actual, "") or "").strip().upper()
        _append_row("TF", q_display, q_actual, answer_text, _tf_compact(student_flags))
    for q_display, q_actual in enumerate(numeric_qs, start=1):
        student_val = _value_by_actual_or_display((result.numeric_answers or {}), q_actual, q_display)
        answer_val = str(
            (key.numeric_answers or {}).get(
                q_actual,
                (invalid_rows.get("NUMERIC", {}) or {}).get(q_actual, ""),
            )
            or ""
        )
        _append_row("NUMERIC", q_display, q_actual, answer_val, str(student_val or ""))

    subject_cfg = self._subject_config_by_subject_key(subject_key) or {}
    try:
        scored_row = self.scoring_engine.score(
            result,
            key,
            student_name=str(getattr(result, "full_name", "") or ""),
            subject_config=subject_cfg,
        )
        formula_txt = self.scoring_engine.describe_formula(key, subject_cfg)
        score_info.setText(f"Điểm hiện tại: {float(getattr(scored_row, 'score', 0.0) or 0.0):g}\nCông thức: {formula_txt}")
    except Exception:
        score_info.setText("Điểm hiện tại: -")

    pix = QPixmap(str(getattr(result, "image_path", "") or ""))
    if pix.isNull():
        preview.setText("Không tải được ảnh bài làm")
    else:
        preview.setText("")
        scaled = pix.scaledToWidth(640, Qt.SmoothTransformation)
        preview.setPixmap(scaled)
        preview.setMinimumSize(scaled.size())

    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([650, 650])
    lay.addWidget(splitter, 1)

    def _current_dialog_snapshot() -> tuple[str, str, list[tuple[str, int, str]]]:
        sid_text = str(inp_sid.currentData() or inp_sid.currentText() or "").strip()
        if sid_text in sid_reverse_map:
            sid_text = sid_reverse_map[sid_text]
        if sid_text.startswith("[") and "]" in sid_text:
            sid_text = sid_text[1:].split("]", 1)[0].strip()
        code_text = str(inp_code.currentData() or inp_code.currentText() or "").strip()
        rows_payload: list[tuple[str, int, str]] = []
        for rr in range(grid.rowCount()):
            sec = str(grid.item(rr, 0).text() if grid.item(rr, 0) else "").strip()
            q_item = grid.item(rr, 1)
            q_no = int(q_item.data(Qt.UserRole) if q_item and q_item.data(Qt.UserRole) is not None else 0)
            val = str(grid.item(rr, 3).text() if grid.item(rr, 3) else "").strip()
            rows_payload.append((sec, q_no, val))
        return sid_text, code_text, rows_payload

    initial_snapshot = _current_dialog_snapshot()
    buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
    lay.addWidget(buttons)
    def _accept_with_confirm() -> None:
        if QMessageBox.question(
            dlg,
            "Xác nhận lưu",
            "Việc lưu sẽ cập nhật dữ liệu bài làm và điểm. Bạn có chắc muốn lưu?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) == QMessageBox.Yes:
            dlg.accept()
    def _reject_with_confirm() -> None:
        if _current_dialog_snapshot() != initial_snapshot:
            choice = QMessageBox.question(
                dlg,
                "Hủy chỉnh sửa",
                "Bạn đang có thay đổi chưa lưu. Hủy sẽ mất dữ liệu vừa sửa. Tiếp tục hủy?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return
        dlg.reject()
    buttons.accepted.connect(_accept_with_confirm)
    buttons.rejected.connect(_reject_with_confirm)
    if dlg.exec() != QDialog.Accepted:
        return

    for r in range(grid.rowCount()):
        section = str(grid.item(r, 0).text() if grid.item(r, 0) else "").strip().upper()
        q_item = grid.item(r, 1)
        q_no = int(q_item.data(Qt.UserRole) if q_item and q_item.data(Qt.UserRole) is not None else 0)
        student_txt = str(grid.item(r, 3).text() if grid.item(r, 3) else "").strip()
        if q_no <= 0:
            continue
        if section == "MCQ":
            target_key = q_no if q_no in (result.mcq_answers or {}) else int(grid.item(r, 1).text() if grid.item(r, 1) else q_no)
            result.mcq_answers[target_key] = student_txt.upper()[:1]
        elif section == "NUMERIC":
            target_key = q_no if q_no in (result.numeric_answers or {}) else int(grid.item(r, 1).text() if grid.item(r, 1) else q_no)
            result.numeric_answers[target_key] = student_txt
        elif section == "TF":
            tf_flags: dict[str, bool] = {}
            compact = "".join(ch for ch in student_txt.upper() if ch in {"Đ", "D", "S", "T", "F", "1", "0"})
            expected_len = max(1, len(_tf_compact((key.true_false_answers or {}).get(q_no, {}) or {})))
            compact = compact[:expected_len]
            for i, ch in enumerate(compact):
                key_flag = ["a", "b", "c", "d"][i] if i < 4 else f"k{i+1}"
                tf_flags[key_flag] = ch in {"Đ", "D", "T", "1"}
            if tf_flags:
                target_key = q_no if q_no in (result.true_false_answers or {}) else int(grid.item(r, 1).text() if grid.item(r, 1) else q_no)
                result.true_false_answers[target_key] = tf_flags

    sid_selected = str(inp_sid.currentData() or inp_sid.currentText() or "").strip()
    if sid_selected in sid_reverse_map:
        sid_selected = sid_reverse_map[sid_selected]
    if sid_selected.startswith("[") and "]" in sid_selected:
        sid_selected = sid_selected[1:].split("]", 1)[0].strip()
    result.student_id = sid_selected
    result.exam_code = str(inp_code.currentData() or inp_code.currentText() or "").strip()

    self.database.update_scan_result_payload(
        self._batch_result_subject_key(subject_key),
        str(getattr(result, "image_path", "") or ""),
        self._serialize_omr_result(result),
        note="scoring_review_edit",
    )
    self.database.log_change(
        "score_review_edit",
        f"{subject_key}::{str(getattr(result, 'student_id', '') or '')}",
        "manual_edit",
        "",
        "Updated answers from scoring review dialog",
        "scoring_review",
    )
    sid_key = str(getattr(result, "student_id", "") or "").strip()
    subject_scores = self.scoring_results_by_subject.get(subject_key, {}) or {}
    if sid_key and sid_key in subject_scores:
        row_payload = dict(subject_scores.get(sid_key, {}) or {})
        row_payload["status"] = "Đã sửa"
        row_payload["note"] = "Sửa từ màn hình giải trình điểm"
        subject_scores[sid_key] = row_payload
        self.scoring_results_by_subject[subject_key] = subject_scores
    self.calculate_scores(subject_key=subject_key, mode="Tính lại toàn bộ", note="scoring_review_edit")


__all__ = ["open_scoring_review_editor_dialog"]
