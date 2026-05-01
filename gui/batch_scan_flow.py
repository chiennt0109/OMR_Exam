from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QComboBox, QDialogButtonBox, QMessageBox, QLineEdit

import csv
import re


def run_batch_scan_from_api_file(self, subject_cfg: dict, file_scope_mode: str, api_file: str) -> None:
    subject_key_for_results = self._subject_key_from_cfg(subject_cfg) if subject_cfg else self._resolve_preferred_scoring_subject()
    scan_folder = str((subject_cfg or {}).get("scan_folder", "") or "").strip()
    if not scan_folder:
        scan_folder = str(getattr(self, "batch_scan_folder_value", QLineEdit("-")).text() if hasattr(self, "batch_scan_folder_value") else "").strip()
    if not scan_folder or scan_folder == "-":
        QMessageBox.warning(self, "API bài thi", "Chưa cấu hình Thư mục bài thi môn.")
        return
    try:
        headers, mapping_rows = self._load_api_mapping_rows(Path(api_file))
    except Exception as exc:
        QMessageBox.warning(self, "API bài thi", f"Không đọc được file mapping:\n{exc}")
        return
    if not mapping_rows:
        QMessageBox.warning(self, "API bài thi", "File mapping không có dữ liệu.")
        return
    if not headers:
        QMessageBox.warning(self, "API bài thi", "Không tìm thấy tiêu đề cột trong file mapping.")
        return
    
    expected_len = self._expected_answer_string_length_for_subject(subject_key_for_results)
    pick = QDialog(self)
    pick.setWindowTitle("Chọn cột API bài thi")
    pick_lay = QVBoxLayout(pick)
    pick_form = QFormLayout()
    file_col = QComboBox(); file_col.addItems(headers)
    sid_col = QComboBox(); sid_col.addItems(["[Không dùng]"] + headers)
    exam_col = QComboBox(); exam_col.addItems(["[Không dùng]"] + headers)
    answer_col = QComboBox(); answer_col.addItems(["[Không dùng]"] + headers)
    
    def _find_idx(alias: set[str], combo: QComboBox) -> int:
        for i in range(combo.count()):
            if self._normalize_mapping_key(combo.itemText(i)) in alias:
                return i
        return 0
    
    file_col.setCurrentIndex(_find_idx({"filename", "file", "image", "tenfile"}, file_col))
    sid_col.setCurrentIndex(_find_idx({"sdb", "studentid", "student_id", "sobaodanh"}, sid_col))
    exam_col.setCurrentIndex(_find_idx({"made", "examcode", "exam_code", "ma_de"}, exam_col))
    answer_col.setCurrentIndex(_find_idx({"bailam", "answer", "answers", "answerstring"}, answer_col))
    
    pick_form.addRow("Cột FileName (bắt buộc)", file_col)
    pick_form.addRow("Cột SBD", sid_col)
    pick_form.addRow("Cột mã đề", exam_col)
    pick_form.addRow("Cột bài làm", answer_col)
    pick_lay.addLayout(pick_form)
    pick_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    pick_buttons.accepted.connect(pick.accept)
    pick_buttons.rejected.connect(pick.reject)
    pick_lay.addWidget(pick_buttons)
    if pick.exec() != QDialog.Accepted:
        return
    
    selected_file_col = file_col.currentText().strip()
    selected_sid_col = sid_col.currentText().strip()
    selected_exam_col = exam_col.currentText().strip()
    selected_answer_col = answer_col.currentText().strip()
    if not selected_file_col or selected_file_col == "[Không dùng]":
        QMessageBox.warning(self, "API bài thi", "Bắt buộc chọn cột FileName.")
        return
    
    mcq_questions, tf_questions, numeric_layout = self._answer_layout_for_subject(subject_key_for_results)
    q_counts = (subject_cfg or {}).get("question_counts", {}) if isinstance(subject_cfg or {}, dict) else {}
    if not any(int((q_counts or {}).get(k, 0) or 0) > 0 for k in ["MCQ", "TF", "NUMERIC"]):
        tpl_path = str((subject_cfg or {}).get("template", "") or "").strip()
        if tpl_path:
            q_counts = self._template_question_counts(tpl_path)
    mcq_count = max(0, int((q_counts or {}).get("MCQ", 0) or 0))
    tf_count = max(0, int((q_counts or {}).get("TF", 0) or 0))
    numeric_count = max(0, int((q_counts or {}).get("NUMERIC", 0) or 0))
    if not mcq_questions and mcq_count > 0:
        mcq_questions = [x for x in range(1, mcq_count + 1)]
    next_q = max(mcq_questions) if mcq_questions else 0
    if not tf_questions and tf_count > 0:
        tf_questions = [x for x in range(next_q + 1, next_q + tf_count + 1)]
    next_q = max(tf_questions) if tf_questions else next_q
    if not numeric_layout and numeric_count > 0:
        numeric_layout = [(x, 0) for x in range(next_q + 1, next_q + numeric_count + 1)]
    tf_true_chars = {"Đ", "đ", "D", "d", "T", "t", "1", "Y", "y"}
    
    configured_answer_keys = self._fetch_answer_keys_for_subject_scoped(subject_key_for_results, subject_cfg) or {}
    imported_answer_keys = self._subject_imported_answer_keys_for_main(subject_cfg or {})
    section_layout_cache: dict[str, tuple[list[int], list[int], list[tuple[int, int]], bool]] = {}
    
    def _section_layout_from_subject_cfg(exam_code_text: str) -> tuple[list[int], list[int], list[tuple[int, int]], bool]:
        exam_text = str(exam_code_text or "").strip()
        exam_norm = self._normalize_exam_code_text(exam_text)
        cache_key = f"{exam_text}::{exam_norm}"
        cached = section_layout_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = None
    
        if configured_answer_keys:
            payload = configured_answer_keys.get(exam_text)
            if payload is None:
                for k, v in configured_answer_keys.items():
                    if self._normalize_exam_code_text(str(k or "")) == exam_norm:
                        payload = v
                        break
    
        if payload is None:
            payload = imported_answer_keys.get(exam_text) if isinstance(imported_answer_keys, dict) else None
            if payload is None and isinstance(imported_answer_keys, dict):
                for k, v in imported_answer_keys.items():
                    if self._normalize_exam_code_text(str(k or "")) == exam_norm:
                        payload = v
                        break
    
        if isinstance(payload, SubjectKey):
            mcq_payload = payload.answers if isinstance(payload.answers, dict) else {}
            tf_payload = payload.true_false_answers if isinstance(payload.true_false_answers, dict) else {}
            numeric_payload = payload.numeric_answers if isinstance(payload.numeric_answers, dict) else {}
        elif isinstance(payload, dict):
            mcq_payload = payload.get("mcq_answers", {}) if isinstance(payload.get("mcq_answers", {}), dict) else {}
            tf_payload = payload.get("true_false_answers", {}) if isinstance(payload.get("true_false_answers", {}), dict) else {}
            numeric_payload = payload.get("numeric_answers", {}) if isinstance(payload.get("numeric_answers", {}), dict) else {}
        else:
            return [], [], [], False
    
        mcq_q: list[int] = []
        tf_q: list[int] = []
        numeric_layout: list[tuple[int, int]] = []
        for q_raw in mcq_payload.keys():
            try:
                mcq_q.append(int(q_raw))
            except Exception:
                pass
        for q_raw in tf_payload.keys():
            try:
                tf_q.append(int(q_raw))
            except Exception:
                pass
        for q_raw, ans in numeric_payload.items():
            try:
                numeric_layout.append((int(q_raw), len(str(ans or ""))))
            except Exception:
                pass
        parsed = (sorted(set(mcq_q)), sorted(set(tf_q)), sorted(numeric_layout, key=lambda x: int(x[0])), True)
        section_layout_cache[cache_key] = parsed
        return parsed
    
    def _parse_answer_string(
        raw_answer: str,
        row_mcq_questions: list[int],
        row_tf_questions: list[int],
        row_numeric_layout: list[tuple[int, int]],
    ) -> tuple[dict[int, str], dict[int, dict[str, bool]], dict[int, str], str]:
        raw_text = str(raw_answer or "").strip()
        compact = re.sub(r"\s+", "", raw_text)
        mcq_span = max(0, len(row_mcq_questions))
        tf_span = max(0, len(row_tf_questions) * 4)
        mcq_source = compact[:mcq_span]
        tf_source = compact[mcq_span:mcq_span + tf_span]
        numeric_tail = compact[mcq_span + tf_span:]
    
        numeric_map: dict[int, str] = {}
        has_fixed_numeric_width = bool(row_numeric_layout) and all(int(expected_len) > 0 for _, expected_len in row_numeric_layout)
        if has_fixed_numeric_width:
            compact_numeric = str(numeric_tail)
            pos = 0
            for q_no, expected_len in row_numeric_layout:
                token = compact_numeric[pos:pos + int(expected_len)] if pos < len(compact_numeric) else ""
                numeric_map[int(q_no)] = token.replace("_", "")
                pos += int(expected_len)
        else:
            # Do not split by comma because decimal answers commonly use comma
            # (e.g. 0,61 / 58,3). Use explicit field separators only.
            numeric_tokens = [tok.strip() for tok in re.split(r"[;|]+", numeric_tail) if tok and tok.strip()]
            if len(numeric_tokens) >= len(row_numeric_layout) and row_numeric_layout:
                for idx_layout, (q_no, expected_len) in enumerate(row_numeric_layout):
                    token = str(numeric_tokens[idx_layout]) if idx_layout < len(numeric_tokens) else ""
                    if int(expected_len) > 0:
                        numeric_map[int(q_no)] = token[: max(0, int(expected_len))].replace("_", "")
                    else:
                        numeric_map[int(q_no)] = token.replace("_", "")
    
        mcq_map: dict[int, str] = {}
        for idx_q, q_no in enumerate(row_mcq_questions):
            raw_ch = str(mcq_source[idx_q]).upper() if idx_q < len(mcq_source) else ""
            mcq_map[int(q_no)] = raw_ch if raw_ch and raw_ch != "_" else ""
    
        tf_map: dict[int, dict[str, bool]] = {}
        for idx_q, q_no in enumerate(row_tf_questions):
            base = idx_q * 4
            tf_row: dict[str, bool] = {}
            for idx_stmt, key in enumerate(["a", "b", "c", "d"]):
                ch = tf_source[base + idx_stmt] if (base + idx_stmt) < len(tf_source) else ""
                if ch in tf_true_chars:
                    tf_row[key] = True
                elif ch in {"S", "s", "0", "N", "n"}:
                    tf_row[key] = False
            tf_map[int(q_no)] = tf_row
    
        rebuilt_parts: list[str] = []
        for q_no in row_mcq_questions:
            rebuilt_parts.append(str((mcq_map or {}).get(int(q_no), "") or "_")[:1])
        for q_no in row_tf_questions:
            flags = (tf_map or {}).get(int(q_no), {}) or {}
            for key in ["a", "b", "c", "d"]:
                if key in flags:
                    rebuilt_parts.append("Đ" if bool(flags.get(key)) else "S")
                else:
                    rebuilt_parts.append("_")
        for q_no, expected_len in row_numeric_layout:
            raw_val = str((numeric_map or {}).get(int(q_no), "") or "")
            if int(expected_len) > 0:
                token = raw_val[: int(expected_len)]
                if len(token) < int(expected_len):
                    token = token + ("_" * (int(expected_len) - len(token)))
                rebuilt_parts.append(token)
            else:
                rebuilt_parts.append(raw_val)
        rebuilt = "".join(rebuilt_parts)
        return mcq_map, tf_map, numeric_map, rebuilt
    
    out: list[OMRResult] = []
    for row in mapping_rows:
        def _pick(col_name: str) -> str:
            if not col_name or col_name == "[Không dùng]":
                return ""
            return str(row.get(col_name, "") or "").strip()
    
        fname = _pick(selected_file_col)
        if not fname:
            continue
        result = OMRResult(image_path=str(Path(scan_folder) / fname))
        result.student_id = _pick(selected_sid_col)
        result.exam_code = _pick(selected_exam_col)
        raw_answer = _pick(selected_answer_col)
        row_mcq_questions = list(mcq_questions)
        row_tf_questions = list(tf_questions)
        row_numeric_layout = list(numeric_layout)
        cfg_mcq_q, cfg_tf_q, cfg_num_layout, exam_code_valid = _section_layout_from_subject_cfg(result.exam_code)
        if cfg_mcq_q:
            row_mcq_questions = list(cfg_mcq_q)
        if cfg_tf_q:
            row_tf_questions = list(cfg_tf_q)
        if exam_code_valid and cfg_num_layout:
            row_numeric_layout = list(cfg_num_layout)
        # Keep default numeric layout from current subject configuration when exam code
        # is missing/invalid, so Numeric answer string is still cut by configured lengths.
        mcq_map, tf_map, numeric_map, rebuilt_answer = _parse_answer_string(raw_answer, row_mcq_questions, row_tf_questions, row_numeric_layout)
        result.mcq_answers = mcq_map
        result.true_false_answers = tf_map
        result.numeric_answers = numeric_map
        setattr(result, "answer_string_api_mode", True)
        if expected_len > 0:
            result.answer_string = rebuilt_answer[:expected_len]
        else:
            result.answer_string = rebuilt_answer
        out.append(self._strip_transient_scan_artifacts(result))
    
    if not out:
        QMessageBox.warning(self, "API bài thi", "Không có dòng hợp lệ (cần cột FileName).")
        return
    
    if file_scope_mode == "all":
        self.database.delete_scan_results_for_subject(self._batch_result_subject_key(subject_key_for_results))
        self.scan_results = []
    else:
        self.scan_results = list(self._refresh_scan_results_from_db(subject_key_for_results) or [])
    by_path = {str(getattr(r, "image_path", "") or ""): r for r in self.scan_results}
    for r in out:
        by_path[str(r.image_path)] = r
    self.scan_results = list(by_path.values())
    self.scan_results_by_subject[self._batch_result_subject_key(subject_key_for_results)] = list(self.scan_results)
    self.database.replace_scan_results_for_subject(self._batch_result_subject_key(subject_key_for_results), [self._serialize_omr_result(x) for x in self.scan_results])
    self._populate_scan_grid_from_results(self.scan_results)
    self._finalize_batch_scan_display()
    self._update_batch_scan_scope_summary()
    QMessageBox.information(self, "API bài thi", f"Đã nạp {len(out)} dòng từ API bài thi.")
    
