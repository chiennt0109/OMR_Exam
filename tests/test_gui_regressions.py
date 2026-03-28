from pathlib import Path
import unittest


class GuiRegressionTests(unittest.TestCase):
    def test_subject_config_dialog_no_left_left_lay_reference(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertNotIn('left_left_lay.addLayout(form)', source)
        self.assertNotIn('left_left_lay.addWidget(buttons)', source)
        self.assertIn('lay.addLayout(form)', source)
        self.assertIn('lay.addWidget(buttons)', source)


    def test_batch_scan_grid_uses_exam_code_name_birth_content_columns_consistently(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('self.scan_list.setHorizontalHeaderLabels(["STUDENT ID", "Mã đề", "Họ tên", "Ngày sinh", "Nội dung", "Status", "Chức năng"])', source)
        self.assertIn('self.scan_list.setItem(idx, 1, QTableWidgetItem((result.exam_code or "").strip() or "-"))', source)
        self.assertIn('setattr(result, "full_name", str(self.scan_list.item(idx, 2).text() if self.scan_list.item(idx, 2) else ""))', source)
        self.assertIn('setattr(result, "birth_date", str(self.scan_list.item(idx, 3).text() if self.scan_list.item(idx, 3) else ""))', source)

    def test_batch_scan_sorting_prioritizes_error_blank_then_student_id(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _row_sort_bucket(self, status: str, blank_map: dict[str, list[int]]) -> int:', source)
        self.assertIn('''if has_error:
            return 0
        if has_blank:
            return 1
        return 2''', source)
        self.assertIn('self._student_sort_token(str(item["sid"]))', source)

    def test_student_sort_token_uses_consistent_tuple_shape_to_avoid_mixed_type_sort_errors(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _student_sort_token(student_id: str) -> tuple[int, int, object]:', source)
        self.assertIn('return (0, 0, int(sid))', source)
        self.assertIn('return (0, 1, sid.casefold())', source)

    def test_student_id_error_status_is_not_treated_as_duplicate(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _student_id_has_recognition_error(student_id: str) -> bool:', source)
        self.assertIn('parts.append("Lỗi SBD")', source)
        self.assertIn('elif sid and duplicate_count > 1:', source)
        self.assertIn('if not self._student_id_has_recognition_error(sid):', source)
        self.assertIn('parts.append("Sai SBD phòng thi")', source)
        self.assertIn('def _subject_student_room_scope(self) -> tuple[set[str], set[str]]:', source)
        self.assertIn('exam_room_sbd_mapping', source)
        self.assertIn('exam_room_name', source)

    def test_batch_scan_supports_new_only_and_all_modes(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('self.batch_file_scope_combo.addItem("Nhận dạng file mới", "new_only")', source)
        self.assertIn('self.batch_file_scope_combo.addItem("Nhận dạng toàn bộ", "all")', source)
        self.assertIn('self.batch_scan_state_value = QLineEdit("-"); self.batch_scan_state_value.setReadOnly(True)', source)
        self.assertIn('if file_scope_mode == "new_only":', source)
        self.assertIn('def _recommended_batch_timeout_sec(template: Template | None) -> float:', source)
        self.assertIn('self.template.metadata["recognition_timeout_sec"] = self._recommended_batch_timeout_sec(self.template)', source)

    def test_batch_scan_replaces_recognition_mechanism_with_api_exam_source(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('batch_form.addRow("API bài thi", api_row)', source)
        self.assertIn('self.batch_recognition_mode_combo.setVisible(False)', source)
        self.assertIn('self.btn_pick_batch_api_file = QPushButton("Chọn file API")', source)
        self.assertIn('def _run_batch_scan_from_api_file(self, subject_cfg: dict, file_scope_mode: str, api_file: str) -> None:', source)
        self.assertIn('if api_file and api_file != "-":', source)
        self.assertIn('self._run_batch_scan_from_api_file(subject_cfg or {}, file_scope_mode, api_file)', source)

    def test_scoring_uses_lightweight_scan_copies_and_batch_results_drop_large_arrays(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _strip_transient_scan_artifacts(result: OMRResult) -> OMRResult:', source)
        self.assertIn('def _lightweight_result_copy(self, result: OMRResult) -> OMRResult:', source)
        self.assertIn('self.scan_results.append(self._strip_transient_scan_artifacts(result))', source)
        self.assertIn('result = self._lightweight_result_copy(base[idx])', source)


    def test_batch_scan_action_column_uses_delete_button(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('SP_TrashIcon', source)
        self.assertIn('def _delete_scan_row_by_index(self, row: int) -> None:', source)
        self.assertIn('delete_scan_result(subject_key, image_path)', source)

    def test_edit_dialog_preview_and_combo_validation_regressions(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('QFrame,', source)
        self.assertIn('preview_result = self._scoped_result_copy(result)', source)
        self.assertIn('preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)', source)
        self.assertIn('preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)', source)
        self.assertIn('def _valid_student_ids() -> list[str]:', source)
        self.assertIn('return None', source)
        self.assertIn('sorted(set(int(q) for q in (key.answers or {}).keys())) or list(default_by_config["MCQ"])', source)
        self.assertIn('expected_by_section[sec] = []', source)
        self.assertIn('self._build_recognition_content_text(scoped, blank_map)', source)
        self.assertIn('self._short_recognition_text_for_result(scoped)', source)
        self.assertIn("Student ID '{student_id_text}' không có trong danh sách học sinh hợp lệ của ca thi.", source)
        self.assertIn("Exam code '{exam_code_text}' không có đáp án hợp lệ cho môn hiện tại.", source)

    def test_exam_code_combo_only_uses_subject_answer_keys(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        start = source.rfind('        def _valid_exam_codes(subject_key: str, current_code: str = "") -> list[str]:')
        end = source.find('        def _populate_exam_code_combo(combo: QComboBox, subject_key: str, current_code: str) -> None:', start)
        block = source[start:end]
        self.assertIn("if not subject:\n                return []", block)
        self.assertNotIn('codes.update(str(x).strip() for x in (self.imported_exam_codes or []) if str(x).strip())', block)
        self.assertNotIn("if current_code and current_code != \"-\":\n                codes.add(str(current_code).strip())", block)
    def test_scoring_syncs_current_batch_snapshot_before_scoring(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _sync_current_batch_subject_snapshot(self, persist_to_db: bool = True) -> tuple[str, list[OMRResult]]:', source)
        self.assertIn('self.database.upsert_scan_result(subject_key, self._serialize_omr_result(result))', source)
        self.assertIn('self._sync_current_batch_subject_snapshot(persist_to_db=True)', source)
        self.assertIn('current_subject, current_results = self._sync_current_batch_subject_snapshot(persist_to_db=True)', source)

    def test_batch_scan_db_refresh_migrates_config_cache_once_and_reloads_from_db(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('cached_results = self._cached_subject_scans_from_config(subject)', source)
        self.assertIn('self.database.replace_scan_results_for_subject(subject, [self._serialize_omr_result(x) for x in cached_results])', source)
        self.assertIn('rows = self.database.fetch_scan_results_for_subject(subject)', source)

    def test_batch_scan_grid_uses_db_refresh_as_single_shared_source(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('existing_results = list(self._refresh_scan_results_from_db(subject_key_for_results) or [])', source)
        self.assertIn('self.scan_results = self._refresh_scan_results_from_db(subject_key)', source)
        self.assertIn('Đã nạp kết quả Batch Scan từ nguồn dữ liệu chuẩn trong cơ sở dữ liệu cho môn này', source)

    def test_batch_scan_initial_load_finalizes_display_and_selects_first_row(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _finalize_batch_scan_display(self, refresh_statuses: bool = True) -> None:', source)
        self.assertIn('self.scan_list.setCurrentCell(target_row, 0)', source)
        self.assertIn('self.scan_list.selectRow(target_row)', source)
        self.assertIn('self._on_scan_selected()', source)

    def test_batch_subject_change_preloads_template_answer_keys_and_finalizes_grid(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn("if tpl_for_view:\n            self.template = tpl_for_view", source)
        self.assertIn('self._ensure_answer_keys_for_subject(subject_key)', source)
        self.assertIn('self._finalize_batch_scan_display(refresh_statuses=False)', source)

    def test_batch_subject_change_uses_cached_db_display_without_status_recompute(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _finalize_batch_scan_display(self, refresh_statuses: bool = True) -> None:', source)
        self.assertIn('if refresh_statuses:\n            self._refresh_all_statuses()', source)
        self.assertIn('cached_blank_map = getattr(result, "cached_blank_summary", None)', source)
        self.assertIn('status = str(getattr(result, "cached_status", "") or "OK")', source)
        self.assertIn('"cached_status": str(getattr(result, "cached_status", "") or "")', source)
        self.assertIn('setattr(result, "cached_status", str(payload.get("cached_status", "") or ""))', source)

    def test_exam_code_change_keeps_existing_tf_and_numeric_rows(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('tf_data = data_snapshot.get("true_false_answers", {}) or {}', source)
        self.assertIn('set(int(q) for q in tf_data.keys())', source)
        self.assertIn('table_tf = _build_tf_table(tf_questions, tf_data)', source)
        self.assertIn('numeric_data = data_snapshot.get("numeric_answers", {}) or {}', source)
        self.assertIn('set(int(q) for q in numeric_data.keys())', source)
        self.assertIn('table_num = _build_pair_table(numeric_questions, numeric_data, "Ví dụ: -12.5")', source)

    def test_edit_dialog_question_numbers_follow_configured_answer_counts(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('default_by_config = {', source)
        self.assertIn('list(range(1, max(0, int(configured_counts.get(sec, 0) or 0)) + 1))', source)
        self.assertIn('"MCQ": list(default_by_config["MCQ"]),', source)


    def test_invalidate_scoring_no_undefined_rows_loop_and_lightweight_path(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        start = source.find('    def _invalidate_scoring_for_student_ids(self, student_ids: list[str], subject_key: str = "", reason: str = "") -> int:')
        end = source.find('    def _record_adjustment(self, idx: int, details: list[str], source: str) -> None:', start)
        block = source[start:end]
        self.assertNotIn('for row in rows:', block)
        self.assertIn('self.database.log_change(', block)
        self.assertIn('return changed', block)

    def test_scoring_view_restores_cached_subject_results_without_recalculate(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _load_cached_scoring_results_for_subject(self, subject_key: str) -> None:', source)
        self.assertIn('self._load_cached_scoring_results_for_subject(subject_key)', source)
        self.assertIn('self._load_cached_scoring_results_for_subject(selected_subject)', source)
        self.assertIn('self.score_rows = list(loaded_rows)', source)
        self.assertIn('loaded_rows.sort(key=lambda row: str(row.student_id or ""))', source)

    def test_scoring_supports_smart_fallback_when_exam_code_is_invalid(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('smart_scored_scans: list[dict[str, str]] = []', source)
        self.assertIn('setattr(best_row, "scoring_note", "Chấm thông minh")', source)
        self.assertIn('"smart_scoring_count": len(smart_scored_scans),', source)
        self.assertIn('"note": str(getattr(r, "scoring_note", "") or ""),', source)
        self.assertIn('Danh sách Chấm thông minh:', source)

    def test_batch_scan_uses_shared_process_batch_flow_instead_of_per_file_deepcopy(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('new_results = self.omr_processor.process_batch(file_paths, self.template, on_progress)', source)
        self.assertNotIn('self.omr_processor.run_recognition_test(image_path, copy.deepcopy(self.template), None)', source)
        self.assertIn('self.database.replace_scan_results_for_subject(', source)
        self.assertNotIn('self.database.upsert_scan_result(subject_key_for_results, self._serialize_omr_result(result))', source)
        self.assertIn('timing_text = f"{elapsed_sec:.1f}s/{total_items} bài"', source)
        self.assertIn('self.progress.setFormat(f"%p% ({timing_text})")', source)
        self.assertIn('omr_batch_timing_manual.log', source)


if __name__ == '__main__':
    unittest.main()
