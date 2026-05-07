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
        self.assertIn('self.exam_room_name = QComboBox(); self.exam_room_name.setEditable(True)', source)
        self.assertIn('self.btn_import_exam_room_mapping = QPushButton("Import mapping SBD/phòng từ Excel")', source)
        self.assertIn('def _import_exam_room_mapping_from_file(self) -> None:', source)
        self.assertIn('def _load_exam_room_mapping_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:', source)
        self.assertIn('pick.setWindowTitle("Chọn cột mapping SBD/phòng")', source)
        self.assertIn('pick_l.addRow("Cột SBD", sid_col)', source)
        self.assertIn('pick_l.addRow("Cột phòng thi", room_col)', source)
        self.assertIn('self.exam_room_mapping_selector = QComboBox()', source)
        self.assertNotIn('self.exam_room_sbd_mapping = QTextEdit', source)

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
        self.assertIn('self.ribbon_recheck_action = toolbar.addAction(style.standardIcon(QStyle.SP_BrowserReload), "Phúc tra", self.action_open_recheck)', source)
        self.assertIn('def action_open_recheck(self) -> None:', source)
        self.assertIn('dlg.setWindowTitle(f"Phúc tra - {subject_key}")', source)
        self.assertIn('batch_form.addRow("API bài thi", api_row)', source)
        self.assertIn('self.batch_recognition_mode_combo.setVisible(False)', source)
        self.assertIn('self.btn_pick_batch_api_file = QPushButton("Chọn file API")', source)
        self.assertIn('def _load_api_mapping_rows(self, path: Path) -> tuple[list[str], list[dict[str, str]]]:', source)
        self.assertIn('def _run_batch_scan_from_api_file(self, subject_cfg: dict, file_scope_mode: str, api_file: str) -> None:', source)
        self.assertIn('pick.setWindowTitle("Chọn cột API bài thi")', source)
        self.assertIn('pick_form.addRow("Cột FileName (bắt buộc)", file_col)', source)
        self.assertIn('pick_form.addRow("Cột SBD", sid_col)', source)
        self.assertIn('pick_form.addRow("Cột mã đề", exam_col)', source)
        self.assertIn('pick_form.addRow("Cột bài làm", answer_col)', source)
        self.assertIn('if not selected_file_col or selected_file_col == "[Không dùng]":', source)
        self.assertIn('def _answer_layout_for_subject(self, subject_key: str) -> tuple[list[int], list[int], list[tuple[int, int]]]:', source)
        self.assertIn('if not any(int((q_counts or {}).get(k, 0) or 0) > 0 for k in ["MCQ", "TF", "NUMERIC"]):', source)
        self.assertIn('q_counts = (subject_cfg or {}).get("question_counts", {}) if isinstance(subject_cfg or {}, dict) else {}', source)
        self.assertIn('q_counts = self._template_question_counts(tpl_path)', source)
        self.assertIn('def _section_layout_from_subject_cfg(exam_code_text: str) -> tuple[list[int], list[int], list[tuple[int, int]]]:', source)
        self.assertIn('def _parse_answer_string(', source)
        self.assertIn('row_numeric_layout: list[tuple[int, int]],', source)
        self.assertIn('compact = re.sub(r"[\\s_]+", "", raw_text)', source)
        self.assertIn('mcq_source = compact[:mcq_span]', source)
        self.assertIn('tf_source = compact[mcq_span:mcq_span + tf_span]', source)
        self.assertIn('numeric_tail = compact[mcq_span + tf_span:]', source)
        self.assertIn('has_fixed_numeric_width = bool(row_numeric_layout) and all(int(expected_len) > 0 for _, expected_len in row_numeric_layout)', source)
        self.assertIn('token = compact_numeric[pos:pos + int(expected_len)] if pos < len(compact_numeric) else ""', source)
        self.assertIn('numeric_tokens = [tok.strip() for tok in re.split(r"[,;|]+", numeric_tail) if tok and tok.strip()]', source)
        self.assertIn('tf_row: dict[str, bool] = {}', source)
        self.assertIn('rebuilt_parts: list[str] = []', source)
        self.assertIn('rebuilt = "".join(rebuilt_parts)', source)
        self.assertNotIn('OMRProcessor.build_answer_string(mcq_map, tf_map, numeric_map)', source)
        self.assertIn('mcq_map, tf_map, numeric_map, rebuilt_answer = _parse_answer_string(raw_answer, row_mcq_questions, row_tf_questions, row_numeric_layout)', source)
        self.assertIn('cfg_mcq_q, cfg_tf_q, cfg_num_layout = _section_layout_from_subject_cfg(result.exam_code)', source)
        self.assertIn('if cfg_tf_q:', source)
        self.assertIn('if cfg_num_layout:', source)
        self.assertIn('result.mcq_answers = mcq_map', source)
        self.assertIn('result.true_false_answers = tf_map', source)
        self.assertIn('result.numeric_answers = numeric_map', source)
        self.assertIn('if api_file and api_file != "-":', source)
        self.assertIn('self._run_batch_scan_from_api_file(subject_cfg or {}, file_scope_mode, api_file)', source)

    def test_realtime_status_refreshes_immediately_after_manual_corrections(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _schedule_correction_update(self, field_name: str, old_value: object, new_value: object, apply_fn) -> None:', source)
        self.assertIn('self._refresh_all_statuses()', source)
        self.assertIn('self.correction_save_timer.start(150)', source)

    def test_edited_status_is_excluded_from_error_duplicate_wrong_code_groups(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('is_edited = "đã sửa" in low', source)
        self.assertIn('if is_edited:\n                    edited_count += 1\n                    continue', source)
        self.assertIn('status_ok = bool(status_text and status_text != "ok" and not is_edited)', source)
        self.assertIn('status_ok = ("trùng sbd" in status_text or "duplicate" in status_text) and not is_edited', source)
        self.assertIn('status_ok = (("mã đề" in status_text) and ("sai" in status_text or "không" in status_text or "?" in status_text)) and not is_edited', source)

    def test_room_scope_status_uses_normalized_student_id_matching(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('sid_norm = self._normalized_student_id_for_match(sid)', source)
        self.assertIn('sid_norm in all_sids and room_sids and sid_norm not in room_sids', source)
        self.assertIn('all_sids.add(self._normalized_student_id_for_match(sid))', source)
        self.assertIn('room_sids = {self._normalized_student_id_for_match(x) for x in chunks if x}', source)
        self.assertIn('def _normalized_room_for_match(room_text: str) -> str:', source)
        self.assertIn('room_name_norm = self._normalized_room_for_match(room_name)', source)
        self.assertIn('self._normalized_room_for_match(exam_room) == room_name_norm', source)
        self.assertIn('"exam_room_sbd_mapping_by_room": {str(k): list(v) for k, v in (self._exam_room_mapping_cache or {}).items()}', source)
        self.assertIn('mapping_by_room = cfg.get("exam_room_sbd_mapping_by_room", {})', source)

    def test_recognition_content_includes_parsed_api_sections_for_debug(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('parsed_parts: list[str] = []', source)
        self.assertIn('parsed_parts.append(f"MCQ: {mcq}")', source)
        self.assertIn('parsed_parts.append(f"TF: {tf}")', source)
        self.assertIn('parsed_parts.append(f"NUM: {num}")', source)
        self.assertIn('merged = parsed_parts + blank_parts', source)
        self.assertIn('return " | ".join(merged) if merged else "-"', source)

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


    def test_batch_scope_key_uses_exam_name_when_session_id_missing(self) -> None:
        source = Path('gui/main_window_batch_subject_mixin.py').read_text(encoding='utf-8')
        self.assertIn('if sid and exam_name:', source)
        self.assertIn('if exam_name:', source)
        self.assertIn('return exam_name', source)

    def test_answer_key_scope_key_uses_exam_name_when_session_id_missing(self) -> None:
        source = Path('gui/main_window_dialogs.py').read_text(encoding='utf-8')
        self.assertIn('if session_id and exam_name:', source)
        self.assertIn('elif session_id:', source)
        self.assertIn('scope_prefix = exam_name', source)

    def test_auto_recognition_hard_session_isolation_guards_cross_exam_jobs(self) -> None:
        auto_source = Path('gui/main_window_auto_recognition_mixin.py').read_text(encoding='utf-8')
        session_source = Path('gui/main_window_session_mixin.py').read_text(encoding='utf-8')
        workspace_source = Path('gui/main_window_workspace_mixin.py').read_text(encoding='utf-8')
        self.assertIn('def _reset_auto_recognition_state(self, *, pause: bool = False) -> None:', auto_source)
        self.assertIn('if scope_prefix and not str(subject_key or "").strip().startswith(f"{scope_prefix}::"):', auto_source)
        self.assertIn('self._reset_auto_recognition_state(pause=True)', session_source)
        self.assertIn('self._reset_auto_recognition_state(pause=False)', session_source)
        self.assertIn('if prev_session_id and next_session_id and prev_session_id != next_session_id:', workspace_source)

    def test_save_as_clone_resets_answer_key_binding_and_copies_scoped_answer_keys(self) -> None:
        source = Path('gui/main_window_session_mixin.py').read_text(encoding='utf-8')
        self.assertIn('subject_cfg.pop("answer_key_key", None)', source)
        self.assertIn('source_answer_key_candidates = [old_subject_key]', source)
        self.assertIn('source_keys = self.database.fetch_answer_keys_for_subject(source_answer_key)', source)
        self.assertIn('self.database.replace_answer_keys_for_subject(new_subject_key, source_keys)', source)
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

    def test_batch_subject_switch_only_caches_previous_working_state_when_needed(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('has_rows = bool(hasattr(self, "scan_list") and self.scan_list.rowCount() > 0)', source)
        self.assertIn('is_dirty = bool(hasattr(self, "btn_save_batch_subject") and self.btn_save_batch_subject.isEnabled())', source)
        self.assertIn('has_cached_state = isinstance(self.batch_working_state_by_subject.get(previous_runtime_key), dict)', source)
        self.assertIn('if has_rows and (is_dirty or not has_cached_state):', source)

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

    def test_blank_question_fallback_uses_answered_questions_when_expected_numbering_disjoint(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('answered_questions = sorted(set(int(q) for q in section_answers.get(sec, set())))', source)
        self.assertIn('if (not expected_set) or (expected_set.isdisjoint(answered_set)):', source)
        self.assertIn('actual_questions = list(answered_questions)', source)

    def test_tf_blank_detection_counts_by_statements_not_by_question(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('missing_tf_statements: list[str] = []', source)
        self.assertIn('missing_tf_statements.append(f"{int(display_q)}{key}")', source)
        self.assertIn('blanks[sec] = missing_tf_statements', source)

    def test_saved_batch_scan_edit_dialog_does_not_expose_content_answer_editor(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertNotIn('txt_content = QTextEdit(content)', source)
        self.assertNotIn('lay.addWidget(QLabel("Nội dung"))', source)
        self.assertIn('setattr(rebuilt, "manual_content_override", "")', source)

    def test_batch_scan_display_keeps_raw_recognition_scope_like_template_editor(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        start = source.find('    def _scoped_result_copy(self, result):')
        end = source.find('    def _count_mismatch_status_parts(self, result) -> list[str]:', start)
        block = source[start:end]
        self.assertIn('scoped = copy.deepcopy(result)', block)
        self.assertNotIn('self._trim_result_answers_to_expected_scope(scoped)', block)
        self.assertIn('để luồng hiển thị Batch Scan thống nhất với Template Editor.', block)

    def test_batch_scan_mcq_preview_uses_display_reindex_helper_to_avoid_missing_question_1(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _mcq_answers_for_display(self, result) -> dict[int, str]:', source)
        self.assertIn('if expected_mcq and 1 in expected_mcq:', source)
        self.assertIn('return {idx + 1: raw[q] for idx, q in enumerate(keys)}', source)
        self.assertIn('mcq = self._format_mcq_answers(self._mcq_answers_for_display(result))', source)
        self.assertIn('self._format_mcq_answers(self._mcq_answers_for_display(preview_result))', source)

    def test_mcq_blank_detection_handles_shifted_contiguous_indexes_to_keep_content_consistent(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('shifted_contiguous = (', source)
        self.assertIn('and display_sorted == list(range(1, len(display_sorted) + 1))', source)
        self.assertIn('and answered_actual_sorted[0] > 1', source)
        self.assertIn('answered_display = set(display_sorted)', source)


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

    def test_edit_dialog_answer_grid_supports_text_entry_navigation_like_scoring(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _answer_grid_key_release(event) -> None:', source)
        self.assertIn('answer_grid.keyReleaseEvent = _answer_grid_key_release  # type: ignore[method-assign]', source)
        self.assertIn('if answer_grid.currentColumn() != 2:', source)
        self.assertIn('if section_text == "MCQ":', source)
        self.assertIn('elif section_text in {"TF", "NUMERIC"}:', source)
        self.assertIn('answer_grid.itemChanged.connect(_normalize_answer_grid_cell_text)', source)


    def test_batch_scan_uses_progress_screen_with_total_current_and_eta(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _open_batch_progress_screen(self, total_items: int, title: str = "Đang nhận dạng Batch Scan") -> QDialog:', source)
        self.assertIn('lbl_total = QLabel(f"Tổng số bài cần nhận dạng: {max(0, int(total_items or 0))}")', source)
        self.assertIn('lbl_current = QLabel("Đang nhận dạng bài thứ: 0/0")', source)
        self.assertIn('lbl_eta = QLabel("Thời gian còn lại ước tính: -")', source)
        self.assertIn('def _update_batch_progress_screen(self, dlg: QDialog | None, current: int, total: int, image_path: str, started_at: float) -> None:', source)
        self.assertIn("lbl_current.setText(f\"Đang nhận dạng bài thứ: {min(current_safe, total_safe)}/{total_safe} - {Path(str(image_path or '')).name or '-'}\")", source)
        self.assertIn('lbl_eta.setText(f"Thời gian còn lại ước tính: {self._format_eta_text(eta_sec)}")', source)
        self.assertIn('batch_progress_dialog = self._open_batch_progress_screen(len(file_paths), title="Batch Scan - Đang nhận dạng")', source)
        self.assertIn('batch_progress_dialog = self._open_batch_progress_screen(len(file_paths), title="Batch Scan API - Đang nhận dạng")', source)


    def test_patched_payload_builder_always_uses_blank_only_content_for_consistency(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _patched_build_scan_row_payload_from_result', source)
        self.assertIn('content_text = manual_content if manual_content else self._patched_build_blank_only_content_text(result, blank_map)', source)
        self.assertIn('payload["content"] = content_text', source)

    def test_restore_cached_working_state_prefers_saved_status_and_content_when_switching_exam(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn("canonical_payload['status'] = str(payload.get('status', '') or canonical_payload.get('status', 'OK') or 'OK')", source)
        self.assertIn("canonical_payload['content'] = str(payload.get('content', '') or canonical_payload.get('content', '') or '')", source)
        self.assertIn("canonical_payload['recognized_short'] = str(payload.get('recognized_short', '') or canonical_payload.get('recognized_short', '') or '')", source)

    def test_status_ignores_fallback_accepted_recognition_messages(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('blocking_rec_error_codes = [code for code in rec_error_codes if "FALLBACK ACCEPTED" not in code]', source)
        self.assertIn('if blocking_rec_error_codes or issue_codes:', source)

    def test_cached_batch_state_rehydrates_status_from_canonical_payload_instead_of_grid_tooltip(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn("'status': str(payload.get('status', 'OK') or 'OK')", source)
        self.assertIn("'forced_status': str(payload.get('forced_status', '') or '')", source)
        self.assertIn('canonical_payload = self._build_scan_row_payload_from_result(', source)
        self.assertIn('forced_status=forced_status,', source)

if __name__ == '__main__':
    unittest.main()
