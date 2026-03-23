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

    def test_student_id_error_status_is_not_treated_as_duplicate(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _student_id_has_recognition_error(student_id: str) -> bool:', source)
        self.assertIn('parts.append("Lỗi SBD")', source)
        self.assertIn('elif sid and duplicate_count > 1:', source)
        self.assertIn('if not self._student_id_has_recognition_error(sid):', source)

    def test_batch_scan_supports_new_only_and_all_modes(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('self.batch_file_scope_combo.addItem("Nhận dạng file mới", "new_only")', source)
        self.assertIn('self.batch_file_scope_combo.addItem("Nhận dạng toàn bộ", "all")', source)
        self.assertIn('self.batch_scan_state_value = QLineEdit("-"); self.batch_scan_state_value.setReadOnly(True)', source)
        self.assertIn('if file_scope_mode == "new_only":', source)
        self.assertIn('def _recommended_batch_timeout_sec(template: Template | None) -> float:', source)
        self.assertIn('self.template.metadata["recognition_timeout_sec"] = self._recommended_batch_timeout_sec(self.template)', source)

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
        self.assertIn('sorted(set(int(q) for q in (key.answers or {}).keys())) or fallback_snapshot["MCQ"]', source)
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


if __name__ == '__main__':
    unittest.main()
