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

    def test_scoring_uses_lightweight_scan_copies_and_batch_results_drop_large_arrays(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def _strip_transient_scan_artifacts(result: OMRResult) -> OMRResult:', source)
        self.assertIn('def _lightweight_result_copy(self, result: OMRResult) -> OMRResult:', source)
        self.assertIn('self.scan_results.append(self._strip_transient_scan_artifacts(result))', source)
        self.assertIn('result = self._lightweight_result_copy(base[idx])', source)


if __name__ == '__main__':
    unittest.main()
