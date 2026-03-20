from pathlib import Path
import unittest


class GuiRegressionTests(unittest.TestCase):
    def test_subject_config_dialog_no_left_left_lay_reference(self) -> None:
        source = Path('gui/main_window.py').read_text(encoding='utf-8')
        self.assertNotIn('left_left_lay.addLayout(form)', source)
        self.assertNotIn('left_left_lay.addWidget(buttons)', source)
        self.assertIn('lay.addLayout(form)', source)


if __name__ == '__main__':
    unittest.main()
