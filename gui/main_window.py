from __future__ import annotations

import copy
import csv
import gc
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import deque
from datetime import date, datetime
from pathlib import Path
import time
import uuid
from typing import TYPE_CHECKING

sys.dont_write_bytecode = True

from PySide6.QtCore import Qt, QEvent, QTimer, QSize
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPixmap, QTransform, QPainter, QPen, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QCompleter,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QInputDialog,
    QToolBar,
    QStyle,
    QGroupBox,
    QScrollArea,
)

from core.answer_key_importer import ImportedAnswerKey, ImportedAnswerKeyPackage, import_answer_key
from core.omr_engine import OMRProcessor, OMRResult, RecognitionContext
from core.scoring_engine import ScoringEngine
from models.answer_key import AnswerKeyRepository, SubjectKey
from models.database import OMRDatabase, bootstrap_application_db
from models.exam_session import ExamSession, Student
from models.template import Template, ZoneType
from models.template_repository import TemplateRepository
from gui.ui_branding import app_icon, load_theme, TOOLBAR, STATUS, logo_symbol, logo_main, apply_widget_branding, brand_button
from gui.batch_scan_flow import run_batch_scan_from_api_file

if TYPE_CHECKING:
    from editor.template_editor import TemplateEditorWindow
from gui.main_window_dialogs import PreviewImageWidget, SubjectConfigDialog, NewExamDialog, StudentListPreviewDialog
from gui.main_window_branding_mixin import MainWindowBrandingMixin
from gui.main_window_auto_recognition_mixin import MainWindowAutoRecognitionMixin
from gui.main_window_session_mixin import MainWindowSessionMixin
from gui.main_window_workspace_mixin import MainWindowWorkspaceMixin
from gui.main_window_template_mixin import MainWindowTemplateMixin
from gui.main_window_import_mixin import MainWindowImportMixin
from gui.main_window_batch_subject_mixin import MainWindowBatchSubjectMixin
from gui.main_window_batch_recognition_mixin import MainWindowBatchRecognitionMixin
from gui.main_window_batch_scope_mixin import MainWindowBatchScopeMixin
from gui.main_window_batch_storage_mixin import MainWindowBatchStorageMixin
from gui.main_window_batch_ui_mixin import MainWindowBatchUiMixin
from gui.main_window_batch_edit_mixin import MainWindowBatchEditMixin
from gui.main_window_scoring_mixin import MainWindowScoringMixin
from gui.main_window_export_mixin import MainWindowExportMixin
from gui.main_window_misc_mixin import MainWindowMiscMixin



class _SignalHook:
    """Tiny signal adapter used by _SimplePageHost.

    This keeps old `currentChanged.connect(...)` call sites working without using
    a Qt stacked page widget.
    """
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        if callable(callback) and callback not in self._callbacks:
            self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            try:
                callback(*args)
            except TypeError:
                callback()


class _SimplePageHost(QWidget):
    """Single-visible-page host that replaces the old stacked-page widget.

    Pages are plain widgets in one VBoxLayout. Only the active page is visible.
    The class intentionally implements the small subset of methods used by the
    existing modules (`addWidget`, `setCurrentIndex`, `currentIndex`) so the rest
    of the application remains stable while the heavy stacked widget dependency
    is removed.
    """
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.currentChanged = _SignalHook()
        self._pages: list[QWidget] = []
        self._current_index = -1
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def addWidget(self, widget: QWidget) -> int:
        index = len(self._pages)
        self._pages.append(widget)
        self._layout.addWidget(widget)
        widget.setVisible(False)
        if self._current_index < 0:
            self.setCurrentIndex(index)
        return index

    def currentIndex(self) -> int:
        return int(self._current_index)

    def setCurrentIndex(self, index: int) -> None:
        try:
            index = int(index)
        except Exception:
            index = 0
        if not (0 <= index < len(self._pages)):
            index = 0 if self._pages else -1
        if index == self._current_index:
            return
        if 0 <= self._current_index < len(self._pages):
            self._pages[self._current_index].setVisible(False)
        self._current_index = index
        if 0 <= index < len(self._pages):
            self._pages[index].setVisible(True)
        self.currentChanged.emit(index)
class MainWindow(MainWindowBrandingMixin, MainWindowAutoRecognitionMixin, MainWindowSessionMixin, MainWindowWorkspaceMixin, MainWindowTemplateMixin, MainWindowImportMixin, MainWindowBatchSubjectMixin, MainWindowBatchRecognitionMixin, MainWindowBatchScopeMixin, MainWindowBatchStorageMixin, MainWindowBatchUiMixin, MainWindowBatchEditMixin, MainWindowScoringMixin, MainWindowExportMixin, MainWindowMiscMixin, QMainWindow):
    SCAN_COL_STT = 0
    SCAN_COL_STUDENT_ID = 1
    SCAN_COL_EXAM_ROOM = 2
    SCAN_COL_EXAM_CODE = 3
    SCAN_COL_FULL_NAME = 4
    SCAN_COL_BIRTH_DATE = 5
    SCAN_COL_CONTENT = 6
    SCAN_COL_STATUS = 7
    SCAN_COL_ACTIONS = 8

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OMR Exam Grading System")
        self.resize(1200, 800)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        try:
            _ico = app_icon()
            if _ico is not None and not _ico.isNull():
                self.setWindowIcon(_ico)
        except Exception:
            pass

        self.session: ExamSession | None = None
        self.template: Template | None = None
        self.answer_keys: AnswerKeyRepository | None = None
        self.scan_results = []
        self.scan_results_by_subject: dict[str, list] = {}
        self.batch_working_state_by_subject: dict[str, dict] = {}
        self.scan_files: list[Path] = []
        self.scan_blank_questions: dict[int, list[int]] = {}
        self.scan_blank_summary: dict[int, dict[str, list[int]]] = {}
        self.scan_manual_adjustments: dict[str, list[str]] = {}
        self.scan_edit_history: dict[str, list[str]] = {}
        self.scan_last_adjustment: dict[str, str] = {}
        self.score_rows = []
        self.scoring_results_by_subject: dict[str, dict[str, dict]] = {}
        self.scoring_phases: list[dict] = []
        self._scoring_dirty_subjects: set[str] = set()
        self.imported_exam_codes: list[str] = []
        self.active_batch_subject_key: str | None = None
        self.subject_catalog: list[str] = ["Toán", "Ngữ văn", "Tiếng Anh", "Vật lý", "Hóa học", "Sinh học"]
        self.block_catalog: list[str] = ["10", "11", "12"]
        self.subjects: list[str] = list(self.subject_catalog)
        self.grades: list[str] = list(self.block_catalog)
        self.subject_configs: list[dict] = []
        self.subject_management_mode = "subjects"
        self.subject_edit_index: int | None = None
        self.batch_editor_return_payload: dict | None = None
        self.batch_editor_return_session_id: str | None = None
        self._current_batch_data_source: str = "empty"
        self._route_history: list[dict] = []
        self._current_route_name: str = "exam_list"
        self._current_route_context: dict = {}
        self._suspend_route_history_push: bool = False

        self.omr_processor = OMRProcessor()
        self.scoring_engine = ScoringEngine()
        self.current_session_path: Path | None = None
        self.current_session_id: str | None = None
        self.session_dirty = False
        self._session_saved_signature = ""

        self.database = OMRDatabase.default()
        self.session_registry: list[dict[str, str | bool]] = self._load_session_registry()
        self.template_repo = self._load_template_repository()
        self.template_editor_embedded: TemplateEditorWindow | None = None
        self.template_editor_mode = "library"

        self.stack = _SimplePageHost(self)  # Compatibility alias; not the old Qt stacked widget.
        self.stack.addWidget(self._build_exam_list_page())
        self.stack.addWidget(self._build_workspace_page())
        self.stack.addWidget(self._build_subject_management_page())
        self.stack.addWidget(self._build_template_management_page())
        self.template_editor_page = QWidget()
        self.template_editor_layout = QVBoxLayout(self.template_editor_page)
        self.template_editor_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.template_editor_page)
        self.exam_editor_page = QWidget()
        self.exam_editor_layout = QVBoxLayout(self.exam_editor_page)
        self.exam_editor_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.exam_editor_page)
        self.embedded_exam_dialog: NewExamDialog | None = None
        self.embedded_exam_session_id: str | None = None
        self.embedded_exam_session: ExamSession | None = None
        self.embedded_exam_original_payload: dict | None = None
        self.embedded_exam_is_new: bool = False
        self.preview_zoom_factor = 0.3
        self.preview_source_pixmap = QPixmap()
        self.preview_rotation_by_index: dict[int, int] = {}
        self.scan_forced_status_by_index: dict[str, str] = {}
        self.deleted_scan_images_by_subject: dict[str, set[str]] = {}
        self._student_option_cache_session_id: str = ""
        self._student_option_labels_cache: list[str] = []
        self._student_option_sid_map: dict[str, str] = {}
        self._student_option_profile_map: dict[str, dict[str, str]] = {}
        self._student_option_sid_set: set[str] = set()
        self._student_option_cache_signature: str = ""
        self._scan_grid_loading = False
        self._switching_batch_subject = False
        self._template_cache_by_path: dict[str, Template] = {}
        self._answer_keys_ready_subjects: set[str] = set()
        self._batch_scan_running = False
        self._batch_cancel_requested = False
        self._batch_loaded_runtime_key: str = ""
        self._batch_loaded_subject_signature: str = ""
        self._auto_recognition_busy = False
        self._auto_recognition_queue: deque[str] = deque()
        self._auto_recognition_enqueued: set[str] = set()
        self._auto_recognition_last_seen: dict[str, tuple[int, float, int]] = {}
        self._auto_recognition_pause_requested = False
        self._auto_recognition_active_subject: str = ""
        self.preview_drag_active = False
        self.preview_last_pos = None
        self.setCentralWidget(self.stack)

        self._build_menu()
        self._apply_application_branding()
        self.stack.currentChanged.connect(self._handle_stack_changed)
        db_subjects = self.database.fetch_catalog("subjects")
        db_blocks = self.database.fetch_catalog("blocks")
        if db_subjects:
            # Persisted catalog order in DB is the source of truth.
            self.subject_catalog = list(db_subjects)
            self.subjects = list(db_subjects)
        else:
            # First run only: seed DB from built-in defaults.
            self.database.replace_catalog("subjects", self.subject_catalog)
        if db_blocks:
            self.block_catalog = list(db_blocks)
            self.grades = list(db_blocks)
        else:
            self.database.replace_catalog("blocks", self.block_catalog)
        self._refresh_exam_list()
        self._refresh_batch_subject_controls()
        self._handle_stack_changed(self.stack.currentIndex())
        self.stack.setCurrentIndex(0)
        self._setup_auto_recognition_timer()


    def _load_api_mapping_rows(self, *args, **kwargs) -> list[dict]:
        """Load SBD/student mapping rows for legacy recheck and scoring flows.

        Older Phúc tra code calls ``self._load_api_mapping_rows(...)`` to read the
        configured SBD list. During the MainWindow split into mixins this helper
        was dropped, so the dialog crashed before it could build the student list.

        This implementation is intentionally tolerant: it accepts a path, a
        subject config dict, a list of rows, or no argument at all. With no
        argument it resolves the active subject config first, then falls back to
        ``self.session.students`` so Phúc tra never depends on already-scanned
        submissions as the only student source.
        """

        def _norm_key(value: object) -> str:
            text = unicodedata.normalize("NFD", str(value or "").strip().lower())
            text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
            text = re.sub(r"[^a-z0-9]+", "", text)
            return text

        def _cell_text(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, bool):
                return "1" if value else "0"
            if isinstance(value, int):
                return str(value)
            if isinstance(value, float):
                if value.is_integer():
                    return str(int(value))
                return (f"{value:.12g}").strip()
            return str(value).strip()

        class _ApiMappingRow(dict):
            """Dict-compatible row that also supports legacy tuple unpacking.

            Some older Phúc tra code iterates rows as ``for sbd, label in rows``.
            A plain dict would be unpacked through its keys and can raise
            "too many values to unpack (expected 2)". This wrapper keeps
            normal ``row.get(...)`` access while direct iteration yields exactly
            two values: SBD and a human-readable label.
            """
            def __iter__(self):
                sid = str(self.get("student_id", "") or self.get("sbd", "") or self.get("SBD", "") or "").strip()
                name = str(self.get("name", "") or self.get("full_name", "") or self.get("Họ tên", "") or "").strip()
                class_name = str(self.get("class_name", "") or self.get("class", "") or self.get("Lớp", "") or "").strip()
                room = str(self.get("exam_room", "") or self.get("room", "") or self.get("Phòng thi", "") or "").strip()
                pieces = [sid]
                if name:
                    pieces.append(name)
                if class_name:
                    pieces.append(class_name)
                if room:
                    pieces.append(f"Phòng {room}")
                label = " - ".join(pieces) if pieces else str(dict.get(self, "label", "") or "")
                yield sid
                yield label

        aliases = {
            "student_id": {
                "sbd", "sobao", "sobd", "sobaodanh", "studentid", "studentcode",
                "mathisinh", "masohocsinh", "mahs", "mahocsinh", "id",
            },
            "name": {
                "hoten", "hovaten", "ten", "tenthisinh", "tenhocsinh", "fullname",
                "name", "studentname",
            },
            "class_name": {
                "lop", "class", "classname", "lopthi", "khoi", "grade", "nhom",
            },
            "birth_date": {
                "ngaysinh", "namsinh", "dob", "birthday", "birthdate", "dateofbirth",
            },
            "exam_room": {
                "phong", "phongthi", "room", "examroom", "diadiem", "maphong",
            },
            "exam_code": {
                "made", "made thi", "examcode", "code", "mamade",
            },
        }

        def _canonical_key(raw_key: object) -> str:
            nk = _norm_key(raw_key)
            for canonical, names in aliases.items():
                if nk in names:
                    return canonical
            return str(raw_key or "").strip()

        def _canonicalize_row(row: object, default_room: str = "") -> dict:
            if row is None:
                return {}
            if isinstance(row, dict):
                source = dict(row)
            else:
                sid = _cell_text(row)
                source = {"student_id": sid}
            out: dict[str, str] = {}
            for k, v in source.items():
                ck = _canonical_key(k)
                val = _cell_text(v)
                if ck in {"student_id", "name", "class_name", "birth_date", "exam_room", "exam_code"}:
                    if val or ck not in out:
                        out[ck] = val
                else:
                    out[str(k or "").strip()] = val
            if default_room and not out.get("exam_room"):
                out["exam_room"] = str(default_room or "").strip()
            sid = str(out.get("student_id", "") or out.get("SBD", "") or "").strip()
            if not sid:
                return {}
            # Keep both canonical keys and legacy/Vietnamese keys because older
            # modules may access either style.
            out["student_id"] = sid
            out.setdefault("sbd", sid)
            out.setdefault("SBD", sid)
            if out.get("name"):
                out.setdefault("full_name", out["name"])
                out.setdefault("Họ tên", out["name"])
            if out.get("class_name"):
                out.setdefault("class", out["class_name"])
                out.setdefault("Lớp", out["class_name"])
            if out.get("birth_date"):
                out.setdefault("Ngày sinh", out["birth_date"])
            if out.get("exam_room"):
                out.setdefault("room", out["exam_room"])
                out.setdefault("Phòng thi", out["exam_room"])
            if out.get("exam_code"):
                out.setdefault("Mã đề", out["exam_code"])
            return out

        def _rows_from_session() -> list[dict]:
            rows: list[dict] = []
            session = getattr(self, "session", None)
            for st in (getattr(session, "students", []) or []):
                row = {
                    "student_id": _cell_text(getattr(st, "student_id", "")),
                    "name": _cell_text(getattr(st, "name", "") or getattr(st, "full_name", "")),
                    "class_name": _cell_text(getattr(st, "class_name", "") or getattr(st, "classroom", "") or getattr(st, "grade", "")),
                    "birth_date": _cell_text(getattr(st, "birth_date", "") or getattr(st, "dob", "")),
                    "exam_room": _cell_text(getattr(st, "exam_room", "") or getattr(st, "room", "")),
                }
                normalized = _canonicalize_row(row)
                if normalized:
                    rows.append(normalized)
            return rows

        def _read_csv(path: Path) -> list[dict]:
            last_exc: Exception | None = None
            for enc in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
                try:
                    text = path.read_text(encoding=enc)
                    sample = text[:4096]
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    except Exception:
                        dialect = csv.excel
                    reader = csv.DictReader(text.splitlines(), dialect=dialect)
                    return [_canonicalize_row(row) for row in reader if _canonicalize_row(row)]
                except Exception as exc:
                    last_exc = exc
            if last_exc:
                raise last_exc
            return []

        def _rows_from_table(raw_rows: list[tuple]) -> list[dict]:
            if not raw_rows:
                return []
            header_index = 0
            for idx, raw in enumerate(raw_rows[:30]):
                keys = {_canonical_key(x) for x in raw if _cell_text(x)}
                if "student_id" in keys or ({"name", "class_name"} & keys):
                    header_index = idx
                    break
            headers = [_cell_text(x) for x in raw_rows[header_index]]
            out: list[dict] = []
            for raw in raw_rows[header_index + 1:]:
                payload = {headers[i] if i < len(headers) else f"col_{i+1}": raw[i] for i in range(len(raw))}
                normalized = _canonicalize_row(payload)
                if normalized:
                    out.append(normalized)
            return out

        def _read_excel(path: Path) -> list[dict]:
            # .xlsx/.xlsm: fast path with openpyxl. .xls or unusual engines: fallback to pandas if available.
            if path.suffix.lower() != ".xls":
                try:
                    from openpyxl import load_workbook

                    wb = load_workbook(path, read_only=True, data_only=True)
                    ws = wb.active
                    return _rows_from_table(list(ws.iter_rows(values_only=True)))
                except Exception:
                    pass
            try:
                import pandas as pd

                df = pd.read_excel(path, dtype=str, header=None)
                raw_rows = [tuple(row) for row in df.fillna("").itertuples(index=False, name=None)]
                return _rows_from_table(raw_rows)
            except Exception as exc:
                raise exc

        def _read_json(path: Path) -> list[dict]:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            return _rows_from_source(data)

        def _rows_from_room_mapping(mapping: object) -> list[dict]:
            rows: list[dict] = []
            if isinstance(mapping, dict):
                for room, values in mapping.items():
                    room_text = _cell_text(room)
                    if isinstance(values, dict):
                        # Accept {room: {sbd: info}} as well as {room: {"rows": [...]}}.
                        nested = values.get("rows") or values.get("students") or values.get("items")
                        if nested is not None:
                            rows.extend(_rows_from_source(nested, default_room=room_text))
                            continue
                        for sid, info in values.items():
                            if isinstance(info, dict):
                                item = dict(info)
                                item.setdefault("student_id", sid)
                                rows.append(_canonicalize_row(item, room_text))
                            else:
                                rows.append(_canonicalize_row({"student_id": sid, "name": info}, room_text))
                    elif isinstance(values, (list, tuple, set)):
                        rows.extend(_rows_from_source(list(values), default_room=room_text))
                    else:
                        rows.append(_canonicalize_row({"student_id": values}, room_text))
            elif isinstance(mapping, (list, tuple)):
                rows.extend(_rows_from_source(mapping))
            return [r for r in rows if r]

        def _rows_from_path(path_like: object) -> list[dict]:
            path_text = _cell_text(path_like)
            if not path_text:
                return []
            path = Path(path_text).expanduser()
            if not path.exists():
                # Some configs store relative paths beside the session file.
                current_session_path = getattr(self, "current_session_path", None)
                if current_session_path:
                    base = Path(current_session_path).parent if Path(current_session_path).suffix else Path(current_session_path)
                    candidate = base / path_text
                    if candidate.exists():
                        path = candidate
            if not path.exists():
                return []
            suffix = path.suffix.lower()
            if suffix in {".csv", ".txt"}:
                return _read_csv(path)
            if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}:
                return _read_excel(path)
            if suffix == ".json":
                return _read_json(path)
            # Try CSV first for unknown simple text files.
            try:
                return _read_csv(path)
            except Exception:
                return []

        def _rows_from_source(source: object, default_room: str = "") -> list[dict]:
            if source is None:
                return []
            if isinstance(source, (str, os.PathLike, Path)):
                return _rows_from_path(source)
            if isinstance(source, dict):
                # Direct row dict.
                if any(_canonical_key(k) == "student_id" for k in source.keys()):
                    row = _canonicalize_row(source, default_room)
                    return [row] if row else []
                # Config dict with embedded row/path/mapping fields.
                config_row_keys = (
                    "api_mapping_rows", "mapping_rows", "student_rows", "students",
                    "sbd_rows", "sbd_mapping_rows", "imported_student_rows",
                    "exam_students", "student_list", "student_mapping_rows",
                )
                rows: list[dict] = []
                for key in config_row_keys:
                    if key in source and source.get(key) not in (None, ""):
                        rows.extend(_rows_from_source(source.get(key), default_room=default_room))
                room_keys = (
                    "exam_room_sbd_mapping_by_room", "sbd_room_mapping",
                    "room_sbd_mapping", "exam_room_mapping", "room_mapping",
                )
                for key in room_keys:
                    if key in source and source.get(key) not in (None, ""):
                        rows.extend(_rows_from_room_mapping(source.get(key)))
                file_keys = (
                    "api_mapping_file", "api_mapping_path", "student_file_path",
                    "student_list_path", "sbd_file", "sbd_file_path",
                    "mapping_file", "mapping_path", "student_mapping_file",
                )
                for key in file_keys:
                    if key in source and source.get(key):
                        rows.extend(_rows_from_path(source.get(key)))
                if rows:
                    return rows
                # Maybe dict is {sid: profile}.
                for sid, info in source.items():
                    if isinstance(info, dict):
                        item = dict(info)
                        item.setdefault("student_id", sid)
                        row = _canonicalize_row(item, default_room)
                    else:
                        row = _canonicalize_row({"student_id": sid, "name": info}, default_room)
                    if row:
                        rows.append(row)
                return rows
            if isinstance(source, (list, tuple, set)):
                rows: list[dict] = []
                for item in source:
                    rows.extend(_rows_from_source(item, default_room=default_room))
                return rows
            row = _canonicalize_row(source, default_room)
            return [row] if row else []

        explicit_sources: list[object] = []
        for value in args:
            if value not in (None, ""):
                explicit_sources.append(value)
        for key in ("rows", "source", "cfg", "config", "path", "file_path", "mapping_file"):
            value = kwargs.get(key)
            if value not in (None, ""):
                explicit_sources.append(value)

        subject_key = str(kwargs.get("subject_key") or kwargs.get("subject") or "").strip()
        if not subject_key:
            subject_key = str(getattr(self, "_current_scoring_subject", "") or getattr(self, "active_batch_subject_key", "") or "").strip()
        if not subject_key and hasattr(self, "scoring_subject_combo"):
            try:
                combo = self.scoring_subject_combo
                subject_key = str(combo.currentData() or combo.currentText() or "").strip()
            except Exception:
                subject_key = ""
        if not subject_key and hasattr(self, "batch_subject_combo"):
            try:
                combo = self.batch_subject_combo
                subject_key = str(combo.currentData() or combo.currentText() or "").strip()
            except Exception:
                subject_key = ""

        if not explicit_sources and subject_key and hasattr(self, "_subject_config_by_subject_key"):
            try:
                cfg = self._subject_config_by_subject_key(subject_key) or {}
                if isinstance(cfg, dict):
                    explicit_sources.append(cfg)
            except Exception:
                pass
        if not explicit_sources and getattr(self, "subject_configs", None):
            explicit_sources.extend([cfg for cfg in (getattr(self, "subject_configs", []) or []) if isinstance(cfg, dict)])

        rows: list[dict] = []
        for source in explicit_sources:
            rows.extend(_rows_from_source(source))
        if not rows:
            rows = _rows_from_session()

        # Deduplicate by SBD, preferring rows with more profile fields.
        dedup: dict[str, dict] = {}
        for row in rows:
            row = _canonicalize_row(row)
            sid = str((row or {}).get("student_id", "") or "").strip()
            if not sid:
                continue
            old = dedup.get(sid)
            if not old:
                dedup[sid] = row
                continue
            old_score = sum(1 for k in ("name", "class_name", "birth_date", "exam_room") if old.get(k))
            new_score = sum(1 for k in ("name", "class_name", "birth_date", "exam_room") if row.get(k))
            if new_score >= old_score:
                merged = dict(old)
                merged.update({k: v for k, v in row.items() if str(v or "").strip()})
                dedup[sid] = merged
        return [_ApiMappingRow(row) for row in dedup.values()]



def run() -> None:
    bootstrap_application_db()
    app = QApplication([])
    window = MainWindow()
    window.showMaximized()
    app.exec()


if __name__ == "__main__":
    run()
