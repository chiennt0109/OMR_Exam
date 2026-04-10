from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.omr_engine import OMRResult
from models.answer_key import SubjectKey


def open_recheck_dialog(self) -> None:
    # Compatibility shim for older patched variants that may still attempt
    # to wire btn_pick/btn_unpick callbacks at the end of this function.
    # Keeping these placeholders prevents NameError in mixed deployments.
    btn_pick = QPushButton()
    btn_unpick = QPushButton()
    def _pick_from_pool() -> None:
        return
    def _remove_selected_recheck_row() -> None:
        return

    subject_cfgs = [cfg for cfg in self._effective_subject_configs_for_batch() if isinstance(cfg, dict)]
    subject_keys = [self._subject_key_from_cfg(cfg) for cfg in subject_cfgs if self._subject_key_from_cfg(cfg)]
    if not subject_keys:
        QMessageBox.warning(self, "Phúc tra", "Chưa có môn để phúc tra.")
        return
    subject_key, ok = QInputDialog.getItem(self, "Phúc tra", "Chọn môn:", subject_keys, 0, False)
    if not ok or not subject_key:
        return

    def _load_sid_list_from_file() -> list[str]:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file SBD phúc tra", "", "Data files (*.xlsx *.csv *.txt *.tsv);;All files (*.*)")
        if not path:
            return []
        try:
            headers, data_rows = self._load_api_mapping_rows(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Phúc tra", f"Không đọc được file SBD:\n{exc}")
            return []
        if not data_rows:
            QMessageBox.warning(self, "Phúc tra", "File SBD rỗng.")
            return []
        sid_col_name, ok_pick = QInputDialog.getItem(self, "Phúc tra", "Chọn cột SBD:", headers, 0, False)
        if not ok_pick or not sid_col_name:
            return []
        out: list[str] = []
        for row_obj in data_rows:
            raw_sid = str((row_obj or {}).get(sid_col_name, "") or "").strip()
            if self._normalized_student_id_for_match(raw_sid):
                out.append(raw_sid)
        return out

    session_cfg = dict(self.session.config or {}) if self.session else {}
    recheck_sid_lists = session_cfg.get("recheck_sid_lists", {}) if isinstance(session_cfg.get("recheck_sid_lists", {}), dict) else {}
    sid_cache_key = f"recheck_sid_list::{str(self.current_session_id or '')}::{subject_key}"
    flag_key = f"recheck_flag::{str(self.current_session_id or '')}::{subject_key}"
    cached_sid_list = [str(x).strip() for x in (self.database.get_app_state(sid_cache_key, []) or []) if str(x).strip()]
    if not cached_sid_list:
        cached_sid_list = [str(x).strip() for x in (recheck_sid_lists.get(subject_key, []) or []) if str(x).strip()]
    requested_sids = list(cached_sid_list)
    def _persist_recheck_sid_list(current_sids: list[str]) -> None:
        if self.session:
            cfg = dict(self.session.config or {})
            sid_cache = cfg.get("recheck_sid_lists", {}) if isinstance(cfg.get("recheck_sid_lists", {}), dict) else {}
            sid_cache[str(subject_key)] = list(current_sids)
            cfg["recheck_sid_lists"] = sid_cache
            self.session.config = cfg
            self.session_dirty = True
            self._persist_session_quietly()
        self.database.set_app_state(sid_cache_key, list(current_sids))
        self.database.set_app_state(flag_key, True)

    def _normalize_sid_list(raw_sids: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in raw_sids:
            sid_text = str(raw or "").strip()
            sid_norm = self._normalized_student_id_for_match(sid_text)
            if not sid_norm or sid_norm in seen:
                continue
            out.append(sid_text)
            seen.add(sid_norm)
        return out

    def _subject_room_for_sid_quick(sid: str) -> str:
        cfg = self._subject_config_by_subject_key(subject_key) or {}
        return self._subject_room_for_student_id(sid, cfg)

    def _open_recheck_list_builder(initial_sids: list[str]) -> list[str] | None:
        builder = QDialog(self)
        builder.setWindowTitle(f"Lập danh sách phúc tra - {subject_key}")
        builder.resize(1080, 680)
        lay = QVBoxLayout(builder)
        lay.addWidget(QLabel("Bước 1/2: Chọn thí sinh cần phúc tra cho môn đã chọn"))
        split_pick = QSplitter(Qt.Horizontal)
        lay.addWidget(split_pick, 1)

        src_tbl = QTableWidget(0, 4)
        src_tbl.setHorizontalHeaderLabels(["SBD", "Họ tên", "Lớp", "Phòng thi"])
        src_tbl.verticalHeader().setVisible(False)
        src_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        src_tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        src_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        src_tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        chosen_tbl = QTableWidget(0, 4)
        chosen_tbl.setHorizontalHeaderLabels(["SBD", "Họ tên", "Lớp", "Phòng thi"])
        chosen_tbl.verticalHeader().setVisible(False)
        chosen_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        chosen_tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        chosen_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        chosen_tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        left_wrap = QWidget(); left_l = QVBoxLayout(left_wrap)
        left_l.addWidget(QLabel("Danh sách theo phòng thi của môn"))
        src_filter = QLineEdit()
        src_filter.setPlaceholderText("Lọc theo SBD / Họ tên / Lớp / Phòng thi...")
        left_l.addWidget(src_filter)
        left_l.addWidget(src_tbl)
        split_pick.addWidget(left_wrap)

        action_wrap = QWidget(); action_l = QVBoxLayout(action_wrap)
        btn_add = QPushButton("Chọn >>")
        btn_remove = QPushButton("<< Bỏ chọn")
        btn_add_all = QPushButton("Chọn tất cả >>")
        btn_remove_all = QPushButton("<< Bỏ chọn tất cả")
        btn_import = QPushButton("Import từ Excel")
        action_l.addStretch(1)
        action_l.addWidget(btn_add)
        action_l.addWidget(btn_remove)
        action_l.addWidget(btn_add_all)
        action_l.addWidget(btn_remove_all)
        action_l.addSpacing(18)
        action_l.addWidget(btn_import)
        action_l.addStretch(1)
        split_pick.addWidget(action_wrap)

        right_wrap = QWidget(); right_l = QVBoxLayout(right_wrap)
        right_l.addWidget(QLabel("Danh sách phúc tra đã chọn"))
        right_l.addWidget(chosen_tbl)
        split_pick.addWidget(right_wrap)
        split_pick.setStretchFactor(0, 1)
        split_pick.setStretchFactor(1, 0)
        split_pick.setStretchFactor(2, 1)
        split_pick.setSizes([420, 140, 420])

        subject_cfg_local = self._subject_config_by_subject_key(subject_key) or {}
        has_room_mapping = False
        if isinstance(subject_cfg_local, dict):
            room_map_by_room = subject_cfg_local.get("exam_room_sbd_mapping_by_room", {})
            room_map_legacy = str(subject_cfg_local.get("exam_room_sbd_mapping", "") or "").strip()
            has_room_mapping = bool(room_map_by_room) or bool(room_map_legacy)

        available_students: list[dict[str, str]] = []
        for st in (self.session.students or []) if self.session else []:
            sid_val = str(getattr(st, "student_id", "") or "").strip()
            if not sid_val:
                continue
            room = _subject_room_for_sid_quick(sid_val)
            if not room or room == "-":
                continue
            profile = self._student_profile_by_id(sid_val)
            available_students.append(
                {
                    "sid": sid_val,
                    "name": str((profile.get("name", "") or getattr(st, "name", "") or "-")),
                    "class_name": str((profile.get("class_name", "") or getattr(st, "class_name", "") or "-")),
                    "room": room,
                }
            )
        if not available_students and not has_room_mapping:
            for st in (self.session.students or []) if self.session else []:
                sid_val = str(getattr(st, "student_id", "") or "").strip()
                if sid_val:
                    profile = self._student_profile_by_id(sid_val)
                    available_students.append(
                        {
                            "sid": sid_val,
                            "name": str((profile.get("name", "") or getattr(st, "name", "") or "-")),
                            "class_name": str((profile.get("class_name", "") or getattr(st, "class_name", "") or "-")),
                            "room": _subject_room_for_sid_quick(sid_val) or "-",
                        }
                    )
        available_students.sort(key=lambda x: (str(x.get("room", "")), str(x.get("sid", ""))))
        profile_by_sid = {str(x.get("sid", "")): dict(x) for x in available_students}

        for row_obj in available_students:
            r = src_tbl.rowCount()
            src_tbl.insertRow(r)
            src_tbl.setItem(r, 0, QTableWidgetItem(str(row_obj.get("sid", "") or "-")))
            src_tbl.setItem(r, 1, QTableWidgetItem(str(row_obj.get("name", "") or "-")))
            src_tbl.setItem(r, 2, QTableWidgetItem(str(row_obj.get("class_name", "") or "-")))
            src_tbl.setItem(r, 3, QTableWidgetItem(str(row_obj.get("room", "") or "-")))

        chosen_sids = _normalize_sid_list(initial_sids)

        def _render_chosen() -> None:
            chosen_tbl.setRowCount(0)
            for sid in chosen_sids:
                prof = profile_by_sid.get(sid, {})
                if not prof:
                    st_prof = self._student_profile_by_id(sid)
                    prof = {"sid": sid, "name": str(st_prof.get("name", "") or "-"), "class_name": str(st_prof.get("class_name", "") or "-"), "room": _subject_room_for_sid_quick(sid) or "-"}
                r = chosen_tbl.rowCount()
                chosen_tbl.insertRow(r)
                chosen_tbl.setItem(r, 0, QTableWidgetItem(str(prof.get("sid", sid) or "-")))
                chosen_tbl.setItem(r, 1, QTableWidgetItem(str(prof.get("name", "") or "-")))
                chosen_tbl.setItem(r, 2, QTableWidgetItem(str(prof.get("class_name", "") or "-")))
                chosen_tbl.setItem(r, 3, QTableWidgetItem(str(prof.get("room", "") or "-")))

        def _add_from_source() -> None:
            rows = sorted({item.row() for item in src_tbl.selectedItems()})
            for row_idx in rows:
                sid_item = src_tbl.item(row_idx, 0)
                sid_text = str(sid_item.text() if sid_item else "").strip()
                if sid_text:
                    chosen_sids.append(sid_text)
            chosen_sids[:] = _normalize_sid_list(chosen_sids)
            _render_chosen()

        def _remove_from_chosen() -> None:
            rows = sorted({item.row() for item in chosen_tbl.selectedItems()}, reverse=True)
            for row_idx in rows:
                sid_item = chosen_tbl.item(row_idx, 0)
                sid_text = str(sid_item.text() if sid_item else "").strip()
                sid_norm = self._normalized_student_id_for_match(sid_text)
                chosen_sids[:] = [x for x in chosen_sids if self._normalized_student_id_for_match(x) != sid_norm]
            _render_chosen()

        def _import_excel_list() -> None:
            imported = _load_sid_list_from_file()
            if not imported:
                return
            decision = QMessageBox.question(
                builder,
                "Import danh sách",
                "Bạn muốn mở rộng danh sách hiện tại?\nYes = Mở rộng, No = Ghi đè, Cancel = Huỷ.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if decision == QMessageBox.Cancel:
                return
            chosen_sids[:] = _normalize_sid_list((chosen_sids + imported) if decision == QMessageBox.Yes else imported)
            _render_chosen()

        def _add_all_from_source() -> None:
            for row_idx in range(src_tbl.rowCount()):
                if src_tbl.isRowHidden(row_idx):
                    continue
                sid_item = src_tbl.item(row_idx, 0)
                sid_text = str(sid_item.text() if sid_item else "").strip()
                if sid_text:
                    chosen_sids.append(sid_text)
            chosen_sids[:] = _normalize_sid_list(chosen_sids)
            _render_chosen()

        def _remove_all_chosen() -> None:
            chosen_sids.clear()
            _render_chosen()

        def _apply_source_filter() -> None:
            keyword = str(src_filter.text() or "").strip().lower()
            for row_idx in range(src_tbl.rowCount()):
                row_text = " | ".join(
                    str(src_tbl.item(row_idx, c).text() if src_tbl.item(row_idx, c) else "").strip().lower()
                    for c in range(src_tbl.columnCount())
                )
                src_tbl.setRowHidden(row_idx, bool(keyword) and keyword not in row_text)

        btn_add.clicked.connect(_add_from_source)
        btn_remove.clicked.connect(_remove_from_chosen)
        btn_add_all.clicked.connect(_add_all_from_source)
        btn_remove_all.clicked.connect(_remove_all_chosen)
        btn_import.clicked.connect(_import_excel_list)
        src_filter.textChanged.connect(_apply_source_filter)
        _render_chosen()

        footer = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        lay.addWidget(footer)

        def _accept_builder() -> None:
            if not chosen_sids:
                QMessageBox.warning(builder, "Phúc tra", "Bạn chưa chọn thí sinh nào để lập danh sách phúc tra.")
                return
            builder.accept()

        footer.accepted.connect(_accept_builder)
        footer.rejected.connect(builder.reject)
        if builder.exec() != QDialog.Accepted:
            return None
        return list(chosen_sids)

    built_sids = _open_recheck_list_builder(requested_sids)
    if built_sids is None:
        return
    requested_sids = _normalize_sid_list(built_sids)
    _persist_recheck_sid_list(requested_sids)

    loading = QProgressDialog("Đang khởi tạo màn hình phúc tra...", "", 0, 4, self)
    loading.setWindowTitle("Vui lòng chờ")
    loading.setCancelButton(None)
    loading.setWindowModality(Qt.ApplicationModal)
    loading.setMinimumDuration(0)
    loading.setValue(1)
    loading.show()
    QApplication.processEvents()

    result_rows = self.database.fetch_scan_results_for_subject(self._batch_result_subject_key(subject_key)) or []
    loading.setValue(2)
    QApplication.processEvents()
    scans = [self._deserialize_omr_result(x) for x in result_rows]
    scan_by_sid_norm: dict[str, OMRResult] = {}
    for scan_item in scans:
        norm_sid = self._normalized_student_id_for_match(str(getattr(scan_item, "student_id", "") or ""))
        if norm_sid and norm_sid not in scan_by_sid_norm:
            scan_by_sid_norm[norm_sid] = scan_item
    requested_sids = _normalize_sid_list(requested_sids)

    recheck_entries: list[dict[str, object]] = []

    def _rebuild_recheck_entries() -> None:
        recheck_entries.clear()
        for raw_sid in requested_sids:
            sid_norm = self._normalized_student_id_for_match(raw_sid)
            if not sid_norm:
                continue
            recheck_entries.append(
                {
                    "requested_sid": raw_sid,
                    "sid_norm": sid_norm,
                    "result": scan_by_sid_norm.get(sid_norm),
                }
            )

    _rebuild_recheck_entries()
    loading.setValue(3)
    QApplication.processEvents()

    exam_codes = sorted(
        {
            str(code or "").strip()
            for code in (self._fetch_answer_keys_for_subject_scoped(subject_key) or {}).keys()
            if str(code or "").strip()
        }
    )
    sid_options: list[str] = []
    sid_to_display: dict[str, str] = {}
    for st in (self.session.students or []) if self.session else []:
        sid_val = str(getattr(st, "student_id", "") or "").strip()
        if not sid_val:
            continue
        display = f"{sid_val} - {str(getattr(st, 'name', '') or '').strip()}"
        sid_options.append(display)
        sid_to_display[sid_val] = display

    def _parse_pairs(raw: str, upper: bool = False) -> dict[int, str]:
        out: dict[int, str] = {}
        tokens = [x.strip() for x in str(raw or "").replace("\n", ",").split(",") if x.strip()]
        for token in tokens:
            if ":" not in token:
                continue
            left, right = token.split(":", 1)
            q_text = "".join(ch for ch in left if ch.isdigit() or ch == "-").strip()
            if not q_text.lstrip("-").isdigit():
                continue
            q_no = int(q_text)
            value = str(right or "").strip()
            out[q_no] = value.upper() if upper else value
        return out

    def _tf_to_display(tf_map: dict[int, dict[str, bool]]) -> str:
        return self._format_tf_answers(tf_map or {})

    def _parse_tf_display(raw: str) -> dict[int, dict[str, bool]]:
        out: dict[int, dict[str, bool]] = {}
        pairs = _parse_pairs(raw, upper=True)
        for q_no, text in pairs.items():
            marks: dict[str, bool] = {}
            for idx, ch in enumerate(text[:4]):
                if ch in {"Đ", "D", "T", "1"}:
                    marks[["a", "b", "c", "d"][idx]] = True
                elif ch in {"S", "F", "0"}:
                    marks[["a", "b", "c", "d"][idx]] = False
            if marks:
                out[q_no] = marks
        return out

    def _current_score_for_result(res: OMRResult) -> float:
        self._ensure_answer_keys_for_subject(subject_key)
        cfg = self._subject_config_by_subject_key(subject_key) or {}
        all_subject_keys: list[SubjectKey] = []
        if self.answer_keys is not None:
            for key_obj in self.answer_keys.keys.values():
                if isinstance(key_obj, SubjectKey) and str(getattr(key_obj, "subject", "") or "").strip() == subject_key:
                    all_subject_keys.append(key_obj)
        fetched_keys = self._fetch_answer_keys_for_subject_scoped(subject_key)
        for exam_code, key_obj in (fetched_keys or {}).items():
            if isinstance(key_obj, SubjectKey):
                candidate_key = key_obj
            elif isinstance(key_obj, dict):
                candidate_key = SubjectKey(
                    subject=subject_key,
                    exam_code=str(exam_code or "").strip(),
                    answers={int(k): str(v) for k, v in ((key_obj.get("mcq_answers", {}) or {}).items() if isinstance(key_obj.get("mcq_answers", {}), dict) else [])},
                    true_false_answers={int(k): dict(v or {}) for k, v in ((key_obj.get("true_false_answers", {}) or {}).items() if isinstance(key_obj.get("true_false_answers", {}), dict) else [])},
                    numeric_answers={int(k): str(v) for k, v in ((key_obj.get("numeric_answers", {}) or {}).items() if isinstance(key_obj.get("numeric_answers", {}), dict) else [])},
                    full_credit_questions={
                        str(sec): [int(x) for x in vals if str(x).strip().lstrip("-").isdigit()]
                        for sec, vals in ((key_obj.get("full_credit_questions", {}) or {}).items() if isinstance(key_obj.get("full_credit_questions", {}), dict) else [])
                    },
                    invalid_answer_rows={
                        str(sec): {int(q): str(v) for q, v in (bucket or {}).items() if str(q).strip().lstrip("-").isdigit()}
                        for sec, bucket in ((key_obj.get("invalid_answer_rows", {}) or {}).items() if isinstance(key_obj.get("invalid_answer_rows", {}), dict) else [])
                    },
                )
            else:
                continue
            if any(str(k.exam_code or "").strip() == str(exam_code or "").strip() for k in all_subject_keys):
                continue
            all_subject_keys.append(candidate_key)
        key = self.answer_keys.get_flexible(subject_key, str(getattr(res, "exam_code", "") or "").strip()) if self.answer_keys else None
        try:
            if key:
                return float(self.scoring_engine.score(res, key, student_name=str(getattr(res, "full_name", "") or ""), subject_config=cfg).score or 0.0)
            best = 0.0
            for candidate in all_subject_keys:
                row = self.scoring_engine.score(res, candidate, student_name=str(getattr(res, "full_name", "") or ""), subject_config=cfg)
                best = max(best, float(getattr(row, "score", 0.0) or 0.0))
            return best
        except Exception:
            return 0.0

    history_all = self.database.fetch_recheck_history(str(self.current_session_id or ""), subject_key=subject_key)

    dlg = QDialog(self)
    dlg.setWindowTitle(f"Phúc tra - {subject_key}")
    dlg.resize(1760, 940)
    root = QVBoxLayout(dlg)
    split = QSplitter(Qt.Horizontal)
    root.addWidget(split)

    left = QWidget()
    left_l = QVBoxLayout(left)
    student_pool_tbl = QTableWidget(0, 4)
    student_pool_tbl.setHorizontalHeaderLabels(["SBD", "Họ tên", "Lớp", "Phòng thi"])
    student_pool_tbl.verticalHeader().setVisible(False)
    student_pool_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
    student_pool_tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
    student_pool_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    student_pool_tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    tbl = QTableWidget(0, 7)
    tbl.setHorizontalHeaderLabels(["STT", "SBD", "Họ tên", "Lớp", "Phòng thi", "Mã đề", "Điểm"])
    tbl.verticalHeader().setVisible(False)
    tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
    tbl.setSelectionMode(QAbstractItemView.SingleSelection)
    tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    left_l.addWidget(tbl)
    btn_build_list = QPushButton("Lập danh sách")
    btn_add_list = QPushButton("Thêm danh sách")
    btn_export = QPushButton("Xuất Excel phúc tra")
    left_actions = QHBoxLayout()
    left_actions.addWidget(btn_build_list)
    left_actions.addWidget(btn_add_list)
    left_actions.addStretch(1)
    left_actions.addWidget(btn_export)
    left_l.addLayout(left_actions)
    split.addWidget(left)

    middle = QWidget()
    middle_l = QVBoxLayout(middle)
    form = QFormLayout()
    inp_sid = QComboBox(); inp_sid.setEditable(True); inp_sid.addItems(sid_options)
    sid_completer = QCompleter(sid_options, inp_sid)
    sid_completer.setCaseSensitivity(Qt.CaseInsensitive)
    sid_completer.setFilterMode(Qt.MatchContains)
    inp_sid.setCompleter(sid_completer)
    inp_exam = QComboBox(); inp_exam.setEditable(True); inp_exam.addItems(exam_codes)
    lbl_score = QLabel("-")
    lbl_recheck_info = QLabel("-")
    lbl_recheck_info.setWordWrap(True)
    form.addRow("SBD (có tìm kiếm)", inp_sid)
    form.addRow("Mã đề", inp_exam)
    form.addRow("Điểm hiện tại", lbl_score)
    form.addRow("Nội dung phúc tra", lbl_recheck_info)
    middle_l.addLayout(form)
    answer_tbl = QTableWidget(0, 4)
    answer_tbl.setHorizontalHeaderLabels(["Phần", "Câu", "Đáp án", "Bài làm"])
    answer_tbl.verticalHeader().setVisible(False)
    answer_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
    answer_tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
    answer_tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    answer_tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
    answer_tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
    middle_l.addWidget(QLabel("Đáp án đúng/sai theo từng câu"))
    middle_l.addWidget(answer_tbl, 1)
    action_row = QHBoxLayout()
    btn_save = QPushButton("Lưu phúc tra + tính lại")
    action_row.addStretch(1)
    action_row.addWidget(btn_save)
    middle_l.addLayout(action_row)
    split.addWidget(middle)

    right = QSplitter(Qt.Vertical)
    image_panel = QWidget()
    image_l = QVBoxLayout(image_panel)
    image_l.addWidget(QLabel("Ảnh bài thi"))
    img_scroll = QScrollArea()
    img_scroll.setWidgetResizable(True)
    img_lbl = QLabel("-")
    img_lbl.setAlignment(Qt.AlignCenter)
    img_scroll.setWidget(img_lbl)
    image_l.addWidget(img_scroll, 1)
    right.addWidget(image_panel)
    history_panel = QWidget()
    history_l = QVBoxLayout(history_panel)
    history_l.addWidget(QLabel("Lịch sử sửa bài"))
    history_txt = QTextEdit()
    history_txt.setReadOnly(True)
    history_l.addWidget(history_txt, 1)
    right.addWidget(history_panel)
    split.addWidget(right)
    split.setStretchFactor(0, 1)
    split.setStretchFactor(1, 2)
    split.setStretchFactor(2, 1)
    split.setSizes([420, 760, 520])
    right.setStretchFactor(0, 2)
    right.setStretchFactor(1, 1)
    editor_refs: dict[str, object] = {"row_map": []}

    def _answer_key_for_exam(exam_code_text: str):
        code = str(exam_code_text or "").strip()
        key_obj = self.answer_keys.get_flexible(subject_key, code) if self.answer_keys else None
        if isinstance(key_obj, SubjectKey):
            return key_obj
        raw = (self._fetch_answer_keys_for_subject_scoped(subject_key) or {}).get(code)
        if isinstance(raw, dict):
            return SubjectKey(
                subject=subject_key,
                exam_code=code,
                answers={int(k): str(v) for k, v in ((raw.get("mcq_answers", {}) or {}).items() if isinstance(raw.get("mcq_answers", {}), dict) else [])},
                true_false_answers={int(k): dict(v or {}) for k, v in ((raw.get("true_false_answers", {}) or {}).items() if isinstance(raw.get("true_false_answers", {}), dict) else [])},
                numeric_answers={int(k): str(v) for k, v in ((raw.get("numeric_answers", {}) or {}).items() if isinstance(raw.get("numeric_answers", {}), dict) else [])},
            )
        return None

    def _expected_questions(exam_code_text: str, res_obj: OMRResult | None = None) -> dict[str, list[int]]:
        key_obj = _answer_key_for_exam(exam_code_text)
        if not key_obj:
            fetched = self._fetch_answer_keys_for_subject_scoped(subject_key) or {}
            for _code, raw_key in fetched.items():
                if isinstance(raw_key, SubjectKey):
                    key_obj = raw_key
                    break
                if isinstance(raw_key, dict):
                    key_obj = SubjectKey(
                        subject=subject_key,
                        exam_code=str(_code or "").strip(),
                        answers={int(k): str(v) for k, v in ((raw_key.get("mcq_answers", {}) or {}).items() if isinstance(raw_key.get("mcq_answers", {}), dict) else [])},
                        true_false_answers={int(k): dict(v or {}) for k, v in ((raw_key.get("true_false_answers", {}) or {}).items() if isinstance(raw_key.get("true_false_answers", {}), dict) else [])},
                        numeric_answers={int(k): str(v) for k, v in ((raw_key.get("numeric_answers", {}) or {}).items() if isinstance(raw_key.get("numeric_answers", {}), dict) else [])},
                    )
                    break
        configured_counts = self._subject_section_question_counts(subject_key)
        if key_obj:
            mcq_all = sorted(int(q) for q in (key_obj.answers or {}).keys() if str(q).strip().lstrip("-").isdigit())
            tf_all = sorted(int(q) for q in (key_obj.true_false_answers or {}).keys() if str(q).strip().lstrip("-").isdigit())
            num_all = sorted(int(q) for q in (key_obj.numeric_answers or {}).keys() if str(q).strip().lstrip("-").isdigit())
            mcq_cnt = max(0, int(configured_counts.get("MCQ", 0) or 0))
            tf_cnt = max(0, int(configured_counts.get("TF", 0) or 0))
            num_cnt = max(0, int(configured_counts.get("NUMERIC", 0) or 0))
            return {
                "MCQ": mcq_all[:mcq_cnt] if mcq_cnt > 0 else mcq_all,
                "TF": tf_all[:tf_cnt] if tf_cnt > 0 else tf_all,
                "NUMERIC": num_all[:num_cnt] if num_cnt > 0 else num_all,
            }
        if isinstance(res_obj, OMRResult):
            mcq_qs = sorted(int(q) for q in ((getattr(res_obj, "mcq_answers", {}) or {}).keys()) if str(q).strip().lstrip("-").isdigit())
            tf_qs = sorted(int(q) for q in ((getattr(res_obj, "true_false_answers", {}) or {}).keys()) if str(q).strip().lstrip("-").isdigit())
            num_qs = sorted(int(q) for q in ((getattr(res_obj, "numeric_answers", {}) or {}).keys()) if str(q).strip().lstrip("-").isdigit())
            return {"MCQ": mcq_qs, "TF": tf_qs, "NUMERIC": num_qs}
        return {"MCQ": [], "TF": [], "NUMERIC": []}

    def _build_answer_table(res_obj: OMRResult, exam_code_text: str) -> None:
        answer_tbl.setRowCount(0)
        row_map: list[tuple[str, int]] = []
        expected = _expected_questions(exam_code_text, res_obj)
        key_obj = _answer_key_for_exam(exam_code_text)

        def _add_row(section: str, q_no: int, correct: str, student: str) -> None:
            r = answer_tbl.rowCount()
            answer_tbl.insertRow(r)
            sec_item = QTableWidgetItem(section)
            q_item = QTableWidgetItem(str(q_no))
            q_item.setData(Qt.UserRole, int(q_no))
            answer_tbl.setItem(r, 0, sec_item)
            answer_tbl.setItem(r, 1, q_item)
            answer_tbl.setItem(r, 2, QTableWidgetItem(correct))
            student_item = QTableWidgetItem(student)
            if str(correct or "").strip().upper() != str(student or "").strip().upper():
                student_item.setBackground(QColor(255, 225, 225))
            answer_tbl.setItem(r, 3, student_item)
            row_map.append((section, int(q_no)))

        for q_no in expected.get("MCQ", []):
            correct = str((getattr(key_obj, "answers", {}) or {}).get(int(q_no), "") or "").strip().upper()
            student = str((getattr(res_obj, "mcq_answers", {}) or {}).get(int(q_no), "") or "").strip().upper()
            _add_row("MCQ", int(q_no), correct, student)
        for q_no in expected.get("TF", []):
            correct = self.scoring_engine._tf_to_canonical_string((getattr(key_obj, "true_false_answers", {}) or {}).get(int(q_no), {}) if key_obj else {})
            student = self.scoring_engine._tf_to_canonical_string((getattr(res_obj, "true_false_answers", {}) or {}).get(int(q_no), {}))
            _add_row("TF", int(q_no), correct, student)
        for q_no in expected.get("NUMERIC", []):
            correct = str((getattr(key_obj, "numeric_answers", {}) or {}).get(int(q_no), "") or "").strip()
            student = str((getattr(res_obj, "numeric_answers", {}) or {}).get(int(q_no), "") or "").strip()
            _add_row("NUMERIC", int(q_no), correct, student)
        editor_refs["row_map"] = row_map

    def _on_answer_table_changed(item: QTableWidgetItem) -> None:
        if item is None or item.column() != 3:
            return
        row_idx = item.row()
        correct_txt = str(answer_tbl.item(row_idx, 2).text() if answer_tbl.item(row_idx, 2) else "").strip().upper()
        student_txt = str(item.text() or "").strip().upper()
        if student_txt != str(item.text() or ""):
            answer_tbl.blockSignals(True)
            item.setText(student_txt)
            answer_tbl.blockSignals(False)
        if correct_txt != student_txt:
            item.setBackground(QColor(255, 225, 225))
        else:
            item.setBackground(QColor(255, 255, 255, 0))

    def _subject_room_for_sid(sid: str) -> str:
        cfg = self._subject_config_by_subject_key(subject_key) or {}
        return self._subject_room_for_student_id(sid, cfg)

    subject_score_map = self.scoring_results_by_subject.get(subject_key, {}) or {}

    def _score_display_for_sid(sid: str, res_obj: OMRResult | None) -> str:
        row_payload = (subject_score_map.get(str(sid or "").strip(), {}) or {}) if isinstance(subject_score_map, dict) else {}
        recheck_val = row_payload.get("recheck_score", "") if isinstance(row_payload, dict) else ""
        if recheck_val not in {"", None}:
            try:
                return f"{float(recheck_val):g}"
            except Exception:
                return str(recheck_val)
        if isinstance(res_obj, OMRResult):
            return f"{_current_score_for_result(res_obj):g}"
        return "Không tìm thấy bài thi"

    def _recheck_content_for_sid(sid: str, res_obj: OMRResult | None) -> str:
        sid_key = str(sid or "").strip()
        row_payload = (subject_score_map.get(sid_key, {}) or {}) if isinstance(subject_score_map, dict) else {}
        if isinstance(row_payload, dict):
            full = str(row_payload.get("baithiphuctra", "") or "").strip()
            if full:
                return full
            base = str(row_payload.get("bailam", "") or "").strip()
            if base:
                return base
        if isinstance(res_obj, OMRResult):
            return (
                f"MCQ:{self._format_mcq_answers(getattr(res_obj, 'mcq_answers', {}) or {})} | "
                f"TF:{self._format_tf_answers(getattr(res_obj, 'true_false_answers', {}) or {})} | "
                f"NUM:{self._format_numeric_answers(getattr(res_obj, 'numeric_answers', {}) or {})}"
            )
        return "-"

    def _render_selected_table() -> None:
        _rebuild_recheck_entries()
        tbl.setRowCount(0)
        for idx, entry in enumerate(recheck_entries, start=1):
            res = entry.get("result")
            sid = str(getattr(res, "student_id", "") or "").strip() if isinstance(res, OMRResult) else str(entry.get("requested_sid", "") or "").strip()
            prof = self._student_profile_by_id(sid)
            score_text = _score_display_for_sid(sid, res if isinstance(res, OMRResult) else None)
            row = tbl.rowCount()
            tbl.insertRow(row)
            tbl.setItem(row, 0, QTableWidgetItem(str(idx)))
            tbl.setItem(row, 1, QTableWidgetItem(sid or "-"))
            tbl.setItem(row, 2, QTableWidgetItem(str(prof.get("name", "") or "-")))
            tbl.setItem(row, 3, QTableWidgetItem(str(prof.get("class_name", "") or "-")))
            tbl.setItem(row, 4, QTableWidgetItem(_subject_room_for_sid(sid) or "-"))
            tbl.setItem(row, 5, QTableWidgetItem(str(getattr(res, "exam_code", "") or "-") if isinstance(res, OMRResult) else "-"))
            tbl.setItem(row, 6, QTableWidgetItem(score_text))
            if any(str(x.get("student_code", "") or "").strip() == sid for x in history_all):
                for c in range(tbl.columnCount()):
                    item = tbl.item(row, c)
                    if item is not None:
                        item.setBackground(Qt.yellow)

    _render_selected_table()

    updating_form = {"busy": False}

    def _refresh_history_for_sid(sid: str) -> None:
        rows_local = [x for x in history_all if str(x.get("student_code", "") or "").strip() == str(sid or "").strip()]
        if not rows_local:
            history_txt.setPlainText("Chưa có lịch sử phúc tra.")
            return
        lines = [
            f"[{str(item.get('created_at', '') or '-')}] {str(item.get('change_text', '') or '-')}"
            for item in rows_local
        ]
        history_txt.setPlainText("\n".join(lines))

    def _on_pick() -> None:
        r = tbl.currentRow()
        if r < 0 or r >= len(recheck_entries):
            return
        updating_form["busy"] = True
        res = recheck_entries[r].get("result")
        if not isinstance(res, OMRResult):
            sid_text = str(recheck_entries[r].get("requested_sid", "") or "").strip()
            inp_sid.setEditText(sid_text)
            inp_exam.setEditText("")
            answer_tbl.setRowCount(0)
            lbl_score.setText("Không tìm thấy bài thi")
            lbl_recheck_info.setText(f"SBD: {sid_text or '-'} | Mã đề: - | Điểm: -\nNội dung: Không tìm thấy bài thi")
            _refresh_history_for_sid(sid_text)
            img_lbl.setText("Không tìm thấy bài thi")
            img_lbl.setPixmap(QPixmap())
            updating_form["busy"] = False
            return
        sid = str(getattr(res, "student_id", "") or "").strip()
        inp_sid.setEditText(sid_to_display.get(sid, sid))
        inp_exam.setEditText(str(getattr(res, "exam_code", "") or ""))
        _build_answer_table(res, str(getattr(res, "exam_code", "") or ""))
        score_here = _score_display_for_sid(sid, res)
        lbl_score.setText(str(score_here))
        detail_content = _recheck_content_for_sid(sid, res)
        lbl_recheck_info.setText(
            f"SBD: {sid or '-'} | Mã đề: {str(getattr(res, 'exam_code', '') or '-')} | Điểm: {score_here}\n"
            f"Nội dung: {detail_content or '-'}"
        )
        prof = self._student_profile_by_id(sid)
        if tbl.item(r, 2):
            tbl.item(r, 2).setText(str(prof.get("name", "") or "-"))
        if tbl.item(r, 4):
            tbl.item(r, 4).setText(_subject_room_for_sid(sid) or "-")
        _refresh_history_for_sid(sid)
        img_path = str(getattr(res, "image_path", "") or "")
        pix = QPixmap(img_path)
        if pix.isNull():
            img_lbl.setText(f"Không đọc được ảnh: {Path(img_path).name}")
            img_lbl.setPixmap(QPixmap())
        else:
            img_lbl.setText("")
            img_lbl.setPixmap(pix.scaledToWidth(max(200, int(img_scroll.viewport().width() * 0.9)), Qt.SmoothTransformation))
        updating_form["busy"] = False

    def _selected_sid_value() -> str:
        text = str(inp_sid.currentText() or "").strip()
        if " - " in text:
            return text.split(" - ", 1)[0].strip()
        return text

    def _save_current() -> None:
        nonlocal subject_score_map
        r = tbl.currentRow()
        if r < 0 or r >= len(recheck_entries):
            return
        if updating_form["busy"]:
            return
        res = recheck_entries[r].get("result")
        if not isinstance(res, OMRResult):
            QMessageBox.information(dlg, "Phúc tra", "SBD này không có bài thi tham chiếu để lưu chỉnh sửa.")
            return
        old_sid = str(getattr(res, "student_id", "") or "").strip()
        old_exam = str(getattr(res, "exam_code", "") or "").strip()
        old_score = _current_score_for_result(res)
        new_sid = _selected_sid_value().strip()
        new_exam = str(inp_exam.currentText() or "").strip()
        new_mcq: dict[int, str] = {}
        new_tf: dict[int, dict[str, bool]] = {}
        new_numeric: dict[int, str] = {}
        for rr in range(answer_tbl.rowCount()):
            sec = str(answer_tbl.item(rr, 0).text() if answer_tbl.item(rr, 0) else "").strip().upper()
            q_item = answer_tbl.item(rr, 1)
            q_no = int(q_item.data(Qt.UserRole) if q_item and q_item.data(Qt.UserRole) is not None else 0)
            student_txt = str(answer_tbl.item(rr, 3).text() if answer_tbl.item(rr, 3) else "").strip()
            if q_no <= 0:
                continue
            if sec == "MCQ":
                new_mcq[q_no] = student_txt.upper()[:1]
            elif sec == "TF":
                parsed = _parse_tf_display(f"{q_no}:{student_txt.upper()[:4]}")
                if q_no in parsed:
                    new_tf[q_no] = parsed[q_no]
            elif sec == "NUMERIC":
                new_numeric[q_no] = student_txt
        if new_sid:
            res.student_id = new_sid
        res.exam_code = new_exam
        res.mcq_answers = {int(k): str(v or "").strip().upper()[:1] for k, v in (new_mcq or {}).items()}
        res.true_false_answers = {int(k): dict(v or {}) for k, v in (new_tf or {}).items()}
        res.numeric_answers = {int(k): str(v or "").strip() for k, v in (new_numeric or {}).items()}
        res.answer_string = ""
        self._persist_single_scan_result_to_db(res, note="recheck_edit")
        new_score = _current_score_for_result(res)
        changes: list[str] = []
        if old_sid != str(getattr(res, "student_id", "") or "").strip():
            changes.append(f"Lỗi SBD -> sửa từ {old_sid or '-'} thành {str(getattr(res, 'student_id', '') or '-')}")
        if old_exam != str(getattr(res, "exam_code", "") or "").strip():
            changes.append(f"Lỗi mã đề -> sửa từ {old_exam or '-'} thành {str(getattr(res, 'exam_code', '') or '-')}")
        if abs(new_score - old_score) > 1e-9:
            changes.append(f"Lỗi nhận dạng bài thi ->thay đổi điểm từ {old_score:g} lên {new_score:g}")
        if not changes:
            changes.append("Cập nhật dữ liệu phúc tra.")
        for message in changes:
            payload = {"image_path": str(getattr(res, "image_path", "") or ""), "subject_key": subject_key}
            self.database.add_recheck_history(
                session_id=str(self.current_session_id or ""),
                exam_name=str(getattr(self.session, "exam_name", "") or ""),
                subject_key=subject_key,
                student_code=str(getattr(res, "student_id", "") or ""),
                exam_code=str(getattr(res, "exam_code", "") or ""),
                change_text=message,
                old_score=float(old_score),
                new_score=float(new_score),
                payload=payload,
            )
        history_all[:] = self.database.fetch_recheck_history(str(self.current_session_id or ""), subject_key=subject_key)
        subject_scores_map = self.scoring_results_by_subject.get(subject_key, {}) or {}
        sid_key_new = str(getattr(res, "student_id", "") or "").strip()
        if sid_key_new:
            row_payload = dict(subject_scores_map.get(sid_key_new, {}) or {})
            row_payload["recheck_score"] = float(new_score)
            row_payload["score"] = row_payload.get("score", float(new_score))
            row_payload["final_score"] = float(new_score)
            row_payload["baithiphuctra"] = (
                f"MCQ:{self._format_mcq_answers(getattr(res, 'mcq_answers', {}) or {})} | "
                f"TF:{self._format_tf_answers(getattr(res, 'true_false_answers', {}) or {})} | "
                f"NUM:{self._format_numeric_answers(getattr(res, 'numeric_answers', {}) or {})}"
            )
            subject_scores_map[sid_key_new] = row_payload
            self.scoring_results_by_subject[subject_key] = subject_scores_map
            subject_score_map = self.scoring_results_by_subject.get(subject_key, {}) or {}
        self.calculate_scores(subject_key=subject_key, mode="Tính lại toàn bộ", note="recheck_edit")
        sid = str(getattr(res, "student_id", "") or "").strip()
        prof = self._student_profile_by_id(sid)
        tbl.setItem(r, 1, QTableWidgetItem(sid or "-"))
        tbl.setItem(r, 2, QTableWidgetItem(str(prof.get("name", "") or "-")))
        tbl.setItem(r, 3, QTableWidgetItem(str(prof.get("class_name", "") or "-")))
        tbl.setItem(r, 4, QTableWidgetItem(_subject_room_for_sid(sid) or "-"))
        tbl.setItem(r, 5, QTableWidgetItem(str(getattr(res, "exam_code", "") or "-")))
        final_score_text = _score_display_for_sid(sid, res)
        tbl.setItem(r, 6, QTableWidgetItem(str(final_score_text)))
        lbl_score.setText(str(final_score_text))
        lbl_recheck_info.setText(
            f"SBD: {sid or '-'} | Mã đề: {str(getattr(res, 'exam_code', '') or '-')} | Điểm: {final_score_text}\n"
            f"Nội dung: {_recheck_content_for_sid(sid, res) or '-'}"
        )
        _refresh_history_for_sid(sid)
        QMessageBox.information(dlg, "Phúc tra", "Đã lưu chỉnh sửa, ghi lịch sử và tính lại điểm.")

    def _rebuild_list_and_refresh() -> None:
        _persist_recheck_sid_list(requested_sids)
        _render_selected_table()
        if tbl.rowCount() > 0:
            tbl.setCurrentCell(0, 0)
            _on_pick()
        else:
            history_txt.setPlainText("Chưa có lịch sử phúc tra.")
            lbl_score.setText("-")
            lbl_recheck_info.setText("-")

    def _open_list_builder_again() -> None:
        nonlocal requested_sids
        rebuilt = _open_recheck_list_builder(requested_sids)
        if rebuilt is None:
            return
        requested_sids = _normalize_sid_list(rebuilt)
        _rebuild_list_and_refresh()

    def _add_sid_list() -> None:
        fresh = _load_sid_list_from_file()
        if not fresh:
            return
        decision = QMessageBox.question(
            dlg,
            "Thêm danh sách",
            "Bạn muốn mở rộng danh sách hiện tại?\nYes = Mở rộng, No = Ghi đè, Cancel = Huỷ.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if decision == QMessageBox.Cancel:
            return
        current = list(requested_sids)
        merged = current + fresh if decision == QMessageBox.Yes else fresh
        requested_sids[:] = _normalize_sid_list(merged)
        _rebuild_list_and_refresh()
        QMessageBox.information(dlg, "Thêm danh sách", "Đã cập nhật danh sách phúc tra.")

    def _export_recheck_excel() -> None:
        path, _ = QFileDialog.getSaveFileName(dlg, "Xuất Excel phúc tra", "", "Excel (*.xlsx)")
        if not path:
            return
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "recheck"
        ws.append(["SBD", "Họ tên", "Ngày sinh", "Lớp", "Điểm cuối cùng", "Lịch sử phúc tra"])
        for i, entry in enumerate(recheck_entries):
            res = entry.get("result")
            sid = str(getattr(res, "student_id", "") or "").strip() if isinstance(res, OMRResult) else str(entry.get("requested_sid", "") or "").strip()
            prof = self._student_profile_by_id(sid)
            score_value = tbl.item(i, 6).text() if tbl.item(i, 6) else "-"
            h_items = [x for x in history_all if str(x.get("student_code", "") or "").strip() == sid]
            h_text = "\n".join(f"[{str(x.get('created_at', '') or '-')}] {str(x.get('change_text', '') or '-')}" for x in h_items) or "-"
            ws.append([
                sid or "-",
                str(prof.get("name", "") or "-"),
                str(prof.get("birth_date", "") or "-"),
                str(prof.get("class_name", "") or "-"),
                str(score_value or "-"),
                h_text,
            ])
        for col in ws.columns:
            width = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(80, max(12, width + 2))
        wb.save(Path(path))
        QMessageBox.information(dlg, "Phúc tra", f"Đã xuất Excel:\n{path}")

    tbl.itemSelectionChanged.connect(_on_pick)
    answer_tbl.itemChanged.connect(_on_answer_table_changed)
    btn_save.clicked.connect(_save_current)
    btn_build_list.clicked.connect(_open_list_builder_again)
    btn_add_list.clicked.connect(_add_sid_list)
    btn_pick.clicked.connect(_pick_from_pool)
    btn_unpick.clicked.connect(_remove_selected_recheck_row)
    btn_export.clicked.connect(_export_recheck_excel)
    # Backward-safe optional hooks: in case older patched layouts still define these controls,
    # connect only when both widget and handler exist to avoid NameError at runtime.
    optional_pick_btn = locals().get("btn_pick")
    optional_pick_handler = locals().get("_pick_from_pool")
    if isinstance(optional_pick_btn, QPushButton) and callable(optional_pick_handler):
        optional_pick_btn.clicked.connect(optional_pick_handler)
    optional_unpick_btn = locals().get("btn_unpick")
    optional_unpick_handler = locals().get("_remove_selected_recheck_row")
    if isinstance(optional_unpick_btn, QPushButton) and callable(optional_unpick_handler):
        optional_unpick_btn.clicked.connect(optional_unpick_handler)
    if tbl.rowCount() > 0:
        tbl.setCurrentCell(0, 0)
        _on_pick()
    loading.setValue(4)
    loading.close()
    dlg.exec()
    return


__all__ = ["open_recheck_dialog"]
