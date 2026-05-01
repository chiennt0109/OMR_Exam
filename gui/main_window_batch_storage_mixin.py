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


class MainWindowBatchStorageMixin:
    """Scan result serialization, DB persistence, cached snapshots and edit registry."""
    def _collect_current_subject_results_for_save(self, subject_key: str) -> list[OMRResult]:
        # scan_list columns: 0 stt, 1 sid, 2 room, 3 exam_code, 4 full_name, 5 birth_date, 6 content, 7 status, 8 actions
        key = str(subject_key or "").strip()
        base_results = list(self.scan_results_by_subject.get(self._batch_result_subject_key(key)) or self.scan_results or [])
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            return base_results

        by_image: dict[str, OMRResult] = {}
        for res in base_results:
            img = str(getattr(res, "image_path", "") or "").strip()
            if img and img not in by_image:
                by_image[img] = res

        out: list[OMRResult] = []
        for r in range(row_count):
            sid_item = self.scan_list.item(r, self.SCAN_COL_STUDENT_ID)
            image_path = str(sid_item.data(Qt.UserRole) if sid_item else "")
            exam_code = str(sid_item.data(Qt.UserRole + 1) if sid_item else "")
            sid_text = str(sid_item.text() if sid_item else "-")
            base = by_image.get(image_path) if image_path else None
            if base is not None:
                res = self._lightweight_result_copy(base)
            else:
                res = OMRResult(image_path=image_path)
            res.student_id = "" if sid_text == "-" else sid_text
            res.exam_code = exam_code
            if hasattr(self, "scan_list"):
                res.full_name = str(self.scan_list.item(r, self.SCAN_COL_FULL_NAME).text() if self.scan_list.item(r, self.SCAN_COL_FULL_NAME) else "")
                res.birth_date = str(self.scan_list.item(r, self.SCAN_COL_BIRTH_DATE).text() if self.scan_list.item(r, self.SCAN_COL_BIRTH_DATE) else "")
            res.sync_legacy_aliases()
            out.append(res)

        return out

    def _refresh_scan_results_from_db(self, subject_key: str) -> list[OMRResult]:
        subject = str(subject_key or "").strip()
        if not subject:
            return []

        candidates = self._subject_scan_storage_key_candidates(subject)
        scoped_subject = self._batch_result_subject_key(subject)
        rows = []
        matched_subject = scoped_subject
        for candidate_key in candidates:
            try:
                rows = self.database.fetch_scan_results_for_subject(candidate_key) or []
            except Exception:
                rows = []
            if rows:
                matched_subject = candidate_key
                break

        refreshed: list[OMRResult] = []
        for item in rows:
            try:
                result = self._deserialize_omr_result(item)
                if result is None:
                    continue
                self._apply_persisted_result_edit_metadata(subject, result)
                refreshed.append(result)
            except Exception:
                continue
        self.scan_results_by_subject[scoped_subject] = list(refreshed)
        if matched_subject != scoped_subject:
            self.scan_results_by_subject[matched_subject] = list(refreshed)
        return refreshed

    @staticmethod
    def _serialize_omr_result(result: OMRResult) -> dict:
        return {
            "image_path": str(getattr(result, "image_path", "") or ""),
            "student_id": str(getattr(result, "student_id", "") or ""),
            "exam_code": str(getattr(result, "exam_code", "") or ""),
            "mcq_answers": {int(k): str(v) for k, v in (getattr(result, "mcq_answers", {}) or {}).items()},
            "true_false_answers": {int(k): dict(v) for k, v in (getattr(result, "true_false_answers", {}) or {}).items()},
            "numeric_answers": {int(k): str(v) for k, v in (getattr(result, "numeric_answers", {}) or {}).items()},
            "confidence_scores": {str(k): float(v) for k, v in (getattr(result, "confidence_scores", {}) or {}).items()},
            "recognition_errors": [str(x) for x in (getattr(result, "recognition_errors", []) or [])],
            "processing_time_sec": float(getattr(result, "processing_time_sec", 0.0) or 0.0),
            "debug_image_path": str(getattr(result, "debug_image_path", "") or ""),
            "full_name": str(getattr(result, "full_name", "") or ""),
            "birth_date": str(getattr(result, "birth_date", "") or ""),
            "exam_room": str(getattr(result, "exam_room", "") or ""),
            "class_name": str(getattr(result, "class_name", "") or ""),
            "cached_status": str(getattr(result, "cached_status", "") or ""),
            "cached_content": str(getattr(result, "cached_content", "") or ""),
            "cached_recognized_short": str(getattr(result, "cached_recognized_short", "") or ""),
            "manual_content_override": str(getattr(result, "manual_content_override", "") or ""),
            "cached_forced_status": str(getattr(result, "cached_forced_status", "") or ""),
            "manually_edited": bool(getattr(result, "manually_edited", False)),
            "edit_history": [str(x) for x in (getattr(result, "edit_history", []) or []) if str(x or "").strip()],
            "last_adjustment": str(getattr(result, "last_adjustment", "") or ""),
            "manual_adjustments": [str(x) for x in (getattr(result, "manual_adjustments", []) or []) if str(x or "").strip()],
            "cached_blank_summary": dict(getattr(result, "cached_blank_summary", {}) or {}),
            "recognized_template_path": str(getattr(result, "recognized_template_path", "") or ""),
            "recognized_alignment_profile": str(getattr(result, "recognized_alignment_profile", "") or ""),
            "recognized_fill_threshold": float(getattr(result, "recognized_fill_threshold", 0.45) or 0.45),
            "recognized_empty_threshold": float(getattr(result, "recognized_empty_threshold", 0.20) or 0.20),
            "recognized_certainty_margin": float(getattr(result, "recognized_certainty_margin", 0.08) or 0.08),
            "answer_string_api_mode": bool(getattr(result, "answer_string_api_mode", False)),
            "answer_string": str(getattr(result, "answer_string", "") or ""),
        }

    @staticmethod
    def _deserialize_omr_result(payload: dict) -> OMRResult:
        result = OMRResult(
            image_path=str(payload.get("image_path", "") or ""),
            student_id=str(payload.get("student_id", "") or ""),
            exam_code=str(payload.get("exam_code", "") or ""),
            mcq_answers={int(k): str(v) for k, v in (payload.get("mcq_answers", {}) or {}).items()},
            true_false_answers={int(k): dict(v) for k, v in (payload.get("true_false_answers", {}) or {}).items()},
            numeric_answers={int(k): str(v) for k, v in (payload.get("numeric_answers", {}) or {}).items()},
            confidence_scores={str(k): float(v) for k, v in (payload.get("confidence_scores", {}) or {}).items()},
            recognition_errors=[str(x) for x in (payload.get("recognition_errors", []) or [])],
            processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
            debug_image_path=str(payload.get("debug_image_path", "") or ""),
        )
        setattr(result, "full_name", str(payload.get("full_name", "") or ""))
        setattr(result, "birth_date", str(payload.get("birth_date", "") or ""))
        setattr(result, "exam_room", str(payload.get("exam_room", "") or ""))
        setattr(result, "class_name", str(payload.get("class_name", "") or ""))
        setattr(result, "cached_status", str(payload.get("cached_status", "") or ""))
        setattr(result, "cached_content", str(payload.get("cached_content", "") or ""))
        setattr(result, "cached_recognized_short", str(payload.get("cached_recognized_short", "") or ""))
        setattr(result, "manual_content_override", str(payload.get("manual_content_override", "") or ""))
        setattr(result, "cached_forced_status", str(payload.get("cached_forced_status", "") or ""))
        setattr(result, "manually_edited", bool(payload.get("manually_edited", False)))
        setattr(result, "edit_history", [str(x) for x in (payload.get("edit_history", []) or []) if str(x or "").strip()])
        setattr(result, "last_adjustment", str(payload.get("last_adjustment", "") or ""))
        setattr(result, "manual_adjustments", [str(x) for x in (payload.get("manual_adjustments", []) or []) if str(x or "").strip()])
        if (not bool(getattr(result, "manually_edited", False))) and bool(getattr(result, "edit_history", [])):
            setattr(result, "manually_edited", True)
            if not str(getattr(result, "cached_forced_status", "") or "").strip():
                setattr(result, "cached_forced_status", "Đã sửa")
        setattr(result, "cached_blank_summary", dict(payload.get("cached_blank_summary", {}) or {}))
        setattr(result, "recognized_template_path", str(payload.get("recognized_template_path", "") or ""))
        setattr(result, "recognized_alignment_profile", str(payload.get("recognized_alignment_profile", "") or ""))
        setattr(result, "recognized_fill_threshold", float(payload.get("recognized_fill_threshold", 0.45) or 0.45))
        setattr(result, "recognized_empty_threshold", float(payload.get("recognized_empty_threshold", 0.20) or 0.20))
        setattr(result, "recognized_certainty_margin", float(payload.get("recognized_certainty_margin", 0.08) or 0.08))
        setattr(result, "answer_string_api_mode", bool(payload.get("answer_string_api_mode", False)))
        setattr(result, "answer_string", str(payload.get("answer_string", "") or ""))
        result.sync_legacy_aliases()
        return result

    def _subject_scan_storage_key_candidates(
        self,
        subject_or_cfg: str | dict | None,
        *,
        session_id: str | None = None,
        session: ExamSession | None = None,
        include_generated: bool = True,
    ) -> list[str]:
        bases: list[str] = []

        def add_base(value: object) -> None:
            key = str(value or "").strip()
            if key and key not in bases:
                bases.append(key)

        if isinstance(subject_or_cfg, dict):
            cfg = subject_or_cfg
            # Rendering the status column must be read-only. Do not create a new
            # subject_instance_key here; key generation belongs to save/scan flows.
            if include_generated:
                try:
                    add_base(self._subject_key_from_cfg(cfg))
                except Exception:
                    pass
            for field_name in (
                "subject_instance_key",
                "logical_subject_key",
                "answer_key_key",
                "subject_key",
                "subject_uid",
            ):
                add_base(cfg.get(field_name, ""))
            try:
                add_base(self._logical_subject_key_from_cfg(cfg))
            except Exception:
                pass
            name = str(cfg.get("name", "") or "").strip()
            block = str(cfg.get("block", "") or "").strip()
            if name and block:
                add_base(f"{name}_{block}")
            elif name:
                add_base(name)
            legacy_keys = cfg.get("legacy_subject_instance_keys", [])
            if isinstance(legacy_keys, list):
                for legacy_key in legacy_keys:
                    add_base(legacy_key)
        else:
            add_base(subject_or_cfg)

        scope_prefixes: list[str] = []

        def add_scope(value: object) -> None:
            key = str(value or "").strip()
            if key and key not in scope_prefixes:
                scope_prefixes.append(key)

        add_scope(self._session_scope_prefix_for(session_id=session_id, session=session))
        add_scope(self._session_scope_prefix())
        add_scope(self.current_session_id if session_id is None else session_id)
        add_scope(self.current_session_id)

        candidates: list[str] = []

        def add_candidate(value: object) -> None:
            key = str(value or "").strip()
            if key and key not in candidates:
                candidates.append(key)

        for base in bases:
            add_candidate(base)
            if "::" in base:
                parts = [p for p in base.split("::") if p]
                if parts:
                    add_candidate(parts[-1])
                if len(parts) >= 2:
                    add_candidate("::".join(parts[-2:]))
            for scope_prefix in scope_prefixes:
                if not scope_prefix:
                    continue
                if base.startswith(f"{scope_prefix}::"):
                    # Compatibility with older builds that accidentally saved a doubly scoped subject key.
                    add_candidate(f"{scope_prefix}::{base}")
                else:
                    add_candidate(f"{scope_prefix}::{base}")
        return candidates

    def _scan_result_counts_for_subject_keys(self, subject_keys: list[str]) -> dict[str, int]:
        """Return scan-result counts without deserializing all OMR rows.

        The subject grid only needs existence/count information. Loading full scan rows
        for every subject is the main source of redundant work when an exam already has
        hundreds of saved sheets.
        """
        keys = [str(k or "").strip() for k in subject_keys or [] if str(k or "").strip()]
        if not keys:
            return {}
        unique_keys = list(dict.fromkeys(keys))
        db = getattr(self, "database", None)
        conn = getattr(db, "conn", None)
        if conn is None:
            conn = getattr(db, "connection", None)
        counts: dict[str, int] = {}
        if conn is not None:
            for pos in range(0, len(unique_keys), 800):
                chunk = unique_keys[pos:pos + 800]
                placeholders = ",".join("?" for _ in chunk)
                try:
                    rows = conn.execute(
                        f"SELECT subject_key, COUNT(*) FROM scan_results WHERE subject_key IN ({placeholders}) GROUP BY subject_key",
                        chunk,
                    ).fetchall()
                    for row in rows:
                        try:
                            key = str(row["subject_key"])
                            count = int(row[1])
                        except Exception:
                            key = str(row[0])
                            count = int(row[1])
                        counts[key] = count
                except Exception:
                    counts = {}
                    break
        if counts:
            return counts

        # Fallback for non-SQL database adapters. This path preserves compatibility,
        # but is only used when the direct count query is unavailable.
        for key in unique_keys:
            try:
                rows = db.fetch_scan_results_for_subject(key) or [] if db is not None else []
            except Exception:
                rows = []
            if isinstance(rows, list) and rows:
                counts[key] = len(rows)
        return counts

    def _subject_saved_data_state_map(
        self,
        cfgs: list[dict] | tuple[dict, ...] | None,
        *,
        session_id: str | None = None,
        session: ExamSession | None = None,
    ) -> dict[int, tuple[bool, int]]:
        state: dict[int, tuple[bool, int]] = {}
        cfg_list = list(cfgs or [])
        if not cfg_list:
            return state

        candidates_by_idx: dict[int, list[str]] = {}
        all_candidates: list[str] = []
        for idx, cfg in enumerate(cfg_list):
            if not isinstance(cfg, dict):
                state[idx] = (False, 0)
                continue
            if self._subject_uses_direct_score_import(cfg):
                rows = self._direct_score_import_rows_for_subject(cfg)
                state[idx] = (bool(rows), len(rows))
                continue
            candidates = self._subject_scan_storage_key_candidates(
                cfg,
                session_id=session_id,
                session=session,
                include_generated=True,
            )
            candidates_by_idx[idx] = candidates
            all_candidates.extend(candidates)

        unresolved: dict[int, list[str]] = {}
        for idx, candidates in candidates_by_idx.items():
            matched = False
            for key in candidates:
                rows = self.scan_results_by_subject.get(key)
                if isinstance(rows, list) and rows:
                    state[idx] = (True, len(rows))
                    matched = True
                    break
            if not matched:
                unresolved[idx] = candidates

        counts = self._scan_result_counts_for_subject_keys(all_candidates)
        for idx, candidates in unresolved.items():
            matched_count = 0
            for key in candidates:
                matched_count = int(counts.get(key, 0) or 0)
                if matched_count > 0:
                    break
            if matched_count > 0:
                state[idx] = (True, matched_count)
                continue
            cfg = cfg_list[idx]
            try:
                legacy_count = int(cfg.get("batch_result_count") or 0)
            except Exception:
                legacy_count = 0
            if legacy_count > 0 or bool(cfg.get("batch_saved")):
                state[idx] = (True, max(legacy_count, 1))
            else:
                state[idx] = (False, 0)
        return state

    def _subject_display_status_map(
        self,
        cfgs: list[dict] | tuple[dict, ...] | None,
        *,
        session_id: str | None = None,
        session: ExamSession | None = None,
    ) -> dict[int, str]:
        state_map = self._subject_saved_data_state_map(cfgs, session_id=session_id, session=session)
        out: dict[int, str] = {}
        for idx, (has_data, count) in state_map.items():
            out[idx] = f"Đã nhận dạng ({count})" if has_data and count > 0 else ("Đã nhận dạng" if has_data else "-")
        return out

    def _subject_saved_data_state(
        self,
        cfg: dict | None,
        *,
        session_id: str | None = None,
        session: ExamSession | None = None,
    ) -> tuple[bool, int]:
        if not isinstance(cfg, dict):
            return False, 0

        if self._subject_uses_direct_score_import(cfg):
            rows = self._direct_score_import_rows_for_subject(cfg)
            return bool(rows), len(rows)

        candidates = self._subject_scan_storage_key_candidates(
            cfg,
            session_id=session_id,
            session=session,
            include_generated=True,
        )
        for key in candidates:
            rows = self.scan_results_by_subject.get(key)
            if isinstance(rows, list) and rows:
                return True, len(rows)

        for key in candidates:
            try:
                rows = self.database.fetch_scan_results_for_subject(key) or []
            except Exception:
                rows = []
            if isinstance(rows, list) and rows:
                return True, len(rows)

        # Backward-compatible UI fallback. It is used only after DB/cache lookup fails.
        # This prevents a save operation from clearing the display before the next restart
        # when older configs still carry batch_saved/batch_result_count metadata.
        try:
            legacy_count = int(cfg.get("batch_result_count") or 0)
        except Exception:
            legacy_count = 0
        if legacy_count > 0 or bool(cfg.get("batch_saved")):
            return True, max(legacy_count, 1)
        return False, 0

    def _subject_has_recognition_data(self, cfg: dict | None) -> bool:
        return self._subject_saved_data_state(cfg)[0]

    def _delete_subject_recognition_data(self, cfg: dict | None) -> None:
        if not isinstance(cfg, dict):
            return
        for key in self._subject_scan_storage_key_candidates(cfg):
            try:
                self.scan_results_by_subject.pop(key, None)
            except Exception:
                pass
            try:
                self.database.delete_scan_results_for_subject(key)
            except Exception:
                pass

    def _subject_display_status_text(
        self,
        cfg: dict | None,
        *,
        session_id: str | None = None,
        session: ExamSession | None = None,
    ) -> str:
        has_data, count = self._subject_saved_data_state(cfg, session_id=session_id, session=session)
        if has_data and count > 0:
            return f"Đã nhận dạng ({count})"
        return "Đã nhận dạng" if has_data else "-"

    def _is_subject_marked_batched(self, cfg: dict) -> bool:
        return self._subject_has_recognition_data(cfg)

    def _cached_subject_scans_from_config(self, subject_key: str) -> list[OMRResult]:
        # Legacy helper giữ lại để tương thích, nhưng luôn hydrate từ DB.
        return list(self._refresh_scan_results_from_db(subject_key) or [])

    def _resolve_current_session_subject_configs_for_update(self) -> tuple[list[dict], str]:
        # batch save source of truth: prefer runtime session currently opened in app.
        if self.session and isinstance(self.session.config, dict):
            cfgs = self.session.config.get("subject_configs", [])
            if isinstance(cfgs, list):
                return cfgs, "runtime"
        if self.current_session_id:
            payload = self.database.fetch_exam_session(self.current_session_id) or {}
            if isinstance(payload, dict) and payload:
                try:
                    db_session = ExamSession.from_dict(payload)
                    if db_session and isinstance(db_session.config, dict):
                        cfgs = db_session.config.get("subject_configs", [])
                        if isinstance(cfgs, list):
                            self.session = db_session
                            self.current_session_path = self._session_path_from_id(self.current_session_id)
                            return cfgs, "database"
                except Exception:
                    pass
        session_path = self.current_session_path
        if not session_path and self.current_session_id:
            candidate = self._session_path_from_id(self.current_session_id)
            if candidate.exists():
                session_path = candidate
        if session_path and session_path.exists():
            try:
                fallback = ExamSession.load_json(session_path)
                if isinstance(fallback.config, dict):
                    cfgs = fallback.config.get("subject_configs", [])
                    if isinstance(cfgs, list):
                        if not self.session:
                            self.session = fallback
                        return cfgs, "file_fallback"
            except Exception:
                pass
        return [], "empty"

    def _find_subject_config_index_for_batch_save(self, subject_cfg: dict, subject_cfgs: list[dict]) -> int:
        # canonical subject match: subject_instance_key > subject_uid > runtime_key > legacy logical (unambiguous).
        selected_instance = str(self._subject_instance_key_from_cfg(subject_cfg) or "").strip()
        selected_uid = str(subject_cfg.get("subject_uid", "") or "").strip()
        selected_runtime = str(self._batch_runtime_key(selected_instance) or "").strip()
        selected_logical = str(self._logical_subject_key_from_cfg(subject_cfg) or "").strip().lower()
        if selected_instance:
            for idx, item in enumerate(subject_cfgs):
                if str(self._subject_instance_key_from_cfg(item) or "").strip() == selected_instance:
                    return idx
        if selected_uid:
            for idx, item in enumerate(subject_cfgs):
                if str(item.get("subject_uid", "") or "").strip() == selected_uid:
                    return idx
        if selected_runtime:
            for idx, item in enumerate(subject_cfgs):
                item_instance = str(self._subject_instance_key_from_cfg(item) or "").strip()
                if item_instance and str(self._batch_runtime_key(item_instance) or "").strip() == selected_runtime:
                    return idx
        if selected_logical:
            matches = [
                idx for idx, item in enumerate(subject_cfgs)
                if str(self._logical_subject_key_from_cfg(item) or "").strip().lower() == selected_logical
            ]
            if len(matches) == 1:
                return matches[0]
        return -1

    def _persist_current_session_subject_configs(self, subject_cfgs: list[dict] | None = None) -> bool:
        if not self.current_session_id:
            return False
        if self.session is None:
            self.session = ExamSession(exam_name="Kỳ thi", exam_date=str(date.today()))
        if subject_cfgs is None:
            subject_cfgs = list((self.session.config or {}).get("subject_configs", []) or [])
        self.session.config = {**(self.session.config or {}), "subject_configs": list(subject_cfgs)}
        try:
            payload = self._build_session_persistence_payload()
            self.session.config = dict(payload.get("config", {}) or {})
            self.database.save_exam_session(self.current_session_id, self.session.exam_name, payload)
            self.current_session_path = self._session_path_from_id(self.current_session_id)
            self.session_dirty = False
            self._remember_current_session_snapshot()
        except Exception:
            self.session_dirty = True
            return False
        if self.current_session_path:
            try:
                self.session.save_json(self.current_session_path)
            except Exception:
                pass
        return True

    def _persist_runtime_session_state_quietly(self) -> bool:
        if not self.session or not self.current_session_id:
            return False
        try:
            payload = self._build_session_persistence_payload()
            self.session.config = dict(payload.get("config", {}) or {})
            self.database.save_exam_session(self.current_session_id, self.session.exam_name, payload)
            self.current_session_path = self._session_path_from_id(self.current_session_id)
            if self.current_session_path and self.session:
                try:
                    self.session.save_json(self.current_session_path)
                except Exception:
                    pass
            self.session_dirty = False
            self._remember_current_session_snapshot()
            return True
        except Exception:
            self.session_dirty = True
            return False

    def _save_batch_for_selected_subject(
        self,
        *,
        show_success_message: bool = True,
        reload_after_save: bool = False,
        refresh_exam_list: bool = False,
    ) -> bool:
        subject_cfg = self._selected_batch_subject_config()
        if not subject_cfg:
            if show_success_message:
                QMessageBox.warning(self, "Lưu Batch", "Vui lòng chọn môn trước khi lưu Batch.")
            return False
        row_count = self.scan_list.rowCount() if hasattr(self, "scan_list") else 0
        if row_count <= 0:
            if show_success_message:
                QMessageBox.warning(self, "Lưu Batch", "Chưa có dữ liệu Batch Scan để lưu.")
            return False

        current_results = list(self._collect_current_subject_results_for_save() or self.scan_results or [])
        if not current_results and row_count > 0:
            current_results = list(self.scan_results or [])
        self.scan_results = current_results
        saved_results = [self._serialize_omr_result(result) for result in current_results]
        subject_key = self._subject_key_from_cfg(subject_cfg)
        saved_at = datetime.now().isoformat(timespec="seconds")

        # Không lưu snapshot lưới vào subject config. Toàn bộ bài nhận dạng đã nằm trong DB.
        subject_cfg.pop("batch_saved_rows", None)
        subject_cfg.pop("batch_saved_preview", None)
        subject_cfg.pop("batch_saved_results", None)
        subject_cfg["batch_saved"] = True
        subject_cfg["batch_saved_at"] = saved_at
        subject_cfg["batch_result_count"] = len(saved_results)

        self._sync_subject_batch_snapshot_after_inline_edit(subject_key, persist_full_subject_rows=False)

        if subject_key:
            try:
                subject_db_key = self._batch_result_subject_key(subject_key)
                self.database.replace_scan_results_for_subject(subject_db_key, saved_results, note="save_batch_subject")
                self._mark_subject_batch_saved(subject_key, len(saved_results), saved_at)
            except Exception as exc:
                QMessageBox.warning(self, "Lưu Batch", f"Không thể lưu dữ liệu Batch Scan vào CSDL: {exc}")
                return False
            self._batch_loaded_runtime_key = self._batch_runtime_key(subject_key)
            self._current_batch_data_source = "database"
            self._mark_scoring_stale_after_batch_save(subject_key)

        self.btn_save_batch_subject.setEnabled(False)
        self._persist_current_session_subject_configs()
        if reload_after_save:
            self._load_batch_subject_state(subject_cfg, source_hint="after_save", force_reload=False)
        else:
            self._update_batch_scan_scope_summary()

        current_route = str(getattr(self, "_current_route", "") or "").strip()
        if refresh_exam_list or current_route in {"exam_list", "dashboard"}:
            self._refresh_exam_list()

        if show_success_message:
            QMessageBox.information(self, "Lưu Batch", f"Đã lưu {len(saved_results)} bài của môn '{subject_key or 'hiện tại'}'.")
        return True

    def _merge_saved_batch_snapshot(self, cfg: dict) -> dict:
        # DB là nguồn sự thật duy nhất cho Batch Scan.
        # Chỉ giữ metadata nhẹ trong subject config; bỏ qua snapshot/cache cũ.
        merged = dict(cfg or {})
        for key in ("batch_saved_rows", "batch_saved_preview", "batch_saved_results"):
            if key in merged:
                merged[key] = []
        return merged

    def _current_scan_results_snapshot(self) -> list[OMRResult]:
        # scan_list columns: 0 stt, 1 sid, 2 exam_room, 3 exam_code, 4 full_name, 5 birth_date, 6 content, 7 status, 8 actions
        base = list(self.scan_results or [])
        if not hasattr(self, "scan_list"):
            return base
        row_count = self.scan_list.rowCount()
        if row_count <= 0:
            return base

        base_by_image: dict[str, list[OMRResult]] = {}
        for src in base:
            key = self._result_identity_key(getattr(src, "image_path", ""))
            if not key:
                continue
            base_by_image.setdefault(key, []).append(src)

        out: list[OMRResult] = []
        for idx in range(row_count):
            sid_item = self.scan_list.item(idx, self.SCAN_COL_STUDENT_ID)
            sid_text = str(sid_item.text() if sid_item else "").strip()
            image_path = str(sid_item.data(Qt.UserRole) if sid_item else "").strip()
            serialized_payload = sid_item.data(Qt.UserRole + 10) if sid_item else None
            result = None
            if isinstance(serialized_payload, dict) and serialized_payload:
                try:
                    result = self._deserialize_omr_result(serialized_payload)
                except Exception:
                    result = None
            if result is None:
                matched_base = None
                image_key = self._result_identity_key(image_path)
                if image_key and image_key in base_by_image and base_by_image[image_key]:
                    matched_base = base_by_image[image_key].pop(0)
                if matched_base is not None:
                    result = self._lightweight_result_copy(matched_base)
                elif image_key:
                    fallback = self._build_result_from_saved_table_row(idx)
                    result = fallback if fallback is not None else OMRResult(image_path=image_path)
                elif idx < len(base):
                    result = self._lightweight_result_copy(base[idx])
                else:
                    fallback = self._build_result_from_saved_table_row(idx)
                    result = fallback if fallback is not None else OMRResult(image_path="")

            result.student_id = "" if sid_text in {"", "-"} else sid_text
            result.exam_code = str(sid_item.data(Qt.UserRole + 1) if sid_item else "").strip()
            if image_path:
                result.image_path = image_path
            setattr(result, "exam_room", str(self.scan_list.item(idx, self.SCAN_COL_EXAM_ROOM).text() if self.scan_list.item(idx, self.SCAN_COL_EXAM_ROOM) else ""))
            setattr(result, "full_name", str(self.scan_list.item(idx, self.SCAN_COL_FULL_NAME).text() if self.scan_list.item(idx, self.SCAN_COL_FULL_NAME) else ""))
            setattr(result, "birth_date", str(self.scan_list.item(idx, self.SCAN_COL_BIRTH_DATE).text() if self.scan_list.item(idx, self.SCAN_COL_BIRTH_DATE) else ""))
            setattr(result, "manual_content_override", str(sid_item.data(Qt.UserRole + 11) if sid_item else "").strip())
            result.answer_string = self._normalize_non_api_answer_string(result)
            payload = self._build_scan_row_payload_from_result(result, row_idx=idx)
            setattr(result, "cached_content", str(payload.get("content", "") or ""))
            setattr(result, "cached_status", str(payload.get("status", "") or ""))
            setattr(result, "cached_recognized_short", str(payload.get("recognized_short", "") or ""))
            self._debug_scan_result_state("current_scan_results_snapshot", result)
            out.append(result)
        return out

    @staticmethod
    def _result_has_recognition_payload(result: OMRResult | None) -> bool:
        if result is None:
            return False
        if bool(str(getattr(result, "answer_string", "") or "").strip()):
            return True
        if bool(getattr(result, "mcq_answers", {}) or {}):
            return True
        if bool(getattr(result, "true_false_answers", {}) or {}):
            return True
        if bool(getattr(result, "numeric_answers", {}) or {}):
            return True
        if bool(getattr(result, "recognition_errors", []) or []):
            return True
        return bool(getattr(result, "issues", []) or [])

    def _restore_full_result_for_row(self, row_idx: int) -> OMRResult | None:
        if row_idx < 0 or row_idx >= self.scan_list.rowCount():
            return None
        sid_item = self.scan_list.item(row_idx, self.SCAN_COL_STUDENT_ID)
        image_path = str(sid_item.data(Qt.UserRole) if sid_item else "")
        sid_text = str(sid_item.text() if sid_item else "").strip()
        exam_code = str(sid_item.data(Qt.UserRole + 1) if sid_item else "").strip()
        if sid_text == "-":
            sid_text = ""

        def _match_result_pool(pool: list[OMRResult]) -> OMRResult | None:
            sid_matches: list[OMRResult] = []
            for cand in (pool or []):
                cand_img = str(getattr(cand, "image_path", "") or "").strip()
                cand_sid = str(getattr(cand, "student_id", "") or "").strip()
                cand_code = str(getattr(cand, "exam_code", "") or "").strip()
                if image_path and cand_img == image_path:
                    return cand
                if (not image_path) and sid_text and cand_sid == sid_text and (not exam_code or cand_code == exam_code):
                    sid_matches.append(cand)
            if len(sid_matches) == 1:
                return sid_matches[0]
            return None

        if 0 <= row_idx < len(self.scan_results):
            direct = self.scan_results[row_idx]
            if self._result_has_recognition_payload(direct):
                return self._lightweight_result_copy(direct)

        subject_key = self._current_batch_subject_key()
        subject_pool = list(self.scan_results_by_subject.get(self._batch_result_subject_key(subject_key), []) or [])
        matched = _match_result_pool(subject_pool)
        if matched is not None and self._result_has_recognition_payload(matched):
            return self._lightweight_result_copy(matched)

        db_rows = []
        cfg = self._subject_config_by_subject_key(subject_key)
        if cfg and hasattr(self, "_subject_scan_storage_key_candidates"):
            for candidate_key in self._subject_scan_storage_key_candidates(cfg):
                db_rows = self.database.fetch_scan_results_for_subject(candidate_key) or []
                if db_rows:
                    break
        if not db_rows:
            db_rows = self.database.fetch_scan_results_for_subject(self._batch_result_subject_key(subject_key)) or []
        db_pool: list[OMRResult] = []
        for item in db_rows:
            try:
                db_pool.append(self._deserialize_omr_result(item))
            except Exception:
                continue
        matched = _match_result_pool(db_pool)
        if matched is not None and self._result_has_recognition_payload(matched):
            return self._lightweight_result_copy(matched)

        return self._build_result_from_saved_table_row(row_idx)

    def _build_result_from_saved_table_row(self, idx: int) -> OMRResult | None:
        # scan_list columns: 0 stt, 1 sid, 2 exam_room, 3 exam_code, 4 full_name, 5 birth_date, 6 content, 7 status, 8 actions
        if idx < 0 or idx >= self.scan_list.rowCount():
            return None
        sid_item = self.scan_list.item(idx, self.SCAN_COL_STUDENT_ID)
        image_path = str(sid_item.data(Qt.UserRole) if sid_item else "")
        if not image_path:
            return None
        student_id = str(sid_item.text() if sid_item else "").strip()
        if student_id == "-":
            student_id = ""
        exam_code = str(sid_item.data(Qt.UserRole + 1) if sid_item else "").strip()
        result = OMRResult(image_path=image_path, student_id=student_id, exam_code=exam_code)
        serialized_payload = sid_item.data(Qt.UserRole + 10) if sid_item else None
        row_payload = sid_item.data(Qt.UserRole + 12) if sid_item else None
        if isinstance(serialized_payload, dict) and serialized_payload:
            try:
                restored = self._deserialize_omr_result(serialized_payload)
                result = restored
            except Exception:
                pass
        result = self._hydrate_result_from_saved_row_payload(result, row_payload) or result
        result.image_path = image_path or str(getattr(result, "image_path", "") or "")
        result.student_id = student_id
        result.exam_code = exam_code
        room_text = str(self.scan_list.item(idx, self.SCAN_COL_EXAM_ROOM).text() if self.scan_list.item(idx, self.SCAN_COL_EXAM_ROOM) else "").strip()
        if not room_text and student_id:
            room_text = str(self._subject_room_for_student_id(student_id) or "").strip()
        result.exam_room = room_text
        result.full_name = str(self.scan_list.item(idx, self.SCAN_COL_FULL_NAME).text() if self.scan_list.item(idx, self.SCAN_COL_FULL_NAME) else "")
        result.birth_date = str(self.scan_list.item(idx, self.SCAN_COL_BIRTH_DATE).text() if self.scan_list.item(idx, self.SCAN_COL_BIRTH_DATE) else "")
        manual_content_override = str(sid_item.data(Qt.UserRole + 11) if sid_item else "").strip()
        setattr(result, "manual_content_override", manual_content_override)
        result.answer_string = self._normalize_non_api_answer_string(result)
        payload = self._build_scan_row_payload_from_result(result, row_idx=idx)
        setattr(result, "cached_content", str(payload.get("content", "") or ""))
        setattr(result, "cached_status", str(payload.get("status", "") or ""))
        setattr(result, "cached_recognized_short", str(payload.get("recognized_short", "") or ""))
        self._debug_scan_result_state("build_result_from_saved_table_row", result)
        result.sync_legacy_aliases()
        return result

    def _set_scan_result_at_row(self, idx: int, result: OMRResult) -> None:
        if idx < 0:
            return
        result_image = self._result_identity_key(getattr(result, "image_path", ""))
        if result_image:
            for i, existing in enumerate(self.scan_results):
                if self._result_identity_key(getattr(existing, "image_path", "")) == result_image:
                    if any(j != i and self.scan_results[j] is result for j in range(len(self.scan_results))):
                        result = self._lightweight_result_copy(result)
                    self.scan_results[i] = result
                    return
        if idx == len(self.scan_results):
            self.scan_results.append(self._lightweight_result_copy(result))
            return
        if idx > len(self.scan_results):
            return
        if any(i != idx and self.scan_results[i] is result for i in range(len(self.scan_results))):
            result = self._lightweight_result_copy(result)
        self.scan_results[idx] = result

    def _subject_edit_registry(self, subject_key: str) -> dict[str, dict]:
        subject = str(subject_key or "").strip()
        if not subject:
            return {}
        cfg = self._subject_config_by_subject_key(subject)
        if not isinstance(cfg, dict):
            return {}
        registry = cfg.get("scan_edit_registry", {})
        return dict(registry) if isinstance(registry, dict) else {}

    def _store_subject_edit_registry(self, subject_key: str, registry: dict[str, dict], *, persist_session: bool = True) -> None:
        subject = str(subject_key or "").strip()
        if not subject:
            return
        cfg = self._subject_config_by_subject_key(subject)
        if not isinstance(cfg, dict):
            return

        normalized_registry: dict[str, dict] = {}
        for image_path, meta in dict(registry or {}).items():
            if not isinstance(meta, dict):
                continue
            image_key = self._result_identity_key(image_path)
            if not image_key:
                continue
            history = self._ordered_unique_text_list(meta.get("history", meta.get("edit_history", [])))
            manual_adjustments = self._ordered_unique_text_list(meta.get("manual_adjustments", []))
            last_adjustment = str(meta.get("last_adjustment", "") or (history[-1] if history else "")).strip()
            manually_edited = bool(meta.get("manually_edited", False) or history or manual_adjustments or last_adjustment)
            if not manually_edited:
                continue
            normalized_registry[image_key] = {
                "manually_edited": True,
                "history": history,
                "last_adjustment": last_adjustment,
                "manual_adjustments": manual_adjustments,
            }

        current_registry = cfg.get("scan_edit_registry", {}) if isinstance(cfg.get("scan_edit_registry", {}), dict) else {}
        if current_registry == normalized_registry:
            return

        cfg["scan_edit_registry"] = normalized_registry
        subject_cfgs = list(getattr(self, "subject_configs", []) or self._effective_subject_configs_for_batch() or [])
        if subject_cfgs:
            idx = self._find_subject_config_index_for_batch_save(cfg, subject_cfgs)
            if idx >= 0:
                subject_cfgs[idx] = cfg
            elif cfg not in subject_cfgs:
                subject_cfgs.append(cfg)
            self.subject_configs = subject_cfgs
        if self.session and isinstance(self.session.config, dict):
            session_subject_cfgs = self.session.config.get("subject_configs", [])
            if isinstance(session_subject_cfgs, list):
                matched = False
                for i, item in enumerate(session_subject_cfgs):
                    if item is cfg:
                        session_subject_cfgs[i] = cfg
                        matched = True
                        break
                    if isinstance(item, dict) and self._subject_key_from_cfg(item) == subject:
                        session_subject_cfgs[i].update(cfg)
                        matched = True
                        break
                if not matched:
                    session_subject_cfgs.append(cfg)
        if isinstance(getattr(self, "batch_editor_return_payload", None), dict):
            payload_subject_cfgs = self.batch_editor_return_payload.get("subject_configs", [])
            if isinstance(payload_subject_cfgs, list):
                matched = False
                for i, item in enumerate(payload_subject_cfgs):
                    if isinstance(item, dict) and self._subject_key_from_cfg(item) == subject:
                        payload_subject_cfgs[i].update(cfg)
                        matched = True
                        break
                if not matched:
                    payload_subject_cfgs.append(dict(cfg))
        if persist_session and subject_cfgs:
            try:
                self._persist_current_session_subject_configs(list(subject_cfgs))
            except Exception:
                pass
            try:
                self._persist_runtime_session_state_quietly()
            except Exception:
                pass

    def _persist_result_edit_registry(self, subject_key: str, result) -> None:
        subject = str(subject_key or "").strip()
        if not subject or result is None:
            return
        image_key = self._result_identity_key(getattr(result, "image_path", ""))
        if not image_key:
            return
        registry = self._subject_edit_registry(subject)
        history = self._ordered_unique_text_list(getattr(result, "edit_history", []) or [])
        manual_adjustments = self._ordered_unique_text_list(getattr(result, "manual_adjustments", []) or [])
        last_adjustment = str(getattr(result, "last_adjustment", "") or (history[-1] if history else "")).strip()
        manually_edited = bool(getattr(result, "manually_edited", False) or history or manual_adjustments or last_adjustment)
        if manually_edited:
            registry[image_key] = {
                "manually_edited": True,
                "history": history,
                "last_adjustment": last_adjustment,
                "manual_adjustments": manual_adjustments,
            }
        else:
            registry.pop(image_key, None)
        self._store_subject_edit_registry(subject, registry)

    def _clear_persisted_result_edit_registry(self, subject_key: str, image_path: str) -> None:
        subject = str(subject_key or "").strip()
        image_key = self._result_identity_key(image_path)
        if not subject or not image_key:
            return
        registry = self._subject_edit_registry(subject)
        if image_key not in registry:
            return
        registry.pop(image_key, None)
        self._store_subject_edit_registry(subject, registry)

    def _apply_persisted_result_edit_metadata(self, subject_key: str, result) -> None:
        subject = str(subject_key or "").strip()
        if not subject or result is None:
            return
        image_key = self._result_identity_key(getattr(result, "image_path", ""))
        if not image_key:
            return
        meta = self._subject_edit_registry(subject).get(image_key, {})
        if not isinstance(meta, dict) or not meta:
            return
        history = self._ordered_unique_text_list(meta.get("history", meta.get("edit_history", [])))
        manual_adjustments = self._ordered_unique_text_list(meta.get("manual_adjustments", []))
        last_adjustment = str(meta.get("last_adjustment", "") or (history[-1] if history else "")).strip()
        manually_edited = bool(meta.get("manually_edited", False) or history or manual_adjustments or last_adjustment)
        if not manually_edited:
            return
        setattr(result, "manually_edited", True)
        setattr(result, "edit_history", history)
        setattr(result, "last_adjustment", last_adjustment)
        setattr(result, "manual_adjustments", manual_adjustments)
        if str(getattr(result, "cached_forced_status", "") or "").strip() != "Đã sửa":
            setattr(result, "cached_forced_status", "Đã sửa")
        if history:
            self.scan_edit_history[image_key] = list(history)
        if last_adjustment:
            self.scan_last_adjustment[image_key] = last_adjustment
        if manual_adjustments:
            self.scan_manual_adjustments[image_key] = list(manual_adjustments)
        self.scan_forced_status_by_index[image_key] = "Đã sửa"

    @staticmethod
    def _result_identity_key(image_path: str) -> str:
        raw = str(image_path or "").strip()
        if not raw:
            return ""
        normalized = os.path.normpath(raw)
        normalized = os.path.normcase(normalized)
        return str(normalized or "").strip()

    def _row_image_key(self, row_idx: int) -> str:
        if row_idx < 0 or (not hasattr(self, "scan_list")) or row_idx >= self.scan_list.rowCount():
            return ""
        sid_item = self.scan_list.item(row_idx, self.SCAN_COL_STUDENT_ID)
        return self._result_identity_key(str(sid_item.data(Qt.UserRole) if sid_item else ""))

    def _row_index_by_image_path(self, image_path: str) -> int:
        key = self._result_identity_key(image_path)
        if not key or not hasattr(self, "scan_list"):
            return -1
        for row in range(self.scan_list.rowCount()):
            sid_item = self.scan_list.item(row, self.SCAN_COL_STUDENT_ID)
            row_image = self._result_identity_key(str(sid_item.data(Qt.UserRole) if sid_item else ""))
            if row_image == key:
                return row
        return -1

    def _result_by_image_path(self, image_path: str) -> OMRResult | None:
        image_key = self._result_identity_key(image_path)
        if not image_key:
            return None
        for result in (self.scan_results or []):
            if self._result_identity_key(getattr(result, "image_path", "")) == image_key:
                return result
        return None

    def _mark_result_manually_edited(self, result: OMRResult, row_idx: int | None = None) -> None:
        setattr(result, "manually_edited", True)
        setattr(result, "cached_forced_status", "Đã sửa")
        setattr(result, "cached_status", "Đã sửa")
        image_key = self._result_identity_key(getattr(result, "image_path", ""))
        resolved_row = row_idx if row_idx is not None and row_idx >= 0 else self._row_index_by_image_path(image_key)
        if image_key:
            self.scan_forced_status_by_index[image_key] = "Đã sửa"
        if resolved_row is not None and resolved_row >= 0 and hasattr(self, "scan_list") and resolved_row < self.scan_list.rowCount():
            status_item = self.scan_list.item(resolved_row, self.SCAN_COL_STATUS)
            if status_item is None:
                status_item = QTableWidgetItem("Đã sửa")
                self.scan_list.setItem(resolved_row, self.SCAN_COL_STATUS, status_item)
            else:
                status_item.setText("Đã sửa")
            status_item.setToolTip("Đã sửa")
            status_item.setForeground(Qt.red)

    def _persist_scan_results_to_db(self, subject_key: str) -> None:
        scoped_key = self._batch_result_subject_key(subject_key)
        active_subject = str(self._current_batch_subject_key() or "").strip()
        if active_subject and self._batch_result_subject_key(active_subject) == scoped_key and self.scan_results:
            source_rows = list(self.scan_results)
            self.scan_results_by_subject[scoped_key] = list(self.scan_results)
        else:
            source_rows = list(self.scan_results_by_subject.get(scoped_key, self.scan_results) or [])
        for result in source_rows:
            result.answer_string = self._normalize_non_api_answer_string(result, subject_key)
        rows = [self._serialize_omr_result(x) for x in source_rows]
        self.database.replace_scan_results_for_subject(scoped_key, rows)
        self.database.log_change("scan_results", scoped_key, "replace_subject_rows", "", f"{len(rows)} rows", "batch_save")
        self._refresh_scan_results_from_db(subject_key)

    @staticmethod
    def _debug_scan_result_state(tag: str, result: OMRResult | None) -> None:
        if result is None:
            return

    def _update_working_batch_state_single_row(self, subject_key: str, result: OMRResult, row_payload: dict, row_idx: int) -> None:
        # DB-only mode: không duy trì working cache theo môn.
        return

    def _persist_single_scan_result_to_db(self, result: OMRResult, note: str = "") -> None:
        subject_key = str(self._current_batch_subject_key() or self.active_batch_subject_key or "").strip()
        image_path = str(getattr(result, "image_path", "") or "").strip()
        if not subject_key or not image_path or result is None:
            return

        history = self._ordered_unique_text_list(getattr(result, "edit_history", []) or self.scan_edit_history.get(image_path, []))
        manual_adjustments = self._ordered_unique_text_list(getattr(result, "manual_adjustments", []) or self.scan_manual_adjustments.get(image_path, []))
        last_adjustment = str(getattr(result, "last_adjustment", "") or self.scan_last_adjustment.get(image_path, "") or (history[-1] if history else "")).strip()
        if history:
            setattr(result, "edit_history", history)
            self.scan_edit_history[image_path] = list(history)
        if manual_adjustments:
            setattr(result, "manual_adjustments", manual_adjustments)
            self.scan_manual_adjustments[image_path] = list(manual_adjustments)
        if last_adjustment:
            setattr(result, "last_adjustment", last_adjustment)
            self.scan_last_adjustment[image_path] = last_adjustment
        if bool(history or manual_adjustments or last_adjustment or getattr(result, "manually_edited", False)):
            setattr(result, "manually_edited", True)
            setattr(result, "cached_forced_status", "Đã sửa")
            self.scan_forced_status_by_index[image_path] = "Đã sửa"

        result.answer_string = self._normalize_non_api_answer_string(result, subject_key)
        row_idx = self._row_index_by_image_path(image_path)
        serialized = self._serialize_omr_result(result)
        subject_db_key = self._batch_result_subject_key(subject_key)

        updated = False
        try:
            update_result = self.database.update_scan_result_payload(
                subject_db_key,
                image_path,
                serialized,
                note=note,
            )
            updated = update_result is not False
        except Exception:
            updated = False

        if not updated:
            try:
                db_rows = list(self.database.fetch_scan_results_for_subject(subject_db_key) or [])
            except Exception:
                db_rows = []
            replaced = False
            for i, payload in enumerate(list(db_rows)):
                if isinstance(payload, dict) and str(payload.get("image_path", "") or "").strip() == image_path:
                    db_rows[i] = dict(serialized)
                    replaced = True
                    break
            if not replaced:
                db_rows.append(dict(serialized))
            try:
                self.database.replace_scan_results_for_subject(subject_db_key, db_rows)
                try:
                    self.database.log_change("scan_results", image_path, "persist_single_fallback", "", note or "inline_edit", note or "inline_edit")
                except Exception:
                    pass
            except Exception:
                pass

        self.scan_results_by_subject[subject_db_key] = list(self.scan_results or [])
        try:
            self._update_single_subject_saved_snapshot(subject_key, result, {}, row_idx)
        except Exception:
            pass
        self._persist_result_edit_registry(subject_key, result)

    def _update_single_subject_saved_snapshot(self, subject_key: str, result: OMRResult, row_payload: dict, row_idx: int) -> None:
        subject = str(subject_key or "").strip()
        if not subject:
            return
        cfg = self._subject_config_by_subject_key(subject)
        if not isinstance(cfg, dict):
            return

        scoped_key = self._batch_result_subject_key(subject)
        result_count = len(self.scan_results_by_subject.get(scoped_key, []) or [])
        if result_count <= 0 and subject == str(self._current_batch_subject_key() or "").strip() and hasattr(self, "scan_list"):
            result_count = self.scan_list.rowCount()

        cfg["batch_saved"] = bool(result_count)
        cfg["batch_saved_at"] = datetime.now().isoformat(timespec="seconds") if result_count else ""
        cfg["batch_result_count"] = result_count
        cfg["batch_saved_rows"] = []
        cfg["batch_saved_preview"] = []
        cfg["batch_saved_results"] = []

        if self.session and isinstance(self.session.config, dict):
            subject_cfgs = self.session.config.get("subject_configs", [])
            if isinstance(subject_cfgs, list):
                for i, item in enumerate(subject_cfgs):
                    if item is cfg:
                        subject_cfgs[i] = cfg
                        break
                    if isinstance(item, dict) and self._subject_key_from_cfg(item) == subject:
                        subject_cfgs[i].update(cfg)
                        break
        if isinstance(self.batch_editor_return_payload, dict):
            subject_cfgs = self.batch_editor_return_payload.get("subject_configs", [])
            if isinstance(subject_cfgs, list):
                for i, item in enumerate(subject_cfgs):
                    if isinstance(item, dict) and self._subject_key_from_cfg(item) == subject:
                        subject_cfgs[i].update(cfg)
                        break

        self.batch_working_state_by_subject.pop(self._batch_runtime_key(subject), None)

    def _sync_subject_batch_snapshot_after_inline_edit(self, subject_key: str = "", persist_full_subject_rows: bool = False) -> None:
        subject = str(subject_key or self._current_batch_subject_key() or "").strip()
        if not subject:
            return
        cfg = self._subject_config_by_subject_key(subject)
        if isinstance(cfg, dict):
            result_count = len(self.scan_results_by_subject.get(self._batch_result_subject_key(subject), []) or [])
            if result_count <= 0 and subject == str(self._current_batch_subject_key() or "").strip() and hasattr(self, "scan_list"):
                result_count = self.scan_list.rowCount()
            cfg["batch_saved"] = bool(result_count)
            cfg["batch_result_count"] = result_count
            if result_count and not cfg.get("batch_saved_at"):
                cfg["batch_saved_at"] = datetime.now().isoformat(timespec="seconds")
        if persist_full_subject_rows:
            try:
                self._persist_scan_results_to_db(subject)
            except Exception:
                pass
        self._update_batch_scan_scope_summary()

    def _refresh_all_statuses(self) -> None:
        duplicate_count_map: dict[str, int] = {}
        canonical_by_image: dict[str, OMRResult] = {}
        for res in self.scan_results:
            canonical_by_image[self._result_identity_key(getattr(res, "image_path", ""))] = res
        for res in canonical_by_image.values():
            sid = str(getattr(res, "student_id", "") or "").strip()
            if not self._student_id_has_recognition_error(sid):
                duplicate_count_map[sid] = duplicate_count_map.get(sid, 0) + 1
        subject_scope = self._subject_student_room_scope()
        available_exam_codes = self._available_exam_codes()
        for row_idx in range(self.scan_list.rowCount()):
            self._refresh_row_status(
                row_idx,
                duplicate_count_map=duplicate_count_map,
                subject_scope=subject_scope,
                available_exam_codes=available_exam_codes,
            )
        self._rebuild_error_list()
        self._apply_scan_filter()
