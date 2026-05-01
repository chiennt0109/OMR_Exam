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


class MainWindowBatchSubjectMixin:
    """Subject key resolution, subject scope, student/profile/room lookup and Batch subject selectors."""
    def _subject_uses_direct_score_import(self, subject_or_cfg: str | dict | None) -> bool:
        cfg = subject_or_cfg if isinstance(subject_or_cfg, dict) else self._subject_config_by_subject_key(str(subject_or_cfg or "").strip())
        if not isinstance(cfg, dict):
            return False
        return bool(cfg.get("is_essay_subject"))

    def _direct_score_import_rows_for_subject(self, subject_or_cfg: str | dict | None) -> list[dict]:
        cfg = subject_or_cfg if isinstance(subject_or_cfg, dict) else self._subject_config_by_subject_key(str(subject_or_cfg or "").strip())
        if not isinstance(cfg, dict) or not self._subject_uses_direct_score_import(cfg):
            return []

        payload = cfg.get("direct_score_import", {}) or {}
        raw_rows = payload.get("rows", []) or []
        normalized: list[dict] = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            sid = str(raw.get("student_id", "") or raw.get("sid", "") or raw.get("sbd", "") or "").strip()
            score_text = str(raw.get("score", "") or raw.get("point", "") or raw.get("mark", "") or "").strip()
            if not sid or score_text == "":
                continue
            normalized.append({
                "student_id": sid,
                "score": score_text,
                "exam_room": str(raw.get("exam_room", "") or raw.get("room", "") or "").strip(),
                "manually_edited": bool(raw.get("manually_edited", False)),
            })
        normalized.sort(key=lambda row: (str(row.get("student_id", "") or ""), str(row.get("exam_room", "") or "")))
        return normalized

    def _build_direct_score_payloads_for_subject(self, subject_key: str) -> list[dict]:
        subject = str(subject_key or "").strip()
        if not subject:
            return []

        cfg = self._subject_config_by_subject_key(subject) or {}
        if not self._subject_uses_direct_score_import(cfg):
            return []

        default_room = str((cfg or {}).get("exam_room_name", "") or "").strip()
        payloads: list[dict] = []
        for row in self._direct_score_import_rows_for_subject(cfg):
            sid = str(row.get("student_id", "") or "").strip()
            score_text = str(row.get("score", "") or "").strip().replace(",", ".")
            if not sid or score_text == "":
                continue
            try:
                score_value = float(score_text)
            except Exception:
                continue

            profile = self._student_profile_by_id(sid)
            room_text = (
                str(row.get("exam_room", "") or "").strip()
                or self._subject_room_for_student_id(sid, cfg)
                or default_room
                or str(profile.get("exam_room", "") or "").strip()
            )

            is_manual_edit = bool(row.get("manually_edited", False))
            payloads.append({
                "student_id": sid,
                "name": str(profile.get("name", "") or "").strip(),
                "subject": subject,
                "exam_code": "",
                "mcq_correct": 0,
                "tf_correct": 0,
                "numeric_correct": 0,
                "tf_compare": "",
                "numeric_compare": "",
                "correct": 0,
                "wrong": 0,
                "blank": 0,
                "score": score_value,
                "class_name": str(profile.get("class_name", "") or "").strip(),
                "birth_date": str(profile.get("birth_date", "") or "").strip(),
                "exam_room": room_text,
                "status": "OK",
                "note": "Nhập điểm trực tiếp (đã sửa)" if is_manual_edit else "Nhập điểm trực tiếp",
                "manually_edited": is_manual_edit,
            })
        payloads.sort(key=lambda row: str(row.get("student_id", "") or ""))
        return payloads

    def _materialize_direct_score_rows_for_subject(self, subject_key: str, *, persist: bool = False) -> dict[str, dict]:
        subject = str(subject_key or "").strip()
        if not subject:
            return {}

        payloads = self._build_direct_score_payloads_for_subject(subject)
        packed: dict[str, dict] = {}
        for payload in payloads:
            sid_key = str(payload.get("student_id", "") or "").strip()
            if sid_key:
                packed[sid_key] = dict(payload)

        if packed:
            self.scoring_results_by_subject[subject] = dict(packed)
            if persist:
                self._persist_scoring_results_for_subject(
                    subject,
                    list(packed.values()),
                    "Nhập điểm trực tiếp",
                    "essay_direct_import",
                    mark_saved=True,
                )
        return packed

    def _subject_config_by_subject_key(self, subject_key: str) -> dict | None:
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        exact_match: dict | None = None
        logical_matches: list[dict] = []
        for cfg in self._subject_configs_for_scoring():
            canonical = self._subject_instance_key_from_cfg(cfg)
            logical = self._logical_subject_key_from_cfg(cfg)
            if key_norm == canonical:
                exact_match = cfg
                break
            if key_norm == logical:
                logical_matches.append(cfg)
        if exact_match is not None:
            return exact_match
        # legacy fallback / migration: only allow logical-key lookup when unambiguous.
        if len(logical_matches) == 1:
            return logical_matches[0]
        return None

    def _session_subject_config_ref_by_subject_key(self, subject_key: str) -> dict | None:
        """Return the mutable subject config stored inside session.config.

        Do not use `_effective_subject_configs_for_batch()` here because that path
        intentionally returns copied configs with merged defaults for UI/runtime
        consumption. For persistence-sensitive fields such as deleted scan images,
        we must mutate the live object inside `session.config["subject_configs"]`.
        """
        key_norm = str(subject_key or "").strip()
        if not key_norm:
            return None
        exact_match: dict | None = None
        logical_matches: list[dict] = []
        for idx, cfg in enumerate(self._subject_configs_in_session()):
            if not isinstance(cfg, dict):
                continue
            self._ensure_subject_instance_key(cfg, idx)
            canonical = self._subject_instance_key_from_cfg(cfg)
            logical = self._logical_subject_key_from_cfg(cfg)
            if key_norm == canonical:
                exact_match = cfg
                break
            if key_norm == logical:
                logical_matches.append(cfg)
        if exact_match is not None:
            return exact_match
        if len(logical_matches) == 1:
            return logical_matches[0]
        return None

    def _ensure_answer_keys_for_subject(self, subject_key: str) -> bool:
        if self.answer_keys and any(str(k).startswith(f"{subject_key}::") for k in self.answer_keys.keys.keys()):
            return True
        cfg = self._subject_config_by_subject_key(subject_key)
        if not cfg:
            return self.answer_keys is not None

        imported_keys = cfg.get("imported_answer_keys", {}) or {}
        if isinstance(imported_keys, dict) and imported_keys:
            repo = self.answer_keys or AnswerKeyRepository()
            for exam_code, kd in imported_keys.items():
                repo.upsert(
                    SubjectKey(
                        subject=subject_key,
                        exam_code=str(exam_code),
                        answers={int(k): str(v) for k, v in (kd.get("mcq_answers", {}) or {}).items()},
                        true_false_answers={int(k): dict(v) for k, v in (kd.get("true_false_answers", {}) or {}).items()},
                        numeric_answers={int(k): str(v) for k, v in (kd.get("numeric_answers", {}) or {}).items()},
                        full_credit_questions={
                            str(sec): [int(x) for x in (vals or []) if str(x).strip().lstrip("-").isdigit()]
                            for sec, vals in (kd.get("full_credit_questions", {}) or {}).items()
                        },
                        invalid_answer_rows={
                            str(sec): {int(k): str(v) for k, v in (vals or {}).items()}
                            for sec, vals in (kd.get("invalid_answer_rows", {}) or {}).items()
                        },
                    )
                )
            self.answer_keys = repo
            self.imported_exam_codes = sorted(str(k) for k in imported_keys.keys())
            self.active_batch_subject_key = subject_key
            return True

        answer_key_path = str(cfg.get("answer_key_path", "") or (self.session.answer_key_path if self.session else "") or "")
        if answer_key_path:
            pth = Path(answer_key_path)
            if pth.exists() and pth.suffix.lower() == ".json":
                try:
                    loaded = AnswerKeyRepository.load_json(pth)
                    repo = self.answer_keys or AnswerKeyRepository()
                    for key_obj in loaded.keys.values():
                        repo.upsert(key_obj)
                    self.answer_keys = repo
                    all_codes: set[str] = set()
                    for key_name in self.answer_keys.keys.keys():
                        parts = str(key_name).split("::", 1)
                        if len(parts) == 2 and parts[0] == subject_key:
                            all_codes.add(parts[1])
                    if all_codes:
                        self.imported_exam_codes = sorted(all_codes)
                    self.active_batch_subject_key = subject_key
                    return True
                except Exception:
                    pass

        return self.answer_keys is not None

    @staticmethod
    def _logical_subject_key_from_cfg(cfg: dict) -> str:
        key = str(cfg.get("answer_key_key", "") or "").strip()
        if key:
            return key
        name = str(cfg.get("name", "") or "").strip()
        block = str(cfg.get("block", "") or "").strip()
        return f"{name}_{block}" if name and block else (name or "General")

    def _ensure_subject_instance_key(self, cfg: dict, index: int | None = None) -> str:
        # canonical subject_instance_key: scoped runtime/storage key, stable across edits of display fields.
        if not isinstance(cfg, dict):
            return ""
        existing = str(cfg.get("subject_instance_key", "") or "").strip()
        scope_prefix = self._session_scope_prefix() or "session"
        # hard isolation guard: if a key belongs to another session scope, remap it to current session scope.
        if existing and scope_prefix and not existing.startswith(f"{scope_prefix}::"):
            legacy_keys = list(cfg.get("legacy_subject_instance_keys", [])) if isinstance(cfg.get("legacy_subject_instance_keys", []), list) else []
            if existing not in legacy_keys:
                legacy_keys.append(existing)
            cfg["legacy_subject_instance_keys"] = legacy_keys[-10:]
            existing = ""
        if existing:
            return existing
        logical = self._logical_subject_key_from_cfg(cfg) or "General"
        uid = str(cfg.get("subject_uid", "") or "").strip()
        if not uid:
            uid = str(uuid.uuid4())
            cfg["subject_uid"] = uid
        suffix = f"{int(index)}" if isinstance(index, int) and index >= 0 else uid
        value = f"{scope_prefix}::{logical}::{suffix}"
        cfg["logical_subject_key"] = logical
        cfg["subject_instance_key"] = value
        return value

    def _subject_instance_key_from_cfg(self, cfg: dict) -> str:
        if not isinstance(cfg, dict):
            return ""
        return self._ensure_subject_instance_key(cfg)

    def _current_subject_instance_key(self) -> str:
        cfg = self._selected_batch_subject_config()
        if not isinstance(cfg, dict):
            return ""
        return self._subject_instance_key_from_cfg(cfg)

    def _current_batch_runtime_key(self) -> str:
        return self._batch_runtime_key(self._current_subject_instance_key())

    def _subject_key_from_cfg(self, cfg: dict) -> str:
        # backward compatibility shim: storage/runtime key is now subject_instance_key.
        return self._subject_instance_key_from_cfg(cfg)

    def _batch_result_subject_key(self, subject_key: str) -> str:
        base = str(subject_key or "").strip()
        scope_prefix = self._session_scope_prefix()
        if scope_prefix and base:
            if base.startswith(f"{scope_prefix}::"):
                return base
            return f"{scope_prefix}::{base}"
        return base

    def _session_scope_prefix(self) -> str:
        return self._session_scope_prefix_for()

    def _session_scope_prefix_for(self, session_id: str | None = None, session: ExamSession | None = None) -> str:
        sid = str(self.current_session_id if session_id is None else session_id or "").strip()
        ses = self.session if session is None else session
        exam_name = str((ses.exam_name if ses else "") or "").strip().lower()
        if sid and exam_name:
            return f"{sid}::{exam_name}"
        return sid

    def _answer_key_subject_key(self, subject_key: str, subject_cfg: dict | None = None) -> str:
        base = str(subject_key or "").strip()
        if not base:
            return ""
        if "::" in base:
            return base
        block = ""
        if isinstance(subject_cfg, dict):
            block = str(subject_cfg.get("block", "") or "").strip()
        if not block and "_" in base:
            block = str(base.rsplit("_", 1)[-1]).strip()
        scope_prefix = self._session_scope_prefix()
        if scope_prefix and block:
            return f"{scope_prefix}::{base}::{block}"
        if scope_prefix:
            return f"{scope_prefix}::{base}"
        return base

    def _fetch_answer_keys_for_subject_scoped(self, subject_key: str, subject_cfg: dict | None = None) -> dict[str, dict[str, Any]]:
        candidates: list[str] = []

        def _push(value: str) -> None:
            key = str(value or "").strip()
            if key and key not in candidates:
                candidates.append(key)

        _push(subject_key)
        if isinstance(subject_cfg, dict):
            _push(str(subject_cfg.get("answer_key_key", "") or ""))
            _push(self._logical_subject_key_from_cfg(subject_cfg))
            _push(self._subject_instance_key_from_cfg(subject_cfg))

        has_session_scope = bool(str(self.current_session_id or "").strip())
        block = str((subject_cfg or {}).get("block", "") or "").strip() if isinstance(subject_cfg, dict) else ""
        sid = str(self.current_session_id or "").strip()

        for candidate in candidates:
            scoped = self._answer_key_subject_key(candidate, subject_cfg)
            rows = self.database.fetch_answer_keys_for_subject(scoped) if scoped else {}
            if not rows and sid:
                legacy_scoped = f"{sid}::{candidate}::{block}" if block else f"{sid}::{candidate}"
                rows = self.database.fetch_answer_keys_for_subject(legacy_scoped)
            if not rows and (not has_session_scope) and scoped and scoped != candidate:
                rows = self.database.fetch_answer_keys_for_subject(candidate)
            if rows:
                return rows or {}

        return {}

    def _save_answer_keys_for_subject_scoped(
        self,
        subject_key: str,
        subject_cfg: dict | None,
        answer_keys: dict,
    ) -> str:
        """Save answer keys using the same scoped key family used by fetch/scoring."""
        cfg = subject_cfg if isinstance(subject_cfg, dict) else self._subject_config_by_subject_key(subject_key)
        storage_key = self._answer_key_subject_key(subject_key, cfg)
        if not storage_key and isinstance(cfg, dict):
            storage_key = self._answer_key_subject_key(self._subject_instance_key_from_cfg(cfg), cfg)
        if not storage_key:
            return ""

        safe_rows = dict(answer_keys or {})
        self.database.replace_answer_keys_for_subject(storage_key, safe_rows)

        try:
            if isinstance(cfg, dict):
                cfg["answer_key_key"] = storage_key
                cfg["imported_answer_keys"] = safe_rows
            cfg_ref = self._session_subject_config_ref_by_subject_key(subject_key)
            if isinstance(cfg_ref, dict):
                cfg_ref["answer_key_key"] = storage_key
                cfg_ref["imported_answer_keys"] = safe_rows
        except Exception:
            pass
        return storage_key

    def _display_subject_label(self, cfg: dict | None) -> str:
        if not isinstance(cfg, dict):
            return "-"
        exam_name = str((self.session.exam_name if self.session else "") or "").strip() or "Kỳ thi hiện tại"
        subject_name = str(cfg.get("name", "") or "").strip() or str(self._logical_subject_key_from_cfg(cfg) or "-")
        block = str(cfg.get("block", "") or "").strip() or "-"
        return f"{exam_name} | {subject_name} | Khối {block}"

    def _batch_cache_subject_key(self, cfg: dict | None, include_session: bool = True) -> str:
        if not isinstance(cfg, dict):
            return ""
        canonical = str(self._subject_instance_key_from_cfg(cfg) or "").strip().lower()
        if canonical:
            return canonical
        name = str(cfg.get("name", "") or "").strip().lower()
        block = str(cfg.get("block", "") or "").strip().lower()
        answer_key = str(cfg.get("answer_key_key", "") or "").strip().lower()
        base = f"{name}::{block}::{answer_key}"
        if include_session:
            scope_prefix = self._session_scope_prefix().strip().lower()
            if scope_prefix:
                return f"{scope_prefix}::{base}"
        return base

    def _subject_configs_in_session(self) -> list[dict]:
        if not self.session:
            return []
        cfg = self.session.config or {}
        raw = cfg.get("subject_configs", [])
        items = raw if isinstance(raw, list) else []
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                # legacy fallback / migration: promote old logical-only configs to canonical subject_instance_key.
                self._ensure_subject_instance_key(item, idx)
        return items

    def _student_profile_by_id(self, student_id: str) -> dict:
        sid = str(student_id or "").strip()
        if not sid or not self.session:
            return {}
        normalized_sid = self._normalized_student_id_for_match(sid)
        for s in (self.session.students or []):
            candidate = str(getattr(s, "student_id", "") or "").strip()
            if candidate == sid:
                return {
                    "name": str(getattr(s, "name", "") or ""),
                    "birth_date": self._format_birth_date_mmddyyyy(str((getattr(s, "extra", {}) or {}).get("birth_date", "") or "")),
                    "class_name": str((getattr(s, "extra", {}) or {}).get("class_name", "") or ""),
                    "exam_room": str((getattr(s, "extra", {}) or {}).get("exam_room", "") or ""),
                }
            if normalized_sid and self._normalized_student_id_for_match(candidate) == normalized_sid:
                return {
                    "name": str(getattr(s, "name", "") or ""),
                    "birth_date": self._format_birth_date_mmddyyyy(str((getattr(s, "extra", {}) or {}).get("birth_date", "") or "")),
                    "class_name": str((getattr(s, "extra", {}) or {}).get("class_name", "") or ""),
                    "exam_room": str((getattr(s, "extra", {}) or {}).get("exam_room", "") or ""),
                }
        return {}

    @staticmethod
    def _format_birth_date_mmddyyyy(value: str) -> str:
        text = str(value or "").strip()
        if not text or text == "-":
            return ""
        normalized = text.replace("-", "/").replace(".", "/")
        candidates = [
            "%m/%d/%Y", "%m/%d/%y",
            "%d/%m/%Y", "%d/%m/%y",
            "%Y/%m/%d", "%Y/%d/%m",
        ]
        for fmt in candidates:
            try:
                parsed = datetime.strptime(normalized, fmt)
                return parsed.strftime("%m/%d/%Y")
            except Exception:
                pass
        chunks = [part for part in normalized.split("/") if part]
        if len(chunks) == 3 and all(part.isdigit() for part in chunks):
            nums = [int(part) for part in chunks]
            if len(chunks[0]) == 4:
                year, month, day = nums
            elif int(chunks[0]) > 12 and int(chunks[1]) <= 12:
                day, month, year = nums
            else:
                month, day, year = nums
            if year < 100:
                year += 2000 if year < 70 else 1900
            try:
                return datetime(year, month, day).strftime("%m/%d/%Y")
            except Exception:
                return text
        return text

    @staticmethod
    def _birth_date_missing(birth_date_text: str) -> bool:
        birth = str(birth_date_text or "").strip()
        return birth in {"", "-"}

    def _refresh_student_profile_for_result(self, result, row_idx: int | None = None) -> None:
        sid = str(getattr(result, "student_id", "") or "").strip()
        profile = self._student_profile_by_id(sid)
        setattr(result, "full_name", str(profile.get("name", "") or ""))
        setattr(result, "birth_date", self._format_birth_date_mmddyyyy(str(profile.get("birth_date", "") or "")))
        setattr(result, "class_name", str(profile.get("class_name", "") or ""))
        exam_room = str(self._subject_room_for_student_id(sid) or "").strip()
        if not exam_room:
            exam_room = str(profile.get("exam_room", "") or "").strip()
        setattr(result, "exam_room", exam_room)
        if row_idx is not None and 0 <= row_idx < self.scan_list.rowCount():
            self.scan_list.setItem(row_idx, self.SCAN_COL_EXAM_ROOM, QTableWidgetItem(str(getattr(result, "exam_room", "") or "-")))
            self.scan_list.setItem(row_idx, self.SCAN_COL_FULL_NAME, QTableWidgetItem(str(getattr(result, "full_name", "") or "-")))
            self.scan_list.setItem(row_idx, self.SCAN_COL_BIRTH_DATE, QTableWidgetItem(str(getattr(result, "birth_date", "") or "-")))

    @staticmethod
    def _normalized_student_id_for_match(student_id: str) -> str:
        sid = str(student_id or "").strip()
        if not sid:
            return ""
        compact = sid.replace(" ", "")
        if compact.endswith(".0"):
            prefix = compact[:-2]
            if prefix.isdigit():
                compact = prefix
        if compact.isdigit():
            compact = compact.lstrip("0") or "0"
        return compact.upper()

    @staticmethod
    def _normalized_room_for_match(room_text: str) -> str:
        room = str(room_text or "").strip().casefold()
        if room.isdigit():
            room = room.lstrip("0") or "0"
        return room

    @staticmethod
    def _subject_imported_answer_keys_for_main(subject_cfg: dict) -> dict:
        if not isinstance(subject_cfg, dict):
            return {}
        raw = subject_cfg.get("imported_answer_keys", {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _effective_subject_configs_for_batch(self) -> list[dict]:
        cfgs = self._subject_configs_in_session()
        common_template = self._normalize_template_path(str(self.session.template_path if self.session else ""))
        if cfgs:
            # Default to exam common template, but keep subject-specific template if explicitly configured.
            out_cfgs: list[dict] = []
            for cfg in cfgs:
                if not isinstance(cfg, dict):
                    continue
                item = dict(cfg)
                subject_template = self._normalize_template_path(str(item.get("template_path", "") or ""))
                item["template_path"] = subject_template or common_template
                out_cfgs.append(item)
            return out_cfgs
        # Fallback for older sessions without subject_configs.
        if not self.session:
            return []
        scan_root = str((self.session.config or {}).get("scan_root", "") or "")
        out: list[dict] = []
        for raw in (self.session.subjects or ["General"]):
            subject = str(raw)
            name, block = (subject.split("_", 1) + [""])[:2] if "_" in subject else (subject, "")
            out.append({
                "name": name,
                "block": block,
                "template_path": common_template or str(self.session.template_path or ""),
                "scan_folder": scan_root,
                "answer_key_key": subject,
                "imported_answer_keys": {},
            })
        return out

    def _resolve_subject_config_for_batch(self) -> dict | None:
        cfg = self._selected_batch_subject_config()
        if cfg:
            return cfg
        cfgs = self._effective_subject_configs_for_batch()
        if not cfgs:
            return None
        # Do not show extra chooser dialog: Batch panel already has subject combo.
        # Fallback to first configured subject when none is currently selected.
        return cfgs[0]

    def _refresh_batch_subject_controls(self) -> None:
        if not hasattr(self, "batch_subject_combo"):
            return
        previous_active_key = str(getattr(self, "active_batch_subject_key", "") or "").strip()
        current_key = str(self.batch_subject_combo.currentData() or "").strip() if self.batch_subject_combo.count() > 0 else ""
        self.batch_subject_combo.blockSignals(True)
        self.batch_subject_combo.clear()
        self.batch_subject_combo.addItem("[Chọn môn]", "")
        target_index = 0
        for idx, cfg in enumerate(self._effective_subject_configs_for_batch()):
            if isinstance(cfg, dict):
                self._ensure_subject_instance_key(cfg, idx)
            label = self._display_subject_label(cfg)
            key = self._subject_instance_key_from_cfg(cfg)
            self.batch_subject_combo.addItem(label, key)
            if current_key and key == current_key:
                target_index = idx + 1
            elif not current_key and previous_active_key and key == previous_active_key:
                target_index = idx + 1
        self.batch_subject_combo.setCurrentIndex(target_index)
        self.batch_subject_combo.blockSignals(False)
        selected_key = str(self.batch_subject_combo.currentData() or "").strip() if self.batch_subject_combo.currentIndex() > 0 else ""
        selected_cfg = self._selected_batch_subject_config() if selected_key else None
        selected_sig = self._batch_subject_refresh_signature(selected_cfg) if isinstance(selected_cfg, dict) else ""
        has_signature_change = bool(selected_key) and bool(selected_sig) and selected_sig != str(getattr(self, "_batch_loaded_subject_signature", "") or "")
        should_load = bool(selected_key) and (
            selected_key != previous_active_key
            or self.scan_list.rowCount() <= 0
            or has_signature_change
        )
        if should_load:
            self._on_batch_subject_changed(self.batch_subject_combo.currentIndex(), force_reload=has_signature_change)
        self._handle_stack_changed(self.stack.currentIndex())

    def _selected_batch_subject_config(self) -> dict | None:
        if not hasattr(self, "batch_subject_combo"):
            return None
        idx = self.batch_subject_combo.currentIndex()
        if idx <= 0:
            return None
        key = str(self.batch_subject_combo.itemData(idx) or "").strip()
        if not key:
            return None
        return self._subject_config_by_subject_key(key)

    def _subject_section_question_counts(self, subject_key: str = "") -> dict[str, int]:
        counts = {"MCQ": 0, "TF": 0, "NUMERIC": 0}
        cfg = self._subject_config_by_subject_key(subject_key) if subject_key else (self._selected_batch_subject_config() or self._resolve_subject_config_for_batch())
        if not cfg:
            return counts
        raw_counts = cfg.get("question_counts", {}) if isinstance(cfg.get("question_counts", {}), dict) else {}
        for sec in counts:
            try:
                counts[sec] = max(0, int(raw_counts.get(sec, 0) or 0))
            except Exception:
                counts[sec] = 0
        if any(counts.values()):
            return counts
        template_path = self._normalize_template_path(str(cfg.get("template_path", "") or ""))
        if template_path:
            try:
                return SubjectConfigDialog._template_question_counts(template_path)
            except Exception:
                return counts
        return counts

    def _apply_subject_section_visibility(self, subject_key: str = "") -> None:
        if not hasattr(self, "score_preview_table"):
            return
        counts = self._subject_section_question_counts(subject_key)
        visibility = {
            5: counts.get("MCQ", 0) > 0,
            6: counts.get("TF", 0) > 0,
            7: counts.get("NUMERIC", 0) > 0,
        }
        for col, is_visible in visibility.items():
            self.score_preview_table.setColumnHidden(col, not is_visible)

    def _normalized_exam_room_mapping_by_room(self, cfg: dict) -> dict[str, set[str]]:
        normalized: dict[str, set[str]] = {}
        mapping_by_room = cfg.get("exam_room_sbd_mapping_by_room", {}) if isinstance(cfg.get("exam_room_sbd_mapping_by_room", {}), dict) else {}
        for room_key, vals in mapping_by_room.items():
            room_text = str(room_key or "").strip()
            if not room_text:
                continue
            raw_vals: list[str] = []
            if isinstance(vals, (list, tuple, set)):
                raw_vals.extend(str(x).strip() for x in vals if str(x).strip())
            elif isinstance(vals, str):
                raw_vals.extend(x.strip() for x in vals.replace(";", ",").replace("\n", ",").split(",") if x.strip())
            normalized_vals = {self._normalized_student_id_for_match(x) for x in raw_vals if x}
            if normalized_vals:
                normalized[room_text] = normalized_vals
        return normalized

    def _subject_student_room_scope(self) -> tuple[set[str], set[str]]:
        all_sids: set[str] = set()
        for s in (self.session.students if self.session else []):
            sid = str(getattr(s, "student_id", "") or "").strip()
            if sid:
                all_sids.add(self._normalized_student_id_for_match(sid))
        cfg = self._selected_batch_subject_config() or {}
        room_name = str(cfg.get("exam_room_name", "") or "").strip()
        room_name_norm = self._normalized_room_for_match(room_name)
        mapping_text = str(cfg.get("exam_room_sbd_mapping", "") or "").strip()
        mapping_by_room = self._normalized_exam_room_mapping_by_room(cfg)
        room_sids: set[str] = set()
        if mapping_by_room:
            matched_rooms = [room for room in mapping_by_room.keys() if self._normalized_room_for_match(room) == room_name_norm] if room_name else []
            if not matched_rooms and len(mapping_by_room) == 1:
                matched_rooms = [next(iter(mapping_by_room.keys()))]
            if matched_rooms:
                for room in matched_rooms:
                    room_sids.update(mapping_by_room.get(room, set()))
            elif not room_name:
                # exam_room_name trống: dùng union của toàn bộ mapping để tránh false negative.
                for values in mapping_by_room.values():
                    room_sids.update(values)
        elif mapping_text:
            chunks = [x.strip() for x in mapping_text.replace(";", ",").replace("\n", ",").split(",")]
            room_sids = {self._normalized_student_id_for_match(x) for x in chunks if x}
        elif room_name:
            for s in (self.session.students if self.session else []):
                sid = str(getattr(s, "student_id", "") or "").strip()
                extra = getattr(s, "extra", {}) or {}
                exam_room = str(extra.get("exam_room", "") or "").strip()
                if sid and exam_room and self._normalized_room_for_match(exam_room) == room_name_norm:
                    room_sids.add(self._normalized_student_id_for_match(sid))
        return all_sids, room_sids

    def _subject_room_for_student_id(self, sid: str, subject_cfg: dict | None = None) -> str:
        sid_norm = self._normalized_student_id_for_match(sid)
        if not sid_norm:
            return ""
        cfg = subject_cfg if isinstance(subject_cfg, dict) else (self._selected_batch_subject_config() or {})
        mapping_by_room = self._normalized_exam_room_mapping_by_room(cfg)
        if mapping_by_room:
            matched_rooms = [room for room, sids in mapping_by_room.items() if sid_norm in sids]
            if matched_rooms:
                preferred_room = str(cfg.get("exam_room_name", "") or "").strip()
                preferred_norm = self._normalized_room_for_match(preferred_room)
                if preferred_norm:
                    for room in matched_rooms:
                        if self._normalized_room_for_match(room) == preferred_norm:
                            return str(room or "").strip()
                picked = sorted(matched_rooms, key=lambda x: self._normalized_room_for_match(x))[0]
                return str(picked or "").strip()

        room_name = str(cfg.get("exam_room_name", "") or "").strip()
        mapping_text = str(cfg.get("exam_room_sbd_mapping", "") or "").strip()
        if room_name and mapping_text:
            chunks = [x.strip() for x in mapping_text.replace(";", ",").replace("\n", ",").split(",") if x.strip()]
            if sid_norm in {self._normalized_student_id_for_match(x) for x in chunks if x}:
                return room_name

        prof = self._student_profile_by_id(sid)
        fallback_room = str(prof.get("exam_room", "") or "")
        return fallback_room

    @staticmethod
    def _is_missing_name_for_status(name_text: str) -> bool:
        return str(name_text or "").strip() in {"", "-"}

    def _resolve_student_profile_for_status(self, student_id: str) -> dict:
        sid = str(student_id or "").strip()
        if not sid:
            return {}
        profile = self._student_profile_by_id(sid)
        return dict(profile) if isinstance(profile, dict) else {}

    def _has_valid_student_reference(self, student_id: str) -> bool:
        profile = self._resolve_student_profile_for_status(student_id)
        return bool(profile) and not self._is_missing_name_for_status(str(profile.get("name", "") or ""))

    def _is_missing_room_for_status(self, room_text: str) -> bool:
        normalized = self._normalized_room_for_match(str(room_text or ""))
        return normalized in {
            "",
            self._normalized_room_for_match("-"),
            self._normalized_room_for_match("Không rõ phòng"),
            self._normalized_room_for_match("[Không rõ phòng]"),
        }

    def _is_valid_exam_code_for_subject(self, exam_code_text: str, available_exam_codes: set[str] | None = None) -> bool:
        code = str(exam_code_text or "").strip()
        if not code or code == "-" or "?" in code:
            return False
        valid_codes = available_exam_codes if available_exam_codes is not None else self._available_exam_codes()
        if not valid_codes:
            return True
        norm_code = self._normalize_exam_code_text(code)
        return code in valid_codes or norm_code in valid_codes

    @staticmethod
    def _name_missing(name_text: str) -> bool:
        name = str(name_text or "").strip()
        return name in {"", "-"}
