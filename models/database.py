from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from pysqlcipher3 import dbapi2 as sqlcipher_sqlite  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sqlcipher_sqlite = None


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS blocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exam_name TEXT NOT NULL,
        session_id TEXT UNIQUE,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT NOT NULL UNIQUE,
        student_name TEXT NOT NULL DEFAULT '',
        class_name TEXT NOT NULL DEFAULT '',
        extra_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS answer_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_key TEXT NOT NULL,
        exam_code TEXT NOT NULL,
        answers_json TEXT NOT NULL DEFAULT '{}',
        true_false_json TEXT NOT NULL DEFAULT '{}',
        numeric_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(subject_key, exam_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scan_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_key TEXT NOT NULL DEFAULT '',
        exam_id INTEGER,
        student_id INTEGER,
        image_path TEXT NOT NULL DEFAULT '',
        exam_code_text TEXT NOT NULL DEFAULT '',
        student_code_text TEXT NOT NULL DEFAULT '',
        mcq_answer TEXT NOT NULL DEFAULT '',
        tf_answer TEXT NOT NULL DEFAULT '',
        numeric_answer TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_key TEXT NOT NULL,
        student_code TEXT NOT NULL,
        exam_code TEXT NOT NULL DEFAULT '',
        score REAL NOT NULL DEFAULT 0,
        correct_count INTEGER NOT NULL DEFAULT 0,
        wrong_count INTEGER NOT NULL DEFAULT 0,
        blank_count INTEGER NOT NULL DEFAULT 0,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(subject_key, student_code, exam_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL DEFAULT '',
        field_name TEXT NOT NULL DEFAULT '',
        old_value TEXT NOT NULL DEFAULT '',
        new_value TEXT NOT NULL DEFAULT '',
        note TEXT NOT NULL DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_subjects_name ON subjects(name)",
    "CREATE INDEX IF NOT EXISTS idx_blocks_name ON blocks(name)",
    "CREATE INDEX IF NOT EXISTS idx_exams_session_id ON exams(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_students_code ON students(student_code)",
    "CREATE INDEX IF NOT EXISTS idx_scan_results_subject ON scan_results(subject_key)",
    "CREATE INDEX IF NOT EXISTS idx_scan_results_student ON scan_results(student_code_text)",
    "CREATE INDEX IF NOT EXISTS idx_scan_results_exam_code ON scan_results(exam_code_text)",
    "CREATE INDEX IF NOT EXISTS idx_scores_subject_student ON scores(subject_key, student_code)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_entity ON audit_logs(entity_type, entity_id)",
]


@dataclass
class DatabaseConfig:
    path: Path
    key: str = ""


class OMRDatabase:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = self._connect(config.path, config.key)
        self._apply_pragmas(config.key)
        self._ensure_schema()

    @staticmethod
    def default(path: str | Path | None = None, key: str | None = None) -> "OMRDatabase":
        db_path = Path(path or (Path.home() / ".omr_exam" / "omr_exam.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return OMRDatabase(DatabaseConfig(path=db_path, key=str(key or os.environ.get("OMR_DB_KEY", ""))))

    def _connect(self, path: Path, key: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        if sqlcipher_sqlite is not None:
            conn = sqlcipher_sqlite.connect(str(path))
            if key:
                conn.execute(f"PRAGMA key = '{key.replace("'", "''")}'")
            return conn
        return sqlite3.connect(str(path))

    def _apply_pragmas(self, key: str) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        if key and sqlcipher_sqlite is None:
            # Compatibility hint when SQLCipher is not installed.
            cur.execute("PRAGMA user_version=1")
        self.conn.commit()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        for stmt in INDEXES:
            cur.execute(stmt)
        self.conn.commit()

    def fetch_catalog(self, table_name: str) -> list[str]:
        if table_name not in {"subjects", "blocks"}:
            return []
        cur = self.conn.cursor()
        rows = cur.execute(f"SELECT name FROM {table_name} ORDER BY name COLLATE NOCASE").fetchall()
        return [str(row[0]) for row in rows]

    def replace_catalog(self, table_name: str, values: Iterable[str]) -> None:
        if table_name not in {"subjects", "blocks"}:
            return
        normalized = [str(v).strip() for v in values if str(v).strip()]
        cur = self.conn.cursor()
        cur.execute(f"DELETE FROM {table_name}")
        cur.executemany(f"INSERT INTO {table_name}(name) VALUES (?)", [(v,) for v in normalized])
        self.conn.commit()

    def log_change(self, entity_type: str, entity_id: str, field_name: str, old_value: Any, new_value: Any, note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO audit_logs(entity_type, entity_id, field_name, old_value, new_value, note) VALUES (?, ?, ?, ?, ?, ?)",
            (entity_type, str(entity_id), field_name, str(old_value), str(new_value), note),
        )
        self.conn.commit()

    def migrate_catalogs_from_json_payloads(self, payloads: Iterable[dict[str, Any]]) -> None:
        subjects: set[str] = set(self.fetch_catalog("subjects"))
        blocks: set[str] = set(self.fetch_catalog("blocks"))
        for payload in payloads:
            cfg = (payload or {}).get("config", {}) if isinstance(payload, dict) else {}
            for name in cfg.get("subject_catalog", []) if isinstance(cfg.get("subject_catalog", []), list) else []:
                if str(name).strip():
                    subjects.add(str(name).strip())
            for name in cfg.get("block_catalog", []) if isinstance(cfg.get("block_catalog", []), list) else []:
                if str(name).strip():
                    blocks.add(str(name).strip())
        if subjects:
            self.replace_catalog("subjects", sorted(subjects, key=str.casefold))
        if blocks:
            self.replace_catalog("blocks", sorted(blocks, key=str.casefold))

    def upsert_scan_result(self, subject_key: str, result_payload: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO scan_results(subject_key, image_path, exam_code_text, student_code_text, mcq_answer, tf_answer, numeric_answer, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                str(subject_key or ""),
                str(result_payload.get("image_path", "") or ""),
                str(result_payload.get("exam_code", "") or ""),
                str(result_payload.get("student_id", "") or ""),
                str(result_payload.get("mcq_answer", "") or ""),
                str(result_payload.get("tf_answer", "") or ""),
                str(result_payload.get("numeric_answer", "") or ""),
                json.dumps(result_payload, ensure_ascii=False),
            ),
        )
        self.conn.commit()
