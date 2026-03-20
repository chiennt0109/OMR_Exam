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

    def replace_answer_keys_for_subject(self, subject_key: str, answer_keys: dict[str, dict[str, Any]]) -> None:
        subject = str(subject_key or "").strip()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM answer_keys WHERE subject_key = ?", (subject,))
        rows = []
        for exam_code, payload in (answer_keys or {}).items():
            if not isinstance(payload, dict):
                continue
            rows.append(
                (
                    subject,
                    str(exam_code or "").strip(),
                    json.dumps(payload.get("mcq_answers", {}) or {}, ensure_ascii=False),
                    json.dumps(payload.get("true_false_answers", {}) or {}, ensure_ascii=False),
                    json.dumps(
                        {
                            "numeric_answers": payload.get("numeric_answers", {}) or {},
                            "full_credit_questions": payload.get("full_credit_questions", {}) or {},
                            "invalid_answer_rows": payload.get("invalid_answer_rows", {}) or {},
                        },
                        ensure_ascii=False,
                    ),
                )
            )
        if rows:
            cur.executemany(
                """
                INSERT INTO answer_keys(subject_key, exam_code, answers_json, true_false_json, numeric_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        self.conn.commit()

    def fetch_answer_keys_for_subject(self, subject_key: str) -> dict[str, dict[str, Any]]:
        subject = str(subject_key or "").strip()
        rows = self.conn.execute(
            "SELECT exam_code, answers_json, true_false_json, numeric_json FROM answer_keys WHERE subject_key = ? ORDER BY exam_code COLLATE NOCASE",
            (subject,),
        ).fetchall()
        payload: dict[str, dict[str, Any]] = {}
        for exam_code, answers_json, true_false_json, numeric_json in rows:
            numeric_payload = json.loads(numeric_json or "{}") if numeric_json else {}
            payload[str(exam_code)] = {
                "mcq_answers": json.loads(answers_json or "{}") if answers_json else {},
                "true_false_answers": json.loads(true_false_json or "{}") if true_false_json else {},
                "numeric_answers": (numeric_payload or {}).get("numeric_answers", {}),
                "full_credit_questions": (numeric_payload or {}).get("full_credit_questions", {}),
                "invalid_answer_rows": (numeric_payload or {}).get("invalid_answer_rows", {}),
            }
        return payload

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

    def replace_scan_results_for_subject(self, subject_key: str, rows: Iterable[dict[str, Any]]) -> None:
        subject = str(subject_key or "")
        cur = self.conn.cursor()
        cur.execute("DELETE FROM scan_results WHERE subject_key = ?", (subject,))
        payload_rows = list(rows)
        cur.executemany(
            """
            INSERT INTO scan_results(subject_key, image_path, exam_code_text, student_code_text, mcq_answer, tf_answer, numeric_answer, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                (
                    subject,
                    str(row.get("image_path", "") or ""),
                    str(row.get("exam_code", "") or ""),
                    str(row.get("student_id", "") or ""),
                    json.dumps(row.get("mcq_answers", {}), ensure_ascii=False),
                    json.dumps(row.get("true_false_answers", {}), ensure_ascii=False),
                    json.dumps(row.get("numeric_answers", {}), ensure_ascii=False),
                    json.dumps(row, ensure_ascii=False),
                )
                for row in payload_rows
            ],
        )
        self.conn.commit()

    def update_scan_result_payload(self, image_path: str, payload: dict[str, Any], note: str = "") -> None:
        image_key = str(image_path or "")
        old_row = self.conn.execute(
            "SELECT payload_json FROM scan_results WHERE image_path = ? ORDER BY id DESC LIMIT 1",
            (image_key,),
        ).fetchone()
        old_payload = json.loads(old_row[0]) if old_row and old_row[0] else {}
        self.conn.execute(
            """
            UPDATE scan_results
            SET exam_code_text = ?, student_code_text = ?, mcq_answer = ?, tf_answer = ?, numeric_answer = ?, payload_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE image_path = ?
            """,
            (
                str(payload.get("exam_code", "") or ""),
                str(payload.get("student_id", "") or ""),
                json.dumps(payload.get("mcq_answers", {}), ensure_ascii=False),
                json.dumps(payload.get("true_false_answers", {}), ensure_ascii=False),
                json.dumps(payload.get("numeric_answers", {}), ensure_ascii=False),
                json.dumps(payload, ensure_ascii=False),
                image_key,
            ),
        )
        self.conn.commit()
        if old_payload != payload:
            self.log_change("scan_results", image_key, "payload_json", old_payload, payload, note or "scan_result_update")

    def upsert_score_row(self, subject_key: str, student_code: str, exam_code: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO scores(subject_key, student_code, exam_code, score, correct_count, wrong_count, blank_count, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(subject_key, student_code, exam_code) DO UPDATE SET
                score = excluded.score,
                correct_count = excluded.correct_count,
                wrong_count = excluded.wrong_count,
                blank_count = excluded.blank_count,
                payload_json = excluded.payload_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                str(subject_key or ""),
                str(student_code or ""),
                str(exam_code or ""),
                float(payload.get("score", 0) or 0),
                int(payload.get("correct", 0) or 0),
                int(payload.get("wrong", 0) or 0),
                int(payload.get("blank", 0) or 0),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def dashboard_summary(self, subject_key: str = "") -> dict[str, Any]:
        params: tuple[Any, ...] = ()
        where = ""
        if str(subject_key or "").strip():
            where = "WHERE subject_key = ?"
            params = (str(subject_key).strip(),)
        cur = self.conn.cursor()
        avg_score = cur.execute(f"SELECT AVG(score) FROM scores {where}", params).fetchone()[0] or 0
        top_rows = cur.execute(
            f"SELECT student_code, score FROM scores {where} ORDER BY score DESC, student_code ASC LIMIT 10",
            params,
        ).fetchall()
        distribution = cur.execute(
            f"SELECT CAST(score AS INT) AS bucket, COUNT(*) FROM scores {where} GROUP BY bucket ORDER BY bucket",
            params,
        ).fetchall()
        return {
            "average_score": float(avg_score or 0),
            "top_students": [{"student_code": str(r[0]), "score": float(r[1] or 0)} for r in top_rows],
            "distribution": [{"bucket": int(r[0] or 0), "count": int(r[1] or 0)} for r in distribution],
        }
