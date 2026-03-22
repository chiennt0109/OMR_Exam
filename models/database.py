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
    """
    CREATE TABLE IF NOT EXISTS app_state (
        state_key TEXT PRIMARY KEY,
        state_value TEXT NOT NULL DEFAULT '',
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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

    @staticmethod
    def default_path(path: str | Path | None = None) -> Path:
        return Path(path or (Path.home() / ".omr_exam" / "omr_exam.db"))

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

    def set_app_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO app_state(state_key, state_value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(state_key) DO UPDATE SET
                state_value = excluded.state_value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(key), json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_app_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT state_value FROM app_state WHERE state_key = ?", (str(key),)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return default

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

    def save_exam_session(self, session_id: str, exam_name: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO exams(session_id, exam_name, payload_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                exam_name = excluded.exam_name,
                payload_json = excluded.payload_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(session_id), str(exam_name or "Kỳ thi"), json.dumps(payload, ensure_ascii=False)),
        )
        self.conn.commit()

    def fetch_exam_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT exam_name, payload_json FROM exams WHERE session_id = ?",
            (str(session_id),),
        ).fetchone()
        if not row:
            return None
        payload = json.loads(row[1] or "{}") if row[1] else {}
        if isinstance(payload, dict) and not payload.get("exam_name"):
            payload["exam_name"] = row[0]
        return payload if isinstance(payload, dict) else None

    def list_exam_sessions(self) -> list[dict[str, Any]]:
        default_id = str(self.get_app_state("default_session_id", "") or "")
        rows = self.conn.execute(
            "SELECT session_id, exam_name, payload_json FROM exams ORDER BY updated_at DESC, created_at DESC"
        ).fetchall()
        items: list[dict[str, Any]] = []
        for session_id, exam_name, payload_json in rows:
            payload = json.loads(payload_json or "{}") if payload_json else {}
            items.append(
                {
                    "session_id": str(session_id or ""),
                    "name": str((payload or {}).get("exam_name") or exam_name or "Kỳ thi"),
                    "default": str(session_id or "") == default_id,
                }
            )
        return items

    def delete_exam_session(self, session_id: str) -> None:
        self.conn.execute("DELETE FROM exams WHERE session_id = ?", (str(session_id),))
        if str(self.get_app_state("default_session_id", "") or "") == str(session_id):
            self.set_app_state("default_session_id", "")
        self.conn.commit()

    def _scan_result_db_tuple(self, subject_key: str, result_payload: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
        payload = dict(result_payload or {})
        return (
            str(subject_key or ""),
            str(payload.get("image_path", "") or ""),
            str(payload.get("exam_code", "") or ""),
            str(payload.get("student_id", "") or ""),
            str(payload.get("answer_string", payload.get("mcq_answer", "")) or ""),
            json.dumps(payload.get("true_false_answers", {}) or {}, ensure_ascii=False),
            json.dumps(payload.get("numeric_answers", {}) or {}, ensure_ascii=False),
            json.dumps(
                {
                    "mcq_answers": payload.get("mcq_answers", {}) or {},
                    "confidence_scores": payload.get("confidence_scores", {}) or {},
                    "recognition_errors": payload.get("recognition_errors", []) or [],
                    "processing_time_sec": float(payload.get("processing_time_sec", 0.0) or 0.0),
                    "debug_image_path": str(payload.get("debug_image_path", "") or ""),
                    "full_name": str(payload.get("full_name", "") or ""),
                    "birth_date": str(payload.get("birth_date", "") or ""),
                },
                ensure_ascii=False,
            ),
        )

    def upsert_scan_result(self, subject_key: str, result_payload: dict[str, Any]) -> None:
        payload_tuple = self._scan_result_db_tuple(subject_key, result_payload)
        image_path = payload_tuple[1]
        row = self.conn.execute(
            "SELECT id FROM scan_results WHERE subject_key = ? AND image_path = ? ORDER BY id DESC LIMIT 1",
            (payload_tuple[0], image_path),
        ).fetchone()
        if row:
            self.conn.execute(
                """
                UPDATE scan_results
                SET exam_code_text = ?, student_code_text = ?, mcq_answer = ?, tf_answer = ?, numeric_answer = ?, payload_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (payload_tuple[2], payload_tuple[3], payload_tuple[4], payload_tuple[5], payload_tuple[6], payload_tuple[7], int(row[0])),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO scan_results(subject_key, image_path, exam_code_text, student_code_text, mcq_answer, tf_answer, numeric_answer, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                payload_tuple,
            )
        self.conn.commit()

    def delete_scan_results_for_subject(self, subject_key: str) -> None:
        self.conn.execute("DELETE FROM scan_results WHERE subject_key = ?", (str(subject_key or ""),))
        self.conn.commit()

    def replace_scan_results_for_subject(self, subject_key: str, rows: Iterable[dict[str, Any]]) -> None:
        subject = str(subject_key or "")
        cur = self.conn.cursor()
        cur.execute("DELETE FROM scan_results WHERE subject_key = ?", (subject,))
        payload_rows = [self._scan_result_db_tuple(subject, row) for row in list(rows)]
        cur.executemany(
            """
            INSERT INTO scan_results(subject_key, image_path, exam_code_text, student_code_text, mcq_answer, tf_answer, numeric_answer, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            payload_rows,
        )
        self.conn.commit()

    def fetch_scan_results_for_subject(self, subject_key: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT image_path, exam_code_text, student_code_text, mcq_answer, tf_answer, numeric_answer, payload_json FROM scan_results WHERE subject_key = ? ORDER BY id ASC",
            (str(subject_key or ""),),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for image_path, exam_code_text, student_code_text, mcq_answer, tf_answer, numeric_answer, payload_json in rows:
            try:
                extras = json.loads(payload_json or "{}") if payload_json else {}
            except Exception:
                extras = {}
            if not isinstance(extras, dict):
                extras = {}
            try:
                tf_answers = json.loads(tf_answer or "{}") if tf_answer else {}
            except Exception:
                tf_answers = {}
            try:
                numeric_answers = json.loads(numeric_answer or "{}") if numeric_answer else {}
            except Exception:
                numeric_answers = {}
            payload = {
                "image_path": str(image_path or ""),
                "student_id": str(student_code_text or ""),
                "exam_code": str(exam_code_text or ""),
                "answer_string": str(mcq_answer or ""),
                "true_false_answers": {int(k): dict(v or {}) for k, v in (tf_answers.items() if isinstance(tf_answers, dict) else [])},
                "numeric_answers": {int(k): str(v) for k, v in (numeric_answers.items() if isinstance(numeric_answers, dict) else [])},
                "mcq_answers": {int(k): str(v) for k, v in ((extras.get("mcq_answers", {}) or {}).items() if isinstance(extras.get("mcq_answers", {}), dict) else [])},
                "confidence_scores": {str(k): float(v) for k, v in ((extras.get("confidence_scores", {}) or {}).items() if isinstance(extras.get("confidence_scores", {}), dict) else [])},
                "recognition_errors": [str(x) for x in (extras.get("recognition_errors", []) if isinstance(extras.get("recognition_errors", []), list) else [])],
                "processing_time_sec": float(extras.get("processing_time_sec", 0.0) or 0.0),
                "debug_image_path": str(extras.get("debug_image_path", "") or ""),
                "full_name": str(extras.get("full_name", "") or ""),
                "birth_date": str(extras.get("birth_date", "") or ""),
            }
            out.append(payload)
        return out

    def update_scan_result_payload(self, subject_key: str, image_path: str, payload: dict[str, Any], note: str = "") -> None:
        image_key = str(image_path or "")
        subject = str(subject_key or "")
        old_payload = next((row for row in self.fetch_scan_results_for_subject(subject) if str(row.get("image_path", "") or "") == image_key), {})
        self.upsert_scan_result(subject, payload)
        if old_payload != payload:
            self.log_change("scan_results", image_key, "payload_json", old_payload, payload, note or "scan_result_update")

    def delete_scan_result(self, subject_key: str, image_path: str) -> None:
        subject = str(subject_key or "")
        image_key = str(image_path or "")
        old_payload = next((row for row in self.fetch_scan_results_for_subject(subject) if str(row.get("image_path", "") or "") == image_key), {})
        self.conn.execute("DELETE FROM scan_results WHERE subject_key = ? AND image_path = ?", (subject, image_key))
        self.conn.commit()
        if old_payload:
            self.log_change("scan_results", image_key, "payload_json", old_payload, {}, "scan_result_delete")

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

    def fetch_scores_for_subject(self, subject_key: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload_json FROM scores WHERE subject_key = ? ORDER BY score DESC, student_code ASC",
            (str(subject_key or ""),),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for (payload_json,) in rows:
            try:
                payload = json.loads(payload_json or "{}") if payload_json else {}
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                out.append(payload)
        return out

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


def bootstrap_application_db(path: str | Path | None = None, key: str | None = None) -> Path:
    db = OMRDatabase.default(path=path, key=key)
    marker = db.get_app_state("db_bootstrap_version", 0)
    if int(marker or 0) < 1:
        db.set_app_state("db_bootstrap_version", 1)
        db.set_app_state("db_path", str(db.config.path))
    return db.config.path
