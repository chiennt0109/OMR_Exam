from __future__ import annotations

import json
from pathlib import Path

from models.database import OMRDatabase


def main() -> None:
    db = OMRDatabase.default()
    session_dir = Path.home() / ".omr_exam" / "sessions"
    payloads = []
    if session_dir.exists():
        for path in session_dir.glob("*.json"):
            try:
                payloads.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
    db.migrate_catalogs_from_json_payloads(payloads)
    print(f"Migrated catalogs into {db.config.path}")


if __name__ == "__main__":
    main()
