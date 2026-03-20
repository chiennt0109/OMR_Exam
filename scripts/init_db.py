from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.database import bootstrap_application_db


def main() -> None:
    db_path = bootstrap_application_db()
    print(f"Initialized application database at: {db_path}")


if __name__ == "__main__":
    main()
