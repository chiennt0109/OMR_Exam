from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.database import OMRDatabase


def main() -> int:
    parser = argparse.ArgumentParser(description="Khôi phục các file scan bị gán sai môn (subject_key).")
    parser.add_argument("--db", required=True, help="Đường dẫn file SQLite (omr.db)")
    parser.add_argument(
        "--map-csv",
        required=True,
        help="CSV mapping gồm 2 cột: source_subject_key,target_subject_key",
    )
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch, không ghi DB.")
    parser.add_argument(
        "--session-scope",
        default="",
        help="Khoá an toàn bổ sung: chỉ cho phép mapping có prefix scope này (ví dụ session_id).",
    )
    args = parser.parse_args()

    db = OMRDatabase.default(Path(args.db))
    mapping_path = Path(args.map_csv)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mapping: {mapping_path}")

    total = 0
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_subject_key", "target_subject_key"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV phải có cột source_subject_key,target_subject_key")

        for row in reader:
            source = str(row.get("source_subject_key", "") or "").strip()
            target = str(row.get("target_subject_key", "") or "").strip()
            if not source or not target:
                continue
            enforced_scope = str(args.session_scope or "").strip()
            if enforced_scope:
                src_scope = source.split("::", 1)[0] if "::" in source else ""
                dst_scope = target.split("::", 1)[0] if "::" in target else ""
                if src_scope != enforced_scope or dst_scope != enforced_scope:
                    raise ValueError(
                        f"Mapping ngoài scope cho phép: source='{source}', target='{target}', scope='{enforced_scope}'"
                    )
            if args.dry_run:
                cnt = len(db.fetch_scan_results_for_subject(source))
                print(f"[DRY-RUN] {source} -> {target}: {cnt} rows")
                total += cnt
                continue
            moved = db.reassign_scan_results_subject_key(source, target, note="recover_cli")
            print(f"[OK] {source} -> {target}: moved={moved}")
            total += moved

    print(f"Tổng số bài xử lý: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
