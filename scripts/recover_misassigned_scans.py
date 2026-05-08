from __future__ import annotations

import argparse
import csv
from pathlib import Path

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

