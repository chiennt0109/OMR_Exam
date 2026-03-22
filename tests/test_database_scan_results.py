from pathlib import Path

from models.database import OMRDatabase


def test_scan_results_roundtrip_without_batch_json_cache(tmp_path: Path) -> None:
    db = OMRDatabase.default(tmp_path / "omr.db")
    payload = {
        "image_path": "scan_001.png",
        "student_id": "S001",
        "exam_code": "A1",
        "mcq_answers": {1: "A", 2: "C"},
        "true_false_answers": {3: {"a": True, "b": False}},
        "numeric_answers": {4: "12,5"},
        "answer_string": "AC",
        "confidence_scores": {"sid": 0.97},
        "recognition_errors": ["warn"],
        "processing_time_sec": 0.25,
        "debug_image_path": "debug.png",
        "full_name": "Nguyen Van A",
        "birth_date": "2008-01-01",
    }

    db.upsert_scan_result("Math", payload)
    rows = db.fetch_scan_results_for_subject("Math")

    assert len(rows) == 1
    row = rows[0]
    assert row["image_path"] == payload["image_path"]
    assert row["student_id"] == payload["student_id"]
    assert row["exam_code"] == payload["exam_code"]
    assert row["mcq_answers"] == payload["mcq_answers"]
    assert row["true_false_answers"] == payload["true_false_answers"]
    assert row["numeric_answers"] == payload["numeric_answers"]
    assert row["full_name"] == payload["full_name"]
    assert row["birth_date"] == payload["birth_date"]


def test_update_scan_result_payload_replaces_row_by_subject_and_image(tmp_path: Path) -> None:
    db = OMRDatabase.default(tmp_path / "omr.db")
    initial = {
        "image_path": "scan_001.png",
        "student_id": "S001",
        "exam_code": "A1",
        "mcq_answers": {1: "A"},
        "true_false_answers": {},
        "numeric_answers": {},
        "answer_string": "A",
    }
    updated = {
        **initial,
        "student_id": "S009",
        "exam_code": "B2",
        "mcq_answers": {1: "D"},
        "numeric_answers": {2: "99"},
        "answer_string": "D99",
        "full_name": "Updated",
    }

    db.upsert_scan_result("Physics", initial)
    db.update_scan_result_payload("Physics", "scan_001.png", updated, note="unit_test")
    rows = db.fetch_scan_results_for_subject("Physics")

    assert len(rows) == 1
    assert rows[0]["student_id"] == "S009"
    assert rows[0]["exam_code"] == "B2"
    assert rows[0]["mcq_answers"] == {1: "D"}
    assert rows[0]["numeric_answers"] == {2: "99"}
    assert rows[0]["full_name"] == "Updated"
