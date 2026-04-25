from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable


@dataclass
class StatusMetrics:
    total: int = 0
    visible: int = 0
    ok: int = 0
    error: int = 0
    duplicate: int = 0
    wrong_code: int = 0
    edited: int = 0
    edited_clean: int = 0
    edited_error: int = 0

    def to_dict(self) -> dict[str, int]:
        return {k: int(v or 0) for k, v in asdict(self).items()}


def _normalize_sid(value: Any) -> str:
    return str(value or "").strip()


def _has_bad_identifier(value: str) -> bool:
    text = str(value or "").strip()
    return (not text) or "?" in text or text == "-"


def aggregate_status_from_results(
    results: Iterable[Any],
    *,
    analyze_result: Callable[[Any, int], dict[str, Any]] | None = None,
    edited_resolver: Callable[[Any], bool] | None = None,
) -> dict[str, int]:
    """Aggregate Batch Scan status from canonical DB/result rows, not from stale UI state.

    The UI can still pass an analyzer to keep the exact existing status rules. This helper
    centralizes counting so status-bar logic no longer recomputes from multiple ad-hoc sources.
    """

    canonical: list[Any] = []
    seen_images: set[str] = set()
    for item in list(results or []):
        image_path = str(getattr(item, "image_path", "") or "").strip()
        image_key = image_path.lower()
        if image_key and image_key in seen_images:
            continue
        if image_key:
            seen_images.add(image_key)
        canonical.append(item)

    metrics = StatusMetrics(total=len(canonical), visible=len(canonical))
    duplicate_count: dict[str, int] = {}
    for item in canonical:
        sid = _normalize_sid(getattr(item, "student_id", ""))
        if not _has_bad_identifier(sid):
            duplicate_count[sid] = duplicate_count.get(sid, 0) + 1

    for item in canonical:
        sid = _normalize_sid(getattr(item, "student_id", ""))
        dup = 0 if _has_bad_identifier(sid) else int(duplicate_count.get(sid, 0) or 0)
        if analyze_result:
            flags = dict(analyze_result(item, dup) or {})
            has_error = bool(flags.get("has_error", False))
            is_clean_ok = bool(flags.get("is_clean_ok", False))
            has_duplicate = bool(flags.get("has_duplicate", False))
            has_wrong_code = bool(flags.get("has_wrong_code", False))
            is_edited = bool(flags.get("is_manual_edited", False))
        else:
            history = list(getattr(item, "edit_history", []) or [])
            forced = str(getattr(item, "cached_forced_status", getattr(item, "forced_status", "")) or "")
            is_edited = bool(getattr(item, "manually_edited", False) or history or forced == "Đã sửa")
            has_duplicate = dup > 1
            exam_code = str(getattr(item, "exam_code", "") or "").strip()
            has_wrong_code = bool(exam_code and "?" in exam_code)
            has_error = _has_bad_identifier(sid) or has_duplicate or has_wrong_code
            is_clean_ok = not has_error
        if edited_resolver and edited_resolver(item):
            is_edited = True
        if is_edited:
            metrics.edited += 1
            if has_error:
                metrics.edited_error += 1
            else:
                metrics.edited_clean += 1
        elif is_clean_ok:
            metrics.ok += 1
        if has_error:
            metrics.error += 1
        if has_duplicate:
            metrics.duplicate += 1
        if has_wrong_code:
            metrics.wrong_code += 1

    return metrics.to_dict()
