from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

try:
    from core.omr_engine import OMRResult
except Exception:  # pragma: no cover - keeps the service importable in unit tests
    OMRResult = Any  # type: ignore


QUESTION_SECTIONS = ("MCQ", "TF", "NUMERIC")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_exam_code_text(value: Any) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.lstrip("0") or digits or text


def _sorted_int_keys(mapping: Any) -> list[int]:
    out: list[int] = []
    for key in (mapping or {}).keys() if isinstance(mapping, dict) else []:
        try:
            out.append(int(key))
        except Exception:
            continue
    return sorted(set(out))


def _expected_from_key(answer_key: Any, section: str) -> list[int]:
    invalid_rows = getattr(answer_key, "invalid_answer_rows", {}) or {}
    if section == "MCQ":
        valid = getattr(answer_key, "answers", {}) or {}
    elif section == "TF":
        valid = getattr(answer_key, "true_false_answers", {}) or {}
    else:
        valid = getattr(answer_key, "numeric_answers", {}) or {}
    invalid = (invalid_rows.get(section, {}) or {}) if isinstance(invalid_rows, dict) else {}
    nums = set(_sorted_int_keys(valid)) | set(_sorted_int_keys(invalid))
    return sorted(nums)


def _filter_map_by_expected(mapping: Any, expected: Iterable[int]) -> dict[int, Any]:
    allowed = {int(x) for x in (expected or [])}
    if not allowed:
        return {}
    out: dict[int, Any] = {}
    for raw_key, value in dict(mapping or {}).items():
        try:
            q_no = int(raw_key)
        except Exception:
            continue
        if q_no in allowed:
            out[q_no] = value
    return out


@dataclass(frozen=True)
class ScopedSubmission:
    """Canonical, subject-scoped representation used by grid, edit, scoring and export.

    raw_result is the original decoded OMRResult. scoped_result is a lightweight copy whose
    answer maps are trimmed to the active subject/exam-code answer scope. UI code should never
    build the Batch Scan content/status/edit payload from the raw full-template maps.
    """

    subject_key: str
    exam_code: str
    raw_result: Any
    scoped_result: Any
    expected: dict[str, list[int]] = field(default_factory=dict)
    blanks: dict[str, list[int]] = field(default_factory=dict)
    answer_string: str = ""


class AnswerScopeService:
    """Resolve the expected question set for one subject/exam code.

    This class is deliberately UI-agnostic. The caller provides an answer-key resolver and,
    optionally, a configured-count resolver as fallback for cases where a key is not available.
    """

    def __init__(
        self,
        answer_key_resolver: Callable[[Any, str], Any] | None = None,
        configured_count_resolver: Callable[[str, str], int] | None = None,
    ) -> None:
        self.answer_key_resolver = answer_key_resolver
        self.configured_count_resolver = configured_count_resolver
        self._cache: dict[tuple[str, str, str], dict[str, list[int]]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def expected(self, result: Any, subject_key: str = "") -> dict[str, list[int]]:
        subject = str(subject_key or "").strip()
        exam_code = str(getattr(result, "exam_code", "") or "").strip()
        normalized_code = _normalize_exam_code_text(exam_code)
        cache_key = (subject, exam_code, normalized_code)
        if cache_key in self._cache:
            return {sec: list(vals) for sec, vals in self._cache[cache_key].items()}

        answer_key = self.answer_key_resolver(result, subject) if self.answer_key_resolver else None
        expected: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        if answer_key is not None:
            expected = {sec: _expected_from_key(answer_key, sec) for sec in QUESTION_SECTIONS}

        if not any(expected.values()) and self.configured_count_resolver:
            for sec in QUESTION_SECTIONS:
                count = max(0, _as_int(self.configured_count_resolver(subject, sec), 0))
                expected[sec] = list(range(1, count + 1)) if count else []

        self._cache[cache_key] = {sec: list(vals) for sec, vals in expected.items()}
        return expected


class SubmissionNormalizer:
    """Build the single canonical ScopedSubmission used by all subject-aware flows."""

    def __init__(
        self,
        scope_service: AnswerScopeService,
        result_copy: Callable[[Any], Any] | None = None,
        answer_string_builder: Callable[[Any, str], str] | None = None,
    ) -> None:
        self.scope_service = scope_service
        self.result_copy = result_copy
        self.answer_string_builder = answer_string_builder

    def scoped_result(self, result: Any, subject_key: str = "") -> Any:
        if result is None:
            return result
        scoped = self.result_copy(result) if self.result_copy else result
        expected = self.scope_service.expected(scoped, subject_key)
        if hasattr(scoped, "mcq_answers"):
            scoped.mcq_answers = _filter_map_by_expected(getattr(scoped, "mcq_answers", {}) or {}, expected.get("MCQ", []))
        if hasattr(scoped, "true_false_answers"):
            scoped.true_false_answers = _filter_map_by_expected(getattr(scoped, "true_false_answers", {}) or {}, expected.get("TF", []))
        if hasattr(scoped, "numeric_answers"):
            scoped.numeric_answers = _filter_map_by_expected(getattr(scoped, "numeric_answers", {}) or {}, expected.get("NUMERIC", []))
        if hasattr(scoped, "sync_legacy_aliases"):
            try:
                scoped.sync_legacy_aliases()
            except Exception:
                pass
        return scoped

    def blank_questions(self, result: Any, expected: dict[str, list[int]] | None = None) -> dict[str, list[int]]:
        expected = expected or self.scope_service.expected(result, "")
        out: dict[str, list[int]] = {"MCQ": [], "TF": [], "NUMERIC": []}
        mcq_answers = getattr(result, "mcq_answers", {}) or {}
        tf_answers = getattr(result, "true_false_answers", {}) or {}
        numeric_answers = getattr(result, "numeric_answers", {}) or {}

        for q_no in expected.get("MCQ", []) or []:
            if not str((mcq_answers or {}).get(int(q_no), "") or "").strip():
                out["MCQ"].append(int(q_no))
        for q_no in expected.get("TF", []) or []:
            flags = (tf_answers or {}).get(int(q_no), {}) or {}
            if not isinstance(flags, dict) or not any(k in flags for k in ("a", "b", "c", "d")):
                out["TF"].append(int(q_no))
        for q_no in expected.get("NUMERIC", []) or []:
            if not str((numeric_answers or {}).get(int(q_no), "") or "").strip():
                out["NUMERIC"].append(int(q_no))
        return out

    def content_text(self, blanks: dict[str, list[int]], expected: dict[str, list[int]] | None = None) -> str:
        parts: list[str] = []
        labels = {"MCQ": "MCQ", "TF": "TF", "NUMERIC": "NUM"}
        for section in QUESTION_SECTIONS:
            values = [int(x) for x in (blanks or {}).get(section, []) if str(x).strip().lstrip("-").isdigit()]
            if values:
                parts.append(f"{labels[section]} chưa tô: " + ", ".join(str(x) for x in sorted(set(values))))
        return " | ".join(parts)

    def normalize(self, result: Any, subject_key: str = "") -> ScopedSubmission:
        scoped = self.scoped_result(result, subject_key)
        expected = self.scope_service.expected(scoped, subject_key)
        blanks = self.blank_questions(scoped, expected)
        answer_string = ""
        if self.answer_string_builder:
            try:
                answer_string = str(self.answer_string_builder(scoped, subject_key) or "")
            except Exception:
                answer_string = ""
        return ScopedSubmission(
            subject_key=str(subject_key or ""),
            exam_code=str(getattr(scoped, "exam_code", "") or ""),
            raw_result=result,
            scoped_result=scoped,
            expected=expected,
            blanks=blanks,
            answer_string=answer_string,
        )
