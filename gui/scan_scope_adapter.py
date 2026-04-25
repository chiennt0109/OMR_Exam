from __future__ import annotations

from typing import Any

from core.submission_scope import AnswerScopeService, SubmissionNormalizer


_SCOPE_SERVICE_ATTR = "_phase3_answer_scope_service"
_NORMALIZER_ATTR = "_phase3_submission_normalizer"


def _resolve_active_subject_key(window: Any, subject_key: str = "") -> str:
    if subject_key:
        return str(subject_key or "").strip()
    fn = getattr(window, "_current_batch_subject_key", None)
    if callable(fn):
        try:
            value = str(fn() or "").strip()
            if value:
                return value
        except Exception:
            pass
    return str(getattr(window, "active_batch_subject_key", "") or "").strip()


def _configured_count(window: Any, subject_key: str, section: str) -> int:
    """Return configured section count without relying on old Phase helper names."""
    fn = getattr(window, "_subject_section_question_counts", None)
    if callable(fn):
        try:
            return max(0, int((fn(subject_key) or {}).get(section, 0) or 0))
        except Exception:
            pass
    legacy = getattr(window, "_configured_question_count_for_section", None)
    if callable(legacy):
        try:
            return max(0, int(legacy(subject_key, section) or 0))
        except Exception:
            pass
    return 0


def _resolve_answer_key(window: Any, result: Any, subject_key: str) -> Any:
    fn = getattr(window, "_subject_answer_key_for_result", None)
    if callable(fn):
        try:
            return fn(result, subject_key)
        except TypeError:
            try:
                return fn(result)
            except Exception:
                return None
        except Exception:
            return None
    return None


def _copy_result(window: Any, result: Any) -> Any:
    fn = getattr(window, "_lightweight_result_copy", None)
    if callable(fn):
        try:
            return fn(result)
        except Exception:
            return result
    return result


def _ensure_scope_services(window: Any) -> tuple[AnswerScopeService, SubmissionNormalizer]:
    """Return per-window runtime services.

    Do not store them under `_answer_scope_service` or `_submission_normalizer`.
    Phase 3 installs methods with these exact names on MainWindow. Using the same names
    for instance attributes makes `getattr(window, "_submission_normalizer")` return a
    bound method before the service is created, which caused:
        AttributeError: 'function' object has no attribute 'scoped_result'
    """
    service = window.__dict__.get(_SCOPE_SERVICE_ATTR)
    if not isinstance(service, AnswerScopeService):
        service = AnswerScopeService(
            answer_key_resolver=lambda result, subject: _resolve_answer_key(window, result, subject),
            configured_count_resolver=lambda subject, section: _configured_count(window, subject, section),
        )
        setattr(window, _SCOPE_SERVICE_ATTR, service)

    normalizer = window.__dict__.get(_NORMALIZER_ATTR)
    if not isinstance(normalizer, SubmissionNormalizer):
        normalizer = SubmissionNormalizer(
            service,
            result_copy=lambda result: _copy_result(window, result),
        )
        setattr(window, _NORMALIZER_ATTR, normalizer)
    else:
        # Keep the normalizer bound to the active service after cache invalidation/recreation.
        try:
            normalizer.scope_service = service
        except Exception:
            pass
    return service, normalizer


# Compatibility wrappers imported directly by Phase 3 main_window.py.
def scan_reset_scope_cache(window: Any) -> None:
    service = window.__dict__.get(_SCOPE_SERVICE_ATTR)
    if isinstance(service, AnswerScopeService):
        try:
            service.clear()
        except Exception:
            pass


def scan_expected_questions_by_section(window: Any, result: Any, subject_key: str = "") -> dict[str, list[int]]:
    subject = _resolve_active_subject_key(window, subject_key)
    service, _normalizer = _ensure_scope_services(window)
    return service.expected(result, subject)


def scan_trim_result_answers_to_expected_scope(window: Any, result: Any, subject_key: str = "") -> Any:
    if result is None:
        return result
    subject = _resolve_active_subject_key(window, subject_key)
    _service, normalizer = _ensure_scope_services(window)
    return normalizer.scoped_result(result, subject)


def scan_compute_blank_questions(window: Any, result: Any, subject_key: str = "") -> dict[str, list[int]]:
    if result is None:
        return {"MCQ": [], "TF": [], "NUMERIC": []}
    expected = scan_expected_questions_by_section(window, result, subject_key)
    scoped = scan_trim_result_answers_to_expected_scope(window, result, subject_key)
    _service, normalizer = _ensure_scope_services(window)
    return normalizer.blank_questions(scoped, expected)


def scan_build_recognition_content_text(
    window: Any,
    result: Any,
    blank_map: dict[str, list[int]] | None = None,
    expected_by_section: dict[str, list[int]] | None = None,
    subject_key: str = "",
) -> str:
    _service, normalizer = _ensure_scope_services(window)
    blanks = blank_map if blank_map is not None else scan_compute_blank_questions(window, result, subject_key)
    expected = expected_by_section if expected_by_section is not None else scan_expected_questions_by_section(window, result, subject_key)
    return normalizer.content_text(blanks, expected)


def install_scan_scope_adapter(MainWindow):
    """Install subject-scope helpers onto MainWindow.

    All Batch Scan data presentation must flow through these helpers. Raw OMRResult may
    contain full-template answers; scoped copies contain only the questions defined by the
    selected subject/exam-code answer key or by the subject configuration fallback.
    """

    def _answer_scope_service(self) -> AnswerScopeService:
        service, _normalizer = _ensure_scope_services(self)
        return service

    def _submission_normalizer(self) -> SubmissionNormalizer:
        _service, normalizer = _ensure_scope_services(self)
        return normalizer

    def _invalidate_scope_cache(self) -> None:
        scan_reset_scope_cache(self)

    def _expected_questions_by_section(self, result, subject_key: str = "") -> dict[str, list[int]]:
        return scan_expected_questions_by_section(self, result, subject_key)

    def _trim_result_answers_to_expected_scope(self, result, subject_key: str = ""):
        return scan_trim_result_answers_to_expected_scope(self, result, subject_key)

    def _scoped_result_copy(self, result, subject_key: str = ""):
        return scan_trim_result_answers_to_expected_scope(self, result, subject_key)

    def _compute_blank_questions(self, result) -> dict[str, list[int]]:
        return scan_compute_blank_questions(self, result)

    def _build_recognition_content_text(self, result, blank_map=None, expected_by_section=None) -> str:
        return scan_build_recognition_content_text(self, result, blank_map, expected_by_section)

    def _build_answer_string_for_result(self, result, subject_key: str = "") -> str:
        scoped_result = self._scoped_result_copy(result, subject_key)
        key = self._subject_answer_key_for_result(scoped_result, subject_key)
        expected_by_section = self._expected_questions_by_section(scoped_result, subject_key)
        if key is not None and any(expected_by_section.get(sec, []) for sec in ["MCQ", "TF", "NUMERIC"]):
            def _filter_question_map(payload: object, expected_questions: list[int]) -> dict[int, Any]:
                expected_set = {int(q) for q in (expected_questions or [])}
                filtered: dict[int, Any] = {}
                for q_raw, value in dict(payload or {}).items():
                    try:
                        q_no = int(q_raw)
                    except Exception:
                        continue
                    if q_no in expected_set:
                        filtered[q_no] = value
                return filtered

            mcq_map = _filter_question_map(getattr(scoped_result, "mcq_answers", {}) or {}, expected_by_section.get("MCQ", []))
            tf_map = _filter_question_map(getattr(scoped_result, "true_false_answers", {}) or {}, expected_by_section.get("TF", []))
            num_map = _filter_question_map(getattr(scoped_result, "numeric_answers", {}) or {}, expected_by_section.get("NUMERIC", []))
            return self._answer_string_from_maps(mcq_map, tf_map, num_map, key, use_semicolon=True)

        parts: list[str] = []
        for q in expected_by_section.get("MCQ", []) or []:
            parts.append(str((getattr(scoped_result, "mcq_answers", {}) or {}).get(q, "") or "_").strip().upper()[:1] or "_")
        for q in expected_by_section.get("TF", []) or []:
            flags = (getattr(scoped_result, "true_false_answers", {}) or {}).get(q, {}) or {}
            for key_name in ["a", "b", "c", "d"]:
                parts.append("Đ" if key_name in flags and bool(flags.get(key_name)) else ("S" if key_name in flags else "_"))
        for q in expected_by_section.get("NUMERIC", []) or []:
            parts.append(str((getattr(scoped_result, "numeric_answers", {}) or {}).get(q, "") or "_").strip() or "_")
        return ";".join(parts)

    MainWindow._answer_scope_service = _answer_scope_service
    MainWindow._submission_normalizer = _submission_normalizer
    MainWindow._invalidate_scope_cache = _invalidate_scope_cache
    MainWindow._expected_questions_by_section = _expected_questions_by_section
    MainWindow._trim_result_answers_to_expected_scope = _trim_result_answers_to_expected_scope
    MainWindow._scoped_result_copy = _scoped_result_copy
    MainWindow._compute_blank_questions = _compute_blank_questions
    MainWindow._build_recognition_content_text = _build_recognition_content_text
    MainWindow._build_answer_string_for_result = _build_answer_string_for_result
