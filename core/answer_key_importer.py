from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


MCQ_CHOICES = {"A", "B", "C", "D", "E"}
MCQ_MARKERS = {"X", "1", "*", "✓", "TRUE", "T"}
TF_TRUE_VALUES = {"T", "TRUE", "D", "Đ", "1"}
TF_FALSE_VALUES = {"F", "FALSE", "S", "0"}


@dataclass
class ImportedAnswerKey:
    exam_id: int = 1
    mcq_answers: dict[int, str] = field(default_factory=dict)
    true_false_answers: dict[int, dict[str, bool]] = field(default_factory=dict)
    numeric_answers: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exam_id": self.exam_id,
            "mcq_answers": self.mcq_answers,
            "true_false_answers": self.true_false_answers,
            "numeric_answers": self.numeric_answers,
        }


def _normalize_columns(columns: list[Any]) -> list[str]:
    return [str(c).strip().lower() for c in columns]


def _read_file(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".xlsx":
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")
    if ext == ".csv":
        return pd.read_csv(path)
    raise ImportError(f"Unsupported file type '{ext}'. Use .xlsx or .csv")


def _question_from_row(df_row_idx: int, value: Any) -> int:
    try:
        q = int(str(value).strip())
        if q <= 0:
            raise ValueError
        return q
    except Exception as exc:
        raise ImportError(
            f"Row {df_row_idx + 2}: invalid question '{value}'. Expected positive integer."
        ) from exc


def _parse_tf_value(df_row_idx: int, column: str, value: Any) -> bool:
    val = str(value).strip().upper()
    if val in TF_TRUE_VALUES:
        return True
    if val in TF_FALSE_VALUES:
        return False
    raise ImportError(
        f"Row {df_row_idx + 2}, column '{column}': invalid value '{value}'. "
        "Expected T/F, Đ/S, D/S, True/False."
    )


def import_answer_key(file_path: str | Path, exam_id: int = 1) -> ImportedAnswerKey:
    df = _read_file(file_path)
    if df.empty:
        raise ImportError("Input file is empty.")

    normalized = _normalize_columns(list(df.columns))
    question_idx = normalized.index("question") if "question" in normalized else -1
    if question_idx < 0:
        raise ImportError("Missing 'Question' column.")

    result = ImportedAnswerKey(exam_id=exam_id)

    answer_idx = normalized.index("answer") if "answer" in normalized else -1
    option_map = {c.upper(): i for i, c in enumerate(normalized) if c in {"a", "b", "c", "d", "e"}}

    if answer_idx >= 0:
        for row_idx, row in df.iterrows():
            q = _question_from_row(row_idx, row.iloc[question_idx])
            raw = str(row.iloc[answer_idx]).strip()
            value = raw.upper()

            if value in MCQ_CHOICES:
                result.mcq_answers[q] = value
            elif raw.isdigit():
                result.numeric_answers[q] = raw
            else:
                raise ImportError(
                    f"Row {row_idx + 2}: invalid answer '{raw}'. "
                    "Expected one of A/B/C/D/E or digits only."
                )
        return result

    if not option_map:
        raise ImportError("Could not detect answer columns. Expected 'Answer' or option columns A-E.")

    ordered_choices = sorted(option_map.keys())
    for row_idx, row in df.iterrows():
        q = _question_from_row(row_idx, row.iloc[question_idx])
        row_values: dict[str, str] = {}
        for choice in ordered_choices:
            v = row.iloc[option_map[choice]]
            text = "" if pd.isna(v) else str(v).strip()
            row_values[choice] = text

        non_empty = [c for c, v in row_values.items() if v != ""]
        upper_non_empty = [row_values[c].upper() for c in non_empty]

        if upper_non_empty and all(v in MCQ_MARKERS for v in upper_non_empty):
            if len(non_empty) != 1:
                raise ImportError(
                    f"Row {row_idx + 2}: invalid MCQ marks {non_empty}. Exactly one option must be marked."
                )
            result.mcq_answers[q] = non_empty[0]
            continue

        tf_payload: dict[str, bool] = {}
        for choice in ordered_choices:
            cell_value = row_values[choice]
            if cell_value == "":
                raise ImportError(
                    f"Row {row_idx + 2}, column '{choice}': empty value. "
                    "Expected T/F, Đ/S, D/S, True/False."
                )
            tf_payload[choice.lower()] = _parse_tf_value(row_idx, choice, cell_value)
        result.true_false_answers[q] = tf_payload

    return result
