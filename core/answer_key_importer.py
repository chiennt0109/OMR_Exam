from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


MCQ_CHOICES = {"A", "B", "C", "D", "E"}
TF_TRUE_VALUES = {"T", "TRUE", "D", "Đ", "1"}
TF_FALSE_VALUES = {"F", "FALSE", "S", "0"}


@dataclass
class ImportedAnswerKey:
    exam_id: int = 1
    mcq_answers: dict[int, str] = field(default_factory=dict)
    true_false_answers: dict[int, dict[str, bool]] = field(default_factory=dict)
    numeric_answers: dict[int, str] = field(default_factory=dict)


@dataclass
class ImportedAnswerKeyPackage:
    exam_keys: dict[str, ImportedAnswerKey] = field(default_factory=dict)


def _read_file(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".xlsx":
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")
    if ext == ".csv":
        return pd.read_csv(path)
    raise ImportError(f"Unsupported file type '{ext}'. Use .xlsx or .csv")


def _read_excel_multilevel(file_path: str | Path) -> pd.DataFrame | None:
    path = Path(file_path)
    if path.suffix.lower() != ".xlsx":
        return None
    try:
        multi = pd.read_excel(path, sheet_name=0, engine="openpyxl", header=[0, 1])
    except Exception:
        return None
    if not isinstance(multi.columns, pd.MultiIndex):
        return None
    first = [str(c[0]).strip().lower() for c in multi.columns]
    second = [str(c[1]).strip() for c in multi.columns]
    if not any("mã đề" in val or "ma de" in val for val in first):
        return None
    if not any(val and not val.lower().startswith("unnamed") for val in second):
        return None
    return multi


def _parse_tf_token(df_row_idx: int, exam_code: str, token: str) -> dict[str, bool]:
    raw = token.strip().upper().replace(" ", "")
    if len(raw) != 4:
        raise ImportError(
            f"Row {df_row_idx + 2}, exam '{exam_code}': invalid TF '{token}'. Expected 4 chars (T/F or Đ/S)."
        )
    payload: dict[str, bool] = {}
    for idx, ch in enumerate(raw):
        key = chr(ord("a") + idx)
        if ch in TF_TRUE_VALUES:
            payload[key] = True
        elif ch in TF_FALSE_VALUES:
            payload[key] = False
        else:
            raise ImportError(
                f"Row {df_row_idx + 2}, exam '{exam_code}': invalid TF character '{ch}'. Expected T/F or Đ/S."
            )
    return payload




def _is_numeric_token(value: str) -> bool:
    token = value.strip().replace(" ", "")
    if not token:
        return False
    if token[0] in "+-":
        token = token[1:]
    if not token:
        return False
    token = token.replace(",", ".")
    if token.count(".") > 1:
        return False
    return token.replace(".", "").isdigit()

def _ensure_question(df_row_idx: int, value: Any) -> int:
    try:
        q = int(str(value).strip())
        if q <= 0:
            raise ValueError
        return q
    except Exception as exc:
        raise ImportError(
            f"Row {df_row_idx + 2}: invalid question '{value}'. Expected positive integer."
        ) from exc


def _parse_single_exam_table(df: pd.DataFrame) -> ImportedAnswerKeyPackage:
    normalized = [str(c).strip().lower() for c in df.columns]
    if "question" not in normalized:
        raise ImportError("Missing 'Question' column.")
    q_idx = normalized.index("question")
    answer_idx = normalized.index("answer") if "answer" in normalized else -1
    option_map = {c.upper(): i for i, c in enumerate(normalized) if c in {"a", "b", "c", "d", "e"}}

    result = ImportedAnswerKey()
    for row_idx, row in df.iterrows():
        q_val = row.iloc[q_idx]
        if pd.isna(q_val) or str(q_val).strip() == "":
            continue
        q = _ensure_question(row_idx, q_val)

        if answer_idx >= 0:
            raw = str(row.iloc[answer_idx]).strip()
            upper = raw.upper()
            if upper in MCQ_CHOICES:
                result.mcq_answers[q] = upper
            elif _is_numeric_token(raw):
                result.numeric_answers[q] = raw
            else:
                raise ImportError(
                    f"Row {row_idx + 2}: invalid value '{raw}'. Expected A/B/C/D/E, T/F(4 chars), or numeric."
                )
            continue

        if not option_map:
            raise ImportError("Expected 'Answer' column or A/B/C/D/E columns.")

        vals = {k: ("" if pd.isna(row.iloc[i]) else str(row.iloc[i]).strip()) for k, i in option_map.items()}
        marked = [k for k, v in vals.items() if v]
        if len(marked) == 1 and vals[marked[0]].strip().upper() in {"X", "1", "✓", "*", "T", "TRUE"}:
            result.mcq_answers[q] = marked[0]
            continue

        tf_payload: dict[str, bool] = {}
        for choice in sorted(option_map):
            token = vals[choice].upper()
            if token in TF_TRUE_VALUES:
                tf_payload[choice.lower()] = True
            elif token in TF_FALSE_VALUES:
                tf_payload[choice.lower()] = False
            else:
                raise ImportError(
                    f"Row {row_idx + 2}, column '{choice}': invalid value '{vals[choice]}'. Expected T/F or Đ/S."
                )
        result.true_false_answers[q] = tf_payload

    return ImportedAnswerKeyPackage(exam_keys={"DEFAULT": result})


def _parse_exam_matrix(df: pd.DataFrame) -> ImportedAnswerKeyPackage:
    cols = list(df.columns)
    if len(cols) < 2:
        raise ImportError("Answer key matrix must include question column and exam code columns.")

    q_col = cols[0]
    exam_cols = cols[1:]
    package = ImportedAnswerKeyPackage(exam_keys={str(code).strip(): ImportedAnswerKey() for code in exam_cols})

    for row_idx, row in df.iterrows():
        q_val = row[q_col]
        if pd.isna(q_val) or str(q_val).strip() == "":
            continue
        q = _ensure_question(row_idx, q_val)

        values: dict[str, str] = {}
        for exam_code in exam_cols:
            val = row[exam_code]
            text = "" if pd.isna(val) else str(val).strip()
            values[str(exam_code).strip()] = text

        non_empty = [v for v in values.values() if v != ""]
        if not non_empty:
            continue

        if all(v.upper() in MCQ_CHOICES for v in non_empty):
            for exam_code, v in values.items():
                if v:
                    package.exam_keys[exam_code].mcq_answers[q] = v.upper()
            continue

        if all(len(v.replace(" ", "")) == 4 for v in non_empty):
            for exam_code, v in values.items():
                if v:
                    package.exam_keys[exam_code].true_false_answers[q] = _parse_tf_token(row_idx, exam_code, v)
            continue

        if all(_is_numeric_token(v) for v in non_empty):
            for exam_code, v in values.items():
                if v:
                    package.exam_keys[exam_code].numeric_answers[q] = v
            continue

        first_invalid = next((v for v in non_empty if not (v.upper() in MCQ_CHOICES or len(v.replace(" ", "")) == 4 or _is_numeric_token(v))), non_empty[0])
        raise ImportError(
            f"Row {row_idx + 2}: invalid value '{first_invalid}'. Expected MCQ(A-E), TF(4 chars T/F or Đ/S), or numeric."
        )

    return package


def import_answer_key(file_path: str | Path, exam_id: int = 1) -> ImportedAnswerKeyPackage:
    multi = _read_excel_multilevel(file_path)
    if multi is not None:
        first_col = multi.columns[0]
        data = pd.DataFrame({"Question": multi[first_col]})
        for col in multi.columns[1:]:
            exam_code = str(col[1]).strip()
            if not exam_code or exam_code.lower().startswith("unnamed"):
                continue
            data[exam_code] = multi[col]
        package = _parse_exam_matrix(data)
    else:
        df = _read_file(file_path)
        if df.empty:
            raise ImportError("Input file is empty.")
        normalized = [str(c).strip().lower() for c in df.columns]
        if len(df.columns) >= 3 and "question" in normalized and "answer" not in normalized:
            package = _parse_exam_matrix(df)
        else:
            package = _parse_single_exam_table(df)

    for key in package.exam_keys.values():
        key.exam_id = exam_id
    return package
