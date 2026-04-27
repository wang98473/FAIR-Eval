from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HLE_DATASET_DESCRIPTION = (
    "### DATASET: Humanity's Last Exam (HLE), ### PUBLISH TIME: 2025, "
    "### CONTENT: expert-level closed-ended academic benchmark at the frontier "
    "of human knowledge, spanning broad subjects with highly challenging questions "
    "that require deep reasoning and cannot be quickly answered by simple internet retrieval."
)

RAW_SUBSET_COLUMNS = [
    "id",
    "question",
    "rationale",
    "rationale_prompt",
]

FEATURE_COLUMNS = [
    "q_len",
    "q_sent_count",
    "q_negation_count",
    "q_conditional_count",
    "q_avg_word_len",
    "q_subordinate_clause_count",
    "answer_len",
    "answer_sent_count",
    "answer_connective_density",
    "answer_clause_count",
    "answer_entity_count",
    "answer_hedge_ratio",
    "answer_lexical_overlap",
    "answer_new_info_ratio",
    "task_type",
]

FINAL_ITEM_COLUMNS = [
    "scenario",
    "question",
    "rationale",
    "rationale_prompt",
    "question_embedding",
    "rationale_embedding",
    *FEATURE_COLUMNS,
]

TASK_TYPE_MAP = {
    "factual": 0,
    "reasoning": 1,
    "calculation": 2,
    "judgment": 3,
    "metacognitive": 4,
}


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for an output path when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write dictionaries to a JSONL file."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when a DataFrame misses required columns."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def count_empty_cells(df: pd.DataFrame) -> dict[str, int]:
    """Count null cells and blank-string cells column by column."""
    counts: dict[str, int] = {}

    for column in df.columns:
        series = df[column]
        null_count = int(series.isna().sum())
        blank_count = 0

        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            blank_mask = series.fillna("").astype(str).str.strip().eq("")
            blank_count = int(blank_mask.sum()) - null_count

        total = null_count + max(blank_count, 0)
        if total:
            counts[column] = total

    return counts


def print_empty_cell_summary(df: pd.DataFrame, label: str) -> None:
    """Print a compact empty-cell summary for a DataFrame."""
    empty_counts = count_empty_cells(df)
    print(f"{label} has empty cells: {bool(empty_counts)}")
    if empty_counts:
        print("Empty-cell counts by column:")
        for column, count in empty_counts.items():
            print(f"  {column}: {count}")
