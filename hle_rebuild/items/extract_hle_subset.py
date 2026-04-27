from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from hle_rebuild.common import RAW_SUBSET_COLUMNS, count_empty_cells, load_jsonl, require_columns


def load_subset_ids(path: Path) -> list[str]:
    """Load target IDs from a JSONL file produced by extract_subset_ids.py."""
    raw_records = load_jsonl(path)
    subset_ids: list[str] = []

    for index, record in enumerate(raw_records, start=1):
        item_id = record.get("id")
        if item_id is None or str(item_id).strip() == "":
            raise ValueError(f"Missing non-empty 'id' in subset ID record {index}")
        subset_ids.append(str(item_id))

    if len(subset_ids) != len(set(subset_ids)):
        raise ValueError("Subset ID file contains duplicate IDs")

    return subset_ids


def extract_subset(hle_jsonl_path: Path, subset_ids: list[str]) -> pd.DataFrame:
    """Extract the requested HLE subset from a user-provided raw HLE JSONL file."""
    target_ids = set(subset_ids)
    matched_records: list[dict[str, str]] = []
    matched_ids: list[str] = []

    with hle_jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {hle_jsonl_path}:{line_number}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {hle_jsonl_path}:{line_number}")

            item_id = str(record.get("id", "")).strip()
            if item_id not in target_ids:
                continue

            question = record.get("question")
            rationale = record.get("rationale")
            if question is None or rationale is None:
                raise ValueError(
                    f"Matched record {item_id} is missing 'question' or 'rationale'"
                )

            matched_ids.append(item_id)
            matched_records.append(
                {
                    "id": item_id,
                    "question": str(question),
                    "rationale": str(rationale),
                    "rationale_prompt": str(question),
                }
            )

    if len(matched_ids) != len(set(matched_ids)):
        duplicates = pd.Series(matched_ids).value_counts()
        duplicate_ids = duplicates[duplicates > 1].index.tolist()
        raise ValueError(
            f"Raw HLE file contains duplicate matched IDs: {duplicate_ids[:10]}"
            f"{' ...' if len(duplicate_ids) > 10 else ''}"
        )

    found_ids = set(matched_ids)
    missing_ids = [item_id for item_id in subset_ids if item_id not in found_ids]
    if missing_ids:
        raise ValueError(
            f"Failed to find {len(missing_ids)} target IDs in the raw HLE file. "
            f"Examples: {missing_ids[:10]}"
        )

    df = pd.DataFrame(matched_records)
    require_columns(df, RAW_SUBSET_COLUMNS, "Extracted HLE subset")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a local HLE subset by item ID and save it as parquet."
    )
    parser.add_argument(
        "--hle-jsonl",
        type=Path,
        required=True,
        help="Path to a user-provided raw HLE JSONL file",
    )
    parser.add_argument(
        "--subset-ids",
        type=Path,
        default=Path("hle_rebuild/metadata/subset_ids.jsonl"),
        help="JSONL file with one {'id': ...} object per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hle_rebuild/items/hle_subset_raw.parquet"),
        help="Output parquet file for the extracted subset",
    )
    args = parser.parse_args()

    subset_ids = load_subset_ids(args.subset_ids)
    subset_df = extract_subset(args.hle_jsonl, subset_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(args.output, index=False)

    print(f"Raw HLE file: {args.hle_jsonl}")
    print(f"Subset ID file: {args.subset_ids}")
    print(f"Output file: {args.output}")
    print(f"Requested IDs: {len(subset_ids)}")
    print(f"Matched rows: {len(subset_df)}")

    empty_counts = count_empty_cells(subset_df)
    print(f"Has empty cells: {bool(empty_counts)}")
    if empty_counts:
        print("Empty-cell counts by column:")
        for column, count in empty_counts.items():
            print(f"  {column}: {count}")


if __name__ == "__main__":
    main()
