from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from hle_rebuild.common import FINAL_ITEM_COLUMNS, count_empty_cells, require_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge locally rebuilt HLE items into irt_data_items_no_hle.parquet."
    )
    parser.add_argument(
        "--items-no-hle",
        type=Path,
        default=Path("irt_data_items_no_hle.parquet"),
        help="Base items parquet without HLE rows",
    )
    parser.add_argument(
        "--hle-features",
        type=Path,
        default=Path("hle_rebuild/items/hle_subset_features.parquet"),
        help="Local HLE parquet produced by generate_features.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("irt_data_items_rebuilt_with_hle.parquet"),
        help="Output merged items parquet",
    )
    args = parser.parse_args()

    base_df = pd.read_parquet(args.items_no_hle)
    hle_df = pd.read_parquet(args.hle_features)

    require_columns(base_df, FINAL_ITEM_COLUMNS, "Base items parquet")
    require_columns(
        hle_df,
        ["question", "rationale", "rationale_prompt", "question_embedding", "rationale_embedding"]
        + FINAL_ITEM_COLUMNS[6:],
        "Local HLE feature parquet",
    )

    prepared_hle_df = hle_df.copy()
    prepared_hle_df["scenario"] = "hle"
    prepared_hle_df = prepared_hle_df[FINAL_ITEM_COLUMNS]

    duplicate_question_mask = prepared_hle_df.duplicated(subset=["scenario", "question"], keep=False)
    if duplicate_question_mask.any():
        examples = prepared_hle_df.loc[duplicate_question_mask, ["scenario", "question"]].head(10)
        raise ValueError(
            "Local HLE feature parquet contains duplicate (scenario, question) rows. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    merged_df = pd.concat(
        [base_df[FINAL_ITEM_COLUMNS], prepared_hle_df],
        axis=0,
        ignore_index=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(args.output, index=False)

    print(f"Base items file: {args.items_no_hle}")
    print(f"Local HLE file: {args.hle_features}")
    print(f"Output file: {args.output}")
    print(f"Base rows: {len(base_df)}")
    print(f"HLE rows added: {len(prepared_hle_df)}")
    print(f"Merged rows: {len(merged_df)}")

    empty_counts = count_empty_cells(merged_df)
    print(f"Has empty cells: {bool(empty_counts)}")
    if empty_counts:
        print("Empty-cell counts by column:")
        for column, count in empty_counts.items():
            print(f"  {column}: {count}")


if __name__ == "__main__":
    main()
