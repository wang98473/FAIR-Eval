from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from hle_rebuild.common import count_empty_cells, require_columns


FINAL_RESPONSE_COLUMNS = ["model", "question", "scenario", "benchmark", "score"]


def build_hle_response_rows(
    validation_csv_path: Path,
    subset_items_path: Path,
) -> pd.DataFrame:
    """Map question IDs back to question text and return FAIR-Eval response rows."""
    validation_df = pd.read_csv(validation_csv_path)
    subset_df = pd.read_parquet(subset_items_path)

    require_columns(
        validation_df,
        ["model", "question_id", "scenario", "benchmark", "score"],
        "HLE validation CSV",
    )
    require_columns(subset_df, ["id", "question"], "Local HLE subset parquet")

    question_map_df = subset_df[["id", "question"]].copy()
    duplicate_id_mask = question_map_df.duplicated(subset=["id"], keep=False)
    if duplicate_id_mask.any():
        examples = question_map_df.loc[duplicate_id_mask, ["id", "question"]].head(10)
        raise ValueError(
            "Local HLE subset parquet contains duplicate IDs. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    merged_df = validation_df.merge(
        question_map_df,
        how="left",
        left_on="question_id",
        right_on="id",
        validate="many_to_one",
    )

    unmatched_mask = merged_df["question"].isna()
    if unmatched_mask.any():
        missing_ids = (
            merged_df.loc[unmatched_mask, "question_id"].astype(str).drop_duplicates().tolist()
        )
        raise ValueError(
            f"Failed to map {len(missing_ids)} question IDs to question text. "
            f"Examples: {missing_ids[:10]}"
        )

    output_df = merged_df[FINAL_RESPONSE_COLUMNS].copy()
    output_df["question"] = output_df["question"].astype(str)
    return output_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge local HLE response rows into irt_data_responses_no_hle.parquet."
    )
    parser.add_argument(
        "--responses-no-hle",
        type=Path,
        default=Path("irt_data_responses_no_hle.parquet"),
        help="Base response parquet without HLE rows",
    )
    parser.add_argument(
        "--hle-validation-csv",
        type=Path,
        default=Path("hle_rebuild/responses/hle_validate_csv_results.csv"),
        help="CSV file with columns model, question_id, scenario, benchmark, score",
    )
    parser.add_argument(
        "--hle-subset-items",
        type=Path,
        default=Path("hle_rebuild/items/hle_subset_raw.parquet"),
        help="Local HLE subset parquet with id and question columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("irt_data_responses_rebuilt_with_hle.parquet"),
        help="Output merged responses parquet",
    )
    args = parser.parse_args()

    base_df = pd.read_parquet(args.responses_no_hle)
    require_columns(base_df, FINAL_RESPONSE_COLUMNS, "Base response parquet")

    hle_response_df = build_hle_response_rows(
        validation_csv_path=args.hle_validation_csv,
        subset_items_path=args.hle_subset_items,
    )

    merged_df = pd.concat(
        [base_df[FINAL_RESPONSE_COLUMNS], hle_response_df[FINAL_RESPONSE_COLUMNS]],
        axis=0,
        ignore_index=True,
    )

    duplicate_mask = merged_df.duplicated(
        subset=["model", "question", "scenario", "benchmark"],
        keep=False,
    )
    duplicate_count = int(duplicate_mask.sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(args.output, index=False)

    print(f"Base response file: {args.responses_no_hle}")
    print(f"HLE validation CSV: {args.hle_validation_csv}")
    print(f"HLE subset parquet: {args.hle_subset_items}")
    print(f"Output file: {args.output}")
    print(f"Base rows: {len(base_df)}")
    print(f"HLE rows added: {len(hle_response_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Duplicate merged rows by response key: {duplicate_count}")

    empty_counts = count_empty_cells(merged_df)
    print(f"Has empty cells: {bool(empty_counts)}")
    if empty_counts:
        print("Empty-cell counts by column:")
        for column, count in empty_counts.items():
            print(f"  {column}: {count}")


if __name__ == "__main__":
    main()
