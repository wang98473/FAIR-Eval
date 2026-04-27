"""
Build an HDF5 file from prepared HELM IRT intermediate parquet tables.

Expected inputs:
- A prepared response parquet with at least: scenario, question, model, score
- A prepared item parquet with at least: scenario, question

Optional item columns that will be written when present:
- question_embedding
- rationale
- rationale_embedding
- item-level feature columns listed in ALL_FEAT_COLS
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_RESPONSES_PATH = Path("path/to/your/irt_data_responses.parquet")
DEFAULT_ITEMS_PATH = Path("path/to/your/irt_data_items.parquet")
DEFAULT_OUTPUT_PATH = Path("path/to/your/irt_data.hdf5")

Q_FEAT_COLS = [
    "q_len",
    "q_sent_count",
    "q_negation_count",
    "q_conditional_count",
    "q_avg_word_len",
    "q_subordinate_clause_count",
]
A_FEAT_COLS = [
    "answer_len",
    "answer_sent_count",
    "answer_connective_density",
    "answer_clause_count",
    "answer_entity_count",
    "answer_hedge_ratio",
]
INTER_FEAT_COLS = [
    "answer_lexical_overlap",
    "answer_new_info_ratio",
]
TASK_FEAT_COLS = ["task_type"]

FEATURE_GROUPS = {
    "question_features": Q_FEAT_COLS,
    "answer_features": A_FEAT_COLS,
    "interaction_features": INTER_FEAT_COLS,
    "task_type_features": TASK_FEAT_COLS,
}
ALL_FEAT_COLS = Q_FEAT_COLS + A_FEAT_COLS + INTER_FEAT_COLS + TASK_FEAT_COLS
TASK_TYPE_MAP = {
    "factual": 0,
    "reasoning": 1,
    "calculation": 2,
    "judgment": 3,
    "metacognitive": 4,
}


def ensure_writable_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. "
            "Pass --overwrite only if you explicitly want to replace it."
        )


def ensure_readable_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def validate_prepared_responses(df: pd.DataFrame) -> None:
    required = {"scenario", "question", "model", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prepared response parquet missing required columns: {sorted(missing)}")

    dup_mask = df.duplicated(subset=["scenario", "question", "model"], keep=False)
    if dup_mask.any():
        examples = (
            df.loc[dup_mask, ["scenario", "question", "model"]]
            .drop_duplicates()
            .head(10)
        )
        raise ValueError(
            "Prepared response parquet contains duplicate (scenario, question, model) rows. "
            f"Example rows:\n{examples.to_string(index=False)}"
        )


def validate_prepared_items(df: pd.DataFrame) -> None:
    required = {"scenario", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prepared item parquet missing required columns: {sorted(missing)}")

    dup_mask = df.duplicated(subset=["scenario", "question"], keep=False)
    if dup_mask.any():
        examples = (
            df.loc[dup_mask, ["scenario", "question"]]
            .drop_duplicates()
            .head(10)
        )
        raise ValueError(
            "Prepared item parquet contains duplicate (scenario, question) rows. "
            f"Example rows:\n{examples.to_string(index=False)}"
        )


def make_item_key(scenario: str, question: str) -> str:
    return json.dumps([str(scenario), str(question)], ensure_ascii=False)


def build_response_matrix(
    df_resp: pd.DataFrame,
    item_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    print("\n[3] Building response matrix")

    item_keys = [
        make_item_key(scenario, question)
        for scenario, question in zip(item_df["scenario"], item_df["question"])
    ]
    question_to_idx = {item_key: idx for idx, item_key in enumerate(item_keys)}
    match_key_to_idx = {
        make_item_key(scenario, question): idx
        for idx, (scenario, question) in enumerate(zip(item_df["scenario"], item_df["question"]))
    }
    all_models = sorted(df_resp["model"].astype(str).unique().tolist())
    model_to_idx = {model: idx for idx, model in enumerate(all_models)}

    n_items = len(question_to_idx)
    n_models = len(model_to_idx)
    matrix = np.full((n_items, n_models), -1, dtype=np.int8)

    df_valid = df_resp.copy()
    df_valid["match_key"] = [
        make_item_key(scenario, question)
        for scenario, question in zip(df_valid["scenario"], df_valid["question"])
    ]
    df_valid = df_valid[df_valid["match_key"].isin(match_key_to_idx)].copy()

    item_idx = df_valid["match_key"].map(match_key_to_idx).values
    model_idx = df_valid["model"].astype(str).map(model_to_idx).values
    scores = df_valid["score"].astype(np.int8).values
    matrix[item_idx, model_idx] = scores

    fill_rate = float((matrix >= 0).mean())
    print(f"    N_items={n_items} | N_models={n_models} | fill_rate={fill_rate:.2%}")
    return matrix, question_to_idx, model_to_idx


def parse_embeddings_col(item_df: pd.DataFrame, col: str) -> tuple[np.ndarray | None, int]:
    def parse_one(raw: object) -> np.ndarray | None:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        if isinstance(raw, (list, np.ndarray)):
            return np.array(raw, dtype=np.float32)

        text = str(raw).strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                return np.array(parser(text), dtype=np.float32)
            except Exception:
                pass

        try:
            cleaned = re.sub(r"[\[\],]", " ", text)
            return np.array([float(x) for x in cleaned.split()], dtype=np.float32)
        except Exception:
            return None

    if col not in item_df.columns:
        print(f"    [WARN] column not found: {col}")
        return None, 0

    parsed = item_df[col].apply(parse_one).tolist()
    valid = [value for value in parsed if value is not None]
    if not valid:
        print(f"    [WARN] failed to parse all values in column: {col}")
        return None, 0

    emb_dim = valid[0].shape[0]
    matrix = np.zeros((len(parsed), emb_dim), dtype=np.float32)
    for idx, value in enumerate(parsed):
        if value is not None:
            matrix[idx] = value

    n_missing = sum(1 for value in parsed if value is None)
    print(f"    {col}: dim={emb_dim} | missing={n_missing}/{len(parsed)}")
    return matrix, emb_dim


def parse_question_embeddings(item_df: pd.DataFrame) -> tuple[np.ndarray | None, int]:
    return parse_embeddings_col(item_df, "question_embedding")


def write_hdf5(
    output_path: Path,
    item_df: pd.DataFrame,
    response_matrix: np.ndarray,
    question_emb_matrix: np.ndarray | None,
    question_to_idx: dict[str, int],
    model_to_idx: dict[str, int],
) -> None:
    print(f"\n[5] Writing HDF5: {output_path}")
    n_items, n_models = response_matrix.shape

    with h5py.File(output_path, "w") as h5_file:
        grp_items = h5_file.create_group("items")
        str_dt = h5py.special_dtype(vlen=str)

        grp_items.create_dataset(
            "question_ids",
            data=np.array(item_df["question"].astype(str).tolist(), dtype=object),
            dtype=str_dt,
        )
        grp_items.create_dataset(
            "scenario",
            data=np.array(item_df["scenario"].astype(str).tolist(), dtype=object),
            dtype=str_dt,
        )

        rationale_col = None
        if "rationale" in item_df.columns:
            rationale_col = "rationale"
        elif "solution" in item_df.columns:
            rationale_col = "solution"

        if rationale_col is not None:
            grp_items.create_dataset(
                "rationale",
                data=np.array(item_df[rationale_col].fillna("").astype(str).tolist(), dtype=object),
                dtype=str_dt,
            )
            print(f"    rationale: {len(item_df)} rows")

        rationale_emb_matrix, rationale_dim = parse_embeddings_col(item_df, "rationale_embedding")
        if rationale_emb_matrix is not None:
            grp_items.create_dataset(
                "rationale_embedding",
                data=rationale_emb_matrix,
                dtype="float32",
                chunks=(min(512, n_items), rationale_dim),
                compression="gzip",
                compression_opts=4,
            )
            print(f"    rationale_embedding: {rationale_emb_matrix.shape}")

        if question_emb_matrix is not None:
            question_dim = question_emb_matrix.shape[1]
            grp_items.create_dataset(
                "question_embedding",
                data=question_emb_matrix,
                dtype="float32",
                chunks=(min(512, n_items), question_dim),
                compression="gzip",
                compression_opts=4,
            )
            print(f"    question_embedding: {question_emb_matrix.shape}")
        else:
            print("    [WARN] question_embedding not written because the data is missing")

        grp_features = grp_items.create_group("features")
        feat_cols_present = [col for col in ALL_FEAT_COLS if col in item_df.columns]
        feat_cols_missing = [col for col in ALL_FEAT_COLS if col not in item_df.columns]
        if feat_cols_missing:
            print(f"    [WARN] missing feature columns: {feat_cols_missing}")

        for col in feat_cols_present:
            arr = item_df[col].values
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)
            elif np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int32)
            else:
                try:
                    arr = arr.astype(np.float32)
                except Exception:
                    arr = np.full(n_items, -1, dtype=np.int32)
            grp_features.create_dataset(col, data=arr, compression="gzip", compression_opts=4)

        print(f"    written feature columns ({len(feat_cols_present)}): {feat_cols_present}")

        grp_responses = h5_file.create_group("responses")
        grp_responses.create_dataset(
            "matrix",
            data=response_matrix,
            dtype="int8",
            chunks=(min(512, n_items), n_models),
            compression="gzip",
            compression_opts=4,
        )
        print(f"    response_matrix: {response_matrix.shape}")

        grp_meta = h5_file.create_group("meta")
        grp_meta.create_dataset("model_index", data=json.dumps(model_to_idx, ensure_ascii=False))
        grp_meta.create_dataset("question_index", data=json.dumps(question_to_idx, ensure_ascii=False))
        grp_meta.create_dataset("feature_names", data=json.dumps(FEATURE_GROUPS, ensure_ascii=False))
        grp_meta.create_dataset("task_type_map", data=json.dumps(TASK_TYPE_MAP, ensure_ascii=False))

    print(f"\n[Done] Wrote HDF5: {output_path}")
    print(f"  items   : {n_items}")
    print(f"  models  : {n_models}")
    print(f"  features: {len(feat_cols_present)}")


def main(
    responses_path: Path = DEFAULT_RESPONSES_PATH,
    items_path: Path = DEFAULT_ITEMS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    overwrite: bool = False,
) -> None:
    ensure_readable_input(responses_path)
    ensure_readable_input(items_path)
    ensure_writable_output(output_path, overwrite)

    print(f"[1] Reading prepared response table: {responses_path}")
    df_resp = pd.read_parquet(responses_path)
    validate_prepared_responses(df_resp)
    print(f"    rows={len(df_resp):,} | columns={list(df_resp.columns)}")

    print(f"\n[2] Reading prepared item table: {items_path}")
    item_df = pd.read_parquet(items_path)
    validate_prepared_items(item_df)
    print(f"    rows={len(item_df):,} | columns={list(item_df.columns)}")

    response_matrix, question_to_idx, model_to_idx = build_response_matrix(df_resp, item_df)

    print("\n[4] Parsing question_embedding")
    question_emb_matrix, _ = parse_question_embeddings(item_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_hdf5(
        output_path=output_path,
        item_df=item_df,
        response_matrix=response_matrix,
        question_emb_matrix=question_emb_matrix,
        question_to_idx=question_to_idx,
        model_to_idx=model_to_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build HELM IRT HDF5 from prepared response and item parquet tables"
    )
    parser.add_argument(
        "--responses",
        type=str,
        default=str(DEFAULT_RESPONSES_PATH),
        help="Path to the prepared response parquet",
    )
    parser.add_argument(
        "--items",
        type=str,
        default=str(DEFAULT_ITEMS_PATH),
        help="Path to the prepared item parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the output HDF5 file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    args = parser.parse_args()

    main(
        responses_path=Path(args.responses),
        items_path=Path(args.items),
        output_path=Path(args.output),
        overwrite=args.overwrite,
    )
