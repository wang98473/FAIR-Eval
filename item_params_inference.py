"""
Export item-level IRT parameters from a trained semantic 2PL checkpoint.

The script selects a checkpoint from `cv_results.json`, reconstructs the model
architecture from the checkpoint tensors, applies the saved feature
normalization, and writes calibrated `a` and `b` values for every item in the
input HDF5 file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


FEAT_NAMES = [
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

CFG = {
    "h5_path": "path/to/your/irt_data.hdf5",
    "cv_results_json": "path/to/your/cv_results.json",
    "output_csv": "path/to/your/items_params.csv",
    "fold_select": "best_auc",
    "batch_size": 256,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SemanticIRT2PL(nn.Module):
    """Semantic 2PL IRT model used for item parameter extraction."""

    def __init__(
        self,
        n_models: int,
        emb_dim: int,
        proj_dim: int,
        n_scalar_feats: int,
        n_task_types: int,
        task_emb_dim: int,
        hidden_dim: int,
        feat_dim: int,
        theta_emb_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_models = n_models

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

        self.task_emb = nn.Embedding(n_task_types, task_emb_dim)

        self.sem_encoder = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.feat_encoder = nn.Sequential(
            nn.Linear(n_scalar_feats + task_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
            nn.GELU(),
        )

        self.head_a = nn.Linear(feat_dim, 1)
        self.head_b = nn.Linear(feat_dim, 1)

        self.model_emb = nn.Embedding(n_models, theta_emb_dim)
        self.encoder_theta = nn.Sequential(
            nn.Linear(theta_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
            nn.GELU(),
        )
        self.head_theta = nn.Linear(feat_dim, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def _encode_item(
        self,
        emb_q: torch.Tensor,
        emb_rat: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        h_q = self.proj(emb_q)
        h_rat = self.proj(emb_rat)
        h_sem = self.sem_encoder(torch.cat([h_q, h_rat], dim=-1))

        scalar = feats[:, :-1]
        task = feats[:, -1].long()
        h_task = self.task_emb(task)
        h_feat = self.feat_encoder(torch.cat([scalar, h_task], dim=-1))

        return self.encoder(torch.cat([h_sem, h_feat], dim=-1))

    @torch.no_grad()
    def get_item_params(
        self,
        emb_q: torch.Tensor,
        emb_rat: torch.Tensor,
        feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encode_item(emb_q, emb_rat, feats)
        a = F.softplus(self.head_a(h)).squeeze(-1)
        b = self.head_b(h).squeeze(-1)
        return a, b


def decode_strings(values: np.ndarray) -> list[str]:
    """Decode an HDF5 string dataset into Python strings."""

    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def load_json_dataset(h5_file: h5py.File, key: str) -> dict[str, Any] | None:
    """Load a JSON-encoded HDF5 dataset when present."""

    if key not in h5_file:
        return None
    raw = h5_file[key][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def load_torch_file(path: Path) -> Any:
    """Load a torch file while supporting multiple PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_existing_path(raw_path: str, cv_results_path: Path) -> Path:
    """Resolve a checkpoint path recorded in `cv_results.json`."""

    candidate = Path(raw_path)
    options = []
    if candidate.is_absolute():
        options.append(candidate)
    else:
        options.extend(
            [
                candidate,
                Path.cwd() / candidate,
                cv_results_path.parent / candidate,
                cv_results_path.parent / candidate.name,
            ]
        )

    for path in options:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Could not resolve checkpoint path: {raw_path}")


def infer_model_config(state_dict: dict[str, torch.Tensor], n_models: int) -> dict[str, int | float]:
    """Infer model dimensions from checkpoint tensor shapes."""

    proj_weight = state_dict["proj.0.weight"]
    task_weight = state_dict["task_emb.weight"]
    sem_weight = state_dict["sem_encoder.0.weight"]
    feat_weight = state_dict["feat_encoder.0.weight"]
    encoder_last = state_dict["encoder.3.weight"]
    model_weight = state_dict["model_emb.weight"]

    hidden_dim = int(sem_weight.shape[0])
    proj_dim = int(proj_weight.shape[0])
    emb_dim = int(proj_weight.shape[1])
    n_task_types = int(task_weight.shape[0])
    task_emb_dim = int(task_weight.shape[1])
    n_scalar_feats = int(feat_weight.shape[1] - task_emb_dim)
    feat_dim = int(encoder_last.shape[0])
    theta_emb_dim = int(model_weight.shape[1])

    dropout = 0.2
    if "proj.2.p" in state_dict:
        dropout = float(state_dict["proj.2.p"])

    return {
        "n_models": n_models,
        "emb_dim": emb_dim,
        "proj_dim": proj_dim,
        "n_scalar_feats": n_scalar_feats,
        "n_task_types": n_task_types,
        "task_emb_dim": task_emb_dim,
        "hidden_dim": hidden_dim,
        "feat_dim": feat_dim,
        "theta_emb_dim": theta_emb_dim,
        "dropout": dropout,
    }


def select_best_fold(cv_results: dict[str, dict[str, Any]], fold_select: str) -> tuple[str, float, str]:
    """Select the checkpoint fold by validation AUC or BCE."""

    fold_keys = sorted(cv_results.keys(), key=lambda x: int(x.split("_")[1]))
    if fold_select == "best_auc":
        best_key = max(fold_keys, key=lambda k: cv_results[k]["best_val_auc"])
        return best_key, float(cv_results[best_key]["best_val_auc"]), "best_val_auc"
    best_key = min(fold_keys, key=lambda k: cv_results[k]["best_val_bce"])
    return best_key, float(cv_results[best_key]["best_val_bce"]), "best_val_bce"


def load_feature_norm(feat_norm_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load saved feature normalization statistics."""

    norm_obj = load_torch_file(feat_norm_path)
    mean = norm_obj["mean"]
    std = norm_obj["std"]
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1)
    std = np.clip(std, 1e-6, None)
    return mean, std


def load_item_metadata(
    h5_path: Path,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
) -> dict[str, Any]:
    """Load embeddings, features, responses, and item metadata from HDF5."""

    with h5py.File(h5_path, "r") as h5_file:
        question_embeddings = h5_file["items/question_embedding"]
        rationale_embeddings = h5_file["items/rationale_embedding"]
        response_matrix = h5_file["responses/matrix"]

        n_items_total = int(question_embeddings.shape[0])
        n_models = int(response_matrix.shape[1])

        if "items/scenario" in h5_file:
            scenarios = decode_strings(h5_file["items/scenario"][:])
        else:
            scenarios = [""] * n_items_total

        selected_indices = np.arange(n_items_total, dtype=np.int64)

        if "items/question_ids" in h5_file:
            question_texts = decode_strings(h5_file["items/question_ids"][:])
        else:
            question_index = load_json_dataset(h5_file, "meta/question_index") or {}
            reverse_map = {int(v): str(k) for k, v in question_index.items()}
            question_texts = [reverse_map.get(i, f"item_{i}") for i in range(n_items_total)]

        feature_columns = []
        for name in FEAT_NAMES:
            if f"items/features/{name}" not in h5_file:
                raise KeyError(f"Missing feature column in HDF5: items/features/{name}")
            col = np.asarray(h5_file[f"items/features/{name}"][selected_indices], dtype=np.float32)
            if np.isnan(col).any():
                fill_value = np.nanmean(col)
                if np.isnan(fill_value):
                    fill_value = 0.0
                col = np.where(np.isnan(col), fill_value, col)
            feature_columns.append(col)

        features = np.stack(feature_columns, axis=1).astype(np.float32)
        features[:, :-1] = (features[:, :-1] - feat_mean) / feat_std

    return {
        "n_items_total": n_items_total,
        "n_models": n_models,
        "selected_indices": selected_indices,
        "question_texts": question_texts,
        "scenarios": scenarios,
        "features": features,
    }


def batched_item_params(
    h5_path: Path,
    model: SemanticIRT2PL,
    selected_indices: np.ndarray,
    features: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer item parameters in batches directly from the HDF5 datasets."""

    all_a: list[np.ndarray] = []
    all_b: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as h5_file:
        emb_q_ds = h5_file["items/question_embedding"]
        emb_rat_ds = h5_file["items/rationale_embedding"]

        for start in range(0, len(selected_indices), batch_size):
            end = min(start + batch_size, len(selected_indices))
            batch_indices = selected_indices[start:end]

            emb_q = torch.tensor(
                emb_q_ds[batch_indices], dtype=torch.float32, device=DEVICE
            )
            emb_rat = torch.tensor(
                emb_rat_ds[batch_indices], dtype=torch.float32, device=DEVICE
            )
            feat_batch = torch.tensor(
                features[start:end], dtype=torch.float32, device=DEVICE
            )

            with torch.no_grad():
                a_batch, b_batch = model.get_item_params(emb_q, emb_rat, feat_batch)

            all_a.append(a_batch.cpu().numpy())
            all_b.append(b_batch.cpu().numpy())

    return np.concatenate(all_a), np.concatenate(all_b)


def build_output_dataframe(
    selected_indices: np.ndarray,
    question_texts: list[str],
    scenarios: list[str],
    a_values: np.ndarray,
    b_values: np.ndarray,
) -> pd.DataFrame:
    """Assemble the final item parameter table."""

    records: list[dict[str, Any]] = []

    for local_idx, item_idx in enumerate(selected_indices):
        record = {
            "question_id": int(item_idx),
            "question_text": question_texts[int(item_idx)],
            "param_a": float(a_values[local_idx]),
            "param_b": float(b_values[local_idx]),
            "scenario": scenarios[int(item_idx)],
        }

        records.append(record)

    column_order = [
        "question_id",
        "question_text",
        "param_a",
        "param_b",
        "scenario",
    ]
    return (
        pd.DataFrame(records)[column_order]
        .sort_values("scenario", kind="stable")
        .reset_index(drop=True)
    )


def extract_item_params(cfg: dict[str, Any]) -> pd.DataFrame:
    """Run the full item-parameter extraction pipeline."""

    h5_path = Path(cfg["h5_path"]).resolve()
    cv_results_path = Path(cfg["cv_results_json"]).resolve()
    output_csv = Path(cfg["output_csv"]).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    if not cv_results_path.exists():
        raise FileNotFoundError(f"cv_results.json not found: {cv_results_path}")

    feat_norm_path = cv_results_path.parent / "feat_norm.pt"
    if not feat_norm_path.exists():
        raise FileNotFoundError(f"Feature normalization file not found: {feat_norm_path}")

    print("=" * 60)
    print("Step 1/5: load CV results and choose checkpoint")
    print("=" * 60)
    with open(cv_results_path, "r", encoding="utf-8") as file:
        cv_results = json.load(file)

    best_fold_key, metric_value, metric_name = select_best_fold(
        cv_results, cfg["fold_select"]
    )
    ckpt_path = resolve_existing_path(cv_results[best_fold_key]["ckpt_path"], cv_results_path)

    print(f"Selected fold: {best_fold_key}")
    print(f"Selection metric: {metric_name} = {metric_value:.6f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Feature norm: {feat_norm_path.resolve()}")

    print("\n" + "=" * 60)
    print("Step 2/5: load normalization and item metadata")
    print("=" * 60)
    feat_mean, feat_std = load_feature_norm(feat_norm_path)
    metadata = load_item_metadata(
        h5_path=h5_path,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )
    selected_indices = metadata["selected_indices"]
    print(f"Total items in HDF5: {metadata['n_items_total']}")
    print(f"Items selected for export: {len(selected_indices)}")

    print("\n" + "=" * 60)
    print("Step 3/5: restore model architecture from checkpoint")
    print("=" * 60)
    state_dict = load_torch_file(ckpt_path)
    model_cfg = infer_model_config(state_dict, n_models=metadata["n_models"])
    model = SemanticIRT2PL(**model_cfg).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Device: {DEVICE}")
    print(f"Inferred config: {model_cfg}")

    print("\n" + "=" * 60)
    print("Step 4/5: infer parameters for all selected items")
    print("=" * 60)
    a_values, b_values = batched_item_params(
        h5_path=h5_path,
        model=model,
        selected_indices=selected_indices,
        features=metadata["features"],
        batch_size=int(cfg["batch_size"]),
    )
    print(
        f"a stats: mean={a_values.mean():.6f}, std={a_values.std():.6f}, "
        f"min={a_values.min():.6f}, max={a_values.max():.6f}"
    )
    print(
        f"b stats: mean={b_values.mean():.6f}, std={b_values.std():.6f}, "
        f"min={b_values.min():.6f}, max={b_values.max():.6f}"
    )

    print("\n" + "=" * 60)
    print("Step 5/5: write full item parameter table")
    print("=" * 60)
    df = build_output_dataframe(
        selected_indices=selected_indices,
        question_texts=metadata["question_texts"],
        scenarios=metadata["scenarios"],
        a_values=a_values,
        b_values=b_values,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {output_csv}")
    print(f"Rows written: {len(df)}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract item-level IRT parameters from a semantic 2PL checkpoint."
    )
    parser.add_argument("--h5", type=str, default=CFG["h5_path"], help="Path to HDF5 data file.")
    parser.add_argument(
        "--cv-results",
        type=str,
        default=CFG["cv_results_json"],
        help="Path to cv_results.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=CFG["output_csv"],
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--fold-select",
        type=str,
        default=CFG["fold_select"],
        choices=["best_auc", "best_bce"],
        help="Rule for selecting the checkpoint fold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CFG["batch_size"],
        help="Inference batch size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cfg = dict(CFG)
    run_cfg.update(
        {
            "h5_path": args.h5,
            "cv_results_json": args.cv_results,
            "output_csv": args.output_csv,
            "fold_select": args.fold_select,
            "batch_size": args.batch_size,
        }
    )
    extract_item_params(run_cfg)
