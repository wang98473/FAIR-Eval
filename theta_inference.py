"""
Export model-level theta estimates from a trained semantic 2PL checkpoint.

The script selects a checkpoint from `cv_results.json`, reconstructs the model
architecture from the checkpoint tensors, extracts theta for every model, and
writes a ranked CSV table.
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


CFG = {
    "h5_path": "path/to/your/irt_data.hdf5",
    "cv_results_json": "path/to/your/cv_results.json",
    "output_csv": "path/to/your/theta_estimates.csv",
    "fold_select": "best_auc",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SemanticIRT2PL(nn.Module):
    """Semantic 2PL IRT model used for theta extraction."""

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

    @torch.no_grad()
    def get_all_theta(self) -> np.ndarray:
        """Return theta for every model in the embedding table."""

        self.eval()
        all_ids = torch.arange(self.n_models, device=next(self.parameters()).device)
        theta_h = self.encoder_theta(self.model_emb(all_ids))
        theta = self.head_theta(theta_h).squeeze(-1)
        return theta.cpu().numpy()


def load_torch_file(path: Path) -> Any:
    """Load a torch file while supporting multiple PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json_dataset(h5_file: h5py.File, key: str) -> dict[str, Any] | None:
    """Load a JSON-encoded HDF5 dataset when present."""

    if key not in h5_file:
        return None
    raw = h5_file[key][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def resolve_existing_path(raw_path: str, cv_results_path: Path) -> Path:
    """Resolve a checkpoint path recorded in `cv_results.json`."""

    candidate = Path(raw_path)
    options: list[Path] = []
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


def load_model_metadata(h5_path: Path) -> dict[str, Any]:
    """Load model counts and model-index mappings from HDF5."""

    with h5py.File(h5_path, "r") as h5_file:
        response_matrix = h5_file["responses/matrix"]
        n_models = int(response_matrix.shape[1])
        model_to_idx = load_json_dataset(h5_file, "meta/model_index")
        if model_to_idx is None:
            raise KeyError("Missing meta/model_index in HDF5")

    idx_to_model = {int(v): str(k) for k, v in model_to_idx.items()}
    return {
        "n_models": n_models,
        "model_to_idx": {str(k): int(v) for k, v in model_to_idx.items()},
        "idx_to_model": idx_to_model,
    }


def build_output_dataframe(
    theta_values: np.ndarray,
    idx_to_model: dict[int, str],
    best_fold_key: str,
    metric_name: str,
    metric_value: float,
    ckpt_path: Path,
) -> pd.DataFrame:
    """Assemble the ranked theta output table."""

    records: list[dict[str, Any]] = []

    for idx, theta_val in enumerate(theta_values):
        records.append(
            {
                "model_index": idx,
                "model_name": idx_to_model.get(idx, f"unknown_model_{idx}"),
                "theta": float(theta_val),
                "best_fold": best_fold_key,
                "selection_metric": metric_name,
                "selection_metric_value": float(metric_value),
                "checkpoint_path": str(ckpt_path),
            }
        )

    df = pd.DataFrame(records).sort_values("theta", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def extract_theta(cfg: dict[str, Any]) -> pd.DataFrame:
    """Run the full theta extraction pipeline."""

    h5_path = Path(cfg["h5_path"]).resolve()
    cv_results_path = Path(cfg["cv_results_json"]).resolve()
    output_csv = Path(cfg["output_csv"]).resolve()

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    if not cv_results_path.exists():
        raise FileNotFoundError(f"cv_results.json not found: {cv_results_path}")

    print("=" * 60)
    print("Step 1/4: load CV results and choose checkpoint")
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

    print("\n" + "=" * 60)
    print("Step 2/4: load model metadata from HDF5")
    print("=" * 60)
    metadata = load_model_metadata(h5_path)
    print(f"HDF5: {h5_path}")
    print(f"Total models: {metadata['n_models']}")
    print(f"Loaded model names: {len(metadata['model_to_idx'])}")

    print("\n" + "=" * 60)
    print("Step 3/4: restore model architecture and extract theta")
    print("=" * 60)
    state_dict = load_torch_file(ckpt_path)
    model_cfg = infer_model_config(state_dict, n_models=metadata["n_models"])
    model = SemanticIRT2PL(**model_cfg).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    theta_values = model.get_all_theta()
    print(f"Device: {DEVICE}")
    print(f"Inferred config: {model_cfg}")
    print(
        "Theta stats: "
        f"mean={theta_values.mean():.6f}, std={theta_values.std():.6f}, "
        f"min={theta_values.min():.6f}, max={theta_values.max():.6f}"
    )

    print("\n" + "=" * 60)
    print("Step 4/4: build ranked table and save CSV")
    print("=" * 60)
    df = build_output_dataframe(
        theta_values=theta_values,
        idx_to_model=metadata["idx_to_model"],
        best_fold_key=best_fold_key,
        metric_name=metric_name,
        metric_value=metric_value,
        ckpt_path=ckpt_path,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {output_csv}")
    print("\nTop 10 models by theta:")
    print(df[["rank", "model_name", "theta"]].head(10).to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract model-level theta estimates from a semantic 2PL checkpoint"
    )
    parser.add_argument(
        "--h5",
        type=str,
        default=CFG["h5_path"],
        help="Path to the HDF5 dataset",
    )
    parser.add_argument(
        "--cv-results",
        type=str,
        default=CFG["cv_results_json"],
        help="Path to cv_results.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=CFG["output_csv"],
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--fold-select",
        type=str,
        default=CFG["fold_select"],
        choices=["best_auc", "best_bce"],
        help="Fold selection rule",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG.update(
        {
            "h5_path": args.h5,
            "cv_results_json": args.cv_results,
            "output_csv": args.output_csv,
            "fold_select": args.fold_select,
        }
    )
    extract_theta(CFG)
