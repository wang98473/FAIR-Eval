"""
Semantic 2PL IRT with item-level cross-validation.

Expected HDF5 structure:
    items/question_embedding   : (N_items, 4096), float32
    items/rationale_embedding  : (N_items, 4096), float32
    items/features/<name>      : (N_items,), float32 or int32
    responses/matrix           : (N_items, N_models), int8, with -1 for missing
    meta/model_index           : JSON string mapping model names to indices

Feature order:
    q_len, q_sent_count, q_negation_count, q_conditional_count,
    q_avg_word_len, q_subordinate_clause_count,
    answer_len, answer_sent_count, answer_connective_density,
    answer_clause_count, answer_entity_count, answer_hedge_ratio,
    answer_lexical_overlap, answer_new_info_ratio,
    task_type
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    "task_type",  # Final column is categorical and handled with an embedding.
]

CFG = dict(
    h5_path="path/to/your/irt_data.hdf5",
    output_dir="path/to/your/model_train_outputs",
    n_splits=10,
    num_epochs=200,
    min_epochs=20,
    patience=15,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-3,
    proj_dim=128,
    hidden_dim=64,
    feat_dim=32,
    theta_emb_dim=64,
    n_scalar_feats=14,
    n_task_types=5,
    task_emb_dim=8,
    dropout=0.2,
    D=1.702,
    lambda_b=0.1,
    lambda_theta=0.1,
    grad_clip=2.0,
    num_workers=0,
    random_seed=42,
)


class IRTDataset(Dataset):
    """Returns one item at a time with its response vector and validity mask."""

    def __init__(self, all_emb_q, all_emb_rat, all_feats, all_responses, item_indices):
        self.all_emb_q = all_emb_q
        self.all_emb_rat = all_emb_rat
        self.all_feats = all_feats
        self.all_responses = all_responses
        self.item_indices = torch.tensor(item_indices, dtype=torch.long)

    def __len__(self):
        return len(self.item_indices)

    def __getitem__(self, idx):
        real_idx = int(self.item_indices[idx])
        emb_q = self.all_emb_q[real_idx]
        emb_rat = self.all_emb_rat[real_idx]
        feats = self.all_feats[real_idx]
        resp = self.all_responses[real_idx]
        mask = (resp >= 0).float()
        resp = torch.clamp(resp, min=0.0)
        return emb_q, emb_rat, feats, resp, mask, real_idx


def make_loader(
    all_emb_q: torch.Tensor,
    all_emb_rat: torch.Tensor,
    all_feats: torch.Tensor,
    all_responses: torch.Tensor,
    item_indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    ds = IRTDataset(all_emb_q, all_emb_rat, all_feats, all_responses, item_indices)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )


class SemanticIRT2PL(nn.Module):
    """Predicts item discrimination and difficulty from question semantics and features."""

    def __init__(
        self,
        n_models: int,
        emb_dim: int = 4096,
        proj_dim: int = 128,
        n_scalar_feats: int = 14,
        n_task_types: int = 5,
        task_emb_dim: int = 8,
        hidden_dim: int = 64,
        feat_dim: int = 32,
        theta_emb_dim: int = 64,
        dropout: float = 0.2,
        d_scale: float = 1.702,
    ):
        super().__init__()
        self.n_models = n_models
        self.d_scale = d_scale

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
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

        nn.init.normal_(self.task_emb.weight, std=0.1)

        for encoder in (self.sem_encoder, self.feat_encoder, self.encoder):
            for module in encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        nn.init.normal_(self.head_a.weight, std=0.01)
        nn.init.constant_(self.head_a.bias, 0.5)

        nn.init.normal_(self.head_b.weight, std=0.01)
        nn.init.zeros_(self.head_b.bias)

        nn.init.normal_(self.model_emb.weight, std=0.1)
        for module in self.encoder_theta.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.head_theta.weight, std=0.01)
        nn.init.zeros_(self.head_theta.bias)

    def _encode_item(
        self,
        emb_q: torch.Tensor,
        emb_rat: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes item inputs into a shared latent representation."""
        h_q = self.proj(emb_q)
        h_rat = self.proj(emb_rat)
        h_sem = self.sem_encoder(torch.cat([h_q, h_rat], dim=-1))

        scalar = feats[:, :-1]
        task = feats[:, -1].long()
        h_task = self.task_emb(task)
        h_feat = self.feat_encoder(torch.cat([scalar, h_task], dim=-1))

        return self.encoder(torch.cat([h_sem, h_feat], dim=-1))

    def get_item_params(
        self,
        emb_q: torch.Tensor,
        emb_rat: torch.Tensor,
        feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns item discrimination `a` and difficulty `b`."""
        h = self._encode_item(emb_q, emb_rat, feats)
        a = F.softplus(self.head_a(h)).squeeze(-1)
        b = self.head_b(h).squeeze(-1)
        return a, b

    def forward(
        self,
        emb_q: torch.Tensor,
        emb_rat: torch.Tensor,
        feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns response probabilities, item parameters, and model abilities."""
        h = self._encode_item(emb_q, emb_rat, feats)
        a = F.softplus(self.head_a(h)).squeeze(-1)
        b = self.head_b(h).squeeze(-1)

        all_ids = torch.arange(self.n_models, device=emb_q.device)
        theta_h = self.encoder_theta(self.model_emb(all_ids))
        theta = self.head_theta(theta_h).squeeze(-1)

        temp = torch.exp(self.log_temp)
        logit = self.d_scale * a.unsqueeze(1) * (theta.unsqueeze(0) - b.unsqueeze(1))
        probs = torch.sigmoid(logit / temp)
        return probs, a, b, theta


def irt_loss(
    P: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    theta: torch.Tensor,
    resp: torch.Tensor,
    mask: torch.Tensor,
    lambda_b: float = 0.1,
    lambda_theta: float = 0.1,
    regularize: bool = True,
) -> torch.Tensor:
    """Computes masked BCE plus optional regularization on `b` and `theta`."""
    P = torch.clamp(P, 1e-7, 1 - 1e-7)

    # Normalize BCE by the number of observed responses for each item.
    bce_elem = -(resp * torch.log(P) + (1 - resp) * torch.log(1 - P))
    valid_per_item = mask.sum(dim=1, keepdim=True).clamp(min=1)
    bce = (bce_elem * mask).sum(dim=1) / valid_per_item
    bce = bce.mean()

    if not regularize:
        return bce

    b_reg = b.mean() ** 2
    reg_theta_mean = theta.mean() ** 2
    reg_theta_var = (theta.var() - 1.0) ** 2
    reg_theta = reg_theta_mean + reg_theta_var

    return bce + lambda_b * b_reg + lambda_theta * reg_theta


@torch.no_grad()
def evaluate(
    model: SemanticIRT2PL,
    loader: DataLoader,
) -> tuple[float, float]:
    """Returns validation BCE and AUC over observed responses only."""
    model.eval()
    total_bce = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for emb_q, emb_rat, feats, resp, mask, _ in loader:
        P, a, b, theta = model(emb_q, emb_rat, feats)
        loss = irt_loss(P, a, b, theta, resp, mask, regularize=False)
        total_bce += loss.item()
        n_batches += 1

        valid = mask.bool()
        all_probs.append(P[valid].cpu().numpy())
        all_labels.append(resp[valid].cpu().numpy())

    val_bce = total_bce / max(n_batches, 1)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    if len(np.unique(labels)) < 2:
        val_auc = float("nan")
    else:
        val_auc = float(roc_auc_score(labels, probs))

    return val_bce, val_auc


def train_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_models: int,
    cfg: dict,
    output_dir: Path,
    all_emb_q: torch.Tensor,
    all_emb_rat: torch.Tensor,
    all_feats: torch.Tensor,
    all_responses: torch.Tensor,
) -> dict:
    """Trains one fold and returns training history plus the best checkpoint path."""
    train_loader = make_loader(
        all_emb_q,
        all_emb_rat,
        all_feats,
        all_responses,
        train_idx,
        cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_loader = make_loader(
        all_emb_q,
        all_emb_rat,
        all_feats,
        all_responses,
        val_idx,
        cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    model = SemanticIRT2PL(
        n_models=n_models,
        emb_dim=4096,
        proj_dim=cfg["proj_dim"],
        n_scalar_feats=cfg["n_scalar_feats"],
        n_task_types=cfg["n_task_types"],
        task_emb_dim=cfg["task_emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        feat_dim=cfg["feat_dim"],
        theta_emb_dim=cfg["theta_emb_dim"],
        dropout=cfg["dropout"],
        d_scale=cfg["D"],
    ).to(DEVICE)

    proj_params = list(model.proj[0].parameters())
    other_params = [
        p
        for n, p in model.named_parameters()
        if n not in {f"proj.0.{suffix}" for suffix in ["weight", "bias"]}
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": cfg["lr"] * 0.1},
            {"params": other_params, "lr": cfg["lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["num_epochs"],
        eta_min=1e-5,
    )

    best_val_bce = float("inf")
    best_val_auc = 0.0
    best_temp = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_bce": [], "val_auc": []}
    ckpt_path = output_dir / f"fold_{fold + 1}_best.pt"

    epoch_bar = tqdm(
        range(cfg["num_epochs"]),
        desc=f"  Fold {fold + 1:>2} Epochs",
        unit="ep",
        leave=False,
    )

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        for emb_q, emb_rat, feats, resp, mask, _ in train_loader:
            optimizer.zero_grad()
            P, a, b, theta = model(emb_q, emb_rat, feats)
            loss = irt_loss(
                P,
                a,
                b,
                theta,
                resp,
                mask,
                lambda_b=cfg["lambda_b"],
                lambda_theta=cfg["lambda_theta"],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        val_bce, val_auc = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_bce"].append(val_bce)
        history["val_auc"].append(val_auc)

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val_bce=f"{val_bce:.4f}",
            val_auc=f"{val_auc:.4f}",
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_bce = val_bce
            best_temp = torch.exp(model.log_temp).detach().item()
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        elif epoch >= cfg["min_epochs"]:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                break

    tqdm.write(
        f"[Fold {fold + 1}/{cfg['n_splits']}] "
        f"Best val_BCE={best_val_bce:.4f} | val_AUC={best_val_auc:.4f} | Temp={best_temp:.4f}"
    )

    return {
        "history": history,
        "best_val_bce": best_val_bce,
        "best_val_auc": best_val_auc,
        "ckpt_path": str(ckpt_path),
        "best_temp": best_temp,
    }


@torch.no_grad()
def compute_param_distribution(
    model: SemanticIRT2PL,
    all_emb_q: torch.Tensor,
    all_emb_rat: torch.Tensor,
    all_feats: torch.Tensor,
    all_responses: torch.Tensor,
    n_items: int,
    batch_size: int = 512,
    num_workers: int = 0,
) -> dict:
    """Runs inference on all items and summarizes the `a` and `b` distributions."""
    model.eval()
    all_indices = np.arange(n_items)
    loader = make_loader(
        all_emb_q,
        all_emb_rat,
        all_feats,
        all_responses,
        all_indices,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_a, all_b = [], []
    for emb_q, emb_rat, feats, _, _, _ in loader:
        a, b = model.get_item_params(emb_q, emb_rat, feats)
        all_a.append(a.cpu().numpy())
        all_b.append(b.cpu().numpy())

    a_arr = np.concatenate(all_a)
    b_arr = np.concatenate(all_b)

    def stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }

    return {
        "a": stats(a_arr),
        "b": stats(b_arr),
        "a_values": a_arr,
        "b_values": b_arr,
    }


def plot_fold_curves(fold_histories: list[dict], output_dir: Path):
    """Plots train loss, validation BCE, and validation AUC for each fold."""
    n = len(fold_histories)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, hist in enumerate(fold_histories):
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[i, 0].plot(epochs, hist["train_loss"], color="steelblue")
        axes[i, 0].set_title(f"Fold {i + 1} Train Loss")
        axes[i, 0].set_xlabel("Epoch")
        axes[i, 0].set_ylabel("Loss")

        axes[i, 1].plot(epochs, hist["val_bce"], color="tomato")
        axes[i, 1].set_title(f"Fold {i + 1} Val BCE")
        axes[i, 1].set_xlabel("Epoch")
        axes[i, 1].set_ylabel("BCE")

        axes[i, 2].plot(epochs, hist["val_auc"], color="seagreen")
        axes[i, 2].set_title(f"Fold {i + 1} Val AUC")
        axes[i, 2].set_xlabel("Epoch")
        axes[i, 2].set_ylabel("AUC")

    plt.tight_layout()
    plt.savefig(output_dir / "fold_curves.png", dpi=150)
    plt.close()


def plot_cv_summary(cv_results: list[dict], output_dir: Path):
    """Plots cross-validation summary boxplots for BCE and AUC."""
    bce_vals = [r["best_val_bce"] for r in cv_results]
    auc_vals = [r["best_val_auc"] for r in cv_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.boxplot(bce_vals, patch_artist=True, boxprops=dict(facecolor="tomato", alpha=0.6))
    ax1.set_title(f"Val BCE across folds\nmean={np.mean(bce_vals):.4f}+-{np.std(bce_vals):.4f}")
    ax1.set_ylabel("BCE")
    ax1.set_xticks([1])
    ax1.set_xticklabels(["10-fold"])

    ax2.boxplot(auc_vals, patch_artist=True, boxprops=dict(facecolor="seagreen", alpha=0.6))
    ax2.set_title(f"Val AUC across folds\nmean={np.mean(auc_vals):.4f}+-{np.std(auc_vals):.4f}")
    ax2.set_ylabel("AUC")
    ax2.set_xticks([1])
    ax2.set_xticklabels(["10-fold"])

    plt.tight_layout()
    plt.savefig(output_dir / "cv_summary.png", dpi=150)
    plt.close()


def plot_param_distribution(param_dist: dict, output_dir: Path):
    """Plots histograms for the inferred item parameter distributions."""
    a_arr = param_dist["a_values"]
    b_arr = param_dist["b_values"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(a_arr, bins=60, color="steelblue", alpha=0.8, edgecolor="white")
    ax1.axvline(a_arr.mean(), color="red", linestyle="--", label=f"mean={a_arr.mean():.3f}")
    ax1.set_title("Discrimination (a) Distribution")
    ax1.set_xlabel("a")
    ax1.set_ylabel("Count")
    ax1.legend()

    ax2.hist(b_arr, bins=60, color="darkorange", alpha=0.8, edgecolor="white")
    ax2.axvline(b_arr.mean(), color="red", linestyle="--", label=f"mean={b_arr.mean():.3f}")
    ax2.set_title("Difficulty (b) Distribution")
    ax2.set_xlabel("b")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "param_distribution.png", dpi=150)
    plt.close()


def train_cv(cfg: dict):
    """Runs full cross-validation, saves artifacts, and plots summary figures."""
    torch.manual_seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(cfg["h5_path"], "r") as f:
        n_items = f["items/question_embedding"].shape[0]
        n_models = f["responses/matrix"].shape[1]

        all_emb_q = torch.tensor(f["items/question_embedding"][:], dtype=torch.float32).to(DEVICE)
        all_emb_rat = torch.tensor(f["items/rationale_embedding"][:], dtype=torch.float32).to(DEVICE)
        all_resp = torch.tensor(f["responses/matrix"][:], dtype=torch.float32).to(DEVICE)

        feat_list = []
        for name in FEAT_NAMES:
            col = f[f"items/features/{name}"][:].astype(np.float32)
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col[nan_mask] = np.nanmean(col)
            feat_list.append(col)
        all_feats_np = np.stack(feat_list, axis=1)

    # Standardize scalar features only; keep task_type as an integer category.
    feats_scalar = all_feats_np[:, :-1]
    feat_mean = feats_scalar.mean(axis=0, keepdims=True)
    feat_std = feats_scalar.std(axis=0, keepdims=True).clip(min=1e-6)
    feats_scalar = (feats_scalar - feat_mean) / feat_std
    all_feats_np = np.concatenate([feats_scalar, all_feats_np[:, -1:]], axis=1)

    # Save normalization statistics so inference can reuse the same transform.
    torch.save(
        {
            "mean": torch.tensor(feat_mean, dtype=torch.float32),
            "std": torch.tensor(feat_std, dtype=torch.float32),
        },
        output_dir / "feat_norm.pt",
    )

    all_feats = torch.tensor(all_feats_np, dtype=torch.float32).to(DEVICE)

    print(f"Items: {n_items} | Models: {n_models}")
    print(f"Device: {DEVICE}")

    all_indices = np.arange(n_items)
    kf = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["random_seed"])

    cv_results = []
    fold_histories = []

    fold_bar = tqdm(
        enumerate(kf.split(all_indices)),
        total=cfg["n_splits"],
        desc="Cross-Validation",
        unit="fold",
    )

    for fold, (train_idx, val_idx) in fold_bar:
        result = train_fold(
            fold=fold,
            train_idx=all_indices[train_idx],
            val_idx=all_indices[val_idx],
            n_models=n_models,
            cfg=cfg,
            output_dir=output_dir,
            all_emb_q=all_emb_q,
            all_emb_rat=all_emb_rat,
            all_feats=all_feats,
            all_responses=all_resp,
        )
        cv_results.append(
            {
                "best_val_bce": result["best_val_bce"],
                "best_val_auc": result["best_val_auc"],
                "best_temp": result["best_temp"],
                "ckpt_path": result["ckpt_path"],
            }
        )
        fold_histories.append(result["history"])

    with open(output_dir / "fold_histories.json", "w", encoding="utf-8") as f:
        json.dump({f"fold_{i + 1}": h for i, h in enumerate(fold_histories)}, f, indent=2)

    with open(output_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump({f"fold_{i + 1}": r for i, r in enumerate(cv_results)}, f, indent=2)

    bce_list = [r["best_val_bce"] for r in cv_results]
    auc_list = [r["best_val_auc"] for r in cv_results]
    temp_list = [r["best_temp"] for r in cv_results]
    print("\n===== Cross-Validation Summary =====")
    for i, (bce, auc, temp) in enumerate(zip(bce_list, auc_list, temp_list)):
        print(f"  Fold {i + 1:>2} | val_BCE={bce:.4f} | val_AUC={auc:.4f} | Temp={temp:.4f}")
    print(f"  Mean val_BCE: {np.mean(bce_list):.4f} +- {np.std(bce_list):.4f}")
    print(f"  Mean val_AUC: {np.mean(auc_list):.4f} +- {np.std(auc_list):.4f}")
    print(f"  Mean temperature: {np.mean(temp_list):.4f} +- {np.std(temp_list):.4f}")

    best_fold_idx = int(np.argmax(auc_list))
    print(f"\nUsing fold {best_fold_idx + 1} checkpoint to summarize parameter distributions...")

    best_model = SemanticIRT2PL(
        n_models=n_models,
        emb_dim=4096,
        proj_dim=cfg["proj_dim"],
        n_scalar_feats=cfg["n_scalar_feats"],
        n_task_types=cfg["n_task_types"],
        task_emb_dim=cfg["task_emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        feat_dim=cfg["feat_dim"],
        theta_emb_dim=cfg["theta_emb_dim"],
        dropout=cfg["dropout"],
        d_scale=cfg["D"],
    ).to(DEVICE)
    best_model.load_state_dict(
        torch.load(cv_results[best_fold_idx]["ckpt_path"], map_location=DEVICE, weights_only=True)
    )
    best_model.eval()

    param_dist = compute_param_distribution(
        best_model,
        all_emb_q,
        all_emb_rat,
        all_feats,
        all_resp,
        n_items,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    param_dist_save = {
        "a": param_dist["a"],
        "b": param_dist["b"],
        "note": f"computed from fold_{best_fold_idx + 1} checkpoint",
    }
    with open(output_dir / "param_dist.json", "w", encoding="utf-8") as f:
        json.dump(param_dist_save, f, indent=2)

    print("\nParameter distribution (a):", param_dist["a"])
    print("Parameter distribution (b):", param_dist["b"])

    print("\nGenerating plots...")
    plot_fold_curves(fold_histories, output_dir)
    plot_cv_summary(cv_results, output_dir)
    plot_param_distribution(param_dist, output_dir)

    print(f"\n[Done] Saved all outputs to {output_dir}/")
    print("  fold_histories.json      - Per-epoch training history for each fold")
    print("  cv_results.json          - Best validation metrics for each fold")
    print("  param_dist.json          - Summary statistics for inferred a/b parameters")
    print("  feat_norm.pt             - Feature normalization statistics for inference")
    print("  fold_curves.png          - Training curves for each fold")
    print("  cv_summary.png           - Cross-validation summary boxplots")
    print("  param_distribution.png   - Histograms of inferred a/b parameters")
    print("  fold_*_best.pt           - Best checkpoint for each fold")

    return cv_results, fold_histories


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic 2PL IRT with item-level cross-validation")
    parser.add_argument("--h5", type=str, default=CFG["h5_path"], help="Path to the input HDF5 file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CFG["output_dir"],
        help="Directory for checkpoints, metrics, and plots",
    )
    parser.add_argument("--epochs", type=int, default=CFG["num_epochs"], help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=CFG["batch_size"], help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=CFG["lr"], help="Base learning rate")
    parser.add_argument("--n-splits", type=int, default=CFG["n_splits"], help="Number of CV folds")
    parser.add_argument("--num-workers", type=int, default=CFG["num_workers"], help="DataLoader workers")
    parser.add_argument("--d-scale", type=float, default=CFG["D"], help="2PL logistic scaling constant")
    args = parser.parse_args()

    CFG.update(
        {
            "h5_path": args.h5,
            "output_dir": args.output_dir,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_splits": args.n_splits,
            "num_workers": args.num_workers,
            "D": args.d_scale,
        }
    )

    train_cv(CFG)
