"""
scGPT + GEARS Fusion Adapter — illustrative architecture scaffold.

STATUS: UNVALIDATED SCAFFOLD, NOT THE PROJECT'S ACTIVE SUBMISSION PIPELINE.
The scGPT and GEARS MODELS are still mock stand-ins (`# REAL:` comments mark
where to swap in real model/graph loaders) - but the DATA pipeline is wired
to load real K562_gwps cells (real control-cell expression, real per-cell
perturbation labels, real measured-minus-control deltas) whenever
K562_GWPS_PATH exists on the machine running this, falling back to pure
synthetic data otherwise so the script stays runnable without any data staged.
Loading real data here is not the same as training against it at full scale -
this script caps at a small cell count for a quick smoke test either way (see
DEMO_N_CELLS in main()); scaling that up and actually training is a deliberate
choice for whoever deploys this, not something this script does on its own.
The project's validated, currently-submitted approach is the CPU-only
signature-transfer baseline in generate_predictions_gwps.py; that one has real
held-out validation numbers behind it. This script, model-wise, does not.

Architecture: a frozen scGPT cell encoder and a frozen GEARS perturbation-graph
encoder are each run ONCE (no_grad, cached to static tensors, then offloaded from
GPU). Only a lightweight 3-layer MLP "Fusion Adapter" is ever trained, taking the
concatenated [cell_embedding, perturbation_embedding] and predicting the gene
expression delta. Hyperparameters (shrinkage/clip applied to the predicted delta,
and the adapter's learning rate) are tuned via a 30-trial multi-objective Optuna
study (maximize Pearson delta correlation, minimize MSE) that reuses the cached
embeddings across every trial — scGPT/GEARS are never re-run during tuning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

SEED = 1594972177
N_CELLS = 360_000
N_GENES = 18_533
N_PERTURBATIONS = 300
SCGPT_EMBED_DIM = 512
GEARS_EMBED_DIM = 128
OUTPUT_DIR = Path(__file__).parent / "fusion_adapter_artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Real K562_gwps location once deployed (65.8GB - not present on a dev machine by
# default). Matches the verified schema: obs['gene'] holds the perturbation label
# ('non-targeting' for controls), var['gene_name'] holds gene symbols (var_names
# itself is Ensembl IDs, not symbols - confirmed against the real file earlier).
K562_GWPS_PATH = Path.home() / "vcc2026_cpu" / "train" / "K562_gwps_raw_singlecell_01.h5ad"
CONTROL_LABEL = "non-targeting"
MIN_CELLS_PER_PERT = 15


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Mock model/data stand-ins. Each is a drop-in placeholder for a real component.
# Swap these for real scGPT/GEARS loaders to deploy against actual data.
# ---------------------------------------------------------------------------

class MockScGPT(nn.Module):
    """
    Stand-in for a frozen, pretrained scGPT cell encoder.

    REAL: load the pretrained scGPT checkpoint and its vocabulary, tokenize raw
    control-cell expression counts the way scGPT expects, and take the [CLS]/
    pooled hidden state as the cell embedding. See scGPT's `load_pretrained`
    and `TransformerModel.encode` for the real equivalents.
    """

    def __init__(self, n_genes: int = N_GENES, embed_dim: int = SCGPT_EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(n_genes, embed_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, raw_counts: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(raw_counts))


class MockGEARS(nn.Module):
    """
    Stand-in for a frozen GEARS perturbation-graph encoder (GO + STRING graph
    message passing over the perturbed-gene node).

    REAL: build the combined GO/STRING graph, run GEARS' own GNN layers (see
    gears/model_new.py's graph convolution stack) over the target-gene node and
    its neighborhood, and take the resulting node embedding.
    """

    def __init__(self, n_perturbations: int = N_PERTURBATIONS, embed_dim: int = GEARS_EMBED_DIM):
        super().__init__()
        self.node_embed = nn.Embedding(n_perturbations, embed_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pert_idx: torch.Tensor) -> torch.Tensor:
        return self.node_embed(pert_idx)


def generate_mock_raw_data(n_cells: int, n_genes: int, n_perturbations: int, seed: int = SEED):
    """
    Pure-synthetic fallback, used only when K562_GWPS_PATH isn't present (e.g.
    for a quick structural smoke test on a laptop with no real data staged).
    Prefer load_real_k562_gwps_data below whenever the real file is available.
    """
    rng = np.random.default_rng(seed)
    raw_control_counts = rng.poisson(lam=1.5, size=(n_cells, n_genes)).astype(np.float32)
    pert_idx = rng.integers(0, n_perturbations, size=n_cells)
    true_delta = rng.normal(loc=0.0, scale=0.3, size=(n_cells, n_genes)).astype(np.float32)
    return raw_control_counts, pert_idx, true_delta


def load_real_k562_gwps_data(h5ad_path: Path, n_perturbations: int,
                              max_cells: int | None = None, block: int = 512, seed: int = SEED):
    """
    Loads real K562_gwps data (verified schema: obs['gene'] holds the
    perturbation label, 'non-targeting' for controls; var['gene_name'] holds
    gene symbols - var_names itself is Ensembl IDs, confirmed against the real
    file earlier). Restricts to the n_perturbations best-represented genes, so
    the fixed-size GEARS embedding table covers real, well-supported
    perturbations. true_delta is each cell's own real measured expression
    minus the real control-cell baseline - actual biological signal, not
    synthetic noise.

    Not run against the full 65.8GB file as part of this script's own test
    pass (that requires the file staged on a pod/VM - deployment is left to
    whoever runs this for real, per the "don't deploy" instruction this was
    written under).
    """
    adata = ad.read_h5ad(h5ad_path, backed='r')
    labels = adata.obs['gene'].astype(str).to_numpy()
    gene_symbols = adata.var['gene_name'].astype(str).to_numpy()
    n_genes = adata.n_vars

    non_ctrl_labels = labels[labels != CONTROL_LABEL]
    groups, counts = np.unique(non_ctrl_labels, return_counts=True)
    groups, counts = groups[counts >= MIN_CELLS_PER_PERT], counts[counts >= MIN_CELLS_PER_PERT]
    top_perts = groups[np.argsort(-counts)][:n_perturbations]
    pert_to_idx = {g: i for i, g in enumerate(top_perts)}

    keep_idx = np.flatnonzero(np.isin(labels, top_perts))
    rng = np.random.default_rng(seed)
    if max_cells is not None and len(keep_idx) > max_cells:
        keep_idx = rng.choice(keep_idx, size=max_cells, replace=False)
    keep_idx = np.sort(keep_idx)

    ctrl_idx = np.flatnonzero(labels == CONTROL_LABEL)
    baseline = np.zeros(n_genes, dtype=np.float64)
    for i in range(0, len(ctrl_idx), block):
        chunk = adata.X[ctrl_idx[i:i + block], :]
        baseline += np.asarray(chunk.sum(axis=0)).ravel() if sp.issparse(chunk) else np.asarray(chunk).sum(axis=0)
    baseline /= max(len(ctrl_idx), 1)

    raw_rows = []
    for i in range(0, len(keep_idx), block):
        chunk = adata.X[keep_idx[i:i + block], :]
        raw_rows.append(chunk.toarray() if sp.issparse(chunk) else np.asarray(chunk))
    raw_counts = (np.vstack(raw_rows) if raw_rows else np.zeros((0, n_genes))).astype(np.float32)

    pert_idx = np.array([pert_to_idx[labels[i]] for i in keep_idx], dtype=np.int64)
    true_delta = (raw_counts - baseline[None, :]).astype(np.float32)

    adata.file.close()
    return raw_counts, pert_idx, true_delta, gene_symbols


# ---------------------------------------------------------------------------
# Step 1 & 2: precompute and cache embeddings ONCE, outside the tuning loop.
# This is what makes the Optuna trials cheap — scGPT/GEARS never run again
# after this point.
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_scgpt_embeddings(raw_counts: np.ndarray, device: str, batch_size: int = 2048) -> torch.Tensor:
    model = MockScGPT(n_genes=raw_counts.shape[1]).to(device).eval()
    chunks = []
    for i in range(0, len(raw_counts), batch_size):
        batch = torch.from_numpy(raw_counts[i:i + batch_size]).to(device)
        chunks.append(model(batch).cpu())
    embeddings = torch.cat(chunks, dim=0)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return embeddings  # static tensor, no grad history, model offloaded


@torch.no_grad()
def precompute_gears_embeddings(n_perturbations: int, device: str) -> torch.Tensor:
    model = MockGEARS(n_perturbations=n_perturbations).to(device).eval()
    all_idx = torch.arange(n_perturbations, device=device)
    embeddings = model(all_idx).cpu()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return embeddings  # static tensor, one row per perturbation, model offloaded


# ---------------------------------------------------------------------------
# Dataset pairing cached static embeddings - no scGPT/GEARS calls happen here.
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    def __init__(self, cell_embeddings: torch.Tensor, pert_idx: np.ndarray,
                 pert_embeddings: torch.Tensor, target_delta: np.ndarray):
        self.cell_embeddings = cell_embeddings
        self.pert_idx = torch.from_numpy(pert_idx).long()
        self.pert_embeddings = pert_embeddings
        self.target_delta = torch.from_numpy(target_delta).float()

    def __len__(self) -> int:
        return self.cell_embeddings.shape[0]

    def __getitem__(self, i: int):
        return (
            self.cell_embeddings[i],
            self.pert_embeddings[self.pert_idx[i]],
            self.target_delta[i],
        )


# ---------------------------------------------------------------------------
# The only trainable component: a lightweight 3-layer MLP fusion adapter.
# ---------------------------------------------------------------------------

class FusionAdapter(nn.Module):
    def __init__(self, cell_dim: int, pert_dim: int, hidden_dim: int, n_genes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cell_dim + pert_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, cell_embed: torch.Tensor, pert_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cell_embed, pert_embed], dim=-1))


# ---------------------------------------------------------------------------
# Leave-one-context-out style split. Real data for this project has only a
# single context (K562), so this is implemented as a grouped split by
# perturbation identity instead - held-out perturbations never appear in the
# train fold, which is the generalization test that actually applies here.
# REAL: if deploying against genuinely multi-context data, split by the
# context label directly instead of by perturbation group.
# ---------------------------------------------------------------------------

def grouped_perturbation_split(pert_idx: np.ndarray, n_perturbations: int,
                                held_out_fraction: float = 0.2, seed: int = SEED):
    rng = np.random.default_rng(seed)
    all_perts = np.arange(n_perturbations)
    rng.shuffle(all_perts)
    n_held_out = max(1, int(n_perturbations * held_out_fraction))
    held_out_perts = set(all_perts[:n_held_out].tolist())
    val_mask = np.isin(pert_idx, list(held_out_perts))
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def compute_vcc_metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline_control: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    delta_true = (y_true - baseline_control).flatten()
    delta_pred = (y_pred - baseline_control).flatten()
    pearson_delta, _ = pearsonr(delta_true, delta_pred)
    return {"MSE": mse, "Pearson_Delta": pearson_delta}


@dataclass
class CachedTensors:
    cell_embeddings: torch.Tensor
    pert_idx: np.ndarray
    pert_embeddings: torch.Tensor
    true_delta: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray


def train_adapter(cache: CachedTensors, shrinkage: float, clip_log_fold: float,
                   adapter_lr: float, device: str, epochs: int = 5, batch_size: int = 512) -> dict:
    """
    Trains ONLY the FusionAdapter's parameters. cell_embeddings and
    pert_embeddings are plain cached tensors (no grad history from precompute,
    since that ran under @torch.no_grad()) - so no additional no_grad wrapping
    is needed here, the isolation already happened upstream.
    """
    train_ds = FusionDataset(
        cache.cell_embeddings[cache.train_idx], cache.pert_idx[cache.train_idx],
        cache.pert_embeddings, cache.true_delta[cache.train_idx],
    )
    val_ds = FusionDataset(
        cache.cell_embeddings[cache.val_idx], cache.pert_idx[cache.val_idx],
        cache.pert_embeddings, cache.true_delta[cache.val_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    adapter = FusionAdapter(
        cell_dim=cache.cell_embeddings.shape[1], pert_dim=cache.pert_embeddings.shape[1],
        hidden_dim=256, n_genes=cache.true_delta.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=adapter_lr)  # only adapter params
    loss_fn = nn.MSELoss()

    adapter.train()
    for _ in range(epochs):
        for cell_embed, pert_embed, target in train_loader:
            cell_embed, pert_embed, target = cell_embed.to(device), pert_embed.to(device), target.to(device)
            optimizer.zero_grad()
            pred_delta = adapter(cell_embed, pert_embed)
            pred_delta = torch.clamp(pred_delta, -clip_log_fold, clip_log_fold) * shrinkage
            loss = loss_fn(pred_delta, target)
            loss.backward()
            optimizer.step()

    adapter.eval()
    with torch.no_grad():
        val_cell = cache.cell_embeddings[cache.val_idx].to(device)
        val_pert = cache.pert_embeddings[cache.pert_idx[cache.val_idx]].to(device)
        pred_delta = adapter(val_cell, val_pert)
        pred_delta = torch.clamp(pred_delta, -clip_log_fold, clip_log_fold) * shrinkage
        pred_delta = pred_delta.cpu().numpy()

    y_true = cache.true_delta[cache.val_idx]
    baseline_control = np.zeros_like(y_true)  # delta is already baseline-relative in this mock setup
    y_pred = pred_delta
    return compute_vcc_metrics(y_true + baseline_control, y_pred + baseline_control, baseline_control)


def build_objective(cache: CachedTensors, device: str):
    def objective(trial: optuna.Trial):
        seed_everything(SEED + trial.number)  # reproducible per-trial seeding
        shrinkage = trial.suggest_float("shrinkage", 0.01, 0.50, step=0.05)
        clip_log_fold = trial.suggest_float("clip_log_fold", 1.0, 5.0)
        adapter_lr = trial.suggest_float("adapter_lr", 1e-4, 1e-2, log=True)

        metrics = train_adapter(cache, shrinkage, clip_log_fold, adapter_lr, device)
        return metrics["Pearson_Delta"], metrics["MSE"]

    return objective


def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # N_CELLS/N_GENES document the REAL competition shape (360,000 x 18,533).
    # Generating mock data at that literal scale needs ~50GB+ RAM for two dense
    # arrays that size - impractical for a demo run, so the synthetic fallback
    # below caps at DEMO_N_CELLS regardless of which path is used.
    DEMO_N_CELLS = 2000
    if K562_GWPS_PATH.exists():
        print(f"Found real K562_gwps data at {K562_GWPS_PATH} - loading real cells "
              f"(capped at {DEMO_N_CELLS} for a reasonable smoke-test runtime; "
              f"remove max_cells for a full real run)...")
        raw_counts, pert_idx, true_delta, real_gene_symbols = load_real_k562_gwps_data(
            K562_GWPS_PATH, N_PERTURBATIONS, max_cells=DEMO_N_CELLS,
        )
        n_genes_actual = raw_counts.shape[1]
        print(f"Loaded {raw_counts.shape[0]} real cells, {n_genes_actual} real genes "
              f"(scGPT encoder below sized to this, not the hardcoded N_GENES constant).")
    else:
        print(f"Real K562_gwps data not found at {K562_GWPS_PATH} (expected - not staged "
              f"on this machine). Generating mock raw data at demo scale ({DEMO_N_CELLS} "
              f"cells) instead...")
        raw_counts, pert_idx, true_delta = generate_mock_raw_data(DEMO_N_CELLS, N_GENES, N_PERTURBATIONS)
        n_genes_actual = N_GENES

    print("Precomputing scGPT cell embeddings (once, no_grad, then offloading model)...")
    cell_embeddings = precompute_scgpt_embeddings(raw_counts, device)

    print("Precomputing GEARS perturbation-graph embeddings (once, no_grad, then offloading model)...")
    pert_embeddings = precompute_gears_embeddings(N_PERTURBATIONS, device)

    train_idx, val_idx = grouped_perturbation_split(pert_idx, N_PERTURBATIONS)
    print(f"Train cells: {len(train_idx)}, held-out-perturbation val cells: {len(val_idx)}")

    cache = CachedTensors(cell_embeddings, pert_idx, pert_embeddings, true_delta, train_idx, val_idx)

    print("\nRunning 30-trial multi-objective Optuna study "
          "(maximize Pearson_Delta, minimize MSE)...")
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(build_objective(cache, device), n_trials=30)

    trials_df = study.trials_dataframe()
    csv_path = OUTPUT_DIR / "optuna_trial_history.csv"
    trials_df.to_csv(csv_path, index=False)
    print(f"\nSaved trial history to {csv_path}")

    print("\n=== Pareto-optimal trials (maximize Pearson_Delta, minimize MSE) ===")
    for t in study.best_trials:
        print(f"  trial {t.number}: Pearson_Delta={t.values[0]:.4f}, MSE={t.values[1]:.4f}, params={t.params}")

    print("\nDONE (mock data - see module docstring before drawing any real conclusions)")


if __name__ == "__main__":
    main()
