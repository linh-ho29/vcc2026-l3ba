"""Training loop for VirtualCellModel: AdamW + CosineAnnealingLR, mixed
precision, SWD + MSE loss. See models/virtual_cell.py for the model and
losses/swd.py for the distribution loss.

Dataset: ``PerturbSeqDataset`` groups perturbed cells by
``(gene, cell_line)`` condition via ``obs_df``, and samples NTC control
cells strictly from the SAME cell line, unpaired (not row-matched to any
particular perturbed cell). This replaces an earlier version of this file
that guessed a single cell-line-agnostic NTC pool - that guess was wrong;
NTC cells are per-cell-line.

Dataset schema (what ``load_dataset_from_disk`` expects on disk):

- ``gears_pert_bank.pt``: ``Dict[gene_id, Tensor[4, 64]]`` - one entry per
  gene, already covering all 4 cell lines (K562, HepG2, RPE, Jurkat order).
- ``stat_delta_bank.pt``: ``Dict[(gene_id, cell_line), Tensor[num_genes]]``.
- ``ntc_latents.pt`` / ``ntc_expression.pt``: ``Dict[cell_line,
  Tensor[N_ctrl_c, d_latent/num_genes]]`` - one pool per cell line.
- ``pert_latents.pt`` / ``pert_expression.pt``: flat ``Tensor[N_total_pert,
  d_latent/num_genes]``, row-aligned with each other and with ``obs.pkl``.
- ``obs.pkl``: a pickled ``pandas.DataFrame`` with (at least) ``gene`` and
  ``cell_line`` columns, one row per perturbed cell, same row order as
  ``pert_latents``/``pert_expression``. Not part of the task spec's
  original file list - added here because grouping perturbed cells by
  condition needs it; stated explicitly rather than assumed silently.
- ``c_target``: NOT read from disk - a learned ``nn.Embedding(4, d_context)``
  over the 4 cell lines, trained jointly with the rest of the model (per
  the spec's "Learned target cell line embedding").
"""

from __future__ import annotations

import pickle
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import Tensor, nn

from losses.swd import sliced_wasserstein_distance
from models.virtual_cell import VirtualCellModel

CELL_LINES = ["K562", "HepG2", "RPE", "Jurkat"]
NUM_CELL_LINES = len(CELL_LINES)
CELL_LINE_TO_IDX = {line: i for i, line in enumerate(CELL_LINES)}


@dataclass
class TrainConfig:
    num_genes: int
    d_latent: int = 128
    d_context: int = 32
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    lambda_swd: float = 0.1
    n_projections: int = 32
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True


class PerturbSeqDataset:
    """Groups perturbed cells by ``(gene, cell_line)`` condition; samples
    unpaired NTC control cells strictly from the same cell line.

    Not a ``torch.utils.data.Dataset`` - there's no meaningful per-index
    ``__getitem__`` here (a "batch" is scoped to one condition, not one
    row), so the training loop below iterates ``condition_keys`` directly
    and calls :meth:`sample_batch` instead of going through a DataLoader.

    ``gears_pert_bank``/``stat_delta_bank`` are NOT stored on this class -
    they're keyed by ``gene``/``(gene, cell_line)`` only, constant within
    a batch, so the caller looks them up once per condition rather than
    broadcasting them into every row here.
    """

    def __init__(
        self,
        ntc_latents: Dict[str, Tensor],
        ntc_expr: Dict[str, Tensor],
        pert_latents: Tensor,
        pert_expr: Tensor,
        obs_df: pd.DataFrame,
        gears_pert_bank: Dict[str, Tensor],
        stat_delta_bank: Dict[Tuple[str, str], Tensor],
    ) -> None:
        self.ntc_latents = ntc_latents
        self.ntc_expr = ntc_expr
        self.pert_latents = pert_latents
        self.pert_expr = pert_expr

        groups = obs_df.groupby(["gene", "cell_line"]).indices
        # Keep only conditions with a GEARS vector, a statistical baseline,
        # and an NTC pool for their cell line - never silently drop a row's
        # dependency and substitute something else, just exclude the
        # condition up front with a visible count.
        self.condition_keys: List[Tuple[str, str]] = [
            (gene, cell_line) for (gene, cell_line) in groups.keys()
            if gene in gears_pert_bank
            and (gene, cell_line) in stat_delta_bank
            and cell_line in ntc_latents and cell_line in ntc_expr
        ]
        n_dropped = len(groups) - len(self.condition_keys)
        if n_dropped:
            print(f"PerturbSeqDataset: dropped {n_dropped}/{len(groups)} conditions "
                  f"missing a GEARS vector, statistical baseline, or NTC pool.")
        if not self.condition_keys:
            raise ValueError("No usable (gene, cell_line) conditions after filtering.")

        self.groups = {k: groups[k] for k in self.condition_keys}
        self.gears_pert_bank = gears_pert_bank
        self.stat_delta_bank = stat_delta_bank

    def __len__(self) -> int:
        return len(self.condition_keys)

    def sample_batch(self, gene: str, cell_line: str, batch_size: int = 256) -> Dict[str, Tensor]:
        # 1. True perturbed cells for this condition.
        pert_indices = self.groups[(gene, cell_line)]
        chosen_pert = torch.randint(0, len(pert_indices), (batch_size,))
        idx_p = pert_indices[chosen_pert.numpy()]

        y_true = self.pert_expr[idx_p]
        z_true = self.pert_latents[idx_p]

        # 2. Unpaired NTC control cells, strictly from the SAME cell line.
        ctrl_pool_size = self.ntc_latents[cell_line].shape[0]
        chosen_ctrl = torch.randint(0, ctrl_pool_size, (batch_size,))

        z_0 = self.ntc_latents[cell_line][chosen_ctrl]
        x_ctrl = self.ntc_expr[cell_line][chosen_ctrl]

        # 3. Condition-level inputs (constant across the batch): broadcast
        # the single GEARS vector / statistical delta for this condition.
        e_p_stack = self.gears_pert_bank[gene].unsqueeze(0).expand(batch_size, -1, -1)  # [B,4,64]
        delta_stat = self.stat_delta_bank[(gene, cell_line)].unsqueeze(0).expand(batch_size, -1)  # [B,G]

        return {
            "z_0": z_0, "x_ctrl": x_ctrl, "z_true": z_true, "y_true": y_true,
            "e_p_stack": e_p_stack, "delta_stat": delta_stat,
            "cell_line_idx": torch.full((batch_size,), CELL_LINE_TO_IDX[cell_line], dtype=torch.long),
        }


def load_dataset_from_disk(data_dir: Path) -> PerturbSeqDataset:
    data_dir = Path(data_dir)
    with open(data_dir / "obs.pkl", "rb") as f:
        obs_df = pickle.load(f)
    return PerturbSeqDataset(
        ntc_latents=torch.load(data_dir / "ntc_latents.pt"),
        ntc_expr=torch.load(data_dir / "ntc_expression.pt"),
        pert_latents=torch.load(data_dir / "pert_latents.pt"),
        pert_expr=torch.load(data_dir / "pert_expression.pt"),
        obs_df=obs_df,
        gears_pert_bank=torch.load(data_dir / "gears_pert_bank.pt"),
        stat_delta_bank=torch.load(data_dir / "stat_delta_bank.pt"),
    )


def compute_loss(
    out: Dict[str, Tensor], y_true: Tensor, z_true: Tensor,
    lambda_swd: float, n_projections: int,
) -> Dict[str, Tensor]:
    loss_recon = nn.functional.mse_loss(out["y_pred"], y_true)
    loss_swd = sliced_wasserstein_distance(out["z_pert"], z_true, n_projections=n_projections)
    loss = loss_recon + lambda_swd * loss_swd
    return {"loss": loss, "loss_recon": loss_recon, "loss_swd": loss_swd}


def train(config: TrainConfig, dataset: PerturbSeqDataset) -> VirtualCellModel:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device(config.device)
    model = VirtualCellModel(
        num_genes=config.num_genes, d_latent=config.d_latent, d_context=config.d_context,
    ).to(device)
    context_embedding = nn.Embedding(NUM_CELL_LINES, config.d_context).to(device)

    params = list(model.parameters()) + list(context_embedding.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    conditions = list(dataset.condition_keys)  # one "batch" per condition per epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs * max(1, len(conditions))
    )

    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    def autocast_ctx():
        if use_amp:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

    model.train()
    for epoch in range(config.max_epochs):
        random.shuffle(conditions)
        epoch_loss = 0.0
        n_batches = 0
        for gene, cell_line in conditions:
            batch = dataset.sample_batch(gene, cell_line, batch_size=config.batch_size)
            e_p_stack = batch["e_p_stack"].to(device)
            cell_line_idx = batch["cell_line_idx"].to(device)
            delta_stat = batch["delta_stat"].to(device)
            z_0 = batch["z_0"].to(device)
            x_ctrl = batch["x_ctrl"].to(device)
            y_true = batch["y_true"].to(device)
            z_true = batch["z_true"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                c_target = context_embedding(cell_line_idx)
                out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
                losses = compute_loss(out, y_true, z_true, config.lambda_swd, config.n_projections)
                loss = losses["loss"]

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch + 1}/{config.max_epochs}  loss={epoch_loss / max(1, n_batches):.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    model.context_embedding = context_embedding  # attach for inference-time reuse
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--num-genes", type=int, required=True)
    parser.add_argument("--d-latent", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()

    dataset = load_dataset_from_disk(args.data_dir)
    config = TrainConfig(
        num_genes=args.num_genes, d_latent=args.d_latent,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
    )
    train(config, dataset)
