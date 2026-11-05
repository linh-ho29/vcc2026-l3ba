"""Training loop for VirtualCellModel: AdamW + CosineAnnealingLR, mixed
precision, SWD + MSE loss. See models/virtual_cell.py for the model and
losses/swd.py for the distribution loss.

Dataset schema assumptions (the task spec leaves these implicit - stated
explicitly here, not guessed silently, so they're easy to correct once the
real files exist):

- ``gears_pert_bank.pt``: ``Dict[gene_id, Tensor[4, 64]]`` - one entry per
  gene, already covering all 4 cell lines (K562, HepG2, RPE, Jurkat order).
- ``stat_delta_bank.pt``, ``pert_cells.pt``, ``pert_latents.pt``: all three
  keyed the same way, ``Dict[(gene_id, cell_line_idx), Tensor]`` -
  ``stat_delta_bank[(g,c)]`` is ``[num_genes]`` (the empirical pseudobulk
  mean of exactly the cells in ``pert_cells[(g,c)]``); ``pert_cells[(g,c)]``
  is ``[n_cells, num_genes]``; ``pert_latents[(g,c)]`` is ``[n_cells,
  d_latent]``, row-aligned with ``pert_cells[(g,c)]``.
- ``ntc_latents.pt`` / ``ntc_expression.pt``: a single shared pool of
  control cells, not pre-split by cell line. ``z_0``/``x_ctrl`` for a
  training example are sampled uniformly at random from this pool,
  independent of the example's target cell line - control-cell state is
  being treated as cell-line-agnostic at the latent level here. If the
  real NTC pool is actually per-cell-line, swap
  ``PerturbationDataset._sample_ntc`` for a per-line lookup; nothing else
  changes.
- ``c_target``: NOT read from disk - a learned ``nn.Embedding(4, d_context)``
  over the 4 cell lines, trained jointly with the rest of the model (per
  the spec's "Learned target cell line embedding").
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from losses.swd import sliced_wasserstein_distance
from models.virtual_cell import VirtualCellModel

CELL_LINES = ["K562", "HepG2", "RPE", "Jurkat"]
NUM_CELL_LINES = len(CELL_LINES)


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


class PerturbationDataset(Dataset):
    """One example = one (perturbed cell, random NTC control cell) pair.

    ``len(dataset)`` is the total number of perturbed cells across every
    (gene, cell_line) key in ``pert_cells`` - every real perturbed cell is
    visited (in expectation, once per epoch), not just one per gene.
    """

    def __init__(
        self,
        ntc_latents: Tensor,
        ntc_expression: Tensor,
        gears_pert_bank: Dict[str, Tensor],
        stat_delta_bank: Dict[Tuple[str, int], Tensor],
        pert_cells: Dict[Tuple[str, int], Tensor],
        pert_latents: Dict[Tuple[str, int], Tensor],
    ) -> None:
        self.ntc_latents = ntc_latents
        self.ntc_expression = ntc_expression
        self.gears_pert_bank = gears_pert_bank
        self.stat_delta_bank = stat_delta_bank
        self.pert_cells = pert_cells
        self.pert_latents = pert_latents

        self.index: List[Tuple[str, int, int]] = []  # (gene_id, cell_line_idx, row_in_pert_cells)
        for (gene_id, cell_line_idx), cells in pert_cells.items():
            if gene_id not in gears_pert_bank:
                continue  # no GEARS vector for this gene - cannot build e_p_stack, skip
            if (gene_id, cell_line_idx) not in stat_delta_bank:
                continue  # no statistical baseline for this (gene, line) - skip
            for row in range(cells.shape[0]):
                self.index.append((gene_id, cell_line_idx, row))

        if not self.index:
            raise ValueError(
                "No usable (gene, cell_line, cell) triples: every entry in pert_cells "
                "was missing either a gears_pert_bank or stat_delta_bank match."
            )

    def __len__(self) -> int:
        return len(self.index)

    def _sample_ntc(self) -> Tuple[Tensor, Tensor]:
        i = random.randrange(self.ntc_latents.shape[0])
        return self.ntc_latents[i], self.ntc_expression[i]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        gene_id, cell_line_idx, row = self.index[idx]
        e_p_stack = self.gears_pert_bank[gene_id]                       # [4, 64]
        delta_stat = self.stat_delta_bank[(gene_id, cell_line_idx)]      # [num_genes]
        y_true = self.pert_cells[(gene_id, cell_line_idx)][row]         # [num_genes]
        z_true = self.pert_latents[(gene_id, cell_line_idx)][row]       # [d_latent]
        z_0, x_ctrl = self._sample_ntc()                                # [d_latent], [num_genes]

        return {
            "e_p_stack": e_p_stack,
            "cell_line_idx": torch.tensor(cell_line_idx, dtype=torch.long),
            "delta_stat": delta_stat,
            "z_0": z_0,
            "x_ctrl": x_ctrl,
            "y_true": y_true,
            "z_true": z_true,
        }


def load_dataset_from_disk(data_dir: Path) -> PerturbationDataset:
    data_dir = Path(data_dir)
    return PerturbationDataset(
        ntc_latents=torch.load(data_dir / "ntc_latents.pt"),
        ntc_expression=torch.load(data_dir / "ntc_expression.pt"),
        gears_pert_bank=torch.load(data_dir / "gears_pert_bank.pt"),
        stat_delta_bank=torch.load(data_dir / "stat_delta_bank.pt"),
        pert_cells=torch.load(data_dir / "pert_cells.pt"),
        pert_latents=torch.load(data_dir / "pert_latents.pt"),
    )


def compute_loss(
    out: Dict[str, Tensor], y_true: Tensor, z_true: Tensor,
    lambda_swd: float, n_projections: int,
) -> Dict[str, Tensor]:
    loss_recon = nn.functional.mse_loss(out["y_pred"], y_true)
    loss_swd = sliced_wasserstein_distance(out["z_pert"], z_true, n_projections=n_projections)
    loss = loss_recon + lambda_swd * loss_swd
    return {"loss": loss, "loss_recon": loss_recon, "loss_swd": loss_swd}


def train(config: TrainConfig, dataset: PerturbationDataset) -> VirtualCellModel:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device(config.device)
    model = VirtualCellModel(
        num_genes=config.num_genes, d_latent=config.d_latent, d_context=config.d_context,
    ).to(device)
    context_embedding = nn.Embedding(NUM_CELL_LINES, config.d_context).to(device)

    params = list(model.parameters()) + list(context_embedding.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs * max(1, len(loader))
    )

    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    def autocast_ctx():
        if use_amp:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

    model.train()
    for epoch in range(config.max_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
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
