"""Training loop for VirtualCellModel: AdamW + configurable LR schedule,
mixed precision, weighted MSE + SWD loss. See models/virtual_cell.py for
the model, losses/swd.py for the distribution loss, and src/config.py for
the full ModelConfig/TrainingConfig schema driving both.

Dataset: ``PerturbSeqDataset`` groups perturbed cells by
``(gene, cell_line)`` condition via ``obs_df``, and samples NTC control
cells strictly from the SAME cell line, unpaired (not row-matched to any
particular perturbed cell).

Dataset schema (what ``load_dataset_from_disk`` expects on disk):

- ``gears_pert_bank.pt``: ``Dict[gene_id, Tensor[4, d_gears]]`` - one entry
  per gene, covering all 4 cell lines (K562, HepG2, RPE, Jurkat order).
- ``stat_delta_bank.pt``: ``Dict[(gene_id, cell_line), Tensor[num_genes]]``.
- ``ntc_latents.pt`` / ``ntc_expression.pt``: ``Dict[cell_line,
  Tensor[N_ctrl_c, d_latent/num_genes]]`` - one pool per cell line.
- ``pert_latents.pt`` / ``pert_expression.pt``: flat ``Tensor[N_total_pert,
  d_latent/num_genes]``, row-aligned with each other and with ``obs.pkl``.
- ``obs.pkl``: a pickled ``pandas.DataFrame`` with (at least) ``gene`` and
  ``cell_line`` columns, one row per perturbed cell, same row order as
  ``pert_latents``/``pert_expression``.
- ``c_target``: NOT read from disk - a learned ``nn.Embedding(num_cell_lines,
  d_context)`` over the cell lines, trained jointly with the rest of the model.
"""

from __future__ import annotations

import pickle
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor, nn

from losses.swd import sliced_wasserstein_distance
from models.virtual_cell import VirtualCellModel
from src.config import ModelConfig, TrainingConfig

CELL_LINES = ["K562", "HepG2", "RPE", "Jurkat"]  # example/default only - see PerturbSeqDataset.cell_lines


class PerturbSeqDataset:
    """Groups perturbed cells by ``(gene, cell_line)`` condition; samples
    unpaired NTC control cells strictly from the same cell line.

    ``cell_lines``/``cell_line_to_idx`` are derived from whatever cell-line
    labels actually appear in ``obs_df`` - NOT the module-level ``CELL_LINES``
    constant, which is just a 4-line example. A dataset built with a 5th
    (or differently-named) context works without any code change here.

    Not a ``torch.utils.data.Dataset`` - there's no meaningful per-index
    ``__getitem__`` here (a "batch" is scoped to one condition, not one
    row), so the training loop below iterates ``condition_keys`` directly
    and calls :meth:`sample_batch` instead of going through a DataLoader.
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

        self.cell_lines: List[str] = sorted({cl for _, cl in self.condition_keys})
        self.cell_line_to_idx: Dict[str, int] = {cl: i for i, cl in enumerate(self.cell_lines)}

    def __len__(self) -> int:
        return len(self.condition_keys)

    def sample_batch(
        self, gene: str, cell_line: str, batch_size: int = 256,
        cell_line_to_idx: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Tensor]:
        """``cell_line_to_idx``: override the index mapping (e.g. pass the
        TRAINING dataset's mapping when sampling from a held-out/val dataset,
        so both share the same context_embedding row per cell line even if
        this dataset's own ``cell_lines`` differ in membership or order)."""
        idx_map = cell_line_to_idx if cell_line_to_idx is not None else self.cell_line_to_idx
        pert_indices = self.groups[(gene, cell_line)]
        chosen_pert = torch.randint(0, len(pert_indices), (batch_size,))
        idx_p = pert_indices[chosen_pert.numpy()]

        y_true = self.pert_expr[idx_p]
        z_true = self.pert_latents[idx_p]

        ctrl_pool_size = self.ntc_latents[cell_line].shape[0]
        chosen_ctrl = torch.randint(0, ctrl_pool_size, (batch_size,))
        z_0 = self.ntc_latents[cell_line][chosen_ctrl]
        x_ctrl = self.ntc_expr[cell_line][chosen_ctrl]

        e_p_stack = self.gears_pert_bank[gene].unsqueeze(0).expand(batch_size, -1, -1)
        delta_stat = self.stat_delta_bank[(gene, cell_line)].unsqueeze(0).expand(batch_size, -1)

        return {
            "z_0": z_0, "x_ctrl": x_ctrl, "z_true": z_true, "y_true": y_true,
            "e_p_stack": e_p_stack, "delta_stat": delta_stat,
            "cell_line_idx": torch.full((batch_size,), idx_map[cell_line], dtype=torch.long),
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
    out: Dict[str, Tensor], y_true: Tensor, z_true: Tensor, train_cfg: TrainingConfig,
) -> Dict[str, Tensor]:
    loss_recon = nn.functional.mse_loss(out["y_pred"], y_true)
    loss_swd = sliced_wasserstein_distance(out["z_pert"], z_true, n_projections=train_cfg.swd_num_projections)
    loss = train_cfg.lambda_recon * loss_recon + train_cfg.lambda_swd * loss_swd
    return {"loss": loss, "loss_recon": loss_recon, "loss_swd": loss_swd}


def _make_scheduler(optimizer, train_cfg: TrainingConfig, steps_per_epoch: int):
    total_steps = train_cfg.max_epochs * max(1, steps_per_epoch)
    if train_cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if train_cfg.lr_scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_cfg.lr, total_steps=total_steps)
    if train_cfg.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    raise ValueError(f"Unknown lr_scheduler {train_cfg.lr_scheduler!r}; use 'cosine', 'onecycle', or 'constant'.")


def train(
    model_cfg: ModelConfig, train_cfg: TrainingConfig, dataset: PerturbSeqDataset,
    val_dataset: Optional[PerturbSeqDataset] = None, verbose: bool = True,
) -> Tuple[VirtualCellModel, Dict[str, list]]:
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)

    device = torch.device(train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu")
    model = VirtualCellModel.from_config(model_cfg).to(device)
    # Sized from the dataset's actual cell lines, not model_cfg.num_cell_lines (which is
    # just a default/example) - see PerturbSeqDataset.cell_lines.
    context_embedding = nn.Embedding(len(dataset.cell_lines), model_cfg.d_context).to(device)

    params = list(model.parameters()) + list(context_embedding.parameters())
    optimizer = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    conditions = list(dataset.condition_keys)
    scheduler = _make_scheduler(optimizer, train_cfg, len(conditions))

    use_amp = train_cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    def autocast_ctx():
        return torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else nullcontext()

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    model.train()
    for epoch in range(train_cfg.max_epochs):
        random.shuffle(conditions)
        epoch_loss, n_batches = 0.0, 0
        for gene, cell_line in conditions:
            batch = dataset.sample_batch(gene, cell_line, batch_size=train_cfg.batch_size)
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
                losses = compute_loss(out, y_true, z_true, train_cfg)
                loss = losses["loss"]

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip_norm)
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        mean_train_loss = epoch_loss / max(1, n_batches)
        history["train_loss"].append(mean_train_loss)

        val_loss = None
        if val_dataset is not None:
            val_loss = evaluate(model, context_embedding, val_dataset, device, train_cfg.batch_size,
                                 cell_line_to_idx=dataset.cell_line_to_idx)
            history["val_loss"].append(val_loss)

        if verbose:
            msg = f"epoch {epoch + 1}/{train_cfg.max_epochs}  train_loss={mean_train_loss:.4f}"
            if val_loss is not None:
                msg += f"  val_loss={val_loss:.4f}"
            msg += f"  lr={scheduler.get_last_lr()[0]:.2e}"
            print(msg)

    model.context_embedding = context_embedding
    return model, history


@torch.no_grad()
def evaluate(
    model: VirtualCellModel, context_embedding: nn.Embedding, dataset: PerturbSeqDataset,
    device: torch.device, batch_size: int = 64,
    cell_line_to_idx: Optional[Dict[str, int]] = None,
) -> float:
    """``cell_line_to_idx``: pass the TRAINING dataset's mapping when
    evaluating on a different (e.g. held-out) dataset - see
    ``PerturbSeqDataset.sample_batch``."""
    model.eval()
    total_se, total_n = 0.0, 0
    for gene, cell_line in dataset.condition_keys:
        batch = dataset.sample_batch(gene, cell_line, batch_size=batch_size, cell_line_to_idx=cell_line_to_idx)
        c_target = context_embedding(batch["cell_line_idx"].to(device))
        out = model(
            batch["e_p_stack"].to(device), c_target, batch["delta_stat"].to(device),
            batch["z_0"].to(device), batch["x_ctrl"].to(device),
        )
        y_true = batch["y_true"].to(device)
        total_se += (out["y_pred"] - y_true).pow(2).sum().item()
        total_n += y_true.numel()
    model.train()
    return total_se / max(1, total_n)


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
    model_cfg = ModelConfig(num_genes=args.num_genes, d_latent=args.d_latent)
    train_cfg = TrainingConfig(batch_size=args.batch_size, max_epochs=args.max_epochs)
    train(model_cfg, train_cfg, dataset)
