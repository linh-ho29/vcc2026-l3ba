"""Training pipeline for the hybrid adapter.

Implements ``train_adapter(model, train_loader, validation_loader, config)`` per
the development assignment: AdamW, configurable lr/weight-decay/epochs, early
stopping (patience 4), gradient clipping, histories, checkpointing, deterministic
seeds, and interruption/resume support.

IMPORTANT — data hygiene
------------------------
* Measured held-out perturbation responses must NEVER be used during fitting or
  hyperparameter selection. The validation loader is used for early stopping and
  reporting only.
* When the residual correction is eventually trained on real data, the
  statistical baseline (``stat_effects``) must be produced out-of-fold: the
  baseline for a training example must not have been fit on that example's own
  measured response, otherwise the residual has nothing to learn and leakage
  inflates apparent performance.
* No biological data is used in this phase; all training here is synthetic.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn

from .checkpoint import save_training_state, load_training_state
from .losses import compute_loss
from .model import HybridAdapter


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    # Loss weights (DE weighting and discrimination disabled initially).
    lambda_all: float = 1.0
    lambda_de: float = 0.0
    lambda_disc: float = 0.0
    disc_temperature: float = 0.1
    # Checkpointing
    checkpoint_path: Optional[str] = None
    resume_path: Optional[str] = None
    # Optional per-batch fields consumed by the loss
    use_eligibility_mask_from_batch: bool = True
    eligibility_mask_key: str = "eligibility_mask"
    de_mask_key: str = "de_mask"
    target_key: str = "target_effects"


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    epochs_completed: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    stopped_early: bool = False


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _extract_target(batch: Mapping[str, Any], cfg: TrainConfig):
    target = batch.get(cfg.target_key)
    if target is None:
        raise ValueError(
            f"Training batches must contain {cfg.target_key!r} (measured effects "
            "for TRAINING perturbations only — never held-out perturbations)."
        )
    return target


def _extract_masks(batch: Mapping[str, Any], cfg: TrainConfig):
    eligibility = batch.get(cfg.eligibility_mask_key) if cfg.use_eligibility_mask_from_batch else None
    de_mask = batch.get(cfg.de_mask_key)
    return eligibility, de_mask


def train_adapter(
    model: HybridAdapter,
    train_loader,
    validation_loader,
    config: Optional[TrainConfig] = None,
    device: str = "cpu",
) -> TrainHistory:
    """Train the adapter on synthetic (or, later, properly out-of-fold real) data.

    Returns a :class:`TrainHistory` with per-epoch training and validation
    losses. Saves the best (lowest validation loss) model state to
    ``config.checkpoint_path`` when provided, and supports resume from
    ``config.resume_path`` after an interruption.
    """
    cfg = config or TrainConfig()
    set_seed(cfg.seed)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = TrainHistory()
    start_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_opt_state = copy.deepcopy(optimizer.state_dict())

    if cfg.resume_path:
        state = load_training_state(cfg.resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history.train_loss = state["history"]["train_loss"]
        history.val_loss = state["history"]["val_loss"]
        history.epochs_completed = state["history"]["epochs_completed"]
        history.best_val_loss = state["history"]["best_val_loss"]
        history.best_epoch = state["history"]["best_epoch"]
        start_epoch = state["epoch"]
        best_state = state["best_model_state"]
        best_opt_state = state["best_optimizer_state"]

    loss_cfg = {
        "lambda_all": cfg.lambda_all,
        "lambda_de": cfg.lambda_de,
        "lambda_disc": cfg.lambda_disc,
        "disc_temperature": cfg.disc_temperature,
    }

    epochs_since_improvement = 0
    for epoch in range(start_epoch, cfg.max_epochs):
        # ---------------- train ----------------
        model.train()
        epoch_losses = []
        for batch in train_loader:
            target = _extract_target(batch, cfg)
            eligibility, de_mask = _extract_masks(batch, cfg)
            optimizer.zero_grad()
            output = model(batch)
            losses = compute_loss(
                output, target, loss_cfg,
                eligibility_mask=eligibility, de_mask=de_mask,
            )
            losses["loss"].backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_losses.append(losses["loss"].item())
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        # ---------------- validate ----------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                target = _extract_target(batch, cfg)
                eligibility, de_mask = _extract_masks(batch, cfg)
                output = model(batch)
                losses = compute_loss(
                    output, target, loss_cfg,
                    eligibility_mask=eligibility, de_mask=de_mask,
                )
                val_losses.append(losses["loss"].item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.val_loss.append(val_loss)

        history.epochs_completed = epoch + 1
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_opt_state = copy.deepcopy(optimizer.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if cfg.checkpoint_path:
            save_training_state(
                cfg.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                best_model_state=best_state,
                best_optimizer_state=best_opt_state,
            )

        if epochs_since_improvement >= cfg.patience:
            history.stopped_early = True
            break

    # Restore the best state.
    model.load_state_dict(best_state)
    return history
