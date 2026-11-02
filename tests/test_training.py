"""Training-loop tests on synthetic data: loss decreases, early stopping,
histories, resume, loss-mask edge cases."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter import (
    AdapterConfig,
    HybridAdapter,
    TrainConfig,
    compute_loss,
    de_weighted_reconstruction_loss,
    generate_whole_line_batch,
    make_synthetic_loaders,
    train_adapter,
)
from hybrid_adapter.synthetic import SyntheticConfig


def test_training_decreases_validation_loss():
    """A small synthetic target function can be learned."""
    torch.manual_seed(0)
    cfg = SyntheticConfig(batch_size=8, n_genes=48, seed=123)
    train_loader, val_loader = make_synthetic_loaders(cfg, n_train_batches=8, n_val_batches=2)
    model = HybridAdapter(AdapterConfig())
    history = train_adapter(
        model, train_loader, val_loader,
        TrainConfig(max_epochs=20, patience=8, lr=5e-3),
    )
    assert history.best_val_loss < history.val_loss[0]
    assert history.epochs_completed >= 1


def test_training_history_recorded():
    train_loader, val_loader = make_synthetic_loaders(
        SyntheticConfig(batch_size=4, n_genes=32), n_train_batches=2, n_val_batches=1
    )
    model = HybridAdapter(AdapterConfig())
    history = train_adapter(model, train_loader, val_loader, TrainConfig(max_epochs=3))
    assert len(history.train_loss) == 3
    assert len(history.val_loss) == 3
    assert history.epochs_completed == 3


def test_early_stopping_triggers():
    train_loader, val_loader = make_synthetic_loaders(
        SyntheticConfig(batch_size=4, n_genes=32), n_train_batches=2, n_val_batches=1
    )
    model = HybridAdapter(AdapterConfig())
    history = train_adapter(model, train_loader, val_loader, TrainConfig(max_epochs=50, patience=2))
    assert history.stopped_early
    assert history.epochs_completed < 50


def test_deterministic_seeds():
    def _run():
        torch.manual_seed(0)
        train_loader, val_loader = make_synthetic_loaders(
            SyntheticConfig(batch_size=4, n_genes=32, seed=7), n_train_batches=2, n_val_batches=1
        )
        model = HybridAdapter(AdapterConfig())
        return train_adapter(model, train_loader, val_loader, TrainConfig(max_epochs=2))

    h1, h2 = _run(), _run()
    assert h1.train_loss == h2.train_loss
    assert h1.val_loss == h2.val_loss


def test_resume_after_interruption(tmp_path):
    ckpt = str(tmp_path / "train_state.pt")
    train_loader, val_loader = make_synthetic_loaders(
        SyntheticConfig(batch_size=4, n_genes=32, seed=11), n_train_batches=2, n_val_batches=1
    )
    model = HybridAdapter(AdapterConfig())
    cfg = TrainConfig(max_epochs=6, patience=10, checkpoint_path=ckpt)
    history = train_adapter(model, train_loader, val_loader, cfg)

    # Resume for more epochs from the saved state.
    model2 = HybridAdapter(AdapterConfig())
    cfg2 = TrainConfig(max_epochs=10, patience=10, resume_path=ckpt)
    history2 = train_adapter(model2, train_loader, val_loader, cfg2)
    assert history2.epochs_completed > history.epochs_completed
    # Resumed history is a continuation, not a restart.
    assert history2.val_loss[: len(history.val_loss)] == history.val_loss


def test_de_weighted_loss_empty_mask_is_zero():
    pred = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8)
    empty = torch.zeros(4, 8)
    loss = de_weighted_reconstruction_loss(pred, target, empty)
    assert loss.item() == 0.0
    # And it stays connected to the graph.
    loss.backward()


def test_de_weighted_loss_nonempty_mask():
    pred = torch.randn(4, 8)
    target = torch.randn(4, 8)
    mask = torch.zeros(4, 8)
    mask[:, :3] = 1.0
    loss = de_weighted_reconstruction_loss(pred, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_compute_loss_default_weights():
    out = {"corrected_effects": torch.randn(4, 8)}
    target = torch.randn(4, 8)
    losses = compute_loss(out, target, config={})
    assert losses["loss_de"].item() == 0.0
    assert losses["loss_disc"].item() == 0.0
    assert torch.isfinite(losses["loss"])
