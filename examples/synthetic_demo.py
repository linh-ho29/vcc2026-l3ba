"""Synthetic demo: train the hybrid adapter on artificial data and verify the
full cycle — baseline identity, training, checkpointing, deterministic reload,
and prediction.

*** ALL DATA HERE IS ARTIFICIAL (see hybrid_adapter.synthetic). ***
Performance on these fixtures is NOT evidence of biological predictive accuracy.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from hybrid_adapter import (
    AdapterConfig,
    HybridAdapter,
    TrainConfig,
    generate_whole_line_batch,
    make_synthetic_loaders,
    predict_corrected_effects,
    save_adapter,
    load_adapter,
    train_adapter,
)


def main() -> None:
    torch.manual_seed(0)
    config = AdapterConfig()
    model = HybridAdapter(config)

    # 1. Baseline identity before training.
    batch = generate_whole_line_batch()
    out = predict_corrected_effects(model, batch)
    identity = torch.allclose(out["corrected_effects"], out["baseline_effects"], atol=1e-6)
    print(f"[1] Baseline identity before training: {identity}")
    assert identity

    # 2. Train on the synthetic target function.
    train_loader, val_loader = make_synthetic_loaders()
    history = train_adapter(
        model, train_loader, val_loader, TrainConfig(max_epochs=20, patience=4)
    )
    print(f"[2] Training: {history.epochs_completed} epochs, "
          f"first val loss {history.val_loss[0]:.4f} -> best {history.best_val_loss:.4f}, "
          f"early-stopped={history.stopped_early}")
    assert history.best_val_loss < history.val_loss[0]

    # 3. Checkpoint round trip.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "adapter.pt")
        manifest = save_adapter(model, config, ckpt)
        reloaded, loaded_manifest = load_adapter(ckpt)
        out2 = predict_corrected_effects(model, batch)
        out3 = predict_corrected_effects(reloaded, batch)
        match = torch.allclose(out2["corrected_effects"], out3["corrected_effects"], atol=1e-6)
        print(f"[3] Checkpoint reload reproduces predictions: {match} "
              f"(model v{loaded_manifest['model_version']})")
        assert match

    # 4. Deterministic chunked inference.
    out_chunked = predict_corrected_effects(model, batch, gene_chunk_size=7)
    match = torch.allclose(out2["corrected_effects"], out_chunked["corrected_effects"], atol=1e-5)
    print(f"[4] Chunked == unchunked inference: {match}")
    assert match

    print("\nAll synthetic demo checks passed. (Artificial data only.)")


if __name__ == "__main__":
    main()
