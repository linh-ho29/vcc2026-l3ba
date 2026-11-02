"""Checkpoint tests: save/load round trip, manifest contents, checksum
verification, training-state resume file."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter import (
    AdapterConfig,
    HybridAdapter,
    generate_whole_line_batch,
    load_adapter,
    predict_corrected_effects,
    save_adapter,
)
from hybrid_adapter.checkpoint import load_training_state, save_training_state
from hybrid_adapter.synthetic import SyntheticConfig


def _model_and_batch():
    torch.manual_seed(3)
    model = HybridAdapter(AdapterConfig())
    batch = generate_whole_line_batch(SyntheticConfig(batch_size=3, n_genes=24, seed=5))
    # Nudge weights so the checkpoint is non-trivial.
    batch["target_effects"] = torch.randn_like(batch["stat_effects"])
    out = model(batch)
    loss = out["corrected_effects"].pow(2).mean()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1e-3 * torch.randn_like(p))
    return model, batch


def test_checkpoint_round_trip_reproduces_predictions(tmp_path):
    model, batch = _model_and_batch()
    path = str(tmp_path / "adapter.pt")
    manifest = save_adapter(model, model.config, path)
    reloaded, _ = load_adapter(path, device="cpu")

    out_a = predict_corrected_effects(model, batch)
    out_b = predict_corrected_effects(reloaded, batch)
    assert torch.allclose(out_a["corrected_effects"], out_b["corrected_effects"], atol=1e-6)
    assert torch.allclose(out_a["residual_effects"], out_b["residual_effects"], atol=1e-6)
    assert manifest["state_dict_checksum"]


def test_manifest_records_required_fields(tmp_path):
    model, _ = _model_and_batch()
    path = str(tmp_path / "adapter.pt")
    manifest = save_adapter(model, model.config, path)
    for key in (
        "architecture", "input_feature_dimensions", "required_modalities",
        "optional_modalities", "gene_axis_requirements", "effect_units",
        "expression_units", "model_version", "state_dict_checksum",
        "dependency_versions", "schema_version",
    ):
        assert key in manifest, f"manifest missing {key}"
    assert manifest["architecture"]["residual_input_dim"] == 104


def test_checkpoint_checksum_mismatch_detected(tmp_path):
    model, _ = _model_and_batch()
    path = str(tmp_path / "adapter.pt")
    save_adapter(model, model.config, path)
    payload = torch.load(path, weights_only=False)
    payload["state_dict"]["fusion.mlp.0.weight"] = torch.randn_like(
        payload["state_dict"]["fusion.mlp.0.weight"]
    )
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_adapter(path, device="cpu")


def test_load_without_gears_or_scgpt_installed(tmp_path, monkeypatch):
    """Loading must not require GEARS or scGPT importable."""
    model, _ = _model_and_batch()
    path = str(tmp_path / "adapter.pt")
    save_adapter(model, model.config, path)

    import builtins
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("gears", "scgpt"):
            raise ImportError(f"{name} blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    reloaded, _ = load_adapter(path, device="cpu")
    assert isinstance(reloaded, HybridAdapter)


def test_training_state_round_trip(tmp_path):
    from hybrid_adapter.train import TrainHistory

    model, _ = _model_and_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = str(tmp_path / "state.pt")
    save_training_state(path, model=model, optimizer=opt, epoch=3,
                        history=TrainHistory(), best_model_state=model.state_dict(),
                        best_optimizer_state=opt.state_dict())
    state = load_training_state(path)
    assert state["epoch"] == 3
    assert "model" in state and "optimizer" in state
