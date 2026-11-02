"""Forward-pass tests: dimensions, baseline identity, gradients, frozen inputs,
multi-context handling, NaN/Inf checks."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter import (
    AdapterConfig,
    HybridAdapter,
    BatchValidationError,
    generate_whole_line_batch,
    predict_corrected_effects,
)

B, G = 4, 32


def make_model(**overrides) -> HybridAdapter:
    return HybridAdapter(AdapterConfig(**overrides))


def test_output_dimensions():
    model = make_model()
    batch = _small_batch()
    out = model(batch)
    assert out["corrected_effects"].shape == (B, G)
    assert out["baseline_effects"].shape == (B, G)
    assert out["residual_effects"].shape == (B, G)
    assert out["gene_ids"] == batch["response_gene_ids"]
    assert out["perturbation_ids"] == batch["perturbation_ids"]
    assert out["context_ids"] == batch["context_ids"]


def _small_batch(**kwargs):
    from hybrid_adapter.synthetic import SyntheticConfig
    return generate_whole_line_batch(SyntheticConfig(batch_size=B, n_genes=G), **kwargs)


def test_forward_with_gears_and_scgpt():
    model = make_model()
    out = model(_small_batch(include_gears=True))
    assert torch.isfinite(out["corrected_effects"]).all()


def test_forward_without_gears():
    model = make_model()
    batch = _small_batch(include_gears=False)
    out = model(batch)
    assert torch.isfinite(out["corrected_effects"]).all()
    # Residual must still be exactly zero before training.
    assert torch.allclose(out["residual_effects"], torch.zeros_like(out["residual_effects"]), atol=1e-7)


def test_baseline_identity_before_training():
    model = make_model()
    batch = _small_batch(include_gears=True)
    out = model(batch)
    assert torch.allclose(out["corrected_effects"], out["baseline_effects"], atol=1e-6)
    assert model.check_baseline_identity(batch)


def test_backward_produces_valid_gradients():
    model = make_model()
    batch = _small_batch(include_gears=True)
    out = model(batch)
    loss = out["corrected_effects"].pow(2).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads if g.numel() > 0)


def test_parameters_change_after_optimizer_step():
    model = make_model()
    batch = _small_batch(include_gears=True)
    batch["target_effects"] = torch.randn(B, G)
    before = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    out = model(batch)
    loss = out["corrected_effects"].pow(2).mean()
    loss.backward()
    opt.step()
    changed = any(
        not torch.equal(before[k], v) for k, v in model.state_dict().items()
    )
    assert changed


def test_frozen_input_embeddings_not_modified():
    model = make_model()
    batch = _small_batch(include_gears=True)
    scgpt_before = batch["scgpt_response_embeddings"].clone()
    gears_before = batch["gears_embeddings"].clone()
    out = model(batch)
    out["corrected_effects"].sum().backward()
    assert torch.equal(batch["scgpt_response_embeddings"], scgpt_before)
    assert torch.equal(batch["gears_embeddings"], gears_before)


def test_multiple_contexts_and_perturbations():
    model = make_model()
    batch = _small_batch(include_gears=True)
    assert len(set(batch["context_ids"])) > 1 or len(set(batch["perturbation_ids"])) > 1
    out = model(batch)
    # Rows must be independent: permuting rows permutes outputs.
    perm = [2, 0, 3, 1]
    permuted = {k: v for k, v in batch.items()}
    for key in ("stat_effects", "scgpt_target_embeddings", "scgpt_response_embeddings",
                "control_features", "control_gene_expression", "gears_effects",
                "gears_embeddings"):
        if key in permuted and isinstance(permuted[key], torch.Tensor):
            permuted[key] = permuted[key][perm]
    permuted["context_ids"] = [batch["context_ids"][i] for i in perm]
    permuted["perturbation_ids"] = [batch["perturbation_ids"][i] for i in perm]
    out_perm = model(permuted)
    assert torch.allclose(
        out["corrected_effects"][perm], out_perm["corrected_effects"], atol=1e-6
    )


def test_nan_and_infinity_checks():
    model = make_model()
    batch = _small_batch(include_gears=True)
    bad = dict(batch)
    bad["stat_effects"] = batch["stat_effects"].clone()
    bad["stat_effects"][0, 0] = float("nan")
    with pytest.raises(BatchValidationError, match="NaN or infinity"):
        model(bad)

    bad2 = dict(_small_batch(include_gears=True))
    bad2["control_features"] = bad2["control_features"].clone()
    bad2["control_features"][1, 3] = math.inf
    with pytest.raises(BatchValidationError, match="NaN or infinity"):
        model(bad2)


def test_cpu_execution_without_accelerator():
    model = make_model()
    batch = _small_batch(include_gears=True)
    out = predict_corrected_effects(model, batch, device="cpu")
    assert out["corrected_effects"].device.type == "cpu"
    assert out["corrected_effects"].dtype == torch.float32
