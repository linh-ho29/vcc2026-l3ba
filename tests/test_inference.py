"""Inference tests: chunked vs unchunked equivalence, determinism, H7
subpopulation modes, metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter import (
    AdapterConfig,
    HybridAdapter,
    generate_h7_batch,
    generate_whole_line_batch,
    predict_corrected_effects,
)
from hybrid_adapter.metrics import (
    direction_accuracy,
    effect_mae,
    effect_mse,
    pearson_correlation,
    perturbation_discrimination_diagnostic,
    spearman_correlation,
)
from hybrid_adapter.synthetic import SyntheticConfig

B, G = 3, 40


def _batch():
    return generate_whole_line_batch(SyntheticConfig(batch_size=B, n_genes=G, seed=42))


def _trained_model():
    torch.manual_seed(1)
    model = HybridAdapter(AdapterConfig())
    batch = _batch()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model, batch


def test_chunked_and_unchunked_inference_match():
    model, batch = _trained_model()
    out_full = predict_corrected_effects(model, batch)
    out_chunked = predict_corrected_effects(model, batch, gene_chunk_size=9)
    assert torch.allclose(
        out_full["corrected_effects"], out_chunked["corrected_effects"], atol=1e-5
    )


def test_inference_is_deterministic():
    model, batch = _trained_model()
    a = predict_corrected_effects(model, batch)["corrected_effects"]
    b = predict_corrected_effects(model, batch)["corrected_effects"]
    assert torch.equal(a, b)


def test_inference_eval_mode_no_dropout_stochasticity():
    model, batch = _trained_model()
    model.train()  # simulate a model left in train mode
    a = predict_corrected_effects(model, batch)["corrected_effects"]
    b = predict_corrected_effects(model, batch)["corrected_effects"]
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# H7 subpopulation modes
# ---------------------------------------------------------------------------
def _h7_batch():
    return generate_h7_batch(
        SyntheticConfig(batch_size=B, n_genes=G, n_phases=3,
                        planted_alpha=(0.8, 0.5, 0.2), seed=17)
    )


@pytest.mark.parametrize("variant,alpha_mode,comp", [
    ("M1", "global", False),
    ("M2", "phase_shared", False),
    ("M3", "phase_per_line", False),
    ("M4", "phase_shared", True),
])
def test_h7_modes_forward(variant, alpha_mode, comp):
    model = HybridAdapter(AdapterConfig(
        response_mode="subpopulation", n_phases=3,
        alpha_mode=alpha_mode, composition_weighting=comp,
    ))
    batch = _h7_batch()
    out = model(batch)
    assert out["corrected_effects"].shape == (B, G)
    assert out["phase_corrected_effects"].shape == (B, 3, G)
    assert torch.isfinite(out["corrected_effects"]).all()

    # Composition-weighted sum over phases must reproduce the line prediction.
    recombined = (batch["phase_proportions"].unsqueeze(-1)
                  * out["phase_corrected_effects"]).sum(dim=1)
    assert torch.allclose(out["corrected_effects"], recombined, atol=1e-5)


def test_h7_alpha_gating_math():
    """With zero-init heads the residual is 0 regardless of alpha; after
    perturbing only the invariant head, alpha_eff scales the residual."""
    model = HybridAdapter(AdapterConfig(
        response_mode="subpopulation", n_phases=3,
        alpha_mode="phase_shared", composition_weighting=False,
    ))
    batch = _h7_batch()
    out0 = model(batch)
    assert torch.allclose(out0["residual_effects"],
                          torch.zeros_like(out0["residual_effects"]), atol=1e-7)

    with torch.no_grad():
        final = model.invariant_mlp.net[-1]
        final.weight.add_(0.1 * torch.randn_like(final.weight))
        final.bias.add_(0.1)
    out1 = model(batch)
    assert not torch.allclose(out1["residual_effects"],
                              torch.zeros_like(out1["residual_effects"]), atol=1e-6)
    # Effective alphas are in (0, 1) and shared across batch rows (M2).
    alpha_eff = out1["alpha_effective"]
    assert (alpha_eff > 0).all() and (alpha_eff < 1).all()
    assert torch.allclose(alpha_eff, alpha_eff[0:1].expand_as(alpha_eff))


def test_h7_composition_weighting_changes_prediction():
    torch.manual_seed(2)
    model = HybridAdapter(AdapterConfig(
        response_mode="subpopulation", n_phases=3,
        alpha_mode="phase_shared", composition_weighting=True,
    ))
    with torch.no_grad():
        for head in (model.invariant_mlp, model.line_dependent_mlp):
            final = head.net[-1]
            final.weight.add_(0.1 * torch.randn_like(final.weight))
            final.bias.add_(0.1)
    batch = _h7_batch()
    out_with = model(batch)

    # Zeroing CompSim must push alpha_eff to 0 and change the prediction.
    batch_zero = dict(batch)
    batch_zero["compsim_weights"] = torch.zeros(B, 3)
    out_zero = model(batch_zero)
    assert (out_zero["alpha_effective"] == 0).all()
    assert not torch.allclose(out_with["corrected_effects"],
                              out_zero["corrected_effects"], atol=1e-5)


def test_h7_requires_subpopulation_fields():
    model = HybridAdapter(AdapterConfig(response_mode="subpopulation", n_phases=3))
    whole_line_batch = _batch()  # lacks phase fields
    with pytest.raises(Exception, match="subpopulation"):
        model(whole_line_batch)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_metrics_basic():
    pred = torch.randn(B, G)
    target = torch.randn(B, G)
    assert -1.0 <= pearson_correlation(pred, target) <= 1.0
    assert -1.0 <= spearman_correlation(pred, target) <= 1.0
    assert effect_mse(pred, target) >= 0
    assert effect_mae(pred, target) >= 0

    mask = (target != 0).float()
    acc = direction_accuracy(pred, target, mask)
    assert 0.0 <= acc <= 1.0

    diag = perturbation_discrimination_diagnostic(pred, target)
    assert "matched_cosine" in diag and "margin" in diag


def test_metrics_mask_support():
    pred = torch.randn(B, G)
    target = torch.randn(B, G)
    mask = torch.zeros(B, G)
    mask[:, :10] = 1.0
    assert pearson_correlation(pred, target, mask=mask) == \
        pearson_correlation(pred[:, :10], target[:, :10])
