"""Tests for GatedMultiContextFusion and its integration into HybridAdapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter import AdapterConfig, HybridAdapter, BatchValidationError
from hybrid_adapter.gated_fusion_flow.gated_fusion import GatedMultiContextFusion
from hybrid_adapter.synthetic import SyntheticConfig, generate_whole_line_batch

B, CONTEXT_DIM, FUSED_DIM = 5, 16, 32
SOURCE_DIMS = (64, 32)  # mirrors the real case: k562=64-d, hepg2=32-d, confirmed mismatched
K = len(SOURCE_DIMS)


def make_fusion(source_dims=SOURCE_DIMS, context_dim=CONTEXT_DIM, fused_dim=FUSED_DIM):
    return GatedMultiContextFusion(source_dims=source_dims, context_dim=context_dim, fused_dim=fused_dim)


def make_inputs(source_dims=SOURCE_DIMS):
    sources = [torch.randn(B, d) for d in source_dims]
    context = torch.randn(B, CONTEXT_DIM)
    return sources, context


def test_requires_at_least_two_sources():
    with pytest.raises(ValueError, match=">= 2 sources"):
        GatedMultiContextFusion(source_dims=(32,), context_dim=CONTEXT_DIM, fused_dim=FUSED_DIM)


def test_output_shapes_with_mismatched_source_dims():
    fusion = make_fusion()
    sources, context = make_inputs()
    fused, alpha = fusion(sources, context)
    assert fused.shape == (B, FUSED_DIM)
    assert alpha.shape == (B, K)


def test_alpha_sums_to_one():
    fusion = make_fusion()
    sources, context = make_inputs()
    _, alpha = fusion(sources, context)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-6)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_masked_source_gets_exactly_zero_weight():
    fusion = make_fusion()
    sources, context = make_inputs()
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[:, 1] = False  # source 1 (hepg2) unavailable for every row
    _, alpha = fusion(sources, context, source_mask=mask)
    assert torch.allclose(alpha[:, 1], torch.zeros(B), atol=1e-7)
    # Remaining sources still renormalize to sum to 1, not silently diluted.
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_all_sources_missing_is_a_hard_error_not_silent_zero():
    fusion = make_fusion()
    sources, context = make_inputs()
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[2, :] = False  # every source missing for row 2
    with pytest.raises(BatchValidationError, match="no available source"):
        fusion(sources, context, source_mask=mask)


def test_row_independence_permutation():
    fusion = make_fusion()
    sources, context = make_inputs()
    fused, alpha = fusion(sources, context)
    perm = [3, 0, 4, 1, 2]
    sources_perm = [s[perm] for s in sources]
    fused_perm, alpha_perm = fusion(sources_perm, context[perm])
    assert torch.allclose(fused[perm], fused_perm, atol=1e-6)
    assert torch.allclose(alpha[perm], alpha_perm, atol=1e-6)


def test_wrong_number_of_sources_raises():
    fusion = make_fusion()
    sources, context = make_inputs()
    with pytest.raises(BatchValidationError, match="Expected 2 source"):
        fusion(sources[:1], context)


def test_wrong_per_source_dim_raises():
    fusion = make_fusion()
    sources, context = make_inputs()
    sources[0] = sources[0][:, :-1]  # wrong width for source 0
    with pytest.raises(BatchValidationError, match=r"source_embeddings\[0\]"):
        fusion(sources, context)


def test_wrong_context_dim_raises():
    fusion = make_fusion()
    sources, context = make_inputs()
    with pytest.raises(BatchValidationError, match="context_embedding"):
        fusion(sources, context[:, :-1])


def test_nan_inf_inputs_raise():
    fusion = make_fusion()
    sources, context = make_inputs()
    sources[0] = sources[0].clone()
    sources[0][0, 0] = float("nan")
    with pytest.raises(BatchValidationError, match="NaN or infinity"):
        fusion(sources, context)


def test_gradients_flow_to_all_per_source_projections_and_gate():
    fusion = make_fusion()
    sources, context = make_inputs()
    fused, alpha = fusion(sources, context)
    fused.sum().backward()
    for k in range(K):
        grad = fusion.input_proj[k].weight.grad
        assert grad is not None and grad.abs().sum() > 0
    assert fusion.gate.weight.grad is not None and fusion.gate.weight.grad.abs().sum() > 0
    assert fusion.proj.weight.grad is not None and fusion.proj.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Integration: HybridAdapter with use_gated_fusion=True
# ---------------------------------------------------------------------------

def _fusion_batch(batch_size=4, n_genes=16, source_dims=(64, 32), context_dim=8, missing_row=None):
    batch = generate_whole_line_batch(SyntheticConfig(batch_size=batch_size, n_genes=n_genes))
    batch.pop("gears_embeddings", None)
    batch["gears_embeddings_multi"] = [torch.randn(batch_size, d) for d in source_dims]
    batch["context_embedding"] = torch.randn(batch_size, context_dim)
    mask = torch.ones(batch_size, len(source_dims), dtype=torch.bool)
    if missing_row is not None:
        mask[missing_row, 0] = False
    batch["gears_source_mask"] = mask
    return batch


def test_adapter_forward_with_gated_fusion_enabled():
    config = AdapterConfig(use_gated_fusion=True, gears_source_dims=(64, 32), context_dim=8)
    model = HybridAdapter(config)
    batch = _fusion_batch(context_dim=8)
    out = model(batch)
    assert torch.isfinite(out["corrected_effects"]).all()
    assert "gated_fusion_alpha" in out
    assert out["gated_fusion_alpha"].shape == (4, 2)


def test_baseline_identity_preserved_with_gated_fusion():
    """Zero-init residual invariant must still hold when fusion is enabled -
    aggregating multiple GEARS sources must not change the untrained-model
    guarantee the rest of the package relies on."""
    config = AdapterConfig(use_gated_fusion=True, gears_source_dims=(64, 32), context_dim=8)
    model = HybridAdapter(config)
    batch = _fusion_batch(context_dim=8)
    assert model.check_baseline_identity(batch)


def test_adapter_rejects_missing_fusion_inputs_when_enabled():
    config = AdapterConfig(use_gated_fusion=True, gears_source_dims=(64, 32), context_dim=8)
    model = HybridAdapter(config)
    batch = _fusion_batch(context_dim=8)
    del batch["context_embedding"]
    with pytest.raises(BatchValidationError, match="use_gated_fusion"):
        model(batch)


def test_adapter_handles_per_row_missing_source():
    config = AdapterConfig(use_gated_fusion=True, gears_source_dims=(64, 32), context_dim=8)
    model = HybridAdapter(config)
    batch = _fusion_batch(context_dim=8, missing_row=1)
    out = model(batch)
    assert torch.isfinite(out["corrected_effects"]).all()
    assert torch.allclose(out["gated_fusion_alpha"][1, 0], torch.tensor(0.0), atol=1e-7)


def test_config_rejects_single_source_fusion():
    with pytest.raises(ValueError, match="num_gears_sources"):
        AdapterConfig(use_gated_fusion=True, gears_source_dims=(32,)).validate()


def test_num_gears_sources_derived_from_source_dims():
    config = AdapterConfig(gears_source_dims=(64, 32, 32))
    assert config.num_gears_sources == 3
