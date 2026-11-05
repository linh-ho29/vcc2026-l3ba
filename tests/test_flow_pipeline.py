"""Tests for the control branch, latent flow/decoder, SWD loss, and the
combined GatedFusionFlowAdapter pipeline. Synthetic data only, per package
convention."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter import (
    BatchValidationError,
    ControlCellEncoder,
    DiscreteLatentFlow,
    GatedFusionFlowAdapter,
    InputConditioningLayer,
    LatentDecoder,
    NTCLatentBank,
    ScGPTGeneEmbeddingPooling,
    compute_flow_loss,
    sliced_wasserstein_distance,
)

N_GENES, EMB_DIM, B = 50, 16, 6


# ---------------------------------------------------------------------------
# ScGPTGeneEmbeddingPooling
# ---------------------------------------------------------------------------

def test_pooling_output_shape():
    gene_emb = torch.randn(N_GENES, EMB_DIM)
    pooling = ScGPTGeneEmbeddingPooling(gene_emb)
    expr = torch.rand(B, N_GENES)
    out = pooling(expr)
    assert out.shape == (B, EMB_DIM)


def test_pooling_is_frozen():
    gene_emb = torch.randn(N_GENES, EMB_DIM)
    pooling = ScGPTGeneEmbeddingPooling(gene_emb)
    assert all(not p.requires_grad for p in pooling.parameters()) or len(list(pooling.parameters())) == 0
    assert "gene_embeddings" in dict(pooling.named_buffers())


def test_pooling_rejects_negative_expression():
    gene_emb = torch.randn(N_GENES, EMB_DIM)
    pooling = ScGPTGeneEmbeddingPooling(gene_emb)
    expr = torch.rand(B, N_GENES)
    expr[0, 0] = -1.0
    with pytest.raises(BatchValidationError, match="non-negative"):
        pooling(expr)


def test_pooling_rejects_wrong_gene_count():
    gene_emb = torch.randn(N_GENES, EMB_DIM)
    pooling = ScGPTGeneEmbeddingPooling(gene_emb)
    with pytest.raises(BatchValidationError, match="must have .* genes"):
        pooling(torch.rand(B, N_GENES - 1))


def test_pooling_all_zero_row_does_not_nan():
    gene_emb = torch.randn(N_GENES, EMB_DIM)
    pooling = ScGPTGeneEmbeddingPooling(gene_emb)
    expr = torch.rand(B, N_GENES)
    expr[0] = 0.0
    out = pooling(expr)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ControlCellEncoder
# ---------------------------------------------------------------------------

def test_control_encoder_fit_and_encode():
    rng = np.random.default_rng(0)
    pooled = rng.normal(size=(500, EMB_DIM)).astype(np.float32)
    encoder = ControlCellEncoder(latent_dim=8).fit(pooled)
    out = encoder(torch.from_numpy(pooled[:10]))
    assert out.shape == (10, 8)
    assert 0.0 <= encoder.explained_variance_ratio_ <= 1.0 + 1e-6


def test_control_encoder_raises_before_fit():
    encoder = ControlCellEncoder(latent_dim=8)
    with pytest.raises(RuntimeError, match="fit"):
        encoder(torch.randn(5, EMB_DIM))


def test_control_encoder_is_frozen_after_fit():
    rng = np.random.default_rng(1)
    pooled = rng.normal(size=(200, EMB_DIM)).astype(np.float32)
    encoder = ControlCellEncoder(latent_dim=4).fit(pooled)
    assert all(not p.requires_grad for p in encoder.parameters())


def test_control_encoder_requires_more_cells_than_latent_dim():
    encoder = ControlCellEncoder(latent_dim=100)
    with pytest.raises(ValueError, match="Need more control cells"):
        encoder.fit(np.random.randn(10, EMB_DIM).astype(np.float32))


# ---------------------------------------------------------------------------
# NTCLatentBank
# ---------------------------------------------------------------------------

def test_latent_bank_sample_shape_and_realness():
    bank = NTCLatentBank()
    real = torch.arange(20 * 4).reshape(20, 4).float()  # distinguishable rows
    bank.add("hepg2", real)
    sample = bank.sample(100, context="hepg2")
    assert sample.shape == (100, 4)
    # Every sampled row must be one of the real stored rows (empirical sampling).
    for row in sample:
        assert any(torch.allclose(row, r) for r in real)


def test_latent_bank_unknown_context_raises():
    bank = NTCLatentBank()
    bank.add("hepg2", torch.randn(10, 4))
    with pytest.raises(BatchValidationError, match="No NTC latents"):
        bank.sample(5, context="k562")


def test_latent_bank_len_and_contexts():
    bank = NTCLatentBank()
    bank.add("hepg2", torch.randn(10, 4))
    bank.add("rpe1", torch.randn(7, 4))
    assert len(bank) == 17
    assert set(bank.contexts) == {"hepg2", "rpe1"}


# ---------------------------------------------------------------------------
# InputConditioningLayer / DiscreteLatentFlow / LatentDecoder
# ---------------------------------------------------------------------------

Z0_DIM, FUSED_DIM, CONTEXT_DIM, FLOW_DIM = 8, 6, 4, 16


def test_conditioning_layer_output_shape():
    layer = InputConditioningLayer(Z0_DIM, FUSED_DIM, CONTEXT_DIM, FLOW_DIM)
    out = layer(torch.randn(B, Z0_DIM), torch.randn(B, FUSED_DIM), torch.randn(B, CONTEXT_DIM))
    assert out.shape == (B, FLOW_DIM)


def test_conditioning_layer_rejects_wrong_width():
    layer = InputConditioningLayer(Z0_DIM, FUSED_DIM, CONTEXT_DIM, FLOW_DIM)
    with pytest.raises(BatchValidationError, match="z0"):
        layer(torch.randn(B, Z0_DIM - 1), torch.randn(B, FUSED_DIM), torch.randn(B, CONTEXT_DIM))


@pytest.mark.parametrize("n_steps", [2, 3])
def test_flow_output_shape(n_steps):
    flow = DiscreteLatentFlow(flow_dim=FLOW_DIM, output_dim=Z0_DIM, fused_dim=FUSED_DIM,
                               context_dim=CONTEXT_DIM, n_steps=n_steps)
    h0 = torch.randn(B, FLOW_DIM)
    z_pert = flow(h0, torch.randn(B, FUSED_DIM), torch.randn(B, CONTEXT_DIM))
    assert z_pert.shape == (B, Z0_DIM)
    assert torch.isfinite(z_pert).all()


def test_flow_rejects_invalid_n_steps():
    with pytest.raises(ValueError, match="n_steps"):
        DiscreteLatentFlow(n_steps=5)


def test_flow_gradients_flow():
    flow = DiscreteLatentFlow(flow_dim=FLOW_DIM, output_dim=Z0_DIM, fused_dim=FUSED_DIM, context_dim=CONTEXT_DIM)
    h0 = torch.randn(B, FLOW_DIM)
    z_pert = flow(h0, torch.randn(B, FUSED_DIM), torch.randn(B, CONTEXT_DIM))
    z_pert.sum().backward()
    assert flow.net[0].weight.grad is not None and flow.net[0].weight.grad.abs().sum() > 0


def test_decoder_output_shape():
    decoder = LatentDecoder(latent_dim=Z0_DIM, n_genes=N_GENES)
    out = decoder(torch.randn(B, Z0_DIM))
    assert out.shape == (B, N_GENES)


def test_decoder_with_hidden_layer():
    decoder = LatentDecoder(latent_dim=Z0_DIM, n_genes=N_GENES, hidden_dim=32)
    out = decoder(torch.randn(B, Z0_DIM))
    assert out.shape == (B, N_GENES)


# ---------------------------------------------------------------------------
# Sliced Wasserstein Distance
# ---------------------------------------------------------------------------

def test_swd_zero_for_identical_distributions():
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    d = sliced_wasserstein_distance(x, x.clone(), n_projections=100)
    assert d.item() < 1e-5


def test_swd_positive_for_different_distributions():
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    y = torch.randn(200, 8) + 5.0  # shifted distribution
    d = sliced_wasserstein_distance(x, y, n_projections=100)
    assert d.item() > 1.0


def test_swd_handles_different_sample_counts():
    d = sliced_wasserstein_distance(torch.randn(50, 4), torch.randn(80, 4), n_projections=20)
    assert torch.isfinite(d)


def test_swd_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="Dimension mismatch"):
        sliced_wasserstein_distance(torch.randn(10, 4), torch.randn(10, 5))


def test_compute_flow_loss_recon_only():
    y_hat = torch.randn(B, N_GENES)
    y_true = torch.randn(B, N_GENES)
    out = compute_flow_loss(y_hat, y_true)
    assert out["loss_swd"].item() == 0.0
    assert torch.isfinite(out["loss"])


def test_compute_flow_loss_with_swd():
    y_hat = torch.randn(B, N_GENES)
    y_true = torch.randn(B, N_GENES)
    z_pert = torch.randn(20, 8)
    z_true = torch.randn(20, 8)
    out = compute_flow_loss(y_hat, y_true, z_pert, z_true, config={"lambda_swd": 0.5})
    assert out["loss_swd"].item() > 0.0


def test_compute_flow_loss_swd_without_latents_raises():
    with pytest.raises(ValueError, match="lambda_swd"):
        compute_flow_loss(torch.randn(B, N_GENES), torch.randn(B, N_GENES), config={"lambda_swd": 1.0})


# ---------------------------------------------------------------------------
# GatedFusionFlowAdapter (full pipeline)
# ---------------------------------------------------------------------------

SOURCE_DIMS = (12, 10)


def make_adapter(**overrides):
    defaults = dict(gears_source_dims=SOURCE_DIMS, context_dim=CONTEXT_DIM,
                     fused_dim=FUSED_DIM, z0_dim=Z0_DIM, flow_dim=FLOW_DIM, n_genes=N_GENES)
    defaults.update(overrides)
    return GatedFusionFlowAdapter(**defaults)


def _pipeline_inputs(batch_size=B):
    z0 = torch.randn(batch_size, Z0_DIM)
    sources = [torch.randn(batch_size, d) for d in SOURCE_DIMS]
    context = torch.randn(batch_size, CONTEXT_DIM)
    return z0, sources, context


def test_full_pipeline_output_shapes():
    model = make_adapter()
    z0, sources, context = _pipeline_inputs()
    out = model(z0, sources, context)
    assert out["predicted_expression"].shape == (B, N_GENES)
    assert out["z_pert"].shape == (B, Z0_DIM)
    assert out["e_p_fused"].shape == (B, FUSED_DIM)
    assert out["gate_alpha"].shape == (B, len(SOURCE_DIMS))
    for v in out.values():
        assert torch.isfinite(v).all()


def test_full_pipeline_rejects_bad_z0_width():
    model = make_adapter()
    _, sources, context = _pipeline_inputs()
    with pytest.raises(BatchValidationError, match="z0"):
        model(torch.randn(B, Z0_DIM - 1), sources, context)


def test_full_pipeline_gradients_flow_through_everything():
    model = make_adapter()
    z0, sources, context = _pipeline_inputs()
    out = model(z0, sources, context)
    out["predicted_expression"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads if g.numel() > 0)


def test_full_pipeline_row_independence():
    model = make_adapter()
    model.eval()
    z0, sources, context = _pipeline_inputs()
    with torch.no_grad():
        out = model(z0, sources, context)
    perm = [2, 0, 4, 1, 5, 3]
    sources_perm = [s[perm] for s in sources]
    with torch.no_grad():
        out_perm = model(z0[perm], sources_perm, context[perm])
    assert torch.allclose(out["predicted_expression"][perm], out_perm["predicted_expression"], atol=1e-5)


def test_full_pipeline_with_missing_source():
    model = make_adapter()
    z0, sources, context = _pipeline_inputs()
    mask = torch.ones(B, len(SOURCE_DIMS), dtype=torch.bool)
    mask[0, 0] = False
    out = model(z0, sources, context, source_mask=mask)
    assert torch.isfinite(out["predicted_expression"]).all()
    assert torch.allclose(out["gate_alpha"][0, 0], torch.tensor(0.0), atol=1e-7)
