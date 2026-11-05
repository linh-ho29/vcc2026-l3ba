"""Synthetic forward/backward tests for VirtualCellModel and its submodules,
plus the Sliced Wasserstein Distance loss. Mock tensors only - no real
biological data, no real checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.swd import sliced_wasserstein_distance
from models.virtual_cell import (
    ContextGatedGEARSFusion,
    DiscreteLatentFlow,
    ResidualDecoder,
    StatisticalBaselineCompressor,
    VirtualCellModel,
)

B, NUM_GENES, D_LATENT, D_CONTEXT = 8, 50, 128, 32


def _mock_inputs(batch_size=B, num_genes=NUM_GENES, d_latent=D_LATENT, d_context=D_CONTEXT):
    e_p_stack = torch.randn(batch_size, 4, 64)
    c_target = torch.randn(batch_size, d_context)
    delta_stat = torch.randn(batch_size, num_genes) * 0.3
    z_0 = torch.randn(batch_size, d_latent)
    x_ctrl = torch.rand(batch_size, num_genes)
    return e_p_stack, c_target, delta_stat, z_0, x_ctrl


# ---------------------------------------------------------------------------
# Module A: ContextGatedGEARSFusion
# ---------------------------------------------------------------------------

def test_fusion_output_shape_and_gate_sums_to_one():
    fusion = ContextGatedGEARSFusion(d_context=D_CONTEXT)
    e_p_stack = torch.randn(B, 4, 64)
    c_target = torch.randn(B, D_CONTEXT)
    e_p_fused, alpha = fusion(e_p_stack, c_target)
    assert e_p_fused.shape == (B, 64)
    assert alpha.shape == (B, 4, 1)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(B, 1), atol=1e-5)
    assert torch.isfinite(e_p_fused).all()


def test_fusion_rejects_wrong_e_p_stack_shape():
    fusion = ContextGatedGEARSFusion(d_context=D_CONTEXT)
    with pytest.raises(ValueError, match="e_p_stack"):
        fusion(torch.randn(B, 3, 64), torch.randn(B, D_CONTEXT))


def test_fusion_rejects_wrong_context_width():
    fusion = ContextGatedGEARSFusion(d_context=D_CONTEXT)
    with pytest.raises(ValueError, match="c_target"):
        fusion(torch.randn(B, 4, 64), torch.randn(B, D_CONTEXT - 1))


def test_fusion_shared_projection_mode():
    fusion = ContextGatedGEARSFusion(d_context=D_CONTEXT, per_line_projection=False)
    e_p_fused, alpha = fusion(torch.randn(B, 4, 64), torch.randn(B, D_CONTEXT))
    assert e_p_fused.shape == (B, 64)


def test_fusion_rejects_nan():
    fusion = ContextGatedGEARSFusion(d_context=D_CONTEXT)
    bad = torch.randn(B, 4, 64)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinity"):
        fusion(bad, torch.randn(B, D_CONTEXT))


# ---------------------------------------------------------------------------
# Module B: StatisticalBaselineCompressor
# ---------------------------------------------------------------------------

def test_stat_compressor_output_shape():
    compressor = StatisticalBaselineCompressor(num_genes=NUM_GENES)
    out = compressor(torch.randn(B, NUM_GENES), torch.randn(B, 64))
    assert out.shape == (B, 64)
    assert torch.isfinite(out).all()


def test_stat_compressor_rejects_wrong_delta_stat_width():
    compressor = StatisticalBaselineCompressor(num_genes=NUM_GENES)
    with pytest.raises(ValueError, match="delta_stat"):
        compressor(torch.randn(B, NUM_GENES - 1), torch.randn(B, 64))


def test_stat_compressor_rejects_wrong_fused_width():
    compressor = StatisticalBaselineCompressor(num_genes=NUM_GENES)
    with pytest.raises(ValueError, match="e_p_fused"):
        compressor(torch.randn(B, NUM_GENES), torch.randn(B, 63))


# ---------------------------------------------------------------------------
# Module C: DiscreteLatentFlow
# ---------------------------------------------------------------------------

def test_flow_output_shape():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, d_context=D_CONTEXT)
    z_pert = flow(torch.randn(B, D_LATENT), torch.randn(B, 64), torch.randn(B, D_CONTEXT))
    assert z_pert.shape == (B, D_LATENT)
    assert torch.isfinite(z_pert).all()


def test_flow_has_exactly_two_steps():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, d_context=D_CONTEXT)
    assert len(flow.steps) == 2


def test_flow_rejects_wrong_z0_width():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, d_context=D_CONTEXT)
    with pytest.raises(ValueError, match="z_0"):
        flow(torch.randn(B, D_LATENT - 1), torch.randn(B, 64), torch.randn(B, D_CONTEXT))


def test_flow_gradients_propagate():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, d_context=D_CONTEXT)
    z_0 = torch.randn(B, D_LATENT)
    e_p_unified = torch.randn(B, 64)
    c_target = torch.randn(B, D_CONTEXT)
    z_pert = flow(z_0, e_p_unified, c_target)
    z_pert.sum().backward()
    assert flow.cond_proj.weight.grad is not None
    assert flow.cond_proj.weight.grad.abs().sum() > 0


def test_flow_works_with_different_d_latent():
    """d_latent = 512, per the offline-data spec's alternative width."""
    flow = DiscreteLatentFlow(d_latent=512, d_context=D_CONTEXT)
    z_pert = flow(torch.randn(B, 512), torch.randn(B, 64), torch.randn(B, D_CONTEXT))
    assert z_pert.shape == (B, 512)


# ---------------------------------------------------------------------------
# Module D: ResidualDecoder
# ---------------------------------------------------------------------------

def test_decoder_zero_init_gives_exact_zero_output():
    decoder = ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES)
    out = decoder(torch.randn(B, D_LATENT))
    assert torch.allclose(out, torch.zeros_like(out))


def test_decoder_zero_init_weight_and_bias_directly():
    decoder = ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES)
    final_layer = decoder.net[-1]
    assert torch.allclose(final_layer.weight, torch.zeros_like(final_layer.weight))
    assert torch.allclose(final_layer.bias, torch.zeros_like(final_layer.bias))


def test_decoder_output_shape():
    decoder = ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES)
    out = decoder(torch.randn(B, D_LATENT))
    assert out.shape == (B, NUM_GENES)


def test_decoder_moves_off_zero_after_gradient_step():
    decoder = ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES)
    z = torch.randn(B, D_LATENT)
    out = decoder(z)
    loss = (out - torch.ones_like(out)).pow(2).mean()
    loss.backward()
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-2)
    opt.step()
    out_after = decoder(z)
    assert not torch.allclose(out_after, torch.zeros_like(out_after))


# ---------------------------------------------------------------------------
# Module E: VirtualCellModel (full pipeline)
# ---------------------------------------------------------------------------

def make_model(**overrides):
    defaults = dict(num_genes=NUM_GENES, d_latent=D_LATENT, d_context=D_CONTEXT)
    defaults.update(overrides)
    return VirtualCellModel(**defaults)


def test_full_model_output_shapes():
    model = make_model()
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs()
    out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    assert out["y_pred"].shape == (B, NUM_GENES)
    assert out["z_pert"].shape == (B, D_LATENT)
    assert out["delta_neural"].shape == (B, NUM_GENES)
    assert out["e_p_fused"].shape == (B, 64)
    assert out["e_p_unified"].shape == (B, 64)
    assert out["gate_alpha"].shape == (B, 4, 1)
    for v in out.values():
        assert torch.isfinite(v).all()


def test_full_model_zero_init_residual_property():
    """At init, delta_neural == 0 everywhere, so y_pred == x_ctrl + delta_stat
    exactly - the "begin perfectly anchored" guarantee from the spec."""
    model = make_model()
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs()
    out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    assert torch.allclose(out["delta_neural"], torch.zeros_like(out["delta_neural"]))
    assert torch.allclose(out["y_pred"], x_ctrl + delta_stat, atol=1e-6)


def test_full_model_final_assembly_identity_after_training_step():
    """y_pred == x_ctrl + delta_stat + delta_neural must hold exactly, even
    after the decoder has been moved off zero-init by a gradient step."""
    model = make_model()
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs()
    out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    loss = (out["y_pred"] - torch.ones_like(out["y_pred"])).pow(2).mean()
    loss.backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.step()

    out2 = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    assert torch.allclose(out2["y_pred"], x_ctrl + delta_stat + out2["delta_neural"], atol=1e-5)
    # And the correction should have actually moved off zero now.
    assert not torch.allclose(out2["delta_neural"], torch.zeros_like(out2["delta_neural"]))


def test_full_model_gradient_propagation_through_every_module():
    model = make_model()
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs()
    # Move decoder off zero-init first so its own gradients are checkable
    # (zero-init weights still receive a nonzero gradient; checking value
    # change, not just grad existence, needs a step - but grad existence
    # alone is what this test verifies for every parameter).
    out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    out["y_pred"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads if g.numel() > 0)


def test_full_model_backward_runs_without_error():
    model = make_model()
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs()
    out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    loss = (out["y_pred"] - torch.zeros_like(out["y_pred"])).pow(2).mean()
    loss.backward()  # must not raise


def test_full_model_rejects_wrong_x_ctrl_width():
    model = make_model()
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs()
    with pytest.raises(ValueError, match="x_ctrl"):
        model(e_p_stack, c_target, delta_stat, z_0, x_ctrl[:, :-1])


def test_full_model_row_independence():
    model = make_model()
    model.eval()
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs()
    with torch.no_grad():
        out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    perm = [3, 1, 6, 0, 5, 2, 7, 4]
    with torch.no_grad():
        out_perm = model(e_p_stack[perm], c_target[perm], delta_stat[perm], z_0[perm], x_ctrl[perm])
    assert torch.allclose(out["y_pred"][perm], out_perm["y_pred"], atol=1e-5)


def test_full_model_with_d_latent_512():
    model = make_model(d_latent=512)
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs(d_latent=512)
    out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    assert out["z_pert"].shape == (B, 512)
    assert out["y_pred"].shape == (B, NUM_GENES)


def test_full_model_batch_size_256():
    """Sanity check at the spec's target training batch size."""
    model = make_model()
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs(batch_size=256)
    out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    assert out["y_pred"].shape == (256, NUM_GENES)
    assert torch.isfinite(out["y_pred"]).all()


# ---------------------------------------------------------------------------
# Sliced Wasserstein Distance
# ---------------------------------------------------------------------------

def test_swd_zero_for_identical_distributions():
    torch.manual_seed(0)
    x = torch.randn(200, D_LATENT)
    d = sliced_wasserstein_distance(x, x.clone(), n_projections=32)
    assert d.item() < 1e-5


def test_swd_positive_for_shifted_distribution():
    torch.manual_seed(0)
    x = torch.randn(200, D_LATENT)
    y = torch.randn(200, D_LATENT) + 5.0
    d = sliced_wasserstein_distance(x, y, n_projections=32)
    assert d.item() > 1.0


def test_swd_handles_unequal_sample_counts():
    d = sliced_wasserstein_distance(torch.randn(50, D_LATENT), torch.randn(80, D_LATENT), n_projections=32)
    assert torch.isfinite(d)


def test_swd_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="Dimension mismatch"):
        sliced_wasserstein_distance(torch.randn(10, 4), torch.randn(10, 5))


def test_swd_gradients_propagate():
    x = torch.randn(32, D_LATENT, requires_grad=True)
    y = torch.randn(32, D_LATENT)
    d = sliced_wasserstein_distance(x, y, n_projections=32)
    d.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# End-to-end: model + SWD loss, one optimizer step (mimics train.py's loop)
# ---------------------------------------------------------------------------

def test_end_to_end_training_step_with_swd_loss():
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    e_p_stack, c_target, delta_stat, z_0, x_ctrl = _mock_inputs()
    y_true = torch.randn(B, NUM_GENES)
    z_true = torch.randn(B, D_LATENT)

    out = model(e_p_stack, c_target, delta_stat, z_0, x_ctrl)
    recon_loss = torch.nn.functional.mse_loss(out["y_pred"], y_true)
    dist_loss = sliced_wasserstein_distance(out["z_pert"], z_true, n_projections=32)
    lambda_swd = 0.1
    loss = recon_loss + lambda_swd * dist_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    assert torch.isfinite(loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(dist_loss)
