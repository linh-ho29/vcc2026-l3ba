"""Tests that src/config.py's knobs are genuinely wired into the model,
not just accepted and ignored. Synthetic data only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.virtual_cell import ContextGatedGEARSFusion, DiscreteLatentFlow, ResidualDecoder, VirtualCellModel
from src.config import ModelConfig, SweepConfig, TrainingConfig

B, NUM_GENES, D_LATENT, D_CONTEXT, D_GEARS = 8, 40, 128, 32, 64


def _inputs(num_genes=NUM_GENES, d_latent=D_LATENT, d_context=D_CONTEXT, d_gears=D_GEARS, batch=B):
    return (
        torch.randn(batch, 4, d_gears),
        torch.randn(batch, d_context),
        torch.randn(batch, num_genes) * 0.3,
        torch.randn(batch, d_latent),
        torch.rand(batch, num_genes),
    )


def test_model_config_defaults_match_spec():
    cfg = ModelConfig()
    assert cfg.d_latent == 128 and cfg.d_gears == 64 and cfg.d_context == 32
    assert cfg.flow_steps == 2 and cfg.decoder_depth == 2
    assert cfg.zero_init_residual is True


def test_from_config_builds_working_model():
    cfg = ModelConfig(num_genes=NUM_GENES, d_latent=D_LATENT)
    model = VirtualCellModel.from_config(cfg)
    out = model(*_inputs())
    assert out["y_pred"].shape == (B, NUM_GENES)
    assert torch.isfinite(out["y_pred"]).all()


def test_gating_temperature_changes_alpha_sharpness():
    torch.manual_seed(0)
    e_p_stack, c_target, _, _, _ = _inputs()
    sharp = ContextGatedGEARSFusion(d_gears=D_GEARS, d_context=D_CONTEXT, gating_temperature=0.1)
    flat = ContextGatedGEARSFusion(d_gears=D_GEARS, d_context=D_CONTEXT, gating_temperature=10.0)
    sharp.load_state_dict(flat.state_dict())  # same weights, only temperature differs
    _, alpha_sharp = sharp(e_p_stack, c_target)
    _, alpha_flat = flat(e_p_stack, c_target)
    # Lower temperature -> more peaked softmax -> higher max-alpha on average.
    assert alpha_sharp.max(dim=1).values.mean() > alpha_flat.max(dim=1).values.mean()


def test_gating_dropout_zero_is_deterministic_and_normalized():
    fusion = ContextGatedGEARSFusion(d_gears=D_GEARS, d_context=D_CONTEXT, gating_dropout=0.0)
    fusion.eval()
    e_p_stack, c_target, _, _, _ = _inputs()
    _, alpha = fusion(e_p_stack, c_target)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(B, 1), atol=1e-5)


def test_invariant_skip_weight_zero_removes_skip_contribution():
    torch.manual_seed(0)
    e_p_stack, c_target, _, _, _ = _inputs()
    with_skip = ContextGatedGEARSFusion(d_gears=D_GEARS, d_context=D_CONTEXT, invariant_skip_weight=1.0)
    without_skip = ContextGatedGEARSFusion(d_gears=D_GEARS, d_context=D_CONTEXT, invariant_skip_weight=0.0)
    without_skip.load_state_dict(with_skip.state_dict())
    out_with, _ = with_skip(e_p_stack, c_target)
    out_without, _ = without_skip(e_p_stack, c_target)
    assert not torch.allclose(out_with, out_without)


def test_flow_steps_3_runs_three_distinct_steps():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, unified_dim=D_GEARS, d_context=D_CONTEXT, n_steps=3)
    assert len(flow.steps) == 3
    z_pert = flow(torch.randn(B, D_LATENT), torch.randn(B, D_GEARS), torch.randn(B, D_CONTEXT))
    assert z_pert.shape == (B, D_LATENT)


def test_flow_rejects_invalid_n_steps():
    with pytest.raises(ValueError, match="n_steps"):
        DiscreteLatentFlow(d_latent=D_LATENT, n_steps=5)


def test_step_size_zero_means_z_pert_equals_z0():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, unified_dim=D_GEARS, d_context=D_CONTEXT, step_size=0.0)
    z_0 = torch.randn(B, D_LATENT)
    z_pert = flow(z_0, torch.randn(B, D_GEARS), torch.randn(B, D_CONTEXT))
    assert torch.allclose(z_pert, z_0)


def test_weight_sharing_uses_same_module_for_every_step():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, unified_dim=D_GEARS, d_context=D_CONTEXT,
                               n_steps=3, weight_sharing=True)
    assert flow.steps[0] is flow.steps[1] is flow.steps[2]


def test_weight_sharing_false_uses_distinct_modules():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, unified_dim=D_GEARS, d_context=D_CONTEXT,
                               n_steps=3, weight_sharing=False)
    assert flow.steps[0] is not flow.steps[1]


def test_silu_activation_selectable():
    flow = DiscreteLatentFlow(d_latent=D_LATENT, unified_dim=D_GEARS, d_context=D_CONTEXT, activation="SiLU")
    assert isinstance(flow.steps[0].act, torch.nn.SiLU)


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="activation"):
        DiscreteLatentFlow(d_latent=D_LATENT, activation="relu")


def test_decoder_depth_1_is_single_linear():
    decoder = ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES, depth=1)
    assert isinstance(decoder.net, torch.nn.Linear)


def test_decoder_depth_2_is_mlp():
    decoder = ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES, depth=2, hidden_dim=32)
    assert isinstance(decoder.net, torch.nn.Sequential)


def test_decoder_rejects_invalid_depth():
    with pytest.raises(ValueError, match="depth"):
        ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES, depth=3)


def test_zero_init_residual_false_gives_nonzero_output():
    torch.manual_seed(0)
    decoder = ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES, zero_init_residual=False)
    out = decoder(torch.randn(B, D_LATENT))
    assert not torch.allclose(out, torch.zeros_like(out))


def test_residual_clamping_bounds_output():
    torch.manual_seed(0)
    decoder = ResidualDecoder(d_latent=D_LATENT, num_genes=NUM_GENES, zero_init_residual=False, clamp=0.5)
    out = decoder(torch.randn(B, D_LATENT) * 100)  # large input, would overflow without clamping
    assert out.abs().max() <= 0.5 + 1e-6


def test_full_model_with_nondefault_config_runs():
    cfg = ModelConfig(
        num_genes=NUM_GENES, d_latent=D_LATENT, flow_steps=3, weight_sharing=True,
        activation="SiLU", decoder_depth=1, residual_clamping=1.0, gating_temperature=0.5,
    )
    model = VirtualCellModel.from_config(cfg)
    out = model(*_inputs())
    assert torch.isfinite(out["y_pred"]).all()
    assert out["delta_neural"].abs().max() <= 1.0 + 1e-6


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.batch_size == 256 and cfg.lr == 1e-3 and cfg.lambda_swd == 0.1


def test_sweep_config_has_budget_guard_and_trials():
    cfg = SweepConfig()
    assert cfg.max_total_minutes > 0
    assert len(cfg.trials) >= 1
    assert all("name" in t for t in cfg.trials)
