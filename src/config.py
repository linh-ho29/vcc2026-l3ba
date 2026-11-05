"""Structured configuration for VirtualCellModel: all architectural and
training knobs exposed through dataclasses, for rapid hyperparameter
tuning, ablations, and budget-conscious sweeps (<$100 GPU budget).

ModelConfig's fields map directly onto models/virtual_cell.py's modules -
every field here is actually wired into a constructor argument or a
forward-pass behavior, not aspirational. See VirtualCellModel.from_config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    # Feature Dimensions
    d_latent: int = 128              # Dimensionality of frozen scGPT latents (z_0)
    d_gears: int = 64                # Dimensionality of GEARS perturbation vectors
    d_context: int = 32              # Target cell line context embedding dimension
    d_stat: int = 64                 # Projected dimension of empirical statistical delta
    num_genes: int = 2000            # Output gene expression space (G)
    num_cell_lines: int = 4          # [K562, HepG2, RPE, Jurkat]

    # Gated Fusion Layer
    gating_hidden_dim: int = 64      # Hidden dim in gating MLP
    gating_temperature: float = 1.0  # Temperature in softmax gating (tau)
    gating_dropout: float = 0.0      # Dropout on gating attention weights
    invariant_skip_weight: float = 0.2  # Scale on the invariant (mean-pooled) skip connection
    per_line_projection: bool = True

    # Discrete Latent Flow Block
    flow_steps: int = 2              # Number of discrete Euler velocity steps (K in {2, 3})
    flow_hidden_dim: int = 256       # Hidden dimension of residual velocity MLP
    flow_dropout: float = 0.1        # Dropout inside velocity blocks
    step_size: float = 1.0           # Integration scaling step (Delta t)
    weight_sharing: bool = False     # Whether steps share weights or use separate MLPs
    activation: str = "GELU"         # "GELU" or "SiLU"

    # Decoder & Output Head
    decoder_depth: int = 2           # 1 for linear, 2 for MLP
    decoder_hidden_dim: int = 256
    zero_init_residual: bool = True  # Zero-init last linear layer to anchor training at baseline
    residual_clamping: Optional[float] = None  # Optional clamp on max abs value of delta_neural


@dataclass
class TrainingConfig:
    # Optimization
    batch_size: int = 256            # Critical for Sliced Wasserstein distribution matching
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    grad_clip_norm: float = 1.0
    lr_scheduler: str = "cosine"     # "cosine", "onecycle", or "constant"

    # Loss Weights
    lambda_recon: float = 1.0        # Weight for MSE on expression space
    lambda_swd: float = 0.1          # Weight for Sliced Wasserstein Distance in latent space
    swd_num_projections: int = 32    # Number of random 1D projections (L)

    # Hardware & Precision
    use_amp: bool = True             # Automatic Mixed Precision (torch.cuda.amp)
    device: str = "cuda"
    seed: int = 0


@dataclass
class SweepConfig:
    """A budget-conscious hyperparameter sweep: a short list of named
    (ModelConfig, TrainingConfig) override dicts, each run for few epochs
    and compared by held-out loss, not a full combinatorial grid."""
    max_total_minutes: float = 90.0     # hard cap across the whole sweep, budget guard
    epochs_per_trial: int = 8           # short trials - enough to rank configs, not to converge
    trials: List[dict] = field(default_factory=lambda: [
        {"name": "baseline"},
        {"name": "wider_flow", "model": {"flow_hidden_dim": 512}},
        {"name": "3_step_flow", "model": {"flow_steps": 3}},
        {"name": "low_swd", "train": {"lambda_swd": 0.0}},
        {"name": "high_swd", "train": {"lambda_swd": 0.3}},
        {"name": "shared_flow_weights", "model": {"weight_sharing": True}},
        {"name": "lower_lr", "train": {"lr": 3e-4}},
        {"name": "silu", "model": {"activation": "SiLU"}},
        {"name": "clamped_residual", "model": {"residual_clamping": 2.0}},
        {"name": "cold_gate", "model": {"gating_temperature": 0.5}},
    ])
