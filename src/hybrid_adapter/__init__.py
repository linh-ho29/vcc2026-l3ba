"""hybrid_adapter — Hybrid MLP Perturbation Adapter for the Virtual Cell Challenge 2026.

SYNTHETIC PHASE ONLY: this package implements and tests the adapter on
artificial data. It must not be trained on real biological data during this
phase of the project.
"""

from .schemas import (
    AdapterConfig,
    BatchValidationError,
    FeatureSpec,
    SCHEMA_VERSION,
    SUPPORTED_EFFECT_UNITS,
    SUPPORTED_EXPRESSION_UNITS,
    describe_schema,
    validate_batch,
)
from .model import MODEL_VERSION, HybridAdapter
from .gated_fusion import GatedMultiContextFusion
from .losses import (
    compute_loss,
    de_weighted_reconstruction_loss,
    global_reconstruction_loss,
    perturbation_discrimination_loss,
)
from .train import TrainConfig, TrainHistory, set_seed, train_adapter
from .inference import predict_corrected_effects
from .checkpoint import load_adapter, save_adapter
from .metrics import (
    direction_accuracy,
    effect_mae,
    effect_mse,
    pearson_correlation,
    perturbation_discrimination_diagnostic,
    spearman_correlation,
)
from .synthetic import (
    SyntheticConfig,
    generate_h7_batch,
    generate_whole_line_batch,
    make_h7_loaders,
    make_synthetic_loaders,
)

__version__ = MODEL_VERSION

__all__ = [
    "AdapterConfig",
    "BatchValidationError",
    "FeatureSpec",
    "SCHEMA_VERSION",
    "MODEL_VERSION",
    "SUPPORTED_EFFECT_UNITS",
    "SUPPORTED_EXPRESSION_UNITS",
    "HybridAdapter",
    "GatedMultiContextFusion",
    "describe_schema",
    "validate_batch",
    "compute_loss",
    "global_reconstruction_loss",
    "de_weighted_reconstruction_loss",
    "perturbation_discrimination_loss",
    "TrainConfig",
    "TrainHistory",
    "train_adapter",
    "set_seed",
    "predict_corrected_effects",
    "save_adapter",
    "load_adapter",
    "pearson_correlation",
    "spearman_correlation",
    "effect_mse",
    "effect_mae",
    "direction_accuracy",
    "perturbation_discrimination_diagnostic",
    "SyntheticConfig",
    "generate_whole_line_batch",
    "generate_h7_batch",
    "make_synthetic_loaders",
    "make_h7_loaders",
]
