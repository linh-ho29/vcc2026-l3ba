"""h7_validation — Subpopulation-Resolved, Identity-Gated Perturbation Response (H7).

Standalone analysis code for the three-phase H7 testing strategy:

* Phase 1 (claims 1, 2, 6): phase scoring, homogeneity variance ratios,
  cross-line phase-classifier transfer.
* Phase 2 (claims 3, 4, 5): per-phase deltas, two-component decomposition with
  shared vs per-line alpha, CompSim vs whole-line-similarity transfer.
* Phase 3: M0-M4 ablation harness (see :mod:`h7_validation.ablation`).

All functions accept plain numpy matrices plus phase labels; AnnData inputs
can be passed via ``adata.X`` / ``adata.obs``. No real data is downloaded or
bundled here.
"""

from .phase_scoring import (
    G2M_GENE_SET,
    PHASE_NAMES,
    S_GENE_SET,
    phase_proportions,
    phase_silhouette,
    score_cell_cycle_phases,
)
from .homogeneity import homogeneity_passes, homogeneity_variance_ratios
from .transfer_classifier import (
    cross_line_transfer_accuracy,
    train_phase_classifier,
    universality_summary,
)
from .phase_deltas import (
    between_within_phase_variance_ratio,
    masked_response_check,
    per_phase_delta,
    whole_line_delta,
)
from .alpha_decomposition import (
    compare_shared_vs_per_line_alpha,
    fit_alpha_for_phase,
    fit_alpha_per_line,
    sign_flip_counts,
)
from .compsim_transfer import (
    compsim,
    head_to_head_transfer,
    total_variation_distance,
    transfer_error,
    whole_line_similarity,
)
from .ablation import ABLATION_MODELS, config_for_ablation, interpret_ordering, run_ablation

__all__ = [
    "S_GENE_SET", "G2M_GENE_SET", "PHASE_NAMES",
    "score_cell_cycle_phases", "phase_silhouette", "phase_proportions",
    "homogeneity_variance_ratios", "homogeneity_passes",
    "train_phase_classifier", "cross_line_transfer_accuracy", "universality_summary",
    "per_phase_delta", "between_within_phase_variance_ratio", "whole_line_delta",
    "masked_response_check",
    "fit_alpha_for_phase", "fit_alpha_per_line", "compare_shared_vs_per_line_alpha",
    "sign_flip_counts",
    "total_variation_distance", "compsim", "whole_line_similarity",
    "transfer_error", "head_to_head_transfer",
    "ABLATION_MODELS", "config_for_ablation", "run_ablation", "interpret_ordering",
]
