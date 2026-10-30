"""
Basal-expression-gated perturbation transfer.

Extends the existing K562 signature-transfer baseline (gwps_signature_bank.py /
generate_predictions_gwps.py) WITHOUT retraining it. The baseline's per-gene
log-fold-change effects, estimated purely from K562, get multiplied by a
context-specific gate w(detection_fraction, percentile) in [0, 1] before the
existing shrinkage/clip/exp conversion to a multiplicative rate - so a gene
that's effectively unexpressed in the target context can't be assigned a
large K562-derived swing just because K562 itself responded strongly there.

Does not touch raw counts directly, does not add negative values, does not
claim the gate is causally validated - it's a conservative confidence
adjustment on an existing untouched effect estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

GateMode = Literal["none", "binary_detection", "percentile", "combined"]


@dataclass
class GateConfig:
    mode: GateMode = "combined"
    detection_threshold: float = 0.05   # fraction of control cells expressing the gene
    detection_sharpness: float = 40.0   # higher = closer to a hard cutoff
    percentile_threshold: float = 10.0  # expression percentile within this context (0-100)
    percentile_sharpness: float = 0.3


def _sigmoid_gate(x: np.ndarray, threshold: float, sharpness: float) -> np.ndarray:
    """Smooth, monotonic gate in (0, 1), not a hard binary cutoff."""
    return 1.0 / (1.0 + np.exp(-sharpness * (x - threshold)))


def compute_context_gate_stats(control_X, block: int = 512) -> dict:
    """
    control_X: (n_control_cells, n_genes) raw counts for ONE context's real
    non-targeting control cells (sparse or dense). Returns per-gene
    detection_fraction, baseline_mean, and expression percentile within
    this context - all computed from real data, no perturbed cells involved.
    """
    n_cells, n_genes = control_X.shape
    nonzero_count = np.zeros(n_genes, dtype=np.float64)
    total = np.zeros(n_genes, dtype=np.float64)
    for i in range(0, n_cells, block):
        chunk = control_X[i:i + block]
        if sp.issparse(chunk):
            chunk = chunk.tocsr()
            nonzero_count += np.asarray((chunk > 0).sum(axis=0)).ravel()
            total += np.asarray(chunk.sum(axis=0)).ravel()
        else:
            chunk = np.asarray(chunk)
            nonzero_count += (chunk > 0).sum(axis=0)
            total += chunk.sum(axis=0)

    detection_fraction = nonzero_count / max(n_cells, 1)
    baseline_mean = total / max(n_cells, 1)

    # Percentile rank of each gene's baseline mean among all genes in THIS
    # context (0 = lowest-expressed gene here, 100 = highest-expressed).
    order = np.argsort(baseline_mean)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order))
    percentile = 100.0 * ranks / max(len(order) - 1, 1)

    return {
        "detection_fraction": detection_fraction,
        "baseline_mean": baseline_mean,
        "percentile": percentile,
    }


def compute_gate_weights(stats: dict, config: GateConfig = GateConfig()) -> np.ndarray:
    """
    Returns per-gene weight in [0, 1]. Undetected genes (detection_fraction
    exactly 0 across every control cell in this context) always gate to 0,
    regardless of mode - a gene with zero evidence of expression here cannot
    be assigned a confident perturbation effect, full stop. Everything above
    that floor uses the smooth, mode-dependent gate so dropout noise doesn't
    get mistaken for genuine absence.
    """
    d = stats["detection_fraction"]
    p = stats["percentile"]
    n_genes = len(d)

    if config.mode == "none":
        w = np.ones(n_genes, dtype=np.float32)
    elif config.mode == "binary_detection":
        w = (d >= config.detection_threshold).astype(np.float32)
    elif config.mode == "percentile":
        w = _sigmoid_gate(p, config.percentile_threshold, config.percentile_sharpness).astype(np.float32)
    elif config.mode == "combined":
        w_detect = _sigmoid_gate(d, config.detection_threshold, config.detection_sharpness)
        w_pct = _sigmoid_gate(p, config.percentile_threshold, config.percentile_sharpness)
        w = (w_detect * w_pct).astype(np.float32)
    else:
        raise ValueError(f"unknown gate mode {config.mode!r}")

    if config.mode != "none":
        hard_zero = d <= 0.0  # genuinely zero detection across every control cell
        w = np.where(hard_zero, 0.0, w).astype(np.float32)
    return w


def apply_gate_to_effect(raw_log_effect: np.ndarray, gate_weight: np.ndarray) -> np.ndarray:
    """Delta_{g,K562,j} x w_{c,j}. Pure elementwise multiply - the K562 effect
    estimate itself is never modified, only how much of it gets transferred."""
    return raw_log_effect * gate_weight


def compute_gate_weights_direction_aware(
    stats: dict, raw_log_effect: np.ndarray, config: GateConfig = GateConfig(),
    induction_discount: float = 0.5,
) -> np.ndarray:
    """
    Zero/low detection in the target context is real evidence against further
    DOWNREGULATION (there's little left to suppress, and no signal it would
    respond). It is NOT equally strong evidence against INDUCTION - a gene
    silent at baseline can still be inducible; single-cell dropout can also
    masquerade as absence. So downregulation keeps the full combined gate
    (detection-fraction floor included); induction instead uses a softer,
    percentile-only gate (ignoring the hard detection-based floor), and is
    additionally discounted by `induction_discount` to reflect that we have
    no direct evidence the gene is actually inducible here - just that
    absence of baseline detection doesn't rule it out.
    """
    downregulation_gate = compute_gate_weights(stats, config)
    induction_gate = _sigmoid_gate(
        stats["percentile"], config.percentile_threshold, config.percentile_sharpness
    ).astype(np.float32) * induction_discount

    is_induction = raw_log_effect > 0
    return np.where(is_induction, np.maximum(downregulation_gate, induction_gate),
                     downregulation_gate).astype(np.float32)
