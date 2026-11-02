"""Phase 2a — per-subpopulation response magnitudes.

For each (line, perturbation, phase), compute the mean expression delta
between perturbed and control cells restricted to cells in that phase::

    Delta(line, pert, phase, g) = mean_{perturbed, phase}(x_g)
                                - mean_{control, phase}(x_g)

Claim 3 predicts per-phase deltas differ substantially across phases and that
the whole-line delta (a proportion-weighted average) can mask a large response
in a minority subpopulation. Reported statistic: the ratio of between-phase
variance to within-phase variance of the delta.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def per_phase_delta(
    perturbed: np.ndarray,
    perturbed_phases: np.ndarray,
    control: np.ndarray,
    control_phases: np.ndarray,
    phase: str,
    min_cells: int = 3,
) -> Tuple[np.ndarray, int]:
    """Gene-wise mean delta restricted to one phase.

    Returns ``(delta [n_genes], n_perturbed_cells_in_phase)``. Raises if fewer
    than ``min_cells`` perturbed or control cells exist in the phase (the
    delta is undefined there).
    """
    p_mask = np.asarray(perturbed_phases) == phase
    c_mask = np.asarray(control_phases) == phase
    n_p, n_c = int(p_mask.sum()), int(c_mask.sum())
    if n_p < min_cells or n_c < min_cells:
        raise ValueError(
            f"Phase {phase!r} has too few cells (perturbed={n_p}, control={n_c}; "
            f"min_cells={min_cells}); per-phase delta is undefined."
        )
    return perturbed[p_mask].mean(axis=0) - control[c_mask].mean(axis=0), n_p


def between_within_phase_variance_ratio(
    phase_deltas: Dict[str, np.ndarray],
) -> float:
    """Ratio of between-phase to within-phase variance of the gene-wise delta.

    ``phase_deltas`` maps phase name -> delta vector [n_genes]. A large ratio
    supports claim 3: the response magnitude is not uniform across phases.
    """
    phases = [p for p, d in phase_deltas.items() if d.size]
    if len(phases) < 2:
        return float("nan")
    stack = np.vstack([phase_deltas[p] for p in phases])  # [K, G]
    gene_mean = stack.mean(axis=0, keepdims=True)
    between = float(((stack - gene_mean) ** 2).mean())
    within = float(stack.var(axis=0, ddof=1).mean()) if stack.shape[0] > 1 else float("nan")
    if not np.isfinite(within) or within == 0.0:
        return float("nan")
    return between / within


def whole_line_delta(
    perturbed: np.ndarray,
    control: np.ndarray,
) -> np.ndarray:
    """Whole-line mean delta (the quantity H7 says can mask minority responses)."""
    return perturbed.mean(axis=0) - control.mean(axis=0)


def masked_response_check(
    phase_deltas: Dict[str, np.ndarray],
    proportions: Dict[str, float],
    line_delta: np.ndarray,
    top_k: int = 1,
) -> Dict[str, float]:
    """Quantify how much the whole-line delta masks minority-subpopulation responses.

    Compares the largest-magnitude per-phase delta with the whole-line delta
    (L2 norms), and reports the composition-weighted reconstruction error of
    the whole-line delta from the per-phase deltas.
    """
    phases = list(phase_deltas.keys())
    if not phases:
        return {"masking_ratio": float("nan"), "reconstruction_error": float("nan")}
    stack = np.vstack([phase_deltas[p] for p in phases])
    props = np.array([proportions.get(p, 0.0) for p in phases])
    reconstructed = (props[:, None] * stack).sum(axis=0)
    recon_err = float(np.linalg.norm(reconstructed - line_delta) / max(np.linalg.norm(line_delta), 1e-12))
    top_phase_norms = np.sort(np.linalg.norm(stack, axis=1))[::-1][:top_k]
    masking = float(top_phase_norms.mean() / max(np.linalg.norm(line_delta), 1e-12))
    return {"masking_ratio": masking, "reconstruction_error": recon_err}
