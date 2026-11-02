"""Phase 1 — within-phase homogeneity and variance ratios.

Claim 2 test: within-phase variance should be substantially lower than
whole-line variance, for every line. Reports a variance ratio per line per
phase.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _mean_within_phase_variance(expression: np.ndarray, phases: np.ndarray, phase: str) -> float:
    """Mean per-gene variance of cells assigned to ``phase``."""
    block = expression[phases == phase]
    if block.shape[0] < 2:
        return float("nan")
    return float(block.var(axis=0, ddof=1).mean())


def homogeneity_variance_ratios(
    expression: np.ndarray,
    phases: np.ndarray,
    phase_order: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Variance ratio per phase: var_within_phase / var_whole_line.

    A ratio well below 1.0 supports claim 2 (within-phase homogeneity): cells
    of the same phase in the same line are much more similar to each other
    than the line as a whole is internally.

    Returns ``{phase: {"var_within": ..., "var_line": ..., "ratio": ...,
    "n_cells": ...}}``.
    """
    from .phase_scoring import PHASE_NAMES

    order = list(phase_order) if phase_order is not None else list(PHASE_NAMES)
    var_line = float(expression.var(axis=0, ddof=1).mean())
    results: Dict[str, Dict[str, float]] = {}
    for phase in order:
        var_within = _mean_within_phase_variance(expression, np.asarray(phases), phase)
        n_cells = int((np.asarray(phases) == phase).sum())
        results[phase] = {
            "var_within": var_within,
            "var_line": var_line,
            "ratio": (var_within / var_line) if var_line > 0 and not np.isnan(var_within) else float("nan"),
            "n_cells": n_cells,
        }
    return results


def homogeneity_passes(
    ratios: Dict[str, Dict[str, float]],
    max_ratio: float = 0.8,
    min_cells: int = 10,
) -> bool:
    """Phase 1 pass criterion helper.

    Homogeneity holds when every phase with at least ``min_cells`` cells has a
    variance ratio below ``max_ratio``. Phases with too few cells are skipped
    (reported as NaN) rather than failing the criterion.
    """
    checked = 0
    for stats in ratios.values():
        if stats["n_cells"] < min_cells:
            continue
        checked += 1
        if not np.isfinite(stats["ratio"]) or stats["ratio"] >= max_ratio:
            return False
    return checked > 0
