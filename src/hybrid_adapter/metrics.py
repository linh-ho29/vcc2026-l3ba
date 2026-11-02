"""Evaluation metrics for real-data testing of the adapter.

These utilities are for later real-data evaluation. They must NOT be used to
derive official VCC scores from synthetic data or pseudobulk deltas alone; the
official evaluator (cell-eval2, metrics: PDS, MSE, NMAE, FID, Reach, Jaccard)
will be integrated separately once compatible raw-count predictions and
reference single-cell observations are available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "pearson_correlation",
    "spearman_correlation",
    "effect_mse",
    "effect_mae",
    "direction_accuracy",
    "perturbation_discrimination_diagnostic",
    "cell_eval2_interface",
]


def _to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def pearson_correlation(
    predicted,
    target,
    mask=None,
) -> float:
    """Pearson correlation of predicted vs true perturbation deltas (flattened)."""
    p = _to_numpy(predicted).ravel()
    t = _to_numpy(target).ravel()
    if mask is not None:
        m = _to_numpy(mask).ravel().astype(bool)
        p, t = p[m], t[m]
    if p.size < 2:
        return float("nan")
    ps = p.std()
    ts = t.std()
    if ps == 0.0 or ts == 0.0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    """Average-rank assignment for tie-corrected Spearman correlation."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_correlation(predicted, target, mask=None) -> float:
    """Spearman rank correlation (pure numpy, average ranks for ties)."""
    p = _to_numpy(predicted).ravel()
    t = _to_numpy(target).ravel()
    if mask is not None:
        m = _to_numpy(mask).ravel().astype(bool)
        p, t = p[m], t[m]
    if p.size < 2:
        return float("nan")
    return pearson_correlation(_rank_with_ties(p), _rank_with_ties(t))


def effect_mse(predicted, target, mask=None) -> float:
    p = _to_numpy(predicted)
    t = _to_numpy(target)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        p, t = p[m], t[m]
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - t) ** 2))


def effect_mae(predicted, target, mask=None) -> float:
    p = _to_numpy(predicted)
    t = _to_numpy(target)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        p, t = p[m], t[m]
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - t)))


def direction_accuracy(predicted, target, eligibility_mask) -> float:
    """Fraction of eligible genes where sign(predicted) == sign(target).

    Requires an externally provided gene-eligibility mask (genes where
    direction is meaningful, e.g. expressed/DE genes). Zero-true-effect genes
    are excluded.
    """
    p = _to_numpy(predicted)
    t = _to_numpy(target)
    m = _to_numpy(eligibility_mask).astype(bool)
    p, t = p[m], t[m]
    keep = t != 0
    p, t = p[keep], t[keep]
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.sign(p) == np.sign(t)))


def perturbation_discrimination_diagnostic(
    predicted, target
) -> Dict[str, float]:
    """Discrimination diagnostic: matched vs unmatched cosine similarity.

    Rows are context-perturbation pairs; a well-calibrated adapter should show
    mean matched cosine similarity clearly above mean unmatched similarity.
    """
    p = _to_numpy(predicted)
    t = _to_numpy(target)
    n = p.shape[0]
    if n < 2:
        return {"matched_cosine": float("nan"), "unmatched_cosine": float("nan"),
                "margin": float("nan")}

    def _row_norm(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, 1e-12, None)

    sim = _row_norm(p) @ _row_norm(t).T
    diag = np.diag(sim).copy()
    off = sim[~np.eye(n, dtype=bool)]
    matched = float(diag.mean())
    unmatched = float(off.mean()) if off.size else float("nan")
    return {
        "matched_cosine": matched,
        "unmatched_cosine": unmatched,
        "margin": matched - unmatched,
    }


def cell_eval2_interface() -> Any:
    """Optional interface to Arc's official cell-eval2 package.

    Returns the imported module if available; otherwise raises an informative
    ImportError. Never fabricates official VCC metrics (PDS, MSE, NMAE, FID,
    Reach, Jaccard) from synthetic data.
    """
    try:
        import cell_eval2  # noqa: F401
        return cell_eval2
    except ImportError as exc:
        raise ImportError(
            "The official 'cell-eval2' package is not installed. Official VCC "
            "metrics (PDS, MSE, NMAE, FID, Reach, Jaccard) require raw-count "
            "predictions and reference single-cell observations and will be "
            "integrated separately; do not derive them from synthetic data."
        ) from exc
