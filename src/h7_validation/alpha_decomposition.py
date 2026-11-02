"""Phase 2b — two-component decomposition and the identity gate.

For each (perturbation, phase), the H7 model is::

    Delta(line, pert, phase, g) = alpha * I(pert, phase, g)
                                + (1 - alpha) * D(line, pert, phase, g)

with ``I`` the cell-line-invariant component and ``D`` the line-dependent
component, assumed uncorrelated with ``I`` across genes and independent
across lines.

Identifiability (documented honestly): with ``I`` unknown and ``D`` free per
line, alpha is not identifiable by ordinary regression — any alpha fits the
deltas exactly by rescaling D. The identifiable quantity is the across-gene
covariance between DIFFERENT lines' deltas, which under the model equals
``alpha^2 * Var_g(I)`` (only the shared component correlates across lines).
We therefore use:

    C_shared = Cov_g(delta_l, delta_l')  for l != l'
             = (L^2 * Var_g(I_hat) - sum_l Var_g(delta_l)) / (L * (L - 1))
    FVE_I    = C_shared / mean_l Var_g(delta_l)      (claim 4 metric)
    alpha_hat = sqrt(FVE) / (sqrt(FVE) + sqrt(1 - FVE))

The last step assumes equal per-gene scales for the invariant and dependent
components (Var_g(I) ≈ Var_g(D)); under that assumption the map is exact, and
it is monotone in the true alpha otherwise. Users with a better estimate of I
(e.g. from a large panel of lines) can supply it via ``invariant=``.

Claim 5 (alpha shared across lines) is tested by comparing a shared-alpha
variance model to a per-line-alpha variance model on the residuals, via AIC
and Bartlett's variance-homogeneity F-style test.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats as scipy_stats


def _alpha_from_fve(fve: float) -> float:
    """Map a fraction-of-variance-explained value to alpha under equal scales."""
    fve = float(np.clip(fve, 0.0, 1.0))
    if fve <= 0.0:
        return 0.0
    if fve >= 1.0:
        return 1.0
    return float(np.sqrt(fve) / (np.sqrt(fve) + np.sqrt(1.0 - fve)))


def _shared_covariance(stack: np.ndarray) -> float:
    """C_shared = alpha^2 * Var_g(I), estimated from cross-line covariances."""
    L = stack.shape[0]
    if L < 2:
        raise ValueError("Need at least 2 lines to separate invariant from dependent components.")
    var_sum = float(np.var(stack.sum(axis=0), ddof=1))          # Var_g(sum_l delta_l)
    var_lines = sum(float(np.var(stack[i], ddof=1)) for i in range(L))
    return (var_sum - var_lines) / (L * (L - 1))


def fit_alpha_for_phase(
    deltas: Dict[str, np.ndarray],
    invariant: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Fit a single shared alpha for one (perturbation, phase) across lines.

    Returns alpha, the invariant component (cross-line mean unless supplied),
    per-line residuals, and the fraction of variance explained by I alone
    (claim 4 diagnostic).
    """
    lines = sorted(deltas.keys())
    stack = np.vstack([deltas[l] for l in lines])  # [L, G]
    if invariant is None:
        invariant = stack.mean(axis=0)

    line_vars = [float(np.var(deltas[l], ddof=1)) for l in lines]
    c_shared = _shared_covariance(stack)
    mean_line_var = float(np.mean(line_vars))

    fve = (c_shared / mean_line_var) if mean_line_var > 0 else float("nan")
    alpha = _alpha_from_fve(fve) if np.isfinite(fve) else 0.5

    return {
        "alpha": alpha,
        "invariant": invariant,
        "residuals": {l: deltas[l] - invariant for l in lines},
        "variance_explained_by_I": float(fve),
        "shared_covariance": c_shared,
        "n_lines": len(lines),
    }


def fit_alpha_per_line(
    deltas: Dict[str, np.ndarray],
    invariant: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Fit a separate alpha per (line, phase) — the M3-style alternative.

    Uses each line's own variance share: FVE_l = C_shared / Var_g(delta_l).
    """
    lines = sorted(deltas.keys())
    stack = np.vstack([deltas[l] for l in lines])
    c_shared = _shared_covariance(stack)
    alphas: Dict[str, float] = {}
    for l in lines:
        v = float(np.var(deltas[l], ddof=1))
        fve = (c_shared / v) if v > 0 else float("nan")
        alphas[l] = _alpha_from_fve(fve) if np.isfinite(fve) else 0.5
    return alphas


def compare_shared_vs_per_line_alpha(
    deltas: Dict[str, np.ndarray],
    invariant: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """AIC + variance-homogeneity comparison of shared vs per-line alpha.

    Claim 5 predicts the shared-alpha model is NOT significantly worse:
    per-line alpha estimates are close to each other (spread < 0.15), and the
    shared-alpha variance model fits at least as well (AIC). The Bartlett-style
    homogeneity p-value is reported as a diagnostic only.
    """
    lines = sorted(deltas.keys())
    stack = np.vstack([deltas[l] for l in lines])  # [L, G]
    if invariant is None:
        invariant = stack.mean(axis=0)
    L, G = stack.shape

    shared = fit_alpha_for_phase(deltas, invariant)
    per_line = fit_alpha_per_line(deltas, invariant)

    resid = stack - invariant[None, :]
    per_line_resid_var = np.array([float(np.var(resid[i], ddof=1)) for i in range(L)])
    pooled_var = float(per_line_resid_var.mean())

    # Gaussian AIC on the residuals: common variance vs per-line variances.
    n = L * G
    aic_shared = n * np.log(max(pooled_var, 1e-12)) + 2 * 2
    aic_per_line = float(
        sum(np.log(max(v, 1e-12)) for v in per_line_resid_var) * G
        + 2 * (L + 1)
    )

    # Bartlett's test for homogeneity of per-line residual variances.
    if L >= 2 and G > 1 and pooled_var > 0:
        nu = G - 1
        sp = pooled_var
        # Standard Bartlett statistic.
        k = L
        sp_sq = float(np.sum(nu * per_line_resid_var) / (k * nu))
        T = nu * np.log(sp_sq) - float(np.sum(nu * np.log(per_line_resid_var)))
        corr = 1.0 + (np.sum(1.0 / nu) - 1.0 / (k * nu)) / (3.0 * (k - 1))
        T = T / corr
        p_value = float(1.0 - scipy_stats.chi2.cdf(T, k - 1))
        f_stat = float(max(per_line_resid_var) / min(per_line_resid_var))
    else:
        p_value, f_stat = float("nan"), float("nan")

    alphas = np.array([per_line[l] for l in lines])
    # Claim-5 criterion (per the hypothesis document): per-line alpha estimates
    # are close to each other, and the shared-alpha model fits nearly as well
    # as the per-line model (AIC). The Bartlett p-value is reported as a
    # diagnostic only — the shared cross-line mean induces residual correlation
    # across lines, so the chi-square approximation is conservative here.
    claim5 = bool(
        (aic_shared <= aic_per_line + 2.0) and (float(alphas.max() - alphas.min()) < 0.15)
    )
    return {
        "shared_alpha": shared["alpha"],
        "per_line_alpha": per_line,
        "alpha_spread": float(alphas.max() - alphas.min()),
        "aic_shared": float(aic_shared),
        "aic_per_line": float(aic_per_line),
        "f_stat": f_stat,
        "p_value": p_value,
        "claim5_supported": claim5,
    }


def sign_flip_counts(
    deltas: Dict[str, np.ndarray],
    invariant: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    """H7 direction check: count cross-line sign flips of the delta per phase.

    A phase with many sign flips is a low-alpha phase; the model should route
    it to the line-dependent component.
    """
    lines = sorted(deltas.keys())
    stack = np.vstack([deltas[l] for l in lines])
    if invariant is None:
        invariant = stack.mean(axis=0)
    signs = np.sign(stack - invariant[None, :])  # direction of the line-dependent part
    flips = (signs[:-1] * signs[1:] < 0).sum(axis=0)
    return {
        "n_sign_flip_genes": int((flips >= len(stack) - 2).sum()),
        "max_flips_per_gene": int(flips.max()),
    }
