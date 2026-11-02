"""Phase 2c — CompSim-based vs whole-line-similarity transfer (head-to-head).

Claim 3 prediction: predicting line c'’s whole-line response magnitude from
line c via whole-line transcriptional similarity is WORSE than predicting it
via composition-similarity-weighted per-phase transfer::

    CompSim(c, c') = 1 - TV(p_c(phase), p_c'(phase))

This is the direct head-to-head test against the original H2 (whole-line
similarity) — the core comparison of the H7 strategy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TV distance between two probability vectors: 0.5 * L1."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if abs(p.sum() - 1.0) > 1e-4 or abs(q.sum() - 1.0) > 1e-4:
        raise ValueError("Phase-proportion vectors must sum to 1.")
    return float(0.5 * np.abs(p - q).sum())


def compsim(p_c: np.ndarray, p_c2: np.ndarray) -> float:
    """Composition similarity in [0, 1]."""
    return 1.0 - total_variation_distance(p_c, p_c2)


def whole_line_similarity(
    expr_c: np.ndarray, expr_c2: np.ndarray
) -> float:
    """Whole-line transcriptional similarity (Pearson of mean profiles)."""
    mu_c = expr_c.mean(axis=0)
    mu_c2 = expr_c2.mean(axis=0)
    sd_c, sd_c2 = mu_c.std(), mu_c2.std()
    if sd_c == 0 or sd_c2 == 0:
        return float("nan")
    return float(np.corrcoef(mu_c, mu_c2)[0, 1])


def transfer_error(
    source_delta: np.ndarray,
    target_true_delta: np.ndarray,
) -> float:
    """Relative L2 error of transferring a whole-line delta between lines."""
    denom = max(float(np.linalg.norm(target_true_delta)), 1e-12)
    return float(np.linalg.norm(source_delta - target_true_delta) / denom)


def head_to_head_transfer(
    line_deltas: Dict[str, np.ndarray],
    phase_deltas: Dict[str, Dict[str, np.ndarray]],
    phase_proportions: Dict[str, np.ndarray],
    held_out_perturbations: List[str],
) -> Dict[str, object]:
    """Compare whole-line-similarity transfer vs CompSim-weighted per-phase transfer.

    Parameters
    ----------
    line_deltas:
        ``{perturbation: {line: whole-line delta [G]}}`` for evaluation
        perturbations.
    phase_deltas:
        ``{perturbation: {line: {phase: delta [G]}}}`` per-phase deltas.
    phase_proportions:
        ``{line: p_c(phase)}``.
    held_out_perturbations:
        Perturbations evaluated (transfer source and target lines are swapped
        within each pair; errors are averaged).

    For each pair of lines (c, c') and each held-out perturbation:

    * H2 baseline: predict c'’s whole-line delta as c's whole-line delta,
      weighted by whole-line similarity (rank-normalised across sources).
    * H7: predict c'’s whole-line delta as the composition-weighted sum of
      c's per-phase deltas, weighted by CompSim.

    Returns mean relative errors for both methods and the per-pair table.
    """
    lines = sorted(phase_proportions.keys())
    pairs: List[Dict[str, object]] = []
    h2_errors, h7_errors = [], []

    for ci in range(len(lines)):
        for cj in range(len(lines)):
            if ci == cj:
                continue
            src, tgt = lines[ci], lines[cj]
            src_sim = whole_line_similarity  # placeholder to keep names local
            for pert in held_out_perturbations:
                if pert not in line_deltas:
                    continue
                true_tgt = line_deltas[pert][tgt]
                src_line_delta = line_deltas[pert][src]

                # --- H2: whole-line similarity weighting across all sources ---
                others = [o for o in lines if o != tgt]
                preds = np.vstack([line_deltas[pert][o] for o in others])
                sims = np.asarray([
                    max(whole_line_similarity(line_deltas[pert][o], line_deltas[pert][tgt]), 0.0)
                    for o in others
                ])
                if sims.sum() > 0:
                    h2_pred = (sims[:, None] * preds).sum(axis=0) / sims.sum()
                else:
                    h2_pred = preds.mean(axis=0)
                h2_err = transfer_error(h2_pred, true_tgt)

                # --- H7: CompSim-weighted per-phase transfer ---
                phases = sorted(phase_deltas[pert][src].keys())
                stack = np.vstack([phase_deltas[pert][src][p] for p in phases])
                w = compsim(phase_proportions[tgt], phase_proportions[src])
                h7_pred = stack.mean(axis=0) if w <= 0 else (
                    w * stack.mean(axis=0) + (1 - w) * src_line_delta
                )
                h7_err = transfer_error(h7_pred, true_tgt)

                h2_errors.append(h2_err)
                h7_errors.append(h7_err)
                pairs.append({
                    "source": src,
                    "target": tgt,
                    "perturbation": pert,
                    "h2_error": h2_err,
                    "h7_error": h7_err,
                    "compsim": w,
                })

    if not h2_errors:
        return {"h2_mean_error": float("nan"), "h7_mean_error": float("nan"),
                "h7_wins": False, "pairs": []}

    h2_mean = float(np.mean(h2_errors))
    h7_mean = float(np.mean(h7_errors))
    return {
        "h2_mean_error": h2_mean,
        "h7_mean_error": h7_mean,
        "h7_wins": bool(h7_mean < h2_mean),
        "claim3_supported": bool(h7_mean < h2_mean),
        "pairs": pairs,
    }
