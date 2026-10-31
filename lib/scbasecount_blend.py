"""
Shared helpers for E06: blending a context's own real control-cell stats with
the E05 scBaseCount curated reference, plus the negative-control variants
used to prove any apparent gain isn't just generic shrinkage.
"""
from __future__ import annotations

import numpy as np


def compute_group_percentile(group_mean_row: np.ndarray) -> np.ndarray:
    """Same rank-based percentile definition as expression_gating's context
    stats, applied over one scBaseCount reference group's own gene vocabulary."""
    order = np.argsort(group_mean_row)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order))
    return 100.0 * ranks / max(len(order) - 1, 1)


def best_and_worst_matching_group(context_detection_fraction, context_symbols, ref):
    """Data-driven match: correlate the context's OWN real detection-fraction
    profile (whole transcriptome, not just the evaluated genes) against each
    scBaseCount reference group's detection-fraction profile, over the shared
    gene vocabulary. No cell-type label is used or assumed - this is exactly
    the information available for an anonymized D/E/F context."""
    ref_symbols = ref["gene_symbols"].astype(str)
    ref_index = {s: i for i, s in enumerate(ref_symbols)}
    ctx_index = {s: i for i, s in enumerate(context_symbols)}
    shared = [s for s in context_symbols if s in ref_index]
    ctx_cols = np.array([ctx_index[s] for s in shared])
    ref_cols = np.array([ref_index[s] for s in shared])

    corrs = {}
    for gi, group in enumerate(ref["reference_groups"].astype(str)):
        ref_vec = ref["group_detection_fraction"][gi, ref_cols]
        ctx_vec = context_detection_fraction[ctx_cols]
        corrs[group] = float(np.corrcoef(ctx_vec, ref_vec)[0, 1])
    best = max(corrs, key=corrs.get)
    worst = min(corrs, key=corrs.get)
    return best, worst, corrs, shared


def reference_gene_stats(ref, group_name, gene_symbol, shuffle_seed=None):
    """detection_fraction and percentile for one gene, within one reference
    group. If shuffle_seed is set, permutes the group's gene axis first (fixed
    seed, applied once per call site) - destroys real gene-level
    correspondence while preserving the marginal distribution, for the
    negative control."""
    ref_symbols = ref["gene_symbols"].astype(str)
    gi = list(ref["reference_groups"].astype(str)).index(group_name)
    det_row = ref["group_detection_fraction"][gi]
    mean_row = ref["group_mean_lognorm"][gi]

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(det_row))
        det_row = det_row[perm]
        mean_row = mean_row[perm]

    pct_row = compute_group_percentile(mean_row)
    ref_index = {s: i for i, s in enumerate(ref_symbols)}
    if gene_symbol not in ref_index:
        return None
    j = ref_index[gene_symbol]
    return float(det_row[j]), float(pct_row[j])


def blend(context_det, context_pct, ref_det, ref_pct, lam):
    """Convex blend of two already-bounded, already-comparable scalars (both
    detection_fraction in [0,1] and percentile in [0,100] are defined
    per-context by construction) - the regularized adjustment. lam=0 must
    return exactly the context-only value (E03), with no approximation."""
    if lam == 0.0 or ref_det is None:
        return context_det, context_pct
    return (1 - lam) * context_det + lam * ref_det, (1 - lam) * context_pct + lam * ref_pct
