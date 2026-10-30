"""
E08/E09: multi-source (K562 + RPE1) perturbation-signature combination and a
simple, regularized context-aware weighting model.

M1: K562-only (= E03's existing backbone).
M2: RPE1-only.
M3: fixed-weight combination, alpha * K562 + (1-alpha) * RPE1.
M4: M3 followed by the existing E03 target-control gate.

Missing-source fallback (documented, not a zero-fill): if a perturbation has a
trained signature in only one bank, that bank's value is used directly,
regardless of alpha - never averaged against an assumed zero effect.
"""
from __future__ import annotations

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from expression_gating import GateConfig, compute_gate_weights, apply_gate_to_effect


def combine_effect_row(k562_row: np.ndarray, rpe1_row: np.ndarray, alpha: float) -> np.ndarray:
    """Per-gene combination over one perturbation's full effect vector. NaN in
    either input means that source has no information for that gene - use
    the other source's value, not zero. NaN in both stays NaN (no source
    provides any information at all for that gene)."""
    k_missing = np.isnan(k562_row)
    r_missing = np.isnan(rpe1_row)
    combined = alpha * np.nan_to_num(k562_row, nan=0.0) + (1 - alpha) * np.nan_to_num(rpe1_row, nan=0.0)
    out = np.where(k_missing & r_missing, np.nan,
          np.where(k_missing, rpe1_row,
          np.where(r_missing, k562_row, combined)))
    return out.astype(np.float32)


class SourceBank:
    """Thin wrapper over one signature_bank_*.npz for name-based lookup."""
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.genes = d["genes"].astype(str)
        self.perts = d["perts"].astype(str)
        self.effects = d["effects"]
        self.control_mean = d["control_mean"]
        self.gene_index = {g: i for i, g in enumerate(self.genes)}
        self.pert_index = {p: i for i, p in enumerate(self.perts)}

    def effect_row(self, pert, gene_order):
        """Full effect vector for `pert`, reindexed onto `gene_order` (a list
        of gene symbols); NaN for genes outside this bank's own vocabulary or
        if `pert` has no trained signature at all."""
        if pert not in self.pert_index:
            return np.full(len(gene_order), np.nan, dtype=np.float32)
        row = self.effects[self.pert_index[pert]]
        out = np.full(len(gene_order), np.nan, dtype=np.float32)
        for j, g in enumerate(gene_order):
            gi = self.gene_index.get(g)
            if gi is not None:
                out[j] = row[gi]
        return out


def m1_k562_only(k562: SourceBank, pert, gene_order):
    return k562.effect_row(pert, gene_order)


def m2_rpe1_only(rpe1: SourceBank, pert, gene_order):
    return rpe1.effect_row(pert, gene_order)


def m3_fixed_weight(k562: SourceBank, rpe1: SourceBank, pert, gene_order, alpha):
    return combine_effect_row(k562.effect_row(pert, gene_order), rpe1.effect_row(pert, gene_order), alpha)


def m4_fixed_weight_gated(k562: SourceBank, rpe1: SourceBank, pert, gene_order, alpha, gate_weights):
    combined = m3_fixed_weight(k562, rpe1, pert, gene_order, alpha)
    return apply_gate_to_effect(combined, gate_weights)


# ---------------------------------------------------------------------------
# E09: simple, regularized context-weighting model.
#
# Learns alpha as a function of per-gene TARGET-CONTEXT features only (never
# the target's own measured perturbation outcomes) from the K562<->RPE1
# development pair: for each gene in the K562∩RPE1 overlap, the "right"
# amount of trust in K562 (vs. a generic transfer-confidence baseline) is
# approximated by how well K562's OWN effect for that gene, across many
# perturbations, correlates with RPE1's real measured effect for that same
# gene. This produces a per-gene reliability score purely from development
# data, then a tiny ridge-regularized logistic-ish mapping from basic
# baseline-expression features (which ARE available in any new context from
# control cells alone) to that reliability score. At inference, only the new
# context's own control-cell features are used - no new perturbation data.
# ---------------------------------------------------------------------------
class ContextWeightModel:
    def __init__(self, l2=1.0):
        self.l2 = l2
        self.coef_ = None
        self.feature_names_ = None

    def fit(self, X, y):
        """X: (n_genes, n_features) target-context features (detection
        fraction, percentile, etc., all computable from control cells alone).
        y: (n_genes,) per-gene reliability target in [0, 1], derived ONLY
        from the K562<->RPE1 development pair (not from any evaluation
        context). Closed-form ridge regression - deliberately not a neural
        network, per the task's explicit guidance to start simple."""
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        n_features = Xb.shape[1]
        A = Xb.T @ Xb + self.l2 * np.eye(n_features)
        b = Xb.T @ y
        self.coef_ = np.linalg.solve(A, b)
        return self

    def predict_weight(self, X):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        raw = Xb @ self.coef_
        return np.clip(raw, 0.0, 1.0)
