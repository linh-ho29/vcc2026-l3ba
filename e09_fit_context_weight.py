"""
E09: fit a simple, regularized context-weighting model using ONLY the K562<->
RPE1 development pair (RPE1 is explicitly marked a development dataset - it
was already used to select E03). The model predicts, per downstream gene, how
much to trust K562's signal when transferring to a NEW context, as a function
of features computable from that new context's own CONTROL cells alone - never
from measured perturbation outcomes in the new context.

Training target: per gene g, how well K562's predicted effect for (pert, g)
actually tracked RPE1's REAL measured effect for (pert, g), across the 2308
perturbations both banks have trained signatures for. This uses RPE1's real
perturbation outcomes - legitimate here because RPE1 is the development
dataset (same role it played selecting E03), and is NEVER repeated against
HepG2's or any other evaluation context's outcomes.

Training features (per gene g, all computable from a NEW context's own
control cells + two static, source-side quantities fixed at K562<->RPE1
fit time): RPE1's own baseline expression and detection fraction for g, 4
standard gene-family pathway flags, a K562-vs-RPE1 baseline-expression
similarity score for g, and whether g is itself one of the perturbed genes
anywhere in the development set (a simple "perturbation target expression"
proxy, per E09's feature list).

Output: context_weight_model.npz (coefficients only - this IS the "frozen
before touching the held-out context" artifact E10 must use unmodified).
"""
import json
from pathlib import Path

import numpy as np
import anndata as ad
import scipy.sparse as sp

import sys
sys.path.insert(0, str(Path(__file__).parent))
from expression_gating import compute_context_gate_stats
from multi_source_transfer import SourceBank, ContextWeightModel

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "predictions"
RPE1_RAW_BULK = ROOT / "data" / "rpe1_raw_bulk_01.h5ad"
OUT_MODEL = PRED_DIR / "context_weight_model.npz"
OUT_REPORT = ROOT / "vcc_agent_harness" / "experiments" / "E09_fit_report.json"

k562 = SourceBank(PRED_DIR / "signature_bank_k562gwps.npz")
rpe1 = SourceBank(PRED_DIR / "signature_bank_rpe1.npz")

shared_genes = sorted(set(k562.genes) & set(rpe1.genes))
print(f"K562 ∩ RPE1 shared genes (feature/target universe): {len(shared_genes)}", flush=True)
overlap_perts = sorted(set(k562.perts) & set(rpe1.perts))
print(f"K562 ∩ RPE1 overlap perturbations (development pairs): {len(overlap_perts)}", flush=True)

# --- Training target: per-gene transfer reliability, K562 -> RPE1 ---------
gene_pos = {g: i for i, g in enumerate(shared_genes)}
k562_mat = np.full((len(overlap_perts), len(shared_genes)), np.nan, dtype=np.float32)
rpe1_mat = np.full((len(overlap_perts), len(shared_genes)), np.nan, dtype=np.float32)
for i, p in enumerate(overlap_perts):
    k_row = k562.effect_row(p, shared_genes)
    r_row = rpe1.effect_row(p, shared_genes)
    k562_mat[i] = k_row
    rpe1_mat[i] = r_row

print("Computing per-gene transfer reliability (correlation across perturbations)...", flush=True)
y = np.full(len(shared_genes), np.nan, dtype=np.float64)
MIN_PAIRS = 30
for j in range(len(shared_genes)):
    k_col, r_col = k562_mat[:, j], rpe1_mat[:, j]
    valid = ~np.isnan(k_col) & ~np.isnan(r_col)
    if valid.sum() < MIN_PAIRS or np.std(k_col[valid]) == 0 or np.std(r_col[valid]) == 0:
        continue
    corr = np.corrcoef(k_col[valid], r_col[valid])[0, 1]
    y[j] = (corr + 1.0) / 2.0  # rescale [-1,1] -> [0,1] weight

trainable = ~np.isnan(y)
print(f"Genes with a defined reliability target (>= {MIN_PAIRS} valid pairs): {trainable.sum()} / {len(shared_genes)}", flush=True)

# --- Features, computed from RPE1's own control cells + static source info ---
print("Computing RPE1's own real control-cell gate stats as training features...", flush=True)
a = ad.read_h5ad(RPE1_RAW_BULK, backed="r")
names = a.obs_names.tolist()
targets = np.array([n.split("_")[1] for n in names])
ctrl_mask = targets == "non-targeting"
X_ctrl = a.X[np.flatnonzero(ctrl_mask), :]
X_ctrl = sp.csr_matrix(X_ctrl) if not sp.issparse(X_ctrl) else X_ctrl.tocsr()
rpe1_bulk_genes = a.var["gene_name"].astype(str).to_numpy()
a.file.close()
rpe1_stats_full = compute_context_gate_stats(X_ctrl, block=64)
rpe1_bulk_gene_idx = {g: i for i, g in enumerate(rpe1_bulk_genes)}

n_genes = len(shared_genes)
baseline_expr = np.full(n_genes, np.nan)
detection_frac = np.full(n_genes, np.nan)
for j, g in enumerate(shared_genes):
    bi = rpe1_bulk_gene_idx.get(g)
    if bi is not None:
        baseline_expr[j] = np.log1p(rpe1_stats_full["baseline_mean"][bi])
        detection_frac[j] = rpe1_stats_full["detection_fraction"][bi]

pathway_ribo = np.array([g.startswith(("RPL", "RPS")) for g in shared_genes], dtype=np.float64)
pathway_mito = np.array([g.startswith("MT-") for g in shared_genes], dtype=np.float64)
pathway_histone = np.array([g.startswith(("HIST", "H1-", "H2A", "H2B", "H3-", "H4-")) for g in shared_genes], dtype=np.float64)
pathway_hemo = np.array([g.startswith(("HBA", "HBB", "HBZ", "HBG", "HBD", "HBE")) for g in shared_genes], dtype=np.float64)

k562_ctrl_idx = k562.gene_index
rpe1_ctrl_idx = rpe1.gene_index
src_tgt_similarity = np.full(n_genes, np.nan)
for j, g in enumerate(shared_genes):
    ki, ri = k562_ctrl_idx.get(g), rpe1_ctrl_idx.get(g)
    if ki is not None and ri is not None:
        k_val, r_val = np.log1p(k562.control_mean[ki]), np.log1p(rpe1.control_mean[ri])
        src_tgt_similarity[j] = -abs(k_val - r_val)  # higher = more similar baseline level

is_perturbation_target = np.array([g in set(overlap_perts) for g in shared_genes], dtype=np.float64)

feature_names = ["baseline_expression", "detection_fraction", "pathway_ribosomal", "pathway_mitochondrial",
                  "pathway_histone", "pathway_hemoglobin", "source_target_similarity", "is_perturbation_target"]
X = np.column_stack([baseline_expr, detection_frac, pathway_ribo, pathway_mito,
                      pathway_histone, pathway_hemo, src_tgt_similarity, is_perturbation_target])

fit_mask = trainable & ~np.isnan(X).any(axis=1)
print(f"Final training set: {fit_mask.sum()} genes with complete features + target", flush=True)
X_fit, y_fit = X[fit_mask], y[fit_mask]

# Standardize features (fit stats saved, applied identically at inference).
feat_mean, feat_std = X_fit.mean(axis=0), X_fit.std(axis=0)
feat_std[feat_std == 0] = 1.0
X_fit_std = (X_fit - feat_mean) / feat_std

model = ContextWeightModel(l2=10.0).fit(X_fit_std, y_fit)
pred = model.predict_weight(X_fit_std)
train_r2 = 1 - np.sum((y_fit - pred) ** 2) / np.sum((y_fit - y_fit.mean()) ** 2)
print(f"Train R^2 (in-sample, on development data only): {train_r2:.4f}", flush=True)
print(f"Coefficients: {dict(zip(feature_names + ['intercept'], model.coef_.round(4)))}", flush=True)

np.savez(OUT_MODEL, coef=model.coef_, feat_mean=feat_mean, feat_std=feat_std,
         feature_names=np.array(feature_names), l2=np.array([10.0]))
print(f"Saved frozen model to {OUT_MODEL} - E10 must load this unmodified.", flush=True)

report = {
    "n_development_genes_total": int(len(shared_genes)),
    "n_genes_with_reliability_target": int(trainable.sum()),
    "n_genes_used_for_fit": int(fit_mask.sum()),
    "train_r2": float(train_r2),
    "coefficients": dict(zip(feature_names + ["intercept"], [float(c) for c in model.coef_])),
    "l2_regularization": 10.0,
    "note": "Fit using ONLY K562<->RPE1 (the development pair). Target = per-gene "
            "correlation between K562's predicted effect and RPE1's REAL measured "
            "effect across perturbations, rescaled to [0,1]. Features use RPE1's own "
            "control-cell stats + static source-side quantities - this is the exact "
            "same category of information (control cells only) that will be available "
            "for HepG2 at E10 inference time. Model coefficients are frozen here, "
            "before E10 ever loads HepG2 data.",
}
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
OUT_REPORT.write_text(json.dumps(report, indent=2))
print(f"Saved fit report to {OUT_REPORT}", flush=True)
print("DONE", flush=True)
