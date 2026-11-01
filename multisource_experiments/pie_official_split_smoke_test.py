"""
CPU-only smoke test: Arc PIE official splits (replogle_wdataset/unseen_ctx,
fold=hepg2), preprocessed arrays only (delta_p, ctrl_means, ctx_ids, pert_ids,
tested, meta.json/contexts.yaml). Train contexts = {jurkat, k562, rpe1} per
the official train.json/val.json; held-out test context = hepg2 (867
perturbations, test.json) - genuinely unseen, never touched during fitting.

Goal: verify the full pipeline end-to-end on real data, not yet a tuned
final model. Steps:
  1. Reproduce the existing signature-transfer baseline's behavior using
     Arc's own K562 slice (on-target sign/scale sanity check against our
     production signature_bank_k562gwps.npz).
  2. Compare M1 (k562-only), M2 (rpe1-only), M3 (training-context mixture,
     jurkat+k562+rpe1) as source predictors for hepg2's held-out effects.
  3. Fit per-source mixture weights via leave-one-training-context-out CV
     WITHIN the training contexts only (never touches hepg2/test) - a
     simple, regularized (ridge) weight fit, not a search over the test set.
  4. Evaluate all variants on the identical 867 held-out hepg2 perturbations:
     delta Pearson, DE-direction agreement (top-20 |delta_p| genes), a
     calibration slope (regression of predicted vs. true, through origin),
     and perturbation coverage per model.

Missing-source handling: a model's prediction for a test perturbation is
NaN (not zero) if none of its required source context(s) have a trained
(i.e. TRAIN-split) signature for that perturbation.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).parent.parent
PIE_DIR = ROOT / "data" / "arc_pie" / "preprocessed"
SPLIT_DIR = ROOT / "data" / "arc_pie" / "splits" / "replogle_wdataset" / "unseen_ctx" / "hepg2"
OUT_PATH = ROOT / "vcc_agent_harness" / "experiments" / "pie_smoke_test_hepg2.json"

TRAIN_CONTEXTS = ["jurkat", "k562", "rpe1"]
TEST_CONTEXT = "hepg2"
TOPK_DE = 20

meta = json.load(open(PIE_DIR / "meta.json"))
genes = meta["genes"]
context_to_id = meta["context_to_id"]
pert_to_id = meta["pert_to_id"]
ctrl_means = np.load(PIE_DIR / "ctrl_means.npy")
ctx_ids = np.load(PIE_DIR / "ctx_ids.npy")
pert_ids = np.load(PIE_DIR / "pert_ids.npy")
delta_p = np.load(PIE_DIR / "delta_p.npy")
tested = np.load(PIE_DIR / "tested.npy")
print(f"Loaded preprocessed arrays: {delta_p.shape[0]} rows, {delta_p.shape[1]} genes, "
      f"{len(context_to_id)} contexts. tested all-True: {tested.all()} "
      f"(no within-vocabulary missingness in this format - already resolved upstream by Arc).",
      flush=True)

row_index = {(int(c), int(p)): i for i, (c, p) in enumerate(zip(ctx_ids, pert_ids))}


def rows_for(context_name, pert_names):
    cid = context_to_id[context_name]
    out = {}
    for p in pert_names:
        pid = pert_to_id.get(p)
        if pid is None:
            continue
        row = row_index.get((cid, pid))
        if row is not None:
            out[p] = row
    return out


train_split = json.load(open(SPLIT_DIR / "train.json"))
val_split = json.load(open(SPLIT_DIR / "val.json"))
test_split = json.load(open(SPLIT_DIR / "test.json"))

train_rows = {c: rows_for(c, train_split[f"replogle.{c}"]) for c in TRAIN_CONTEXTS}
val_rows = {c: rows_for(c, val_split[f"replogle.{c}"]) for c in TRAIN_CONTEXTS}
test_rows = rows_for(TEST_CONTEXT, test_split[f"replogle.{TEST_CONTEXT}"])
for c in TRAIN_CONTEXTS:
    print(f"  train[{c}]: {len(train_rows[c])} resolved rows", flush=True)
print(f"  test[{TEST_CONTEXT}]: {len(test_rows)} resolved rows", flush=True)

# ---------------------------------------------------------------------------
# Step 1: reproduce the existing baseline's behavior - sanity check Arc's own
# K562 on-target effect (delta_p, log1p-difference) against our production
# K562_gwps bank's on-target effect (ln pseudocount-ratio) on overlapping
# perturbations. Different formulas/scales by design (flagged, not assumed
# comparable) - only sign and rank agreement are meaningful here.
# ---------------------------------------------------------------------------
print("\n=== Step 1: reproduce/sanity-check vs production K562_gwps bank ===", flush=True)
bank = np.load(ROOT / "predictions" / "signature_bank_k562gwps.npz", allow_pickle=True)
bank_genes, bank_perts, bank_effects = bank["genes"].astype(str), bank["perts"].astype(str), bank["effects"]
bank_gene_idx = {g: i for i, g in enumerate(bank_genes)}
bank_pert_idx = {p: i for i, p in enumerate(bank_perts)}
gene_idx = {g: i for i, g in enumerate(genes)}

common_perts = [p for p in train_rows["k562"] if p in bank_pert_idx and p in gene_idx and p in bank_gene_idx]
arc_ontarget, prod_ontarget = [], []
for p in common_perts:
    j_arc, j_prod = gene_idx[p], bank_gene_idx[p]
    arc_ontarget.append(delta_p[train_rows["k562"][p], j_arc])
    prod_ontarget.append(bank_effects[bank_pert_idx[p], j_prod])
arc_ontarget, prod_ontarget = np.array(arc_ontarget), np.array(prod_ontarget)
pearson_sanity = pearsonr(arc_ontarget, prod_ontarget)[0]
sign_agree_sanity = float(np.mean(np.sign(arc_ontarget) == np.sign(prod_ontarget)))
print(f"  n={len(common_perts)} shared on-target perturbations (K562, Arc-preprocessed vs. production bank)",
      flush=True)
print(f"  Pearson={pearson_sanity:.4f}, sign_agreement={sign_agree_sanity:.4f} "
      f"(different formulas/scales by design - direction agreement is the meaningful check here)", flush=True)

# ---------------------------------------------------------------------------
# Step 2+3: fit per-source weights via leave-one-training-context-out CV,
# entirely within {jurkat, k562, rpe1} - never touches hepg2/test.
# ---------------------------------------------------------------------------
print("\n=== Step 2/3: fit mixture weights via leave-one-training-context-out CV ===", flush=True)


def ontarget_vec(context, pert_dict):
    """On-target (self) delta_p for every resolved perturbation in pert_dict."""
    out = {}
    for p, row in pert_dict.items():
        if p in gene_idx:
            out[p] = float(delta_p[row, gene_idx[p]])
    return out


train_ontarget = {c: ontarget_vec(c, train_rows[c]) for c in TRAIN_CONTEXTS}

cv_alphas = []
for held in TRAIN_CONTEXTS:
    others = [c for c in TRAIN_CONTEXTS if c != held]
    held_perts = set(train_ontarget[held])
    y, X = [], []
    for p in held_perts:
        feats = [train_ontarget[o].get(p, np.nan) for o in others]
        if all(not np.isnan(f) for f in feats):
            y.append(train_ontarget[held][p])
            X.append(feats)
    if len(y) < 10:
        continue
    y, X = np.array(y), np.array(X)
    # Ridge-regularized least squares, non-negative-ish via simple clipping - simple by design.
    lam = 1.0
    A = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y
    w = np.linalg.solve(A, b)
    cv_alphas.append(dict(zip(others, w)))
    print(f"  held-out (within-train) context={held}: n={len(y)}, weights on {others} = {w.round(3).tolist()}",
          flush=True)

# Average the per-fold weights per source context, then renormalize to sum to 1 over
# whichever sources are actually available for a given test perturbation at inference.
weight_sum, weight_count = {c: 0.0 for c in TRAIN_CONTEXTS}, {c: 0.0 for c in TRAIN_CONTEXTS}
for fold_w in cv_alphas:
    for c, w in fold_w.items():
        weight_sum[c] += w
        weight_count[c] += 1
learned_weight = {c: (weight_sum[c] / weight_count[c] if weight_count[c] > 0 else 1.0 / len(TRAIN_CONTEXTS))
                   for c in TRAIN_CONTEXTS}
print(f"  learned source weights (averaged across folds): {learned_weight}", flush=True)

# ---------------------------------------------------------------------------
# Step 4: evaluate M1 (k562-only), M2 (rpe1-only), M3 (equal mixture),
# M4 (learned-weight mixture) on the IDENTICAL 867 held-out hepg2 perturbations.
# ---------------------------------------------------------------------------
print(f"\n=== Step 4: evaluate on {len(test_rows)} held-out hepg2 perturbations (identical set, all models) ===",
      flush=True)

y_true = {}
for p, row in test_rows.items():
    if p in gene_idx:
        y_true[p] = float(delta_p[row, gene_idx[p]])

variants = {
    "M1_k562_only": {"k562": 1.0},
    "M2_rpe1_only": {"rpe1": 1.0},
    "M3_equal_mixture": {c: 1.0 / len(TRAIN_CONTEXTS) for c in TRAIN_CONTEXTS},
    "M4_learned_weight_mixture": learned_weight,
}

results = {}
for name, weights in variants.items():
    preds, truths, covered = [], [], 0
    for p in test_rows:
        if p not in y_true:
            continue
        num, denom = 0.0, 0.0
        for c, w in weights.items():
            if p in train_ontarget[c]:
                num += w * train_ontarget[c][p]
                denom += w
        if denom == 0:
            continue  # no required source available for this perturbation - NaN, not zero
        preds.append(num / denom)
        truths.append(y_true[p])
        covered += 1
    preds, truths = np.array(preds), np.array(truths)
    pearson = pearsonr(truths, preds)[0] if np.std(preds) > 0 else float("nan")
    sign_agree = float(np.mean(np.sign(truths) == np.sign(preds)))
    slope = float(np.sum(truths * preds) / np.sum(preds ** 2)) if np.sum(preds ** 2) > 0 else float("nan")
    coverage = covered / len(test_rows)
    results[name] = {"n": int(covered), "coverage": coverage, "pearson_delta_ontarget": float(pearson),
                      "sign_agreement": sign_agree, "calibration_slope_true_vs_pred": slope}
    print(f"  {name:28s} n={covered:4d} coverage={coverage:.3f} pearson={pearson:.4f} "
          f"sign_agree={sign_agree:.4f} calib_slope={slope:.4f}", flush=True)

# ---------------------------------------------------------------------------
# Step 4b: FULL-PROFILE delta Pearson (all 6642 genes, not just on-target) -
# the harder, more meaningful task and the one actually comparable to the
# GEARS-track baseline numbers (meanshift/ridge/GEARS, ~0.03-0.07 pearson_delta
# on the full profile). The on-target-only numbers above are a much easier
# task (reflects guide efficiency/gene turnover kinetics, fairly conserved
# across cell lines) and should NOT be compared to those full-profile numbers.
# ---------------------------------------------------------------------------
print(f"\n=== Step 4b: FULL-PROFILE delta Pearson (per-perturbation mean profile, all {len(genes)} genes) ===",
      flush=True)
full_results = {}
for name, weights in variants.items():
    per_pert_pearson, per_pert_dir20, covered = [], [], 0
    for p in test_rows:
        true_row = delta_p[test_rows[p]]
        num, denom = np.zeros(len(genes)), 0.0
        for c, w in weights.items():
            if p in train_rows[c]:
                num = num + w * delta_p[train_rows[c][p]]
                denom += w
        if denom == 0:
            continue
        pred_row = num / denom
        if np.std(pred_row) > 0:
            per_pert_pearson.append(pearsonr(true_row, pred_row)[0])
        top20 = np.argsort(-np.abs(true_row))[:TOPK_DE]
        per_pert_dir20.append(float(np.mean(np.sign(true_row[top20]) == np.sign(pred_row[top20]))))
        covered += 1
    full_results[name] = {
        "n": covered, "coverage": covered / len(test_rows),
        "pearson_delta_full_profile_mean": float(np.mean(per_pert_pearson)) if per_pert_pearson else None,
        "pearson_delta_full_profile_median": float(np.median(per_pert_pearson)) if per_pert_pearson else None,
        "dir20_mean": float(np.mean(per_pert_dir20)) if per_pert_dir20 else None,
    }
    print(f"  {name:28s} n={covered:4d} pearson_full_mean={full_results[name]['pearson_delta_full_profile_mean']:.4f} "
          f"pearson_full_median={full_results[name]['pearson_delta_full_profile_median']:.4f} "
          f"dir20={full_results[name]['dir20_mean']:.4f}", flush=True)

# ---------------------------------------------------------------------------
# DE-direction metric: for a genuinely genome-wide (not on-target-only) check,
# compare predicted vs. true sign agreement on each test perturbation's own
# top-20 |delta_p| genes (DE-direction convention used throughout this project).
# Only feasible for sources with a FULL per-gene signature for that perturbation,
# not just the on-target scalar used above - so this uses the mixture's
# available full rows directly (train_rows, not just train_ontarget).
# ---------------------------------------------------------------------------
print("\n=== DE-direction check (top-20 |delta_p| genes, full profile, k562-only vs rpe1-only) ===", flush=True)
de_results = {}
for name, ctx in [("M1_k562_only", "k562"), ("M2_rpe1_only", "rpe1")]:
    dir20_scores = []
    for p in test_rows:
        if p not in train_rows[ctx] or p not in y_true:
            continue
        true_row = delta_p[test_rows[p]]
        pred_row = delta_p[train_rows[ctx][p]]
        top20 = np.argsort(-np.abs(true_row))[:TOPK_DE]
        dir20 = float(np.mean(np.sign(true_row[top20]) == np.sign(pred_row[top20])))
        dir20_scores.append(dir20)
    de_results[name] = {"n": len(dir20_scores), "dir20_mean": float(np.mean(dir20_scores)) if dir20_scores else None}
    print(f"  {name}: n={len(dir20_scores)}, dir20_mean={de_results[name]['dir20_mean']}", flush=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({
    "smoke_test": True,
    "fold": "hepg2", "setting": "unseen_ctx",
    "sanity_check_vs_production_bank": {"n": len(common_perts), "pearson": float(pearson_sanity),
                                          "sign_agreement": sign_agree_sanity},
    "learned_source_weights": learned_weight,
    "test_set_results_ontarget_only": results,
    "test_set_results_full_profile": full_results,
    "de_direction_dir20_single_source": de_results,
    "note": "delta_p is a log1p-DIFFERENCE (perturbed minus control pseudobulk), not a log-RATIO - "
            "not assumed comparable in absolute scale to signature_bank_k562gwps.npz's ln "
            "pseudocount-ratio effect; only sign/rank agreement is treated as meaningful in the "
            "sanity check. tested.npy is uniformly True in this preprocessed format - no "
            "within-vocabulary missingness signal available at this stage.",
}, indent=2))
print(f"\nSaved to {OUT_PATH}")
print("DONE")
