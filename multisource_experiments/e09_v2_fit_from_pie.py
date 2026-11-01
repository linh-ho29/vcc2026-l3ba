"""
E09 v2: refit the context-weighting model using Arc's 4-context PIE data
instead of just the K562<->RPE1 pair E09 v1 used. Goal: more training
signal (4 contexts x ~1850 perturbations each, official splits) to see if
a feature-based weighting model can finally beat naive equal-weighting -
v1 could not (0.664 vs 0.729 on the HepG2 check).

Per-gene reliability target, pooled across folds: for each of the 4
official unseen_ctx folds, and each of its 3 training contexts T, average
the per-gene correlation between every OTHER training context's effect and
T's own real effect, across perturbations in both contexts' TRAIN splits
only (never a fold's test split, never any val split used for fitting).
Each context contributes one reliability vector per fold it trains in (3
each, slightly different per fold's exact train-split) - pooled as
separate rows rather than picking one, since ridge handles near-duplicates
fine and it avoids an arbitrary "pick one fold" choice.

Features, all derived from ctrl_means.npy (log1p pseudobulk control
expression - same pipeline, same scale across all 4 contexts, an
improvement over v1 which mixed RPE1's raw-bulk file with the K562 bank's
control_mean from a different source): target's own baseline expression
and expression percentile, 4 standard gene-family pathway flags, and
whether the gene is itself a perturbation target anywhere in the pooled
training data.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
PIE_DIR = ROOT / "data" / "arc_pie" / "preprocessed"
SPLITS_ROOT = ROOT / "data" / "arc_pie" / "splits" / "replogle_wdataset" / "unseen_ctx"
OUT_MODEL = ROOT / "predictions" / "context_weight_model_v2.npz"
OUT_REPORT = ROOT / "vcc_agent_harness" / "experiments" / "E09_v2_fit_report.json"

ALL_CONTEXTS = ["hepg2", "jurkat", "k562", "rpe1"]
MIN_PAIRS = 30
RIDGE_LAMBDA = 10.0

meta = json.load(open(PIE_DIR / "meta.json"))
genes = meta["genes"]
context_to_id = meta["context_to_id"]
pert_to_id = meta["pert_to_id"]
ctrl_means = np.load(PIE_DIR / "ctrl_means.npy")  # (4, 6642), log1p pseudobulk control mean
ctx_ids = np.load(PIE_DIR / "ctx_ids.npy")
pert_ids = np.load(PIE_DIR / "pert_ids.npy")
delta_p = np.load(PIE_DIR / "delta_p.npy")
gene_idx = {g: i for i, g in enumerate(genes)}
row_index = {(int(c), int(p)): i for i, (c, p) in enumerate(zip(ctx_ids, pert_ids))}
all_perts_by_context = {c: set(json.load(open(SPLITS_ROOT / c / "train.json")).get(f"replogle.{c}", []))
                         for c in ALL_CONTEXTS}  # for is_perturbation_target - from fold=c's own train list


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


def context_features(context_name, is_pert_target_mask):
    cid = context_to_id[context_name]
    baseline = ctrl_means[cid]  # (n_genes,), already log1p
    order = np.argsort(baseline)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order))
    percentile = 100.0 * ranks / max(len(order) - 1, 1)
    pathway_ribo = np.array([g.startswith(("RPL", "RPS")) for g in genes], dtype=np.float64)
    pathway_mito = np.array([g.startswith("MT-") for g in genes], dtype=np.float64)
    pathway_histone = np.array([g.startswith(("HIST", "H1-", "H2A", "H2B", "H3-", "H4-")) for g in genes],
                                dtype=np.float64)
    pathway_hemo = np.array([g.startswith(("HBA", "HBB", "HBZ", "HBG", "HBD", "HBE")) for g in genes],
                             dtype=np.float64)
    return np.column_stack([baseline, percentile, pathway_ribo, pathway_mito, pathway_histone,
                             pathway_hemo, is_pert_target_mask])


feature_names = ["baseline_expression", "percentile", "pathway_ribosomal", "pathway_mitochondrial",
                  "pathway_histone", "pathway_hemoglobin", "is_perturbation_target"]

pooled_X, pooled_y = [], []
print("Pooling per-gene reliability targets across all 4 folds...", flush=True)
for held_out in ALL_CONTEXTS:
    train_contexts = [c for c in ALL_CONTEXTS if c != held_out]
    train_split = json.load(open(SPLITS_ROOT / held_out / "train.json"))
    train_rows = {c: rows_for(c, train_split[f"replogle.{c}"]) for c in train_contexts}

    for target in train_contexts:
        sources = [c for c in train_contexts if c != target]
        reliab_sum = np.zeros(len(genes))
        reliab_count = np.zeros(len(genes))
        for source in sources:
            shared = sorted(set(train_rows[source]) & set(train_rows[target]))
            if len(shared) < MIN_PAIRS:
                continue
            s_mat = np.stack([delta_p[train_rows[source][p]] for p in shared])
            t_mat = np.stack([delta_p[train_rows[target][p]] for p in shared])
            s_c = s_mat - s_mat.mean(axis=0, keepdims=True)
            t_c = t_mat - t_mat.mean(axis=0, keepdims=True)
            num = (s_c * t_c).sum(axis=0)
            denom = np.sqrt((s_c ** 2).sum(axis=0) * (t_c ** 2).sum(axis=0))
            valid = denom > 0
            corr = np.zeros(len(genes))
            corr[valid] = num[valid] / denom[valid]
            reliab_sum[valid] += corr[valid]
            reliab_count[valid] += 1
        has_signal = reliab_count > 0
        reliability = np.full(len(genes), np.nan)
        reliability[has_signal] = (reliab_sum[has_signal] / reliab_count[has_signal] + 1.0) / 2.0  # [-1,1] -> [0,1]

        is_pert_target = np.array([g in all_perts_by_context[target] for g in genes], dtype=np.float64)
        X = context_features(target, is_pert_target)
        valid_rows = has_signal & ~np.isnan(X).any(axis=1)
        pooled_X.append(X[valid_rows])
        pooled_y.append(reliability[valid_rows])
        print(f"  fold(held_out={held_out}) target={target}: {int(valid_rows.sum())} genes with signal "
              f"(from sources {sources})", flush=True)

pooled_X = np.vstack(pooled_X)
pooled_y = np.concatenate(pooled_y)
print(f"\nPooled training table: {pooled_X.shape[0]} rows (genes x fold-target appearances), "
      f"{pooled_X.shape[1]} features", flush=True)

feat_mean, feat_std = pooled_X.mean(axis=0), pooled_X.std(axis=0)
feat_std[feat_std == 0] = 1.0
X_std = (pooled_X - feat_mean) / feat_std
Xb = np.hstack([X_std, np.ones((X_std.shape[0], 1))])
A = Xb.T @ Xb + RIDGE_LAMBDA * np.eye(Xb.shape[1])
coef = np.linalg.solve(A, Xb.T @ pooled_y)
pred = np.clip(Xb @ coef, 0.0, 1.0)
r2 = 1 - np.sum((pooled_y - pred) ** 2) / np.sum((pooled_y - pooled_y.mean()) ** 2)
print(f"Train R^2 (in-sample, pooled): {r2:.4f}", flush=True)
print(f"Coefficients: {dict(zip(feature_names + ['intercept'], coef.round(4)))}", flush=True)

np.savez(OUT_MODEL, coef=coef, feat_mean=feat_mean, feat_std=feat_std,
         feature_names=np.array(feature_names), l2=np.array([RIDGE_LAMBDA]))
print(f"Saved frozen v2 model to {OUT_MODEL}", flush=True)

OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
OUT_REPORT.write_text(json.dumps({
    "n_pooled_training_rows": int(pooled_X.shape[0]),
    "n_unique_genes": len(genes),
    "train_r2": float(r2),
    "coefficients": dict(zip(feature_names + ["intercept"], [float(c) for c in coef])),
    "l2_regularization": RIDGE_LAMBDA,
    "note": "Pooled across all 4 official unseen_ctx folds' internal train-context relationships - "
            "never uses any fold's held-out test context. v1 (predictions/context_weight_model.npz) "
            "was fit on the single K562<->RPE1 pair only; this version pools 12 (fold, target-context) "
            "reliability vectors across all 4 Arc PIE contexts.",
}, indent=2))
print(f"Saved fit report to {OUT_REPORT}", flush=True)
print("DONE", flush=True)
