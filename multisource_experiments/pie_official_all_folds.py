"""
Scaled-up version of pie_official_split_smoke_test.py: runs the same
leave-one-context-out comparison across ALL FOUR folds of Arc's official
replogle_wdataset/unseen_ctx splits (hepg2, jurkat, k562, rpe1 each held out
in turn), not just hepg2. Same data, same no-leakage discipline (weights
fit via CV entirely within the 3 training contexts for a given fold, never
touching that fold's held-out test context), same full-profile delta
Pearson / dir20 / coverage metrics as the smoke test - generalized to loop
over every fold instead of hardcoding hepg2.

Goal: establish whether "multi-source mixture beats any single source"
holds consistently across all 4 held-out contexts, or only for hepg2 -
before treating it as a result worth acting on for the real pipeline.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).parent.parent
PIE_DIR = ROOT / "data" / "arc_pie" / "preprocessed"
SPLITS_ROOT = ROOT / "data" / "arc_pie" / "splits" / "replogle_wdataset" / "unseen_ctx"
OUT_PATH = ROOT / "vcc_agent_harness" / "experiments" / "pie_all_folds_results.json"

ALL_CONTEXTS = ["hepg2", "jurkat", "k562", "rpe1"]
TOPK_DE = 20
RIDGE_LAMBDA = 1.0

meta = json.load(open(PIE_DIR / "meta.json"))
genes = meta["genes"]
context_to_id = meta["context_to_id"]
pert_to_id = meta["pert_to_id"]
ctx_ids = np.load(PIE_DIR / "ctx_ids.npy")
pert_ids = np.load(PIE_DIR / "pert_ids.npy")
delta_p = np.load(PIE_DIR / "delta_p.npy")
gene_idx = {g: i for i, g in enumerate(genes)}
row_index = {(int(c), int(p)): i for i, (c, p) in enumerate(zip(ctx_ids, pert_ids))}
print(f"Loaded preprocessed arrays: {delta_p.shape[0]} rows, {delta_p.shape[1]} genes, "
      f"{len(context_to_id)} contexts.", flush=True)


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


def run_fold(test_context):
    train_contexts = [c for c in ALL_CONTEXTS if c != test_context]
    split_dir = SPLITS_ROOT / test_context
    train_split = json.load(open(split_dir / "train.json"))
    val_split = json.load(open(split_dir / "val.json"))
    test_split = json.load(open(split_dir / "test.json"))

    train_rows = {c: rows_for(c, train_split[f"replogle.{c}"]) for c in train_contexts}
    test_rows = rows_for(test_context, test_split[f"replogle.{test_context}"])
    print(f"\n{'='*70}\nFold: held-out context = {test_context} "
          f"(train contexts = {train_contexts})\n{'='*70}", flush=True)
    for c in train_contexts:
        print(f"  train[{c}]: {len(train_rows[c])} resolved rows", flush=True)
    print(f"  test[{test_context}]: {len(test_rows)} resolved rows", flush=True)

    def ontarget_vec(pert_dict):
        return {p: float(delta_p[row, gene_idx[p]]) for p, row in pert_dict.items() if p in gene_idx}

    train_ontarget = {c: ontarget_vec(train_rows[c]) for c in train_contexts}

    # Fit mixture weights via leave-one-training-context-out CV, entirely within train_contexts.
    weight_sum = {c: 0.0 for c in train_contexts}
    weight_count = {c: 0.0 for c in train_contexts}
    for held in train_contexts:
        others = [c for c in train_contexts if c != held]
        y, X = [], []
        for p in train_ontarget[held]:
            feats = [train_ontarget[o].get(p, np.nan) for o in others]
            if all(not np.isnan(f) for f in feats):
                y.append(train_ontarget[held][p])
                X.append(feats)
        if len(y) < 10:
            continue
        y, X = np.array(y), np.array(X)
        A = X.T @ X + RIDGE_LAMBDA * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        for c, wi in zip(others, w):
            weight_sum[c] += wi
            weight_count[c] += 1
    learned_weight = {c: (weight_sum[c] / weight_count[c] if weight_count[c] > 0 else 1.0 / len(train_contexts))
                       for c in train_contexts}
    print(f"  learned source weights: { {k: round(v, 3) for k, v in learned_weight.items()} }", flush=True)

    variants = {f"single_{c}": {c: 1.0} for c in train_contexts}
    variants["equal_mixture"] = {c: 1.0 / len(train_contexts) for c in train_contexts}
    variants["learned_weight_mixture"] = learned_weight

    fold_results = {}
    for name, weights in variants.items():
        per_pert_pearson, per_pert_dir20, covered = [], [], 0
        for p, test_row_idx in test_rows.items():
            true_row = delta_p[test_row_idx]
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
        fold_results[name] = {
            "n": covered, "coverage": covered / len(test_rows),
            "pearson_full_profile_mean": float(np.mean(per_pert_pearson)) if per_pert_pearson else None,
            "pearson_full_profile_median": float(np.median(per_pert_pearson)) if per_pert_pearson else None,
            "dir20_mean": float(np.mean(per_pert_dir20)) if per_pert_dir20 else None,
        }
        r = fold_results[name]
        print(f"  {name:28s} n={r['n']:4d} coverage={r['coverage']:.3f} "
              f"pearson_full={r['pearson_full_profile_mean']:.4f} dir20={r['dir20_mean']:.4f}", flush=True)

    best_single = max((fold_results[f"single_{c}"]["pearson_full_profile_mean"] for c in train_contexts),
                       default=None)
    mixture_beats_best_single = fold_results["equal_mixture"]["pearson_full_profile_mean"] > best_single
    print(f"  best single-source pearson={best_single:.4f}; equal_mixture beats it: "
          f"{mixture_beats_best_single}", flush=True)
    return {"train_contexts": train_contexts, "learned_weight": learned_weight, "results": fold_results,
            "best_single_source_pearson": best_single, "equal_mixture_beats_best_single": mixture_beats_best_single}


all_fold_results = {}
for test_context in ALL_CONTEXTS:
    all_fold_results[test_context] = run_fold(test_context)

print(f"\n{'='*70}\nSummary across all 4 folds\n{'='*70}", flush=True)
n_mixture_wins = sum(1 for f in all_fold_results.values() if f["equal_mixture_beats_best_single"])
print(f"Equal mixture beats the best single source in {n_mixture_wins}/4 folds", flush=True)
for test_context, f in all_fold_results.items():
    eq = f["results"]["equal_mixture"]["pearson_full_profile_mean"]
    lw = f["results"]["learned_weight_mixture"]["pearson_full_profile_mean"]
    print(f"  {test_context:8s}: best_single={f['best_single_source_pearson']:.4f}  "
          f"equal_mixture={eq:.4f}  learned_mixture={lw:.4f}", flush=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({
    "folds": all_fold_results,
    "n_folds_mixture_beats_best_single": n_mixture_wins,
    "note": "delta_p is a log1p-difference (perturbed minus control pseudobulk); full-profile "
            "Pearson is the per-perturbation mean-profile correlation across all genes, the "
            "metric comparable to the GEARS-track baseline framing (though that was an "
            "unseen-GENE task within one cell line, while this is a known-gene, "
            "cross-CELL-LINE task - not directly comparable numbers, same caveat as the smoke test).",
}, indent=2))
print(f"\nSaved to {OUT_PATH}")
print("DONE")
