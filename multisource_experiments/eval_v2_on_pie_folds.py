"""
Does the v2 context-weight model (fit on pooled 4-context Arc data) beat
plain equal-weight mixture within Arc's own all-folds framework? Reuses
the exact same fold/metric logic as pie_official_all_folds.py, adding one
more variant: per-gene-weighted mixture using context_weight_model_v2.npz.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).parent.parent
PIE_DIR = ROOT / "data" / "arc_pie" / "preprocessed"
SPLITS_ROOT = ROOT / "data" / "arc_pie" / "splits" / "replogle_wdataset" / "unseen_ctx"
MODEL_PATH = ROOT / "predictions" / "context_weight_model_v2.npz"
OUT_PATH = ROOT / "vcc_agent_harness" / "experiments" / "E09_v2_vs_mixture_on_pie.json"

ALL_CONTEXTS = ["hepg2", "jurkat", "k562", "rpe1"]
TOPK_DE = 20

meta = json.load(open(PIE_DIR / "meta.json"))
genes = meta["genes"]
context_to_id = meta["context_to_id"]
pert_to_id = meta["pert_to_id"]
ctrl_means = np.load(PIE_DIR / "ctrl_means.npy")
ctx_ids = np.load(PIE_DIR / "ctx_ids.npy")
pert_ids = np.load(PIE_DIR / "pert_ids.npy")
delta_p = np.load(PIE_DIR / "delta_p.npy")
gene_idx = {g: i for i, g in enumerate(genes)}
row_index = {(int(c), int(p)): i for i, (c, p) in enumerate(zip(ctx_ids, pert_ids))}

model = np.load(MODEL_PATH, allow_pickle=True)
coef, feat_mean, feat_std = model["coef"], model["feat_mean"], model["feat_std"]


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


def v2_weight_for_context(context_name, train_perts_for_target):
    cid = context_to_id[context_name]
    baseline = ctrl_means[cid]
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
    is_pert_target = np.array([g in train_perts_for_target for g in genes], dtype=np.float64)
    X = np.column_stack([baseline, percentile, pathway_ribo, pathway_mito, pathway_histone,
                          pathway_hemo, is_pert_target])
    Xb = np.hstack([(X - feat_mean) / feat_std, np.ones((X.shape[0], 1))])
    return np.clip(Xb @ coef, 0.0, 1.0)


all_fold_results = {}
for test_context in ALL_CONTEXTS:
    train_contexts = [c for c in ALL_CONTEXTS if c != test_context]
    split_dir = SPLITS_ROOT / test_context
    train_split = json.load(open(split_dir / "train.json"))
    test_split = json.load(open(split_dir / "test.json"))
    train_rows = {c: rows_for(c, train_split[f"replogle.{c}"]) for c in train_contexts}
    test_rows = rows_for(test_context, test_split[f"replogle.{test_context}"])

    w_v2 = v2_weight_for_context(test_context, set(train_split[f"replogle.{test_context}"])
                                  if f"replogle.{test_context}" in train_split else set())

    equal_pearsons, v2_pearsons = [], []
    for p, test_row_idx in test_rows.items():
        true_row = delta_p[test_row_idx]
        available = [c for c in train_contexts if p in train_rows[c]]
        if not available:
            continue
        equal_pred = np.mean([delta_p[train_rows[c][p]] for c in available], axis=0)
        v2_pred = np.zeros(len(genes))
        for c in available:
            v2_pred = v2_pred + w_v2 * delta_p[train_rows[c][p]]
        v2_pred = v2_pred / len(available)
        if np.std(equal_pred) > 0:
            equal_pearsons.append(pearsonr(true_row, equal_pred)[0])
        if np.std(v2_pred) > 0:
            v2_pearsons.append(pearsonr(true_row, v2_pred)[0])

    all_fold_results[test_context] = {
        "equal_mixture_pearson_mean": float(np.mean(equal_pearsons)),
        "v2_weighted_pearson_mean": float(np.mean(v2_pearsons)),
        "v2_beats_equal": float(np.mean(v2_pearsons)) > float(np.mean(equal_pearsons)),
    }
    r = all_fold_results[test_context]
    print(f"{test_context:8s}: equal_mixture={r['equal_mixture_pearson_mean']:.4f}  "
          f"v2_weighted={r['v2_weighted_pearson_mean']:.4f}  v2_beats_equal={r['v2_beats_equal']}", flush=True)

n_wins = sum(1 for r in all_fold_results.values() if r["v2_beats_equal"])
print(f"\nv2 beats equal-mixture in {n_wins}/4 folds", flush=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({"folds": all_fold_results, "n_folds_v2_beats_equal_mixture": n_wins}, indent=2))
print(f"Saved to {OUT_PATH}")
print("DONE")
