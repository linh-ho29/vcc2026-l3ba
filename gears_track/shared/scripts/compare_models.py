#!/usr/bin/env python
"""
Lightweight per-perturbation metrics computed directly from prediction arrays
(no h5ad / DE uns needed). Works uniformly for GEARS and the custom baselines.

For each model preds dir (standard contract: test_pred.npy, test_truth.npy,
test_pertcat.npy, test_de_idx.npy, ctrl_mean.npy), computes per-perturbation:
  pearson_delta      - pearson(pred_delta, truth_delta) over all genes
  pearson_delta_de   - same over top-20 DE genes
  pearson_de         - pearson(pred, truth) over top-20 DE genes (profile, not delta)
  dir20              - fraction of top-20 DE genes with matching delta sign
  mse_de             - MSE of mean profiles over top-20 DE genes
Metrics are computed on per-perturbation mean profiles (GEARS convention).

Outputs:
  --out_pkl   : dict model -> {"rows": [per-pert dicts], "summary": {...}}
  --out_table : CSV comparison table (models x metrics)
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def load_arrays(preds_dir):
    pred = np.load(f"{preds_dir}/test_pred.npy")
    truth = np.load(f"{preds_dir}/test_truth.npy")
    if truth.ndim == 3:  # (cells, 1, genes) -> (cells, genes)
        truth = truth[:, 0, :]
    pert = np.load(f"{preds_dir}/test_pertcat.npy", allow_pickle=True)
    de_idx = np.load(f"{preds_dir}/test_de_idx.npy")
    ctrl = np.load(f"{preds_dir}/ctrl_mean.npy")
    return pred, truth, pert, de_idx, ctrl


def per_pert_rows(pred, truth, pert, de_idx, ctrl):
    uniq = np.unique(pert)
    rows = []
    for p in uniq:
        m = pert == p
        pm = pred[m].mean(0); tm = truth[m].mean(0)
        idx = de_idx[m][0] if de_idx.ndim == 2 else de_idx
        pd_ = pm - ctrl; td = tm - ctrl
        r = {
            "pert": str(p),
            "n_cells": int(m.sum()),
            "pearson_delta": pearson(pd_, td),
            "pearson_delta_de": pearson(pd_[idx], td[idx]),
            "pearson_de": pearson(pm[idx], tm[idx]),
            "dir20": float(np.mean(np.sign(pd_[idx]) == np.sign(td[idx]))),
            "mse_de": float(np.mean((pm[idx] - tm[idx]) ** 2)),
        }
        rows.append(r)
    return rows


def summarize(rows):
    keys = ["pearson_delta", "pearson_delta_de", "pearson_de", "dir20", "mse_de"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r[k] == r[k]])
        out[f"{k}_mean"] = float(v.mean()); out[f"{k}_median"] = float(np.median(v))
    out["n_perts"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="model=path pairs, e.g. gears=/mnt/shared-workspace/preds/gears")
    ap.add_argument("--out_pkl", required=True)
    ap.add_argument("--out_table", required=True)
    args = ap.parse_args()

    all_rows, summaries = {}, {}
    for spec in args.preds:
        model, path = spec.split("=", 1)
        if not Path(path, "test_pred.npy").exists():
            print(f"[skip] {model}: no test_pred.npy in {path}")
            continue
        pred, truth, pert, de_idx, ctrl = load_arrays(path)
        rows = per_pert_rows(pred, truth, pert, de_idx, ctrl)
        all_rows[model] = rows
        summaries[model] = summarize(rows)
        print(f"[{model}] n_perts={summaries[model]['n_perts']}  "
              f"pearson_delta={summaries[model]['pearson_delta_mean']:.4f}  "
              f"pearson_delta_de={summaries[model]['pearson_delta_de_mean']:.4f}  "
              f"dir20={summaries[model]['dir20_mean']:.4f}  "
              f"mse_de={summaries[model]['mse_de_mean']:.4f}", flush=True)

    with open(args.out_pkl, "wb") as f:
        pickle.dump({"rows": all_rows, "summary": summaries}, f)

    # comparison table
    metrics = ["pearson_delta_mean", "pearson_delta_de_mean", "pearson_de_mean",
               "dir20_mean", "mse_de_mean"]
    with open(args.out_table, "w") as f:
        f.write("model," + ",".join(metrics) + ",n_perts\n")
        for model, s in summaries.items():
            f.write(model + "," + ",".join(f"{s[m]:.4f}" for m in metrics)
                    + f",{s['n_perts']}\n")
    print("[done] ->", args.out_pkl, args.out_table)


if __name__ == "__main__":
    main()
