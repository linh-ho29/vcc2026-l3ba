#!/usr/bin/env python
"""Track J evaluate.py: held-out evaluation of the Jurkat GEARS model.

Metrics per held-out target (and vs. control-only / mean-shift baselines):
  pearson_delta, spearman_delta, de_delta_corr (top-20 DE genes),
  effect MSE / MAE (delta space), top20_de_direction_accuracy,
  magnitude calibration (corr of |pred delta| vs |true delta| on top-20 DE).

Stratified by: GO community, GO-degree bin, basal-expression bin, cell count
(exposure), and GO-graph distance to nearest training target.
"""
import argparse
import json
import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from gears.inference import evaluate
from gears.pertdata import PertData


def delta_metrics(pred_delta, true_delta, de_idx):
    out = {
        "pearson_delta": pearsonr(pred_delta, true_delta)[0],
        "spearman_delta": spearmanr(pred_delta, true_delta)[0],
        "effect_mse": float(np.mean((pred_delta - true_delta) ** 2)),
        "effect_mae": float(np.mean(np.abs(pred_delta - true_delta))),
    }
    d = de_idx[:20]
    pd_, td_ = pred_delta[d], true_delta[d]
    out["de_delta_corr_top20"] = pearsonr(pd_, td_)[0] if len(d) > 2 else np.nan
    out["top20_de_direction_accuracy"] = float(np.mean(np.sign(pd_) == np.sign(td_)))
    out["magnitude_calibration_r"] = pearsonr(np.abs(pd_), np.abs(td_))[0] if len(d) > 2 else np.nan
    out["magnitude_ratio_med"] = float(np.median(np.abs(pd_) / (np.abs(td_) + 1e-8)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/results/jurkat_gears/best_checkpoint.pt")
    ap.add_argument("--data_dir", default="/workspace/jurkat/gears_data")
    ap.add_argument("--dataset", default="nadig25jurkat")
    ap.add_argument("--results_dir", default="/mnt/results/jurkat_gears")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random_control", action="store_true",
                    help="evaluate the matched random-sampling control model instead")
    args = ap.parse_args()

    pd_ = PertData(data_path=args.data_dir)
    pd_.load(data_path=os.path.join(args.data_dir, args.dataset))
    # Track J custom split (biology-informed, response-blind) — must match train.py
    custom_split = os.path.join(args.data_dir, args.dataset, "splits", "custom_trackJ.pkl")
    pd_.prepare_split(split="custom", split_dict_path=custom_split)
    pd_.get_dataloader(batch_size=8, test_batch_size=16)
    adata = pd_.adata

    from gears import GEARS
    model = GEARS(pd_, device="cpu")
    model.model_initialize(hidden_size=32)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.best_model.load_state_dict(ck["state_dict"])
    model.best_model.eval()

    test_res = evaluate(pd_.dataloader["test_loader"], model.best_model,
                        model.config["uncertainty"], "cpu")

    # control mean + training-mean delta (mean-shift baseline)
    conds = adata.obs["condition"].values
    X = adata.X
    ctrl_mean = np.asarray(X[conds == "ctrl"].mean(axis=0)).ravel()
    split_manifest = json.load(open(os.path.join(args.results_dir, "split_manifest.json")))
    train_targets = split_manifest["train_targets"]
    train_deltas = [np.asarray(X[conds == f"{g}+ctrl"].mean(axis=0)).ravel() - ctrl_mean
                    for g in train_targets if f"{g}+ctrl" in set(conds)]
    mean_shift_delta = np.mean(train_deltas, axis=0)

    # GO metadata for stratification
    cov = pd.read_csv(os.path.join(args.results_dir, "pathway_coverage_report.csv"))
    cov = cov.set_index("gene")
    gene2go = pickle.load(open("/workspace/data/gene2go_all.pkl", "rb"))
    go_edges = pd.read_csv("/workspace/data/go_essential_all/go_essential_all.csv")
    import networkx as nx
    G = nx.from_pandas_edgelist(go_edges, "source", "target")

    def graph_dist_to_train(gene):
        if gene not in G:
            return np.inf
        best = np.inf
        for t in train_targets:
            if t in G and nx.has_path(G, gene, t):
                best = min(best, nx.shortest_path_length(G, gene, t))
        return best

    rows = []
    perts = [p for p in np.unique(test_res["pert_cat"]) if p != "ctrl"]
    for pert in perts:
        i = np.where(test_res["pert_cat"] == pert)[0]
        gene = pert.replace("+ctrl", "")
        pred_mean = test_res["pred"][i].mean(0)
        true_mean = test_res["truth"][i].mean(0)
        pred_delta, true_delta = pred_mean - ctrl_mean, true_mean - ctrl_mean
        de_genes = adata.uns["rank_genes_groups_cov_all"].get(
            dict(adata.obs[["condition", "condition_name"]].values)[pert])
        de_idx = [list(adata.var_names).index(g) for g in de_genes[:20]
                  if g in set(adata.var_names)]
        m = delta_metrics(pred_delta, true_delta, de_idx)
        m.update({"perturbation": gene,
                  "n_cells": int(i.sum()),
                  "go_category": cov.at[gene, "go_category"] if gene in cov.index else "na",
                  "deg_bin": cov.at[gene, "deg_bin"] if gene in cov.index else "na",
                  "basal_bin": cov.at[gene, "basal_bin"] if gene in cov.index else "na",
                  "graph_dist_to_train": graph_dist_to_train(gene),
                  "n_go_terms": len(gene2go.get(gene, []))})
        # baselines for the same target
        m["ctrl_only_pearson_delta"] = np.nan  # control-only predicts zero delta (undefined corr)
        m["ctrl_only_effect_mse"] = float(np.mean(true_delta ** 2))
        m["ctrl_only_top20_dir"] = float(np.mean(np.sign(true_delta[de_idx[:20]]) == 0)) if de_idx else np.nan
        ms_pred = ctrl_mean + mean_shift_delta
        m["meanshift_pearson_delta"] = pearsonr(ms_pred - ctrl_mean, true_delta)[0]
        m["meanshift_effect_mse"] = float(np.mean((ms_pred - true_mean) ** 2))
        m["meanshift_top20_dir"] = float(np.mean(
            np.sign((ms_pred - ctrl_mean)[de_idx[:20]]) == np.sign(true_delta[de_idx[:20]]))) if de_idx else np.nan
        rows.append(m)

    df = pd.DataFrame(rows)
    label = "random_control" if args.random_control else "balanced"
    df.to_csv(os.path.join(args.results_dir, f"metrics_by_perturbation_{label}.csv"), index=False)

    num = df.select_dtypes(include=[np.number]).columns
    summary = {"model": label, "n_test_perts": len(df),
               "mean_metrics": df[num].mean().round(4).to_dict(),
               "median_graph_dist_to_train": float(df.graph_dist_to_train.replace(np.inf, np.nan).median())}
    with open(os.path.join(args.results_dir, f"evaluation_summary_{label}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[eval] " + json.dumps(summary["mean_metrics"], default=str), flush=True)


if __name__ == "__main__":
    main()
