#!/usr/bin/env python
"""
Combined benchmark figures across models (GEARS + additive + meanshift + ridge
+ control-mean), from the comparison_rows pkl produced by compare_models.py.
Colorblind-friendly Phylo palette; PNG (150 dpi) + SVG with editable text.
"""
import argparse, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

MODEL_COLORS = {
    "gears": "#0279EE",
    "additive": "#E9ED4C",
    "meanshift": "#FF9400",
    "ridge": "#75A025",
    "ctrl_mean": "#888888",
}
MODEL_ORDER = ["gears", "additive", "meanshift", "ridge", "ctrl_mean"]
PRETTY = {"gears": "GEARS", "additive": "Additive", "meanshift": "Global mean-shift",
          "ridge": "Ridge (seen-gene)", "ctrl_mean": "Control-mean"}


def save(fig, figdir, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{figdir}/{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}.png / .svg", flush=True)


def fig1_overview(figdir):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    a = ax[0]
    a.bar(["cells", "genes", "conditions"], [78190, 8248, 7619],
          color=["#0279EE", "#75A025", "#FF9400"])
    for i, v in enumerate([78190, 8248, 7619]):
        a.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    a.set_title("K562 genome-scale CRISPRi (10 cells/pert)", fontsize=10)
    a.set_ylabel("count")
    b = ax[1]
    ss = {"train": 5143, "val": 572, "test": 1905}
    b.bar(list(ss.keys()), list(ss.values()), color=["#0279EE", "#75A025", "#FF9400"])
    for i, v in enumerate(ss.values()):
        b.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    b.set_title("GEARS simulation split (conditions)", fontsize=10)
    b.set_ylabel("conditions")
    c = ax[2]
    c.bar(["unseen_single"], [1905], color="#FF9400")
    c.text(0, 1905, "1,905", ha="center", va="bottom", fontsize=9)
    c.set_title("Held-out test regimes", fontsize=10)
    c.set_ylabel("conditions")
    fig.tight_layout()
    save(fig, figdir, "F1_dataset_overview")


def fig3_summary(data, figdir):
    models = [m for m in MODEL_ORDER if m in data]
    metrics_hi = [("pearson_delta_mean", "pearson_delta"),
                  ("pearson_delta_de_mean", "pearson_delta_de"),
                  ("dir20_mean", "dir_top20")]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    x = np.arange(len(metrics_hi)); w = 0.8 / len(models)
    for j, m in enumerate(models):
        vals = [data[m]["summary"][k] for k, _ in metrics_hi]
        ax[0].bar(x + (j - (len(models) - 1) / 2) * w, vals, w,
                  label=PRETTY[m], color=MODEL_COLORS[m])
        for xi, v in zip(x + (j - (len(models) - 1) / 2) * w, vals):
            ax[0].text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in metrics_hi], fontsize=9)
    ax[0].set_ylabel("score (higher better)")
    ax[0].set_title("Delta correlation & direction accuracy", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8, ncol=2)
    # mse_de, lower better
    ax[1].bar([PRETTY[m] for m in models], [data[m]["summary"]["mse_de_mean"] for m in models],
              color=[MODEL_COLORS[m] for m in models])
    for i, m in enumerate(models):
        ax[1].text(i, data[m]["summary"]["mse_de_mean"], f"{data[m]['summary']['mse_de_mean']:.3f}",
                   ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("MSE on top-20 DE genes (lower better)")
    ax[1].set_title("Reconstruction error (DE genes)", fontsize=11)
    ax[1].set_xticklabels([PRETTY[m] for m in models], rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    save(fig, figdir, "F3_benchmark_summary")


def fig4_distributions(data, figdir):
    models = [m for m in MODEL_ORDER if m in data and m != "ctrl_mean"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, key, title in zip(ax, ["pearson_delta", "pearson_delta_de"],
                             ["Pearson delta (all genes)", "Pearson delta (top-20 DE genes)"]):
        for m in models:
            vals = np.array([r[key] for r in data[m]["rows"] if r[key] == r[key]])
            a.hist(vals, bins=25, color=MODEL_COLORS[m], alpha=0.55,
                   label=PRETTY[m], edgecolor="none")
        a.axvline(0, color="#555", lw=0.8, ls=":")
        a.set_title(title, fontsize=10); a.set_xlabel(key); a.set_ylabel("# perturbations")
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, figdir, "F4_metric_distributions")


def fig6_examples(data, preds_dir, figdir, model="gears", named=None):
    pred = np.load(f"{preds_dir}/test_pred.npy")
    truth = np.load(f"{preds_dir}/test_truth.npy")
    if truth.ndim == 3:
        truth = truth[:, 0, :]
    pert = np.load(f"{preds_dir}/test_pertcat.npy", allow_pickle=True)
    de_idx = np.load(f"{preds_dir}/test_de_idx.npy")
    ctrl = np.load(f"{preds_dir}/ctrl_mean.npy")
    rows = {r["pert"]: r for r in data[model]["rows"]}
    uniq = [p for p in np.unique(pert) if p != "ctrl"]
    scored = sorted([(p, rows.get(p, {}).get("pearson_delta", np.nan)) for p in uniq
                     if rows.get(p, {}).get("pearson_delta", np.nan) == rows.get(p, {}).get("pearson_delta", np.nan)],
                    key=lambda x: x[1])
    worst, best = scored[0][0], scored[-1][0]
    med = scored[len(scored) // 2][0]
    chosen_named = named if (named and named in rows) else med
    picks = [("best", best), ("named", chosen_named), ("median", med), ("worst", worst)]
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    for a, (tag, p) in zip(ax.ravel(), picks):
        m = pert == p
        idx = de_idx[m][0] if de_idx.ndim == 2 else de_idx
        pm = pred[m].mean(0); tm = truth[m].mean(0)
        dx = tm - ctrl; dy = pm - ctrl
        r = rows.get(p, {}).get("pearson_delta", np.nan)
        a.scatter(dx, dy, s=6, alpha=0.35, color=MODEL_COLORS[model], edgecolor="none")
        lim = np.percentile(np.abs(np.concatenate([dx, dy])), 99.5)
        a.plot([-lim, lim], [-lim, lim], ls="--", color="#888", lw=1)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_xlabel("true delta (pert - ctrl)"); a.set_ylabel("predicted delta")
        a.set_title(f"{tag}: {p} | pearson_delta={r:.3f}", fontsize=10)
    fig.suptitle(f"{PRETTY[model]}: predicted vs true expression change", fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, figdir, "F6_example_scatters")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="comparison_rows pkl from compare_models.py")
    ap.add_argument("--gears_preds", default=None, help="GEARS preds dir (for F6)")
    ap.add_argument("--figdir", default="/mnt/results/figures")
    ap.add_argument("--named_example", default=None)
    ap.add_argument("--only", default="F1,F3,F4,F6")
    ap.add_argument("--official_jsons", nargs="*", default=[],
                    help="model=path to official metrics JSONs; overrides F3 bar values")
    args = ap.parse_args()
    import os
    os.makedirs(args.figdir, exist_ok=True)
    blob = pickle.load(open(args.pkl, "rb"))
    data = {k: {"rows": v, "summary": blob["summary"][k]} for k, v in blob["rows"].items()}
    # Override headline bars with official aggregate_metrics values where available
    KEYMAP = {"pearson_delta_mean": "pearson_delta_mean",
              "pearson_delta_de_mean": "pearson_delta_de_mean",
              "pearson_de_mean": "pearson_de_mean",
              "dir20_mean": "frac_correct_direction_20_mean",
              "mse_de_mean": "mse_de_mean"}
    for spec in args.official_jsons:
        model, path = spec.split("=", 1)
        if model in data:
            om = json.load(open(path))["overall_model"]
            for lk, ok in KEYMAP.items():
                if ok in om:
                    data[model]["summary"][lk] = float(om[ok])
            print(f"[official] {model} headline metrics overridden from {path}")
    want = set(args.only.split(","))
    if "F1" in want: fig1_overview(args.figdir)
    if "F3" in want: fig3_summary(data, args.figdir)
    if "F4" in want: fig4_distributions(data, args.figdir)
    if "F6" in want and args.gears_preds:
        fig6_examples(data, args.gears_preds, args.figdir, model="gears",
                      named=args.named_example)
    print("[done] figures ->", args.figdir)


if __name__ == "__main__":
    main()
