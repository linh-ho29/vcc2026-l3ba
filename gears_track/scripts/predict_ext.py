#!/usr/bin/env python
"""
Custom linear baselines for the K562 gwps benchmark, extending the skill's
predict.py adapter interface. Emits the SAME output contract as predict.py:
  test_pred.npy, test_truth.npy, test_pertcat.npy, test_de_idx.npy,
  ctrl_mean.npy, meta.json

Models (--model):
  additive   pred = ctrl_mean + per-gene mean delta from TRAINING perturbations;
             unseen genes get delta = 0 (falls back to control mean).
  ridge      multi-output ridge: features = control co-expression profile of the
             targeted gene (n_genes-dim Pearson-corr vector), target = expression
             delta; generalizes to unseen genes. Alpha by inner validation.
  meanshift  pred = ctrl_mean + global mean delta over training perturbations
             (one shared CRISPRi-response shift for every perturbation).
"""
import argparse, json, os, time, warnings, shutil
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")

from predict import collect_truth_and_deidx


def load_pertdata(args):
    from gears import PertData
    pd = PertData(args.data_dir)
    if args.data_path:
        pd.load(data_path=args.data_path)
    else:
        pd.load(data_name=args.dataset)
    pd.prepare_split(split=args.split, seed=args.split_seed)
    pd.get_dataloader(batch_size=args.batch_size, test_batch_size=args.batch_size)
    return pd


# ---------------------------------------------------------------- shared utils
def get_split_conditions(pd):
    s2c = pd.set2conditions
    train_perts = sorted(s2c.get("train", []))
    test_perts = sorted(s2c.get("test", []))
    return train_perts, test_perts


def gene_of(label):
    """'CDK1+ctrl' -> 'CDK1' (single-gene dataset)."""
    gs = [g for g in label.split("+") if g != "ctrl"]
    return gs[0] if gs else None


def condition_means(pd, conditions, batch=500):
    """Mean expression per condition from pd.adata (sparse-safe). Returns dict."""
    adata = pd.adata
    conds = np.asarray(adata.obs["condition"].values)
    X = adata.X
    out = {}
    for i in range(0, len(conditions), batch):
        chunk = conditions[i:i + batch]
        for c in chunk:
            m = conds == c
            out[c] = np.asarray(X[m].mean(axis=0)).ravel().astype(np.float32)
    return out


def training_deltas(pd, ctrl_mean, log_prefix=""):
    """Per-gene mean delta (vs ctrl) over TRAINING perturbations."""
    t0 = time.time()
    train_perts, _ = get_split_conditions(pd)
    train_perts = [c for c in train_perts if gene_of(c)]
    means = condition_means(pd, train_perts)
    delta_by_gene = {}
    for c, m in means.items():
        delta_by_gene[gene_of(c)] = m - ctrl_mean
    print(f"[{log_prefix}] {len(delta_by_gene)} training gene deltas "
          f"({time.time()-t0:.0f}s)", flush=True)
    return delta_by_gene


# ------------------------------------------------------------------- adapters
def predict_additive(pd, args, device):
    """pred = ctrl + delta_g for seen genes; ctrl + 0 for unseen genes."""
    adata = pd.adata
    genes = adata.var["gene_name"].tolist(); n_genes = len(genes)
    ctrl_idx = np.where(adata.obs["condition"].values == "ctrl")[0]
    ctrl_mean = np.asarray(adata.X[ctrl_idx].mean(axis=0)).ravel().astype(np.float32)
    delta_by_gene = training_deltas(pd, ctrl_mean, "additive")

    pert_cat, truth, de_idx_all = collect_truth_and_deidx(pd.dataloader["test_loader"])
    pred = np.repeat(ctrl_mean[None, :], truth.shape[0], axis=0).astype(np.float32)
    seen, unseen = 0, 0
    for i, lbl in enumerate(pert_cat):
        g = gene_of(lbl)
        if g is not None and g in delta_by_gene:
            pred[i] += delta_by_gene[g]; seen += 1
        else:
            unseen += 1
    print(f"[additive] seen-gene cells: {seen}, unseen-gene cells: {unseen}", flush=True)
    return pert_cat, pred, truth, de_idx_all, genes, n_genes


def control_coexpression(pd, ctrl_mean):
    """Pearson correlation of each gene's control expression with all genes,
    computed across control cells (n_genes x n_genes feature matrix)."""
    adata = pd.adata
    ctrl_idx = np.where(adata.obs["condition"].values == "ctrl")[0]
    Xc = adata.X[ctrl_idx]
    Xc = np.asarray(Xc.todense() if hasattr(Xc, "todense") else Xc, dtype=np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, keepdims=True); sd[sd == 0] = 1e-8
    Xc = Xc / sd
    C = (Xc.T @ Xc) / Xc.shape[0]
    np.fill_diagonal(C, 1.0)
    print(f"[ridge] co-expression features {C.shape} (mean |r|={np.abs(C).mean():.3f})",
          flush=True)
    return C.astype(np.float32)


def predict_ridge(pd, args, device):
    """Ridge: delta_g ~ W @ coexpr_profile(g). Fit on training perts."""
    from sklearn.linear_model import Ridge
    adata = pd.adata
    genes = adata.var["gene_name"].tolist(); n_genes = len(genes)
    ctrl_idx = np.where(adata.obs["condition"].values == "ctrl")[0]
    ctrl_mean = np.asarray(adata.X[ctrl_idx].mean(axis=0)).ravel().astype(np.float32)

    delta_by_gene = training_deltas(pd, ctrl_mean, "ridge")
    train_genes = sorted(delta_by_gene)
    C = control_coexpression(pd, ctrl_mean)
    g2i = {g: i for i, g in enumerate(genes)}
    F = np.stack([C[g2i[g]] for g in train_genes])          # (n_train, n_genes)
    Y = np.stack([delta_by_gene[g] for g in train_genes])   # (n_train, n_genes)

    # inner validation: hold out 10% of training genes, pick alpha by pearson_delta
    rng = np.random.RandomState(args.split_seed)
    perm = rng.permutation(len(train_genes))
    n_val = max(1, len(train_genes) // 10)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    Fv, Yv = F[val_idx], Y[val_idx]
    Ff, Yf = F[fit_idx], Y[fit_idx]
    # center features/targets on the fit set; predict with stored offsets
    mu_f, mu_y = Ff.mean(0), Yf.mean(0)
    best = (None, -np.inf)
    for alpha in args.ridge_alphas:
        r = Ridge(alpha=alpha, solver="lsqr", fit_intercept=False)
        r.fit(Ff - mu_f, Yf - mu_y)
        Pv = r.predict(Fv - mu_f) + mu_y
        d_true = Yv - Yv.mean(0, keepdims=True)
        d_pred = Pv - Pv.mean(0, keepdims=True)
        num = (d_true * d_pred).sum(1)
        den = np.linalg.norm(d_true, axis=1) * np.linalg.norm(d_pred, axis=1) + 1e-12
        r_delta = float(np.mean(num / den))
        print(f"[ridge] alpha={alpha}: inner val pearson_delta={r_delta:.4f}", flush=True)
        if r_delta > best[1]:
            best = (alpha, r_delta)
    alpha = best[0]
    print(f"[ridge] selected alpha={alpha}", flush=True)

    model = Ridge(alpha=alpha, solver="lsqr", fit_intercept=False)
    model.fit(F - F.mean(0), Y - Y.mean(0))  # refit on all training genes
    mu_f_all, mu_y_all = F.mean(0), Y.mean(0)

    pert_cat, truth, de_idx_all = collect_truth_and_deidx(pd.dataloader["test_loader"])
    test_genes = [gene_of(l) for l in pert_cat]
    Ft = np.stack([C[g2i[g]] if g in g2i else np.zeros(n_genes, dtype=np.float32)
                   for g in test_genes])
    delta = model.predict(Ft - mu_f_all) + mu_y_all
    pred = np.repeat(ctrl_mean[None, :], truth.shape[0], axis=0) + delta
    print(f"[ridge] predictions done for {truth.shape[0]} cells", flush=True)
    return pert_cat, pred.astype(np.float32), truth, de_idx_all, genes, n_genes


def predict_meanshift(pd, args, device):
    """pred = ctrl + global mean training delta (same shift for every perturbation)."""
    adata = pd.adata
    genes = adata.var["gene_name"].tolist(); n_genes = len(genes)
    ctrl_idx = np.where(adata.obs["condition"].values == "ctrl")[0]
    ctrl_mean = np.asarray(adata.X[ctrl_idx].mean(axis=0)).ravel().astype(np.float32)
    delta_by_gene = training_deltas(pd, ctrl_mean, "meanshift")
    global_delta = np.stack(list(delta_by_gene.values())).mean(axis=0).astype(np.float32)
    print(f"[meanshift] global delta: ||d||={np.linalg.norm(global_delta):.2f}, "
          f"top genes: {[genes[i] for i in np.argsort(-np.abs(global_delta))[:5]]}", flush=True)

    pert_cat, truth, de_idx_all = collect_truth_and_deidx(pd.dataloader["test_loader"])
    pred = np.repeat((ctrl_mean + global_delta)[None, :], truth.shape[0], axis=0)
    return pert_cat, pred.astype(np.float32), truth, de_idx_all, genes, n_genes


ADAPTERS = {"additive": predict_additive, "ridge": predict_ridge,
            "meanshift": predict_meanshift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(ADAPTERS), required=True)
    ap.add_argument("--dataset", default="custom")
    ap.add_argument("--split", default="simulation")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--data_dir", default="/workspace/data")
    ap.add_argument("--data_path", default=None,
                    help="GEARS-processed data path (PertData.load(data_path=...))")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ridge_alphas", type=float, nargs="+",
                    default=[0.1, 1.0, 10.0, 100.0])
    args = ap.parse_args()

    device = torch.device("cpu")
    local = "/workspace/preds"; os.makedirs(local, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[data] {args.dataset} + {args.split} split (seed {args.split_seed}) "
          f"| model={args.model}", flush=True)
    pd = load_pertdata(args)
    adata = pd.adata

    pert_cat, pred, truth, de_idx_all, genes, vocab_match = ADAPTERS[args.model](pd, args, device)

    print(f"[pred] {pred.shape}. SAVING NOW (before metrics).", flush=True)
    np.save(f"{local}/test_pred.npy", pred)
    np.save(f"{local}/test_truth.npy", truth)
    np.save(f"{local}/test_pertcat.npy", pert_cat)
    np.save(f"{local}/test_de_idx.npy", de_idx_all)
    ctrl_idx = np.where(adata.obs["condition"].values == "ctrl")[0]
    ctrl_mean = np.asarray(adata.X[ctrl_idx].mean(axis=0)).ravel().astype(np.float32)
    np.save(f"{local}/ctrl_mean.npy", ctrl_mean)
    json.dump({
        "dataset": args.dataset, "model": args.model, "split": args.split,
        "seed": args.split_seed, "n_genes": len(genes), "genes": genes,
        "vocab_match": vocab_match, "subgroup": pd.subgroup,
        "set2conditions": {k: list(v) for k, v in pd.set2conditions.items()},
    }, open(f"{local}/meta.json", "w"))
    for f in ["test_pred.npy", "test_truth.npy", "test_pertcat.npy",
              "test_de_idx.npy", "ctrl_mean.npy", "meta.json"]:
        shutil.copy(f"{local}/{f}", f"{args.out_dir}/{f}")
    print(f"[save] all arrays -> {args.out_dir}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
