#!/usr/bin/env python
"""
Convert the Replogle 2022 K562 gwps single-cell h5ad (dense, ~2M cells x 8248
genes, z-normalized per gemgroup) into a GEARS-format AnnData without ever
loading the full matrix into memory.

- obs['gene_transcript'] = '<idx>_<SYMBOL>_<construct>_<ENSG>' (or non-targeting)
  -> gene-level condition labels: 'SYMBOL+ctrl', controls -> 'ctrl'
- perturbations whose target gene is not in the expression panel are DROPPED
  (GEARS resolves perturbation targets by var gene_name lookup and crashes otherwise)
- subsample (seed 42): up to --cells_per_pert cells per perturbation,
  --n_ctrl control cells; drop conditions with < --min_cells cells
"""
import argparse, json, time
import h5py
import numpy as np
import pandas as pd
import anndata as ad


def read_obs_col(f, name):
    g = f["obs"][name]
    if isinstance(g, h5py.Group):  # categorical: codes + categories
        cats = g["categories"][:]
        codes = g["codes"][:]
        return np.array([cats[c] if c >= 0 else b"NA" for c in codes])
    return g[:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="/workspace/data/K562_gwps_normalized_singlecell_01.h5ad")
    ap.add_argument("--out", default="/workspace/data/gwps_k562_gears.h5ad")
    ap.add_argument("--cells_per_pert", type=int, default=40)
    ap.add_argument("--n_ctrl", type=int, default=10000)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--max_perts", type=int, default=0,
                    help="keep only the first N perturbations (0 = all; for smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.RandomState(args.seed)
    f = h5py.File(args.h5ad, "r")
    n_cells, n_genes_total = f["X"].shape
    print(f"[open] {n_cells} cells x {n_genes_total} genes", flush=True)

    # ---- var: gene names (categorical decode)
    gn_cat = f["var"]["__categories"]["gene_name"][:]
    gn_codes = f["var"]["gene_name"][:]
    gene_names = np.array([gn_cat[c].decode() for c in gn_codes])
    gid = f["var"]["gene_id"][:]
    gene_ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in gid])
    gene_name_set = set(gene_names.tolist())

    # ---- obs: perturbation labels -> gene-level conditions
    # single-cell file stores categoricals as obs-level __categories + integer codes
    gcats = f["obs"]["__categories"]["gene"][:]
    gcodes = f["obs"]["gene"][:]
    labels = np.array([gcats[c].decode() for c in gcodes])
    is_ctrl = labels == "non-targeting"
    conds = np.where(is_ctrl, "ctrl", ["%s+ctrl" % x for x in labels])
    print(f"[labels] {int(is_ctrl.sum())} ctrl cells, {int((~is_ctrl).sum())} "
          f"perturbed cells ({time.time()-t0:.0f}s)", flush=True)

    # ---- cells per condition
    uniq, counts = np.unique(conds, return_counts=True)
    count_map = dict(zip(uniq.tolist(), counts.tolist()))
    keep_conds = {c: n for c, n in count_map.items() if n >= args.min_cells}
    pert_conds = [c for c in sorted(keep_conds.keys()) if c != "ctrl"]
    if args.max_perts:
        pert_conds = pert_conds[:args.max_perts]
    kept_perts = [c for c in pert_conds if c.split("+")[0] in gene_name_set]
    dropped = [c for c in pert_conds if c.split("+")[0] not in gene_name_set]
    print(f"[conds] {len(pert_conds)} perturbations with >= {args.min_cells} cells; "
          f"dropping {len(dropped)} whose target gene is not in the expression panel",
          flush=True)
    if dropped:
        print(f"  dropped examples: {sorted(dropped)[:5]}", flush=True)
    keep_conds = {c: keep_conds[c] for c in kept_perts + ["ctrl"] if c in count_map}
    n_perts = len(kept_perts)

    # ---- sample cell indices per condition
    order = np.argsort(conds, kind="stable")
    sorted_conds = conds[order]
    keys = sorted(keep_conds.keys())
    picked, per_cond_final = [], {}
    for c in keys:
        lo = np.searchsorted(sorted_conds, c)
        hi = np.searchsorted(sorted_conds, c, side="right")
        idx = order[lo:hi]
        cap = args.n_ctrl if c == "ctrl" else args.cells_per_pert
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        picked.append(idx)
        per_cond_final[c] = len(idx)
    picked = np.sort(np.concatenate(picked))
    n_ctrl_final = per_cond_final.get("ctrl", 0)
    print(f"[sample] {len(picked)} cells ({n_perts} perts x <= {args.cells_per_pert}, "
          f"{n_ctrl_final} ctrl) ({time.time()-t0:.0f}s)", flush=True)

    # ---- read X rows in chunks
    X = np.empty((len(picked), n_genes_total), dtype=np.float32)
    n_nonfinite = 0
    chunk = 20000
    for s in range(0, len(picked), chunk):
        idx = picked[s:s + chunk]
        X[s:s + len(idx)] = f["X"][idx, :]
        if (s // chunk) % 10 == 0:
            print(f"  read {s + len(idx)}/{len(picked)} rows ({time.time()-t0:.0f}s)", flush=True)
    n_nonfinite = int((~np.isfinite(X)).sum())
    X[~np.isfinite(X)] = 0.0  # z-score of zero-variance gemgroup genes -> inf; impute 0 (= mean)
    print(f"[sanitize] replaced {n_nonfinite} non-finite values with 0", flush=True)
    f.close()

    # ---- assemble AnnData (var index made unique; GEARS requires sparse X)
    import scipy.sparse as sp
    obs_df = pd.DataFrame(
        {"condition": conds[picked], "cell_type": "K562"},
        index=[f"cell_{i}" for i in range(len(picked))],
    )
    var_df = pd.DataFrame({"gene_name": gene_names, "gene_id": gene_ids},
                          index=gene_names.tolist())
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs_df, var=var_df)
    adata.var_names_make_unique()
    adata.obs["condition"] = adata.obs["condition"].astype("category")
    print(f"[adata] {adata}", flush=True)

    adata.write_h5ad(args.out)
    stats = {
        "source": args.h5ad, "n_cells_source": int(n_cells), "n_genes": int(n_genes_total),
        "n_cells_sampled": int(len(picked)), "n_ctrl": int(n_ctrl_final),
        "n_perturbations": int(n_perts),
        "n_perturbations_dropped_gene_not_in_panel": int(len(dropped)),
        "cells_per_pert_cap": args.cells_per_pert,
        "min_cells": args.min_cells, "seed": args.seed,
    }
    json.dump(stats, open(args.out.replace(".h5ad", "_stats.json"), "w"), indent=2)
    print(f"[save] {args.out} ({time.time()-t0:.0f}s total)", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
