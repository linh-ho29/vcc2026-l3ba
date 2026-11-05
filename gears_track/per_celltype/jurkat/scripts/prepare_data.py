#!/usr/bin/env python
"""Track J prepare_data: Jurkat Nadig et al. CRISPRi -> GEARS format with
biology-informed, response-blind target sampling and leakage-safe splits.

Pipeline:
  1. Load scPerturb NadigOConner2024_jurkat.h5ad (DENSE raw counts -> chunked
     sparse conversion), normalize CP10k+log1p.
  2. Response-blind target metadata: GO functional category (rarest-anchor
     assignment over GO terms covering >=50 targets), GO-graph degree tercile,
     basal expression (control cells), essential-panel membership.
  3. Leakage-safe split: test (60) -> val (40) -> train (300), largest-remainder
     proportional allocation across category x degree strata. Never uses
     perturbation-effect magnitudes.
  4. Sample <=50 cells/target + 2,000 controls (seed 42).
  5. Convert to GEARS format via PertData.new_data_process (DE genes, pyg graphs).
  6. Write manifests + coverage report.
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

SEED = 42
N_TRAIN_TARGETS = 300
N_VAL_TARGETS = 40
N_TEST_TARGETS = 60
CELLS_PER_PERT = 50
N_CTRL = 2000
MIN_CELLS = 10
GO_ANCHOR_MIN_TARGETS = 50


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ram_gb():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
    return (info["MemTotal"] - info["MemAvailable"]) / 1e6


def go_degree(edge_df, genes):
    cnt = Counter(pd.concat([edge_df["source"], edge_df["target"]]))
    return {g: int(cnt.get(g, 0)) for g in genes}


def assign_go_categories(genes, gene2go, min_targets=GO_ANCHOR_MIN_TARGETS):
    """Functional category per gene. Overlapping GO terms are handled by
    anchoring: terms covering >= min_targets targets become anchors, and each
    gene is assigned to its RAREST anchor term (most specific). Genes with no
    anchor term -> 'other'. Full multi-membership is preserved in the manifest."""
    anchor_counts = Counter()
    for g in genes:
        for t in gene2go.get(g, []):
            anchor_counts[t] += 1
    anchors = {t for t, c in anchor_counts.items() if c >= min_targets}
    cats, memberships = {}, {}
    for g in genes:
        terms = [t for t in gene2go.get(g, []) if t in anchors]
        memberships[g] = sorted(terms)
        if not terms:
            cats[g] = "other"
        else:
            cats[g] = min(terms, key=lambda t: (anchor_counts[t], t))
    return cats, memberships, anchor_counts


def alloc_proportional(targets, strata_key, n, chosen, rng):
    """Largest-remainder proportional allocation of n genes across strata,
    drawing only from genes not already chosen."""
    free = targets[~targets.gene.isin(chosen)]
    sizes = free.groupby(strata_key, observed=True).size()
    total = sizes.sum()
    quotas = sizes * n / total
    base = quotas.astype(int)
    rem = n - base.sum()
    for s in (quotas - base).sort_values(ascending=False).index[:max(rem, 0)]:
        base[s] += 1
    picked = []
    for s, k in base.items():
        avail = sorted(set(free[free[strata_key] == s].gene) - chosen)
        if not avail:
            continue
        pick = rng.choice(avail, min(k, len(avail)), replace=False)
        picked += list(pick)
        chosen.update(pick)
    while len(picked) < n:  # top up if some strata were exhausted
        avail = sorted(set(targets.gene) - chosen)
        if not avail:
            break
        pick = rng.choice(avail, min(n - len(picked), len(avail)), replace=False)
        picked += list(pick)
        chosen.update(pick)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="/workspace/jurkat/NadigOConner2024_jurkat.h5ad")
    ap.add_argument("--gene2go", default="/workspace/data/gene2go_all.pkl")
    ap.add_argument("--go_csv", default="/workspace/data/go_essential_all/go_essential_all.csv")
    ap.add_argument("--out_dir", default="/workspace/jurkat/gears_data")
    ap.add_argument("--results_dir", default="/workspace/jurkat/manifests")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ---- 1. Load + normalize (chunked: source X is DENSE, ~9-19 GB if read whole)
    a = ad.read_h5ad(args.h5ad, backed="r")
    full_shape = list(a.shape)
    n_batches = int(a.obs["batch"].nunique())
    obs_df = a.obs.copy()
    var_df = a.var.copy()
    print(f"[load] {full_shape} in {time.time()-t0:.0f}s", flush=True)
    chunks = []
    step = 20000
    for start in range(0, a.n_obs, step):
        Xc = a.X[start:start + step]
        chunks.append(sp.csr_matrix(Xc, dtype=np.float32))
    X = sp.vstack(chunks).tocsr()
    del chunks
    a = ad.AnnData(X=X, obs=obs_df, var=var_df)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    print(f"[norm] done, ram={ram_gb():.1f} GB", flush=True)

    ctrl_mask = (a.obs["perturbation"] == "control").values
    ctrl_mean = np.asarray(a.X[ctrl_mask].mean(axis=0)).ravel()
    gene2idx = {g: i for i, g in enumerate(a.var_names)}

    # ---- 2. Response-blind target metadata ----------------------------------
    gene2go = pickle.load(open(args.gene2go, "rb"))
    edge_df = pd.read_csv(args.go_csv)
    targets = (a.obs["perturbation"].value_counts()
                 .drop("control").rename_axis("gene").reset_index(name="n_cells"))
    targets = targets[targets.n_cells >= MIN_CELLS].reset_index(drop=True)
    n_all = len(targets)
    targets = targets[targets.gene.isin(set(a.var_names))].reset_index(drop=True)
    n_meas = len(targets)
    # GEARS PertData.load silently drops perturbations outside its GO/essential
    # gene universe (gene2go ∩ essential_all_data_pert_genes), which breaks the
    # loss filter — exclude them here so every sampled target is modelable.
    ess_path = os.path.join(args.out_dir, "essential_all_data_pert_genes.pkl")
    if not os.path.exists(ess_path):
        # same source GEARS uses in PertData.load (default_pert_graph branch)
        from gears.utils import dataverse_download
        dataverse_download("https://dataverse.harvard.edu/api/access/datafile/6934320",
                           ess_path)
    ess = pickle.load(open(ess_path, "rb"))
    universe = set(ess) & set(gene2go)
    targets = targets[targets.gene.isin(universe)].reset_index(drop=True)
    print(f"[targets] {len(targets)} eligible (>= {MIN_CELLS} cells, in measured "
          f"panel, in GEARS GO universe; {n_all - len(targets)} dropped: "
          f"{n_all - n_meas} unmeasured, {n_meas - len(targets)} outside GO universe)",
          flush=True)

    deg = go_degree(edge_df, targets.gene.tolist())
    targets["go_degree"] = targets.gene.map(deg)
    targets["deg_bin"] = pd.qcut(targets.go_degree.rank(method="first"), 3,
                                 labels=["deg_low", "deg_mid", "deg_high"])
    cats, memberships, anchor_counts = assign_go_categories(targets.gene.tolist(), gene2go)
    targets["go_category"] = targets.gene.map(cats)
    targets["basal_expr"] = [float(ctrl_mean[gene2idx[g]]) if g in gene2idx else np.nan
                             for g in targets.gene]
    targets["basal_bin"] = "basal_na"
    meas = targets.basal_expr.notna().values
    if meas.sum() >= 10:
        bins = pd.qcut(targets.loc[meas, "basal_expr"].rank(method="first"), 3,
                       labels=["basal_low", "basal_mid", "basal_high"])
        targets.loc[meas, "basal_bin"] = bins.astype(str).values
    targets["n_go_terms"] = targets.gene.map(lambda g: len(gene2go.get(g, [])))
    targets["essential_panel"] = True  # all targets are essential-panel genes by design
    targets["stratum"] = targets.go_category.astype(str) + "|" + targets.deg_bin.astype(str)

    # ---- 3. Leakage-safe splits ---------------------------------------------
    rng = np.random.default_rng(SEED)
    chosen = set()
    test_t = alloc_proportional(targets, "stratum", N_TEST_TARGETS, chosen, rng)
    val_t = alloc_proportional(targets, "stratum", N_VAL_TARGETS, chosen, rng)
    train_t = alloc_proportional(targets, "stratum", N_TRAIN_TARGETS, chosen, rng)
    assert not (set(test_t) & set(val_t)) and not (set(test_t) & set(train_t))
    print(f"[split] test={len(test_t)} val={len(val_t)} train={len(train_t)}", flush=True)

    # ---- 4. Cell sampling -----------------------------------------------------
    obs_pert = a.obs["perturbation"].values
    keep_idx, sampling_rows = [], []
    for split, genes in [("train", train_t), ("val", val_t), ("test", test_t)]:
        for g in genes:
            sel = np.where(obs_pert == g)[0]
            n_take = min(CELLS_PER_PERT, len(sel))
            take = rng.choice(sel, n_take, replace=False) if len(sel) > n_take else sel
            keep_idx.append(take)
            sampling_rows.append({"gene": g, "split": split,
                                  "n_cells_available": int(len(sel)),
                                  "n_cells_sampled": int(n_take)})
    ctrl_idx = np.where(ctrl_mask)[0]
    keep_idx.append(rng.choice(ctrl_idx, N_CTRL, replace=False))
    sampling_rows.append({"gene": "control", "split": "control",
                          "n_cells_available": int(ctrl_mask.sum()),
                          "n_cells_sampled": N_CTRL})
    keep = np.sort(np.concatenate(keep_idx))
    sub = a[keep].copy()
    del a
    import gc
    gc.collect()
    print(f"[sample] working set: {sub.n_obs} cells "
          f"({sub.n_obs - N_CTRL} perturbed + {N_CTRL} ctrl), ram={ram_gb():.1f} GB", flush=True)

    # ---- 5. GEARS conversion --------------------------------------------------
    cond = sub.obs["perturbation"].astype(str).values
    sub.obs["condition"] = np.array(["ctrl" if c == "control" else c + "+ctrl" for c in cond])
    sub.obs["cell_type"] = "Jurkat"
    sub.var["gene_name"] = sub.var_names.astype(str)
    sub.X = sp.csr_matrix(sub.X, dtype=np.float32)

    from gears.pertdata import PertData
    pd_ = PertData(data_path=args.out_dir)
    t_de = time.time()
    pd_.new_data_process("nadig25jurkat", sub)  # DE genes + pyg graphs (cached)
    de_time = time.time() - t_de
    print(f"[gears] processed data + DE in {de_time:.0f}s", flush=True)

    # ---- 6. Manifests + coverage report ---------------------------------------
    split_manifest = {
        "seed": SEED,
        "procedure": "test targets drawn first, then val, then train; "
                     "largest-remainder proportional allocation across "
                     "GO-category x GO-degree-tercile strata; response-blind "
                     "(no effect-size features used anywhere)",
        "n_test_targets": len(test_t), "n_val_targets": len(val_t),
        "n_train_targets": len(train_t),
        "test_targets": sorted(test_t),
        "val_targets": sorted(val_t),
        "train_targets": sorted(train_t),
    }
    sampling_manifest = {
        "seed": SEED,
        "strategy": "biology-informed balance: proportional allocation across "
                    "GO-category x degree strata; <=50 cells/target; "
                    "2000 non-targeting controls",
        "effect_blindness": "no perturbation-effect magnitude, DE, or fold-change "
                            "feature was used at any sampling step",
        "go_overlap_handling": f"GO terms covering >= {GO_ANCHOR_MIN_TARGETS} eligible "
                               "targets are anchors; each gene assigned to its rarest "
                               "anchor (most specific); full multi-membership recorded",
        "go_anchor_counts": {t: int(c) for t, c in anchor_counts.most_common()},
        "per_target": sampling_rows,
    }
    universe = targets.set_index("gene")
    cov_rows = []
    for g in universe.index:
        split = ("test" if g in test_t else
                 "val" if g in val_t else
                 "train" if g in train_t else "unsampled")
        cov_rows.append({"gene": g, "split": split,
                         "go_category": universe.at[g, "go_category"],
                         "go_degree": int(universe.at[g, "go_degree"]),
                         "deg_bin": str(universe.at[g, "deg_bin"]),
                         "basal_bin": str(universe.at[g, "basal_bin"]),
                         "n_cells": int(universe.at[g, "n_cells"])})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(os.path.join(args.results_dir, "pathway_coverage_report.csv"), index=False)

    rep = []
    for c, grp in cov.groupby("go_category"):
        f_universe = len(grp) / len(cov)
        f_train = (grp.split == "train").mean()
        rep.append({"go_category": c, "n_universe": len(grp),
                    "universe_frac": round(f_universe, 4),
                    "train_frac": round(f_train, 4),
                    "train_minus_universe": round(f_train - f_universe, 4)})
    pd.DataFrame(rep).sort_values("train_minus_universe").to_csv(
        os.path.join(args.results_dir, "train_vs_universe_representation.csv"), index=False)

    dataset_manifest = {
        "source_h5ad": args.h5ad,
        "source_md5": md5sum(args.h5ad),
        "preferred_source_status": "scperteval GCS bucket is requester-pays; used the "
                                   "scPerturb Zenodo distribution of the same Nadig et al. "
                                   "Jurkat screen (record 13350497)",
        "shape_full": full_shape,
        "normalization": "normalize_total(1e4) + log1p (source X was raw integer counts)",
        "verified": {"raw_counts": True, "gene_identifiers": "ensembl_id + var_names",
                     "single_gene_labels": "obs.perturbation",
                     "controls": "obs.perturbation=='control'",
                     "go_mappings": "GEARS gene2go_all.pkl + go_essential_all graph",
                     "target_coverage": f"{len(targets)} targets with >= {MIN_CELLS} cells",
                     "sample_batch_metadata": f"obs.batch ({n_batches} batches)"},
        "gears_dataset_dir": os.path.join(args.out_dir, "nadig25jurkat"),
        "de_calc_time_s": round(de_time, 1),
        "prep_time_s": round(time.time() - t0, 1),
    }
    for name, obj in [("split_manifest.json", split_manifest),
                      ("sampling_manifest.json", sampling_manifest),
                      ("dataset_manifest.json", dataset_manifest)]:
        with open(os.path.join(args.results_dir, name), "w") as f:
            json.dump(obj, f, indent=2)
    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
