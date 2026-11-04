#!/usr/bin/env python
"""
Fast replacement for GEARS's get_go_auto (gears/utils.py): builds the same
go.csv (GO-similarity Jaccard edges, score > 0.1) via a vectorized binary
matrix product instead of an O(n^2) nested np.intersect1d loop.

Run once per dataset folder; GEARS then loads go.csv from cache unchanged.
"""
import argparse, os, pickle, time
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/workspace/data")
    ap.add_argument("--data_name", required=True)
    ap.add_argument("--gene_list", default=None,
                    help="optional file with one gene name per line; "
                         "default: genes of <data_path>/<data_name>/perturb_processed.h5ad")
    args = ap.parse_args()

    t0 = time.time()
    ds_dir = os.path.join(args.data_path, args.data_name)
    go_path = os.path.join(ds_dir, "go.csv")
    if os.path.exists(go_path):
        print(f"[go] {go_path} already exists; nothing to do")
        return

    g2g_path = os.path.join(args.data_path, "gene2go.pkl")
    if not os.path.exists(g2g_path):
        from gears.utils import dataverse_download
        dataverse_download("https://dataverse.harvard.edu/api/access/datafile/6153417",
                           g2g_path)
    gene2go = pickle.load(open(g2g_path, "rb"))

    if args.gene_list:
        gene_list = [l.strip() for l in open(args.gene_list) if l.strip()]
    else:
        import anndata as ad
        gene_list = ad.read_h5ad(os.path.join(ds_dir, "perturb_processed.h5ad"),
                                 backed="r").var["gene_name"].tolist()

    genes = [g for g in gene_list if g in gene2go]
    print(f"[go] {len(genes)}/{len(gene_list)} genes with GO annotations", flush=True)

    terms = sorted({t for g in genes for t in gene2go[g]})
    term2i = {t: i for i, t in enumerate(terms)}
    g2i = {g: i for i, g in enumerate(genes)}
    B = np.zeros((len(genes), len(terms)), dtype=np.float32)
    for g, ts in gene2go.items():
        if g in g2i:
            B[g2i[g], [term2i[t] for t in ts]] = 1.0
    sizes = B.sum(axis=1)

    # Jaccard in row blocks: J = inter / (|g1| + |g2| - inter)
    n = len(genes)
    J = np.empty((n, n), dtype=np.float32)
    block = 512
    for s in range(0, n, block):
        e = min(s + block, n)
        inter = B[s:e] @ B.T
        union = sizes[s:e, None] + sizes[None, :] - inter
        J[s:e] = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    np.fill_diagonal(J, 1.0)  # GEARS's nested loop includes self-pairs (J=1)

    src, tgt = np.nonzero(J > 0.1)
    df = pd.DataFrame({"source": [genes[i] for i in src],
                       "target": [genes[i] for i in tgt],
                       "importance": J[src, tgt]})
    df.to_csv(go_path, index=False)
    print(f"[go] {len(df)} edges (score>0.1) -> {go_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
