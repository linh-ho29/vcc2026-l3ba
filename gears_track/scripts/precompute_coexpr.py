#!/usr/bin/env python
"""
Streaming precompute of GEARS's co-expression network cache, replicating
gears.utils.get_coexpression_network_from_train exactly but in row blocks
(GEARS materializes all training cells as a dense matrix -> OOM at this scale).

Reads the GEARS-processed h5ad via h5py (CSR components), accumulates the
gene-gene gram matrix over TRAINING-split cells, then writes the same
co_expression_network.csv cache file GEARS expects (top k+1 neighbors per
gene by |pearson r|, kept if |r| > threshold).
"""
import argparse, glob, os, pickle, time
import h5py
import numpy as np
import pandas as pd


def read_condition(f):
    codes = f["obs"]["condition"][:]
    cats = f["obs"]["__categories"]["condition"][:]
    return np.array([cats[c].decode() for c in codes])


def iter_csr_blocks(f, n_cells, block=20000):
    data = f["X"]["data"]
    indices = f["X"]["indices"]
    indptr = f["X"]["indptr"]
    for s in range(0, n_cells, block):
        e = min(s + block, n_cells)
        lo, hi = indptr[s], indptr[e]
        n_rows = e - s
        d = data[lo:hi]
        c = indices[lo:hi]
        ptr = (indptr[s:e] - lo)
        from scipy.sparse import csr_matrix
        yield csr_matrix((d, c, ptr), shape=(n_rows, f["X"]["indptr"].shape[0] - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/workspace/data")
    ap.add_argument("--data_name", required=True)
    ap.add_argument("--split", default="simulation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_gene_set_size", type=float, default=0.75)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    ds_dir = os.path.join(args.data_path, args.data_name)
    fname = os.path.join(
        ds_dir, f"{args.split}_{args.seed}_{args.train_gene_set_size}_"
                f"{args.threshold}_{args.k}_co_expression_network.csv")
    if os.path.exists(fname):
        print(f"[coexpr] cache exists: {fname}")
        return

    # train conditions from the GEARS split pickle
    split_files = glob.glob(os.path.join(ds_dir, "splits",
        f"{args.split}_{args.seed}_{args.train_gene_set_size}*.pkl"))
    assert split_files, f"no split pickle found in {ds_dir}/splits"
    split = pickle.load(open(split_files[0], "rb"))
    train_conds = set(split["train"] if isinstance(split, dict) else split)
    print(f"[coexpr] {len(train_conds)} train conditions", flush=True)

    h5ad = os.path.join(ds_dir, "perturb_processed.h5ad")
    with h5py.File(h5ad, "r") as f:
        n_cells = f["X"]["indptr"].shape[0] - 1
        conds = read_condition(f)
        gene_names = [c.decode() for c in f["var"]["__categories"]["gene_name"][:]]
        g = len(gene_names)
        mask = np.isin(conds, list(train_conds))
        n_train = int(mask.sum())
        print(f"[coexpr] {n_cells} cells, {n_train} train cells, {g} genes", flush=True)

        S = np.zeros((g, g), dtype=np.float64)
        sums = np.zeros(g, dtype=np.float64)
        sumsq = np.zeros(g, dtype=np.float64)
        done = 0
        for blk in iter_csr_blocks(f, n_cells):
            b_mask = mask[done:done + blk.shape[0]]
            B = blk[b_mask].toarray().astype(np.float64)
            S += B.T @ B
            sums += B.sum(0)
            sumsq += (B ** 2).sum(0)
            done += blk.shape[0]
            print(f"  processed {done}/{n_cells} cells ({time.time()-t0:.0f}s)", flush=True)

    n = n_train
    mean = sums / n
    var = sumsq / n - mean ** 2
    sd = np.sqrt(np.maximum(var, 1e-12))
    C = S - n * np.outer(mean, mean)
    denom = n * np.outer(sd, sd)
    r = np.divide(C, denom, out=np.zeros_like(C), where=denom > 0)
    r = np.abs(r)
    r[np.isnan(r)] = 0.0

    k1 = args.k + 1
    idx = np.argsort(r, axis=1)[:, -k1:]
    val = np.sort(r, axis=1)[:, -k1:]
    src = np.repeat(np.arange(g), k1)
    rows = {
        "source": [gene_names[j] for j in idx.ravel()],
        "target": [gene_names[i] for i in src],
        "importance": val.ravel().astype(np.float32),
    }
    df = pd.DataFrame(rows)
    df = df[df["importance"] > args.threshold]
    df.to_csv(fname, index=False)
    print(f"[coexpr] {len(df)} edges -> {fname} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
