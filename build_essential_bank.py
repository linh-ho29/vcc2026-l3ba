"""
Generic version of rpe1_signature_bank.py's build logic, parameterized by
source file - used here to build a K562-essential bank (same guide-level raw
pseudobulk format as RPE1's bank) to test whether M2 (RPE1-only) beating M1
(K562-genome-wide-only) on the HepG2 held-out validation reflects genuine
biology or a screen-methodology confound (K562_gwps is a genome-wide screen;
RPE1/HepG2/Jurkat are all the same "essential-gene" screen family - this
K562-essential screen matches that family instead).
"""
import hashlib
import sys
from pathlib import Path

import anndata as ad
import numpy as np

PSEUDOCOUNT = 0.5
CLIP_LOG_FOLD = 1.0
MIN_CELLS_PER_GROUP = 15
CONTROL_LABEL = "non-targeting"


def sha256_of(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def build(src_path, out_path):
    print(f"Hashing {src_path.name}...", flush=True)
    print(f"  sha256={sha256_of(src_path)}", flush=True)

    a = ad.read_h5ad(src_path, backed="r")
    print(f"Shape: {a.shape}", flush=True)
    gene_symbols = a.var["gene_name"].astype(str).to_numpy()
    names = a.obs_names.tolist()
    targets = np.array([n.split("_")[1] for n in names])
    ncells = a.obs["num_cells_unfiltered"].to_numpy().astype(np.float64)
    X = a.X[:, :]
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    a.file.close()

    def weighted_group_mean(mask):
        w = ncells[mask]
        total_w = w.sum()
        if total_w <= 0:
            return None, 0.0
        return np.average(X[mask], axis=0, weights=w), float(total_w)

    ctrl_mask = targets == CONTROL_LABEL
    control_mean, control_total_cells = weighted_group_mean(ctrl_mask)
    print(f"Control: {int(ctrl_mask.sum())} guide rows, {control_total_cells:.0f} cells", flush=True)

    unique_targets = np.unique(targets)
    effects, n_omitted = {}, 0
    for target in unique_targets:
        if target == CONTROL_LABEL:
            continue
        mean, total_cells = weighted_group_mean(targets == target)
        if total_cells < MIN_CELLS_PER_GROUP:
            n_omitted += 1
            continue
        effect = np.log((mean + PSEUDOCOUNT) / (control_mean + PSEUDOCOUNT))
        effects[target] = np.clip(effect, -CLIP_LOG_FOLD, CLIP_LOG_FOLD).astype(np.float32)

    print(f"Trained signatures: {len(effects)} (omitted <{MIN_CELLS_PER_GROUP} cells: {n_omitted})", flush=True)
    np.savez_compressed(out_path, genes=gene_symbols, perts=np.array(list(effects.keys())),
                         effects=np.stack(list(effects.values())), control_mean=control_mean)
    print(f"Saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    ROOT = Path(__file__).parent
    build(ROOT / "data" / "K562_essential_raw_bulk_01.h5ad",
          ROOT / "predictions" / "signature_bank_k562_essential.npz")
    print("DONE", flush=True)
