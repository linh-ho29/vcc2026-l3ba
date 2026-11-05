"""Extract and serialize post-GNN perturbation embeddings across the
per-cell-type GEARS checkpoints (K562, HepG2, RPE1, Jurkat).

"Post-GNN perturbation embedding" is defined precisely, not assumed: in
GEARS_Model (see gears_track/shared/gears_patched_source/model.py on the
GEARS branch), the raw per-gene embedding `self.pert_emb` is propagated
through `self.sim_layers` (SGConv GNN layers over the co-expression graph)
and then fused by `self.pert_fuse` (an MLP) - THAT fused output is the
post-GNN embedding. This matches exactly what each handoff already saved as
`perturbation_embeddings*.npz`/`.npy` ("fused"/"post_graph" in their own
field names) - extraction has already been run once per cell line, by
whoever trained each model; this script's job is to standardize dimensions
and merge across cell lines into one artifact, not to re-run GEARS.

Two extraction paths, both implemented:
  - from_precomputed_embeddings(): loads an already-saved embeddings file
    (.npz/.npy). This is the path actually exercised in this run, for all
    3 cell lines with real data available (K562, HepG2, Jurkat).
  - from_checkpoint(): genuinely loads a raw GEARS_Model checkpoint,
    .eval(), torch.no_grad(), and recomputes pert_fuse(sim_layers(pert_emb))
    directly from the model's own weights. Implemented against the real
    GEARS_Model class, but NOT exercised in this run: it additionally needs
    the co-expression/GO graph edge-index buffers that were registered on
    the model at training time, which are not bundled in any of today's
    handoffs on their own - calling it without them raises a clear error
    rather than silently extracting something wrong.

Standardization to d_pert=64 (requirement: "a fixed projection... not
learned"): identity if already 64-d (K562); a FIXED (seeded, untrained)
Gaussian random projection for anything narrower (HepG2, Jurkat: 32-d);
truncation for anything wider. Fixed and seeded so it's reproducible, not
fit to any particular cell line's data.

RPE1: Courtney's full epoch-8 embeddings export (gears_adapter_ready/
embeddings/representations.npz on her courtneychen branch) was not in the
pushed git tree - it ships separately as a ~543MB GitHub Release
(gears-rpe1-8epoch-v1) because of size. The release's representations.npz +
target_index.csv give a real target_fused vector (the post-GNN fused
perturbation embedding, same quantity as the other 3 lines) for all 9,853
perturbation-vocabulary genes; sha256 verified against her
trained_handoff_manifest.json before use. IMPORTANT CAVEAT (from her own
reports/embedding_diagnostics.csv, carried through standardize_dim as a
dimension change only - it does not fix this): target_fused has effective
rank ~1.0 out of its nominal 32 dimensions at both epoch 3 (her
validation-best) and epoch 8 (final) - i.e. it is close to a single scalar
repeated across 32 coordinates, not 32 independent features. Treat RPE1's
slot in the merged table as real coverage but very low information
content, not on par with the other 3 lines' embeddings. Both checkpoints'
embeddings are available (--rpe1-epoch 8, the default, or 3); they are
correlated (flattened Pearson r=0.92) but not identical (per-gene cosine
similarity is bimodal - median 1.0, mean 0.72, with some genes at -1.0 -
consistent with a near-rank-1 representation where sign/scale along the
single dominant direction can flip between checkpoints).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

CELL_LINES = ["K562", "HepG2", "RPE1", "Jurkat"]  # fixed stack order, per spec
D_PERT = 64
PROJECTION_SEED = 20261105  # fixed, not fit to data - reproducible across runs


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def from_precomputed_embeddings(
    path: Path, gene_key: str, embedding_key: str
) -> Dict[str, np.ndarray]:
    """Load an already-extracted post-GNN embeddings file (.npz/.npy index
    pair or a single .npz with both gene ids and embeddings)."""
    data = np.load(path, allow_pickle=True)
    genes = data[gene_key].astype(str)
    embeddings = np.asarray(data[embedding_key], dtype=np.float32)
    if len(genes) != embeddings.shape[0]:
        raise ValueError(
            f"{path}: {len(genes)} gene ids but {embeddings.shape[0]} embedding rows."
        )
    return dict(zip(genes, embeddings))


def from_precomputed_embeddings_split(
    embeddings_path: Path, index_csv: Path, gene_col: str, embedding_dim: int
) -> Dict[str, np.ndarray]:
    """Variant for cell lines whose genes/embeddings live in separate files
    (K562: perturbation_embeddings_fused.npy + perturbation_embedding_index.csv)."""
    import csv
    embeddings = np.load(embeddings_path).astype(np.float32)
    genes = []
    with open(index_csv) as f:
        for row in csv.DictReader(f):
            genes.append(row[gene_col])
    if len(genes) != embeddings.shape[0]:
        raise ValueError(
            f"{index_csv}: {len(genes)} rows but {embeddings_path} has {embeddings.shape[0]}."
        )
    return dict(zip(genes, embeddings))


def from_representations_npz(
    npz_path: Path, index_csv: Path, array_key: str, gene_col: str = "gene_symbol"
) -> Dict[str, np.ndarray]:
    """Variant for Courtney's RPE1 export: a multi-array representations.npz
    (gene_lookup/target_pre_go/target_post_go/target_fused/...) plus a
    separate target_index.csv mapping row -> gene_symbol. Pulls a single
    named array (``target_fused``, the post-GNN fused perturbation
    embedding) and aligns it to gene symbols by row order."""
    import csv
    data = np.load(npz_path)  # plain numeric arrays only, no allow_pickle
    embeddings = np.asarray(data[array_key], dtype=np.float32)
    genes = []
    with open(index_csv) as f:
        for row in csv.DictReader(f):
            genes.append(row[gene_col])
    if len(genes) != embeddings.shape[0]:
        raise ValueError(
            f"{index_csv}: {len(genes)} rows but {npz_path}[{array_key}] has {embeddings.shape[0]}."
        )
    return dict(zip(genes, embeddings))


def from_checkpoint(checkpoint_path: Path, model_args: dict, gears_source_root: Path) -> Dict[str, np.ndarray]:
    """Genuinely load a raw GEARS_Model checkpoint, eval(), no_grad(), and
    recompute the post-GNN embedding (pert_fuse(sim_layers(pert_emb))) for
    every gene in the model's perturbation graph.

    Requires the SAME co-expression/GO graph edge-index buffers the model
    was trained with - these are set by GEARS's own model_initialize() from
    PertData, not recoverable from the checkpoint's state_dict alone. This
    function does not fabricate them: it requires them to be passed in
    model_args, and raises if they are absent, rather than guessing.
    """
    import sys
    sys.path.insert(0, str(gears_source_root))
    from gears import GEARS_Model  # the patched source on the GEARS branch

    required_graph_keys = ("G_go", "G_go_weight", "G_sim", "G_sim_weight")
    missing = [k for k in required_graph_keys if k not in model_args]
    if missing:
        raise ValueError(
            f"from_checkpoint requires {required_graph_keys} in model_args (the "
            f"co-expression/GO graph buffers GEARS registers at training time via "
            f"model_initialize() from PertData) - missing {missing}. These are not "
            f"recoverable from the checkpoint's state_dict alone and are not bundled "
            f"in any handoff used in this run; refusing to extract against the "
            f"wrong graph rather than silently producing incorrect embeddings."
        )

    model = GEARS_Model(model_args)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        num_genes = model_args["num_genes"]
        gene_idx = torch.arange(num_genes, dtype=torch.long)
        pert_global_emb = model.pert_emb(gene_idx)
        emb = pert_global_emb
        for layer in model.sim_layers:
            emb = layer(emb, model_args["G_sim"])
        fused = model.pert_fuse(emb)

    gene_names = model_args["gene_names"]
    return dict(zip(gene_names, fused.numpy().astype(np.float32)))


# ---------------------------------------------------------------------------
# Standardization to d_pert
# ---------------------------------------------------------------------------

def make_fixed_projection(from_dim: int, to_dim: int, seed: int = PROJECTION_SEED) -> np.ndarray:
    """Fixed (seeded, untrained) Gaussian random projection matrix, scaled
    so projected vectors keep a comparable norm (standard random-projection
    practice: scale by 1/sqrt(from_dim))."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((from_dim, to_dim)) / np.sqrt(from_dim)).astype(np.float32)


def standardize_dim(embeddings: np.ndarray, target_dim: int = D_PERT) -> np.ndarray:
    """embeddings: [N, d]. Identity if d == target_dim; fixed random
    projection if d < target_dim; truncation if d > target_dim."""
    n, d = embeddings.shape
    if d == target_dim:
        return embeddings
    if d < target_dim:
        projection = make_fixed_projection(d, target_dim)
        return embeddings @ projection
    return embeddings[:, :target_dim]  # truncation


# ---------------------------------------------------------------------------
# Merge across cell lines
# ---------------------------------------------------------------------------

def merge_cell_lines(
    per_line_embeddings: Dict[str, Dict[str, np.ndarray]],
    cell_lines: Sequence[str] = CELL_LINES,
    d_pert: int = D_PERT,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """``per_line_embeddings[cell_line][gene] -> raw_dim embedding`` (already
    standardize_dim'd to d_pert by the caller). Returns
    ``(gene -> [len(cell_lines), d_pert] tensor, gene -> [len(cell_lines)] mask)``.
    Missing entries are imputed with the mean of whichever lines ARE
    available for that gene (never zero-filled silently); mask is 0 there.
    """
    all_genes = sorted(set().union(*[set(d.keys()) for d in per_line_embeddings.values()]))
    result_emb: Dict[str, torch.Tensor] = {}
    result_mask: Dict[str, torch.Tensor] = {}
    n_fully_missing = 0

    for gene in all_genes:
        rows = []
        mask = torch.zeros(len(cell_lines))
        available_rows = []
        for i, line in enumerate(cell_lines):
            vec = per_line_embeddings.get(line, {}).get(gene)
            if vec is not None:
                mask[i] = 1.0
                available_rows.append(vec)
            rows.append(vec)
        if not available_rows:
            n_fully_missing += 1
            continue  # no cell line has this gene at all - cannot impute from nothing
        cross_line_mean = np.mean(available_rows, axis=0)
        filled_rows = [r if r is not None else cross_line_mean for r in rows]
        result_emb[gene] = torch.tensor(np.stack(filled_rows), dtype=torch.float32)
        result_mask[gene] = mask

    if n_fully_missing:
        print(f"  {n_fully_missing} genes present in no cell line's output vocabulary at all - "
              f"excluded (cannot impute from zero sources).", flush=True)
    return result_emb, result_mask


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_hdf5(
    out_path: Path,
    embeddings: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    cell_lines: Sequence[str],
    dtype: str = "float32",
) -> None:
    genes = sorted(embeddings.keys())
    emb_dtype = np.float16 if dtype == "float16" else np.float32
    emb_stack = np.stack([embeddings[g].numpy() for g in genes]).astype(emb_dtype)
    mask_stack = np.stack([masks[g].numpy() for g in genes]).astype(np.uint8)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("embeddings", data=emb_stack, compression="gzip")  # [n_genes, n_lines, d_pert]
        f.create_dataset("mask", data=mask_stack, compression="gzip")       # [n_genes, n_lines]
        f.create_dataset("genes", data=np.array(genes, dtype=h5py.string_dtype()))
        f.attrs["cell_lines"] = json.dumps(list(cell_lines))
        f.attrs["d_pert"] = emb_stack.shape[-1]
        f.attrs["dtype"] = dtype
        f.attrs["projection_seed"] = PROJECTION_SEED
        f.attrs["n_genes"] = len(genes)
    print(f"Saved {out_path}: {emb_stack.shape} embeddings, {mask_stack.mean():.3f} mean coverage "
          f"across (gene, cell_line) pairs", flush=True)


# ---------------------------------------------------------------------------
# Main: real extraction run for K562, HepG2, Jurkat, RPE1
# ---------------------------------------------------------------------------

GEARS_TRACK_ROOT = Path(__file__).resolve().parents[2]  # gears_track/ (this script lives in shared/scripts/)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Defaults to merged/gears_embeddings_merged.h5, "
                         "or merged/gears_embeddings_merged_rpe1_epoch3.h5 if --rpe1-epoch 3.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--rpe1-epoch", type=int, choices=[8, 3], default=8,
                         help="Which of Courtney's two verified RPE1 checkpoints to use: 8 (final, "
                         "default, matches the other lines' convention) or 3 (her validation-best, "
                         "for comparison - see gears_track/README.md for how the two differ).")
    args = parser.parse_args()
    if args.out is None:
        suffix = "" if args.rpe1_epoch == 8 else "_rpe1_epoch3"
        args.out = GEARS_TRACK_ROOT / "merged" / f"gears_embeddings_merged{suffix}.h5"

    per_line: Dict[str, Dict[str, np.ndarray]] = {}

    print("K562: loading perturbation_embeddings_fused.npy (already 64-d, no projection needed)", flush=True)
    k562_dir = GEARS_TRACK_ROOT / "per_celltype" / "k562" / "embeddings"
    k562_raw = from_precomputed_embeddings_split(
        k562_dir / "perturbation_embeddings_fused.npy", k562_dir / "perturbation_embedding_index.csv",
        gene_col="target_gene_symbol", embedding_dim=64,
    )
    per_line["K562"] = {g: standardize_dim(v[None, :], D_PERT)[0] for g, v in k562_raw.items()}

    print("HepG2: loading gene_embeddings.npz post_graph_embeddings as the perturbation-embedding "
          "proxy (the real standalone perturbation_embeddings.npz did not survive an earlier upload "
          "- see gears_track/README.md); projecting 32-d -> 64-d", flush=True)
    hepg2_path = GEARS_TRACK_ROOT / "per_celltype" / "hepg2" / "embeddings" / "gene_embeddings.npz"
    hepg2_data = np.load(hepg2_path, allow_pickle=True)
    hepg2_genes = hepg2_data["gene_symbols"].astype(str)
    hepg2_emb = standardize_dim(hepg2_data["post_graph_embeddings"].astype(np.float32), D_PERT)
    per_line["HepG2"] = dict(zip(hepg2_genes, hepg2_emb))

    print("Jurkat: loading real perturbation_embeddings.npz (32-d); projecting 32-d -> 64-d", flush=True)
    jurkat_path = GEARS_TRACK_ROOT / "per_celltype" / "jurkat" / "embeddings" / "perturbation_embeddings.npz"
    jurkat_raw = from_precomputed_embeddings(
        jurkat_path, gene_key="perturbations", embedding_key="embeddings",
    )
    per_line["Jurkat"] = {g: standardize_dim(v[None, :], D_PERT)[0] for g, v in jurkat_raw.items()}

    rpe1_subdir = "embeddings" if args.rpe1_epoch == 8 else "embeddings_best_epoch_03"
    print(f"RPE1 (epoch {args.rpe1_epoch}): loading {rpe1_subdir}/representations.npz[target_fused] "
          "(32-d, real coverage over all 9,853 perturbation-vocabulary genes, but effective "
          "rank ~1.0 at both epochs - see module docstring); projecting 32-d -> 64-d", flush=True)
    rpe1_dir = GEARS_TRACK_ROOT / "per_celltype" / "rpe1" / rpe1_subdir
    rpe1_raw = from_representations_npz(
        rpe1_dir / "representations.npz", rpe1_dir / "target_index.csv",
        array_key="target_fused", gene_col="gene_symbol",
    )
    per_line["RPE1"] = {g: standardize_dim(v[None, :], D_PERT)[0] for g, v in rpe1_raw.items()}

    for line in CELL_LINES:
        print(f"  {line}: {len(per_line.get(line, {}))} genes", flush=True)

    embeddings, masks = merge_cell_lines(per_line, CELL_LINES, D_PERT)
    print(f"\nMerged: {len(embeddings)} genes with at least one real cell-line source", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_hdf5(args.out, embeddings, masks, CELL_LINES, dtype=args.dtype)

    # Sanity print: a handful of genes' mask vectors (which lines are real vs imputed)
    sample_genes = sorted(embeddings.keys())[:5]
    print("\nSample mask vectors [K562, HepG2, RPE1, Jurkat] (1=real, 0=imputed):", flush=True)
    for g in sample_genes:
        print(f"  {g}: {masks[g].tolist()}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
