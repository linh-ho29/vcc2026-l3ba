"""
E05: scBaseCount curated reference.

Loads the 12 real, verified accessions in data/manifests/scbasecount_selection.csv
(human, non-perturbed, well-annotated; selection rationale and leakage checks
recorded in that manifest - none are RPE1, which is held out as our real
ground-truth validation line). For each of the 6 reference groups (k562_like,
erythroid, immune, epithelial, stromal, stem_progenitor), builds:

  - reference-group pseudobulk expression (mean of library-size-normalized,
    log1p counts, pooled across that group's studies - NOT raw counts, since
    studies differ substantially in sequencing depth)
  - per-group detection fraction (same definition as expression_gating's
    context stats: fraction of cells with nonzero raw count)
  - simple, well-established gene-family/pathway features (ribosomal protein,
    mitochondrial, histone, hemoglobin/erythroid - standard HGNC symbol
    prefixes, not an external curated pathway DB we don't have access to)
  - study-balanced PCA: each of the 12 accessions contributes exactly one
    pseudobulk row regardless of its cell count, so a large study cannot
    dominate the PCA the way raw per-cell PCA would let it.

Output: data/scbasecount_reference.npz, keyed by gene symbol (the same
identifier space as the K562_gwps signature bank and the competition panel).
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).parent
MANIFEST = ROOT / "data" / "manifests" / "scbasecount_selection.csv"
RAW_DIR = ROOT / "data" / "scbasecount"
OUT_PATH = ROOT / "predictions" / "scbasecount_reference.npz"
SEED = 2026


def load_subsampled(path, n_cap, seed):
    a = ad.read_h5ad(path, backed="r")
    rng = np.random.default_rng(seed)
    n = a.n_obs
    idx = np.sort(rng.choice(n, size=min(n_cap, n), replace=False))
    X = a.X[idx, :]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    symbols = a.var["gene_symbols"].astype(str).to_numpy()
    a.file.close()
    return X, symbols


def collapse_to_symbols(X, symbols, all_symbols_index):
    """Sum duplicate Ensembl IDs mapping to the same symbol; project onto a
    fixed, shared symbol vocabulary (same order for every accession), via a
    single sparse matmul against a (n_local_genes x n_out_genes) 0/1 mapping
    matrix - vectorized, not a per-gene Python loop."""
    n_out = len(all_symbols_index)
    out_cols = np.array([all_symbols_index.get(s, -1) for s in symbols])
    valid = out_cols >= 0
    local_rows = np.flatnonzero(valid)
    mapping = sp.csr_matrix(
        (np.ones(len(local_rows), dtype=np.float32), (local_rows, out_cols[valid])),
        shape=(len(symbols), n_out),
    )
    return (X.tocsr() @ mapping).tocsr()


manifest = pd.read_csv(MANIFEST)
assert len(manifest) <= 12, "exceeds the 12-source-file cap"
assert manifest["file_size_bytes"].sum() <= 5 * 1e9, "exceeds the 5GB download cap"
print(f"Manifest: {len(manifest)} accessions across {manifest['reference_group'].nunique()} reference groups", flush=True)

# Shared gene-symbol vocabulary: union of symbols seen in any of the 12 files.
print("Pass 1: collecting shared gene-symbol vocabulary...", flush=True)
all_symbols = set()
for _, row in manifest.iterrows():
    a = ad.read_h5ad(RAW_DIR / f"{row['srx_accession']}.h5ad", backed="r")
    all_symbols.update(a.var["gene_symbols"].astype(str).tolist())
    a.file.close()
all_symbols = sorted(all_symbols)
symbol_index = {s: i for i, s in enumerate(all_symbols)}
n_genes = len(all_symbols)
print(f"  {n_genes} unique gene symbols across all 12 accessions", flush=True)

study_pseudobulk = []       # one row per accession (study-balanced)
study_labels = []
group_sum_lognorm = {g: np.zeros(n_genes, dtype=np.float64) for g in manifest["reference_group"].unique()}
group_nonzero_count = {g: np.zeros(n_genes, dtype=np.float64) for g in manifest["reference_group"].unique()}
group_n_cells = {g: 0 for g in manifest["reference_group"].unique()}

for i, (_, row) in enumerate(manifest.iterrows()):
    acc, group, cap = row["srx_accession"], row["reference_group"], int(row["cells_used_cap"])
    print(f"  loading {acc} ({group}), cap={cap}...", flush=True)
    X_raw, symbols = load_subsampled(RAW_DIR / f"{acc}.h5ad", cap, SEED + i)
    X_sym = collapse_to_symbols(X_raw, symbols, symbol_index)

    lib_size = np.asarray(X_sym.sum(axis=1)).ravel()
    lib_size[lib_size == 0] = 1.0
    cpm = X_sym.multiply(1e4 / lib_size[:, None]).tocsr()
    lognorm = cpm.copy()
    lognorm.data = np.log1p(lognorm.data)

    study_pseudobulk.append(np.asarray(lognorm.mean(axis=0)).ravel())
    study_labels.append(acc)

    group_sum_lognorm[group] += np.asarray(lognorm.sum(axis=0)).ravel()
    group_nonzero_count[group] += np.asarray((X_sym > 0).sum(axis=0)).ravel()
    group_n_cells[group] += X_sym.shape[0]

study_pseudobulk = np.vstack(study_pseudobulk)  # (12, n_genes)
reference_groups = sorted(group_n_cells.keys())
group_mean_lognorm = np.vstack([group_sum_lognorm[g] / max(group_n_cells[g], 1) for g in reference_groups])
group_detection_fraction = np.vstack([group_nonzero_count[g] / max(group_n_cells[g], 1) for g in reference_groups])

# Study-balanced PCA: one row per accession regardless of cell count, so the
# 52,515-cell SRX10338552 accession does not outweigh the 2,157-cell
# SRX24763709 accession the way a pooled per-cell PCA would.
print("Running study-balanced PCA (12 studies x genes, mean-centered)...", flush=True)
centered = study_pseudobulk - study_pseudobulk.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
n_components = min(5, len(S))
pca_scores = U[:, :n_components] * S[:n_components]
pca_loadings = Vt[:n_components]
explained_var_ratio = (S[:n_components] ** 2) / (S ** 2).sum()
print(f"  explained variance ratio (top {n_components}): {explained_var_ratio.round(3).tolist()}", flush=True)

# Simple, well-established HGNC symbol-prefix gene families (not an external
# pathway DB - we don't have offline access to one, and these four families
# are standard, unambiguous nomenclature, directly relevant to this dataset's
# own biology: HBZ surfaced earlier as the single most gated-down gene in the
# D/E/F ablation, consistent with it being an erythroid/hemoglobin marker).
symbols_arr = np.array(all_symbols)
pathway_features = {
    "ribosomal": np.array([s.startswith(("RPL", "RPS")) for s in all_symbols]),
    "mitochondrial": np.array([s.startswith("MT-") for s in all_symbols]),
    "histone": np.array([s.startswith(("HIST", "H1-", "H2A", "H2B", "H3-", "H4-")) for s in all_symbols]),
    "hemoglobin_erythroid": np.array([s.startswith(("HBA", "HBB", "HBZ", "HBG", "HBD", "HBE", "HBQ", "HBM")) for s in all_symbols]),
}
for name, mask in pathway_features.items():
    print(f"  pathway feature '{name}': {int(mask.sum())} genes flagged", flush=True)

np.savez(
    OUT_PATH,
    gene_symbols=symbols_arr,
    reference_groups=np.array(reference_groups),
    group_mean_lognorm=group_mean_lognorm.astype(np.float32),
    group_detection_fraction=group_detection_fraction.astype(np.float32),
    group_n_cells=np.array([group_n_cells[g] for g in reference_groups]),
    study_labels=np.array(study_labels),
    study_pseudobulk=study_pseudobulk.astype(np.float32),
    pca_scores=pca_scores.astype(np.float32),
    pca_loadings=pca_loadings.astype(np.float32),
    pca_explained_var_ratio=explained_var_ratio.astype(np.float32),
    pathway_ribosomal=pathway_features["ribosomal"],
    pathway_mitochondrial=pathway_features["mitochondrial"],
    pathway_histone=pathway_features["histone"],
    pathway_hemoglobin_erythroid=pathway_features["hemoglobin_erythroid"],
)
print(f"\nSaved E05 curated reference to {OUT_PATH}", flush=True)
print(f"Reference groups: {reference_groups}", flush=True)
print(f"Cells per group: {[group_n_cells[g] for g in reference_groups]}", flush=True)
print("DONE", flush=True)
