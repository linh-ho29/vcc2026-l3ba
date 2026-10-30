"""
Regenerates the final (D/E/F) predictions with E03 (combined) basal-expression
gating applied - the mode validated against real RPE1 ground truth to show a
genuine improvement over the ungated baseline (Pearson 0.501 vs 0.481,
Spearman 0.696 vs 0.686; see cross_context_validation_rpe1.py).

Does NOT touch or overwrite the already-submitted final_predictions_DEF_gwps.h5ad
or final.vcc - writes to a clearly separate file. Does not run vcc prep.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent))
from expression_gating import GateConfig, compute_context_gate_stats, compute_gate_weights, apply_gate_to_effect

ROOT = Path(__file__).parent
FINAL_DIR = Path.home() / "Downloads" / "vcc_2026_controls_final"
PRED_DIR = ROOT / "predictions"

SEED = 2026
SHRINKAGE = 0.65
CLIP_LOG_FOLD = 1.0
CELLS_PER_PERT = 400
MAX_CELL_COUNTS = 1_000_000
GATE_CONFIG = GateConfig(mode="combined")  # E03, validated against real RPE1 ground truth

bank = np.load(PRED_DIR / "signature_bank_k562gwps.npz", allow_pickle=True)
source_genes = bank["genes"].astype(str)
source_perts = bank["perts"].astype(str)
source_effects = bank["effects"]
source_gene_index = {x: i for i, x in enumerate(source_genes)}
source_pert_index = {x: i for i, x in enumerate(source_perts)}
print(f"Loaded bank: {len(source_perts)} trained perturbations, {len(source_genes)} genes, "
      f"gating mode={GATE_CONFIG.mode}", flush=True)


def bundle_metadata(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    gene_list = pd.read_csv(directory / "gene_names.csv").iloc[:, 0].astype(str).tolist()
    pert_list = pd.read_csv(directory / "pert_counts.csv").iloc[:, 0].astype(str).tolist()
    assert len(gene_list) == len(set(gene_list))
    assert len(pert_list) == len(set(pert_list))
    assert len(gene_list) == manifest["n_genes"]
    assert len(pert_list) == manifest["n_constructs"]
    return manifest, gene_list, pert_list


def make_predicted_rows(X_ctrl, sample_idx, log_effect, strength, rng):
    subset = X_ctrl[sample_idx, :].tocsr(copy=True).astype(np.float32)
    if log_effect is not None:
        factor = np.exp(np.clip(strength * log_effect, -CLIP_LOG_FOLD, CLIP_LOG_FOLD))
        subset.data *= factor[subset.indices]
    lo = np.floor(subset.data)
    subset.data = (lo + (rng.random(len(lo)) < (subset.data - lo))).astype(np.float32)
    subset.eliminate_zeros()
    totals = np.asarray(subset.sum(axis=1)).ravel()
    if np.any(totals > MAX_CELL_COUNTS):
        raise ValueError("Cell count cap violated")
    assert np.isfinite(subset.data).all() and np.all(subset.data >= 0)
    return subset


def generate_context_gated(directory, context, genes, perts, src_idx, outpath):
    a = ad.read_h5ad(directory / f"context_{context}.h5ad", backed="r")
    assert a.n_vars == len(genes)
    assert a.var_names.astype(str).tolist() == genes
    assert a.obs["context"].astype(str).eq(context).all()
    assert a.obs["target_gene"].astype(str).eq("non-targeting").all()

    X = a.X[:, :]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()

    # Real context-specific gate, computed from this context's own real control cells.
    stats = compute_context_gate_stats(X)
    gate = compute_gate_weights(stats, GATE_CONFIG)
    print(f"  {context}: mean gate weight={gate.mean():.4f}, "
          f"genes hard-zeroed={int((gate == 0).sum())}/{len(gate)}", flush=True)

    rng = np.random.default_rng(SEED + sum(map(ord, context)))
    out_parts, out_labels, missing = [], [], []
    match = src_idx >= 0
    for j, p in enumerate(perts):
        sel = rng.choice(a.n_obs, size=CELLS_PER_PERT, replace=True)
        if p not in source_pert_index:
            delta = None
            missing.append(p)
        else:
            raw_delta = np.zeros(len(genes), dtype=np.float32)
            raw_delta[match] = source_effects[source_pert_index[p], src_idx[match]]
            delta = apply_gate_to_effect(raw_delta, gate)  # the only change vs the ungated version
        part = make_predicted_rows(X, sel, delta, SHRINKAGE, rng)
        out_parts.append(part)
        out_labels.extend([p] * CELLS_PER_PERT)
        if (j + 1) % 50 == 0:
            print(f"  {context} generated {j + 1}/{len(perts)} targets", flush=True)

    X_out = sp.vstack(out_parts, format="csr")
    X_out.eliminate_zeros()
    obs = pd.DataFrame({
        "target_gene": pd.Categorical(out_labels),
        "context": pd.Categorical([context] * len(out_labels)),
    }, index=[f"{context}_{i:07d}" for i in range(len(out_labels))])
    result = ad.AnnData(X=X_out, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene_name")))
    print(f"  {context} shape: {result.shape}, nnz: {result.X.nnz}, unseen perturbations: {len(missing)}", flush=True)
    result.write_h5ad(outpath, compression="gzip")
    print(f"  Wrote {outpath} ({outpath.stat().st_size / 1e9:.2f} GB)", flush=True)
    a.file.close()
    del a, X, X_out, out_parts
    gc.collect()
    return outpath


manifest, genes, perts = bundle_metadata(FINAL_DIR)
assert len(manifest["contexts"]) == 3
src_idx = np.asarray([source_gene_index.get(g, -1) for g in genes], dtype=np.int64)

print("=== Generating GATED final predictions (D/E/F), E03 combined mode ===", flush=True)
context_files = []
for ctx in manifest["contexts"]:
    p = PRED_DIR / f"{manifest['partition']}_{ctx}_prediction_gated_e03.h5ad"
    generate_context_gated(FINAL_DIR, ctx, genes, perts, src_idx, p)
    context_files.append(p)

ads = [ad.read_h5ad(p) for p in context_files]
combo = ad.concat(ads, axis=0, join="inner", merge="same", index_unique=None)
assert combo.n_obs == len(manifest["contexts"]) * len(perts) * CELLS_PER_PERT
combo.X = combo.X.tocsr()
combo.X.eliminate_zeros()
outfile = PRED_DIR / "final_predictions_DEF_gwps_gated_e03.h5ad"
combo.write_h5ad(outfile, compression="gzip")
print(f"Complete gated panel: {outfile} {combo.shape} nnz={combo.X.nnz}", flush=True)
print(f"(Original ungated final_predictions_DEF_gwps.h5ad and final.vcc left untouched)", flush=True)
print("DONE", flush=True)
