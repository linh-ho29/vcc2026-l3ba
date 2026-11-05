"""Full-pipeline real-data compatibility smoke test.

NOT TRAINING. Forward pass only, eval mode, no_grad - no parameter ever
changes. Exercises every new component together against real data:

  - 2,000 real HepG2 control cells (data/scgpt/hepg2_real_control_cells_pie_axis.npy,
    extracted from GSE264667_hepg2_raw_singlecell_01.h5ad, aligned to the PIE
    6,642-gene axis - 6,641/6,642 genes present)
  - the real frozen fine-tuned scGPT gene embeddings
    (data/scgpt/gene_emb_finetuned_pie.npy, per the user's explicit
    direction to freeze the fine-tuned scGPT for the control branch)
  - real k562 (64-d) and hepg2 (32-d) GEARS perturbation embeddings, for
    genuinely overlapping genes (same data as gated_fusion's own real-data
    check)

Pipeline exercised: scGPT pooling -> PCA fit on real control cells -> NTC
latent bank -> empirical z_0 sampling -> gated fusion of real k562+hepg2
embeddings -> input conditioning -> discrete latent flow -> decoder ->
predicted expression. The context embedding has no real source yet (no
control-cell-derived context encoder exists in this repo) - zeros, clearly
labeled as a placeholder.

Run: python3 examples/real_data_pipeline_smoketest.py
(needs data/scgpt/*.npy and data/arc_pie/preprocessed/meta.json locally -
not bundled in this repo; see gears_track/ on the GEARS branch and the PIE
smoke test on the linhho branch for how to obtain them.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter import ControlCellEncoder, GatedFusionFlowAdapter, NTCLatentBank, ScGPTGeneEmbeddingPooling

ROOT = Path(__file__).resolve().parents[1]
Z0_DIM = 128
CONTEXT_DIM = 8
K562_SOURCE_DIM = 64
HEPG2_SOURCE_DIM = 32


def main() -> None:
    # --- Control branch: real HepG2 control cells -> pooled scGPT features -> PCA -> z_0 bank ---
    control_cells = np.load(ROOT / "data" / "scgpt" / "hepg2_real_control_cells_pie_axis.npy")
    scgpt_emb = np.load(ROOT / "data" / "scgpt" / "gene_emb_finetuned_pie.npy")
    print(f"Real HepG2 control cells: {control_cells.shape}, scGPT embeddings: {scgpt_emb.shape}")

    pooling = ScGPTGeneEmbeddingPooling(torch.from_numpy(scgpt_emb))
    pooling.eval()
    with torch.no_grad():
        pooled = pooling(torch.from_numpy(control_cells).float())
    print(f"Pooled control-cell features: {tuple(pooled.shape)}, finite: {torch.isfinite(pooled).all().item()}")

    encoder = ControlCellEncoder(latent_dim=Z0_DIM).fit(pooled.numpy())
    print(f"PCA fit on {control_cells.shape[0]} real control cells, "
          f"explained variance ratio: {encoder.explained_variance_ratio_:.4f}")
    with torch.no_grad():
        z0_all = encoder(pooled)

    bank = NTCLatentBank()
    bank.add("hepg2", z0_all)
    print(f"NTC latent bank: {len(bank)} real control-cell latents stored for hepg2")

    # --- Perturbation branch: real k562 + hepg2 GEARS embeddings, overlapping genes ---
    k562_emb = np.load("/tmp/k562_pert_emb.npy")
    k562_idx = pd.read_csv("/tmp/k562_pert_idx.csv")
    hepg2_gene_emb = np.load("/tmp/hepg2_gene_emb.npz", allow_pickle=True)

    trained_mask = k562_idx["exposure_status"] == "trained_on_direct_K562_examples"
    k562_gene_to_row = dict(zip(k562_idx[trained_mask]["target_gene_symbol"], k562_idx[trained_mask]["row_index"]))
    hepg2_genes = hepg2_gene_emb["gene_symbols"].astype(str)
    hepg2_gene_to_row = {g: i for i, g in enumerate(hepg2_genes)}
    overlap = sorted(set(k562_gene_to_row) & set(hepg2_gene_to_row))
    print(f"Real overlapping genes (k562 trained-direct AND hepg2 gene-embedding): {len(overlap)}")

    n = min(len(overlap), control_cells.shape[0])  # one perturbation batch row per control cell, for this demo
    k562_tensor = torch.from_numpy(k562_emb[[k562_gene_to_row[g] for g in overlap[:n]]]).float()
    hepg2_tensor = torch.from_numpy(hepg2_gene_emb["post_graph_embeddings"][[hepg2_gene_to_row[g] for g in overlap[:n]]]).float()

    # --- Sample real z_0 (empirical, not mean/variance) for this batch ---
    z0 = bank.sample(n, context="hepg2", generator=torch.Generator().manual_seed(2026))

    # Placeholder context embedding - no real control-cell-derived context encoder
    # exists yet in this repo; zeros, not a claim about any actual context.
    context = torch.zeros(n, CONTEXT_DIM)

    model = GatedFusionFlowAdapter(
        gears_source_dims=(K562_SOURCE_DIM, HEPG2_SOURCE_DIM),
        context_dim=CONTEXT_DIM, z0_dim=Z0_DIM, n_genes=len(overlap[:n]),
    )
    model.eval()
    with torch.no_grad():
        out = model(z0, [k562_tensor, hepg2_tensor], context)

    print(f"\nFull pipeline output (n={n} rows):")
    print(f"  predicted_expression: {tuple(out['predicted_expression'].shape)}, "
          f"finite: {torch.isfinite(out['predicted_expression']).all().item()}")
    print(f"  z_pert: {tuple(out['z_pert'].shape)}")
    print(f"  gate_alpha (first 5 rows, [k562, hepg2]):")
    for i in range(min(5, n)):
        print(f"    {overlap[i]:12s} {out['gate_alpha'][i].tolist()}")
    print("\nDONE - forward pass only, no training, no gradients computed.")


if __name__ == "__main__":
    main()
