"""Real-data compatibility smoke test for GatedMultiContextFusion.

NOT TRAINING. This script does a forward pass only, in ``eval()`` mode under
``torch.no_grad()`` - no parameter ever changes, no gradient is ever
computed against real data. It exists purely to confirm the module's
plumbing (shapes, dtypes, the heterogeneous k562/hepg2 embedding widths)
actually works against the real per-cell-type GEARS embeddings checked into
the ``GEARS`` branch, rather than only against synthetic fixtures. This does
NOT violate the package's "no training on real biological data in this
phase" rule - nothing here is a fit, an optimizer step, or a formal
validation of predictive accuracy.

Sources used (both genuinely real, read directly from the GEARS branch):
  - k562: perturbation_embeddings_fused.npy, restricted to genes with
    exposure_status == "trained_on_direct_K562_examples" (actually learned
    from direct examples, not graph-propagated placeholders), 64-d.
  - hepg2: gene_embeddings.npz's post_graph_embeddings, used as a documented
    proxy for a per-cell-type perturbation representation (hepg2's own
    standalone perturbation_embeddings.npz did not survive its upload -
    see gears_track/README.md on the GEARS branch), 32-d.

The context embedding ``c`` has no real source yet (Phase 1's control-cell
VAE/PCA does not exist in this repo) - a fixed zero vector is used, clearly
labeled as a placeholder, not a claim about any real context.

Run: python3 examples/real_data_fusion_smoketest.py <k562_pert_emb.npy>
     <k562_pert_index.csv> <hepg2_gene_emb.npz>
(paths from `git show GEARS:gears_track/per_celltype/.../...` - see the
GEARS branch; not bundled in this repo.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter.gated_fusion import GatedMultiContextFusion

K562_SOURCE_DIM = 64
HEPG2_SOURCE_DIM = 32
CONTEXT_DIM = 8


def main(k562_pert_emb_path: str, k562_pert_index_path: str, hepg2_gene_emb_path: str) -> None:
    k562_emb = np.load(k562_pert_emb_path)
    k562_idx = pd.read_csv(k562_pert_index_path)
    assert k562_emb.shape[1] == K562_SOURCE_DIM, k562_emb.shape

    trained_mask = k562_idx["exposure_status"] == "trained_on_direct_K562_examples"
    k562_trained = k562_idx[trained_mask]
    k562_gene_to_row = dict(zip(k562_trained["target_gene_symbol"], k562_trained["row_index"]))

    hepg2 = np.load(hepg2_gene_emb_path, allow_pickle=True)
    hepg2_genes = hepg2["gene_symbols"].astype(str)
    hepg2_emb = hepg2["post_graph_embeddings"]
    assert hepg2_emb.shape[1] == HEPG2_SOURCE_DIM, hepg2_emb.shape
    hepg2_gene_to_row = {g: i for i, g in enumerate(hepg2_genes)}

    overlap = sorted(set(k562_gene_to_row) & set(hepg2_gene_to_row))
    print(f"Real overlapping genes (k562 trained-direct AND hepg2 gene-embedding): {len(overlap)}")
    assert len(overlap) > 0, "no real overlap found - check the source files"

    k562_rows = [k562_gene_to_row[g] for g in overlap]
    hepg2_rows = [hepg2_gene_to_row[g] for g in overlap]
    k562_tensor = torch.from_numpy(k562_emb[k562_rows]).float()
    hepg2_tensor = torch.from_numpy(hepg2_emb[hepg2_rows]).float()
    B = len(overlap)

    # Placeholder context embedding - no real control-cell VAE/PCA exists yet
    # in this repo (Phase 1's other half); zeros, not a claim about any
    # actual context. Real usage must replace this once that module exists.
    context = torch.zeros(B, CONTEXT_DIM)

    fusion = GatedMultiContextFusion(
        source_dims=(K562_SOURCE_DIM, HEPG2_SOURCE_DIM), context_dim=CONTEXT_DIM, fused_dim=32
    )
    fusion.eval()
    with torch.no_grad():
        fused, alpha = fusion([k562_tensor, hepg2_tensor], context)

    print(f"Fused output: {tuple(fused.shape)}, finite: {torch.isfinite(fused).all().item()}")
    print(f"Gate weights (alpha): {tuple(alpha.shape)}, "
          f"row sums all 1.0: {torch.allclose(alpha.sum(-1), torch.ones(B), atol=1e-5)}")
    print("\nFirst 10 genes' gate weights [k562, hepg2] (untrained gate - not meaningful "
          "yet, confirms plumbing only):")
    for i in range(min(10, B)):
        print(f"  {overlap[i]:12s} alpha={alpha[i].tolist()}")
    print("\nDONE - forward pass only, no training, no gradients computed.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
