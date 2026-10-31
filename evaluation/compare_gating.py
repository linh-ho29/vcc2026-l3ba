"""
Real comparison: gated vs. ungated K562->context transfer, using the actual
signature bank and the REAL control cells for contexts D/E/F (already
downloaded locally). Does not retrain or modify the existing signature bank -
only adjusts how much of its effect gets applied per context/gene.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from expression_gating import GateConfig, compute_context_gate_stats, compute_gate_weights, apply_gate_to_effect

PRED_DIR = Path(__file__).parent.parent / "predictions"
FINAL_DIR = Path.home() / "Downloads" / "vcc_2026_controls_final"
SHRINKAGE = 0.65
CLIP_LOG_FOLD = 1.0

bank = np.load(PRED_DIR / "signature_bank_k562gwps.npz", allow_pickle=True)
source_genes = bank["genes"].astype(str)
source_perts = bank["perts"].astype(str)
source_effects = bank["effects"]  # (n_perts, n_genes), raw log-effect, PRE shrinkage/clip
source_gene_index = {g: i for i, g in enumerate(source_genes)}
source_pert_index = {p: i for i, p in enumerate(source_perts)}
print(f"Loaded bank: {len(source_perts)} perturbations, {len(source_genes)} genes\n")

panel_genes = pd.read_csv(FINAL_DIR / "gene_names.csv").iloc[:, 0].astype(str).tolist()
panel_perts = pd.read_csv(FINAL_DIR / "pert_counts.csv").iloc[:, 0].astype(str).tolist()
src_idx_for_panel = np.asarray([source_gene_index.get(g, -1) for g in panel_genes], dtype=np.int64)

for context in ["D", "E", "F"]:
    print(f"=== Context {context} ===")
    a = ad.read_h5ad(FINAL_DIR / f"context_{context}.h5ad", backed="r")
    assert a.obs["target_gene"].astype(str).eq("non-targeting").all(), "expected pure-control bundle"
    X = a.X[:, :]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()

    stats = compute_context_gate_stats(X)
    gate = compute_gate_weights(stats, GateConfig(mode="combined"))
    print(f"  control cells: {a.n_obs}, genes: {a.n_vars}")
    print(f"  genes with zero detection in this context: {(stats['detection_fraction'] == 0).sum()} / {len(gate)}")
    print(f"  mean gate weight: {gate.mean():.4f}, genes fully gated to 0: {(gate == 0).sum()}, "
          f"genes with gate > 0.9: {(gate > 0.9).sum()}")

    # For each real target perturbation this context needs, measure how much the
    # K562 effect gets suppressed by gating, specifically on the genes it targets.
    biggest_suppressions = []
    for p in panel_perts:
        if p not in source_pert_index:
            continue
        raw_effect = source_effects[source_pert_index[p]]
        match = src_idx_for_panel >= 0
        panel_raw_effect = np.zeros(len(panel_genes), dtype=np.float32)
        panel_raw_effect[match] = raw_effect[src_idx_for_panel[match]]
        gated_effect = apply_gate_to_effect(panel_raw_effect, gate)
        suppression = np.abs(panel_raw_effect) - np.abs(gated_effect)
        strongest_idx = np.argmax(suppression)
        if suppression[strongest_idx] > 0.05:  # meaningful suppression on at least one gene
            biggest_suppressions.append((p, panel_genes[strongest_idx], panel_raw_effect[strongest_idx],
                                          gated_effect[strongest_idx], gate[strongest_idx]))

    print(f"  perturbations with >=1 meaningfully-suppressed downstream gene: {len(biggest_suppressions)} / "
          f"{sum(p in source_pert_index for p in panel_perts)} covered perturbations")
    biggest_suppressions.sort(key=lambda r: abs(r[2]) - abs(r[3]), reverse=True)
    print("  top 5 examples (perturbation, suppressed gene, raw K562 effect, gated effect, gate weight):")
    for p, g, raw, gated, w in biggest_suppressions[:5]:
        print(f"    {p:10s} -> {g:10s}  raw={raw:+.3f}  gated={gated:+.3f}  gate={w:.3f}")
    a.file.close()
    print()

print("DONE")
