"""
E01-E04: gating mode ablation on real D/E/F control cells.

IMPORTANT: this is a prediction-DIFFERENCE diagnostic, not an accuracy
measurement - we have no ground truth for D/E/F (confirmed earlier this
project; the organizers withhold it). It shows how much each gating mode
changes the existing baseline's output and why, not whether that change is
more accurate. Genuine accuracy validation requires a labeled unseen line
(RPE1), tracked separately pending the >5GB download approval it needs.

E01 = mode='none'              (constant global shrinkage; current submitted baseline)
E02 = mode='binary_detection'
E03 = mode='combined'          (smooth percentile+detection hybrid)
E04 = direction-aware           (E03's gate for downregulation, softer gate for induction)
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from expression_gating import (GateConfig, compute_context_gate_stats, compute_gate_weights,
                                compute_gate_weights_direction_aware, apply_gate_to_effect)

PRED_DIR = Path(__file__).resolve().parents[2] / "predictions"
FINAL_DIR = Path.home() / "Downloads" / "vcc_2026_controls_final"
OUT_PATH = Path(__file__).resolve().parent / "E01_E04_metrics.json"

bank = np.load(PRED_DIR / "signature_bank_k562gwps.npz", allow_pickle=True)
source_genes = bank["genes"].astype(str)
source_perts = bank["perts"].astype(str)
source_effects = bank["effects"]
source_gene_index = {g: i for i, g in enumerate(source_genes)}
source_pert_index = {p: i for i, p in enumerate(source_perts)}

panel_genes = pd.read_csv(FINAL_DIR / "gene_names.csv").iloc[:, 0].astype(str).tolist()
panel_perts = pd.read_csv(FINAL_DIR / "pert_counts.csv").iloc[:, 0].astype(str).tolist()
src_idx = np.asarray([source_gene_index.get(g, -1) for g in panel_genes], dtype=np.int64)
match = src_idx >= 0

results = {"experiments": {}, "note": "prediction-difference diagnostics only; no D/E/F ground truth exists"}

for context in ["D", "E", "F"]:
    a = ad.read_h5ad(FINAL_DIR / f"context_{context}.h5ad", backed="r")
    X = a.X[:, :]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    stats = compute_context_gate_stats(X)
    a.file.close()

    covered_perts = [p for p in panel_perts if p in source_pert_index]
    raw_effects = np.zeros((len(covered_perts), len(panel_genes)), dtype=np.float32)
    for i, p in enumerate(covered_perts):
        raw_effects[i, match] = source_effects[source_pert_index[p], src_idx[match]]

    modes = {
        "E01_none": compute_gate_weights(stats, GateConfig(mode="none")),
        "E02_binary_detection": compute_gate_weights(stats, GateConfig(mode="binary_detection")),
        "E03_combined": compute_gate_weights(stats, GateConfig(mode="combined")),
    }

    for name, gate in modes.items():
        gated = raw_effects * gate[None, :]
        suppression_frac = 1.0 - (np.abs(gated).sum() / max(np.abs(raw_effects).sum(), 1e-9))
        results["experiments"].setdefault(name, {})[context] = {
            "mean_gate_weight": float(gate.mean()),
            "fraction_hard_zeroed": float((gate == 0).mean()),
            "total_effect_magnitude_suppressed_frac": float(suppression_frac),
        }

    # E04: direction-aware, applied per (perturbation, gene) since direction depends on raw_effects' sign.
    e04_gated = np.zeros_like(raw_effects)
    induction_rescued_count = 0
    for i in range(len(covered_perts)):
        gate_i = compute_gate_weights_direction_aware(stats, raw_effects[i], GateConfig(mode="combined"))
        e04_gated[i] = raw_effects[i] * gate_i
        combined_gate_i = modes["E03_combined"]
        was_hard_zeroed_by_combined = (combined_gate_i == 0) & (raw_effects[i] > 0)
        induction_rescued_count += int((was_hard_zeroed_by_combined & (gate_i > 0)).sum())

    suppression_frac_e04 = 1.0 - (np.abs(e04_gated).sum() / max(np.abs(raw_effects).sum(), 1e-9))
    results["experiments"].setdefault("E04_direction_aware", {})[context] = {
        "total_effect_magnitude_suppressed_frac": float(suppression_frac_e04),
        "induction_cases_rescued_from_hard_zero": int(induction_rescued_count),
        "note": "cases where E03 would hard-zero a POSITIVE (induction) effect purely for low "
                "baseline detection, but E04 assigns it nonzero (discounted) weight instead",
    }

print(json.dumps(results, indent=2))
OUT_PATH.write_text(json.dumps(results, indent=2))
