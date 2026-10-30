"""
E07: RPE1 perturbation signature bank, built with the SAME effect definition,
pseudocount, clip bound, and minimum-cell threshold as the existing K562_gwps
bank (gwps_signature_bank.py) - natural-log pseudocount ratio on RAW (not
normalized) per-cell mean counts:

    effect = clip(ln((mean_perturbed_raw + 0.5) / (mean_control_raw + 0.5)), -1, 1)

Source: gs://2026-vcc-llsa/train/rpe1_raw_bulk_01.h5ad - a guide-level raw
pseudobulk file (one row per sgRNA, "raw" meaning per-cell mean UMI count,
NOT the z-score-normalized rpe1_normalized_singlecell_01.h5ad used for cross-
context validation elsewhere in this project). 2393 unique gene targets, 130
non-targeting control rows, 135 genes represented by >1 guide (up to 3) -
aggregated here by cell-count-weighted mean, matching how the K562 pipeline
pools all cells for a gene target regardless of which guide produced them.

Missing-value handling (distinguishing "unmeasured" from "zero effect"):
  - A gene target is OMITTED from the bank entirely if its total raw cell
    count (summed across its guides) is below MIN_CELLS_PER_GROUP - same
    threshold K562 uses to skip low-confidence groups.
  - Within a kept perturbation's effect vector, a (perturbation, gene) pair
    is marked NaN (not a numerically-defined-but-meaningless ~0) if neither
    the control nor the perturbed mean raw count clears MIN_DETECTABLE_COUNT -
    i.e. a gene with no real measured expression in that pseudobulk is
    recorded as untested, not as "no change."

Does NOT touch, retrain, or modify signature_bank_k562gwps.npz.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).parent
SRC_PATH = ROOT / "data" / "rpe1_raw_bulk_01.h5ad"
OUT_NPZ = ROOT / "predictions" / "signature_bank_rpe1.npz"
OUT_MANIFEST = ROOT / "predictions" / "signature_bank_rpe1_manifest.json"

PSEUDOCOUNT = 0.5          # identical to gwps_signature_bank.py
CLIP_LOG_FOLD = 1.0        # identical to gwps_signature_bank.py
MIN_CELLS_PER_GROUP = 15   # identical to gwps_signature_bank.py
MIN_DETECTABLE_COUNT = 0.05  # below this in BOTH control and perturbed mean raw count,
                             # a gene is "untested" for that perturbation, not "unchanged"
CONTROL_LABEL = "non-targeting"


def sha256_of(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


print("Hashing source file...", flush=True)
src_hash = sha256_of(SRC_PATH)
print(f"  {SRC_PATH.name}: sha256={src_hash}", flush=True)

a = ad.read_h5ad(SRC_PATH, backed="r")
print(f"Shape: {a.shape}", flush=True)
gene_symbols = a.var["gene_name"].astype(str).to_numpy()
n_genes = len(gene_symbols)

names = a.obs_names.tolist()
targets = np.array([n.split("_")[1] for n in names])
ncells = a.obs["num_cells_unfiltered"].to_numpy().astype(np.float64)  # no NaNs, unlike num_cells_filtered

X = a.X[:, :]
X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
a.file.close()

unique_targets, counts = np.unique(targets, return_counts=True)
print(f"Unique gene targets (incl. control): {len(unique_targets)}", flush=True)
print(f"Targets with >1 guide row: {int(np.sum(counts[unique_targets != CONTROL_LABEL] > 1))}", flush=True)


def weighted_group_mean(rows_mask):
    """Cell-count-weighted mean across all guide rows for one gene target -
    matches K562's per-cell pooling (not a plain mean-of-guide-means)."""
    w = ncells[rows_mask]
    total_w = w.sum()
    if total_w <= 0:
        return None, 0.0
    return np.average(X[rows_mask], axis=0, weights=w), float(total_w)


ctrl_mask = targets == CONTROL_LABEL
control_mean, control_total_cells = weighted_group_mean(ctrl_mask)
print(f"Control (non-targeting): {int(ctrl_mask.sum())} guide rows, "
      f"{control_total_cells:.0f} total cells", flush=True)

effects = {}
n_tested_per_pert = {}
n_omitted_low_cells = 0
t0 = time.time()
for i, target in enumerate(unique_targets):
    if target == CONTROL_LABEL:
        continue
    rows_mask = targets == target
    mean, total_cells = weighted_group_mean(rows_mask)
    if total_cells < MIN_CELLS_PER_GROUP:
        n_omitted_low_cells += 1
        continue

    effect = np.log((mean + PSEUDOCOUNT) / (control_mean + PSEUDOCOUNT))
    effect = np.clip(effect, -CLIP_LOG_FOLD, CLIP_LOG_FOLD).astype(np.float32)

    untested = (mean < MIN_DETECTABLE_COUNT) & (control_mean < MIN_DETECTABLE_COUNT)
    effect[untested] = np.nan

    effects[target] = effect
    n_tested_per_pert[target] = int((~untested).sum())
    if i % 500 == 0:
        print(f"  {i}/{len(unique_targets)} targets processed, {time.time()-t0:.1f}s elapsed", flush=True)

print(f"\nTrained signatures: {len(effects)} (omitted for <{MIN_CELLS_PER_GROUP} cells: {n_omitted_low_cells})", flush=True)
tested_counts = np.array(list(n_tested_per_pert.values()))
print(f"Genes marked 'tested' per perturbation: mean={tested_counts.mean():.0f}/{n_genes}, "
      f"min={tested_counts.min()}, max={tested_counts.max()}", flush=True)

perts_arr = np.array(list(effects.keys()))
effects_arr = np.stack(list(effects.values()))
np.savez_compressed(
    OUT_NPZ,
    genes=gene_symbols,
    perts=perts_arr,
    effects=effects_arr,  # NaN = untested (gene below detection in both groups), not zero-effect
    control_mean=control_mean,
)
print(f"\nSaved {OUT_NPZ} ({OUT_NPZ.stat().st_size/1e6:.1f} MB)", flush=True)

manifest = {
    "bank_version": "1.0",
    "built_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_file": "gs://2026-vcc-llsa/train/rpe1_raw_bulk_01.h5ad",
    "source_file_sha256": src_hash,
    "source_description": "Guide-level raw pseudobulk (per-cell mean UMI count per guide), "
                           "Replogle RPE1 essential-gene CRISPRi screen. NOT the z-score-normalized "
                           "rpe1_normalized_singlecell_01.h5ad used elsewhere for cross-context validation.",
    "effect_definition": "clip(ln((mean_perturbed_raw + pseudocount) / (mean_control_raw + pseudocount)), "
                          "-clip_bound, clip_bound) - identical formula, pseudocount, and clip bound to "
                          "signature_bank_k562gwps.npz (gwps_signature_bank.py), computed on raw (not "
                          "CP10k- or z-score-normalized) per-cell mean counts for definitional consistency.",
    "pseudocount": PSEUDOCOUNT,
    "clip_log_fold": CLIP_LOG_FOLD,
    "min_cells_per_group": MIN_CELLS_PER_GROUP,
    "min_detectable_count_for_tested_flag": MIN_DETECTABLE_COUNT,
    "n_genes": int(n_genes),
    "n_guide_rows_total": int(len(names)),
    "n_control_guide_rows": int(ctrl_mask.sum()),
    "n_control_total_cells": control_total_cells,
    "n_unique_gene_targets_total": int(len(unique_targets) - 1),
    "n_gene_targets_with_multiple_guides": int(np.sum(counts[unique_targets != CONTROL_LABEL] > 1)),
    "n_perturbations_trained": int(len(effects)),
    "n_perturbations_omitted_low_cells": int(n_omitted_low_cells),
    "missing_value_handling": "NaN in the effects matrix = gene untested for that perturbation "
                               "(both control and perturbed raw mean count below "
                               f"{MIN_DETECTABLE_COUNT}), distinct from a numerically-near-zero "
                               "measured effect. Perturbations with <min_cells_per_group total cells "
                               "across all their guides are omitted entirely, not represented as "
                               "zero-effect. IN PRACTICE this within-vocabulary NaN mechanism found "
                               "zero cases in this file (checked directly, not assumed) - the source "
                               "file's 8749-gene vocabulary was already pre-filtered upstream "
                               "(min control_mean=0.055, right at this threshold; var['in_matrix'] "
                               "suggests an existing QC filter). The REAL missing-value axis for this "
                               "bank is vocabulary coverage: it has effects for only 8749 genes total, "
                               "not the full competition panel or even K562_gwps's 8248-gene "
                               "vocabulary. Any downstream combination (E08+) must treat a gene absent "
                               "from this bank's vocabulary as missing information, never zero-fill it.",
    "mean_genes_tested_per_perturbation": float(tested_counts.mean()),
    "min_genes_tested_per_perturbation": int(tested_counts.min()),
    "max_genes_tested_per_perturbation": int(tested_counts.max()),
    "gene_identifier_space": "HGNC gene symbols (var['gene_name']) - same identifier space as "
                              "signature_bank_k562gwps.npz, the competition panel, and the "
                              "scBaseCount reference.",
    "note_on_scale": "This bank's raw-count basis makes it formula-identical to the K562 bank, but "
                     "the two cell lines' sequencing depth, capture efficiency, and essential-gene "
                     "library composition differ - E08 must verify effect-magnitude comparability "
                     "before linearly combining the two banks, not assume it from formula identity alone.",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
print(f"Saved manifest to {OUT_MANIFEST}", flush=True)
print("DONE", flush=True)
