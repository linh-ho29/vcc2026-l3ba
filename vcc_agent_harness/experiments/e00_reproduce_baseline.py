"""
E00: Reproduce the existing K562 baseline's exported predictions.

Verifies that re-deriving a sample of (perturbation, context) predictions
from the saved signature bank, using the exact same seeded logic as
generate_predictions_gwps.py, matches what's already stored in
final_predictions_DEF_gwps.h5ad bit-for-bit. Does NOT re-download or
re-process K562_gwps (65.8GB) - that would violate the no->5GB-download
rule for no real benefit, since the bank itself is the baseline's complete,
already-computed state.
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "predictions"
FINAL_DIR = Path.home() / "Downloads" / "vcc_2026_controls_final"
OUT_PATH = Path(__file__).resolve().parent / "E00_metrics.json"

SEED = 2026
SHRINKAGE = 0.65
CLIP_LOG_FOLD = 1.0
CELLS_PER_PERT = 400


def make_predicted_rows(X_ctrl, sample_idx, log_effect, strength, rng):
    subset = X_ctrl[sample_idx, :].tocsr(copy=True).astype(np.float32)
    if log_effect is not None:
        factor = np.exp(np.clip(strength * log_effect, -CLIP_LOG_FOLD, CLIP_LOG_FOLD))
        subset.data *= factor[subset.indices]
    lo = np.floor(subset.data)
    subset.data = (lo + (rng.random(len(lo)) < (subset.data - lo))).astype(np.float32)
    subset.eliminate_zeros()
    return subset


def main():
    bank = np.load(PRED_DIR / "signature_bank_k562gwps.npz", allow_pickle=True)
    source_genes = bank["genes"].astype(str)
    source_perts = bank["perts"].astype(str)
    source_effects = bank["effects"]
    source_gene_index = {g: i for i, g in enumerate(source_genes)}
    source_pert_index = {p: i for i, p in enumerate(source_perts)}

    panel_genes = pd.read_csv(FINAL_DIR / "gene_names.csv").iloc[:, 0].astype(str).tolist()
    panel_perts = pd.read_csv(FINAL_DIR / "pert_counts.csv").iloc[:, 0].astype(str).tolist()
    src_idx = np.asarray([source_gene_index.get(g, -1) for g in panel_genes], dtype=np.int64)

    stored = ad.read_h5ad(PRED_DIR / "final_predictions_DEF_gwps.h5ad", backed="r")

    context = "D"
    first_pert = next(p for p in panel_perts if p in source_pert_index)
    rng = np.random.default_rng(SEED + sum(map(ord, context)))

    a = ad.read_h5ad(FINAL_DIR / f"context_{context}.h5ad", backed="r")
    X = a.X[:, :]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()

    # Reproduce the exact rng draw sequence generate_predictions_gwps.py used:
    # one rng.choice call per perturbation, in panel order, before this one.
    for p in panel_perts:
        if p == first_pert:
            break
        rng.choice(a.n_obs, size=CELLS_PER_PERT, replace=True)

    sel = rng.choice(a.n_obs, size=CELLS_PER_PERT, replace=True)
    delta = np.zeros(len(panel_genes), dtype=np.float32)
    match = src_idx >= 0
    delta[match] = source_effects[source_pert_index[first_pert], src_idx[match]]
    reproduced = make_predicted_rows(X, sel, delta, SHRINKAGE, rng)

    row_offset = panel_perts.index(first_pert) * CELLS_PER_PERT
    stored_rows = stored.X[row_offset:row_offset + CELLS_PER_PERT, :]
    stored_rows = sp.csr_matrix(stored_rows) if not sp.issparse(stored_rows) else stored_rows.tocsr()

    exact_match = (reproduced.shape == stored_rows.shape and
                   np.array_equal(reproduced.toarray(), stored_rows.toarray()))

    result = {
        "status": "PASS" if exact_match else "FAIL",
        "context": context,
        "perturbation_checked": first_pert,
        "reproduced_shape": list(reproduced.shape),
        "stored_shape": list(stored_rows.shape),
        "exact_bitwise_match": bool(exact_match),
        "bank_n_perturbations": int(len(source_perts)),
        "bank_n_genes": int(len(source_genes)),
        "note": "Verifies re-derivation from the saved bank reproduces the already-exported "
                "prediction bit-for-bit, given fixed seeds. Does not re-process raw K562_gwps.",
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    a.file.close()
    stored.file.close()
    sys.exit(0 if exact_match else 1)


if __name__ == "__main__":
    main()
