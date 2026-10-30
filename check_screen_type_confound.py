"""
Tests whether M2 (RPE1-only) beating M1 (K562-gwps-only) on the HepG2 held-out
validation reflects genuine biology or a screen-methodology confound: K562_gwps
is a genome-wide screen, while RPE1/HepG2/Jurkat (this Arc dataset) are all the
same "essential-gene" screen family. Compares K562_gwps vs K562-essential
(same screen family as the target) as the K562-side predictor, holding
everything else (gate, shrinkage, ground truth) identical to e10's run.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent))
from expression_gating import GateConfig, compute_context_gate_stats, compute_gate_weights
from multi_source_transfer import SourceBank

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "predictions"
SHRINKAGE = 0.65

k562_gwps = SourceBank(PRED_DIR / "signature_bank_k562gwps.npz")
k562_ess = SourceBank(PRED_DIR / "signature_bank_k562_essential.npz")
rpe1 = SourceBank(PRED_DIR / "signature_bank_rpe1.npz")

print("Loading HepG2 raw single-cell file (control cells only)...", flush=True)
a = ad.read_h5ad(ROOT / "data" / "GSE264667_hepg2_raw_singlecell_01.h5ad", backed="r")
labels = a.obs["gene"].astype(str).to_numpy()
hepg2_gene_symbols = a.var["gene_name"].astype(str).to_numpy()
ctrl_idx = np.flatnonzero(labels == "non-targeting")
X_ctrl = a.X[np.sort(ctrl_idx), :]
X_ctrl = sp.csr_matrix(X_ctrl) if not sp.issparse(X_ctrl) else X_ctrl.tocsr()
hepg2_stats = compute_context_gate_stats(X_ctrl, block=256)
hepg2_gene_idx = {g: i for i, g in enumerate(hepg2_gene_symbols)}
a.file.close()
gate_full = compute_gate_weights(hepg2_stats, GateConfig(mode="combined"))

de = pd.read_parquet(ROOT / "data" / "arc_pie" / "hepg2.parquet",
                      columns=["target", "feature", "target_mean", "ref_mean"])
de_ontarget = de[de["target"] == de["feature"]].set_index("target")
PSEUDOCOUNT = 0.5
de_ontarget = de_ontarget.assign(
    y_true=np.log((de_ontarget["target_mean"] + PSEUDOCOUNT) / (de_ontarget["ref_mean"] + PSEUDOCOUNT)))

print(f"\nK562-gwps perturbations: {len(k562_gwps.perts)}", flush=True)
print(f"K562-essential perturbations: {len(k562_ess.perts)}", flush=True)
print(f"RPE1 perturbations: {len(rpe1.perts)}", flush=True)

targets = sorted(set(de_ontarget.index) & set(hepg2_gene_idx))
print(f"Evaluation targets: {len(targets)}", flush=True)

rows = []
for t in targets:
    gi = hepg2_gene_idx[t]
    gate = float(gate_full[gi])
    y_true = float(de_ontarget.loc[t, "y_true"])
    k_gwps = k562_gwps.effect_row(t, [t])[0]
    k_ess = k562_ess.effect_row(t, [t])[0]
    r_eff = rpe1.effect_row(t, [t])[0]
    rows.append({
        "target": t, "y_true": y_true,
        "M1_k562_gwps": k_gwps * gate * SHRINKAGE if not np.isnan(k_gwps) else np.nan,
        "M1b_k562_essential": k_ess * gate * SHRINKAGE if not np.isnan(k_ess) else np.nan,
        "M2_rpe1": r_eff * gate * SHRINKAGE if not np.isnan(r_eff) else np.nan,
    })
df = pd.DataFrame(rows)

print(f"\nCoverage: k562_gwps={df['M1_k562_gwps'].notna().sum()}, "
      f"k562_essential={df['M1b_k562_essential'].notna().sum()}, rpe1={df['M2_rpe1'].notna().sum()}\n")
for col in ["M1_k562_gwps", "M1b_k562_essential", "M2_rpe1"]:
    sub = df[["y_true", col]].dropna()
    yt, yp = sub["y_true"].to_numpy(), sub[col].to_numpy()
    pearson = pearsonr(yt, yp)[0]
    spearman = spearmanr(yt, yp)[0]
    print(f"  {col:22s} n={len(sub):4d}  Pearson={pearson:.4f}  Spearman={spearman:.4f}")
print("\nDONE")
