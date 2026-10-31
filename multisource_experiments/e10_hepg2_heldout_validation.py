"""
E10: independent cross-cell-line validation on HepG2 - genuinely held out.
HepG2 was not used anywhere in E07 (RPE1 bank), E08 (M1-M4 code), or E09 (the
context-weight model, fit using only K562<->RPE1). This script is the first
time HepG2 data is loaded anywhere in this project.

Architecture and hyperparameters are frozen before this script runs:
  - signature_bank_k562gwps.npz / signature_bank_rpe1.npz: built in prior steps.
  - predictions/context_weight_model.npz: fit in e09_fit_context_weight.py,
    loaded here UNMODIFIED - no refitting against HepG2.
  - GateConfig(mode="combined") defaults: fixed in expression_gating.py,
    committed before RPE1 or HepG2 data existed in this project.
  - alpha grid {0, 0.25, 0.5, 0.75, 1.0}: prespecified, not searched.

Only HepG2's own non-targeting control cells are used for target-context
features (real detection_fraction/percentile, computed from the raw
single-cell file - not approximated from the DE table's aggregate means).
HepG2's measured perturbation outcomes are used ONLY as evaluation ground
truth, never as a model input.
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from expression_gating import GateConfig, compute_context_gate_stats, compute_gate_weights
from multi_source_transfer import SourceBank, combine_effect_row

ROOT = Path(__file__).parent.parent
PRED_DIR = ROOT / "predictions"
HEPG2_RAW = ROOT / "data" / "GSE264667_hepg2_raw_singlecell_01.h5ad"
HEPG2_DE = ROOT / "data" / "arc_pie" / "hepg2.parquet"
MODEL_PATH = PRED_DIR / "context_weight_model.npz"
OUT_PATH = ROOT / "vcc_agent_harness" / "experiments" / "E10_hepg2_results.json"

SHRINKAGE = 0.65
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_BOOTSTRAP = 2000
SEED = 2026

k562 = SourceBank(PRED_DIR / "signature_bank_k562gwps.npz")
rpe1 = SourceBank(PRED_DIR / "signature_bank_rpe1.npz")

model_npz = np.load(MODEL_PATH, allow_pickle=True)
model_coef = model_npz["coef"]
model_feat_mean, model_feat_std = model_npz["feat_mean"], model_npz["feat_std"]
model_feature_names = model_npz["feature_names"].astype(str).tolist()
print(f"Loaded frozen E09 model (fit on K562<->RPE1 only): {model_feature_names}", flush=True)

# --- HepG2's own real control-cell stats (never its perturbed cells) -----
print("Loading HepG2 raw single-cell file (backed, control cells only)...", flush=True)
a = ad.read_h5ad(HEPG2_RAW, backed="r")
labels = a.obs["gene"].astype(str).to_numpy() if "gene" in a.obs else a.obs["target_gene"].astype(str).to_numpy()
hepg2_gene_symbols = (a.var["gene_name"].astype(str).to_numpy() if "gene_name" in a.var
                      else a.var_names.astype(str).to_numpy())
ctrl_label_candidates = [l for l in np.unique(labels) if "non-target" in l.lower() or l.lower() == "control"]
ctrl_label = ctrl_label_candidates[0] if ctrl_label_candidates else "non-targeting"
ctrl_idx = np.flatnonzero(labels == ctrl_label)
print(f"  shape={a.shape}, control label='{ctrl_label}', n_control_cells={len(ctrl_idx)}", flush=True)
X_ctrl = a.X[np.sort(ctrl_idx), :]
X_ctrl = sp.csr_matrix(X_ctrl) if not sp.issparse(X_ctrl) else X_ctrl.tocsr()
hepg2_stats = compute_context_gate_stats(X_ctrl, block=256)
hepg2_gene_idx = {g: i for i, g in enumerate(hepg2_gene_symbols)}
a.file.close()
gate_weights_full = compute_gate_weights(hepg2_stats, GateConfig(mode="combined"))
print(f"  E03 gate: mean={gate_weights_full.mean():.4f}, hard-zeroed={(gate_weights_full==0).sum()}/{len(gate_weights_full)}", flush=True)


def hepg2_gate_and_features(gene_symbol, overlap_pert_set):
    """Real target-context (HepG2) gate weight and E09 feature vector for one
    gene - built only from HepG2's own control cells + static source info."""
    gi = hepg2_gene_idx.get(gene_symbol)
    if gi is None:
        return None, None
    gate = float(gate_weights_full[gi])
    baseline_expr = np.log1p(hepg2_stats["baseline_mean"][gi])
    detection_frac = hepg2_stats["detection_fraction"][gi]
    pathway_ribo = float(gene_symbol.startswith(("RPL", "RPS")))
    pathway_mito = float(gene_symbol.startswith("MT-"))
    pathway_histone = float(gene_symbol.startswith(("HIST", "H1-", "H2A", "H2B", "H3-", "H4-")))
    pathway_hemo = float(gene_symbol.startswith(("HBA", "HBB", "HBZ", "HBG", "HBD", "HBE")))
    ki, ri = k562.gene_index.get(gene_symbol), rpe1.gene_index.get(gene_symbol)
    if ki is None or ri is None:
        src_tgt_sim = 0.0  # neutral (mean-imputed at standardization) if either source lacks this gene
    else:
        src_tgt_sim = -abs(np.log1p(k562.control_mean[ki]) - np.log1p(rpe1.control_mean[ri]))
    is_pert_target = float(gene_symbol in overlap_pert_set)
    feat = np.array([baseline_expr, detection_frac, pathway_ribo, pathway_mito,
                      pathway_histone, pathway_hemo, src_tgt_sim, is_pert_target])
    return gate, feat


def e09_weight(feat):
    std = (feat - model_feat_mean) / model_feat_std
    raw = np.dot(np.append(std, 1.0), model_coef)
    return float(np.clip(raw, 0.0, 1.0))


# --- HepG2's real measured ground truth (on-target effect) ---------------
print("Loading HepG2 DE table (on-target rows only)...", flush=True)
de = pd.read_parquet(HEPG2_DE, columns=["target", "feature", "log2_fold_change", "p_adj",
                                          "target_ncells", "target_mean", "ref_mean"])
de_ontarget = de[de["target"] == de["feature"]].set_index("target")
print(f"  on-target rows: {len(de_ontarget)}", flush=True)

# Arc's own log2_fold_change has no pseudocount: target_mean=0.0 (complete knockdown, 882/1253
# on-target rows here) produces an arbitrary large-negative FLOOR artifact (-28 to -31, clustered,
# not a graded magnitude) rather than log2(0/ref_mean) = -inf. Checked directly - not assumed.
# Recomputed instead with the SAME pseudocount formula as the K562/RPE1 banks, on the table's own
# target_mean/ref_mean (CP10k-normalized), which bounds gracefully instead of saturating.
PSEUDOCOUNT = 0.5
de_ontarget = de_ontarget.assign(
    y_true_pseudocount=np.log((de_ontarget["target_mean"] + PSEUDOCOUNT) / (de_ontarget["ref_mean"] + PSEUDOCOUNT))
)
print(f"  recomputed on-target effect (pseudocount formula): mean={de_ontarget['y_true_pseudocount'].mean():.4f}, "
      f"std={de_ontarget['y_true_pseudocount'].std():.4f}, min={de_ontarget['y_true_pseudocount'].min():.4f}, "
      f"max={de_ontarget['y_true_pseudocount'].max():.4f}", flush=True)

overlap_pert_set = set(k562.perts) & set(rpe1.perts)
targets = sorted(set(de_ontarget.index) & (set(k562.perts) | set(rpe1.perts)))
print(f"HepG2 targets usable for evaluation (DE measured AND at least one source trained): {len(targets)}", flush=True)

rows = []
for t in targets:
    if t not in hepg2_gene_idx:
        continue
    y_true_log2fc = float(de_ontarget.loc[t, "log2_fold_change"])
    y_true = float(de_ontarget.loc[t, "y_true_pseudocount"])

    k_eff = k562.effect_row(t, [t])[0]
    r_eff = rpe1.effect_row(t, [t])[0]
    gate, feat = hepg2_gate_and_features(t, overlap_pert_set)
    if gate is None:
        continue

    row = {"target": t, "y_true": y_true, "y_true_log2fc": y_true_log2fc,
           "k562_available": not np.isnan(k_eff), "rpe1_available": not np.isnan(r_eff)}
    row["M1_k562_gated"] = (k_eff * gate * SHRINKAGE) if not np.isnan(k_eff) else np.nan
    row["M2_rpe1_gated"] = (r_eff * gate * SHRINKAGE) if not np.isnan(r_eff) else np.nan
    for alpha in ALPHAS:
        combined = combine_effect_row(np.array([k_eff]), np.array([r_eff]), alpha)[0]
        row[f"M3_alpha{alpha}"] = combined * SHRINKAGE if not np.isnan(combined) else np.nan
        row[f"M4_alpha{alpha}"] = (combined * gate * SHRINKAGE) if not np.isnan(combined) else np.nan
    w = e09_weight(feat)
    e09_combined = combine_effect_row(np.array([k_eff]), np.array([r_eff]), w)[0]
    row["E09_learned_weight"] = e09_combined * SHRINKAGE if not np.isnan(e09_combined) else np.nan
    row["E09_w"] = w
    rows.append(row)

results_df = pd.DataFrame(rows)
print(f"\nEvaluation rows built: {len(results_df)}", flush=True)
print(f"Coverage: K562 available for {results_df['k562_available'].sum()}, "
      f"RPE1 available for {results_df['rpe1_available'].sum()}, "
      f"both for {(results_df['k562_available'] & results_df['rpe1_available']).sum()}", flush=True)

model_cols = ["M1_k562_gated", "M2_rpe1_gated"] + [f"M3_alpha{a}" for a in ALPHAS] + \
             [f"M4_alpha{a}" for a in ALPHAS] + ["E09_learned_weight"]

rng = np.random.default_rng(SEED)
summary = {}
for col in model_cols:
    sub = results_df[["y_true", col]].dropna()
    n = len(sub)
    if n < 10:
        summary[col] = {"coverage": n, "note": "too few predictions"}
        continue
    yt, yp = sub["y_true"].to_numpy(), sub[col].to_numpy()
    pearson = pearsonr(yt, yp)[0] if np.std(yp) > 0 else float("nan")
    spearman = spearmanr(yt, yp)[0] if np.std(yp) > 0 else float("nan")
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    sign_agree = float(np.mean(np.sign(yt) == np.sign(yp)))

    boot_pearsons = []
    idx = np.arange(n)
    for _ in range(N_BOOTSTRAP):
        b = rng.choice(idx, size=n, replace=True)
        if np.std(yp[b]) > 0 and np.std(yt[b]) > 0:
            boot_pearsons.append(np.corrcoef(yt[b], yp[b])[0, 1])
    boot_pearsons = np.array(boot_pearsons)
    ci_lo, ci_hi = np.percentile(boot_pearsons, [2.5, 97.5])

    summary[col] = {"coverage": int(n), "pearson": float(pearson), "spearman": float(spearman),
                     "mae": mae, "rmse": rmse, "sign_agreement": sign_agree,
                     "pearson_bootstrap_ci95": [float(ci_lo), float(ci_hi)]}
    print(f"  {col:20s} n={n:4d}  Pearson={pearson:.4f} [{ci_lo:.4f},{ci_hi:.4f}]  "
          f"Spearman={spearman:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  sign_agree={sign_agree:.4f}", flush=True)

m1_pearson = summary["M1_k562_gated"]["pearson"]
m1_ci = summary["M1_k562_gated"]["pearson_bootstrap_ci95"]
print(f"\n=== Promotion decision vs M1 (K562-only E03), Pearson={m1_pearson:.4f} [{m1_ci[0]:.4f},{m1_ci[1]:.4f}] ===", flush=True)
decision = {}
for col in model_cols:
    if col == "M1_k562_gated" or "pearson" not in summary[col]:
        continue
    beats = summary[col]["pearson"] > m1_pearson
    ci_overlap = not (summary[col]["pearson_bootstrap_ci95"][1] < m1_ci[0] or summary[col]["pearson_bootstrap_ci95"][0] > m1_ci[1])
    verdict = "ACCEPTED" if (beats and not ci_overlap) else ("INCONCLUSIVE" if beats and ci_overlap else "REJECTED")
    decision[col] = verdict
    print(f"  {col:20s} Pearson={summary[col]['pearson']:.4f}  beats_M1={beats}  "
          f"CI_overlaps_M1={ci_overlap}  -> {verdict}", flush=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({
    "n_targets_evaluated": len(results_df),
    "coverage_summary": {"k562_available": int(results_df["k562_available"].sum()),
                          "rpe1_available": int(results_df["rpe1_available"].sum())},
    "metrics": summary,
    "decision_vs_M1": decision,
    "e09_learned_weight_stats": {"mean": float(results_df["E09_w"].mean()), "std": float(results_df["E09_w"].std())},
    "note": "y_true converted from HepG2's CP10k-normalized log2_fold_change to natural log for "
            "internal consistency only - NOT assumed to be on the same absolute scale as the "
            "raw-count K562/RPE1 banks (see E07's scale check, which was run on K562-vs-RPE1 raw "
            "effects, not against this CP10k-based ground truth). Pearson/Spearman/sign-agreement "
            "are scale-robust; MAE/RMSE should be read with this caveat.",
}, indent=2))
print(f"\nSaved to {OUT_PATH}")
print("DONE")
