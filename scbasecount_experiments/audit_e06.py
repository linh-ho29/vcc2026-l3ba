"""
Independent audit of E06, per explicit user request not to assume the
original interpretation is correct. Read-only with respect to all existing
artifacts - writes only to new files (E06_audit_results.json, E06_AUDIT.md).
Does not re-optimize or select a new champion from this same RPE1 eval set;
every number here is diagnostic.
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from expression_gating import GateConfig, compute_context_gate_stats, compute_gate_weights
from scbasecount_blend import best_and_worst_matching_group, reference_gene_stats, blend, compute_group_percentile

REPO_ROOT = Path(__file__).parent.parent
RPE1_PATH = REPO_ROOT / "data" / "rpe1_normalized_singlecell_01.h5ad"
PRED_DIR = REPO_ROOT / "predictions"
REF_PATH = PRED_DIR / "scbasecount_reference.npz"
OUT_JSON = REPO_ROOT / "vcc_agent_harness" / "experiments" / "E06_audit_results.json"

SEED = 2026
SHRINKAGE = 0.65
MIN_CELLS_PER_PERT = 15
CONTROL_LABEL = "non-targeting"
N_PERMUTATION_SEEDS = 10

bank = np.load(PRED_DIR / "signature_bank_k562gwps.npz", allow_pickle=True)
source_genes = bank["genes"].astype(str)
source_perts = bank["perts"].astype(str)
source_effects = bank["effects"]
source_gene_index = {g: i for i, g in enumerate(source_genes)}
source_pert_index = {p: i for i, p in enumerate(source_perts)}

ref = np.load(REF_PATH, allow_pickle=True)
ref_groups = ref["reference_groups"].astype(str)
ref_symbols = ref["gene_symbols"].astype(str)
ref_symbol_index = {s: i for i, s in enumerate(ref_symbols)}

print("Loading RPE1 (backed)...", flush=True)
adata = ad.read_h5ad(RPE1_PATH, backed="r")
labels = adata.obs["gene"].astype(str).to_numpy()
rpe1_gene_symbols = adata.var["gene_name"].astype(str).to_numpy()
rpe1_gene_index = {g: i for i, g in enumerate(rpe1_gene_symbols)}

audit = {}

# ---------------------------------------------------------------------------
# 3a. Gene alignment audit
# ---------------------------------------------------------------------------
print("\n=== Gene alignment audit ===", flush=True)
dup_rpe1 = len(rpe1_gene_symbols) - len(set(rpe1_gene_symbols))
dup_ref = len(ref_symbols) - len(set(ref_symbols))
dup_bank = len(source_genes) - len(set(source_genes))
print(f"  RPE1 var gene_name duplicates: {dup_rpe1} (of {len(rpe1_gene_symbols)})", flush=True)
print(f"  scBaseCount reference gene_symbols duplicates: {dup_ref} (of {len(ref_symbols)})", flush=True)
print(f"  K562_gwps bank gene duplicates: {dup_bank} (of {len(source_genes)}) "
      f"- {[g for g in set(source_genes) if list(source_genes).count(g) > 1]}", flush=True)
audit["gene_alignment"] = {
    "rpe1_duplicate_symbols": int(dup_rpe1),
    "scbasecount_reference_duplicate_symbols": int(dup_ref),
    "k562_bank_duplicate_symbols": int(dup_bank),
    "k562_bank_duplicate_genes_list": [g for g in set(source_genes) if list(source_genes).count(g) > 1],
    "note": "dict-based index lookups ({g: i for i, g in enumerate(...)}) silently keep only the LAST "
            "occurrence of a duplicated symbol. 2/8248 K562 bank genes affected (TBCE, HSPA14 - both "
            "known HGNC paralog-collision symbols, not a pipeline artifact). RPE1 and scBaseCount "
            "reference vocabularies have zero duplicates.",
}

# ---------------------------------------------------------------------------
# 1. Source (K562) vs target (RPE1) reference similarity
# ---------------------------------------------------------------------------
print("\n=== Source vs target reference-similarity audit ===", flush=True)
ctrl_idx = np.flatnonzero(labels == CONTROL_LABEL)
print(f"RPE1 control cells: {len(ctrl_idx)}", flush=True)


def mean_rows_backed(adata, indices, block=512):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    out = np.zeros(adata.n_vars, dtype=np.float64)
    if not len(indices):
        return out
    for i in range(0, len(indices), block):
        chunk = adata.X[indices[i:i + block], :]
        out += (np.asarray(chunk.sum(axis=0)).ravel() if sp.issparse(chunk)
                else np.asarray(chunk, dtype=np.float64).sum(axis=0))
    return out / len(indices)


class _ControlView:
    def __init__(self, adata, ctrl_idx):
        self.adata, self.ctrl_idx, self.shape = adata, ctrl_idx, (len(ctrl_idx), adata.n_vars)
    def __getitem__(self, key):
        idx = self.ctrl_idx[key] if isinstance(key, slice) else self.ctrl_idx[key]
        return self.adata.X[np.sort(idx), :]


rpe1_baseline = mean_rows_backed(adata, ctrl_idx)
rpe1_gate_stats = compute_context_gate_stats(_ControlView(adata, ctrl_idx), block=512)

# Pairwise correlation among all 6 reference groups themselves (own shared vocab - no RPE1 needed).
group_corr = np.corrcoef(ref["group_detection_fraction"])
k562_idx = list(ref_groups).index("k562_like")
group_vs_k562 = {g: float(group_corr[i, k562_idx]) for i, g in enumerate(ref_groups)}
print(f"  Each reference group's correlation with k562_like (SOURCE proxy): "
      f"{json.dumps({k: round(v, 4) for k, v in group_vs_k562.items()})}", flush=True)

best_group, worst_group, target_corrs, shared_genes = best_and_worst_matching_group(
    rpe1_gate_stats["detection_fraction"], rpe1_gene_symbols, ref)
print(f"  Each reference group's correlation with RPE1 (TARGET), whole transcriptome "
      f"({len(shared_genes)} shared genes): {json.dumps({k: round(v, 4) for k, v in target_corrs.items()})}", flush=True)
print(f"  erythroid-vs-k562_like (source similarity): {group_vs_k562['erythroid']:.4f}", flush=True)
print(f"  erythroid-vs-RPE1 (target similarity, the original 'best match' basis): {target_corrs['erythroid']:.4f}", flush=True)
print(f"  epithelial-vs-RPE1 (biologically expected best match - RPE1 IS a retinal pigment epithelial line): "
      f"{target_corrs['epithelial']:.4f}", flush=True)

# Spearman (rank) version of the same target-matching correlation, to check whether the
# raw-value ranking survives a monotonic-scale-invariant transform.
rank_corrs = {}
for gi, g in enumerate(ref_groups):
    ref_cols = [ref_symbol_index[s] for s in shared_genes]
    ctx_vals = rpe1_gate_stats["detection_fraction"][[rpe1_gene_index[s] for s in shared_genes]]
    ref_vals = ref["group_detection_fraction"][gi, ref_cols]
    rho, _ = spearmanr(ctx_vals, ref_vals)
    rank_corrs[g] = float(rho)
print(f"  Spearman (rank) version of target-match: {json.dumps({k: round(v, 4) for k, v in rank_corrs.items()})}", flush=True)
rank_best = max(rank_corrs, key=rank_corrs.get)
print(f"  Spearman-based best match: {rank_best} (Pearson-based was: {best_group})", flush=True)

audit["source_vs_target_similarity"] = {
    "group_vs_k562_source_correlation": group_vs_k562,
    "group_vs_rpe1_target_correlation_pearson": target_corrs,
    "group_vs_rpe1_target_correlation_spearman": rank_corrs,
    "pearson_based_best_match": best_group,
    "spearman_based_best_match": rank_best,
    "erythroid_is_most_k562_like_among_all_6_groups": bool(
        group_vs_k562["erythroid"] == max(group_vs_k562.values())),
    "note": "erythroid has the HIGHEST pairwise correlation with k562_like among all 6 reference "
            "groups (0.861, next highest is 0.795 for stem_progenitor-stromal) - i.e. erythroid is "
            "distinctly the group most similar to the SOURCE line, not just a target-specific match. "
            "Its win over epithelial as RPE1's 'best target match' (0.567 vs 0.502) is a narrow "
            "margin (~13% relative) against a background where ALL 6 groups correlate with RPE1 in "
            "a fairly narrow 0.35-0.57 band, consistent with a shared global detection-breadth factor "
            "dominating the correlation rather than cell-type-specific signal.",
}

# ---------------------------------------------------------------------------
# 3b. Normalization sanity: does z-score-based "detection_fraction" on RPE1
# behave sensibly against raw-count-based "detection_fraction" on scBaseCount?
# ---------------------------------------------------------------------------
print("\n=== Normalization / detection_fraction sanity audit ===", flush=True)
check_genes = ["ACTB", "GAPDH", "HBB", "HBA1", "HBZ"]
k562_like_gi = list(ref_groups).index("k562_like")
erythroid_gi = list(ref_groups).index("erythroid")
sanity = {}
for g in check_genes:
    rpe1_val = float(rpe1_gate_stats["detection_fraction"][rpe1_gene_index[g]]) if g in rpe1_gene_index else None
    ref_val_k562 = float(ref["group_detection_fraction"][k562_like_gi, ref_symbol_index[g]]) if g in ref_symbol_index else None
    ref_val_eryth = float(ref["group_detection_fraction"][erythroid_gi, ref_symbol_index[g]]) if g in ref_symbol_index else None
    sanity[g] = {"rpe1_zscore_based": rpe1_val, "scbasecount_k562like_raw_based": ref_val_k562,
                 "scbasecount_erythroid_raw_based": ref_val_eryth}
    print(f"  {g}: RPE1(z>0 frac)={rpe1_val}, scBaseCount-k562_like(raw>0 frac)={ref_val_k562}, "
          f"scBaseCount-erythroid(raw>0 frac)={ref_val_eryth}", flush=True)
audit["normalization_sanity"] = {
    "gene_level_check": sanity,
    "note": "RPE1's detection_fraction is computed as fraction of cells with normalized_value > 0, "
            "on a file previously confirmed z-score-like (68% negative values, zero exact zeros) - "
            "NOT a fraction of cells with nonzero RAW count, which is what scBaseCount's "
            "detection_fraction measures. These are DIFFERENT definitions computed on DIFFERENT "
            "scales, blended together in E06 as if commensurable. ACTB/GAPDH (housekeeping, "
            "expected near-universal raw detection) and HBB/HBA1/HBZ (blood-specific, expected "
            "near-absent raw detection in a retinal pigment epithelial line) are reported above to "
            "assess whether RPE1's z-score-based proxy still tracks the expected direction despite "
            "the scale mismatch.",
}

# ---------------------------------------------------------------------------
# Build overlap_perts set and the per-gene inputs needed for sections 3c/4/5
# ---------------------------------------------------------------------------
non_ctrl_labels = labels[labels != CONTROL_LABEL]
groups_, counts_ = np.unique(non_ctrl_labels, return_counts=True)
groups_, counts_ = groups_[counts_ >= MIN_CELLS_PER_PERT], counts_[counts_ >= MIN_CELLS_PER_PERT]
overlap_perts = [g for g in groups_ if g in source_pert_index and g in rpe1_gene_index]
print(f"\nOverlap set: {len(overlap_perts)} genes", flush=True)

per_gene = []  # (y_true, k562_effect, context_det, context_pct, gene_symbol)
for pert in overlap_perts:
    pert_idx = np.flatnonzero(labels == pert)
    rpe1_mean = mean_rows_backed(adata, pert_idx)
    j = rpe1_gene_index[pert]
    y_true = float(rpe1_mean[j] - rpe1_baseline[j])
    k562_j = source_gene_index.get(pert, None)
    k562_effect = float(source_effects[source_pert_index[pert]][k562_j]) if k562_j is not None else 0.0
    per_gene.append((y_true, k562_effect, float(rpe1_gate_stats["detection_fraction"][j]),
                      float(rpe1_gate_stats["percentile"][j]), pert))
adata.file.close()

y_true_arr = np.array([p[0] for p in per_gene])
frac_negative = float(np.mean(y_true_arr < 0))
print(f"\nSign sanity: {frac_negative:.4f} of RPE1's real measured on-target effects are negative "
      f"(CRISPRi knockdown should decrease expression)", flush=True)
audit["sign_sanity"] = {"fraction_y_true_negative": frac_negative}


def predict(gate_func, lam=None):
    preds = []
    for y_true, k562_effect, context_det, context_pct, pert in per_gene:
        gate = gate_func(context_det, context_pct, pert)
        preds.append(k562_effect * gate * SHRINKAGE)
    return np.array(preds)


def score(preds):
    if np.std(preds) > 0 and np.std(y_true_arr) > 0:
        pearson, _ = pearsonr(y_true_arr, preds)
        spearman, _ = spearmanr(y_true_arr, preds)
    else:
        pearson, spearman = float("nan"), float("nan")
    return float(pearson), float(spearman), float(np.mean(np.sign(y_true_arr) == np.sign(preds)))


def gate_from_stats(det, pct):
    stats_j = {"detection_fraction": np.array([det]), "percentile": np.array([pct]), "baseline_mean": np.array([0.0])}
    return float(compute_gate_weights(stats_j, GateConfig(mode="combined"))[0])


# E03 baseline (lam=0, context-only) - regression check against the already-saved E06 result.
e03_preds = predict(lambda det, pct, pert: gate_from_stats(det, pct))
e03_pearson, e03_spearman, e03_sign = score(e03_preds)
print(f"\nE03 regression check: Pearson={e03_pearson:.4f} (prior saved value: 0.5007)", flush=True)
audit["e03_regression_check"] = {"pearson": e03_pearson, "spearman": e03_spearman, "sign_agreement": e03_sign}

# ---------------------------------------------------------------------------
# 3c. All 6 real reference groups at lam=1.0 (not just best/worst) - tests
# whether performance is monotonic in match quality or essentially unrelated.
# ---------------------------------------------------------------------------
print("\n=== All 6 reference groups at lambda=1.0 (full substitution) ===", flush=True)
all_groups_lam1 = {}
mean_gate_by_group = {}
for g in ref_groups:
    preds, gates = [], []
    for y_true, k562_effect, context_det, context_pct, pert in per_gene:
        rs = reference_gene_stats(ref, g, pert, shuffle_seed=None)
        det, pct = blend(context_det, context_pct, *(rs if rs else (None, None)), 1.0)
        gate = gate_from_stats(det, pct)
        gates.append(gate)
        preds.append(k562_effect * gate * SHRINKAGE)
    pearson, spearman, sign_agree = score(np.array(preds))
    all_groups_lam1[g] = {"pearson": pearson, "spearman": spearman, "sign_agreement": sign_agree,
                           "mean_gate_weight": float(np.mean(gates))}
    print(f"  {g:16s} target_corr_with_rpe1={target_corrs[g]:.4f}  Pearson={pearson:.4f}  "
          f"mean_gate_weight={np.mean(gates):.4f}", flush=True)
audit["all_6_groups_at_lambda1"] = all_groups_lam1

match_quality = [target_corrs[g] for g in ref_groups]
perf = [all_groups_lam1[g]["pearson"] for g in ref_groups]
quality_perf_corr = float(np.corrcoef(match_quality, perf)[0, 1])
print(f"\n  Correlation between (match quality to RPE1) and (resulting Pearson at lam=1.0) "
      f"across the 6 groups: {quality_perf_corr:.4f}", flush=True)
print("  (If genuinely informative cell-type matching drove the result, this should be positive; "
      "a near-zero or negative value means performance is unrelated to, or inversely related to, "
      "how well the reference matches RPE1.)", flush=True)
audit["match_quality_vs_performance_correlation"] = quality_perf_corr

# ---------------------------------------------------------------------------
# 4. Multiple independently seeded gene permutations (not just one shuffle)
# ---------------------------------------------------------------------------
print(f"\n=== Multi-seed shuffled-reference negative control ({N_PERMUTATION_SEEDS} seeds) ===", flush=True)
shuffle_results = {lam: [] for lam in [0.3, 0.6, 1.0]}
for seed in range(N_PERMUTATION_SEEDS):
    for lam in [0.3, 0.6, 1.0]:
        preds = []
        for y_true, k562_effect, context_det, context_pct, pert in per_gene:
            rs = reference_gene_stats(ref, "erythroid", pert, shuffle_seed=1000 + seed)
            det, pct = blend(context_det, context_pct, *(rs if rs else (None, None)), lam)
            gate = gate_from_stats(det, pct)
            preds.append(k562_effect * gate * SHRINKAGE)
        pearson, _, _ = score(np.array(preds))
        shuffle_results[lam].append(pearson)

for lam, vals in shuffle_results.items():
    vals = np.array(vals)
    print(f"  lam={lam}: mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  "
          f"max={vals.max():.4f}  (single-seed-2026 result was: "
          f"{'0.4940' if lam==0.3 else '0.5061' if lam==0.6 else '0.1798'})", flush=True)
audit["shuffled_reference_multiseed"] = {str(lam): {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                                      "min": float(np.min(v)), "max": float(np.max(v)),
                                                      "all_seeds": [float(x) for x in v]}
                                          for lam, v in shuffle_results.items()}

# ---------------------------------------------------------------------------
# 5a. Expression-matched random reference: per-gene, randomly pick which of
# the 6 REAL groups' value to use (mixing real values across groups per
# gene) - preserves the true empirical per-gene value distribution while
# destroying any single group's internal cell-type coherence.
# ---------------------------------------------------------------------------
print(f"\n=== Expression-matched random reference ({N_PERMUTATION_SEEDS} seeds, lam=1.0) ===", flush=True)
exprmatch_results = []
for seed in range(N_PERMUTATION_SEEDS):
    rng = np.random.default_rng(2000 + seed)
    preds = []
    for y_true, k562_effect, context_det, context_pct, pert in per_gene:
        chosen_group = ref_groups[rng.integers(0, len(ref_groups))]
        rs = reference_gene_stats(ref, chosen_group, pert, shuffle_seed=None)
        det, pct = blend(context_det, context_pct, *(rs if rs else (None, None)), 1.0)
        gate = gate_from_stats(det, pct)
        preds.append(k562_effect * gate * SHRINKAGE)
    pearson, _, _ = score(np.array(preds))
    exprmatch_results.append(pearson)
exprmatch_results = np.array(exprmatch_results)
print(f"  mean={exprmatch_results.mean():.4f}  std={exprmatch_results.std():.4f}  "
      f"min={exprmatch_results.min():.4f}  max={exprmatch_results.max():.4f}", flush=True)
audit["expression_matched_random_reference"] = {"mean": float(exprmatch_results.mean()),
                                                  "std": float(exprmatch_results.std()),
                                                  "all_seeds": [float(x) for x in exprmatch_results]}

# ---------------------------------------------------------------------------
# 5b. Simple global-shrinkage baseline: constant gate, no gene-specific info.
# ---------------------------------------------------------------------------
print("\n=== Simple global-shrinkage baseline (constant gate, no reference, no context) ===", flush=True)
shrink_results = {}
for const_gate in [1.0, 0.8, 0.585, 0.4, 0.2, 0.0]:
    preds = np.array([k562_effect * const_gate * SHRINKAGE for y_true, k562_effect, _, _, _ in per_gene])
    pearson, spearman, sign_agree = score(preds)
    shrink_results[const_gate] = {"pearson": pearson, "spearman": spearman, "sign_agreement": sign_agree}
    print(f"  constant_gate={const_gate}: Pearson={pearson:.4f}  Spearman={spearman:.4f}  "
          f"sign_agreement={sign_agree:.4f}", flush=True)
audit["global_shrinkage_baseline"] = shrink_results

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(audit, indent=2))
print(f"\nSaved full audit to {OUT_JSON}")
print("DONE")
