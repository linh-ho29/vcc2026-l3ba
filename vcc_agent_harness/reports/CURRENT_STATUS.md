# Current status

## Completed
- E00: PASS - existing baseline reproduces its exported predictions bit-for-bit from the saved signature bank (no re-processing of raw K562_gwps needed)
- E01-E03: gating ablation on real D/E/F control cells (prediction-difference diagnostics, no ground truth exists for D/E/F)
- E04: direction-aware gating implemented and tested (10/10 unit tests pass)

## Verified metrics (E01-E04_metrics.json)
| Mode | Mean gate weight | % hard-zeroed | % total effect magnitude suppressed |
|---|---|---|---|
| E01 none (current submitted baseline) | 1.000 | 0.0% | 0.0% |
| E02 binary_detection | 0.54-0.56 | 44-46% | 6.6-8.8% |
| E03 combined (smooth) | 0.58-0.59 | 12-16% | 6.0-8.4% |
| E04 direction-aware | n/a (per-pert) | n/a | 5.1-7.2% |

E04 rescues 5,327-10,996 (perturbation, gene) pairs per context where E03 would
hard-zero a positive/induction-direction effect purely for low baseline
detection - absence of detection constrains downregulation, not induction.

## Blockers
RPE1 cross-context accuracy validation (the one approach that would give real
ground-truth accuracy, not just prediction-difference diagnostics) requires
downloading rpe1_normalized_singlecell_01.h5ad (8.7GB), which exceeds this
harness's own non-negotiable 5GB-without-approval rule. BLOCKED pending
explicit user approval.

## Next command (pending approval)
Download rpe1_normalized_singlecell_01.h5ad from gs://2026-vcc-llsa/train/,
then build a K562->RPE1 cross-context validation using RPE1's own real
measured perturbation responses as ground truth.

## RPE1 cross-context validation (real ground truth, not a diagnostic)
Downloaded rpe1_normalized_singlecell_01.h5ad (8.7GB, local Mac, no GCP cost).
File is z-score-like normalized (68% negative values, zero exact zeros) - NOT
raw counts. Original log-pseudocount-ratio formula was invalid for this scale;
corrected to compare via correlation (direction/rank agreement) rather than
MSE, since K562's log-fold effect and RPE1's native-units effect are not on
directly comparable absolute scales.

1,995 genes overlap (K562-trained signature AND RPE1's own real measured
on-target knockdown effect). Results:
- E01 (ungated): Pearson 0.481, Spearman 0.686
- E02 (binary detection): identical to E01 - overlap set is pre-selected for
  genes RPE1 successfully targeted, so almost all are already well-detected;
  binary threshold rarely triggers
- E03 (combined smooth) / E04 (direction-aware): Pearson 0.501, Spearman 0.696
  - real, modest improvement over ungated baseline
- E04 = E03 exactly on this test since it only covers on-target (always
  downregulation) effects - direction-awareness only matters for downstream,
  non-self-target genes, not covered here
- Sign agreement ~97% across all modes, uniformly high and less informative
  for on-target effects specifically (knockdown is trivially downregulation)

## E05: scBaseCount curated reference - COMPLETE
Verified manifest (data/manifests/scbasecount_selection.csv): 12 real SRX/ERX
accessions, 3.84GB, human, non-perturbed, well-annotated, across 6 reference
groups (k562_like, erythroid, immune, epithelial, stromal, stem_progenitor).
RPE1 explicitly excluded (would leak into its own validation set). Built
reference-group pseudobulk expression, detection fractions, study-balanced
PCA (one row per accession regardless of cell count), and 4 standard HGNC
gene-family features (ribosomal/mitochondrial/histone/hemoglobin). Biology
check passed: HBB/HBA1 detection highest in erythroid group (0.99/0.92); HBZ
and HBA1 elevated in k562_like relative to other non-erythroid groups,
consistent with K562 being erythroleukemia (matches the HBZ finding from the
E01-E04 ablation). Saved to predictions/scbasecount_reference.npz.

## E06: context-control + scBaseCount blend - REJECTED
Blended RPE1's own real context stats with its best-data-matched scBaseCount
reference group (erythroid, found via detection-fraction correlation, no
cell-type label used - correlations: erythroid 0.567, epithelial 0.502,
k562_like 0.493, stem_progenitor 0.491, stromal 0.479, immune 0.349), at
lambda in {0, 0.3, 0.6, 1.0}. lambda=0 sanity check passed exactly (all three
variants collapse to the prior E03 result, Pearson=0.5007, as required by
construction).

Negative-control results are the finding: the MISMATCHED reference (immune -
the worst-correlating group, deliberately wrong) outperforms the correctly
best-matched reference (erythroid) at every lambda>0, and beats plain E03 at
lambda=1.0 (Pearson 0.600 vs 0.501). A gene-shuffled version of the correct
reference also briefly outperforms the real one at lambda=0.6. This is
exactly the "scBaseCount only acts as generic shrinkage / noise" rejection
criterion in AGENT_PROMPT.md's evaluation protocol.

**Decision: REJECTED. Do not incorporate E06 into the production pipeline.**
E03 (context-control-only gating, lambda=0) remains the validated choice -
already regenerated as predictions/final_predictions_DEF_gwps_gated_e03.h5ad,
not yet submitted. Full metrics: vcc_agent_harness/experiments/E05_E06_metrics.json.

## Champion
E03 (combined basal-expression gating, context-control-only). No further
open experiments per the harness's planned list (E00-E06 all run).
