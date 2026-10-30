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

## E07: RPE1 signature bank - COMPLETE
Built predictions/signature_bank_rpe1.npz from gs://2026-vcc-llsa/train/
rpe1_raw_bulk_01.h5ad (raw, not z-score-normalized), identical formula/
pseudocount/clip/min-cells as the K562 bank. 2317 perturbations trained.
Scale check vs K562 on 1989 shared on-target pairs: Pearson 0.689, slope 1.05,
std ratio 1.02 - genuinely comparable, not assumed. **Headline finding: RPE1's
trained perturbations have ZERO overlap with the competition's 300-target
panel** - E08's per-perturbation combination formula cannot fire on a single
real required prediction; the fallback degenerates to K562-only there, by
construction. Full report: reports/E07_COVERAGE_NORMALIZATION.md.

## E08/E09: multi-source combination + learned weighting - COMPLETE (code)
multi_source_transfer.py: M1 (K562-only)/M2 (RPE1-only)/M3 (fixed-weight
combination, documented missing-source fallback, never zero-fills)/M4
(M3 + E03 gate). e09_fit_context_weight.py: simple ridge-regularized per-gene
weighting model, fit using ONLY the K562<->RPE1 development pair (RPE1 already
marked development, consistent with its role selecting E03), frozen before
E10 ever loads HepG2 data.

## E10: HepG2 held-out validation - ACCEPTED (research finding), NOT PROMOTED
Genuinely held-out (never touched before this step); Jurkat untouched, held in
reserve. Caught and fixed a ground-truth bug first (Arc's own log2_fold_change
has no pseudocount, produces -28 to -31 floor artifacts on complete knockdowns,
886/1253 rows affected - recomputed from target_mean/ref_mean with the
project's own pseudocount formula instead).

Combining K562+RPE1 beats K562-only with non-overlapping bootstrap CIs
(M1=0.5744 [0.532,0.615] vs M3 alpha=0.25=0.7290 [0.710,0.750], n~1250,
2000 resamples). Checked for a screen-methodology confound (K562_gwps is
genome-wide; RPE1/HepG2/Jurkat are all the same "essential" screen family) by
building a parallel K562-essential bank - confound real but partial (closes
0.060 of the 0.147 gap) and the combination benefit survives it (K562-essential
alone 0.634 vs combined-with-RPE1 0.735). E09's learned model underperformed
the naive fixed blend (0.664 vs 0.729) - not yet the generalization mechanism
needed to reach the real panel (which E07 shows has zero RPE1 overlap, so a
direct per-perturbation combination can never apply there). Full report:
reports/E10_HEPG2_VALIDATION.md.

**Not promoted.** No D/E/F or A/B/C prediction file generated or modified;
final_predictions_DEF_gwps.h5ad and final_predictions_DEF_gwps_gated_e03.h5ad
untouched. Promotion needs a working generalizable weighting mechanism, which
doesn't exist yet - E09's model needs more feature engineering or a different
functional form before it beats the naive blend, and only a generalizable
mechanism (not per-perturbation combination) can ever reach the real panel.

## Champion
E03 (combined basal-expression gating, context-control-only) remains the
production choice. E10's multi-source result is a validated research finding,
not yet a path to a better real submission - stopping here per the task's own
rule pending either a working E09 model or further direction from the user.
