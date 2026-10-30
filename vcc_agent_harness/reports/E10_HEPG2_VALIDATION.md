# E10: independent HepG2 held-out validation

Jurkat was never downloaded or touched - held in reserve. All numbers below are
from a single, genuinely held-out context (HepG2, Nadig et al. 2025 essential-
gene CRISPRi screen) never used in E07 (RPE1 bank), E08 (M1-M4 code), or E09
(context-weight model, fit using only K562<->RPE1).

## A ground-truth bug, caught and fixed before trusting any result

The first run used Arc's own `log2_fold_change` column directly as ground
truth and got **negative** Pearson across every single model variant (-0.42 to
-0.57), despite ~100% sign agreement. Investigated rather than accepted: Arc's
fold-change has no pseudocount, so the 882/1253 on-target rows with
`target_mean == 0.0` (complete knockdown) produce an arbitrary large-negative
floor artifact (-28 to -31, clustered, not a graded magnitude) instead of a
real continuous effect size. Fixed by recomputing ground truth from the
table's own `target_mean`/`ref_mean` with the identical pseudocount formula
used throughout this project (`ln((mean+0.5)/(ref+0.5))`). All results below
use the corrected ground truth.

## Headline result (production K562_gwps bank, as actually submitted)

| Model | n | Pearson | 95% bootstrap CI | Spearman | vs M1 |
|---|---|---|---|---|---|
| M1 (K562-gwps only, E03-gated) | 1246 | 0.5744 | [0.532, 0.615] | 0.679 | - |
| M2 (RPE1 only, gated) | 1240 | 0.7214 | [0.702, 0.741] | 0.850 | ACCEPTED |
| M3 alpha=0.25 (best fixed blend) | 1253 | **0.7290** | [0.710, 0.750] | 0.841 | ACCEPTED |
| M4 alpha=0.25 (blend + E03 gate) | 1253 | 0.7290 | [0.709, 0.749] | 0.845 | ACCEPTED |
| E09 learned weight | 1253 | 0.6641 | [0.634, 0.692] | 0.791 | ACCEPTED |

Every alpha from 0.0-0.75 beats M1 with non-overlapping bootstrap CIs (2000
resamples over perturbation targets). Coverage: 1246/1253 targets via K562,
1240/1253 via RPE1, 1233/1253 via both. The E03 gate is close to a no-op on
HepG2 (mean weight 0.896, zero genes hard-zeroed) - M3 and M4 are nearly
identical here, unlike on D/E/F where gating mattered more.

## A confound found and checked before trusting the magnitude

M2 (RPE1) beating M1 by such a wide margin (0.72 vs 0.57) was suspicious on
its face: `signature_bank_k562gwps.npz` comes from K562's **genome-wide**
Perturb-seq screen, while RPE1, HepG2, and Jurkat (this entire Arc dataset) are
all the same **essential-gene** screen family - a methodology match that has
nothing to do with cell-line biology. Checked directly, not assumed: built a
parallel K562-essential bank (`K562_essential_raw_bulk_01.h5ad`, same formula,
same pipeline) and re-ran the comparison.

| Source | Pearson vs HepG2 on-target |
|---|---|
| K562-gwps (genome-wide, production bank) | 0.5744 |
| K562-essential (same screen family as target) | **0.6342** |
| RPE1 (same screen family as target) | 0.7214 |

Screen-methodology match explains part of the gap (+0.060 of the 0.147 total)
but not all of it - RPE1 still beats K562-essential by 0.087 Pearson even after
controlling for screen type. Then checked whether the *combination* benefit
specifically survives this fair, methodology-matched comparison (not just
"RPE1 is better"):

| Model (K562-essential + RPE1) | Pearson |
|---|---|
| K562-essential alone | 0.6342 |
| RPE1 alone | 0.7221 |
| **combo, alpha=0.25 (best)** | **0.7345** |

Combining still beats the fair, screen-type-matched K562-only baseline by a
clear margin (0.735 vs 0.634). The combination benefit is not an artifact of
the genome-wide-vs-essential confound - it survives controlling for it.

## What this does and doesn't mean for the real VCC submission

**Established:** combining K562 and RPE1 signatures gives a real, bootstrap-
supported, confound-checked improvement over K562-only when predicting a
genuinely held-out cell line's (HepG2) own measured perturbation effects.

**Does not yet mean:** this can directly improve the actual D/E/F or A/B/C
submission. Per E07's finding, RPE1 has zero trained signatures among the
competition's 300 required perturbations - M3/M4's per-perturbation
combination formula cannot fire on a single real required prediction; the
source-availability fallback triggers 100% of the time there, degenerating to
plain K562-only. The only route from this result to a real submission
improvement is a *generalizable* weighting mechanism (E09's actual goal) that
doesn't require RPE1 to share the specific perturbed gene - and in this run,
the learned E09 model (0.664) underperformed the simple fixed blend (0.729),
so it is not yet that mechanism.

**Not promoted to D/E/F.** Per the task's own rule ("if the model does not
beat E03 on independent data, do not promote it") - here E03-equivalent (M1)
*was* beaten, clearly, by M3/M4. But promotion to the real panel specifically
requires a working generalization path that doesn't exist yet (E09's current
model is weaker than the naive fixed blend, and the fixed blend itself has
nothing to combine with on the real 300-perturbation panel). No prediction
file was generated or modified for D/E/F; `final_predictions_DEF_gwps.h5ad`
and `final_predictions_DEF_gwps_gated_e03.h5ad` are untouched.

## Caveats

- Ground truth is HepG2's CP10k-normalized pseudocount ratio; predictions are
  raw-count pseudocount ratios. Pearson/Spearman/sign-agreement are scale-
  robust; MAE/RMSE (reported in `E10_hepg2_results.json`) should be read with
  this caveat, same as E07's K562-vs-RPE1 check.
- Primary analysis is on-target (self-knockdown) effect only, for direct
  comparability with the established RPE1 cross-validation convention used in
  E01-E06. Downstream (non-self) differential-expression recovery was not
  separately evaluated here due to time - a natural next check if this track
  continues.
- Single held-out context (HepG2). Jurkat remains untouched and available as
  an independent second check if this result needs replication before any
  future promotion decision.
