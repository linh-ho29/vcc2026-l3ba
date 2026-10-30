# Independent audit of E06

Commissioned explicitly to not assume the original E06 writeup's interpretation is
correct. All numbers below come from `audit_e06.py` (new, read-only script) and
`vcc_agent_harness/experiments/E06_audit_results.json` (new output). Nothing in
E05, the original E06 run, `expression_gating.py`, or `CURRENT_STATUS.md` was
modified. No new champion was selected from the RPE1 evaluation set used here.

## 1. Erythroid as "the correct match" for RPE1 - source vs. target similarity

The original E06 selected erythroid as RPE1's best-matching scBaseCount group by
correlating RPE1's own detection-fraction profile against each of the 6 reference
groups (target similarity only - K562, the source, was never part of that
selection step).

Two problems surface on audit:

- **Erythroid is also, by a wide margin, the reference group most similar to the
  source line (K562).** Pairwise correlation among the 6 groups' detection
  fractions: erythroid-vs-k562_like = **0.861**, the single highest pairwise value
  among all 15 group pairs (next highest: stem_progenitor-vs-stromal at 0.795).
  Erythroid's win as "RPE1's best match" (0.567) over epithelial (0.502) - the
  biologically expected match, since RPE1 *is* a retinal pigment epithelial line -
  is only a 13% relative margin, set against a background where erythroid is
  already known to be unusually similar to K562. The selection procedure cannot,
  by its own construction, distinguish "resembles RPE1" from "resembles K562,
  which happens to correlate broadly with everything via shared global detection
  breadth."
- **The winner is not robust to the choice of correlation.** Switching from
  Pearson to Spearman (rank) correlation for the same target-matching step changes
  the winner entirely: stromal (0.591) and stem_progenitor (0.585) both edge out
  erythroid (0.585, tied to 3 decimal places) under rank correlation, while
  erythroid led clearly under Pearson. A selection rule that flips depending on a
  monotonic transform of the same inputs is not a stable biological finding.

**Observation, not yet distinguished:** whether erythroid's apparent RPE1-match is
real cell-identity signal or a byproduct of erythroid having a generically denser,
more K562-like detection profile cannot be resolved from this data. Both are
consistent with what was measured.

## 2. Does the immune-reference improvement reflect regularization, a bug, or real signal?

None of the three cleanly, but the audit rules two of them out more precisely
than the original writeup did:

- **Not uniform/generic shrinkage, provably.** Pearson correlation is invariant to
  multiplying every prediction by the same positive constant. The global-shrinkage
  sweep confirms this directly: constant gates of 1.0, 0.8, 0.585, 0.4, and 0.2 all
  give **exactly Pearson = 0.4813** (identical to 4 decimal places - this is a
  mathematical identity, not noise). So "it's just generic shrinkage" - the
  original rejection reason - is not the correct mechanism. Whatever is happening
  depends on how unevenly a reference distributes attenuation **across genes**,
  not on the average amount of attenuation.
- **Not an implementation bug.** Gene alignment, the lambda=0 identity, and the
  sign of every prediction were checked directly (section 3) and are correct.
- **The real mechanism: differential (gene-specific), not uniform, attenuation -
  and it tracks reference sparsity, not match quality.** Across all 6 real
  reference groups at full substitution (lambda=1.0):

  | Group | Target-match corr. | Mean gate weight | Pearson |
  |---|---|---|---|
  | erythroid | 0.567 (best match) | 0.975 | 0.473 |
  | epithelial | 0.502 | 0.958 | 0.486 |
  | k562_like | 0.493 | 0.868 | 0.498 |
  | stem_progenitor | 0.491 | 0.858 | 0.520 |
  | stromal | 0.479 | 0.758 | 0.558 |
  | immune | 0.349 (worst match) | 0.639 | 0.600 |

  Match quality and resulting Pearson are correlated at **-0.908** across these 6
  groups - strongly *negative*. Mean gate weight and Pearson move together almost
  perfectly instead: the sparser/more-dissimilar a reference's overall detection
  profile, the more it varies gene-to-gene when substituted in for these ~2,000
  already-well-detected RPE1 genes, and the better the resulting correlation on
  this one test set. That is a signature of **differential shrinkage acting as a
  generic variance-reduction device**, not of immune cells carrying biologically
  relevant information about RPE1.

## 3. Verification: gene alignment, normalization, similarity calc, sign

- **Gene alignment:** RPE1 has 1 duplicate symbol (HSPA14), the K562 bank has 2
  (TBCE, HSPA14) - both are known HGNC paralog-collision symbols, not pipeline
  artifacts, and affect a negligible fraction of each vocabulary (36591-gene
  scBaseCount reference has zero duplicates). Dict-based index lookups keep the
  last occurrence silently; immaterial here given the tiny count, but worth
  knowing.
- **Normalization is the audit's clearest finding of an actual methodological
  flaw, independent of the direction of any result.** RPE1's `detection_fraction`
  is computed as "fraction of cells with normalized value > 0" on a file already
  known to be z-score-like (not raw counts). scBaseCount's `detection_fraction` is
  "fraction of cells with nonzero raw count." These are different quantities on
  different scales, blended as if commensurable. Concretely: ACTB/GAPDH (universal
  housekeeping genes) show RPE1 z>0 fractions of 0.46/0.53 - nowhere near the
  near-ceiling raw-detection fraction (~0.86-00.97) scBaseCount's erythroid group
  shows for the same genes, but much closer to k562_like's raw-detection fraction
  (0.50/0.49). So even on genes expected to be universally detected, RPE1's proxy
  sits on a visibly different scale than most scBaseCount groups - undermining the
  premise that a linear blend of the two is a principled "regularized adjustment"
  rather than an arbitrary rescaling. (HBB/HBA1/HBZ could not be checked this way -
  they are simply absent from RPE1's measured CRISPRi panel, not a detection-zero
  result.)
- **Similarity calculation:** recomputed independently in this audit using the
  same formula as the original E06 and confirmed to reproduce the original
  group-correlation numbers exactly.
- **Sign:** 99.55% of RPE1's real measured on-target effects are negative, as
  expected for CRISPRi knockdown (consistent with the ~97% sign-agreement
  reported originally). The gate itself is a sigmoid product bounded to [0, 1] and
  cannot flip sign by construction - confirmed in code, not just assumed.
- **Lambda=0 regression check:** reproduced exactly (Pearson = 0.5007, matching
  the saved E03 result bit-for-bit in this independent run).

## 4. Multiple independently seeded permutations

The original E06 used a single shuffle seed. Repeating with 10 independent seeds
(permuting erythroid's gene axis before blending):

| Lambda | Mean Pearson | Std | Min | Max |
|---|---|---|---|---|
| 0.3 | 0.4929 | 0.0011 | 0.4911 | 0.4945 |
| 0.6 | 0.5040 | 0.0020 | 0.4990 | 0.5060 |
| 1.0 | 0.1883 | 0.0276 | 0.1520 | 0.2417 |

The single-seed result (0.4940 / 0.5061 / 0.1798) was representative, not a fluke
- variability across seeds is small relative to the effect sizes being discussed.
One clarifying pattern: at **partial** blending (lambda 0.3-0.6), shuffled and
real-ordered erythroid references are nearly indistinguishable from each other and
from plain E03 (~0.48-0.51 across the board) - the real gene-level content of the
reference barely matters until full substitution, where it matters a great deal
(collapse to ~0.19 when shuffled vs. 0.473 when correctly ordered). This says real
gene correspondence is doing *something* at lambda=1, but says little about
whether that something is transferable biology versus this one dataset's specific
noise structure.

## 5. Comparison baselines

- **Global shrinkage (constant gate, no reference, no context):** Pearson
  identical (0.4813) across every tested nonzero constant, by mathematical
  necessity (see section 2). This baseline cannot be beaten or lost to in Pearson
  terms - it only moves other metrics. Reported for completeness; it confirms the
  mechanism question more than it offers a true competing baseline.
- **Expression-matched random reference** (per gene, independently draw which of
  the 6 real groups' value to use - preserves real per-gene values, destroys
  coherent cell-identity structure), 10 seeds, lambda=1.0: mean Pearson **0.5147**,
  std 0.0024 (tight). This *beats* both plain E03 (0.5007) and the "correctly
  matched" erythroid reference (0.4726), while being clearly below immune alone
  (0.5998) and far above the fully-shuffled control (0.1883). It sits in between -
  consistent with "some generic value in injecting real per-gene external
  detection variance, regardless of which cell type it came from" being a real,
  separate, weaker effect from "this specific cell type is a meaningful match,"
  which the original E06 writeup did not distinguish.

## 6. Observations vs. hypotheses

**Established by direct measurement in this audit:**
- Erythroid's win over epithelial as RPE1's "best match" is a narrow, Pearson-only
  result that flips under Spearman - not a robust selection.
- Erythroid is the single most K562-like reference group of the 6, confounding the
  "best match to RPE1" label with "most similar to the source line."
- Across all 6 real groups, match quality to RPE1 and resulting Pearson at full
  substitution are strongly *negatively* correlated (-0.908).
- Uniform/generic shrinkage is mathematically incapable of producing the observed
  Pearson changes - ruled out as the mechanism, contradicting the original
  writeup's framing.
- RPE1's detection-fraction proxy (z-score-based) and scBaseCount's (raw-count
  based) sit on visibly different scales even for universal housekeeping genes -
  a genuine normalization mismatch in the blend, independent of any result.
- The 10-seed shuffled and random-reference results are stable (low variance),
  not single-seed flukes.
- Random expression-matched mixing outperforms the "correct" cell-type match.

**Hypotheses, not established here:**
- Whether immune's gate pattern reflects any transferable biological information
  at all, versus being a dataset-specific artifact of this one ~2,000-gene,
  single-RPE1-dataset evaluation (no bootstrap/independent replicate was run - a
  real limitation of this audit, not papered over).
- Whether a properly scale-matched, non-z-score detection measure for RPE1 would
  change which reference group is selected as "best match."

**What invalidates the original E06 interpretation specifically:**
- The claim that the mismatched-reference result was explainable as "generic
  shrinkage/noise" is wrong in a precise, provable sense (shrinkage can't move
  Pearson at all). The *recommendation* to reject E06 survives, but on different
  grounds: a fragile, confounded match-selection criterion, a measured negative
  correlation between match quality and performance across all 6 real references,
  and a normalization mismatch between the two blended quantities - not generic
  shrinkage.

## Bottom line

Original recommendation (reject E06, keep E03 as champion) is **not reversed**,
but was reached for an **imprecise reason**. The audit replaces "it's just
shrinkage" with a more specific and better-supported finding: the match-selection
step is confounded and non-robust, performance across real reference groups runs
*opposite* to match quality, and the two blended quantities are not on a
comparable scale to begin with. E03 (context-control-only gating) remains the
validated choice; nothing here changes that production recommendation.
