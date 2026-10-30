# E07: RPE1 signature bank - coverage and normalization report

## Headline finding - read before E08

**RPE1's trained perturbations have zero overlap with the competition's actual
300-target panel (0/300).** For every one of the 300 perturbations the real D/E/F
(and A/B/C) submissions must predict, RPE1 has no measured CRISPRi effect at all -
its essential-gene screen targeted a materially different 2,393-gene set than the
panel's 300 genes. This means E08's literal combination formula
(`Δ = α·Δ_K562 + (1-α)·Δ_RPE1`, "for a perturbation measured in both source
contexts") **can never fire for a single one of the real required predictions** -
the documented source-availability fallback will trigger 100% of the time on the
actual panel, which means M3/M4 degenerate to M1 (plain K562-only E03) on every
real prediction, by construction, not by a weak result.

This does not block E10's actual question (does K562+RPE1 generalize better to a
held-out line than K562 alone, evaluated on that line's *own* measured
perturbations) - that's a valid, well-posed question regardless of panel overlap.
It does mean the only route by which this research track could ever change a real
VCC submission is a *generalizable* mechanism (a learned weighting/gating
function, E09's actual target) - not direct per-perturbation signature
combination, which is moot for the real deliverable from the start.

## Source and build

- `data/rpe1_raw_bulk_01.h5ad` (gs://2026-vcc-llsa/train/, sha256
  `603c655f...d2e2db2`, full hash in `signature_bank_rpe1_manifest.json`) -
  guide-level raw pseudobulk, 2679 rows x 8749 genes. NOT the z-score-normalized
  `rpe1_normalized_singlecell_01.h5ad` used for the earlier RPE1 cross-validation.
- Same effect formula, pseudocount (0.5), clip bound (1.0), and minimum-cell
  threshold (15) as `signature_bank_k562gwps.npz`. 135/2393 gene targets had
  multiple guide rows (up to 3), aggregated by cell-count-weighted mean, matching
  how the K562 pipeline pools cells regardless of guide.
- 2317 perturbations trained (76 omitted for <15 total cells).

## Coverage

| | genes | perturbations |
|---|---|---|
| K562 bank | 8,246 | 9,831 |
| RPE1 bank | 8,748 | 2,317 |
| Competition panel | 18,533 (output) | 300 (required) |

| Overlap | count | |
|---|---|---|
| K562 genes ∩ RPE1 genes | 7,093 | 86.0% of K562, 81.1% of RPE1 |
| K562 ∩ RPE1 ∩ panel genes | 6,714 | |
| K562 perts ∩ RPE1 perts (dev-usable) | 2,308 | |
| K562 perts ∩ panel | 275 / 300 | 91.7% |
| **RPE1 perts ∩ panel** | **0 / 300** | **0.0%** |
| Panel perts with neither source | 25 / 300 | |

## Normalization check - good news

Applying the *identical* raw-count pseudocount-ratio formula to both cell lines
(rather than mixing bases, e.g. K562 raw vs. RPE1 CP10k-normalized) produces
genuinely comparable scales. Checked directly on the 1,989 perturbations with a
valid on-target (self-knockdown) effect in both banks:

- K562 on-target effect: mean -0.576, std 0.291
- RPE1 on-target effect: mean -0.656, std 0.298
- Pearson correlation: **0.689**
- 99.3% agree in sign (both show knockdown, as expected for CRISPRi on-target)
- Scale ratio (RPE1 std / K562 std): **1.024**
- Linear regression slope (RPE1 ~ K562, through origin): **1.051**

This is a real, checked result, not an assumption: the two banks' effect
magnitudes are close to 1:1 comparable. This specifically validates building
RPE1's bank from raw counts with the same formula, rather than from Arc's
CP10k-normalized `log2_fold_change` table (whose basis was confirmed different
and not pre-verified as comparable).

## Missing-value handling

- Perturbation-level: 76 gene targets omitted entirely for <15 total cells - not
  represented as zero-effect.
- Within-vocabulary gene-level: a NaN-on-untested mechanism was implemented
  (mirrors Arc's own `tested.npy`), but checked directly and found zero
  qualifying cases - this file's 8,749-gene vocabulary was already pre-filtered
  upstream (`var['in_matrix']`; min control_mean = 0.055, right at the
  threshold). The real missing-value axis for this bank is **vocabulary
  coverage**: effects exist for only 8,749 of the competition's 18,533 output
  genes. Any downstream combination must treat genes outside this vocabulary as
  missing, never zero-fill them.

## Note on Arc's PIE benchmark splits (per your recommendation)

Arc's dataset card states this collection is the test set of `replogle_xdataset`
(cross-dataset/cross-cell-line generalization) and the train+test data of
`replogle_wdataset` (within-dataset). This confirms Arc treats "held-out cell
line" as a first-class evaluation axis, matching E10's design (K562+RPE1 as
development, one of HepG2/Jurkat genuinely held out). Not reusing their trained
models or their 6,642-gene preprocessed vocabulary - E10 will use their raw `de/`
parquet tables only as a source of real measured effects for the held-out line,
and keep the full 18,533-gene VCC output vocabulary throughout.

## Next

Proceeding to a minimal, functional E08 (M1-M4) sized to support E10 - the
milestone you flagged as most valuable - rather than exhaustively polishing E08
in isolation, given M3/M4's combination step is inert on the actual panel per
the finding above.
