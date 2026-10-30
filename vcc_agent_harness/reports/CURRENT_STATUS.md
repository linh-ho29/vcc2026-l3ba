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
