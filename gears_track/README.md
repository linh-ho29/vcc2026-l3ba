# GEARS track - handoff from local training (3 epochs, paused)

This is a **separate, standalone GEARS run** (plain `gears`/`torch_geometric`
library, GEARS_Model class) - not related to the `PertAdapt` submodule
(AIDOCell/scFoundation-based), and not the CPU statistical baseline that is
the actual production pipeline for this project (see root `README.md`). This
track was paused earlier in the project for GPU infrastructure reasons; these
are the artifacts from before that pause, handed off for future resumption.

## What's here vs. what's not

Copied from the original handoff (`~/Downloads/gears_handoff/`), verified
bit-identical via `manifest_checksums.json`:
- `checkpoints/final_checkpoint.pt` - model weights only (**no optimizer
  state** - resuming will need a freshly-initialized optimizer, losing any
  Adam momentum from the original run).
- `checkpoints/provenance.json` - architecture, dataset, splits, seeds,
  determinism characterization. Read this first.
- `embeddings/*.npy` + `*_index.csv` - gene and perturbation embeddings
  (pre-graph and post-graph/fused), 8,248 x 64.
- `evaluation/metrics.json`, `per_perturbation_metrics.csv`,
  `training_history.csv` - honest, modest results (see below).
- `validation/integrity_report.json` - shape/finiteness checks, leakage
  check (passed - no held-out perturbation outcomes used as input),
  determinism check.

**Not copied - two files referenced in `manifest_checksums.json` were never
in the handoff folder to begin with** (not a copying error on this end -
checked directly, they're simply absent from `~/Downloads/gears_handoff/`):
`manifest.json` and `README.md`. Whatever top-level description/manifest
they would have contained isn't available; this README is reconstructed from
`provenance.json` + `integrity_report.json` + `metrics.json` instead.

**Not pushed to git - too large, consistent with how this repo treats large
prediction arrays everywhere else** (see root `README.md`'s note on
`data/`/`predictions/` being gitignored): `predictions/perturbation_delta.npz`
(188.6 MB - exceeds GitHub's 100 MB hard per-file limit, would need Git LFS)
and `predictions/validation_predictions.npz` (37.8 MB). Both remain only at
`~/Downloads/gears_handoff/predictions/` locally. If you want these in git,
say so explicitly - it would mean setting up Git LFS for this repo, which
hasn't been done yet.

## Current state (honest numbers, from `evaluation/metrics.json`)

Test split: `simulation_seed42_test_unseen_single`, 1,905 perturbations,
**zero gene overlap between train and test targets** (genuinely unseen-gene
regime, not just unseen-combination).

| Metric | Mean | Median |
|---|---|---|
| Pearson (full delta profile) | 0.028 | 0.020 |
| Pearson (DE genes only) | 0.177 | 0.202 |
| dir20 (top-20 DE direction hit rate) | 0.286 | 0.300 |
| MSE (DE genes) | 0.506 | 0.487 |

Weak on the full profile, modest on DE genes - consistent with a model that's
had only 3 total epochs (1 original + 2 resumed, ~7,332 optimizer steps) on a
small subsample (10 cells/pert + 2,000 ctrl = 78,190 cells, 8,248-gene HVG
panel from Replogle 2022 K562 gwps). Not yet competitive with the statistical
baseline's validated numbers - this is expected for 3 epochs and isn't a
claim of readiness, just where training was paused.

One data-quality note already caught in `integrity_report.json`: 2 duplicate
gene symbols in the panel (recorded, not filled/deduped).

## Determinism caveat (important before trusting any single prediction run)

GEARS's own `predict()` resamples control cells per condition
(`gears/utils.py:283`) - re-running prediction on the same checkpoint and
condition is **not bitwise reproducible** (up to ~0.38 log1p units of
difference on the worst genes in the 5-condition check in
`provenance.json`). This is a property of the upstream GEARS library, not a
bug introduced here. Averaging multiple prediction draws, or using the full
2,000-cell control mean as basal instead of a 300-cell resample, would reduce
this noise floor - worth doing before any serious comparison against the
statistical baseline.

## To resume training

1. Re-create the environment: `torch 2.1.0+cpu`, `torch_geometric 2.4.0` (see
   `checkpoints/provenance.json` for exact architecture: `hidden_size=64`,
   1 co-expression GNN layer, 1 GO GNN layer, cross-gene MLP decoder).
2. Load `checkpoints/final_checkpoint.pt` as a starting point - weights only,
   initialize a fresh optimizer.
3. Same data split and seeds are in `provenance.json`
   (`split_seed=42`, `extraction_seed=42`) - no explicit training-loop seed
   was set in the original run, so exact reproducibility of further training
   dynamics isn't guaranteed even with the same code.
4. `evaluation/training_history.csv` has the per-step loss log for the
   original run's first 2 epochs; the resumed run's step-level log was lost
   (only epoch-end summaries were reconstructed) - a good argument for
   logging more defensively this time around.

## Future direction noted by the user (not yet implemented)

Idea for a possible adapter: use the **statistical signature-transfer
baseline** (this repo's actual production approach) for predicting
**perturbed-cell** outcomes, and use **GEARS/the GNN** specifically for
modeling **unperturbed/control-cell** state - i.e. split responsibility by
cell state rather than trying to have one model do both. Not scoped or
started - recorded here so it isn't lost, and so a well-reasoned starting
point (the checkpoint and embeddings above) is sitting next to it when work
on it begins.
