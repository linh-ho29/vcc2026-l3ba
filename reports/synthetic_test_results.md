# Synthetic Test Results — Hybrid MLP Perturbation Adapter

**Date:** 2026-11-02
**Scope:** Implementation and synthetic testing only. No real biological data
was used, trained on, or evaluated. All fixtures are artificial (see
`src/hybrid_adapter/synthetic.py`); their performance is NOT evidence of
biological predictive accuracy.

## Environment

- Python 3.11.13, torch 2.7.1+cpu, numpy 2.1.0, scikit-learn 1.7.1, scipy
- CPU-only execution; no accelerator dependencies
- Package installed editable (`pip install -e .`)

## Unit-test results

```
63 passed in 4.94s
```

Coverage by file:

| Test file | What it verifies |
|---|---|
| `test_forward.py` | Output dimensions; forward with GEARS + scGPT; forward without GEARS; **baseline identity before training** (`corrected == stat` within 1e-6); valid gradients; parameters change after an optimizer step; frozen input embeddings unmodified; row-permutation equivariance over multiple contexts/perturbations; NaN/Inf rejection; CPU float32 execution |
| `test_missing_inputs.py` | Missing GEARS effects produce no invalid values; `gears_only` mode without GEARS raises; `scgpt_only` mode; duplicated gene axis raises; **incompatible effect units raise**; incompatible expression units raise; unknown batch keys / mask keys raise; explicit availability masks override tensor presence; missing required keys raise; shape mismatches raise; non-binary `stat_coverage` raises |
| `test_training.py` | **Synthetic target function is learned (validation loss decreases)**; histories recorded; early stopping triggers; deterministic seeding (two identical runs match); interrupt/resume continues histories; DE-weighted loss is exactly 0 on an empty mask and stays graph-connected; default loss weights (`lambda_de = lambda_disc = 0`) |
| `test_checkpoint.py` | Checkpoint save/load reproduces predictions; manifest records architecture, dims, modalities, gene-axis requirements, units, versions, checksum; **tampered weights fail the SHA-256 checksum**; loading works with `gears`/`scgpt` imports blocked; training-state round trip |
| `test_inference.py` | **Chunked == unchunked inference in eval mode**; deterministic inference; eval-mode inference regardless of caller's train flag; H7 modes M1–M4 forward passes with correct shapes; composition-weighted recombination identity; α-gating math (zero-init heads → zero residual; invariant-head perturbation scales residual by α_eff); composition weighting changes predictions and drives α_eff to 0 at zero CompSim; subpopulation mode without phase fields raises; metrics (Pearson, Spearman, MSE, MAE, direction accuracy, discrimination diagnostic) |
| `test_h7_validation.py` | Phase scoring recovers the three phases; positive silhouette for discrete phases; homogeneity variance ratios < 1; cross-line transfer accuracy above chance; universality summary; per-phase deltas with min-cells guard; **α estimator recovers a planted α (0.7) within 0.15**; shared-vs-per-line comparison supports claim 5 under a truly shared α; sign-flip counts; TV/CompSim identities; head-to-head transfer; ablation config mapping; ordering interpretation rules |

## Example runs

`examples/synthetic_demo.py`:

```
[1] Baseline identity before training: True
[2] Training: 20 epochs, first val loss 0.0367 -> best 0.0099, early-stopped=False
[3] Checkpoint reload reproduces predictions: True (model v1.1.0)
[4] Chunked == unchunked inference: True
```

`examples/h7_synthetic_validation.py` (artificial H7-generative data,
planted α = [0.8, 0.5, 0.2]):

- Phase 1: phase silhouettes 0.30–0.42 (discrete clusters); homogeneity
  variance ratios 0.10–0.73 (all lines pass); cross-line phase-classifier
  transfer accuracy 0.89–0.93 vs chance 0.41–0.51; no missing phases.
- Phase 2: shared-α fit recovers the planted phase-0 α (0.808 vs 0.8);
  per-line α spread 0.047; claim 5 supported by AIC + spread.
- Phase 3 ablation (M0–M4, perturbation-holdout proxy): observed ordering
  **M4 > M2 > M3 > M1 > M0** — matches the H7 prediction (M4 > M2 > M1 > M0,
  with M3 ≈ M2 under a shared α). Loss differences are small (0.0142 → 0.0141)
  because the synthetic task is easy; the ordering, not the margin, is the
  signal here.

## Known limitations and incomplete functionality

1. **No real-data training or evaluation has been performed.** The adapter is
   integration-ready but unvalidated biologically.
2. **Official VCC metrics are not implemented** (PDS, MSE, NMAE, FID, Reach,
   Jaccard). They require raw-count predictions and reference single-cell
   observations; `metrics.cell_eval2_interface()` raises an informative error
   until Arc's `cell-eval2` package is available.
3. **α identifiability.** With the invariant component estimated as the
   cross-line mean, α is not identifiable by plain regression; the Phase 2b
   estimator uses a cross-line-covariance variance partition with an explicit
   equal-scale assumption (documented in `h7_validation/alpha_decomposition.py`).
   It is exact under that assumption and monotone otherwise.
4. **M3 (per-line α)** requires context identity at training time via
   `model.register_line_alpha`; at inference on unseen contexts it falls back
   to phase-shared α (M2 behaviour).
5. **Claim-5 homogeneity p-value** is diagnostic only: the shared cross-line
   mean induces residual correlation across lines, so the Bartlett chi-square
   approximation is conservative. The criterion actually used is AIC + α spread.
6. Phase scoring supports the G1/S/G2M triad (Seurat convention); M-phase and
   quiescent states are not separated by the bundled marker sets.
7. The synthetic ablation margin is small; on real data the ordering test
   requires the Phase 1/2 pass criteria to hold first.
