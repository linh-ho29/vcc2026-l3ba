# Track J: Jurkat GEARS with Biology-Informed, Response-Blind Target Sampling

VCC 2026 Research Track J experiment: train a GEARS model on the Nadig et al.
(2025) Jurkat essential-gene CRISPRi Perturb-seq dataset using **biology-informed,
response-blind target sampling**, and test whether sampling across diverse
GO/network neighborhoods improves prediction of unseen perturbation responses
relative to high-effect selection.

## Headline result

On 60 held-out targets sampled blind to response magnitude, stratified across
GO functional categories and GO-graph degree:

| Metric (mean over 60 test targets) | GEARS | Mean-shift baseline | Control-only |
|---|---|---|---|
| Pearson delta (all genes) | **0.073** | **0.239** | undefined (zero delta) |
| Spearman delta | 0.052 | — | — |
| Effect MSE | 0.0102 | **0.0063** | 0.0067 |
| Top-20 DE direction accuracy | 0.505 (chance) | 0.530 | 0.0 |
| Magnitude calibration r | 0.299 | — | — |
| Magnitude ratio (median pred/true) | 0.266 | — | — |

**GEARS underperforms the naive mean-shift baseline on this split** (better
Pearson on only 7/60 targets, and worse MSE). Predicted deltas are shrunk ~4x
(median pred/true |delta| ratio 0.266) and direction calls on the top-20 DE
genes are at chance. Under response-blind, biology-diverse sampling, the model
collapses toward predicting near-average responses: the target-specific signal
it can extract from GO/graph features does not beat the global CRISPRi
shift-plus-shrinkage strategy. This is the central Track J finding: the
regime of the holdout, not just the model, determines whether learned
perturbation response models beat one-line baselines.

Where signal survives (see `where_signal_survives.png`): gene-level sign
accuracy rises with true effect size (0.51 all genes -> 0.56 top-1% ->
0.64 top-0.1% |delta|); per-target Pearson tracks response magnitude
(r = +0.49); target-level magnitude ordering is weak but real (r = +0.31,
p = 0.016). GEARS never identifies which genes are DE (top-20 set overlap
mean 4.7%). The 7 targets where GEARS beats the baseline are near-zero
correlation cases, not learned signal.

Caveats before over-interpreting:
- Single seed (42); no error bars.
- 55 sequencing batches are documented but not corrected.
- The GO kNN graph is dense: 59/60 test targets are 1 hop from a train target,
  so graph-distance stratification has almost no spread.
- pearson_delta is computed over all 8,882 genes; on DE genes only
  (de_delta_corr_top20) GEARS is also weak (0.048), so the conclusion is not an
  artifact of the all-gene averaging.

## Under-training vs structural shrinkage test

To test whether the ~4x magnitude shrinkage is under-training rather than
structural, two retraining arms were run on the identical split/seed/recipe,
with the LR schedule stretched (StepLR step=4, gamma=0.5) so learning continues
past epoch 8 (the official step=1 schedule freezes LR at ~4e-6 by epoch 8):

| Arm | pearson_delta | magnitude_ratio_med | effect_mse | top20 dir acc | best val mse_de |
|---|---|---|---|---|---|
| baseline h32, 8 ep (official schedule) | 0.073 | 0.266 | 0.0102 | 0.505 | 0.0605 |
| A: h32, 16 ep, stretched | 0.067 | 0.264 | 0.0102 | 0.514 | 0.0604 |
| B: h128, 16 ep, stretched | 0.071 | 0.256 | 0.0101 | 0.524 | 0.0603 |

**The shrinkage is structural, not under-training.** Doubling training with a
live LR and 4x width leaves every metric unchanged (magnitude ratio ~0.26 in
all arms; train loss floor ~0.525 in all arms). Under response-blind sampling
the MSE-optimal predictor conditional on GO/graph features is close to the
global CRISPRi shift, so the trained model shrinks toward it. Files:
`*_h32_e16*`, `*_h128_e16*`.

## Data

- Source: scPerturb (Zenodo record 13350497) `NadigOConner2024_jurkat.h5ad`,
  md5 `d8b05d00bfbd686d37ffdd4293bc6c8c`, 262,956 cells x 8,882 genes, 55
  batches. The preferred scperteval GCS source is requester-pays; the scPerturb
  distribution of the same screen was used instead.
- X is dense raw integer counts -> normalize_total(1e4) + log1p.
- 2,123 targets pass eligibility (>=10 cells, in the measured gene panel, in
  GEARS's GO/essential perturbation universe); 230 dropped as unmeasured, 8
  outside the GEARS GO universe (incl. SEM1, which GEARS cannot model).
- Working set: 20,469 cells = 18,469 perturbed (<=50/target) + 2,000 controls.

## Sampling design (response-blind)

- Functional neighborhoods: GO terms covering >=50 eligible targets are
  anchors (91 terms); each gene is assigned to its rarest anchor term
  (most specific), giving 86 categories; full multi-membership is preserved in
  `sampling_manifest.json`.
- Degree: terciles of GO-graph degree (deg_low/mid/high).
- Strata = category x degree tercile; largest-remainder proportional
  allocation; **test (60) drawn first, then val (40), then train (300)** —
  target-disjoint, seed 42. No effect-size feature is used anywhere.
- Train covers 76/86 GO categories; degree terciles balanced
  (test 19/21/20, train ~100 each).

## Files

- `prepare_data.py` — chunked load/normalize, response-blind metadata,
  leakage-safe splits, cell sampling, GEARS conversion (~3.5 min total).
- `train.py` — official GEARS recipe (Adam lr 1e-3, wd 5e-4, StepLR(1, 0.5),
  grad clip 1.0, direction_lambda 1e-3), hidden 32, batch 8, eval batch 16,
  8 epochs, seed 42; per-epoch checkpoints; `--benchmark_batches N` mode;
  `--scheduler_step/--scheduler_gamma/--tag` for retraining arms.
- `evaluate.py` — held-out metrics vs control-only and mean-shift baselines,
  per-target stratifications (GO category, degree, basal expression, graph
  distance, GO term count); `--hidden_size`, `--label` for arm evaluation.
- `predict.py` — `JurkatGearsAdapter.predict_perturbation(perturbation_id,
  control_cells)` adapter; `--export_manifest` writes `MANIFEST.json`.
- `export_artifacts.py` — embeddings + held-out predictions export.
- `config.yaml` — full configuration.
- `split_manifest.json` / `sampling_manifest.json` / `dataset_manifest.json`.
- `training_history.csv` — per-epoch losses and val metrics (main arm);
  `training_history_h32_e16.csv`, `training_history_h128_e16.csv` for arms.
- `checkpoints/epoch{1..8}_checkpoint.pt`, `best_checkpoint.pt` (epoch 2),
  `epoch8_checkpoint.pt`; arm checkpoints `best_h32_e16.pt`, `best_h128_e16.pt`.
- `gene_embeddings.npz` (8,882 x 32), `perturbation_embeddings.npz`
  (9,853 x 32), `heldout_effect_predictions.npz` (2,673 test cells x 8,882).
- `metrics_by_perturbation_balanced.csv` (60 rows) + arm variants,
  `evaluation_summary_balanced.json` + arm variants.
- `pathway_coverage_report.csv` (all 2,123 eligible targets with split,
  category, degree, basal bins), `train_vs_universe_representation.csv`
  (train vs universe GO-category representation).
- `where_signal_survives.png` — signal-vs-magnitude figure.

## Training history (val, main arm)

| Epoch | val mse | val mse_de | val pearson_de | lr |
|---|---|---|---|---|
| 1 | 0.01519 | 0.07727 | 0.747 | 5e-4 |
| 2 | 0.00876 | **0.06048** | 0.819 | 2.5e-4 |
| 3 | 0.00872 | 0.06090 | 0.819 | 1.25e-4 |
| 4 | 0.00872 | 0.06103 | 0.817 | 6.25e-5 |
| 5 | 0.00870 | 0.06089 | 0.819 | 3.13e-5 |
| 6 | 0.00870 | 0.06086 | 0.819 | 1.56e-5 |
| 7 | 0.00870 | 0.06085 | 0.820 | 7.81e-6 |
| 8 | 0.00870 | 0.06085 | 0.820 | 3.91e-6 |

Note: val pearson_de (0.82, DE genes only, GEARS's internal metric) and the
all-gene test pearson_delta (0.073) measure different things; both are
reported to avoid a misleading picture.

## Benchmark (100 batches, GEARS random split — timing only)

0.243 s/train batch, 5.6 GB RAM, 1.78M params. With the final 300-target
split: ~1,966 batches/epoch, ~4-7 min/epoch, ~35 min total on 16 CPU cores.

## Known issues encountered (documented for reproducibility)

1. GEARS's `PertData.load` silently drops perturbations outside its GO/essential
   universe from `adata` while the pyg cell graphs keep them -> KeyError in
   `loss_fct`. Fixed by restricting eligibility to the GEARS GO universe.
2. GEARS's own `prepare_split('single')` is a random 75/10/15 split and would
   silently discard the Track J design; the custom split pickle is used
   everywhere (`split='custom'`).
3. `np.savez` cannot write directly to the S3-backed results mount; artifacts
   are written locally then copied.
