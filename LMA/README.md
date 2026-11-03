# HepG2 GEARS — Broad Random Perturbation Coverage

HepG2-specific GEARS model trained on the Nadig et al. 2025 essential-gene CRISPRi
Perturb-seq screen (GEO GSE264667), testing whether maximizing perturbation-target
diversity under a fixed cell budget yields useful generalization to unseen genetic
perturbations. This model is intended as the HepG2 context of a future multi-context
hybrid model; no cross-cell-type training is performed in this phase.

## Design summary

| Element | Value |
| --- | --- |
| Data source | GEO GSE264667 `GSE264667_hepg2_raw_singlecell_01.h5ad` (raw counts) |
| Eligible universe | 1,818 single-gene CRISPRi targets with >= 30 cells |
| Selected targets | 500, random, seed 42, without replacement, blinded to effect size |
| Split | 400 train / 50 val / 50 test, target-disjoint |
| Cell cap | 30 cells per target (500/500 targets at cap); 2,000 non-targeting controls |
| Working set | 17,000 cells x 5,000 HVG panel |
| Expression scale | X = normalize_total(1e4) + log1p (official GEARS convention); raw counts in `layers['counts']` |
| Model | Official snap-stanford GEARS, hidden 32, GO + co-expression graphs |
| Training | 8 epochs, batch 8, eval batch 16, Adam lr 1e-3, wd 5e-4, StepLR gamma 0.5, seed 42, CPU |
| Checkpoint selection | Best epoch by validation `mse_de` (GEARS official criterion); test evaluated once, after selection |

The scperteval hosted copy of this dataset (`nadig25hepg2`) is a requester-pays GCS
object and stores log-normalized X only; the GEO raw file was used instead and raw
counts were verified (integer-valued) before normalization.

## Reproduction

```bash
# 1. Environment (Python 3.11)
uv pip install -r requirements.txt

# 2. Download raw data (~5.6 GB)
curl -o data/GSE264667_hepg2_raw_singlecell_01.h5ad -C - \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad

# 3. Audit, sample, build working dataset (seed 42; writes manifests)
python3 prepare_data.py

# 4. Optional: benchmark (100 steps) before the full run
python3 train.py --benchmark 100

# 5. Train 8 epochs (per-epoch checkpoints; safe to interrupt and rerun to resume)
python3 train.py

# 6. Evaluate checkpoints 2/4/6/8 on validation, select best, evaluate test
python3 evaluate.py

# 7. Export embeddings, held-out predictions, representation analysis
python3 export_outputs.py

# 8. Standalone predictions for arbitrary genes
python3 predict.py --checkpoint best_checkpoint.pt --genes TFAM GFM1
```

The GO graph asset (`gene2go.pkl`) downloads automatically from Harvard Dataverse on
first use and is cached in `data/`.

## Files

- `prepare_data.py` — audit, sampling, working-dataset construction, manifests
- `train.py` — GEARS training loop (official math) + per-epoch checkpoints + benchmark mode
- `evaluate.py` — checkpoint selection on validation, held-out test metrics, baselines
- `export_outputs.py` — embedding/prediction exports and representation analysis
- `predict.py` — standalone predictor (no training environment needed)
- `config.yaml` — full configuration record
- `split_manifest.json` — sampled targets, split assignment, per-target cell counts
- `dataset_manifest.json` — source audit and transformation record
- `training_history.csv` — per-epoch train/val metrics
- `checkpoint_validation_metrics.csv` — epochs 2/4/6/8 on validation
- `metrics_by_perturbation.csv` — per-target test metrics (best model)
- `metrics_summary.json` — aggregate results and selection record
- `epoch8_checkpoint.pt`, `best_checkpoint.pt` — model weights
- `gene_embeddings.npz`, `perturbation_embeddings.npz` — pre-graph, post-graph, fused
- `heldout_effect_predictions.npz` — adapter-ready predictions for held-out targets
- `representation_analysis.json` — effective ranks, cosine similarities, graph connectivity
- `MANIFEST.json` — file inventory with checksums

## Metric conventions

GEARS-standard formulas are used throughout: per-perturbation metrics compare the
mean predicted vs mean observed expression across that perturbation's cells, on all
panel genes (`mse`, `pearson`) and on the perturbation's top-20 DE genes (`mse_de`,
`pearson_de`); aggregates average over perturbations. Delta metrics subtract the
control mean before comparison (`pearson_delta`, `spearman_delta`, `mse_delta`) and
are prioritized for interpretation, per project convention. Baselines: control-only
(predict zero delta) and mean-shift (predict the mean training delta vector).

## Known limitations

- CPU-only training; no GPU was available in the execution environment.
- Predictions cover the 5,000-HVG panel; `gene_coverage_mask` records this scope.
- Targets whose genes lack GO annotations cannot be represented in the GEARS
  perturbation graph; any such exclusions are printed by `train.py` during setup.
- 32-dim embeddings were specified; no claim is made that they outperform 64-dim
  without a controlled comparison.
