# GEARS RPE1: eight-epoch baseline and future-adapter interface

## Scope and status

This project implements the approved CPU GEARS baseline with hidden size 32,
training batch 8, evaluation batch 16, seed 42, and eight requested epochs.
**All eight epochs are now completed and checksum-verified**, at global step
90,448. In the release archive, the authoritative completion status is
`reports/training_progress.json`, and all eight checkpoints are in `checkpoints/`.
The validation-selected checkpoint is epoch 3, not epoch 8. Both checkpoints'
embeddings are supplied. `reports/primary_artifact_validation.json` records the
completed artifact checks. The optional raw-count evaluation is a separate stage;
its status is not implied by completion of training or normalized evaluation.
The future adapter has **not** been trained. Code and reports are now organized
under `gears_adapter_ready/` in the team repository; trained artifacts are in the
GitHub Release linked below.

The exact official GEARS RPE1 release contains 162,733 cells, 5,000 response genes,
1,543 perturbation conditions, and 11,485 controls. The official default GO
vocabulary cannot represent nine conditions (1,310 cells); these are explicitly
listed in `data/dataset_manifest.json` and `data/cell_manifest.csv.gz`. There is no
additional CPU-driven cell cap in the approved full-data run.

After the default GEARS compatibility filter, the seed-42 simulation split has
1,035 fitting-target conditions, 115 validation targets, and 384 unseen-test
targets. Validation and unseen-test target assignments and cells are unchanged.
Before fitting, 20% of cells within training targets and 20% of controls were
reserved, leaving:

| Role | Cells |
|---|---:|
| Perturbation fitting cells | 81,263 |
| Fitting controls | 9,188 |
| Independent seen-target test cells | 19,799 |
| Unseen-target test cells | 36,740 |
| Validation cells | 12,136 |
| Independent reference controls | 2,297 |
| Unsupported default-GO cells | 1,310 |

The file contains model-normalized expression, **not raw counts**. Its existing
HVG/strong-perturbation selection is inherited frozen preprocessing and may use
the original full study. Consequently this is a target-holdout benchmark
conditional on that published feature selection, not a claim of fully inductive
raw-data feature discovery. New preprocessing, coexpression, loss masks, and
baseline fitting exclude reserved/test outcomes.

## Environment

Python 3.11, CPU PyTorch 2.7.1, GEARS source commit
`f374e43e197b295016d80395d7a54ddb81cc6769`, and PyG 2.7.0 were used.
Core scientific packages are pinned in `requirements.txt`; exact execution
versions and resource measurements accompany the reports.

Clean-install note: the observed environment combined scikit-learn 1.5.1 with
UMAP 0.5.9.post2, whose declared dependency requires a newer scikit-learn.
The current clean-install requirements use UMAP 0.5.7 instead. This workflow
does not run UMAP; no trained weights, predictions, or scientific settings were
changed. The original primary-v2 archive retains the observed requirement list;
apply the accompanying reproducibility patch before a fresh installation.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps ./vendor/GEARS
```

The pinned GEARS source and license are bundled; repository access is not needed.
`requirements_resolved.txt` in the reproducibility patch records a successfully
resolved clean dependency set, not the original training environment. CPU Torch
is installed separately as above. Dependency resolution was validated; the
original training environment was not replaced or upgraded.

Only load trusted checkpoints. These PyTorch files contain complete Python/NumPy
restart state, not merely a safe-tensor weights dictionary.

The CPU benchmark tested 1/2/4/8 threads. Four threads were fastest; the confirmed
100-batch pilot projected approximately 3.37 hours for eight training passes,
before full validation/checkpoint overhead, with peak RSS approximately 3.79 GiB.
Actual elapsed time and memory supersede that projection in the final report.

## Reproduce fitting or resume

Use local POSIX storage for HDF5, NumPy memory maps, temporary checkpoint saves,
and archives. `--checkpoint-dir` retains verified public download objects.

```bash
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
PY=.venv/bin/python
$PY data/download_resources.py --root ./resources --checkpoint-dir ./resource_backup
$PY data/prepare.py --data-root ./resources --out ./prepared
$PY training/train.py benchmark --prepared ./prepared --out ./run --threads 4 --batch-size 8
$PY tests/smoke_interface.py --checkpoint ./run/initial_state.pt --prepared ./prepared --output ./run/interface_smoke.json
$PY training/train.py train --prepared ./prepared --out ./run --threads 4 --batch-size 8 --resume ./run/initial_state.pt --publish-dir ./artifacts/checkpoints --chunk-minutes 100
```

After a healthy chunk pause, replace `--resume` with
`./run/checkpoints/recovery.pt`. After an epoch-boundary pause, an epoch checkpoint
can also be resumed. The checkpoint stores the exact split, role row indices,
vocabularies, graphs, model/optimizer/scheduler, RNGs, shuffle state, cursor,
global step, and history. Data row identity is anchored to the public H5AD
checksum and the accompanying cell manifest. The original paired-control
semantics are reproduced with fixed seed-42 pairing restricted to fitting
controls. Sparse lazy batching avoids eager per-cell dense graph allocation.
The upstream `drop_last=True` policy consumes 90,448/90,451 fitting rows per epoch;
the three omitted rows vary with the deterministic epoch permutation.

The smoke test verifies exact next-step loss and parameter replay and exact
lazy/eager prediction parity. Benchmark updates are discarded before epoch 1.
Adam uses learning rate 0.001 and weight decay 0.0005; StepLR halves the learning
rate after each epoch. The official GEARS loss includes its quartic error and
direction term (lambda 0.1), with gradient-value clipping at 1.0.

Validation target outcomes are used only for the prespecified validation rule,
macro top-20-DE MSE. Validation DE ranks are recomputed against fitting controls.
Reserved seen-target and unseen-test responses are not model-selection inputs.
Their evaluation occurs only after fitting and prediction/count settings freeze.

## Load trained artifacts without fitting

Full inference needs a checkpoint and model-scale controls in exact gene order;
it does not need measured perturbed outcomes or the entire original dataset.
The exported control panel and adapter bundle provide a small working example.

```python
import numpy as np
from inference.predict_effects import GEARSPredictor

model = GEARSPredictor("checkpoints/epoch_08.pt")
panel = np.load("adapter_interface/control_states.npz", allow_pickle=False)
controls = panel["control_state"]
unseen = model.metadata["frozen_split"]["conditions"]["test"][0]
bundle = model.predict_effects([unseen], controls, context_id="rpe1")
effects = bundle["predicted_effects"]        # targets x control cells x 5000
target_features = bundle["perturbation_embeddings"]  # targets x 32
```

Effects are additive changes in natural-log1p normalized expression. They are
not raw counts or log2 fold changes. Unsupported target identifiers and
non-RPE1 contexts raise errors. A response gene is not automatically a valid
perturbation target. AnnData inputs must match the exact symbol or Ensembl axis;
plain arrays are the caller's responsibility and must follow `gene_index.csv`.

Optional external gene/cell embeddings are separate adapter inputs:

```python
from adapter_interface.schema import ExternalEmbedding, align_external_embedding

# Supply real teammate-produced values and identifiers here.
# external = ExternalEmbedding(identifiers, values, provenance="model/version")
# aligned = align_external_embedding(external, model.genes)
# A future adapter consumes aligned["values"], aligned["coverage"], target
# features, controls, and effects. Nothing changes the GEARS checkpoint.
```

## Export representations and predictions

After fitting, export epoch 8 and the validation-best checkpoint as appropriate.
Preserve all eight epoch checkpoints and compare epochs 2, 4, 6, and 8.

```bash
$PY inference/extract_embeddings.py --checkpoint ./run/checkpoints/epoch_08.pt --output ./exports/embeddings
$PY adapter_interface/export_features.py --checkpoint ./run/checkpoints/epoch_08.pt --prepared ./prepared --output ./exports/adapter_interface
$PY inference/export_predictions.py --checkpoint ./run/checkpoints/epoch_08.pt --prepared ./prepared --output ./exports/predictions
```

`representations.npz` contains 32-dimensional learned gene lookup, separate
gene positional embeddings before/after coexpression propagation, combined gene
representations, perturbation lookups before/after GO propagation, and fused
single-target representations. `embedding_manifest.json` defines every tensor
and reports norm/effective-rank/coverage diagnostics.

**Upstream implementation caveat:** the pinned forward pass repeats positional
embeddings across the batch but uses a single-copy coexpression edge index.
Representations therefore have potentially distinct first-slot and later-slot
semantics. This project preserves that architecture, exports both definitions,
and fixes inference batch size/order at 16, padding the final chunk. It does not
silently claim batch-invariance or silently change the original model.

**Representation-quality caveat:** several epoch-8 arrays have uncentered
entropy effective rank near one despite 32 columns. Inspect
`reports/embedding_diagnostics.csv`; do not assume the exported width represents
32 independent adapter features. No weights were changed to hide this result.

Prediction archives contain target means over the same 64 fitting-control inputs,
model-scale effects, first/other-slot effects, gene IDs, and input control IDs.
They are compact normalized profiles, not synthetic single-cell references.
The adapter bundle contains actual per-control predictions for one unseen target,
its control states, target features, masks, and provenance. No measured test
response is included in that bundle.

## Evaluate and interpret

The statistical baseline uses an equal-weight mean of fitting-target effects.
The control-only baseline predicts no perturbation effect. Both use the same
control-input panel as GEARS.

After producing all four prespecified prediction archives:

```bash
$PY evaluation/evaluate_gears.py --prepared ./prepared --prediction-dir ./exports/predictions --output ./exports/reports
```

These are **supplemental normalized-scale metrics**, including target-mean MSE,
Pearson/Spearman delta, and top-20 direction diagnostics. Macro and cell-weighted
aggregates have separate columns; undefined correlations remain unavailable.
Independent seen-target and unseen-target performance are separate.

The completed epoch-8 macro MSE is 0.025269 on seen targets and 0.015259 on unseen
targets. The statistical baseline is 0.026159 and 0.013987, respectively.
Accordingly, the trained model does not beat that baseline on unseen-target
macro MSE. On that split, cell-weighted MSE gives a different ordering:
0.011509 for epoch 8 versus 0.011804 for the statistical baseline. Both aggregation
rules are reported without selecting one after observing these results.

The optional count module is a **heuristic prediction layer**, not a trained
adapter or learned count-distribution model. It reweights real fitting-control
count proportions by `exp(predicted_log_effect)`, uses a fixed fitting-control
background fraction of 0.0001, and samples multinomial integers at library depths
from fitting controls. Effects, reconstructed counts, measured counts, and their
evaluation are distinct artifacts.

`evaluation/evaluate_vcc2026.py` invokes Arc's official `vcc2026` preset, with an
explicit supported CPU DE backend, only when matched measured raw single-cell
RPE1 counts are available. Install cell-eval2 in a separate evaluation environment
from commit `d86be3f4cd47015cd41046e9cc6a834777453594`.
The original-study raw RPE1 source is Figshare file 35775606 (8,700,873,216 bytes).
Do not substitute inverse-log normalized values for measured raw counts.

The reproducibility patch supplies the pinned evaluator source and a compatible
CPU constraint file. From the project directory:

```bash
uv venv .eval-venv --python 3.11
uv pip install --python .eval-venv/bin/python -c evaluation/cpu_constraints.txt ./vendor/cell-eval2 psutil==7.0.0
```

For memory-bounded scoring use `evaluation/evaluate_counts_bounded_cpu.py`.
It precomputes DE per target through the official implementation, against the
same real control pool, then passes the combined tables to official
`compute_metrics`. This requires the frozen per-perturbation FDR rule and fixed
per-cell normalization. On an actual two-target pilot it matched the standard
official API at absolute and relative tolerances of 1e-10, including NaNs.
The count writer's optimized direct CSR appends were separately checked to
preserve every pilot count and both axes. Neither operational change changes
the count-sampling model.

The raw source has been matched to all 162,733 model cell IDs and 5,000 gene IDs,
with all retained entries checked as nonnegative integral measured counts.
The full unseen count panel contains 384 targets at 400 generated cells each;
evaluation uses 36,740 measured unseen cells and 2,297 independent reference
controls. Full count-score outputs are delivered separately as they finish.
The two-target pilot is a software parity test, not the full benchmark.
Native aggregation includes `expr_mse_unbiased_capped_norm`, a derived
ratio-of-sums with no per-target rows; diagnostic expression metrics are not
additional canonical leaderboard members.

Any such result is an RPE1 benchmark on the explicitly declared matched gene
panel, **not an official challenge leaderboard result**. The six canonical
metrics are reported individually where calculable. Do not average their raw
values. A calibrated aggregate requires a compatible reference bundle; official
challenge-context calibration anchors cannot simply be transferred to RPE1.
The reports explicitly state any unavailable official metrics or aggregate.

## Repository layout and released artifacts

This directory is the GEARS project root. From the repository root, run:

```bash
cd gears_adapter_ready
```

Run the examples above from here. For interactive Python, use this directory as
the working directory (or add it to `PYTHONPATH`) before importing `inference`,
`common`, or `adapter_interface`. Script entry points add their own project root
using `Path(__file__).resolve().parents[1]`; `common.ROOT` resolves here too.
All nine GEARS code/report directories and the handoff documents, requirements,
and manifests live together here. Existing `adapter_interface/` files are
preserved unchanged.

The previous placeholder title was **GEARS RPE1 – 8-Epoch Trained Model**; this
complete README retains that model identity and expands its usage instructions.

Trained checkpoints, embeddings, and predictions are already published in
[GitHub Release `gears-rpe1-8epoch-v1`](https://github.com/linh-ho29/vcc2026-l3ba/releases/tag/gears-rpe1-8epoch-v1).
The release assets are `gears_rpe1_trained_project_handoff_v4.tar.gz` and
`gears_rpe1_trained_project_handoff_v4_checksums.json`. They were not downloaded,
retrained, or regenerated during this reorganization. Use their checksum records
when restoring artifacts; training/export examples above describe the original
workflow and are not required to obtain the existing trained model.

### Transfer gaps and execution prerequisites

The GitHub code upload is not yet a standalone runnable copy of the full release:

- `data/` is absent. This includes `csr_rows.py` (imported as
  `data.csr_rows` by count evaluation/tests), `download_resources.py`,
  `prepare.py`, `download_raw_counts.py`, `prepare_raw_counts.py`,
  `export_count_references.py`, and the prepared metadata, split, gene/target
  indices, cell roles, graphs, and training statistics recorded in
  `trained_handoff_manifest.json`. The download/preparation examples cannot run
  until the corresponding sources and inputs are transferred.
- `vendor/` is absent. The local-install commands require `vendor/GEARS`
  at commit `f374e43e197b295016d80395d7a54ddb81cc6769` and, for count
  evaluation, `vendor/cell-eval2` at commit
  `d86be3f4cd47015cd41046e9cc6a834777453594`, with their licenses.
- Checkpoints, embedding exports, and prediction exports are Release assets,
  not files in this Git tree. Full inference, artifact verification, and
  scientific parity tests require those existing artifacts and appropriate
  inputs. Do not rerun training or exports just to fill this checkout.
- Core dependencies are in `requirements.txt`; CPU Torch is installed
  separately. Count evaluation additionally needs the pinned `cell_eval2`
  environment (including `polars`) and
  `evaluation/cpu_constraints.txt`. The core requirements alone do not provide
  this separate evaluator environment.
- Several `manifests/` recovery/packaging scripts and data-dependent tests
  preserve historical absolute paths under `/workspace`, `/mnt/results`,
  and `/mnt/shared-workspace`. For example,
  `manifests/restore_handoff_inputs.py` points to
  `/workspace/gears_adapter_ready`. These are records of the original runtime,
  not portable checkout defaults; review their input/output paths before any
  future use. They were not executed as part of this move.

The JSON artifact/handoff manifests and reports retain the original archive
paths, hashes, and historical status fields. Their paths are relative to this
GEARS directory when referring to project files. They describe the archived
snapshot, not a new checksum inventory of this reorganized checkout; in
particular, this README has changed. Do not rewrite historical checksums to
make missing assets appear present.

### Reorganization verification

All 99 moved files were absent from the team's pre-upload commit
`93534e47841191b3586472a95fb50cb375eb973a`; the source branch was verified at
`74feafdcf2557aa459815a3d6a91de5a697846f9`, following the GEARS uploads by
`courtney-chen`. All moved content except this README retains its original Git
blob. Python syntax and local import/path structure were checked without
training, inference, or binary downloads. Full runtime/scientific checks remain
blocked by the missing inputs, vendor sources, and runtime packages described
above.

## Provenance sources

- Official GEARS implementation: https://github.com/snap-stanford/GEARS
- Exact data/GO IDs and SHA-256 hashes: `data/resource_manifest.json`
- GEARS data release: https://dataverse.harvard.edu/api/access/datafile/7458694
- Original raw RPE1: https://ndownloader.figshare.com/files/35775606
- Official evaluator: https://github.com/ArcInstitute/cell-eval2
- Evaluator metric specification: https://github.com/ArcInstitute/cell-eval2/tree/main/docs/vcc2026_metrics
