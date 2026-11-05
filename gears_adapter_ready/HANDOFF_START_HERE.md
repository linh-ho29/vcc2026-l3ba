# Trained GEARS RPE1 project: independent preservation handoff

## Status and scope

**All eight training epochs are complete. This archive is usable independently
of the unfinished count-evaluation campaign.** It contains every epoch checkpoint,
final and validation-best gene/perturbation embeddings, four normalized prediction
exports, the untrained adapter-ready interface, configurations, frozen identities,
training logs, source, dependency files and read-only verification tools.

The validation-selected checkpoint is epoch 3; epoch 8 is the final requested
checkpoint. No retraining, new model inference, embedding regeneration or
normalized-prediction regeneration was performed to create this handoff.

The separate count campaign previously saved two of twelve panel receipts
(epoch-8 unseen and statistical-baseline unseen). Those results are being
revalidated separately. This archive deliberately does not claim completion of
that campaign or include a complete twelve-panel count benchmark.

## Verify before use

1. Check the archive SHA-256 against the separately supplied checksum receipt.
2. Extract into a new staging directory, never directly over team files.
3. From the extracted `gears_adapter_ready/` directory run:

```bash
uv venv .review-venv --python 3.11
uv pip install --python .review-venv/bin/python numpy==1.26.4
.review-venv/bin/python tests/verify_trained_handoff.py --root .
```

This hashes every manifest-listed member, verifies all eight checkpoint hashes
against the original completion receipt, and checks the exported-array schemas,
finite values, gene/condition alignment and adapter arrays plus JSON metadata.
It does **not** unpickle a checkpoint, run a model forward pass or train anything.
Only load checkpoints after establishing trust; these `.pt` files contain Python
and NumPy restart state, not just a safe tensor dictionary.

`trained_handoff_manifest.json` and this document describe the present snapshot.
Historical manifests/reports remain unchanged and refer to their original stage.
The final complete-evaluation verifier/guide is retained as source/template
material, not as a claim that twelve panels are already complete. Do not run
`tests/verify_review_bundle.py` until the separately delivered complete-review
manifest and all twelve result panels exist.

## Contents

- `checkpoints/epoch_01.pt` through `epoch_08.pt`, with original sidecars.
- `embeddings/`: final epoch-8 gene and perturbation representations and indexes.
- `embeddings_best_epoch_03/`: validation-best representations and indexes.
- `predictions/epoch_02_profiles.npz`, `epoch_04_profiles.npz`,
  `epoch_06_profiles.npz`, `epoch_08_profiles.npz`: existing normalized outputs.
- `adapter_interface/`: aligned controls, unseen-target example bundle, schemas,
  validation code and manifests. The adapter itself has **not** been trained.
- `data/`, `configs/`, `reports/`: frozen data/split/graph identities, fitting
  statistics, training progress/logs, validation history and prior audit records.
- `vendor/GEARS/` and `vendor/cell-eval2/`: pinned sources and licenses.
- `README.md`, `requirements.txt`, `requirements_resolved.txt`,
  `evaluation/cpu_constraints.txt`: model setup and separate CPU-evaluation setup.

The compact archive excludes original large expression datasets and generated
count H5ADs. Their provenance and retrieval/reconstruction instructions are
retained. Existing embeddings and normalized predictions need neither raw data
nor a retraining run for review.

## Optional model environment, not needed for read-only verification

From the extracted project directory:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps ./vendor/GEARS
```

The observed training environment had an unused UMAP/scikit-learn metadata
conflict. The clean-install requirements use UMAP 0.5.7, without changing trained
weights, predictions or scientific settings. `requirements_resolved.txt`
documents that clean dependency resolution, not a new training run.
Some session-recovery scripts contain original machine paths for provenance;
use explicit local paths rather than invoking those orchestration scripts blindly.
The README's training/export commands are optional provenance instructions,
**not** required handoff-verification steps.

## Scientific limitations and challenge use

- This model is for the published **RPE1 CRISPRi** context. It does not establish
  generalization to a challenge cell line, perturbation regime or unseen context.
- There are 5,000 response genes but a distinct 9,853-entry perturbation
  vocabulary. A measured gene is not automatically a supported model target.
  Nine source conditions (1,310 cells) were unsupported by the default GO
  vocabulary and explicitly excluded; consult the dataset manifest.
- Frozen target splits are 1,035 fitting-target conditions, 115 validation
  targets and 384 unseen-test targets. Seen-target test cells are held out.
- Only 159 of the 384 unseen target transcripts occur in the 5,000-gene response
  panel. Source ID/symbol checks found no retained-gene alias conflict under
  the available mapping; this is not an exhaustive global alias resolution.
- Outputs are additive **natural-log1p normalized effects**, not measured counts
  or log2 fold changes. Future count-space scores additionally depend on a
  fixed heuristic reconstruction layer, not a trained count-distribution adapter.
- Embeddings have width 32, but several have effective rank near one; retain the
  diagnostics rather than treating them as 32 independent features.
- Upstream batch-slot dependence is preserved. Keep batch size/order/padding and
  first/other-slot definitions consistent; do not assume batch-invariant inference.
- Published upstream feature selection is inherited. This is a target-holdout
  benchmark conditional on that preprocessing, not fully inductive feature discovery.
- The statistical baseline beats epoch 8 on unseen macro normalized MSE, while
  cell-weighted MSE reverses that ordering. Both aggregations are retained.
- RPE1 metrics are **not official Virtual Cell Challenge leaderboard scores**.
  No compatible calibration bundle was available; no calibrated aggregate is
  claimed. Do not average raw metric values into an invented overall score.

## Codex and later GitHub transfer

Give Codex this archive and checksum receipt; request read-only verification and
review first. Keep trained artifacts immutable, label any new analysis separately,
and do not default to retraining or regenerating predictions.

No GitHub access or upload was performed. Extract separately, compare the
destination branch/files, merge intentionally, and use Git LFS or an approved
artifact store for binaries. Retain checksums and avoid force pushes or silent
overwrites. Challenge applicability and submission compatibility must be assessed
separately; possession of these trained RPE1 artifacts is not a validated submission.
