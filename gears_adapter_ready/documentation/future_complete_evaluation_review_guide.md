# Independent review of the eight-epoch RPE1 project

## Start here: read-only review, not retraining

This package preserves the completed eight-epoch GEARS run, both embedding
exports, four normalized prediction exports, the **untrained** adapter interface,
frozen data identities, and the six-method seen/unseen count evaluation.
The release builder refuses to publish a complete review bundle unless all
twelve count panels have valid completion receipts and metric coverage.

1. Verify the downloaded archive against its external checksum receipt.
2. Extract into a new directory; do not overlay a working team checkout.
3. From `gears_adapter_ready/`, run the read-only verifier below.
4. Read the final count report and the historical primary report together.
5. Review the code and provenance before optionally rerunning any evaluation.

No reviewer command below trains a model, regenerates embeddings, or reruns
normalized model inference. Never deserialize an untrusted PyTorch checkpoint.

## Minimal integrity/schema checks

Python 3.11 and NumPy suffice; neither Torch nor a GPU is needed:

```bash
uv venv .review-venv --python 3.11
uv pip install --python .review-venv/bin/python numpy==1.26.4
.review-venv/bin/python tests/verify_review_bundle.py --root .
```

The verifier reads every manifest-listed file and SHA-256, verifies eight
checkpoint filenames without unpickling them, checks both embedding exports,
the four normalized-profile schemas, and the adapter arrays plus their JSON
metadata. It also checks every completed count-panel receipt, target coverage,
the six requested metrics, undefined-value counts, and native aggregation.

`review_bundle_manifest.json` is authoritative for this release. Historical
reports and manifests are retained unchanged; a historical “count evaluation
pending” flag describes that earlier stage, not this release. The external
archive checksum authenticates the manifest as well as the other members.

## Optional isolated evaluation environment

```bash
uv venv .eval-venv --python 3.11
uv pip install --python .eval-venv/bin/python \
  -c evaluation/cpu_constraints.txt ./vendor/cell-eval2 psutil==7.0.0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

The bundled evaluator is cell-eval2 0.20.0, pinned by
`vendor/cell-eval2/SOURCE_COMMIT`. The helper checks the installed Python source
against this vendor tree. It does not fetch the current upstream branch.
Torch is deliberately absent from this isolated evaluation environment.

Optional large measured/generated count inputs are supplied separately; they
are not silently embedded in the compact code/model archive. Use their count
manifests to verify SHA-256 before running any numerical work. Evaluate a full
panel into a **new local directory**:

```bash
.eval-venv/bin/python evaluation/evaluate_counts_bounded_cpu.py \
  --predicted ./counts/gears_epoch_8_unseen_test.h5ad \
  --real ./counts/measured_unseen_test.h5ad \
  --output ./review_recomputed/epoch08_unseen \
  --cache ./review_cache/epoch08_unseen \
  --checkpoint ./review_de_checkpoint \
  --groupwise-bulk
```

The same CLI accepts other method/exposure count files. The groupwise option
uses the pinned official functions for per-group sufficient statistics, then
dispatches metrics once over the **complete panel**. Do not score arbitrary
subpanels and average PDS: that changes its candidate universe. Eight-target
parity tests are software checks, not benchmark panels. The native normalized
expression error is a ratio of sums, not a mean of per-target ratios.

Session-specific campaign/recovery helpers retain the original execution paths
as an audit trail; they are not the portable entry point. Use the explicit-path
CLI above when reviewing elsewhere. Do not run old training, post-training,
embedding-export, or prediction-export commands merely to verify this bundle.

## Raw source and regeneration boundaries

The original raw count H5AD is the public Figshare file 35775606, size
8,700,873,216 bytes, SHA-256
`9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1`.
It is not in the compact archive. The range downloader is optional:

```bash
.eval-venv/bin/python data/download_raw_counts.py \
  --work ./raw_source --checkpoint ./raw_parts
```

Raw matching, count generation and model inference are distinct operations.
Existing normalized profiles suffice for count reconstruction; no checkpoint
forward pass is required. The matched measured cache and fitting statistics
are required for that reconstruction. Retain exact gene order, cell roles,
frozen per-condition seeds, count manifests and sampler-provenance records.
Do not replace measured counts with inverse-log normalized values.

## Scientific review priorities

- **Selection:** epoch 3 was selected by the prespecified validation rule.
  Epoch 8 is the requested endpoint, not a newly test-selected winner.
- **No trained adapter:** the count layer is a fixed heuristic, not adapter
  training or a learned single-cell count distribution.
- **No challenge leaderboard claim:** these are public RPE1 results on a
  restricted 5,000-gene response panel. No compatible calibration bundle was
  available, so no calibrated aggregate is reported. Never average raw metrics.
- **Leakage boundaries:** fitting controls, reference controls, seen-target
  held-out cells and unseen targets have separate roles. Published upstream
  feature filtering is inherited; this is not fully inductive raw-data feature
  discovery. New fitting components exclude reserved outcomes.
- **Representation quality:** 32 columns do not imply 32 independent features.
  Several epoch-8 embeddings have effective rank near one; retain those findings.
- **Architecture:** upstream batch-slot dependence is preserved. Inference
  batch size, ordering, padding and first/other-slot exports must remain fixed.
- **Response coverage:** 159 of 384 unseen target transcripts occur in the
  response panel. The available raw-source ID/symbol mapping revealed no
  retained-gene alias conflict; that is not an exhaustive global alias audit.
- **Baseline comparison:** unseen macro normalized MSE favors the statistical
  baseline, while cell-weighted MSE reverses that order. Report both. Count-space
  metrics describe an additional heuristic layer and need not rank methods
  identically.
- **Uncertainty:** no biological replication, formal significance test or
  post-hoc metric selection is implied by these point estimates.

## Later GitHub handoff

No GitHub access, upload or push was performed. Review in a separate staging
directory, inspect the destination branch and existing files, then merge
intentionally. Use Git LFS or an approved external artifact store for binaries,
retain checksum receipts, and avoid force-pushes or silent overwrites.
