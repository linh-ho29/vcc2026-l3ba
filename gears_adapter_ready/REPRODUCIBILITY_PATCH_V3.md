# Reproducibility patch v3 for primary release v2

Apply this overlay only after verifying `gears_rpe1_8epoch_primary_v2.tar.gz`
against its supplied checksum receipt. Extract the patch into the same parent
directory so the `gears_adapter_ready/` directories merge. No checkpoint,
embedding, normalized prediction, split, graph, or training statistic is replaced.
Keep the original archive unchanged for provenance.

## Changes

- Clean-install requirements replace unused UMAP 0.5.9.post2 with compatible
  0.5.7 while retaining scikit-learn 1.5.1. The original observed environment
  remains recorded. A clean dependency resolution succeeded; a fresh training
  installation or retraining is not claimed.
- Install the bundled GEARS source without repository access.
- Include pinned cell-eval2 0.20.0 source at commit
  `d86be3f4cd47015cd41046e9cc6a834777453594`, its license, CPU constraints,
  and the recorded isolated evaluation environment.
- Use framework-free I/O in the evaluation environment and Polars `write_csv`.
- Add matched measured-count preparation, deterministic count-panel exports,
  independent reference exports, and memory-bounded official DE precomputation.
- Include configuration-aware, source-identified DE caches. Preserve the official
  derived normalized-expression metric in native aggregation.
- Include actual-data parity receipts and alignment checks. Full unseen-target
  scoring was in progress when this patch was made; no pilot score is claimed
  as a full-panel result.

## Integrity and interpretation

`artifact_manifest.json` describes the immutable original primary archive.
After applying the overlay, use `reproducibility_patch_v3_manifest.json` for the
changed and added files. Their expected relative paths, sizes, and SHA-256 hashes
are included. The patch's own archive checksum receipt is separate.

The reconstruction layer is heuristic, not a trained adapter. The future adapter
remains untrained. Count scores characterize the selected GEARS checkpoint or
baseline **plus** that fixed layer on the restricted 5,000-gene RPE1 benchmark.
These are not official challenge leaderboard scores. No compatible RPE1
calibration bundle is available, so no calibrated aggregate is supplied. No
GitHub upload, new machine, paid resource reservation, or additional training
was performed to make this patch.
