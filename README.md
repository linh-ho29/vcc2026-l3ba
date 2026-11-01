# VCC 2026 - CPU signature-transfer baseline + research tracks

## Layout

- `lib/` - shared modules imported by everything else (`expression_gating.py`,
  `multi_source_transfer.py`, `scbasecount_blend.py`, and their unit tests).
- `baseline/` - the production K562_gwps signature-transfer pipeline (what's
  actually submitted): bank construction and prediction generation.
- `evaluation/` - cross-cutting validation tooling shared across experiments
  (held-out K562_gwps evaluation, hyperparameter tuning, the RPE1
  cross-context ground-truth check, gating comparisons, scale checks).
- `scbasecount_experiments/` - E05/E06: scBaseCount-enhanced gating.
  **Rejected** - see `vcc_agent_harness/reports/E06_AUDIT.md`.
- `multisource_experiments/` - E07-E10: K562+RPE1 multi-source signature
  combination. Validated research finding on held-out HepG2, **not yet
  promoted to production** - see
  `vcc_agent_harness/reports/E10_HEPG2_VALIDATION.md`.
- `unvalidated/` - explicitly-flagged scaffolds not wired into the actual
  pipeline (scGPT+GEARS fusion mock, an externally-sourced ensemble utility).
- `notebooks/` - exploratory notebooks.
- `vcc_agent_harness/` - the resumable experiment-tracking harness: mission
  spec (`AGENT_PROMPT.md`), configs, per-experiment results
  (`experiments/*.json`), and status/decision reports (`reports/`). Read
  `vcc_agent_harness/reports/CURRENT_STATUS.md` for the current state of
  every experiment (E00-E10) and which one is the champion.
- `gears_track/` - standalone GEARS/GNN training handoff (3 epochs, paused for
  GPU infrastructure reasons), kept separate from `PertAdapt/`. Checkpoint,
  embeddings, and honest (modest) metrics for future resumption - see
  `gears_track/README.md`, which also records a proposed future adapter
  (statistical baseline for perturbed cells + GEARS for control-cell state).
- `PertAdapt/` - git submodule, a separate deep-learning track (paused).
- `data/`, `predictions/` - gitignored (large files); `data/manifests/` and a
  couple of small provenance manifests are tracked as exceptions.

## Current champion

E03 (K562_gwps signature transfer + combined basal-expression gating).
Already-submitted baseline: `predictions/final_predictions_DEF_gwps.h5ad`.
Validated-but-unsubmitted improvement:
`predictions/final_predictions_DEF_gwps_gated_e03.h5ad`.

Everything else in `scbasecount_experiments/` and `multisource_experiments/`
is a research finding, honestly scored (accepted/rejected/inconclusive), not
a production change - see `vcc_agent_harness/reports/CURRENT_STATUS.md`.

## Running things

Every script resolves paths relative to this repo root
(`Path(__file__).parent[.parent...]`), regardless of which subfolder it lives
in - run any script directly, e.g. `python3 baseline/generate_predictions_gwps.py`
or `python3 multisource_experiments/e10_hepg2_heldout_validation.py`. No
`PYTHONPATH` setup needed; each script inserts `lib/` onto `sys.path` itself.
