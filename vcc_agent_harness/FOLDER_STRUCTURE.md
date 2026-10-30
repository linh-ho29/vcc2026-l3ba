# VCC 2026 agent harness — suggested project layout

```text
virtual-cell-project/
├── AGENT_PROMPT.md                 # Persistent instructions for coding agent
├── FOLDER_STRUCTURE.md            # This architecture guide
├── baseline/                      # Existing K562 statistical model: do not overwrite
├── configs/
│   ├── experiments.json           # Pending experiment command queue
│   └── scbasecount.yaml           # Dataset reference selection limits
├── data/
│   ├── manifests/
│   │   └── scbasecount_selection.csv  # Verified accessions only
│   ├── reference_features/        # Study-balanced pseudobulks, PCA, etc.
│   └── processed/                 # Reusable aligned matrices
├── src/
│   ├── atlas_selector.py
│   ├── reference_builder.py
│   ├── context_encoder.py
│   ├── gating.py
│   ├── predictor.py
│   ├── count_generator.py
│   └── evaluator.py
├── harness/
│   ├── runner.py                  # Deterministic command runner / process timeout
│   ├── state.json                 # Generated; resumable state
│   └── logs/                      # Per-experiment stdout/stderr
├── experiments/
│   └── results.jsonl             # Append-only run results
├── reports/
│   ├── CURRENT_STATUS.md         # Generated compact agent handoff
│   └── decisions.md              # Full decisions and reasons
└── tests/
```

## Data flow
K562 baseline -> K562 effect matrix -> target control-state features -> candidate gating -> count generator -> format validation -> held-out public-line evaluation -> accept/reject. scBaseCount enters only at *reference selection -> contextual features*, not as perturbation ground truth.

## Entry point
`python harness/runner.py --config configs/experiments.json --root .`

`runner.py` is a functional **command harness**, not an implementation of perturbation prediction. Configure experiment commands to invoke scripts already present or that the coding agent creates. `experiments.json` starts with a disabled example; replace it before running. Runner intentionally does not choose statistical winners from opaque metrics. A separate evaluator must promote candidates after prespecified decisions.
