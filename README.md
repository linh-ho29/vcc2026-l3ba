# Hybrid MLP Perturbation Adapter (VCC 2026)

A modular, lightweight PyTorch MLP adapter that combines independently
generated genetic perturbation representations (GEARS), frozen scGPT gene-token
embeddings, statistical perturbation-effect estimates, and measured
non-targeting-control (NTC) features to predict corrected transcriptional
responses.

**Scope: implementation, synthetic testing, and integration readiness only.**
No real biological data is used or bundled in this phase. All fixtures are
artificial; their performance is not evidence of biological predictive accuracy.

## Core contract

```
corrected_effects = statistical_effects + residual
```

The residual output layer is zero-initialised, so the untrained adapter
exactly reproduces the statistical baseline.

## Architecture

| Module | Input | Output |
|---|---|---|
| A — Perturbation encoder | GEARS embedding (32) + projected scGPT target embedding (512→32) | `z_pert [B, 32]` |
| B — Control-state encoder | precomputed NTC features (32) | `z_ctrl [B, 32]` |
| C — Context fusion | `[z_pert, z_ctrl, z_pert ⊙ z_ctrl]` | `h [B, 64]` |
| D — Gene-wise residual MLP | h (64) + projected response-gene embedding (32) + basal (1) + stat effect (1) + optional GEARS effect (1) + availability/coverage indicators (5) = **104** | residual per gene |

Module D is one shared network for every responding gene (no per-gene heads);
the gene dimension is processed in configurable chunks (default 4096) so an
18,533-gene response axis stays memory-predictable on CPU.

## Installation

```bash
pip install -e ".[dev]"
pytest            # run the synthetic test suite
```

Runs on CPU; no accelerator dependencies.

## Quickstart

```python
from hybrid_adapter import (
    AdapterConfig, HybridAdapter, generate_whole_line_batch,
    predict_corrected_effects, save_adapter, load_adapter,
)

config = AdapterConfig()
model = HybridAdapter(config)

batch = generate_whole_line_batch()          # synthetic, clearly labeled
out = predict_corrected_effects(model, batch) # deterministic, eval mode
# out["corrected_effects"] == out["baseline_effects"] before any training
```

See `examples/synthetic_demo.py` for a full train → checkpoint → reload →
predict cycle, and `examples/h7_synthetic_validation.py` for the H7 pipeline.

## Input batch schema

Required keys (shapes): `context_ids [B]`, `perturbation_ids [B]`,
`response_gene_ids [G]` (one shared 1-D gene axis), `stat_effects [B,G]`,
`scgpt_target_embeddings [B,512]`, `scgpt_response_embeddings [B,G,512]`,
`control_features [B,32]`, `control_gene_expression [B,G]`, `effect_units`
(string, e.g. `"log2_fold_change"`), optional `availability_masks`.

Optional: `gears_effects [B,G]`, `gears_embeddings [B,32]`,
`stat_coverage [B,G]` (0/1 coverage indicators).

Validation is strict and raises `BatchValidationError` on: shape mismatches,
inconsistent gene ordering, unit mismatch, NaN/Inf inputs, unknown mask keys.
Missing GEARS inputs are handled through explicit availability masks and
configurable model branches (`perturbation_mode: both | scgpt_only |
gears_only | auto`); nothing is silently mixed across expression scales.

## H7 subpopulation modes (M0–M4)

Implements the *Subpopulation-Resolved, Identity-Gated Perturbation Response*
hypothesis: the unit of transfer is the (perturbation × subpopulation) pair,
not the cell line. Enable with `AdapterConfig(response_mode="subpopulation", ...)`;
the batch then additionally carries `phase_proportions [B,K]`,
`phase_stat_effects [B,K,G]`, optional `compsim_weights [B,K]`.

| Variant | Subpopulations | Response unit | Mixing fraction | Tests |
|---|---|---|---|---|
| M0 | no | whole line | one global α | H3 baseline (docx-spec adapter) |
| M1 | yes | subpopulation | one global α | H6 alone |
| M2 | yes | subpopulation | α(phase), shared across lines | H7 (claims 4+5) |
| M3 | yes | subpopulation | α(phase, line) | is claim 5 too strong? |
| M4 | yes | subpopulation | α(phase) + composition weighting | H7 + claim 3 (full model) |

Predicted ordering under H7: **M4 > M2 > M1 > M0**, with M3 ≈ M2 if claim 5
holds. The residual is decomposed per phase as
`r = α_eff · r_invariant + (1 − α_eff) · r_line_dependent` with
`α_eff(c,k) = α(k) · CompSim(c,k)` in M4, and the whole-line prediction is the
composition-weighted sum over phases. M3 requires context identity at train
time (`model.register_line_alpha`); at inference on unseen contexts it falls
back to phase-shared α (M2 behaviour).

## H7 validation package (`h7_validation`)

Standalone analysis code for the three-phase testing strategy (runs on public
perturbation datasets with paired perturbed/control cells; no data bundled):

- **Phase 1** (claims 1, 2, 6): Seurat-style phase scoring (bundled
  Tirosh/Seurat marker sets, overridable), per-phase silhouette, within-phase
  vs whole-line variance ratios, cross-line phase-classifier transfer.
- **Phase 2** (claims 3, 4, 5): per-phase deltas with between/within variance
  ratio; two-component decomposition with shared-α vs per-line-α fits (AIC +
  F-test); CompSim (1 − TV of phase proportions) vs whole-line-similarity
  transfer head-to-head on held-out perturbations.
- **Phase 3**: `ablation.run_ablation` trains M0–M4 on perturbation-holdout
  splits and applies the H7 interpretation rules.

If H7 survives all three phases, the adapter should be restructured so ΔP is
indexed by (perturbation, subpopulation) — which is exactly what the
subpopulation modes above provide — rather than retuned from the H2/H3
adapter.

## Training (implemented; not run on biological data)

`train_adapter(model, train_loader, validation_loader, config)` provides
AdamW (lr 1e-3, wd 1e-4), max 20 epochs, early stopping (patience 4),
gradient clipping, histories, checkpointing, deterministic seeds, and
interrupt/resume. Objectives: global Huber reconstruction (on), optional
DE-weighted reconstruction and perturbation-discrimination contrastive loss
(off by default: `lambda_de = lambda_disc = 0.0`).

**Data-hygiene requirements for future real training:**

- Measured held-out perturbation responses must never be used during fitting
  or hyperparameter selection.
- The statistical baseline (`stat_effects`) must be produced **out-of-fold**:
  the baseline for a training example must not have been fit on that example's
  own measured response, otherwise the residual has nothing to learn and
  leakage inflates apparent performance.

## Checkpoints and portability

`save_adapter(model, config, path)` / `load_adapter(path, device="cpu")`
write a versioned manifest (architecture config, feature dimensions,
required/optional modalities, gene-axis requirements, effect units, model
version, state-dict SHA-256 checksum, dependency versions). Checkpoints load
without GEARS or scGPT installed; no foundation-model weights or datasets are
bundled.

## Metrics

Pearson/Spearman correlation of predicted deltas, effect MSE/MAE, direction
accuracy with externally provided eligibility masks, and perturbation-
discrimination diagnostics live in `hybrid_adapter.metrics`. An optional
interface to Arc's official `cell-eval2` package raises an informative error
when the package is absent. Official VCC metrics (PDS, MSE, NMAE, FID, Reach,
Jaccard) require raw-count predictions and reference single-cell observations
and will be integrated separately — they cannot be derived from synthetic data
or pseudobulk deltas alone.

## Explicit scope limitations

- No training on actual biological data in this phase.
- No paid computing resources.
- No transformer/attention replacement for the MLP.
- No assumption that upstream embeddings have already been delivered — the
  adapter consumes cached feature tensors only.
