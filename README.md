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

## Project hypothesis

> Our virtual cell rests on the hypothesis that CRISPR perturbation responses
> occupy a low-dimensional transcriptional manifold whose dynamics follow a
> context-conditioned velocity field. We hypothesize that lineage-specific
> cellular responses can be decomposed into shared and cell-specific
> subnetwork components, enabling zero-shot or cross-context prediction by
> dynamically gating perturbation-induced graph representations learned from
> distinct cellular contexts.

The H7 subpopulation modes below and the gated multi-context fusion module
are both expressions of this hypothesis at different granularities:
subpopulation (cell-cycle phase) and cell line, respectively.

## Gated multi-context fusion (`hybrid_adapter.gated_fusion_flow.gated_fusion`)

Aligns Module A with the system architecture posted on `main`'s README
(gated latent multi-context fusion, Phase 2): instead of one GEARS
embedding, Module A can aggregate K independently pretrained per-cell-type
GEARS checkpoints (currently k562 and hepg2, on the `GEARS` branch's
`gears_track/per_celltype/`) via context-conditioned gating:

```
alpha_k = Softmax( W_gate . [ e_p^(k) || c ] )
e_p^(fused) = sum_k alpha_k . Linear(e_p^(k))
```

Enable with `AdapterConfig(use_gated_fusion=True, gears_source_dims=(64, 32), context_dim=...)`
and supply `gears_embeddings_multi` (a **list** of K per-source tensors, not
one stacked tensor - k562 and hepg2's real GEARS checkpoints do not share an
embedding width, confirmed directly, so each source gets its own input
projection before the shared gate), `gears_source_mask` (which sources are
real for each row - a missing source is excluded from the softmax, never
zero-filled), and `context_embedding`.

This is an incremental integration into the existing Module A only - the
fused vector is pinned to `gears_dim` so it slots in without touching
Modules B-D. `corrected = stat + residual` remains `HybridAdapter`'s only
prediction pathway, untouched by any of the below.

## Full Phase 2 pipeline (`hybrid_adapter.gated_fusion_flow.flow_model.GatedFusionFlowAdapter`)

A second, separate model implementing the rest of the architecture -
control-cell distribution sampling, the discrete latent flow block, the
decoder, and the Sliced Wasserstein distribution-matching loss - with no
statistical-baseline dependency at all:

```
z_0 ~ P_ctrl  (real control-cell latent, sampled empirically)
  -> gated fusion of {e_p^(k)}  ->  input conditioning  ->  discrete latent flow (2-3 steps)
  -> z_pert  ->  light decoder  ->  predicted expression
```

- **Control branch** (`hybrid_adapter.gated_fusion_flow.control_branch`): `ScGPTGeneEmbeddingPooling`
  is a frozen, expression-weighted pooling of per-gene scGPT embeddings into
  one per-cell vector - per the user's explicit direction, the **fine-tuned**
  scGPT gene embeddings are used here, frozen, as an offline feature
  extractor. Read this project's own `s05c` evaluation before expecting much
  from it: fine-tuning did not measurably beat frozen embeddings under a
  diagonal-ridge probe (deltas -0.0001 to -0.0004 across folds), and `s04`
  found embeddings (frozen or fine-tuned) match random Gaussian features
  within ~0.01 Pearson - most of a linear probe's apparent value is generic
  per-gene capacity, not unique semantic structure. Using the fine-tune here
  is a consistency choice, not an accuracy claim. `ControlCellEncoder` is a
  PCA fit once on real control-cell pooled features (closed-form, not
  gradient descent - not "training" in the sense the package's synthetic-
  phase rule guards against) then frozen. `NTCLatentBank` samples real,
  stored per-cell latents empirically - never a mean or parametric
  distribution, per the architecture's explicit "sample actual single-cell
  baseline vectors" requirement.
- **Latent flow** (`hybrid_adapter.gated_fusion_flow.latent_flow`): `InputConditioningLayer`
  projects `[z0 || e_p^(fused) || c]` into the flow's working space;
  `DiscreteLatentFlow` runs 2-3 explicit-Euler residual steps, each
  conditioned on `e_p^(fused)`, `c`, and a step embedding
  (Pre-LayerNorm -> Linear -> GELU -> Dropout); `LatentDecoder` is a light
  linear (or 2-layer) head to gene-expression space. One dimensional
  inconsistency in the posted diagram (z0/z_pert stated as R^128, the
  conditioning layer/flow internals stated as 256-wide) is resolved with an
  explicit final 256->128 projection rather than assumed away - documented
  in the module docstring.
- **Loss** (`hybrid_adapter.losses.sliced_wasserstein_distance`,
  `compute_flow_loss`): closed-form 1-D optimal transport averaged over
  random projections, comparing a batch of predicted vs. real single-cell
  latents for one perturbation (not a per-row loss like the others);
  combined with plain MSE reconstruction, SWD weight defaults to 0 (off)
  until there's real paired single-cell data to validate it against.

Verified against real data (forward pass only - no training, no gradients,
eval mode throughout): `examples/real_data_pipeline_smoketest.py` runs the
entire chain on 2,000 real HepG2 control cells (PCA explains 98.3% of
variance with 128 components), the real fine-tuned scGPT embeddings, and
real k562+hepg2 GEARS embeddings for 2,122 overlapping genes - finite
output confirmed end to end, not just on synthetic fixtures of convenient
matching shape. `examples/real_data_fusion_smoketest.py` covers the fusion
module alone. The context embedding `c` has no real source yet in either
script (zeros, clearly labeled) - Phase 1's control-cell-derived context
encoder doesn't exist yet.

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
