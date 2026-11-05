"""Batch schemas and validation for the hybrid perturbation adapter.

This module defines the input/output contracts for :class:`hybrid_adapter.model.HybridAdapter`.
All tensors are float32 unless stated otherwise. Gene ordering is defined by a single
1-D ``response_gene_ids`` tensor shared by every gene-axis input; inconsistent ordering
is a hard error.

Effect units are explicit: every batch must declare ``effect_units`` (e.g.
``"log2_fold_change"``) and it must match the units the model was configured with.
Expression scales are likewise declared via ``expression_units``; the adapter never
silently mixes incompatible scales.

SYNTHETIC DATA NOTICE
---------------------
Nothing in this package touches real biological data. All fixtures produced by
:mod:`hybrid_adapter.synthetic` are artificial and their performance is not evidence
of biological predictive accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

SCHEMA_VERSION = "1.1.0"

# Canonical effect-unit labels. The adapter refuses to mix scales.
SUPPORTED_EFFECT_UNITS: Tuple[str, ...] = (
    "log2_fold_change",
    "log_fold_change",
    "delta_log1p_normalized",
)
SUPPORTED_EXPRESSION_UNITS: Tuple[str, ...] = (
    "log1p_cpm",
    "log1p_counts",
    "z_score",
)

# Availability-mask keys. Each is a scalar bool describing whether an input
# modality is usable for the whole batch.
SOURCE_MASK_KEYS: Tuple[str, ...] = (
    "gears_effects",
    "gears_embeddings",
    "scgpt_target",
    "scgpt_response",
    "control_features",
)

REQUIRED_BATCH_KEYS: Tuple[str, ...] = (
    "context_ids",
    "perturbation_ids",
    "response_gene_ids",
    "stat_effects",
    "scgpt_target_embeddings",
    "scgpt_response_embeddings",
    "control_features",
    "control_gene_expression",
    "effect_units",
)

OPTIONAL_BATCH_KEYS: Tuple[str, ...] = (
    "gears_effects",
    "gears_embeddings",
    "stat_coverage",
    "expression_units",
    "availability_masks",
    # --- training-side keys (ignored by forward, consumed by the loss) ---
    "target_effects",        # measured effects for TRAINING perturbations only
    "eligibility_mask",      # optional 0/1 mask for the global loss
    "de_mask",               # optional 0/1 DE-eligibility mask
    "artificial",            # synthetic-data labeling string
    # --- H7 subpopulation-mode extensions (opt-in via config) ---
    "phase_proportions",     # [B, K] composition of each context
    "phase_stat_effects",    # [B, K, G] per-phase statistical baseline
    "phase_labels",          # [K] phase names, e.g. ["G1", "S", "G2M", "M"]
    "compsim_weights",       # [B, K] composition similarity to reference lines
    "ground_truth_alpha",    # synthetic-only diagnostic (ignored by forward)
    # --- gated multi-context fusion extension (opt-in via config) ---
    "gears_embeddings_multi",  # [B, num_gears_sources, gears_dim] per-cell-type GEARS embeddings
    "gears_source_mask",       # [B, num_gears_sources] bool/0-1, which sources are real (not missing)
    "context_embedding",       # [B, context_dim] target-context embedding "c"
)

REQUIRED_SUBPOPULATION_KEYS: Tuple[str, ...] = (
    "phase_proportions",
    "phase_stat_effects",
)


class BatchValidationError(ValueError):
    """Raised when a batch violates the adapter's input contract."""


@dataclass(frozen=True)
class FeatureSpec:
    """Declared feature layout for the gene-wise residual MLP (Module D).

    ``input_dim`` is computed from the spec, never hard-coded, per the development
    assignment ("Compute input_dim from the implemented feature schema").
    """

    context_dim: int = 64            # Module C output h
    response_gene_dim: int = 32      # projected scGPT response-gene embedding
    basal_dim: int = 1               # basal expression for the gene
    stat_effect_dim: int = 1         # statistical predicted effect
    gears_effect_dim: int = 1        # optional GEARS predicted effect (zero-filled)
    indicator_dim: int = 5           # availability + coverage indicators

    @property
    def input_dim(self) -> int:
        return (
            self.context_dim
            + self.response_gene_dim
            + self.basal_dim
            + self.stat_effect_dim
            + self.gears_effect_dim
            + self.indicator_dim
        )


DEFAULT_FEATURE_SPEC = FeatureSpec()

# Indicator layout (order matters; documented for reproducibility):
#   0: has_gears_effect      (source availability)
#   1: has_gears_embedding   (source availability)
#   2: has_scgpt             (source availability, target or response)
#   3: has_control           (source availability)
#   4: stat_coverage         (per-gene coverage of the statistical baseline)
INDICATOR_NAMES: Tuple[str, ...] = (
    "has_gears_effect",
    "has_gears_embedding",
    "has_scgpt",
    "has_control",
    "stat_coverage",
)


@dataclass
class AdapterConfig:
    """Configuration for :class:`hybrid_adapter.model.HybridAdapter`."""

    # Input modality dimensions
    gears_dim: int = 32
    scgpt_dim: int = 512
    control_dim: int = 32
    latent_dim: int = 32
    fusion_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1

    # "both" | "scgpt_only" | "gears_only" | "auto"
    # "auto" selects per-batch based on availability masks.
    perturbation_mode: str = "auto"

    # Units the model expects; batches declaring other units are rejected.
    effect_units: str = "log2_fold_change"
    expression_units: str = "log1p_cpm"

    # --- H7 subpopulation extension ---
    # response_mode: "whole_line" (docx spec, M0) or "subpopulation" (M1-M4).
    response_mode: str = "whole_line"
    n_phases: int = 4
    # alpha_mode: "global" (M1) | "phase_shared" (M2/M4) | "phase_per_line" (M3)
    alpha_mode: str = "phase_shared"
    # M4: gate the invariant component by composition similarity.
    composition_weighting: bool = False

    # --- Gated multi-context fusion extension (system architecture Phase 2) ---
    # When enabled, Module A additionally accepts K per-cell-type GEARS
    # embeddings (one per pretrained GEARS checkpoint, e.g. k562/hepg2) and
    # fuses them via context-conditioned gating instead of using a single
    # gears_embeddings vector. See hybrid_adapter.gated_fusion.
    #
    # gears_source_dims is a tuple of per-source embedding widths, in a fixed
    # order the caller must use consistently (e.g. (64, 32) for (k562, hepg2) -
    # the two real per-cell-type GEARS checkpoints in this project do NOT
    # share an embedding width, confirmed directly; treated as the general
    # case, not an edge case). num_gears_sources is derived from it.
    use_gated_fusion: bool = False
    gears_source_dims: Tuple[int, ...] = (32, 32)
    context_dim: int = 32

    @property
    def num_gears_sources(self) -> int:
        return len(self.gears_source_dims)

    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)

    # Inference behaviour
    gene_chunk_size: int = 4096

    def validate(self) -> None:
        if self.perturbation_mode not in ("both", "scgpt_only", "gears_only", "auto"):
            raise ValueError(f"Unknown perturbation_mode: {self.perturbation_mode!r}")
        if self.effect_units not in SUPPORTED_EFFECT_UNITS:
            raise ValueError(
                f"Unsupported effect_units {self.effect_units!r}; "
                f"expected one of {SUPPORTED_EFFECT_UNITS}"
            )
        if self.expression_units not in SUPPORTED_EXPRESSION_UNITS:
            raise ValueError(
                f"Unsupported expression_units {self.expression_units!r}; "
                f"expected one of {SUPPORTED_EXPRESSION_UNITS}"
            )
        if self.response_mode not in ("whole_line", "subpopulation"):
            raise ValueError(f"Unknown response_mode: {self.response_mode!r}")
        if self.response_mode == "subpopulation":
            if self.alpha_mode not in ("global", "phase_shared", "phase_per_line"):
                raise ValueError(f"Unknown alpha_mode: {self.alpha_mode!r}")
            if self.n_phases < 1:
                raise ValueError("n_phases must be >= 1")
            if self.alpha_mode == "phase_shared" and self.composition_weighting:
                # M4 is phase_shared alpha + composition weighting; allowed.
                pass
        if self.gene_chunk_size < 1:
            raise ValueError("gene_chunk_size must be >= 1")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.use_gated_fusion:
            if self.num_gears_sources < 2:
                raise ValueError(
                    "num_gears_sources must be >= 2 when use_gated_fusion is enabled "
                    "(fusing a single source is just the existing gears_only path)."
                )
            if self.context_dim < 1:
                raise ValueError("context_dim must be >= 1")


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise BatchValidationError(
                f"Batch field {name!r} could not be interpreted as a tensor: {exc}"
            ) from exc
    if value.dtype != torch.float32:
        value = value.float()
    return value


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise BatchValidationError(
            f"Batch field {name!r} contains NaN or infinity; "
            "inputs must be finite before inference."
        )


def _check_shape(tensor: torch.Tensor, name: str, expected: Sequence[Optional[int]]) -> None:
    if tuple(tensor.shape) != tuple(expected):
        raise BatchValidationError(
            f"Batch field {name!r} has shape {tuple(tensor.shape)}, "
            f"expected {tuple(expected)}."
        )


def validate_batch(
    batch: Mapping[str, Any],
    config: AdapterConfig,
    require_subpopulation: Optional[bool] = None,
) -> Dict[str, Any]:
    """Validate a batch against the adapter contract.

    Returns a normalised dict containing validated tensors and resolved
    availability masks. Raises :class:`BatchValidationError` on any violation.

    Parameters
    ----------
    batch:
        Mapping following the documented input schema.
    config:
        The adapter configuration (defines expected units and mode).
    require_subpopulation:
        Force-require (True) or forbid (False) the H7 subpopulation fields.
        Defaults to ``config.response_mode == "subpopulation"``.
    """
    config.validate()
    if require_subpopulation is None:
        require_subpopulation = config.response_mode == "subpopulation"

    unknown = set(batch.keys()) - set(REQUIRED_BATCH_KEYS) - set(OPTIONAL_BATCH_KEYS)
    if unknown:
        raise BatchValidationError(f"Unknown batch keys: {sorted(unknown)}")

    missing = [k for k in REQUIRED_BATCH_KEYS if k not in batch]
    if missing:
        raise BatchValidationError(f"Missing required batch keys: {missing}")

    # --- Units ---------------------------------------------------------
    effect_units = batch["effect_units"]
    if effect_units != config.effect_units:
        raise BatchValidationError(
            f"Incompatible effect units: batch declares {effect_units!r} but the "
            f"model expects {config.effect_units!r}. Refusing to silently mix scales."
        )
    expression_units = batch.get("expression_units", config.expression_units)
    if expression_units != config.expression_units:
        raise BatchValidationError(
            f"Incompatible expression units: batch declares {expression_units!r} but "
            f"the model expects {config.expression_units!r}."
        )

    # --- Identifiers and gene axis --------------------------------------
    context_ids = list(batch["context_ids"])
    perturbation_ids = list(batch["perturbation_ids"])
    B = len(context_ids)
    if B == 0:
        raise BatchValidationError("Batch is empty (B == 0).")
    if len(perturbation_ids) != B:
        raise BatchValidationError(
            f"perturbation_ids has length {len(perturbation_ids)}, expected {B} "
            "(one per context-perturbation pair)."
        )

    gene_ids = list(batch["response_gene_ids"])
    G = len(gene_ids)
    if G == 0:
        raise BatchValidationError("response_gene_ids is empty (G == 0).")
    if len(set(gene_ids)) != G:
        raise BatchValidationError("response_gene_ids contains duplicates.")
    if not isinstance(batch["response_gene_ids"], (list, tuple)) and (
        batch["response_gene_ids"].dim() != 1
    ):
        # response_gene_ids may be a 1-D tensor; anything else is ambiguous ordering.
        raise BatchValidationError(
            "response_gene_ids must be a 1-D shared gene axis; per-row gene axes "
            "are not supported because gene ordering must be consistent."
        )

    # --- Core tensors ----------------------------------------------------
    stat_effects = _as_tensor(batch["stat_effects"], "stat_effects")
    _check_shape(stat_effects, "stat_effects", (B, G))
    _check_finite(stat_effects, "stat_effects")

    scgpt_target = _as_tensor(batch["scgpt_target_embeddings"], "scgpt_target_embeddings")
    _check_shape(scgpt_target, "scgpt_target_embeddings", (B, config.scgpt_dim))
    _check_finite(scgpt_target, "scgpt_target_embeddings")

    scgpt_response = _as_tensor(
        batch["scgpt_response_embeddings"], "scgpt_response_embeddings"
    )
    _check_shape(scgpt_response, "scgpt_response_embeddings", (B, G, config.scgpt_dim))
    _check_finite(scgpt_response, "scgpt_response_embeddings")

    control_features = _as_tensor(batch["control_features"], "control_features")
    _check_shape(control_features, "control_features", (B, config.control_dim))
    _check_finite(control_features, "control_features")

    control_gene_expression = _as_tensor(
        batch["control_gene_expression"], "control_gene_expression"
    )
    _check_shape(control_gene_expression, "control_gene_expression", (B, G))
    _check_finite(control_gene_expression, "control_gene_expression")

    # --- Optional GEARS inputs -------------------------------------------
    gears_effects: Optional[torch.Tensor] = None
    if batch.get("gears_effects") is not None:
        gears_effects = _as_tensor(batch["gears_effects"], "gears_effects")
        _check_shape(gears_effects, "gears_effects", (B, G))
        _check_finite(gears_effects, "gears_effects")

    gears_embeddings: Optional[torch.Tensor] = None
    if batch.get("gears_embeddings") is not None:
        gears_embeddings = _as_tensor(batch["gears_embeddings"], "gears_embeddings")
        _check_shape(gears_embeddings, "gears_embeddings", (B, config.gears_dim))
        _check_finite(gears_embeddings, "gears_embeddings")

    # --- Gated multi-context fusion inputs --------------------------------
    # A LIST of K tensors, not one stacked tensor: per-cell-type GEARS
    # checkpoints need not share an embedding width (k562 is 64-d, hepg2 is
    # 32-d in this project - confirmed, not an edge case), so stacking into
    # a single [B, K, D] tensor is not generally possible.
    gears_embeddings_multi: Optional[List[torch.Tensor]] = None
    gears_source_mask: Optional[torch.Tensor] = None
    context_embedding: Optional[torch.Tensor] = None
    if config.use_gated_fusion:
        if batch.get("gears_embeddings_multi") is None or batch.get("context_embedding") is None:
            raise BatchValidationError(
                "use_gated_fusion=True requires both 'gears_embeddings_multi' and "
                "'context_embedding' in the batch."
            )
    if batch.get("gears_embeddings_multi") is not None:
        raw_multi = batch["gears_embeddings_multi"]
        if not isinstance(raw_multi, (list, tuple)):
            raise BatchValidationError(
                "gears_embeddings_multi must be a list/tuple of K per-source tensors "
                f"(one per cell type), got {type(raw_multi)}."
            )
        if len(raw_multi) != config.num_gears_sources:
            raise BatchValidationError(
                f"gears_embeddings_multi has {len(raw_multi)} sources, expected "
                f"{config.num_gears_sources} (config.gears_source_dims)."
            )
        gears_embeddings_multi = []
        for k, (tensor, expected_dim) in enumerate(zip(raw_multi, config.gears_source_dims)):
            t = _as_tensor(tensor, f"gears_embeddings_multi[{k}]")
            _check_shape(t, f"gears_embeddings_multi[{k}]", (B, expected_dim))
            _check_finite(t, f"gears_embeddings_multi[{k}]")
            gears_embeddings_multi.append(t)
        if batch.get("gears_source_mask") is not None:
            gears_source_mask = _as_tensor(batch["gears_source_mask"], "gears_source_mask")
            _check_shape(gears_source_mask, "gears_source_mask", (B, config.num_gears_sources))
            if ((gears_source_mask != 0) & (gears_source_mask != 1)).any():
                raise BatchValidationError("gears_source_mask must be binary (0/1).")
    if batch.get("context_embedding") is not None:
        context_embedding = _as_tensor(batch["context_embedding"], "context_embedding")
        _check_shape(context_embedding, "context_embedding", (B, config.context_dim))
        _check_finite(context_embedding, "context_embedding")

    stat_coverage: Optional[torch.Tensor] = None
    if batch.get("stat_coverage") is not None:
        stat_coverage = _as_tensor(batch["stat_coverage"], "stat_coverage")
        _check_shape(stat_coverage, "stat_coverage", (B, G))
        _check_finite(stat_coverage, "stat_coverage")
        if ((stat_coverage != 0) & (stat_coverage != 1)).any():
            raise BatchValidationError("stat_coverage must be binary (0/1).")

    # --- Availability masks ----------------------------------------------
    masks = {key: True for key in SOURCE_MASK_KEYS}
    provided_masks = batch.get("availability_masks") or {}
    if not isinstance(provided_masks, Mapping):
        raise BatchValidationError("availability_masks must be a mapping of bools.")
    unknown_mask_keys = set(provided_masks.keys()) - set(SOURCE_MASK_KEYS)
    if unknown_mask_keys:
        raise BatchValidationError(
            f"Unknown availability mask keys: {sorted(unknown_mask_keys)}; "
            f"expected a subset of {SOURCE_MASK_KEYS}."
        )
    for key, value in provided_masks.items():
        if not isinstance(value, (bool, int)) or isinstance(value, torch.Tensor):
            raise BatchValidationError(
                f"availability_masks[{key!r}] must be a scalar bool, got {type(value)}."
            )
        masks[key] = bool(value)

    # Presence of a tensor implies availability unless a mask explicitly denies it.
    masks["gears_effects"] = masks["gears_effects"] and gears_effects is not None
    masks["gears_embeddings"] = masks["gears_embeddings"] and gears_embeddings is not None
    masks["scgpt_target"] = masks["scgpt_target"] and scgpt_target is not None
    masks["scgpt_response"] = masks["scgpt_response"] and scgpt_response is not None
    masks["control_features"] = masks["control_features"] and control_features is not None

    if config.perturbation_mode == "gears_only" and not masks["gears_embeddings"]:
        raise BatchValidationError(
            "perturbation_mode='gears_only' requires GEARS embeddings, but the "
            "availability mask reports them missing."
        )
    if config.perturbation_mode == "scgpt_only" and not (
        masks["scgpt_target"] and masks["scgpt_response"]
    ):
        raise BatchValidationError(
            "perturbation_mode='scgpt_only' requires scGPT target and response "
            "embeddings, but the availability masks report them missing."
        )

    # --- H7 subpopulation fields -----------------------------------------
    phase_proportions: Optional[torch.Tensor] = None
    phase_stat_effects: Optional[torch.Tensor] = None
    phase_labels: Optional[List[str]] = None
    compsim_weights: Optional[torch.Tensor] = None

    has_subpop_fields = all(k in batch for k in REQUIRED_SUBPOPULATION_KEYS)
    if require_subpopulation and not has_subpop_fields:
        raise BatchValidationError(
            f"response_mode='subpopulation' requires batch keys "
            f"{REQUIRED_SUBPOPULATION_KEYS}."
        )
    if has_subpop_fields:
        # In subpopulation mode K must match the configured phase count; in
        # whole-line mode (where phase fields are informational, e.g. for the
        # M0 ablation on H7-structured data) K is inferred from the batch.
        if require_subpopulation:
            K = config.n_phases
        else:
            K = int(torch.as_tensor(batch["phase_proportions"]).shape[1])
        phase_proportions = _as_tensor(batch["phase_proportions"], "phase_proportions")
        _check_shape(phase_proportions, "phase_proportions", (B, K))
        _check_finite(phase_proportions, "phase_proportions")
        if (phase_proportions < 0).any() or (phase_proportions > 1).any():
            raise BatchValidationError("phase_proportions must lie in [0, 1].")
        sums = phase_proportions.sum(dim=1)
        if not torch.allclose(sums, torch.ones(B), atol=1e-4):
            raise BatchValidationError(
                "phase_proportions rows must sum to 1 "
                f"(got sums in [{sums.min():.6f}, {sums.max():.6f}])."
            )

        phase_stat_effects = _as_tensor(
            batch["phase_stat_effects"], "phase_stat_effects"
        )
        _check_shape(phase_stat_effects, "phase_stat_effects", (B, K, G))
        _check_finite(phase_stat_effects, "phase_stat_effects")

        raw_labels = batch.get("phase_labels")
        if raw_labels is not None:
            phase_labels = [str(x) for x in raw_labels]
            if len(phase_labels) != K:
                raise BatchValidationError(
                    f"phase_labels has length {len(phase_labels)}, expected {K}."
                )

        if batch.get("compsim_weights") is not None:
            compsim_weights = _as_tensor(batch["compsim_weights"], "compsim_weights")
            _check_shape(compsim_weights, "compsim_weights", (B, K))
            _check_finite(compsim_weights, "compsim_weights")
            if (compsim_weights < 0).any() or (compsim_weights > 1).any():
                raise BatchValidationError("compsim_weights must lie in [0, 1].")

    return {
        "context_ids": context_ids,
        "perturbation_ids": perturbation_ids,
        "response_gene_ids": gene_ids,
        "effect_units": effect_units,
        "expression_units": expression_units,
        "stat_effects": stat_effects,
        "scgpt_target_embeddings": scgpt_target,
        "scgpt_response_embeddings": scgpt_response,
        "control_features": control_features,
        "control_gene_expression": control_gene_expression,
        "gears_effects": gears_effects,
        "gears_embeddings": gears_embeddings,
        "stat_coverage": stat_coverage,
        "availability_masks": masks,
        "phase_proportions": phase_proportions,
        "phase_stat_effects": phase_stat_effects,
        "phase_labels": phase_labels,
        "compsim_weights": compsim_weights,
        "gears_embeddings_multi": gears_embeddings_multi,
        "gears_source_mask": gears_source_mask,
        "context_embedding": context_embedding,
    }


def check_output_finite(output: Mapping[str, torch.Tensor]) -> None:
    """Verify that every tensor in a model output dict is finite."""
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise BatchValidationError(
                f"Model output {key!r} contains NaN or infinity."
            )


def describe_schema() -> str:
    """Human-readable description of the batch contract (for docs/manifests)."""
    lines = [
        f"hybrid_adapter batch schema v{SCHEMA_VERSION}",
        f"  required keys : {', '.join(REQUIRED_BATCH_KEYS)}",
        f"  optional keys : {', '.join(OPTIONAL_BATCH_KEYS)}",
        f"  effect units  : {', '.join(SUPPORTED_EFFECT_UNITS)}",
        f"  expression units: {', '.join(SUPPORTED_EXPRESSION_UNITS)}",
        f"  residual input_dim = {DEFAULT_FEATURE_SPEC.input_dim} "
        f"(indicators: {', '.join(INDICATOR_NAMES)})",
    ]
    return "\n".join(lines)
