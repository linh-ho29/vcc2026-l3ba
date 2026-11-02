"""Phase 3 — M0-M4 ablation harness.

Trains/evaluates the five adapter variants on perturbation-holdout splits of
the Model B training line (proxy) and reports the ordering against the H7
prediction::

    M4 > M2 > M1 > M0,  with M3 ≈ M2 if claim 5 holds, M3 > M2 otherwise.

Interpretation rules (from the hypothesis document):
    * M4 does not beat M2  -> composition weighting (claim 3) adds nothing.
    * M2 does not beat M1  -> the identity gate (claim 5) adds nothing.
    * M1 does not beat M0  -> subpopulation resolution itself adds nothing;
      the whole hypothesis fails at its first step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from hybrid_adapter.model import HybridAdapter
from hybrid_adapter.schemas import AdapterConfig
from hybrid_adapter.train import TrainConfig, train_adapter

ABLATION_MODELS = ("M0", "M1", "M2", "M3", "M4")


def config_for_ablation(
    variant: str,
    n_phases: int = 4,
    base: Optional[AdapterConfig] = None,
) -> AdapterConfig:
    """Build the AdapterConfig for one ablation variant."""
    base = base or AdapterConfig()
    if variant == "M0":
        cfg = AdapterConfig(**{k: v for k, v in vars(base).items() if k != "feature_spec"})
        cfg.response_mode = "whole_line"
        cfg.n_phases = n_phases
        return cfg
    if variant not in ABLATION_MODELS:
        raise ValueError(f"Unknown ablation variant {variant!r}; expected one of {ABLATION_MODELS}.")
    alpha_mode = {"M1": "global", "M2": "phase_shared", "M3": "phase_per_line", "M4": "phase_shared"}[variant]
    composition = variant == "M4"
    return AdapterConfig(
        gears_dim=base.gears_dim,
        scgpt_dim=base.scgpt_dim,
        control_dim=base.control_dim,
        latent_dim=base.latent_dim,
        fusion_dim=base.fusion_dim,
        hidden_dim=base.hidden_dim,
        dropout=base.dropout,
        perturbation_mode=base.perturbation_mode,
        effect_units=base.effect_units,
        expression_units=base.expression_units,
        response_mode="subpopulation",
        n_phases=n_phases,
        alpha_mode=alpha_mode,
        composition_weighting=composition,
        gene_chunk_size=base.gene_chunk_size,
    )


def run_ablation(
    train_loader,
    validation_loader,
    n_phases: int = 4,
    train_config: Optional[TrainConfig] = None,
    variants: tuple = ABLATION_MODELS,
    seed: int = 0,
) -> Dict[str, Dict[str, object]]:
    """Train each variant and return final validation losses.

    Every variant is trained from its own freshly seeded model on identical
    data so the comparison isolates the architectural change.
    """
    torch.manual_seed(seed)
    results: Dict[str, Dict[str, object]] = {}
    for variant in variants:
        cfg = config_for_ablation(variant, n_phases=n_phases)
        model = HybridAdapter(cfg)
        tcfg = train_config or TrainConfig()
        history = train_adapter(model, train_loader, validation_loader, tcfg)
        results[variant] = {
            "final_val_loss": history.val_loss[-1] if history.val_loss else float("nan"),
            "best_val_loss": history.best_val_loss,
            "epochs_completed": history.epochs_completed,
            "stopped_early": history.stopped_early,
        }
    results["_ordering"] = _ordering(results, variants)
    return results


def _ordering(results: Dict[str, Dict[str, object]], variants: tuple) -> str:
    """Variants sorted by best validation loss (lower is better)."""
    ranked = sorted(
        (v for v in variants if v in results),
        key=lambda v: results[v]["best_val_loss"],
    )
    return " > ".join(ranked)


def interpret_ordering(ordering: str) -> Dict[str, bool]:
    """Apply the H7 interpretation rules to an observed ordering string."""
    parts = ordering.split(" > ")
    rank = {name: i for i, name in enumerate(parts)}

    def _beats(a: str, b: str) -> bool:
        return a in rank and b in rank and rank[a] < rank[b]

    return {
        "subpopulation_resolution_helps": _beats("M1", "M0"),
        "identity_gate_helps": _beats("M2", "M1"),
        "composition_weighting_helps": _beats("M4", "M2"),
        "shared_alpha_sufficient": abs(rank.get("M3", 0) - rank.get("M2", 0)) <= 1,
        "h7_predicted_ordering_holds": (
            _beats("M4", "M2") and _beats("M2", "M1") and _beats("M1", "M0")
        ),
    }
