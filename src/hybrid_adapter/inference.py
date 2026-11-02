"""Deterministic inference utilities for the hybrid adapter."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from .model import HybridAdapter
from .schemas import AdapterConfig


@torch.no_grad()
def predict_corrected_effects(
    model: HybridAdapter,
    batch: Mapping[str, Any],
    gene_chunk_size: int | None = None,
    device: str | None = None,
) -> Dict[str, Any]:
    """Predict corrected perturbation effects deterministically.

    The model is switched to eval mode, inputs are validated, computation is
    float32, and the gene dimension is processed in configurable chunks so that
    18,533-gene response axes stay memory-predictable on CPU.

    Returns the model output dict (``corrected_effects``, ``baseline_effects``,
    ``residual_effects``, plus identifiers and, in subpopulation mode, the
    per-phase predictions and effective alphas).
    """
    was_training = model.training
    model.eval()
    try:
        if gene_chunk_size is not None and gene_chunk_size != model.config.gene_chunk_size:
            # Temporarily override chunking without mutating the caller's config.
            original = model.config.gene_chunk_size
            model.config.gene_chunk_size = gene_chunk_size
            try:
                if device is not None:
                    model = model.to(device)
                return model.forward(batch)
            finally:
                model.config.gene_chunk_size = original
        if device is not None:
            model = model.to(device)
        return model.forward(batch)
    finally:
        if was_training:
            model.train()


def make_default_config(**overrides) -> AdapterConfig:
    """Convenience constructor for the default (docx-spec) configuration."""
    config = AdapterConfig()
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown AdapterConfig field: {key!r}")
        setattr(config, key, value)
    config.validate()
    return config
