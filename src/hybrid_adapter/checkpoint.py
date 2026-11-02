"""Checkpoint save/load with a versioned manifest.

The adapter checkpoint is self-contained: it can be loaded without installing
or initialising GEARS or scGPT, because only cached feature tensors are used as
inputs and no foundation-model weights are bundled.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .model import MODEL_VERSION, HybridAdapter
from .schemas import SCHEMA_VERSION, AdapterConfig

CHECKPOINT_FORMAT = "hybrid_adapter_checkpoint"


def _dependency_versions() -> Dict[str, str]:
    def _ver(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except Exception:
            return "unknown"

    return {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "hybrid_adapter": MODEL_VERSION,
    }


def _state_dict_checksum(state_dict: Dict[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_adapter(
    model: HybridAdapter,
    config: AdapterConfig,
    path: str | Path,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save model weights, config, and a versioned manifest.

    The manifest records architecture configuration, input feature dimensions,
    required/optional modalities, gene-axis requirements, expression-effect
    units, model version, checkpoint checksum, and dependency versions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    checksum = _state_dict_checksum(state_dict)

    manifest: Dict[str, Any] = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "architecture": {
            "gears_dim": config.gears_dim,
            "scgpt_dim": config.scgpt_dim,
            "control_dim": config.control_dim,
            "latent_dim": config.latent_dim,
            "fusion_dim": config.fusion_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "perturbation_mode": config.perturbation_mode,
            "response_mode": config.response_mode,
            "n_phases": config.n_phases,
            "alpha_mode": config.alpha_mode,
            "composition_weighting": config.composition_weighting,
            "residual_input_dim": config.feature_spec.input_dim,
        },
        "input_feature_dimensions": {
            "gears_embedding": config.gears_dim,
            "scgpt_embedding": config.scgpt_dim,
            "control_features": config.control_dim,
        },
        "required_modalities": [
            "stat_effects",
            "scgpt_target_embeddings",
            "scgpt_response_embeddings",
            "control_features",
            "control_gene_expression",
        ],
        "optional_modalities": ["gears_effects", "gears_embeddings", "stat_coverage"],
        "gene_axis_requirements": {
            "ordering": "single shared 1-D response_gene_ids axis",
            "must_match_across": [
                "stat_effects",
                "control_gene_expression",
                "scgpt_response_embeddings",
            ],
        },
        "effect_units": config.effect_units,
        "expression_units": config.expression_units,
        "state_dict_checksum": checksum,
        "dependency_versions": _dependency_versions(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    torch.save(
        {"manifest": manifest, "state_dict": state_dict, "config": vars(config) | {}},
        path,
    )
    return manifest


def load_adapter(path: str | Path, device: str = "cpu") -> tuple:
    """Load an adapter checkpoint.

    Returns ``(model, manifest)``. Raises on checksum mismatch. Works without
    GEARS or scGPT installed.
    """
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    manifest = payload["manifest"]
    if manifest.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Not a hybrid_adapter checkpoint (format={manifest.get('checkpoint_format')!r})."
        )

    config = AdapterConfig(**payload["config"])
    model = HybridAdapter(config)
    state_dict = payload["state_dict"]

    # Verify checksum before trusting the weights.
    checksum = _state_dict_checksum(state_dict)
    if checksum != manifest.get("state_dict_checksum"):
        raise ValueError(
            "Checkpoint checksum mismatch: manifest records "
            f"{manifest.get('state_dict_checksum')}, computed {checksum}. "
            "The file may be corrupted or tampered with."
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, manifest


def save_training_state(
    path: str | Path,
    model: HybridAdapter,
    optimizer,
    epoch: int,
    history,
    best_model_state: Dict[str, Any],
    best_optimizer_state: Dict[str, Any],
) -> None:
    """Persist full resumable training state (used by ``train_adapter``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "training_state_format": "hybrid_adapter_train_state",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": {
                "train_loss": list(history.train_loss),
                "val_loss": list(history.val_loss),
                "epochs_completed": history.epochs_completed,
                "best_val_loss": history.best_val_loss,
                "best_epoch": history.best_epoch,
                "stopped_early": history.stopped_early,
            },
            "best_model_state": best_model_state,
            "best_optimizer_state": best_optimizer_state,
        },
        path,
    )


def load_training_state(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load full training state for resume."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("training_state_format") != "hybrid_adapter_train_state":
        raise ValueError(f"Not a hybrid_adapter training state file: {path}")
    return payload
