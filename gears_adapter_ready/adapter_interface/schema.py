"""Array contract for a future adapter; no adapter is trained here."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ExternalEmbedding:
    identifiers: list[str]
    values: np.ndarray
    provenance: str
    context_id: Optional[str] = None


def align_external_embedding(embedding, desired_ids):
    """Missing identifiers are masked, never silently assumed covered."""
    if len(set(embedding.identifiers)) != len(embedding.identifiers):
        raise ValueError("External embedding identifiers must be unique")
    if embedding.values.ndim != 2 or len(embedding.values) != len(embedding.identifiers):
        raise ValueError("Embedding shape/identifier mismatch")
    lookup = {g:i for i,g in enumerate(embedding.identifiers)}
    values = np.zeros((len(desired_ids), embedding.values.shape[1]), np.float32)
    mask = np.array([g in lookup for g in desired_ids], bool)
    for i,g in enumerate(desired_ids):
        if mask[i]:
            values[i] = embedding.values[lookup[g]]
    return dict(values=values, coverage=mask, provenance=embedding.provenance)
