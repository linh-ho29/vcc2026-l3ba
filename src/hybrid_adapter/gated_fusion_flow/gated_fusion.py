"""Gated latent multi-context fusion (system architecture, Phase 2 step 2/3).

Aggregates the pre-extracted perturbation vectors from K independently
pretrained, per-cell-type GEARS models into a single fused representation,
weighted by how relevant each cell type's response is to the current target
context::

    alpha_k = Softmax( W_gate . [ e_p^(k) || c ] )
    e_p^(fused) = sum_k alpha_k . Linear(e_p^(k))

Per-source input projection
----------------------------
The two per-cell-type GEARS checkpoints actually available in this project
do NOT share an embedding dimension: k562's are 64-d, hepg2's are 32-d (see
``gears_track/per_celltype/{k562,hepg2}/embeddings/`` on the ``GEARS``
branch - confirmed directly, not assumed). Independently trained GEARS
models have no reason to agree on a hidden size, so this is treated as the
general case, not an edge case: each source ``k`` first gets its own input
projection ``Linear(source_dims[k], common_dim)``, and the shared gate +
``Linear(common_dim, fused_dim)`` from the formula above operate on the
common-dimension projections. This is an addition on top of the formula (it
specifies the fusion math, not how to reconcile mismatched input widths),
documented here rather than silently assumed away.

A source can be unavailable for a given (context, perturbation) row - e.g. a
cell type's GEARS model was never trained on that target gene. Per this
project's established convention (never silently zero-fill a missing
source), unavailable sources are excluded from the softmax via
``source_mask`` rather than contributing a zero vector that would otherwise
dilute the gate; if every source is unavailable for a row, this is a hard
error, not a silent zero-fused-vector.

SYNTHETIC PHASE ONLY: nothing in this module touches real biological data on
its own; see ``hybrid_adapter.synthetic`` for fixtures and the package root
docstring for the project's data-hygiene rules. A forward-pass-only
compatibility check against the real per-cell-type embeddings (no training,
no gradient updates) lives outside this package - see the GEARS branch's
``gears_track/`` tooling.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..schemas import BatchValidationError


class GatedMultiContextFusion(nn.Module):
    """Fuse K per-cell-type GEARS perturbation embeddings via context gating.

    Parameters
    ----------
    source_dims:
        Per-source embedding dimensionality, one entry per cell type ``k``
        (e.g. ``[64, 32]`` for k562 and hepg2). May differ across sources.
    context_dim:
        Dimensionality of the context embedding ``c``.
    fused_dim:
        Output dimensionality of the fused vector ``e_p^(fused)``.
    common_dim:
        Shared dimension each source is projected to before gating and
        fusion. Defaults to ``fused_dim`` if not given.
    """

    def __init__(
        self,
        source_dims: Sequence[int],
        context_dim: int,
        fused_dim: int,
        common_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(source_dims) < 2:
            raise ValueError(
                "GatedMultiContextFusion requires >= 2 sources; with a single "
                "source there is nothing to gate between."
            )
        self.source_dims = list(source_dims)
        self.num_sources = len(source_dims)
        self.context_dim = context_dim
        self.fused_dim = fused_dim
        self.common_dim = common_dim if common_dim is not None else fused_dim

        self.input_proj = nn.ModuleList(
            [nn.Linear(d, self.common_dim) for d in self.source_dims]
        )
        self.gate = nn.Linear(self.common_dim + context_dim, 1)
        self.proj = nn.Linear(self.common_dim, fused_dim)

    def forward(
        self,
        source_embeddings: Sequence[Tensor],
        context_embedding: Tensor,
        source_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the fused perturbation vector and the gate weights used.

        Parameters
        ----------
        source_embeddings:
            Sequence of length ``K`` (one per cell type, in the same order
            as ``source_dims``); entry ``k`` has shape ``[B, source_dims[k]]``.
            Rows for an unavailable source should still be a finite
            placeholder value (e.g. zeros) - ``source_mask`` is what actually
            excludes them from the gate, not the values themselves.
        context_embedding:
            ``[B, context_dim]`` - the target context's embedding ``c``.
        source_mask:
            Optional ``[B, K]`` bool/0-1 tensor; ``True``/1 means that cell
            type's embedding is a real, trained value for this row's target
            gene (not a placeholder). If omitted, every source is assumed
            available.

        Returns
        -------
        fused:
            ``[B, fused_dim]`` - ``e_p^(fused)``.
        alpha:
            ``[B, K]`` - the softmax gate weights actually used (masked
            entries are exactly 0), returned for inspection/diagnostics.
        """
        K = self.num_sources
        if len(source_embeddings) != K:
            raise BatchValidationError(
                f"Expected {K} source embedding tensors, got {len(source_embeddings)}."
            )
        B = context_embedding.shape[0]
        if tuple(context_embedding.shape) != (B, self.context_dim):
            raise BatchValidationError(
                f"context_embedding must have shape [{B}, {self.context_dim}], "
                f"got {tuple(context_embedding.shape)}."
            )
        if not torch.isfinite(context_embedding).all():
            raise BatchValidationError("context_embedding contains NaN or infinity.")

        projected_common: List[Tensor] = []
        for k, (emb, expected_dim) in enumerate(zip(source_embeddings, self.source_dims)):
            if tuple(emb.shape) != (B, expected_dim):
                raise BatchValidationError(
                    f"source_embeddings[{k}] must have shape [{B}, {expected_dim}], "
                    f"got {tuple(emb.shape)}."
                )
            if not torch.isfinite(emb).all():
                raise BatchValidationError(f"source_embeddings[{k}] contains NaN or infinity.")
            projected_common.append(self.input_proj[k](emb))
        common = torch.stack(projected_common, dim=1)  # [B, K, common_dim]

        if source_mask is None:
            mask = torch.ones(B, K, dtype=torch.bool, device=context_embedding.device)
        else:
            mask = source_mask.bool()
            if tuple(mask.shape) != (B, K):
                raise BatchValidationError(
                    f"source_mask must have shape [{B}, {K}], got {tuple(mask.shape)}."
                )

        all_missing = ~mask.any(dim=1)
        if all_missing.any():
            bad_rows = torch.nonzero(all_missing, as_tuple=True)[0].tolist()
            raise BatchValidationError(
                "GatedMultiContextFusion: no available source for row(s) "
                f"{bad_rows} - every cell-type GEARS model lacks this target gene. "
                "This must be handled upstream (e.g. excluded from the batch), "
                "not silently zero-filled."
            )

        c_expanded = context_embedding.unsqueeze(1).expand(B, K, self.context_dim)
        gate_input = torch.cat([common, c_expanded], dim=-1)   # [B, K, common_dim + context_dim]
        logits = self.gate(gate_input).squeeze(-1)             # [B, K]
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(logits, dim=-1)                  # [B, K], masked entries -> 0

        projected = self.proj(common)                          # [B, K, fused_dim]
        fused = (alpha.unsqueeze(-1) * projected).sum(dim=1)    # [B, fused_dim]
        return fused, alpha
