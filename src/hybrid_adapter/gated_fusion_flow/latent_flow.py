"""Input conditioning, discrete latent flow, and decoder (system architecture,
Phase 2 steps 3-5).

A dimensional note, documented rather than silently resolved: the posted
diagram states ``z_0`` and the final ``z_pert`` are both in R^128, while the
input-conditioning layer and the flow block's internal linear are both
stated as 256-wide ("Linear 224->256", "Linear(256->256)"). Taken literally
these are inconsistent (a 256-wide state can't equal a 128-dim one without
a projection somewhere). Resolved here as: the flow operates entirely in
the stated 256-dim conditioned space (matching the explicit "Linear(256,
256)" at every step), with one explicit final 256->128 projection producing
``z_pert`` - so both explicit width labels in the diagram are honored
exactly, with the bridging layer made explicit instead of assumed away.

::

    [Input Conditioning Layer]
        h0 = LayerNorm(Linear([z0 (128) || e_p^(fused) (64) || c (32)]))   # -> 256

    [Discrete Latent Flow Block] (2-3 steps, explicit Euler - no ODE solver)
        cond_t = Linear([e_p^(fused) || c || time_embed(t)])               # -> 256
        z(t+1) = z(t) + Dropout(GELU(Linear(LayerNorm(z(t) + cond_t))))    # Pre-LN residual

    z_pert = Linear(z(T))                                                  # 256 -> 128

    [Light Decoder Head]
        y_hat_pert = Linear(z_pert)                                        # 128 -> G
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..schemas import BatchValidationError


class InputConditioningLayer(nn.Module):
    """``[z0 || e_p^(fused) || c] -> Linear -> LayerNorm -> h0 (flow_dim)``."""

    def __init__(self, z0_dim: int, fused_dim: int, context_dim: int, flow_dim: int) -> None:
        super().__init__()
        self.z0_dim, self.fused_dim, self.context_dim = z0_dim, fused_dim, context_dim
        self.proj = nn.Linear(z0_dim + fused_dim + context_dim, flow_dim)
        self.norm = nn.LayerNorm(flow_dim)

    def forward(self, z0: Tensor, e_p_fused: Tensor, context: Tensor) -> Tensor:
        for name, t, expected in (("z0", z0, self.z0_dim), ("e_p_fused", e_p_fused, self.fused_dim),
                                   ("context", context, self.context_dim)):
            if t.shape[-1] != expected:
                raise BatchValidationError(f"{name} must have width {expected}, got {t.shape[-1]}.")
        return self.norm(self.proj(torch.cat([z0, e_p_fused, context], dim=-1)))


class DiscreteLatentFlow(nn.Module):
    """2-3 step explicit-Euler residual flow, conditioned at every step.

    Parameters
    ----------
    flow_dim:
        Width the flow operates in internally (256 in the diagram).
    output_dim:
        Width of the final projected latent ``z_pert`` (128 in the diagram).
    fused_dim, context_dim:
        Widths of the per-step conditioning inputs.
    n_steps:
        Number of discrete Euler steps (2 or 3 per the architecture).
    """

    def __init__(
        self,
        flow_dim: int = 256,
        output_dim: int = 128,
        fused_dim: int = 64,
        context_dim: int = 32,
        n_steps: int = 3,
        time_embed_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_steps not in (2, 3):
            raise ValueError(f"n_steps must be 2 or 3 per the architecture, got {n_steps}")
        self.flow_dim, self.output_dim = flow_dim, output_dim
        self.fused_dim, self.context_dim = fused_dim, context_dim
        self.n_steps = n_steps

        self.time_embed = nn.Embedding(n_steps, time_embed_dim)
        self.cond_proj = nn.Linear(fused_dim + context_dim + time_embed_dim, flow_dim)
        self.pre_norm = nn.LayerNorm(flow_dim)
        self.net = nn.Sequential(
            nn.Linear(flow_dim, flow_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Linear(flow_dim, output_dim)

    def _step(self, z: Tensor, e_p_fused: Tensor, context: Tensor, t: int) -> Tensor:
        B = z.shape[0]
        t_idx = torch.full((B,), t, dtype=torch.long, device=z.device)
        t_emb = self.time_embed(t_idx)
        cond = self.cond_proj(torch.cat([e_p_fused, context, t_emb], dim=-1))
        h = self.pre_norm(z + cond)
        delta = self.net(h)
        return z + delta

    def forward(self, h0: Tensor, e_p_fused: Tensor, context: Tensor) -> Tensor:
        if h0.shape[-1] != self.flow_dim:
            raise BatchValidationError(f"h0 must have width {self.flow_dim}, got {h0.shape[-1]}.")
        if e_p_fused.shape[-1] != self.fused_dim:
            raise BatchValidationError(
                f"e_p_fused must have width {self.fused_dim}, got {e_p_fused.shape[-1]}."
            )
        if context.shape[-1] != self.context_dim:
            raise BatchValidationError(
                f"context must have width {self.context_dim}, got {context.shape[-1]}."
            )
        z = h0
        for t in range(self.n_steps):
            z = self._step(z, e_p_fused, context, t)
        return self.out_proj(z)  # z_pert, [B, output_dim]


class LatentDecoder(nn.Module):
    """Light decoder head: ``z_pert [B, latent_dim] -> y_hat_pert [B, G]``."""

    def __init__(self, latent_dim: int, n_genes: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(latent_dim, n_genes)
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_genes),
            )

    def forward(self, z_pert: Tensor) -> Tensor:
        return self.net(z_pert)
