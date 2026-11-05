"""Compute-efficient Perturbation Virtual Cell model.

Pipeline: frozen scGPT + frozen per-cell-line GEARS checkpoints (K562,
HepG2, RPE, Jurkat) are offline feature extractors only - nothing in this
file trains them. Online training runs entirely on their low-dimensional
outputs (a control-cell latent ``z_0``, four 64-d GEARS perturbation
vectors, and a gene-space statistical delta), through a 2-step discrete
latent velocity flow and a zero-initialized residual decoder. Designed to
fit under 2 GB VRAM on a free T4: every online module operates on vectors
of width <= 512, batch size 256-512, mixed precision (see ``train.py``).

Final prediction is anchored to the existing statistical baseline, not
learned from scratch:

::

    y_pred = x_ctrl + delta_stat + delta_neural

with ``delta_neural`` exactly zero at initialization (``ResidualDecoder``'s
final layer is zero-initialized) - training starts perfectly at the
already-validated statistical baseline and only has to learn a correction,
never has to re-learn the baseline itself.

One dimensional ambiguity in the spec, resolved explicitly rather than
silently assumed: ``DiscreteLatentFlow``'s update rule states
``v_t = MLP_step_t(LayerNorm(z_t) + cond)`` with ``cond`` 256-wide and
``LayerNorm(z_t)`` ``d_latent``-wide (not generally 256) - these can't be
added directly unless ``d_latent == 256``. Resolved here by applying the
addition *inside* ``MLP_step_t``, right after its first ``Linear(d_latent,
256)`` (the only point where both operands are actually 256-wide), not
before it:

::

    h = Linear_in(LayerNorm(z_t))   # d_latent -> 256
    h = h + cond                    # now both 256-wide
    v_t = Linear_out(Dropout(GELU(h)))   # 256 -> d_latent

This keeps every stated layer (``Linear(d_latent,256) -> GELU ->
Dropout(0.1) -> Linear(256,d_latent)``) exactly as specified; only the
*position* of the addition is pinned down.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Module A
# ---------------------------------------------------------------------------

class ContextGatedGEARSFusion(nn.Module):
    """``e_p_stack [B,4,64], c_target [B,d_context] -> e_p_fused [B,64]``.

    Gated mixture over the 4 cell-line GEARS vectors (softmax weights from
    a 2-layer MLP conditioned on the target context), each line projected
    through its own ``Linear(64,64)`` (``per_line_projection=True``, the
    default - set ``False`` for one shared projection across all 4 lines),
    plus an invariant skip connection from the raw (unprojected,
    un-gated) per-line mean, so the fused output never fully loses the
    "what do all 4 lines agree on" signal even if the gate collapses onto
    one line.
    """

    N_LINES = 4
    PERT_DIM = 64

    def __init__(
        self,
        d_context: int = 32,
        gate_hidden_dim: int = 64,
        per_line_projection: bool = True,
    ) -> None:
        super().__init__()
        self.d_context = d_context
        self.per_line_projection = per_line_projection

        gate_in_dim = self.PERT_DIM + d_context
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )

        if per_line_projection:
            self.proj = nn.ModuleList(
                [nn.Linear(self.PERT_DIM, self.PERT_DIM) for _ in range(self.N_LINES)]
            )
        else:
            self.proj = nn.Linear(self.PERT_DIM, self.PERT_DIM)

        self.skip_proj = nn.Linear(self.PERT_DIM, self.PERT_DIM)
        self.norm = nn.LayerNorm(self.PERT_DIM)

    def forward(self, e_p_stack: Tensor, c_target: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns ``(e_p_fused [B,64], alpha [B,4,1])``."""
        if e_p_stack.dim() != 3 or e_p_stack.shape[1:] != (self.N_LINES, self.PERT_DIM):
            raise ValueError(
                f"e_p_stack must have shape [B, {self.N_LINES}, {self.PERT_DIM}], "
                f"got {tuple(e_p_stack.shape)}."
            )
        if c_target.shape[-1] != self.d_context:
            raise ValueError(f"c_target must have width {self.d_context}, got {c_target.shape[-1]}.")
        if not torch.isfinite(e_p_stack).all():
            raise ValueError("e_p_stack contains NaN or infinity.")

        B = e_p_stack.shape[0]
        c_expanded = c_target.unsqueeze(1).expand(-1, self.N_LINES, -1)  # [B,4,d_context]
        gate_input = torch.cat([e_p_stack, c_expanded], dim=-1)  # [B,4,64+d_context]
        gate_logits = self.gate_net(gate_input)  # [B,4,1]
        alpha = torch.softmax(gate_logits, dim=1)  # [B,4,1]

        if self.per_line_projection:
            e_p_proj = torch.stack(
                [self.proj[k](e_p_stack[:, k, :]) for k in range(self.N_LINES)], dim=1
            )  # [B,4,64]
        else:
            e_p_proj = self.proj(e_p_stack)  # [B,4,64], broadcast over line dim

        e_p_fused = (alpha * e_p_proj).sum(dim=1)  # [B,64]

        invariant = self.skip_proj(e_p_stack.mean(dim=1))  # [B,64]
        e_p_fused = self.norm(e_p_fused + invariant)
        return e_p_fused, alpha


# ---------------------------------------------------------------------------
# Module B
# ---------------------------------------------------------------------------

class StatisticalBaselineCompressor(nn.Module):
    """``delta_stat [B,G], e_p_fused [B,64] -> e_p_unified [B,64]``.

    Compresses the gene-space statistical delta to 64-d
    (``Linear -> LayerNorm -> GELU``), then combines with the GEARS fusion
    output via concat + projection back down to 64-d (``Linear(128,64) ->
    LayerNorm -> GELU``) - not a bare concat, so ``e_p_unified`` stays the
    same width the rest of the pipeline (``DiscreteLatentFlow``) expects.
    """

    FUSED_DIM = 64
    STAT_DIM = 64
    UNIFIED_DIM = 64

    def __init__(self, num_genes: int) -> None:
        super().__init__()
        self.num_genes = num_genes

        self.compress = nn.Sequential(
            nn.Linear(num_genes, self.STAT_DIM),
            nn.LayerNorm(self.STAT_DIM),
            nn.GELU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.FUSED_DIM + self.STAT_DIM, self.UNIFIED_DIM),
            nn.LayerNorm(self.UNIFIED_DIM),
            nn.GELU(),
        )

    def forward(self, delta_stat: Tensor, e_p_fused: Tensor) -> Tensor:
        if delta_stat.shape[-1] != self.num_genes:
            raise ValueError(f"delta_stat must have width {self.num_genes}, got {delta_stat.shape[-1]}.")
        if e_p_fused.shape[-1] != self.FUSED_DIM:
            raise ValueError(f"e_p_fused must have width {self.FUSED_DIM}, got {e_p_fused.shape[-1]}.")
        if not torch.isfinite(delta_stat).all():
            raise ValueError("delta_stat contains NaN or infinity.")

        e_p_stat = self.compress(delta_stat)  # [B,64]
        e_p_unified = self.combine(torch.cat([e_p_fused, e_p_stat], dim=-1))  # [B,64]
        return e_p_unified


# ---------------------------------------------------------------------------
# Module C
# ---------------------------------------------------------------------------

class _FlowStep(nn.Module):
    """One explicit-Euler velocity step: its own LayerNorm + MLP, not shared
    across steps (``MLP_step_t`` with distinct parameters per ``t``)."""

    def __init__(self, d_latent: int, cond_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_latent)
        self.linear_in = nn.Linear(d_latent, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, d_latent)

    def forward(self, z_t: Tensor, cond: Tensor) -> Tensor:
        h = self.linear_in(self.norm(z_t))  # d_latent -> hidden_dim (256)
        h = h + cond                        # addition at the 256-wide point - see module docstring
        h = self.dropout(self.act(h))
        return self.linear_out(h)           # hidden_dim -> d_latent


class DiscreteLatentFlow(nn.Module):
    """``z_0 [B,d_latent], e_p_unified [B,64], c_target [B,d_context] -> z_pert [B,d_latent]``.

    2-step explicit Euler residual flow (``z_{t+1} = z_t + v_t``), no ODE
    solver. Each step has its own conditioned MLP (:class:`_FlowStep`);
    the conditioning vector ``cond`` (derived once from
    ``[e_p_unified || c_target]``) is shared across both steps.
    """

    N_STEPS = 2
    COND_HIDDEN = 256

    def __init__(
        self, d_latent: int, unified_dim: int = 64, d_context: int = 32,
        hidden_dim: int = 256, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.unified_dim = unified_dim
        self.d_context = d_context

        self.cond_proj = nn.Linear(unified_dim + d_context, self.COND_HIDDEN)
        self.steps = nn.ModuleList(
            [_FlowStep(d_latent, self.COND_HIDDEN, hidden_dim, dropout) for _ in range(self.N_STEPS)]
        )

    def forward(self, z_0: Tensor, e_p_unified: Tensor, c_target: Tensor) -> Tensor:
        if z_0.shape[-1] != self.d_latent:
            raise ValueError(f"z_0 must have width {self.d_latent}, got {z_0.shape[-1]}.")
        if e_p_unified.shape[-1] != self.unified_dim:
            raise ValueError(f"e_p_unified must have width {self.unified_dim}, got {e_p_unified.shape[-1]}.")
        if c_target.shape[-1] != self.d_context:
            raise ValueError(f"c_target must have width {self.d_context}, got {c_target.shape[-1]}.")
        if not torch.isfinite(z_0).all():
            raise ValueError("z_0 contains NaN or infinity.")

        cond = self.cond_proj(torch.cat([e_p_unified, c_target], dim=-1))  # [B,256]
        z_t = z_0
        for step in self.steps:
            v_t = step(z_t, cond)
            z_t = z_t + v_t
        return z_t  # z_pert


# ---------------------------------------------------------------------------
# Module D
# ---------------------------------------------------------------------------

class ResidualDecoder(nn.Module):
    """``z_pert [B,d_latent] -> delta_neural [B,num_genes]``.

    Final layer is zero-initialized (weight AND bias): at step 0 of
    training, ``delta_neural == 0`` for every input, so ``y_pred`` starts
    exactly at ``x_ctrl + delta_stat`` - the baseline-identity invariant
    this whole architecture is built around.
    """

    def __init__(self, d_latent: int, num_genes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_genes),
        )
        final_layer = self.net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, z_pert: Tensor) -> Tensor:
        return self.net(z_pert)


# ---------------------------------------------------------------------------
# Module E
# ---------------------------------------------------------------------------

class VirtualCellModel(nn.Module):
    """Full assembly: A -> B -> C -> D -> final additive expression assembly.

    Parameters
    ----------
    num_genes:
        Width of ``delta_stat``, ``x_ctrl``, and the model's output.
    d_latent:
        Width of ``z_0`` / ``z_pert`` (the frozen scGPT <cls> latent's own
        width - 128 or 512 per the offline data spec).
    d_context:
        Width of the target cell-line context embedding ``c_target``.
    """

    def __init__(
        self,
        num_genes: int,
        d_latent: int = 128,
        d_context: int = 32,
        gate_hidden_dim: int = 64,
        per_line_projection: bool = True,
        flow_hidden_dim: int = 256,
        flow_dropout: float = 0.1,
        decoder_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.d_latent = d_latent
        self.d_context = d_context

        self.fusion = ContextGatedGEARSFusion(
            d_context=d_context, gate_hidden_dim=gate_hidden_dim,
            per_line_projection=per_line_projection,
        )
        self.stat_compressor = StatisticalBaselineCompressor(num_genes=num_genes)
        self.flow = DiscreteLatentFlow(
            d_latent=d_latent, unified_dim=StatisticalBaselineCompressor.UNIFIED_DIM,
            d_context=d_context, hidden_dim=flow_hidden_dim, dropout=flow_dropout,
        )
        self.decoder = ResidualDecoder(d_latent=d_latent, num_genes=num_genes, hidden_dim=decoder_hidden_dim)

    def forward(
        self,
        e_p_stack: Tensor,
        c_target: Tensor,
        delta_stat: Tensor,
        z_0: Tensor,
        x_ctrl: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        e_p_stack:
            ``[B, 4, 64]`` - stacked post-GNN GEARS perturbation vectors,
            fixed order ``[K562, HepG2, RPE, Jurkat]``.
        c_target:
            ``[B, d_context]`` - target cell-line context embedding.
        delta_stat:
            ``[B, num_genes]`` - empirical pseudobulk mean delta.
        z_0:
            ``[B, d_latent]`` - sampled NTC single-cell latent (real,
            empirically sampled - not a mean or parametric distribution).
        x_ctrl:
            ``[B, num_genes]`` - the paired control cell's own expression.

        Returns
        -------
        dict with ``y_pred [B,num_genes]``, ``z_pert [B,d_latent]``,
        ``delta_neural [B,num_genes]``, ``e_p_fused [B,64]``,
        ``e_p_unified [B,64]``, ``gate_alpha [B,4,1]``.
        """
        if x_ctrl.shape[-1] != self.num_genes:
            raise ValueError(f"x_ctrl must have width {self.num_genes}, got {x_ctrl.shape[-1]}.")
        if not torch.isfinite(x_ctrl).all():
            raise ValueError("x_ctrl contains NaN or infinity.")

        e_p_fused, gate_alpha = self.fusion(e_p_stack, c_target)
        e_p_unified = self.stat_compressor(delta_stat, e_p_fused)
        z_pert = self.flow(z_0, e_p_unified, c_target)
        delta_neural = self.decoder(z_pert)
        y_pred = x_ctrl + delta_stat + delta_neural

        return {
            "y_pred": y_pred,
            "z_pert": z_pert,
            "delta_neural": delta_neural,
            "e_p_fused": e_p_fused,
            "e_p_unified": e_p_unified,
            "gate_alpha": gate_alpha,
        }
