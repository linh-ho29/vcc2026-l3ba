"""Compute-efficient Perturbation Virtual Cell model.

Pipeline: frozen scGPT + frozen per-cell-line GEARS checkpoints (K562,
HepG2, RPE, Jurkat) are offline feature extractors only - nothing in this
file trains them. Online training runs entirely on their low-dimensional
outputs (a control-cell latent ``z_0``, four GEARS perturbation vectors,
and a gene-space statistical delta), through a discrete latent velocity
flow and a zero-initialized residual decoder. Designed to fit under 2 GB
VRAM on a free T4: every online module operates on vectors of width <=
512, batch size 256-512, mixed precision (see ``train.py``).

Final prediction is anchored to the existing statistical baseline, not
learned from scratch:

::

    y_pred = x_ctrl + delta_stat + delta_neural

with ``delta_neural`` exactly zero at initialization when
``zero_init_residual=True`` (the default) - training starts at the
already-validated statistical baseline and only has to learn a
correction, never has to re-learn the baseline itself.

One dimensional ambiguity in the original spec, resolved explicitly
rather than silently assumed: ``DiscreteLatentFlow``'s update rule states
``v_t = MLP_step_t(LayerNorm(z_t) + cond)`` with ``cond`` ``flow_hidden_dim``-
wide and ``LayerNorm(z_t)`` ``d_latent``-wide (not generally equal) - these
can't be added directly unless ``d_latent == flow_hidden_dim``. Resolved
here by applying the addition *inside* ``MLP_step_t``, right after its
first ``Linear(d_latent, flow_hidden_dim)`` (the only point where both
operands are actually the same width), not before it. Every other knob
below (``src/config.py``) is a real constructor/forward argument, not an
unused schema field - see ``VirtualCellModel.from_config``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

try:
    from src.config import ModelConfig
except ImportError:  # running from within src/ or without the package root on sys.path
    ModelConfig = None  # type: ignore


def _make_activation(name: str) -> nn.Module:
    name = name.upper()
    if name == "GELU":
        return nn.GELU()
    if name == "SILU":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation {name!r}; use 'GELU' or 'SiLU'.")


# ---------------------------------------------------------------------------
# Module A
# ---------------------------------------------------------------------------

class ContextGatedGEARSFusion(nn.Module):
    """``e_p_stack [B,4,d_gears], c_target [B,d_context] -> e_p_fused [B,d_gears]``.

    Gated mixture over the 4 cell-line GEARS vectors (softmax weights from
    a 2-layer MLP conditioned on the target context), each line projected
    through its own ``Linear(d_gears,d_gears)`` (``per_line_projection=True``,
    the default - set ``False`` for one shared projection across all 4
    lines), plus an invariant skip connection (scaled by
    ``invariant_skip_weight``) from the raw (unprojected, un-gated)
    per-line mean, so the fused output never fully loses the "what do all
    4 lines agree on" signal even if the gate collapses onto one line.

    ``gating_temperature`` divides the gate logits before softmax (lower
    = sharper/more confident gate); ``gating_dropout`` drops the resulting
    attention weights (renormalized) during training.
    """

    N_LINES = 4

    def __init__(
        self,
        d_gears: int = 64,
        d_context: int = 32,
        gating_hidden_dim: int = 64,
        gating_temperature: float = 1.0,
        gating_dropout: float = 0.0,
        invariant_skip_weight: float = 0.2,
        per_line_projection: bool = True,
    ) -> None:
        super().__init__()
        if gating_temperature <= 0:
            raise ValueError(f"gating_temperature must be > 0, got {gating_temperature}.")
        self.d_gears = d_gears
        self.d_context = d_context
        self.gating_temperature = gating_temperature
        self.invariant_skip_weight = invariant_skip_weight
        self.per_line_projection = per_line_projection

        gate_in_dim = d_gears + d_context
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, gating_hidden_dim),
            nn.GELU(),
            nn.Linear(gating_hidden_dim, 1),
        )
        self.gate_dropout = nn.Dropout(gating_dropout)

        if per_line_projection:
            self.proj = nn.ModuleList(
                [nn.Linear(d_gears, d_gears) for _ in range(self.N_LINES)]
            )
        else:
            self.proj = nn.Linear(d_gears, d_gears)

        self.skip_proj = nn.Linear(d_gears, d_gears)
        self.norm = nn.LayerNorm(d_gears)

    def forward(self, e_p_stack: Tensor, c_target: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns ``(e_p_fused [B,d_gears], alpha [B,4,1])``."""
        if e_p_stack.dim() != 3 or e_p_stack.shape[1:] != (self.N_LINES, self.d_gears):
            raise ValueError(
                f"e_p_stack must have shape [B, {self.N_LINES}, {self.d_gears}], "
                f"got {tuple(e_p_stack.shape)}."
            )
        if c_target.shape[-1] != self.d_context:
            raise ValueError(f"c_target must have width {self.d_context}, got {c_target.shape[-1]}.")
        if not torch.isfinite(e_p_stack).all():
            raise ValueError("e_p_stack contains NaN or infinity.")

        c_expanded = c_target.unsqueeze(1).expand(-1, self.N_LINES, -1)  # [B,4,d_context]
        gate_input = torch.cat([e_p_stack, c_expanded], dim=-1)
        gate_logits = self.gate_net(gate_input) / self.gating_temperature  # [B,4,1]
        alpha = torch.softmax(gate_logits, dim=1)
        alpha = self.gate_dropout(alpha)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)  # renormalize post-dropout

        if self.per_line_projection:
            e_p_proj = torch.stack(
                [self.proj[k](e_p_stack[:, k, :]) for k in range(self.N_LINES)], dim=1
            )
        else:
            e_p_proj = self.proj(e_p_stack)

        e_p_fused = (alpha * e_p_proj).sum(dim=1)

        invariant = self.skip_proj(e_p_stack.mean(dim=1))
        e_p_fused = self.norm(e_p_fused + self.invariant_skip_weight * invariant)
        return e_p_fused, alpha


# ---------------------------------------------------------------------------
# Module B
# ---------------------------------------------------------------------------

class StatisticalBaselineCompressor(nn.Module):
    """``delta_stat [B,G], e_p_fused [B,d_gears] -> e_p_unified [B,d_gears]``.

    Compresses the gene-space statistical delta to ``d_stat``-d
    (``Linear -> LayerNorm -> GELU``), then combines with the GEARS fusion
    output via concat + projection back down to ``d_gears``-d
    (``Linear(d_gears+d_stat, d_gears) -> LayerNorm -> GELU``) - not a bare
    concat, so ``e_p_unified`` stays the same width the rest of the
    pipeline (``DiscreteLatentFlow``) expects.
    """

    def __init__(self, num_genes: int, d_gears: int = 64, d_stat: int = 64) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.d_gears = d_gears
        self.unified_dim = d_gears

        self.compress = nn.Sequential(
            nn.Linear(num_genes, d_stat),
            nn.LayerNorm(d_stat),
            nn.GELU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(d_gears + d_stat, d_gears),
            nn.LayerNorm(d_gears),
            nn.GELU(),
        )

    def forward(self, delta_stat: Tensor, e_p_fused: Tensor) -> Tensor:
        if delta_stat.shape[-1] != self.num_genes:
            raise ValueError(f"delta_stat must have width {self.num_genes}, got {delta_stat.shape[-1]}.")
        if e_p_fused.shape[-1] != self.d_gears:
            raise ValueError(f"e_p_fused must have width {self.d_gears}, got {e_p_fused.shape[-1]}.")
        if not torch.isfinite(delta_stat).all():
            raise ValueError("delta_stat contains NaN or infinity.")

        e_p_stat = self.compress(delta_stat)
        e_p_unified = self.combine(torch.cat([e_p_fused, e_p_stat], dim=-1))
        return e_p_unified


# ---------------------------------------------------------------------------
# Module C
# ---------------------------------------------------------------------------

class _FlowStep(nn.Module):
    """One explicit-Euler velocity step: its own LayerNorm + MLP unless
    ``weight_sharing=True`` (same module instance reused for every step,
    set up by :class:`DiscreteLatentFlow`)."""

    def __init__(
        self, d_latent: int, cond_dim: int, hidden_dim: int = 256,
        dropout: float = 0.1, activation: str = "GELU",
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_latent)
        self.linear_in = nn.Linear(d_latent, hidden_dim)
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, d_latent)

    def forward(self, z_t: Tensor, cond: Tensor) -> Tensor:
        h = self.linear_in(self.norm(z_t))  # d_latent -> hidden_dim
        h = h + cond                        # addition at the matched-width point - see module docstring
        h = self.dropout(self.act(h))
        return self.linear_out(h)           # hidden_dim -> d_latent


class DiscreteLatentFlow(nn.Module):
    """``z_0 [B,d_latent], e_p_unified [B,d_gears], c_target [B,d_context] -> z_pert [B,d_latent]``.

    ``n_steps``-step explicit Euler residual flow
    (``z_{t+1} = z_t + step_size * v_t``), no ODE solver. ``weight_sharing``
    controls whether every step reuses the same conditioned MLP
    (:class:`_FlowStep`) or each gets its own distinct parameters (the
    default, ``False``). The conditioning vector ``cond`` (derived once
    from ``[e_p_unified || c_target]``) is shared across all steps either way.
    """

    COND_HIDDEN_DEFAULT = 256

    def __init__(
        self, d_latent: int, unified_dim: int = 64, d_context: int = 32,
        hidden_dim: int = 256, dropout: float = 0.1, n_steps: int = 2,
        step_size: float = 1.0, weight_sharing: bool = False, activation: str = "GELU",
    ) -> None:
        super().__init__()
        if n_steps not in (2, 3):
            raise ValueError(f"n_steps must be 2 or 3 per the architecture, got {n_steps}.")
        self.d_latent = d_latent
        self.unified_dim = unified_dim
        self.d_context = d_context
        self.n_steps = n_steps
        self.step_size = step_size
        self.weight_sharing = weight_sharing

        self.cond_proj = nn.Linear(unified_dim + d_context, hidden_dim)
        if weight_sharing:
            shared = _FlowStep(d_latent, hidden_dim, hidden_dim, dropout, activation)
            self.steps = nn.ModuleList([shared for _ in range(n_steps)])
        else:
            self.steps = nn.ModuleList(
                [_FlowStep(d_latent, hidden_dim, hidden_dim, dropout, activation) for _ in range(n_steps)]
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

        cond = self.cond_proj(torch.cat([e_p_unified, c_target], dim=-1))
        z_t = z_0
        for step in self.steps:
            v_t = step(z_t, cond)
            z_t = z_t + self.step_size * v_t
        return z_t  # z_pert


# ---------------------------------------------------------------------------
# Module D
# ---------------------------------------------------------------------------

class ResidualDecoder(nn.Module):
    """``z_pert [B,d_latent] -> delta_neural [B,num_genes]``.

    ``depth=1`` is a single linear layer; ``depth=2`` is a 2-layer MLP
    (``hidden_dim`` wide). When ``zero_init_residual=True`` (the default),
    the final layer's weight AND bias are zero-initialized: at step 0 of
    training, ``delta_neural == 0`` for every input, so ``y_pred`` starts
    exactly at ``x_ctrl + delta_stat`` - the baseline-identity invariant
    this whole architecture is built around. ``clamp`` (if given) bounds
    ``|delta_neural|`` so a still-undertrained correction can't blow up
    the final assembly.
    """

    def __init__(
        self, d_latent: int, num_genes: int, depth: int = 2, hidden_dim: int = 256,
        zero_init_residual: bool = True, activation: str = "GELU",
        clamp: Optional[float] = None,
    ) -> None:
        super().__init__()
        if depth not in (1, 2):
            raise ValueError(f"depth must be 1 or 2, got {depth}.")
        self.clamp = clamp
        if depth == 1:
            self.net = nn.Linear(d_latent, num_genes)
            final_layer = self.net
        else:
            self.net = nn.Sequential(
                nn.Linear(d_latent, hidden_dim),
                _make_activation(activation),
                nn.Linear(hidden_dim, num_genes),
            )
            final_layer = self.net[-1]
        if zero_init_residual:
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(self, z_pert: Tensor) -> Tensor:
        out = self.net(z_pert)
        if self.clamp is not None:
            out = out.clamp(-self.clamp, self.clamp)
        return out


# ---------------------------------------------------------------------------
# Module E
# ---------------------------------------------------------------------------

class VirtualCellModel(nn.Module):
    """Full assembly: A -> B -> C -> D -> final additive expression assembly.

    Every argument here corresponds 1:1 to a field in ``src.config.ModelConfig``
    - use :meth:`from_config` to construct from that dataclass directly.
    """

    def __init__(
        self,
        num_genes: int,
        d_latent: int = 128,
        d_gears: int = 64,
        d_context: int = 32,
        d_stat: int = 64,
        gating_hidden_dim: int = 64,
        gating_temperature: float = 1.0,
        gating_dropout: float = 0.0,
        invariant_skip_weight: float = 0.2,
        per_line_projection: bool = True,
        flow_steps: int = 2,
        flow_hidden_dim: int = 256,
        flow_dropout: float = 0.1,
        step_size: float = 1.0,
        weight_sharing: bool = False,
        activation: str = "GELU",
        decoder_depth: int = 2,
        decoder_hidden_dim: int = 256,
        zero_init_residual: bool = True,
        residual_clamping: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.d_latent = d_latent
        self.d_context = d_context

        self.fusion = ContextGatedGEARSFusion(
            d_gears=d_gears, d_context=d_context, gating_hidden_dim=gating_hidden_dim,
            gating_temperature=gating_temperature, gating_dropout=gating_dropout,
            invariant_skip_weight=invariant_skip_weight, per_line_projection=per_line_projection,
        )
        self.stat_compressor = StatisticalBaselineCompressor(
            num_genes=num_genes, d_gears=d_gears, d_stat=d_stat,
        )
        self.flow = DiscreteLatentFlow(
            d_latent=d_latent, unified_dim=d_gears, d_context=d_context,
            hidden_dim=flow_hidden_dim, dropout=flow_dropout, n_steps=flow_steps,
            step_size=step_size, weight_sharing=weight_sharing, activation=activation,
        )
        self.decoder = ResidualDecoder(
            d_latent=d_latent, num_genes=num_genes, depth=decoder_depth,
            hidden_dim=decoder_hidden_dim, zero_init_residual=zero_init_residual,
            activation=activation, clamp=residual_clamping,
        )

    @classmethod
    def from_config(cls, cfg: "ModelConfig") -> "VirtualCellModel":
        return cls(
            num_genes=cfg.num_genes, d_latent=cfg.d_latent, d_gears=cfg.d_gears,
            d_context=cfg.d_context, d_stat=cfg.d_stat,
            gating_hidden_dim=cfg.gating_hidden_dim, gating_temperature=cfg.gating_temperature,
            gating_dropout=cfg.gating_dropout, invariant_skip_weight=cfg.invariant_skip_weight,
            per_line_projection=cfg.per_line_projection, flow_steps=cfg.flow_steps,
            flow_hidden_dim=cfg.flow_hidden_dim, flow_dropout=cfg.flow_dropout,
            step_size=cfg.step_size, weight_sharing=cfg.weight_sharing, activation=cfg.activation,
            decoder_depth=cfg.decoder_depth, decoder_hidden_dim=cfg.decoder_hidden_dim,
            zero_init_residual=cfg.zero_init_residual, residual_clamping=cfg.residual_clamping,
        )

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
            ``[B, 4, d_gears]`` - stacked post-GNN GEARS perturbation
            vectors, fixed order ``[K562, HepG2, RPE, Jurkat]``.
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
        ``delta_neural [B,num_genes]``, ``e_p_fused [B,d_gears]``,
        ``e_p_unified [B,d_gears]``, ``gate_alpha [B,4,1]``.
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
