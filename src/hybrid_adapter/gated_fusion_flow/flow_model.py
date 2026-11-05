"""Full Phase 2 pipeline: gated multi-context fusion -> input conditioning ->
discrete latent flow -> decoder (system architecture, posted on ``main``).

This is a SEPARATE model from :class:`hybrid_adapter.model.HybridAdapter`,
not a replacement - ``HybridAdapter``'s ``corrected = stat + residual``
pathway is untouched. This module implements the generative alternative
pathway: sample a real control-cell latent ``z_0``, condition it on the
fused perturbation representation and target context, advance it through a
short residual flow, and decode to predicted expression - with no
statistical-baseline dependency at all.

SYNTHETIC PHASE ONLY, same as the rest of this package: training this model
on real biological data is out of scope here. See
``examples/real_data_pipeline_smoketest.py`` for a forward-pass-only (no
training, no gradients) check against real per-cell-type GEARS embeddings
and a real control-cell NTC latent bank.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

from .control_branch import ControlCellEncoder, NTCLatentBank, ScGPTGeneEmbeddingPooling
from .gated_fusion import GatedMultiContextFusion
from .latent_flow import DiscreteLatentFlow, InputConditioningLayer, LatentDecoder
from ..schemas import BatchValidationError


class GatedFusionFlowAdapter(nn.Module):
    """``z_0, {e_p^(k)}, c -> gated fusion -> conditioning -> flow -> decode``.

    Parameters
    ----------
    gears_source_dims:
        Per-source GEARS embedding widths (e.g. ``(64, 32)`` for k562, hepg2).
    context_dim:
        Width of the context embedding ``c``.
    fused_dim:
        Output width of the gated fusion (``e_p^(fused)``).
    z0_dim:
        Width of the control-cell latent ``z_0`` (and of the final ``z_pert``).
    flow_dim:
        Internal width the discrete latent flow operates in.
    n_genes:
        Size of the predicted expression vector (decoder output width).
    n_flow_steps:
        2 or 3, per the architecture.
    decoder_hidden:
        If given, the decoder is a 2-layer MLP with this hidden width
        instead of a single linear layer ("light decoder head" either way).
    """

    def __init__(
        self,
        gears_source_dims: Sequence[int],
        context_dim: int = 32,
        fused_dim: int = 64,
        z0_dim: int = 128,
        flow_dim: int = 256,
        n_genes: int = 6642,
        n_flow_steps: int = 3,
        decoder_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gears_source_dims = list(gears_source_dims)
        self.context_dim = context_dim
        self.z0_dim = z0_dim

        self.fusion = GatedMultiContextFusion(
            source_dims=gears_source_dims, context_dim=context_dim, fused_dim=fused_dim
        )
        self.conditioning = InputConditioningLayer(
            z0_dim=z0_dim, fused_dim=fused_dim, context_dim=context_dim, flow_dim=flow_dim
        )
        self.flow = DiscreteLatentFlow(
            flow_dim=flow_dim, output_dim=z0_dim, fused_dim=fused_dim,
            context_dim=context_dim, n_steps=n_flow_steps,
        )
        self.decoder = LatentDecoder(latent_dim=z0_dim, n_genes=n_genes, hidden_dim=decoder_hidden)

    def forward(
        self,
        z0: Tensor,
        source_embeddings: List[Tensor],
        context_embedding: Tensor,
        source_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        z0:
            ``[B, z0_dim]`` - real, sampled control-cell latents (e.g. from
            :class:`hybrid_adapter.control_branch.NTCLatentBank`), NOT a
            mean or parametric sample.
        source_embeddings:
            List of K tensors, one per cell-type GEARS source, entry k of
            shape ``[B, gears_source_dims[k]]``.
        context_embedding:
            ``[B, context_dim]``.
        source_mask:
            Optional ``[B, K]`` - which sources are real for each row.
        """
        if z0.shape[-1] != self.z0_dim:
            raise BatchValidationError(f"z0 must have width {self.z0_dim}, got {z0.shape[-1]}.")
        if not torch.isfinite(z0).all():
            raise BatchValidationError("z0 contains NaN or infinity.")

        e_p_fused, gate_alpha = self.fusion(source_embeddings, context_embedding, source_mask)
        h0 = self.conditioning(z0, e_p_fused, context_embedding)
        z_pert = self.flow(h0, e_p_fused, context_embedding)
        y_hat_pert = self.decoder(z_pert)

        output = {
            "predicted_expression": y_hat_pert,
            "z_pert": z_pert,
            "e_p_fused": e_p_fused,
            "gate_alpha": gate_alpha,
        }
        for key, value in output.items():
            if not torch.isfinite(value).all():
                raise BatchValidationError(f"Model output {key!r} contains NaN or infinity.")
        return output
