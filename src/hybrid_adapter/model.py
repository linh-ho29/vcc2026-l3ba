"""Hybrid MLP perturbation adapter (VCC 2026).

Implements the four-module architecture from the development assignment:

    Module A — Perturbation representation   (GEARS 32-d + projected scGPT 512->32)
    Module B — Control-state encoder         (measured NTC features, 32 -> 32)
    Module C — Context-conditioned fusion    ([z_pert, z_ctrl, z_pert*z_ctrl])
    Module D — Gene-wise residual MLP        (shared across responding genes)

Core contract::

    corrected_effects = statistical_effects + residual

The residual output layer is zero-initialised so the untrained model exactly
reproduces the statistical baseline.

H7 subpopulation extension (opt-in via ``AdapterConfig.response_mode``)
-----------------------------------------------------------------------
Implements the "Subpopulation-Resolved, Identity-Gated Perturbation Response"
hypothesis as ablation modes M0-M4:

    M0: whole-line response (the docx-spec adapter; H3 baseline)
    M1: subpopulation unit, one global learned alpha
    M2: subpopulation unit, alpha(phase) shared across lines (claims 4+5)
    M3: alpha(phase, line); unseen contexts fall back to phase-shared alpha
    M4: M2 + composition weighting, alpha_eff(c,k) = alpha(k) * CompSim(c,k)

In subpopulation mode the residual is computed per (context, perturbation,
phase, gene) and the whole-line prediction is the composition-weighted sum
over phases::

    corrected(c, g, j) = sum_k p_c(k) * [ stat(c,g,k,j) + r(c,g,k,j) ]

with the residual decomposed as ``r = alpha_eff * r_invariant +
(1 - alpha_eff) * r_line_dependent``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor, nn

from .gated_fusion import GatedMultiContextFusion
from .schemas import (
    AdapterConfig,
    BatchValidationError,
    check_output_finite,
    validate_batch,
)

MODEL_VERSION = "1.2.0"


class PerturbationEncoder(nn.Module):
    """Module A — combine GEARS embedding with projected frozen scGPT embedding."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.scgpt_proj = nn.Linear(config.scgpt_dim, config.latent_dim)
        self.norm = nn.LayerNorm(config.latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * config.latent_dim, 2 * config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.latent_dim, config.latent_dim),
        )

    def forward(
        self,
        scgpt_target: Tensor,
        gears_embeddings: Optional[Tensor],
        masks: Mapping[str, bool],
    ) -> Tensor:
        """Return ``z_pert`` of shape ``[B, latent_dim]``.

        Missing sources are zero-filled according to the availability masks and
        the configured ``perturbation_mode``. ``gears_embeddings`` here may be
        either the single-source embedding from the batch, or - when
        ``config.use_gated_fusion`` is enabled - the already-fused
        ``e_p^(fused)`` computed by :class:`HybridAdapter` from multiple
        per-cell-type GEARS sources; this method treats both identically,
        since fusion only changes where the vector came from, not its shape.
        """
        mode = self.config.perturbation_mode
        use_scgpt = masks["scgpt_target"] and mode in ("both", "scgpt_only", "auto")
        use_gears = masks["gears_embeddings"] and mode in ("both", "gears_only", "auto")

        if mode == "scgpt_only":
            use_gears = False
        elif mode == "gears_only":
            use_scgpt = False

        if use_scgpt:
            scgpt_part = self.norm(self.scgpt_proj(scgpt_target))
        else:
            scgpt_part = torch.zeros(
                scgpt_target.shape[0], self.config.latent_dim,
                device=scgpt_target.device, dtype=scgpt_target.dtype,
            )
        if use_gears and gears_embeddings is not None:
            gears_part = gears_embeddings
        else:
            gears_part = torch.zeros(
                scgpt_target.shape[0], self.config.gears_dim,
                device=scgpt_target.device, dtype=scgpt_target.dtype,
            )
        return self.mlp(torch.cat([scgpt_part, gears_part], dim=-1))


class ControlStateEncoder(nn.Module):
    """Module B — encode precomputed control-state features (no foundation model)."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.control_dim, 2 * config.control_dim),
            nn.GELU(),
            nn.Linear(2 * config.control_dim, config.control_dim),
        )

    def forward(self, control_features: Tensor) -> Tensor:
        return self.mlp(control_features)


class ContextFusion(nn.Module):
    """Module C — fuse perturbation and control representations with interaction."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        d = config.latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.fusion_dim),
        )

    def forward(self, z_pert: Tensor, z_ctrl: Tensor) -> Tensor:
        return self.mlp(torch.cat([z_pert, z_ctrl, z_pert * z_ctrl], dim=-1))


class ResidualMLP(nn.Module):
    """Module D — shared gene-wise residual network.

    One shared network for every responding gene (no per-gene output heads).
    The final linear layer's weights and bias are zero-initialised so the
    untrained adapter reproduces the statistical baseline exactly.
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        input_dim = config.feature_spec.input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self._zero_init_final_layer()

    def _zero_init_final_layer(self) -> None:
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: Tensor) -> Tensor:
        """features: ``[..., input_dim]`` -> residual ``[..., 1]``."""
        return self.net(features)


class InvariantResidualMLP(ResidualMLP):
    """Shared (line-invariant) residual head used by the H7 modes."""


class LineDependentResidualMLP(ResidualMLP):
    """Line-dependent residual head used by the H7 modes."""


class HybridAdapter(nn.Module):
    """The hybrid MLP perturbation adapter.

    Default mode (``response_mode="whole_line"``) is the docx-spec adapter:
    ``corrected = stat + r_theta(context, perturbation, gene)``.

    Subpopulation mode (``response_mode="subpopulation"``) implements the H7
    ablation variants M1-M4; see the module docstring.
    """

    def __init__(self, config: Optional[AdapterConfig] = None) -> None:
        super().__init__()
        self.config = config or AdapterConfig()
        self.config.validate()

        self.perturbation_encoder = PerturbationEncoder(self.config)
        self.control_encoder = ControlStateEncoder(self.config)
        self.fusion = ContextFusion(self.config)
        self.response_gene_proj = nn.Linear(self.config.scgpt_dim, self.config.latent_dim)
        self.response_gene_norm = nn.LayerNorm(self.config.latent_dim)
        self.residual_mlp = ResidualMLP(self.config)

        # --- Gated multi-context fusion (system architecture Phase 2 step 2) ---
        # fused_dim is pinned to gears_dim (not a separate config value) so the
        # fused vector slots directly into PerturbationEncoder's existing
        # gears_embeddings input without changing any other module's shape.
        # The architecture diagram's own e_p^(fused) in R^64 describes the full
        # Phase 2 pipeline (distribution sampling + discrete latent flow +
        # decoder), which does not exist in this codebase yet; this is the
        # incremental integration into the existing Module A only.
        if self.config.use_gated_fusion:
            self.gated_fusion = GatedMultiContextFusion(
                source_dims=self.config.gears_source_dims,
                context_dim=self.config.context_dim,
                fused_dim=self.config.gears_dim,
            )

        # --- H7 subpopulation components (created only when enabled) ---
        if self.config.response_mode == "subpopulation":
            K = self.config.n_phases
            self.invariant_mlp = InvariantResidualMLP(self.config)
            self.line_dependent_mlp = LineDependentResidualMLP(self.config)
            self.phase_embedding = nn.Embedding(K, self.config.latent_dim)
            nn.init.normal_(self.phase_embedding.weight, std=0.05)
            if self.config.alpha_mode == "global":
                self.alpha_logit = nn.Parameter(torch.zeros(1))
            elif self.config.alpha_mode == "phase_shared":
                self.alpha_logit = nn.Parameter(torch.zeros(K))
            elif self.config.alpha_mode == "phase_per_line":
                # Registered per-context alpha vectors are stored in this buffer
                # index; unseen contexts fall back to the phase-shared vector.
                self.alpha_logit = nn.Parameter(torch.zeros(K))  # shared fallback
                self._per_line_alpha = nn.ParameterDict()
            if self.config.composition_weighting:
                # Reference composition used to compute CompSim when the batch
                # does not supply compsim_weights explicitly.
                self.register_buffer(
                    "reference_phase_proportions", torch.full((K,), 1.0 / K)
                )
        else:
            self._last_phase_corrected: Optional[Tensor] = None
            self._last_alpha_eff: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Alpha gating (H7)
    # ------------------------------------------------------------------
    def _alpha(
        self,
        context_ids: List[Any],
        device: torch.device,
    ) -> Tensor:
        """Return alpha logits of shape ``[B, K]`` for the current batch."""
        K = self.config.n_phases
        B = len(context_ids)
        if self.config.alpha_mode == "global":
            logits = self.alpha_logit.expand(B, K)
        elif self.config.alpha_mode == "phase_shared":
            logits = self.alpha_logit.unsqueeze(0).expand(B, K)
        elif self.config.alpha_mode == "phase_per_line":
            shared = self.alpha_logit.unsqueeze(0).expand(B, K)
            rows = []
            for i, cid in enumerate(context_ids):
                key = str(cid)
                if key in self._per_line_alpha:
                    rows.append(self._per_line_alpha[key].unsqueeze(0))
                else:
                    rows.append(shared[i].unsqueeze(0))
            logits = torch.cat(rows, dim=0).to(device)
        else:  # pragma: no cover - validated in config
            raise ValueError(f"Unknown alpha_mode: {self.config.alpha_mode}")
        return logits

    def register_line_alpha(self, context_id: Any, alpha_logits: Tensor) -> None:
        """Register a per-line alpha logit vector (M3 training-time use)."""
        if self.config.response_mode != "subpopulation":
            raise BatchValidationError(
                "register_line_alpha is only valid in subpopulation mode."
            )
        if self.config.alpha_mode != "phase_per_line":
            raise BatchValidationError(
                "register_line_alpha requires alpha_mode='phase_per_line'."
            )
        K = self.config.n_phases
        if tuple(alpha_logits.shape) != (K,):
            raise BatchValidationError(
                f"alpha_logits must have shape ({K},), got {tuple(alpha_logits.shape)}."
            )
        self._per_line_alpha[str(context_id)] = nn.Parameter(alpha_logits.detach().clone())

    def _compsim(self, validated: Dict[str, Any]) -> Tensor:
        """Composition similarity CompSim(c, k) in [0, 1], shape [B, K]."""
        supplied = validated["compsim_weights"]
        if supplied is not None:
            return supplied
        if not self.config.composition_weighting:
            return torch.ones(1)
        p = validated["phase_proportions"]
        ref = self.reference_phase_proportions.to(p.dtype)
        # Per-phase TV between {phase k vs rest} partitions reduces to
        # |p_c(k) - p_ref(k)|; CompSim(c,k) = 1 - that.
        return (1.0 - (p - ref.unsqueeze(0)).abs()).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        validated = validate_batch(batch, self.config)
        masks = validated["availability_masks"]
        stat = validated["stat_effects"]                       # [B, G]
        B, G = stat.shape
        device = stat.device

        # Modules A-C: context-conditioned representation h [B, fusion_dim]
        gears_input = validated["gears_embeddings"]
        gate_alpha = None
        if self.config.use_gated_fusion:
            fused, gate_alpha = self.gated_fusion(
                validated["gears_embeddings_multi"],
                validated["context_embedding"],
                validated["gears_source_mask"],
            )
            gears_input = fused
            masks = dict(masks)
            masks["gears_embeddings"] = True  # fusion already enforces >=1 real source

        z_pert = self.perturbation_encoder(
            validated["scgpt_target_embeddings"],
            gears_input,
            masks,
        )
        z_ctrl = self.control_encoder(validated["control_features"])
        h = self.fusion(z_pert, z_ctrl)                        # [B, 64]

        has_gears_effect = masks["gears_effects"]
        gears_effects = validated["gears_effects"]
        if gears_effects is None:
            gears_effects = torch.zeros_like(stat)
        stat_coverage = validated["stat_coverage"]
        if stat_coverage is None:
            stat_coverage = torch.ones_like(stat)

        source_indicators = torch.stack(
            [
                torch.full((B, G), float(has_gears_effect), device=device),
                torch.full(
                    (B, G), float(masks["gears_embeddings"]), device=device
                ),
                torch.full(
                    (B, G),
                    float(masks["scgpt_target"] and masks["scgpt_response"]),
                    device=device,
                ),
                torch.full((B, G), float(masks["control_features"]), device=device),
                stat_coverage,
            ],
            dim=-1,
        )                                                       # [B, G, 5]

        if self.config.response_mode == "whole_line":
            corrected, residual = self._forward_whole_line(
                validated, h, gears_effects, source_indicators
            )
        else:
            corrected, residual = self._forward_subpopulation(
                validated, h, gears_effects, source_indicators
            )

        output: Dict[str, Any] = {
            "corrected_effects": corrected,
            "baseline_effects": stat,
            "residual_effects": residual,
            "gene_ids": validated["response_gene_ids"],
            "perturbation_ids": validated["perturbation_ids"],
            "context_ids": validated["context_ids"],
        }
        if self.config.response_mode == "subpopulation":
            output["phase_corrected_effects"] = self._last_phase_corrected
            output["alpha_effective"] = self._last_alpha_eff
        if self.config.use_gated_fusion:
            output["gated_fusion_alpha"] = gate_alpha
        check_output_finite(output)
        return output

    # ------------------------------------------------------------------
    def _gene_features(
        self,
        validated: Dict[str, Any],
        h: Tensor,
        gears_effects: Tensor,
        source_indicators: Tensor,
        gene_slice: slice,
    ) -> Tensor:
        """Assemble Module D input features for a gene chunk: [B, Gc, input_dim]."""
        stat = validated["stat_effects"][:, gene_slice]          # [B, Gc]
        basal = validated["control_gene_expression"][:, gene_slice]
        gears_eff = gears_effects[:, gene_slice]
        indicators = source_indicators[:, gene_slice, :]

        # Project response-gene embeddings for this chunk (frozen inputs reused).
        resp_emb = validated["scgpt_response_embeddings"][:, gene_slice, :]
        B, Gc, D = resp_emb.shape
        proj = self.response_gene_norm(
            self.response_gene_proj(resp_emb.reshape(B * Gc, D))
        ).reshape(B, Gc, -1)

        h_exp = h.unsqueeze(1).expand(B, Gc, h.shape[-1])
        return torch.cat([h_exp, proj, basal.unsqueeze(-1), stat.unsqueeze(-1),
                          gears_eff.unsqueeze(-1), indicators], dim=-1)

    def _forward_whole_line(
        self,
        validated: Dict[str, Any],
        h: Tensor,
        gears_effects: Tensor,
        source_indicators: Tensor,
    ):
        stat = validated["stat_effects"]
        B, G = stat.shape
        chunk = self.config.gene_chunk_size
        residual_chunks = []
        for start in range(0, G, chunk):
            sl = slice(start, min(start + chunk, G))
            feats = self._gene_features(validated, h, gears_effects, source_indicators, sl)
            residual_chunks.append(self.residual_mlp(feats).squeeze(-1))
        residual = torch.cat(residual_chunks, dim=1)             # [B, G]
        return stat + residual, residual

    def _forward_subpopulation(
        self,
        validated: Dict[str, Any],
        h: Tensor,
        gears_effects: Tensor,
        source_indicators: Tensor,
    ):
        """H7 modes M1-M4.

        Residual per phase: r = alpha_eff * r_inv + (1 - alpha_eff) * r_dep.
        Whole-line prediction: composition-weighted sum over phases.
        """
        stat_phase = validated["phase_stat_effects"]             # [B, K, G]
        proportions = validated["phase_proportions"]             # [B, K]
        B, K, G = stat_phase.shape
        chunk = self.config.gene_chunk_size

        alpha_logits = self._alpha(validated["context_ids"], stat_phase.device)
        alpha = torch.sigmoid(alpha_logits)                      # [B, K]
        compsim = self._compsim(validated)                       # [B, K] or [1]
        if compsim.shape[0] == 1 and self.config.composition_weighting:
            compsim = compsim.expand(B, K)
        alpha_eff = alpha * compsim if self.config.composition_weighting else alpha

        phase_corrected_chunks = []
        residual_chunks = []
        for start in range(0, G, chunk):
            sl = slice(start, min(start + chunk, G))
            feats = self._gene_features(validated, h, gears_effects, source_indicators, sl)
            Bc, Gc = feats.shape[0], feats.shape[1]

            # Phase conditioning: the context block of the features is augmented
            # with the phase embedding, evaluated once per phase (input_dim fixed).
            phase_emb = self.phase_embedding.weight              # [K, latent]
            h_block = feats[..., : self.config.fusion_dim]
            rest = feats[..., self.config.fusion_dim:]
            pad = torch.zeros(
                Bc, Gc, self.config.fusion_dim - self.config.latent_dim,
                device=feats.device, dtype=feats.dtype,
            )

            r_phase = torch.zeros(Bc, K, Gc, device=feats.device, dtype=feats.dtype)
            for k in range(K):
                emb = phase_emb[k].view(1, 1, -1).expand(Bc, Gc, -1)
                feats_k = torch.cat([h_block + torch.cat([emb, pad], dim=-1), rest], dim=-1)
                r_inv = self.invariant_mlp(feats_k).squeeze(-1)      # [B, Gc]
                r_dep = self.line_dependent_mlp(feats_k).squeeze(-1)
                a = alpha_eff[:, k].unsqueeze(-1)                    # [B, 1]
                r_phase[:, k, :] = a * r_inv + (1.0 - a) * r_dep

            corrected_phase = stat_phase[..., sl] + r_phase          # [B, K, Gc]
            phase_corrected_chunks.append(corrected_phase)
            residual_chunks.append(r_phase)

        phase_corrected = torch.cat(phase_corrected_chunks, dim=-1)  # [B, K, G]
        residual = torch.cat(residual_chunks, dim=-1)                # [B, K, G]

        # Composition-weighted whole-line prediction.
        corrected = (proportions.unsqueeze(-1) * phase_corrected).sum(dim=1)  # [B, G]
        residual_line = (proportions.unsqueeze(-1) * residual).sum(dim=1)

        self._last_phase_corrected = phase_corrected
        self._last_alpha_eff = alpha_eff
        return corrected, residual_line

    # ------------------------------------------------------------------
    @torch.no_grad()
    def check_baseline_identity(self, batch: Mapping[str, Any], atol: float = 1e-6) -> bool:
        """True if the untrained/zero-residual model reproduces the baseline."""
        was_training = self.training
        self.eval()
        try:
            out = self.forward(batch)
            return torch.allclose(
                out["corrected_effects"], out["baseline_effects"], atol=atol
            )
        finally:
            self.train(was_training)
