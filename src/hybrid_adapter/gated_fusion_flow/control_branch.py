"""Control-cell branch (system architecture, Phase 1 - control-cell side).

Implements::

    [Control Single Cells] -> Frozen Lightweight VAE/PCA (trained on all NTCs)
                            -> Saved NTC Latents z_0 ~ P_ctrl in R^128

Two frozen stages, matching the user's explicit direction to "freeze the
fine-tuned scGPT and use it strictly as an offline feature extractor for
the control branch":

1. :class:`ScGPTGeneEmbeddingPooling` - a non-trainable, expression-weighted
   pooling of frozen per-gene scGPT embeddings into one per-cell vector.
   This is ONE reasonable pooling choice, not the only one; it is also not
   assumed to carry strong unique semantic signal - this project's own
   evaluation (``s04``/``s05c`` in the scGPT handoff, see
   ``data/scgpt/ft_meta.json`` and the manifest it came with) found that
   under a diagonal-ridge probe, scGPT embeddings (frozen OR fine-tuned)
   match random Gaussian features within ~0.01 Pearson - most of the probe's
   apparent value is generic per-gene capacity, not semantic gene-gene
   structure, and the fine-tune did not measurably improve on frozen
   embeddings (deltas of -0.0001 to -0.0004 across folds). Using the
   fine-tuned embeddings here is the user's explicit choice (for pipeline
   consistency / narrative), not a claim that it outperforms frozen.

2. :class:`ControlCellEncoder` - a PCA (not a VAE; PCA is the simpler of the
   two alternatives the architecture names, and sampling real encoded cells
   makes a VAE's generative prior unnecessary - see point 3 below) fit ONCE
   on real control-cell pooled features, then frozen.

:class:`NTCLatentBank` stores the resulting per-cell latents and samples
from them empirically - "rather than reducing controls to mean and
variance, sample actual single-cell baseline vectors z_0 ~ P_ctrl" (the
user's own phrasing). Fitting a PCA on real control-cell data is a closed-
form preprocessing step (no gradient descent, no adapter parameters
learned), not "training on real biological data" in the sense the
package's synthetic-phase rule guards against - that rule is about the
adapter's own learned weights, not about fitting a frozen upstream feature
extractor, which is exactly what scGPT's own embeddings already are.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn

from ..schemas import BatchValidationError


class ScGPTGeneEmbeddingPooling(nn.Module):
    """Frozen, expression-weighted pooling of per-gene scGPT embeddings.

    Parameters
    ----------
    gene_embeddings:
        ``[n_genes, emb_dim]`` frozen per-gene embeddings (e.g.
        ``data/scgpt/gene_emb_finetuned_pie.npy``), on a fixed gene axis.
    """

    def __init__(self, gene_embeddings: Tensor) -> None:
        super().__init__()
        if gene_embeddings.dim() != 2:
            raise ValueError(f"gene_embeddings must be 2-D, got shape {tuple(gene_embeddings.shape)}")
        self.register_buffer("gene_embeddings", gene_embeddings.float())
        self.n_genes, self.emb_dim = gene_embeddings.shape
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, expression: Tensor) -> Tensor:
        """``expression``: ``[B, n_genes]`` non-negative expression values
        (raw counts or any non-negative normalization) on the SAME gene
        axis as ``gene_embeddings``. Returns pooled ``[B, emb_dim]``."""
        if expression.shape[-1] != self.n_genes:
            raise BatchValidationError(
                f"expression must have {self.n_genes} genes (scGPT embedding axis), "
                f"got {expression.shape[-1]}."
            )
        if not torch.isfinite(expression).all():
            raise BatchValidationError("expression contains NaN or infinity.")
        if (expression < 0).any():
            raise BatchValidationError(
                "expression must be non-negative (pooling weights are normalized "
                "expression values); got negative entries - check it is not a "
                "z-score-normalized representation."
            )
        total = expression.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = expression / total
        return weights @ self.gene_embeddings


class ControlCellEncoder(nn.Module):
    """Frozen PCA encoder, fit once on real control-cell pooled features.

    Not trainable via backprop: :meth:`fit` is a closed-form PCA computed
    from real data directly (not gradient descent), after which the
    projection is frozen (registered buffers, ``requires_grad=False``).
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._fitted = False

    def fit(self, pooled_features: np.ndarray) -> "ControlCellEncoder":
        """``pooled_features``: ``[N, emb_dim]`` real control-cell pooled
        vectors (e.g. from :class:`ScGPTGeneEmbeddingPooling`). Fits PCA via
        SVD and freezes the resulting projection."""
        if pooled_features.ndim != 2:
            raise ValueError(f"pooled_features must be 2-D, got shape {pooled_features.shape}")
        n, d = pooled_features.shape
        if n <= self.latent_dim:
            raise ValueError(
                f"Need more control cells ({n}) than latent_dim ({self.latent_dim}) to fit PCA."
            )
        mean = pooled_features.mean(axis=0)
        centered = pooled_features - mean
        # Economy SVD: centered = U S Vt: components are rows of Vt.
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[: self.latent_dim]              # [latent_dim, emb_dim]
        explained_var = (S[: self.latent_dim] ** 2) / max(n - 1, 1)
        total_var = (S ** 2).sum() / max(n - 1, 1)

        self.register_buffer("mean_", torch.from_numpy(mean).float())
        self.register_buffer("components_", torch.from_numpy(components).float())
        self.explained_variance_ratio_ = float(explained_var.sum() / total_var) if total_var > 0 else 0.0
        self._fitted = True
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def forward(self, pooled_features: Tensor) -> Tensor:
        if not self._fitted:
            raise RuntimeError("ControlCellEncoder.fit() must be called before use.")
        if pooled_features.shape[-1] != self.components_.shape[-1]:
            raise BatchValidationError(
                f"pooled_features must have {self.components_.shape[-1]} features, "
                f"got {pooled_features.shape[-1]}."
            )
        centered = pooled_features - self.mean_
        return centered @ self.components_.T


class NTCLatentBank:
    """Saved per-cell NTC latents; samples real encoded cells empirically.

    Deliberately NOT a parametric distribution (no mean/covariance fit) -
    ``sample`` draws (with replacement) from the actual stored per-cell
    latent vectors, matching "sample actual single-cell baseline vectors
    z_0 ~ P_ctrl" rather than reducing controls to summary statistics.
    """

    def __init__(self) -> None:
        self._latents: Dict[str, Tensor] = {}

    def add(self, context: str, latents: Tensor) -> None:
        """``latents``: ``[N, latent_dim]`` real per-cell z_0 for this context."""
        if latents.dim() != 2:
            raise ValueError(f"latents must be 2-D, got shape {tuple(latents.shape)}")
        self._latents[context] = latents.detach().clone()

    def __len__(self) -> int:
        return sum(t.shape[0] for t in self._latents.values())

    @property
    def contexts(self):
        return list(self._latents.keys())

    def sample(self, n: int, context: str, generator: Optional[torch.Generator] = None) -> Tensor:
        """Sample ``n`` real control-cell latents (with replacement) for ``context``."""
        if context not in self._latents:
            raise BatchValidationError(
                f"No NTC latents stored for context {context!r}; available: {self.contexts}. "
                "This must be handled upstream, not silently substituted with another context."
            )
        pool = self._latents[context]
        idx = torch.randint(0, pool.shape[0], (n,), generator=generator)
        return pool[idx]
