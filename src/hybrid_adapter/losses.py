"""Training objectives for the hybrid adapter.

Three configurable objectives (weights default to docx-spec values):

1. Global reconstruction (always on): Huber loss between corrected and true
   perturbation effects.
2. DE-weighted reconstruction (optional): Huber loss restricted to genes in an
   externally provided differential-expression eligibility mask. Returns 0 when
   the mask is empty (no division by zero).
3. Perturbation discrimination (optional): a contrastive cosine-similarity
   objective encouraging each predicted perturbation effect to match its own
   measured response more strongly than other perturbations' responses.

A fourth objective, :func:`sliced_wasserstein_distance`, supports the
gated-fusion + latent-flow pipeline (``model.GatedFusionFlowAdapter``, system
architecture Phase 2 step 5): it compares a BATCH of predicted single-cell
latents against a batch of real single-cell latents for the same
perturbation, preserving cell-to-cell heterogeneity that an MSE-on-mean-
profile loss would wash out. It is a distribution-matching loss between two
SETS of samples, not a per-row loss like the three above.

All losses respect explicit eligibility masks. No objective ever sees measured
held-out perturbation responses during fitting or hyperparameter selection.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def global_reconstruction_loss(
    predicted: Tensor,
    target: Tensor,
    eligibility_mask: Optional[Tensor] = None,
    delta: float = 1.0,
) -> Tensor:
    """Huber loss over all (context, perturbation, gene) entries.

    ``eligibility_mask`` (1.0 = eligible, 0.0 = excluded) restricts the loss;
    if omitted, all entries are eligible.
    """
    if eligibility_mask is None:
        return F.huber_loss(predicted, target, delta=delta)
    mask = eligibility_mask.float()
    denom = mask.sum()
    if denom.item() == 0:
        return predicted.sum() * 0.0  # zero, connected to the graph
    per_elem = F.huber_loss(predicted, target, delta=delta, reduction="none")
    return (per_elem * mask).sum() / denom


def de_weighted_reconstruction_loss(
    predicted: Tensor,
    target: Tensor,
    de_mask: Optional[Tensor],
    delta: float = 1.0,
) -> Tensor:
    """Huber loss restricted to DE-eligible genes.

    ``de_mask`` is a 0/1 tensor of the same shape as ``predicted``. An empty
    (all-zero or None) mask yields 0.0 while keeping the graph connected.
    """
    if de_mask is None:
        return predicted.sum() * 0.0
    mask = de_mask.float()
    if mask.sum().item() == 0:
        return predicted.sum() * 0.0
    per_elem = F.huber_loss(predicted, target, delta=delta, reduction="none")
    return (per_elem * mask).sum() / mask.sum()


def perturbation_discrimination_loss(
    predicted: Tensor,
    target: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Contrastive cosine-similarity discrimination objective.

    Each row of ``predicted`` (one context-perturbation pair's gene-wise effect
    vector) should be more similar to its own measured response row in
    ``target`` than to any other row. Computes a cosine-similarity matrix
    between predicted and target rows and applies cross-entropy toward the
    diagonal.
    """
    if predicted.shape[0] < 2:
        return predicted.sum() * 0.0
    pred_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    logits = pred_norm @ target_norm.T / temperature
    labels = torch.arange(predicted.shape[0], device=predicted.device)
    return F.cross_entropy(logits, labels)


def sliced_wasserstein_distance(
    predicted_samples: Tensor,
    true_samples: Tensor,
    n_projections: int = 50,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sliced Wasserstein distance between two sets of samples.

    ``predicted_samples``, ``true_samples``: ``[N_pred, D]`` and ``[N_true,
    D]`` - two (possibly differently-sized) batches of points in the same
    D-dimensional space, e.g. predicted vs. real single-cell latents for one
    perturbation. Projects both onto ``n_projections`` random unit
    directions; in 1-D the optimal-transport cost between two empirical
    distributions is just the L2 distance between their SORTED values
    (closed form, no inner optimization), averaged over projections and
    interpolated to a common size when ``N_pred != N_true``.
    """
    if predicted_samples.dim() != 2 or true_samples.dim() != 2:
        raise ValueError(
            "predicted_samples and true_samples must be 2-D [N, D]; got "
            f"{tuple(predicted_samples.shape)} and {tuple(true_samples.shape)}."
        )
    if predicted_samples.shape[-1] != true_samples.shape[-1]:
        raise ValueError(
            f"Dimension mismatch: predicted_samples has D={predicted_samples.shape[-1]}, "
            f"true_samples has D={true_samples.shape[-1]}."
        )
    D = predicted_samples.shape[-1]
    directions = torch.randn(n_projections, D, generator=generator, device=predicted_samples.device)
    directions = F.normalize(directions, dim=-1)

    pred_proj = predicted_samples @ directions.T   # [N_pred, n_projections]
    true_proj = true_samples @ directions.T        # [N_true, n_projections]
    pred_sorted, _ = torch.sort(pred_proj, dim=0)
    true_sorted, _ = torch.sort(true_proj, dim=0)

    n_pred, n_true = pred_sorted.shape[0], true_sorted.shape[0]
    if n_pred != n_true:
        # Interpolate the smaller set's sorted values onto the larger set's
        # quantile grid so both empirical CDFs are compared at matched quantiles.
        n_common = max(n_pred, n_true)
        def _resample(x, n_from, n_to):
            if n_from == n_to:
                return x
            idx = torch.linspace(0, n_from - 1, n_to, device=x.device)
            lo = idx.floor().long().clamp(max=n_from - 1)
            hi = idx.ceil().long().clamp(max=n_from - 1)
            frac = (idx - lo.float()).unsqueeze(-1)
            return x[lo] * (1 - frac) + x[hi] * frac
        pred_sorted = _resample(pred_sorted, n_pred, n_common)
        true_sorted = _resample(true_sorted, n_true, n_common)

    return ((pred_sorted - true_sorted) ** 2).mean()


def compute_flow_loss(
    y_hat_pert: Tensor,
    y_true_pert: Tensor,
    z_pert: Optional[Tensor] = None,
    z_true: Optional[Tensor] = None,
    config: Optional[Mapping[str, float]] = None,
) -> dict:
    """Loss for the gated-fusion + latent-flow pipeline: MSE on predicted vs.
    true expression, plus (optionally) Sliced Wasserstein distance between
    predicted and real single-cell latents for the same perturbation.

    Parameters
    ----------
    y_hat_pert, y_true_pert:
        ``[B, G]`` predicted and real (mean or per-cell) expression.
    z_pert, z_true:
        Optional ``[N, latent_dim]`` predicted/real single-cell latents for
        ONE perturbation (not batched across different perturbations - SWD
        compares two point clouds in the same space). Omit to train with
        reconstruction loss only.
    config:
        ``lambda_recon`` (default 1.0), ``lambda_swd`` (default 0.0, so SWD
        is off unless explicitly requested), ``swd_projections`` (default 50).
    """
    cfg = dict(config or {})
    lambda_recon = float(cfg.get("lambda_recon", 1.0))
    lambda_swd = float(cfg.get("lambda_swd", 0.0))
    n_projections = int(cfg.get("swd_projections", 50))

    loss_recon = F.mse_loss(y_hat_pert, y_true_pert)
    total = lambda_recon * loss_recon

    loss_swd = torch.zeros((), device=y_hat_pert.device)
    if lambda_swd > 0.0:
        if z_pert is None or z_true is None:
            raise ValueError("lambda_swd > 0 requires both z_pert and z_true.")
        loss_swd = sliced_wasserstein_distance(z_pert, z_true, n_projections=n_projections)
        total = total + lambda_swd * loss_swd

    return {"loss": total, "loss_recon": loss_recon.detach(), "loss_swd": loss_swd.detach()}


def compute_loss(
    output: Mapping[str, Tensor],
    target_effects: Tensor,
    config: Optional[Mapping[str, float]] = None,
    eligibility_mask: Optional[Tensor] = None,
    de_mask: Optional[Tensor] = None,
) -> dict:
    """Weighted total loss.

    Parameters
    ----------
    output:
        Model output dict (must contain ``corrected_effects``).
    target_effects:
        Measured perturbation effects matching ``corrected_effects`` in shape
        and units.
    config:
        Loss weights: ``lambda_all`` (default 1.0), ``lambda_de`` (default 0.0),
        ``lambda_disc`` (default 0.0), optional ``disc_temperature``.
    eligibility_mask:
        Optional 0/1 mask applied to the global reconstruction term.
    de_mask:
        Optional 0/1 DE-eligibility mask for the DE-weighted term.
    """
    cfg = dict(config or {})
    lambda_all = float(cfg.get("lambda_all", 1.0))
    lambda_de = float(cfg.get("lambda_de", 0.0))
    lambda_disc = float(cfg.get("lambda_disc", 0.0))
    temperature = float(cfg.get("disc_temperature", 0.1))

    predicted = output["corrected_effects"]
    loss_all = global_reconstruction_loss(predicted, target_effects, eligibility_mask)
    total = lambda_all * loss_all

    loss_de = torch.zeros((), device=predicted.device)
    if lambda_de > 0.0:
        loss_de = de_weighted_reconstruction_loss(predicted, target_effects, de_mask)
        total = total + lambda_de * loss_de

    loss_disc = torch.zeros((), device=predicted.device)
    if lambda_disc > 0.0:
        loss_disc = perturbation_discrimination_loss(
            predicted, target_effects, temperature=temperature
        )
        total = total + lambda_disc * loss_disc

    return {
        "loss": total,
        "loss_all": loss_all.detach(),
        "loss_de": loss_de.detach(),
        "loss_disc": loss_disc.detach(),
    }
