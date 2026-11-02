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
