"""Sliced Wasserstein Distance (SWD) between two point clouds.

Vectorized, O(B log B) per projection via sorting - not the O(B^2) quadratic
cost of Sinkhorn optimal transport, to keep this cheap enough for a free T4
GPU (VRAM target < 2 GB, see models/virtual_cell.py's module docstring).

Method: for L random unit-vector projections theta ~ N(0, I) in the latent
space, project both point clouds onto each theta, sort the 1-D projections,
and take the L1 distance between the sorted sequences (closed-form 1-D
optimal transport). The final distance is the mean over projections.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def sliced_wasserstein_distance(
    x: Tensor,
    y: Tensor,
    n_projections: int = 32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Parameters
    ----------
    x, y:
        ``[N_x, D]`` and ``[N_y, D]`` point clouds in the same D-dim space.
        ``N_x`` and ``N_y`` need not match.
    n_projections:
        Number of random projection directions L (default 32, per spec).
    generator:
        Optional ``torch.Generator`` for reproducible projections.

    Returns
    -------
    Scalar tensor: the mean L1 distance between sorted 1-D projections,
    averaged over ``n_projections`` random directions.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"x and y must be 2-D [N, D], got shapes {tuple(x.shape)} and {tuple(y.shape)}.")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(f"Dimension mismatch: x has D={x.shape[-1]}, y has D={y.shape[-1]}.")

    d = x.shape[-1]
    theta = torch.randn(d, n_projections, generator=generator, device=x.device, dtype=x.dtype)
    theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-12)  # unit vectors, [D, L]

    x_proj = x @ theta  # [N_x, L]
    y_proj = y @ theta  # [N_y, L]

    x_sorted, _ = torch.sort(x_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)

    if x_sorted.shape[0] != y_sorted.shape[0]:
        # Unequal sample counts: interpolate the smaller sorted sequence onto
        # the larger's quantile grid (matches quantile positions, not raw
        # ranks) - still O(N log N), no O(B^2) pairwise cost anywhere.
        n_x, n_y = x_sorted.shape[0], y_sorted.shape[0]
        n_q = max(n_x, n_y)
        q = torch.linspace(0.0, 1.0, n_q, device=x.device, dtype=x.dtype)
        x_sorted = _interp_quantiles(x_sorted, q)
        y_sorted = _interp_quantiles(y_sorted, q)

    return (x_sorted - y_sorted).abs().mean()


def _interp_quantiles(sorted_vals: Tensor, q: Tensor) -> Tensor:
    """Linearly interpolate a [N, L] sorted tensor's empirical quantiles at
    positions ``q`` in [0, 1], per projection column. Returns [len(q), L]."""
    n = sorted_vals.shape[0]
    positions = q * (n - 1)
    lower = positions.floor().long().clamp(0, n - 1)
    upper = positions.ceil().long().clamp(0, n - 1)
    frac = (positions - lower.float()).unsqueeze(-1)
    lower_vals = sorted_vals[lower]
    upper_vals = sorted_vals[upper]
    return lower_vals + frac * (upper_vals - lower_vals)
