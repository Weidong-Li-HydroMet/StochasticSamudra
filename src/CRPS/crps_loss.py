"""Fair (unbiased) ensemble CRPS for SamudraV2 fine-tuning.

All functions operate on NORMALIZED model-space tensors of shape
(M, C, H, W) for the ensemble and (C, H, W) for the ground truth, where
C = (hist+1) * n_prognostic (SamudraV2: 2 * 77 = 154 channels).

Fair CRPS (Ferro 2014), per grid point and channel:

    CRPS_fair = (1/M) * sum_i |x_i - y|                      # skill
              - 1/(2*M*(M-1)) * sum_{i != j} |x_i - x_j|     # spread

The spread term is computed with the O(M log M) sorted form instead of the
O(M^2) pairwise sum:

    sum_{i<j} (x_(j) - x_(i)) = sum_k (2k - M + 1) * x_(k)   # k = 0..M-1, ascending

so that no (M, M, C, H, W) intermediate is ever materialized.

Reduction is a weighted mean with weights = cos(lat) * wet_mask, i.e. land
points and the polar convergence are excluded exactly as in the latitude-
weighted losses used elsewhere in this project.
"""

from __future__ import annotations

import torch
from torch import Tensor


def make_loss_weights(label_mask: Tensor, lat_deg: Tensor | None = None) -> Tensor:
    """Per-channel loss weights: wet points only, optionally cos(lat) scaled.

    label_mask: bool (C, H, W)  — ds.ctx.label_mask (land = False)
    lat_deg:    float (H,) latitude in degrees, or None for uniform weights
                (grid-uniform over wet points — matches the original SamudraV2
                training loss).
    Returns float32 (C, H, W).
    """
    weights = label_mask.to(torch.float32)
    if lat_deg is not None:
        coslat = torch.cos(torch.deg2rad(lat_deg.to(torch.float32)))
        coslat = coslat.clamp_min(0.0).view(1, -1, 1)
        weights = weights * coslat.to(label_mask.device)
    return weights


def weighted_mean(field: Tensor, weights: Tensor) -> Tensor:
    """Weighted mean of `field` (C, H, W) with `weights` (C, H, W)."""
    return (field * weights).sum() / weights.sum()


def weighted_mean_per_channel(field: Tensor, weights: Tensor) -> Tensor:
    """Per-channel weighted mean: (C, H, W) -> (C,).

    Each channel normalizes by its own wet area, so deep channels with less
    ocean are not down-weighted relative to surface channels.
    """
    return (field * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1e-12)


def fair_spread_field(ens: Tensor) -> Tensor:
    """Fair spread term 1/(2M(M-1)) * sum_{i!=j} |x_i - x_j|, per grid point.

    ens: (M, C, H, W). Returns (C, H, W). Uses the sorted O(M log M) identity.
    """
    M = ens.shape[0]
    if M < 2:
        return torch.zeros_like(ens[0])
    x_sorted, _ = torch.sort(ens, dim=0)
    coef = (2.0 * torch.arange(M, device=ens.device, dtype=ens.dtype) - M + 1).view(
        M, 1, 1, 1
    )
    return (coef * x_sorted).sum(dim=0) / (M * (M - 1))


def fair_crps_decomposed(ens: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Exact fair CRPS skill/spread fields for a full ensemble.

    ens: (M, C, H, W), target: (C, H, W).
    Returns (skill, spread) fields of shape (C, H, W); crps = skill - spread.
    """
    skill = (ens - target.unsqueeze(0)).abs().mean(dim=0)
    spread = fair_spread_field(ens)
    return skill, spread


def fair_crps(ens: Tensor, target: Tensor, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Weighted scalar (crps, skill, spread) for logging/validation."""
    skill_f, spread_f = fair_crps_decomposed(ens, target)
    skill = weighted_mean(skill_f, weights)
    spread = weighted_mean(spread_f, weights)
    return skill - spread, skill, spread


def fair_crps_per_channel(
    ens: Tensor, target: Tensor, weights: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-channel weighted (crps, skill, spread), each of shape (C,)."""
    skill_f, spread_f = fair_crps_decomposed(ens, target)
    skill = weighted_mean_per_channel(skill_f, weights)
    spread = weighted_mean_per_channel(spread_f, weights)
    return skill - spread, skill, spread


def fair_crps_surrogate(
    local: Tensor,
    remote: Tensor | None,
    target: Tensor,
    m_total: int,
    weights: Tensor,
    channel_scale: Tensor | None = None,
    alpha: float = 1.0,
) -> Tensor:
    """Ensemble-parallel surrogate loss for (almost-)fair CRPS.

    `alpha` interpolates between the fair and the standard CRPS, following
    AIFS-CRPS (Lang et al., arXiv:2412.15832):

        afCRPS = alpha * fCRPS + (1 - alpha) * CRPS

    The two differ only in the spread coefficient — 1/(2M(M-1)) for fair,
    1/(2M^2) for standard — so the blend is a single coefficient here. The
    standard term rewards spread slightly less, which pulls the optimum a
    little toward the deterministic solution: it costs a little calibration
    and buys back ensemble-mean RMSE, and it removes the degeneracy pure
    fair CRPS can hit at small M in reduced precision.

    alpha=1.0 (the default) is exactly the pure fair CRPS used before, so
    existing callers are unaffected. Diagnostics and validation should keep
    using `fair_crps` regardless, so metrics stay comparable across runs.

    `local` (m_local, C, H, W) carries gradients; `remote`
    (m_total - m_local, C, H, W) holds the other ranks' members, detached.
    The gradient of this scalar w.r.t. the local members equals the gradient
    of the full fair CRPS (its *value* is only the local share — use
    `fair_crps` on the gathered ensemble for logging):

        dL/dx_r = (1/M) sign(x_r - y) - 1/(M(M-1)) * sum_{j != r} sign(x_r - x_j)

    Both local-local pairs (counted once here, twice in sum_{i!=j}) and
    local-remote pairs (only the local-differentiable direction exists here)
    end up with the same 1/(M(M-1)) coefficient.

    channel_scale (C,), if given, applies DynamicLoss-style per-channel
    weights: the loss becomes mean_c(scale_c * per-channel weighted CRPS)
    instead of one global weighted mean.
    """
    M = m_total
    m = local.shape[0]
    skill = (local - target.unsqueeze(0)).abs().sum(dim=0) / M
    spread = torch.zeros_like(skill)
    for i in range(m):
        for j in range(i + 1, m):
            spread = spread + (local[i] - local[j]).abs()
    if remote is not None and remote.shape[0] > 0:
        for i in range(m):
            for j in range(remote.shape[0]):
                spread = spread + (local[i] - remote[j].to(local.dtype)).abs()
    # Pairs are counted once above, twice in the sum_{i!=j} of the definitions,
    # hence 1/(M(M-1)) and 1/M^2 rather than half of each.
    coef = alpha / (M * (M - 1)) + (1.0 - alpha) / (M * M)
    field = skill - spread * coef
    if channel_scale is None:
        return weighted_mean(field, weights)
    return (weighted_mean_per_channel(field, weights) * channel_scale).mean()
