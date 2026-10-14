"""Latent-noise-conditioned SamudraV2: FiLM on the InstanceNorm slots.

Turns the deterministic ConvNeXt U-Net into a genuinely stochastic map
`y = f(x, z)` so that fair-CRPS training can learn a *model-error*
distribution instead of only amplifying the IC perturbation.

Why the InstanceNorm slots
--------------------------
`ConvNeXtBlock` uses `nn.InstanceNorm2d(..., affine=False)` (the default),
so after each normalization the per-channel amplitude information has been
erased and *nothing* writes it back. Inserting

    h <- (1 + gamma(z)) * h + beta(z)

there fills an empty slot rather than competing with pretrained weights,
and it is the one place the network cannot route around: the amplitude of
that feature is now a function of `z` alone. That is what stops the
"noise ignoring" collapse a plain extra input channel suffers from.

`gamma`/`beta` come from two sources that are added together:
  * a global latent vector `z_vec` (m, z_dim) -> per-channel scalars,
    i.e. classic FiLM: how strongly each learned mode fires this draw;
  * a coarse random field `z_field` (m, field_ch, H0, W0), bilinearly
    resampled to each block's resolution -> spatially varying gamma/beta
    (SPADE-lite). Since the U-Net runs at four resolutions, one coarse
    field automatically acts at several scales. Without this term the
    perturbation is globally uniform, which is far too weak for the ocean.

Zero initialization
-------------------
The last layer of every FiLM head is zero-initialized, so at construction
`gamma = 0, beta = 0` and the model is **bitwise identical** to the
pretrained deterministic checkpoint. The CRPS validation baseline and the
`ckpt_best` gate therefore stay comparable with the non-FiLM run.
`assert_zero_init_identity` is the acceptance test for this.

Gradient checkpointing
----------------------
The SamudraV2 config runs with `checkpointing: all`, i.e. every U-Net layer
is wrapped in `torch.utils.checkpoint`. `z` is therefore sampled *outside*
the model and threaded through as an **explicit argument** to
`checkpoint(...)`, so the backward recomputation always sees the same noise.
Sampling inside a checkpointed region would rely on RNG-state preservation
and would silently break under any refactor.

Nothing in `ocean_emulators/` is modified: the pretrained model is built and
loaded exactly as before, then `make_stochastic()` swaps in the subclasses
and wraps the ConvNeXt blocks in place, reusing the original submodules so
no pretrained weight is touched.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ocean_emulators.models.modules.blocks import (
    BilinearUpsample,
    ConvNeXtBlock,
    CoreBlock,
    TransposedConvUpsample,
    ZonallyPeriodicBilinearUpsample,
)
from ocean_emulators.models.modules.unet_backbone import UNetBackbone
from ocean_emulators.models.samudra import Samudra

import numpy as np

logger = logging.getLogger("stochastic_samudra")

__all__ = [
    "FiLMHead",
    "FiLMConvNeXtBlock",
    "FiLMUNetBackbone",
    "StochasticSamudra",
    "make_stochastic",
    "sample_z",
    "film_parameters",
    "film_weight_norm",
    "assert_zero_init_identity",
]

# Normalization layers whose (missing) affine slot FiLM takes over.
_NORM_TYPES = (nn.InstanceNorm2d, nn.BatchNorm2d, nn.GroupNorm)


class FiLMHead(nn.Module):
    """(z_vec, z_field) -> (gamma, beta) for one normalization slot.

    Output layers are zero-initialized, so the head is the identity
    transform until training moves it.
    """

    def __init__(self, z_dim: int, channels: int, field_ch: int = 0,
                 hidden: int = 128):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        if field_ch > 0:
            self.spatial = nn.Conv2d(field_ch, 2 * channels, kernel_size=1)
            nn.init.zeros_(self.spatial.weight)
            nn.init.zeros_(self.spatial.bias)
        else:
            self.spatial = None

    def forward(self, h: torch.Tensor, z_vec: torch.Tensor,
                z_field: torch.Tensor | None = None) -> torch.Tensor:
        # (m, 2C) -> (m, 2C, 1, 1): global, per-channel modulation.
        mod = self.mlp(z_vec.to(h.dtype))[:, :, None, None]
        if self.spatial is not None and z_field is not None:
            # Resample the coarse field to this block's resolution: the same
            # draw therefore modulates every U-Net level at its own scale.
            zf = F.interpolate(z_field.to(h.dtype), size=h.shape[-2:],
                               mode="bilinear", align_corners=False)
            mod = mod + self.spatial(zf)
        gamma, beta = mod.chunk(2, dim=1)
        return h * (1.0 + gamma) + beta


class FiLMConvNeXtBlock(CoreBlock):
    """A ConvNeXtBlock with FiLM applied after every normalization layer.

    Constructed *from an existing block*: `convblock` and `skip_module` are
    the very same objects, so all pretrained weights are preserved by
    reference and `state_dict()` keys for them are unchanged apart from the
    added `heads.*` entries.

    Subclassing `CoreBlock` matters: `UNetBackbone.forward` selects the skip
    connections with `isinstance(layer, CoreBlock)`, so a plain `nn.Module`
    wrapper would silently sever every skip connection in the U-Net.
    """

    def __init__(self, block: ConvNeXtBlock, z_dim: int, field_ch: int,
                 hidden: int = 128):
        # kernel_size/dilation only feed CoreBlock's N_pad computation, which
        # we override with the source block's value right after.
        super().__init__(
            in_channels=block.in_channels,
            out_channels=block.out_channels,
            kernel_size=1,
            dilation=1,
            pad=block.pad,
        )
        self.N_pad = block.N_pad
        self.convblock = block.convblock
        self.skip_module = block.skip_module
        self.checkpoint_simple = block.checkpoint_simple

        self.heads = nn.ModuleList([
            FiLMHead(z_dim, _norm_channels(layer), field_ch, hidden)
            for layer in self.convblock if isinstance(layer, _NORM_TYPES)
        ])

    def forward(self, x: torch.Tensor, z_vec: torch.Tensor | None = None,
                z_field: torch.Tensor | None = None) -> torch.Tensor:
        # Mirrors ConvNeXtBlock.forward, with a FiLM head after each norm.
        skip = self.skip_module(x)
        h = 0
        for layer in self.convblock:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1:
                x = F.pad(x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad)
                x = F.pad(x, (0, 0, self.N_pad, self.N_pad), mode="constant")
            if self.checkpoint_simple and not isinstance(layer, nn.Conv2d):
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if isinstance(layer, _NORM_TYPES):
                if z_vec is not None:
                    x = self.heads[h](x, z_vec, z_field)
                h += 1
        return skip + x


def _norm_channels(layer: nn.Module) -> int:
    if isinstance(layer, nn.GroupNorm):
        return layer.num_channels
    return layer.num_features


class FiLMUNetBackbone(UNetBackbone):
    """UNetBackbone that threads (z_vec, z_field) into its FiLM blocks.

    The conditioning is handed over by `set_conditioning` and read **once**,
    at the top of `forward`, before any checkpointing happens; from there it
    travels as explicit arguments, which is what makes recomputation in the
    backward pass safe.
    """

    def set_conditioning(self, z: tuple | None) -> None:
        self._z = z

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        z_vec, z_field = getattr(self, "_z", None) or (None, None)

        skip_inputs: list[torch.Tensor] = [
            torch.zeros_like(fts) for _ in range(self.num_steps)
        ]
        count = 0
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                fts = F.pad(fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad)
                fts = F.pad(fts, (0, 0, self.N_pad, self.N_pad), mode="constant")

            conditioned = isinstance(layer, FiLMConvNeXtBlock) and z_vec is not None
            args = (fts, z_vec, z_field) if conditioned else (fts,)
            if self.checkpoint_all:
                fts = checkpoint(layer, *args, use_reentrant=False)  # type: ignore[arg-type]
            else:
                fts = layer(*args)

            # Skip-connection bookkeeping, unchanged from UNetBackbone.
            if count < self.num_steps:
                if isinstance(layer, CoreBlock):
                    skip_inputs[count] = fts
                    count += 1
            else:
                if isinstance(layer, (BilinearUpsample, TransposedConvUpsample,
                                      ZonallyPeriodicBilinearUpsample)):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        skip_inputs[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads = [
                        pads[1] // 2, pads[1] - pads[1] // 2,
                        pads[0] // 2, pads[0] - pads[0] // 2,
                    ]
                    fts = F.pad(fts, pads)
                    fts += skip_inputs[int(2 * self.num_steps - count - 1)]
                    count += 1
        return fts


class StochasticSamudra(Samudra):
    """Samudra whose `forward_once` accepts a latent noise draw.

    Two ways to supply it:

    * explicitly — `forward_once(x, ctx, z=(z_vec, z_field))`;
    * automatically — call `enable_auto_z(...)` once and every forward draws
      its own `z` from the model's own generator. This keeps the signature
      `forward_once(x, ctx)` intact so helpers written against the plain
      model (e.g. the CRPS script's `forward_members` / `validate`) work
      unchanged. The draw happens here, outside any checkpointed region.
    """

    def enable_auto_z(self, z_dim: int, field_ch: int,
                      field_hw: tuple[int, int], seed: int) -> None:
        self._auto_z = {"z_dim": z_dim, "field_ch": field_ch,
                        "field_hw": tuple(field_hw)}
        self._z_gen = torch.Generator()  # CPU: identical behaviour on any device
        self._z_gen.manual_seed(int(seed))

    def set_z_seed(self, seed: int) -> None:
        """Pin the auto-z stream (used so every validation sees the same draws)."""
        if getattr(self, "_z_gen", None) is not None:
            self._z_gen.manual_seed(int(seed))

    def _draw_auto_z(self, batch: int, device: torch.device):
        cfg = getattr(self, "_auto_z", None)
        if cfg is None:
            return None
        z_vec = torch.randn(batch, cfg["z_dim"], generator=self._z_gen)
        z_field = (
            torch.randn(batch, cfg["field_ch"], *cfg["field_hw"], generator=self._z_gen)
            if cfg["field_ch"] > 0 else None
        )
        return (z_vec.to(device),
                None if z_field is None else z_field.to(device))

    def forward_once(self, fts: torch.Tensor, ctx, z=None) -> torch.Tensor:
        if z is None:
            z = self._draw_auto_z(fts.shape[0], fts.device)
        self.unet.set_conditioning(z)
        try:
            return super().forward_once(fts, ctx)
        finally:
            self.unet.set_conditioning(None)


# =====================================================================
# Conversion / helpers
# =====================================================================
WHERE_CHOICES = ("all", "bottleneck", "decoder")


def make_stochastic(model: Samudra, z_dim: int = 64, field_ch: int = 4,
                    hidden: int = 128, where: str = "bottleneck") -> Samudra:
    """Wrap `model`'s ConvNeXt blocks with FiLM, in place.

    `where` selects the injection depth:
      * "all"        — every ConvNeXt block;
      * "bottleneck" — the U-Net middle block and the whole decoder path
                       (default): large-scale, coherent perturbations, which
                       is the right prior for *structural* model error when a
                       separate mechanism already handles the finest scales;
      * "decoder"    — decoder blocks only (finest scales).

    Class reassignment is used so the config-driven builder in
    `ocean_emulators.config` needs no changes and the pretrained weights are
    never re-loaded; the subclasses add no instance state of their own.
    """
    if where not in WHERE_CHOICES:
        raise ValueError(f"--film-where must be one of {WHERE_CHOICES}, got {where!r}")

    layers = model.unet.layers
    block_idx = [i for i, l in enumerate(layers) if isinstance(l, ConvNeXtBlock)]
    if not block_idx:
        raise RuntimeError("no ConvNeXtBlock found in unet.layers — "
                           "FiLM injection only supports the ConvNeXt backbone")
    if where == "bottleneck":
        block_idx = block_idx[len(block_idx) // 2:]
    elif where == "decoder":
        block_idx = block_idx[len(block_idx) // 2 + 1:]

    for i in block_idx:
        layers[i] = FiLMConvNeXtBlock(layers[i], z_dim, field_ch, hidden)

    n_layers = sum(len(l.heads) for l in model.unet.layers
                   if isinstance(l, FiLMConvNeXtBlock))
    logger.info("FiLM: %d blocks wrapped, %d modulation layers",
                len(block_idx), n_layers)

    model.unet.__class__ = FiLMUNetBackbone
    model.unet.set_conditioning(None)
    model.__class__ = StochasticSamudra
    return model


def film_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Every parameter belonging to a FiLM head (i.e. everything new)."""
    return [p for m in model.modules() if isinstance(m, FiLMHead)
            for p in m.parameters()]


@torch.no_grad()
def film_weight_norm(model: nn.Module) -> float:
    """Mean |w| of the FiLM output layers — 0 at init.

    Stays at ~0 iff the network is ignoring the noise, so it is the cheap
    early-warning signal for the collapse this design exists to prevent.
    """
    total, count = 0.0, 0
    for m in model.modules():
        if isinstance(m, FiLMHead):
            for w in (m.mlp[-1].weight, None if m.spatial is None else m.spatial.weight):
                if w is not None:
                    total += w.abs().sum().item()
                    count += w.numel()
    return total / max(1, count)


@torch.no_grad()
def scale_film(model: nn.Module, s: float) -> int:
    """Multiply every FiLM head's zero-init output layer by `s`; returns #tensors.

    `mod` in FiLMHead.forward is linear in exactly these tensors, so this scales
    gamma/beta EXACTLY by `s` — s=0 restores the deterministic map, s>1 amplifies
    the learned model-error amplitude. Used by the strength sweep
    (`eval_film_candidates.py`) and to build a shipping checkpoint
    (`bake_film_scale.py`, which performs the same multiplication on a state
    dict instead of a live module).
    """
    n = 0
    for m in model.modules():
        if isinstance(m, FiLMHead):
            for t in (m.mlp[-1].weight, m.mlp[-1].bias):
                t.mul_(s)
                n += 1
            if m.spatial is not None:
                for t in (m.spatial.weight, m.spatial.bias):
                    t.mul_(s)
                    n += 1
    return n


def sample_z(global_members: list[int], z_dim: int, field_ch: int,
             field_hw: tuple[int, int], device: torch.device,
             seed0: int, step: int = 0):
    """Reproducible per-member latent draw -> (z_vec, z_field).

    Seeded by the *global* member index (as `perturbed_members` is), so each
    ensemble member gets its own draw no matter which rank simulates it, and
    a rerun reproduces the ensemble exactly.

    `z_field` is drawn at coarse resolution and bilinearly upsampled inside
    the FiLM heads; the coarse grid spacing sets the correlation length of
    the resulting perturbation, which is why plain `randn` suffices here.
    """
    gen = torch.Generator()
    z_vec, z_field = [], []
    for g in global_members:
        gen.manual_seed(int(seed0 + g * 1_000_003 + step * 7_919))
        z_vec.append(torch.randn(z_dim, generator=gen))
        if field_ch > 0:
            z_field.append(torch.randn(field_ch, *field_hw, generator=gen))
    return (
        torch.stack(z_vec).to(device),
        torch.stack(z_field).to(device) if field_ch > 0 else None,
    )


@torch.no_grad()
def assert_zero_init_identity(model: StochasticSamudra, x: torch.Tensor, ctx,
                              z, atol: float = 0.0) -> None:
    """Acceptance test: with zero-initialized FiLM, `z` must not matter.

    If this fails the heads are wired into the wrong place, and every metric
    compared against the deterministic baseline would be meaningless.
    """
    was_training = model.training
    model.eval()
    try:
        y_off = model.forward_once(x, ctx, z=None if getattr(model, "_auto_z", None)
                                   is None else z)
        model.unet.set_conditioning(None)
        y_ref = Samudra.forward_once(model, x, ctx)
        y_on = model.forward_once(x, ctx, z=z)
    finally:
        model.train(was_training)

    for name, y in (("z=given", y_off), ("z=explicit", y_on)):
        if atol == 0.0:
            ok = torch.equal(y, y_ref)
        else:
            ok = torch.allclose(y, y_ref, atol=atol, rtol=0.0)
        if not ok:
            diff = (y - y_ref).abs().max().item()
            raise RuntimeError(
                f"zero-init identity check failed for {name}: max|diff|={diff:.3e}. "
                "FiLM is not zero-initialized or is attached in the wrong place."
            )
