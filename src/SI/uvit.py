"""U-ViT backbone: conv encoder -> MaskedDiT at patch_size=1 -> conv decoder.

1deg fork of Samudra-SI/uvit.py (2026-10-13). Changes, and only these:

1. NO big skip. Samudra-SI returned y_hat = carry(x_raw) + s_inc * g, an
   identity-initialised 3x3 conv on the source state plus a scaled increment, so
   the network learned the tendency. Here the decoder head IS the output:
   y_hat = g, the full (Samudra-normalized) field of the next state. The U-Net
   encoder->decoder skips inside the conv ladder stay -- they are part of the
   backbone, not a residual around it.
2. Depth is a parameter: len(widths) conv levels, total stride 2**len(widths).
   The 0.25deg ladder was 3 levels (stride 8, 720x1440 -> 90x180); at 1deg
   180/8 is not an integer, so the default here is 2 levels (stride 4,
   180x360 -> 45x90 latent = 4050 tokens before land dropping).
3. Input = [X_t (77), cond (154 = [a_t, da_t]), c_grid (6 forcing)] = 237 ch.

Unchanged: t-conditioned partial-conv blocks with FiLM (zero-init = identity),
static PConv renormalisation on per-level union masks, nearest-upsample decoder
(no checkerboard), MaskedDiT on the latent with calendar embedding + cross-attn
blocks, zero-initialised head (so y_hat = 0 at init; there is no persistence
start any more -- that was the carry's job).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from modules.layers.unpatchify import sphere_pad
from masked_dit import MaskedDiT


def _pconv_stats(mask_2d):
    """(renorm, mask) buffers for one resolution: mask (1,1,h,w) in {0,1},
    renorm = 9 / (#ocean cells in the spherically-padded 3x3 window)."""
    m = mask_2d.reshape(1, 1, *mask_2d.shape[-2:]).float()
    count = F.conv2d(sphere_pad(m, (1, 1, 1, 1)),
                     torch.ones(1, 1, 3, 3, dtype=m.dtype, device=m.device))
    return 9.0 / count.clamp(min=1.0), m


class FiLM(nn.Module):
    """Per-channel scale/shift from t_emb; zero-init -> identity at start."""

    def __init__(self, t_dim, width):
        super().__init__()
        self.proj = nn.Linear(t_dim, 2 * width)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, t_emb):
        scale, shift = self.proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        return h * (1.0 + scale) + shift


class PConvBlock(nn.Module):
    """sphere_pad -> 3x3 conv -> static PConv renorm -> FiLM(t) -> GELU -> mask."""

    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=0, bias=True)
        self.film = FiLM(t_dim, c_out)

    def forward(self, h, t_emb, renorm, mask):
        h = self.conv(sphere_pad(h, (1, 1, 1, 1))) * renorm
        h = F.gelu(self.film(h, t_emb))
        return h * mask


class Down(nn.Module):
    """Stride-2 3x3 conv (spherically padded), masked at the coarse level."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=0)

    def forward(self, h, mask_coarse):
        return self.conv(sphere_pad(h, (1, 1, 1, 1))) * mask_coarse


class Up(nn.Module):
    """Nearest x2 upsample + PConvBlock -- no transposed conv, no checkerboard."""

    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.block = PConvBlock(c_in, c_out, t_dim)

    def forward(self, h, t_emb, renorm, mask):
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        return self.block(h, t_emb, renorm, mask)


class UViT(nn.Module):
    """Conv U-Net (len(widths) levels) around a patch_size=1 MaskedDiT.

    forward(x_noised, cond, t, c_grid, c_scalar) -> y_hat (B, n_out, H, W),
    the full field. keep_idx / n_tokens must come from an OceanMask built with
    patch_size = self.stride.
    """

    def __init__(self, keep_idx, n_tokens, surf_mask, n_out=77, cond_ch=154,
                 forcing_ch=6, nlat=180, nlon=360, widths=(256, 384),
                 latent_dim=512, dit_dim=768, dit_heads=12, dit_blocks=12,
                 dit_ca_blocks=2, dit_dropout=0.0, grad_checkpoint=True):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.n_levels = len(widths)
        self.stride = 2 ** self.n_levels
        assert nlat % self.stride == 0 and nlon % self.stride == 0, \
            f'{nlat}x{nlon} not divisible by encoder stride {self.stride}'
        c_in = n_out + cond_ch + forcing_ch                    # 77 + 154 + 6 = 237

        self.dit = MaskedDiT(
            keep_idx=keep_idx, n_tokens=n_tokens,
            in_channels=latent_dim, out_channels=latent_dim,
            dim=dit_dim, num_heads=dit_heads, num_blocks=dit_blocks,
            patch_size=1, nlat=nlat // self.stride, nlon=nlon // self.stride,
            dropout=dit_dropout,
            # 'subpixel', not 'vanilla': vanilla's FinalLayer JIT-fuses via
            # nvFuser, which crashed in the Samudra-SI env (missing nvrtc builtins).
            unpatch='subpixel',
            scalar_dim=2, c_grid_dim=0, c_grid_embed_dim=32,
            c_grid_downsample=0, num_ca_blocks=dit_ca_blocks,
            num_output_blocks=0)
        t_dim = dit_dim

        # Encoder: enc[l] at level l, down[l] to level l+1 (last one -> latent).
        widths = list(widths)
        nxt = widths[1:] + [latent_dim]
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c = c_in
        for l, w in enumerate(widths):
            self.enc.append(PConvBlock(c, w, t_dim))
            self.down.append(Down(w, nxt[l]))
            c = nxt[l]
        # Decoder, coarse -> fine: up[l] brings level l+1 to level l, dec[l]
        # fuses the skip from enc[l].
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = latent_dim
        for l in reversed(range(self.n_levels)):
            w = widths[l]
            self.up.append(Up(prev, w, t_dim))
            self.dec.append(PConvBlock(2 * w, w, t_dim))
            prev = w
        self.head = nn.Conv2d(widths[0], n_out, kernel_size=3, padding=0, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Per-level union masks: coarse cell is ocean if ANY subcell is.
        m = surf_mask.reshape(1, 1, *surf_mask.shape[-2:]).float()
        for l in range(self.n_levels):
            renorm, mask = _pconv_stats(F.max_pool2d(m, 2 ** l)[0, 0] if l else m[0, 0])
            self.register_buffer(f'renorm{l}', renorm, persistent=False)
            self.register_buffer(f'mask{l}', mask, persistent=False)
        self.register_buffer('mask_lat', F.max_pool2d(m, self.stride), persistent=False)

    def _level(self, l):
        return getattr(self, f'renorm{l}'), getattr(self, f'mask{l}')

    def _t_emb(self, t):
        if t.dim() == 1:
            t = t[:, None]
        return self.dit.t_embedder(t)

    def _run(self, module, *args):
        """Apply a conv stage, checkpointed during training-with-grad."""
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return _ckpt(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x_noised, cond, t, c_grid=None, c_scalar=None):
        t_emb = self._t_emb(t)
        h = torch.cat([x_noised, cond, c_grid], dim=1)
        skips = []
        for l in range(self.n_levels):
            renorm, mask = self._level(l)
            s = self._run(self.enc[l], h, t_emb, renorm, mask)
            skips.append(s)
            nmask = self._level(l + 1)[1] if l + 1 < self.n_levels else self.mask_lat
            h = self.down[l](s, nmask)

        # Latent transformer; zero-width cond slot (conditioning entered the encoder).
        h = self.dit(h, h[:, :0], t, None, c_scalar=c_scalar)

        for i, l in enumerate(reversed(range(self.n_levels))):
            renorm, mask = self._level(l)
            h = self._run(self.up[i], h, t_emb, renorm, mask)
            h = self._run(self.dec[i], torch.cat([h, skips[l]], dim=1), t_emb, renorm, mask)
        return self.head(sphere_pad(h, (1, 1, 1, 1))) * self.mask0
