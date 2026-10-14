"""MaskedDiT: the SI_PlaSim DiT with all-land patch tokens dropped from the
transformer body.

Subclasses `modules.models.DiT` and overrides only `forward`, so
`modules/models/DiT.py` stays as copied. Everything else -- patch embed, RoPE,
the blocks, the subpixel head, weight init, the Muon parameter groups -- is
inherited untouched.

What changes, and only this:

    x = patch_embed(...)                  # (B, N, dim)   as before
    x = x[:, keep_idx]                    # (B, N_keep, dim)   <-- gather
    ... all transformer blocks run on N_keep ...
    full = zeros(B, N, dim); full[:, keep_idx] = x           <-- scatter
    x = unpatchify(full)                  # as before

RoPE frequencies and the cross-attention context are gathered with the same
index, so geometry and conditioning stay aligned with the surviving tokens.

Why this is simpler than MaskDiT (arXiv:2306.09305), which drops tokens too:
its mask is redrawn every step and hides real information, so it needs a learnable
mask token, an auxiliary MAE loss on the masked patches, a separate lightweight
decoder, and an unmasking fine-tuning phase to close the train/inference gap.
Ours is static and the dropped tokens are structurally zero, so:

  * dropped patches are all-land -> their correct output is exactly 0, and the
    subpixel head's 1x1 conv has bias=False, so zero tokens produce zero output.
    Scattering zeros back IS the exact inverse, no mask token required.
  * the mask is identical at train and sample time -> no distribution shift, no
    fine-tuning phase.
  * N_keep is a compile-time constant -> no attention mask, no ragged kernels.

Caveat that bites if forgotten: a KEPT patch is only 87.7% ocean on average, so
token dropping does NOT zero the land inside surviving patches. The cell-level
mask must still be applied to the model output. That is done by the caller
(SI_Ocean.py wraps the model call), not here.
"""
import torch
from einops import rearrange

from modules.layers.unpatchify import sphere_pad
from modules.models.DiT import DiT, DiTCrossAttentionBlock


class MaskedDiT(DiT):
    """DiT that runs its transformer body on ocean tokens only.

    Args:
        keep_idx: (N_keep,) int64 token indices to keep, row-major over the
            (nlat//p, nlon//p) patch grid -- i.e. matching DiT's own flatten.
        n_tokens: total token count before dropping, for the scatter target.
        **dit_kwargs: forwarded verbatim to DiT.
    """

    def __init__(self, keep_idx, n_tokens, **dit_kwargs):
        super().__init__(**dit_kwargs)
        assert self.pad_lat == 0 and self.pad_lon == 0, (
            'MaskedDiT assumes patch_size divides the grid exactly; with padding '
            'the token grid would not match the mask used to build keep_idx.')
        assert self.grid_x * self.grid_y == n_tokens, (
            f'keep_idx was built for {n_tokens} tokens but this DiT has '
            f'{self.grid_x * self.grid_y}')
        self.n_tokens = n_tokens
        self.register_buffer('keep_idx', torch.as_tensor(keep_idx, dtype=torch.long))

    def forward(self, x_noised, cond, t, c_grid=None, c_scalar=None):
        """Same signature and semantics as DiT.forward."""
        nlat, nlon = self.nlat, self.nlon

        if self.c_grid_embed is not None and c_grid is not None:
            c_grid_emb = self.c_grid_embed(c_grid)
        elif c_grid is not None:
            c_grid_emb = c_grid
        else:
            c_grid_emb = None

        if self.scalar_embedder is not None and c_scalar is not None:
            c_scalar_emb = self.scalar_embedder(c_scalar)
        else:
            c_scalar_emb = None

        cond_list = [x_noised, cond]
        if c_grid_emb is not None:
            cond_list.append(c_grid_emb)
        if c_scalar_emb is not None:
            cond_list.append(c_scalar_emb)
        x_input = torch.cat(cond_list, dim=1)

        x_input = sphere_pad(x_input, padding=(self.pad_lon_left, self.pad_lon_right,
                                               self.pad_lat_top, self.pad_lat_bottom))
        x = self.patch_embed_main(x_input.permute(0, 2, 3, 1))
        x = rearrange(x, 'b ny nx c -> b (ny nx) c')            # (B, N, dim)

        rope = self.compute_rope_freqs(x.device)
        rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon = rope

        context = None
        if self.ca_embed is not None:
            ctx_parts = []
            if c_grid_emb is not None:
                ctx_parts.append(c_grid_emb)
            if c_scalar_emb is not None:
                ctx_parts.append(c_scalar_emb)
            assert len(ctx_parts) > 0, 'num_ca_blocks > 0 requires c_grid or c_scalar'
            ctx_input = torch.cat(ctx_parts, dim=1)
            ctx_input = sphere_pad(ctx_input, padding=(self.pad_lon_left, self.pad_lon_right,
                                                       self.pad_lat_top, self.pad_lat_bottom))
            context = self.ca_embed(ctx_input.permute(0, 2, 3, 1))
            context = rearrange(context, 'b ny nx c -> b (ny nx) c')

        # ---- drop all-land tokens -------------------------------------------
        n_full = x.shape[1]
        x = x[:, self.keep_idx]
        if context is not None:
            context = context[:, self.keep_idx]
        rope_cos_lat = rope_cos_lat[:, self.keep_idx]
        rope_sin_lat = rope_sin_lat[:, self.keep_idx]
        rope_cos_lon = rope_cos_lon[:, self.keep_idx]
        rope_sin_lon = rope_sin_lon[:, self.keep_idx]

        if len(t.shape) == 1:
            t = t[:, None]
        t_emb = self.t_embedder(t)

        for block in self.sa_blocks:
            if isinstance(block, DiTCrossAttentionBlock):
                x = block(x, context, t_emb,
                          rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
            else:
                x = block(x, t_emb,
                          rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        # ---- scatter back into the full token grid --------------------------
        # Dropped tokens go back as exact zeros. Their patches are all-land, and
        # the subpixel head's conv has bias=False, so zero token -> zero output.
        full = x.new_zeros(x.shape[0], n_full, x.shape[2])
        full = full.index_copy(1, self.keep_idx, x)
        x = full

        x = self.unpatchify_layer(x, t_emb)
        x = x.permute(0, 3, 1, 2)
        if self.pad_lat > 0 or self.pad_lon > 0:
            x = x[:, :, self.pad_lat_top:self.pad_lat_top + nlat,
                  self.pad_lon_left:self.pad_lon_left + nlon]
        return x
