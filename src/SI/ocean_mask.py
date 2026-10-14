"""Static land/ocean mask: token keep-index, cell mask, and the loss weights.

Everything here is computed ONCE at startup and registered as buffers. That is the
whole advantage of a land mask over MaskDiT's random mask: the index set never
changes, so the transformer's sequence length is a compile-time constant and we
need no mask token, no auxiliary reconstruction loss, no asymmetric decoder, and
no unmasking fine-tuning phase (see MDiT/DISCUSSION.md section 10 for the
side-by-side).

Three distinct objects, easy to confuse:

  cell_mask  (77, H, W)   per-channel wet cells. Zeroes model output and noise.
  keep_idx   (N_keep,)    which PATCH tokens survive. A token is kept if its
                          p x p footprint contains any ocean at all, in any
                          channel -- and because mask_0 is a superset of every
                          depth mask, that is exactly mask_0's patch cover.
  loss_w     (1,77,H,W)   cell_mask * cos(lat), normalised so each channel sums
                          to 1. Folds three fixes into one tensor: land carries no
                          loss, polar cells stop being over-counted, and every
                          channel gets equal weight.

Measured at p=16 on the OM4 0.25 deg grid:
  tokens 4050 -> 3075 kept (75.9%): 1.32x fewer tokens, 1.73x less attention.
  Modest, because coastlines are fine-grained -- most patches touch some water.
  Kept patches are 87.7% ocean by area, so a CELL-level mask is still required on
  the output; dropping tokens alone does not zero the land inside a kept patch.
  45.7% of the un-masked loss was being spent on structural zeros.
  22.8% of ocean cells sit poleward of 60 deg but only 8.7% of the ocean AREA.
"""
import os

import numpy as np
import torch

# 1deg: built by 0-prepare_data.py (mask_k per level, mask_0 for zos).
SI_DATA = os.environ.get('SI_DATA', '/data/weidong/InMOS_1deg/SI_data')
MASK_PATH = os.path.join(SI_DATA, 'prog_mask.npy')
LAT_PATH = os.path.join(SI_DATA, 'lat.npy')


def load_cell_mask(path=MASK_PATH):
    """(77, H, W) float32, 1 = ocean."""
    return np.load(path).astype(np.float32)


def token_keep_index(cell_mask, patch_size):
    """Indices of the patch tokens that contain at least one ocean cell.

    Returns (keep_idx int64 (N_keep,), n_tokens int). Token order matches
    DiT's `rearrange(x, 'b ny nx c -> b (ny nx) c')`, i.e. row-major over
    (nlat//p, nlon//p).
    """
    _, H, W = cell_mask.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f'patch_size {patch_size} must divide {H}x{W} (DiT would pad otherwise)'
    any_ocean = cell_mask.max(axis=0)                       # mask_0, the superset
    grid = any_ocean.reshape(H // patch_size, patch_size,
                             W // patch_size, patch_size).max(axis=(1, 3))
    flat = grid.reshape(-1) > 0
    return np.nonzero(flat)[0].astype(np.int64), int(flat.size)


def area_weights(cell_mask, lat=None):
    """(1, 77, H, W) float32: cell_mask * cos(lat), each channel summing to 1.

    1deg: ``lat`` = the zarr cell-centre latitudes (not exactly uniform);
    None falls back to a uniform grid.

    Used as the loss reduction weight. Per-channel normalisation is what makes the
    reduction equal-weighted across the 77 channels; combined with the anomaly-unit
    data from L0, every channel then contributes on the same scale.
    """
    C, H, W = cell_mask.shape
    if lat is None:
        lat = -90.0 + (180.0 / H) * (np.arange(H) + 0.5)
    w = cell_mask * np.cos(np.deg2rad(lat))[None, :, None]
    w = w / np.maximum(w.sum(axis=(1, 2), keepdims=True), 1e-12)
    return w.astype(np.float32)[None]


class OceanMask(torch.nn.Module):
    """Holds the three tensors above as buffers so `.to(device)` moves them all."""

    def __init__(self, patch_size, path=MASK_PATH, lat_path=LAT_PATH):
        super().__init__()
        cm = load_cell_mask(path)
        lat = np.load(lat_path) if lat_path and os.path.exists(lat_path) else None
        keep, n_tokens = token_keep_index(cm, patch_size)
        self.n_tokens = n_tokens
        self.n_keep = int(keep.size)
        self.patch_size = patch_size
        self.register_buffer('cell_mask', torch.from_numpy(cm).unsqueeze(0))   # (1,77,H,W)
        self.register_buffer('keep_idx', torch.from_numpy(keep))               # (N_keep,)
        self.register_buffer('loss_w', torch.from_numpy(area_weights(cm, lat)))     # (1,77,H,W)

    def summary(self):
        cm = self.cell_mask
        return (f'tokens {self.n_tokens} -> {self.n_keep} kept '
                f'({100*self.n_keep/self.n_tokens:.1f}%, '
                f'{self.n_tokens/self.n_keep:.2f}x fewer, '
                f'{(self.n_tokens/self.n_keep)**2:.2f}x less attention); '
                f'ocean cells {float(cm.mean()):.4f} of the grid')
