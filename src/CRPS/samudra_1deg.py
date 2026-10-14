"""1-degree replacements for what the CRPS scripts imported from the 0.25deg rollout scripts.

The Sampling/Derecho CRPS code imports, from `samudra_om4_multi_ic_rollout` /
`samudra_om4_ensemble_rollout`:
    DEFAULT_CKPT_PATH, DEFAULT_CONFIG_PATH, DEFAULT_DATA_ROOT, DEFAULT_SURFACE_VARS,
    load_model_and_data, perlin_3octaves
and builds `ocean_emulators.datasets.InferenceDataset`s. This module provides the same
names for the 1deg checkpoint, so the training scripts change only their import lines
plus the few one-step lines marked `1deg:`.

What is 1deg-specific (see InMOS_1degree/README.md):
  * ONE-STEP model: decoder = 77 ch = x(t+1) from [x(t-1), x(t), f(t-1), f(t)].
    `load_model_and_data` reads `num_out` from the checkpoint (77), and
    `InferenceDataset` re-windows the dataset to [t, t+1 | t+2] with stride 1
    (5 days), with a 77-channel label mask, so `ds[i][0]` is the 154-ch input
    state, `_get_boundary` gives f(t-1), f(t) and `inference_target(i)` gives x(t+2).
  * BatchNorm (norm: batch). DEFAULT `set_bn_mode(model, "sample")`: per-sample statistics
    (how the net was trained; the only stable inference mode, checked 2026-10-12).
    `set_bn_mode(model, "eval")` pins every BatchNorm layer
    to eval mode (checkpoint running stats, never updated) whatever model.train() /
    model.eval() is called afterwards; "train" leaves PyTorch's default behaviour
    (batch statistics in train mode -- with 1 member per rank these are per-sample
    spatial statistics -- and running-stat updates). BN weight/bias train either way.
  * Data from OM4.zarr only: the 1deg npz cache lacks 2003-04-13..2009-12-29, which
    lies inside the 1975-2013 training window.
"""
from __future__ import annotations

import logging
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS, TensorMap
from ocean_emulators.datasets import InferenceDataset as _ZarrInferenceDataset
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.data import DataSource, Normalize
from ocean_emulators.utils.location import LocalLocation

logger = logging.getLogger("samudra_1deg")

REPO_ROOT = Path(__file__).resolve().parents[2]                        # .../InMOS_1degree
DEFAULT_CONFIG_PATH = REPO_ROOT / "checkpoints" / "SamudraV2_1deg" / "config.yaml"
DEFAULT_CKPT_PATH = Path("/data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt")
DEFAULT_DATA_ROOT = Path("/data/weidong/InMOS_1deg/om4_onedeg_v3")
DEFAULT_OUT_ROOT = Path("/data/weidong/InMOS_1deg/ckpt/CRPS")         # ALL CRPS stages, files prefixed 1-/2-/3-
DEFAULT_SURFACE_VARS = ["thetao_0", "so_0", "uo_0", "vo_0", "zos"]
BN_MODES = ("sample", "eval", "train")


def load_model_and_data(args, cfg, device):
    """Same signature/returns as samudra_om4_multi_ic_rollout.load_model_and_data.

    num_out comes from the checkpoint (1deg: 77 = one step); strict load.
    """
    prog_names = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
    bound_names = BOUNDARY_VARS[cfg.experiment.boundary_vars_key]
    n_prog, n_bound = len(prog_names), len(bound_names)
    num_in = (cfg.data.hist + 1) * (n_prog + n_bound)

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    state = state.get("model", state)
    state = OrderedDict((k.removeprefix("module."), v) for k, v in state.items())
    num_out = int(state["decoder.bias"].shape[0])

    src = DataSource.from_locations(
        data_location=LocalLocation(path=args.data_root / "OM4.zarr"),
        means_location=LocalLocation(path=args.data_root / "OM4_means.zarr"),
        stds_location=LocalLocation(path=args.data_root / "OM4_stds.zarr"),
        prognostic_var_names=prog_names,
        boundary_var_names=bound_names,
        static_data_vars=cfg.data.static_data_vars,
        use_dask=True,
    )
    TensorMap.init_instance(cfg.experiment.prognostic_vars_key,
                            cfg.experiment.boundary_vars_key).to(device)
    normalize = Normalize.init_instance(src, prognostic_var_names=prog_names,
                                        boundary_var_names=bound_names)
    model = cfg.model.build(
        in_channels=num_in, out_channels=num_out, hist=cfg.data.hist,
        static_data_for_corrector=None, srcs=[src],
    ).to(device)
    # Strict for plain checkpoints; a FiLM checkpoint (extra heads.* keys) is loaded
    # onto the backbone here and strictly again after make_stochastic by the caller.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or any(".heads." not in k for k in unexpected):
        raise RuntimeError(f"checkpoint/architecture mismatch: missing={missing[:5]} "
                           f"unexpected={[k for k in unexpected if '.heads.' not in k][:5]}")
    model.eval()
    logger.info("loaded %s: %.1fM params, out=%d ch (%d step/call), %d FiLM keys left for later",
                args.ckpt_path, sum(p.numel() for p in model.parameters()) / 1e6,
                num_out, num_out // n_prog, len(unexpected))
    return src, normalize, model, prog_names, bound_names, n_prog, num_out


def _target_time_one_step(self, start_step: int, num_steps: int):
    first = int(self.rolling_indices.isel(window_dim=start_step).values[self.hist + 1])
    return self._times.isel(time=slice(first, first + num_steps))


def InferenceDataset(src, **kwargs):  # noqa: N802 -- drop-in for the class at call sites
    """ocean_emulators InferenceDataset re-windowed for the one-step model.

    window i = time indices [i, i+1 | i+2], stride 1 (5 d); label mask 77 ch.
    Input (`_get_prognostic`/`_get_boundary`, slots [:hist+1]) and label
    (`_get_label`, slots [hist+1:]) slicing are untouched.
    """
    ds = _ZarrInferenceDataset(src=src, **kwargs)
    hist = ds.hist
    times = ds._times.values
    bad = [i for i in range(len(times) - 1) if (times[i + 1] - times[i]).days not in (5, 6)]
    if bad:
        raise ValueError(f"time gap {times[bad[0]]} -> {times[bad[0] + 1]} in the window")
    win = np.arange(len(times) - (hist + 1))[:, None] + np.arange(hist + 2)[None, :]
    ds.rolling_indices = xr.DataArray(win, dims=["window_dim", "time"]).astype(int)
    ds.size = len(ds.rolling_indices)
    ds.wet_label = ds.wet.clone()
    ds.ctx = GridContext(ds.wet_label, ds.input_res, ds.input_res)
    ds.get_target_time = types.MethodType(_target_time_one_step, ds)
    return ds


def set_bn_mode(model: nn.Module, mode: str) -> nn.Module:
    """"eval": BatchNorm layers stay in eval mode through every later model.train().

    Implemented by wrapping this instance's `train` (survives make_stochastic's class
    reassignment, which only swaps __class__). "train": no change.
    """
    if mode not in BN_MODES:
        raise ValueError(f"bn mode must be one of {BN_MODES}, got {mode!r}")
    n_bn = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if mode == "sample":
        # Per-sample statistics, in train AND eval: each member's field is normalized by its own
        # spatial mean/var, with the BN weight/bias. This is what the 1deg net saw in training
        # (BN train mode, 1 sample per GPU), and it keeps members independent when several
        # share a batch. Module type and state dict are unchanged; running stats are unused.
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.forward = types.MethodType(
                    lambda self, x: F.instance_norm(x, weight=self.weight, bias=self.bias, eps=self.eps), m)
    else:
        logger.warning("!" * 78 + "\n  BatchNorm mode %r is NOT how the 1deg Samudra was trained and NOT what the "
                       "inference scripts use.\n  Running stats (eval) make the base model blow up after ~3.5 yr. "
                       "Use --bn-mode sample unless this is a deliberate experiment.\n" + "!" * 78, mode)
    if mode == "eval":
        cls_train = type(model).train

        def train(self, flag: bool = True):
            cls_train(self, flag)
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
            return self

        model.train = types.MethodType(train, model)
        model.train(model.training)
    logger.info("BatchNorm mode: %s (%d BN layers)%s", mode, n_bn, {
        "sample": " -- per-sample statistics (train and eval), running stats unused, weight/bias trainable",
        "eval": " -- running stats frozen, weight/bias trainable",
        "train": " -- batch stats in train mode, running stats updated"}[mode])
    return model


# ---- Perlin (verbatim from samudra_om4_ensemble_rollout.py) -------------------------
def _perlin(shape, periods, rng):
    """2D Perlin noise on `shape`, tiled with `periods` lattice cells."""
    ny, nx = shape
    py, px = periods
    angles = rng.uniform(0, 2 * np.pi, (py + 1, px + 1)).astype(np.float32)
    angles[-1, :] = angles[0, :]
    angles[:, -1] = angles[:, 0]
    gx, gy = np.cos(angles), np.sin(angles)

    ys = np.linspace(0, py, ny, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, px, nx, endpoint=False, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)

    y0, x0 = yg.astype(int), xg.astype(int)
    y1, x1 = y0 + 1, x0 + 1
    fy, fx = yg - y0, xg - x0

    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)  # noqa: E731
    uy, ux = fade(fy), fade(fx)

    dot = lambda iy, ix: gx[iy, ix] * (xg - ix) + gy[iy, ix] * (yg - iy)  # noqa: E731
    n00, n10, n01, n11 = dot(y0, x0), dot(y1, x0), dot(y0, x1), dot(y1, x1)

    nx0 = n00 + uy * (n10 - n00)
    nx1 = n01 + uy * (n11 - n01)
    noise = nx0 + ux * (nx1 - nx0)
    noise -= noise.mean()
    s = noise.std()
    if s > 0:
        noise /= s
    return noise


def perlin_3octaves(shape, seeds, periods=(6, 12, 24), scales=(0.2, 0.1, 0.05)):
    """3-octave Perlin noise on `shape`, zero-mean unit-variance."""
    noise = np.zeros(shape, dtype=np.float32)
    for per, w, sd in zip(periods, scales, seeds):
        noise += w * _perlin(shape, (per, per), np.random.default_rng(sd))
    noise -= noise.mean()
    s = noise.std()
    if s > 0:
        noise /= s
    return noise
