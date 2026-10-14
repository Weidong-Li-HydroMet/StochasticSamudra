"""Shared pieces of the 1-degree inference scripts (1-, 2-, 3-, 4-*.py).

Data, Perlin IC noise, the SamudraV2 1deg loader and the output zarr, all as in
0-SamudraV2_long.py (which stays self-contained), except that the data comes from the
prepared memmaps in /data/weidong/InMOS_1deg/SI_data (src/SI/0-prepare_data.py):
the same normalization as 0-SamudraV2_long.read_frame on OM4.zarr ((x - mean) / std
with OM4_means/stds, land = 0, raw hfds), but one fast array read per frame.

Output layout (every script): one zarr per member, 5-day frames, all 77 channels,
physical units, NaN on land, variables `prediction` and `target` on one time axis.
"""
import copy
import sys
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numcodecs
import numpy as np
import torch
import xarray as xr
import yaml
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]                     # .../InMOS_1degree
SRC = REPO_ROOT / "src"
SI_DATA = Path("/data/weidong/InMOS_1deg/SI_data")
OM4_ROOT = Path("/data/weidong/InMOS_1deg/om4_onedeg_v3")
SAMUDRA_CONFIG = REPO_ROOT / "checkpoints/SamudraV2_1deg/config.yaml"
SAMUDRA_CKPT = Path("/data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt")
PRED_ROOT = Path("/data/weidong/InMOS_1deg/predictions")

PROG = [f"{v}_{k}" for v in ("uo", "vo", "thetao", "so") for k in range(19)] + ["zos"]
FORC = ["tauuo", "tauvo", "hfds"]
NP, NF = len(PROG), len(FORC)                                       # 77, 3
SST = PROG.index("thetao_0")


# ---------------------------------------------------------------- data -----------
class Data:
    """Normalized states / forcing (land = 0), dates, masks and physical-unit stats."""

    def __init__(self):
        self.states = np.load(SI_DATA / "states.npy", mmap_mode="r")      # (T, 77, Y, X)
        self.forcing = np.load(SI_DATA / "forcing.npy", mmap_mode="r")    # (T, 3, Y, X)
        t = np.load(SI_DATA / "times.npz")
        self.dates = np.array([f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(t["year"], t["month"], t["day"])])
        self.doy = t["doy"].astype(np.float32)
        self.wet = np.load(SI_DATA / "prog_mask.npy") > 0                 # (77, Y, X)
        self.lat = np.load(SI_DATA / "lat.npy")
        self.lon = xr.open_zarr(OM4_ROOT / "OM4.zarr")["x"].values
        means = xr.open_zarr(OM4_ROOT / "OM4_means.zarr").load()
        stds = xr.open_zarr(OM4_ROOT / "OM4_stds.zarr").load()
        self.mu = np.array([float(means[v]) for v in PROG], np.float32)[:, None, None]
        self.sd = np.array([float(stds[v]) for v in PROG], np.float32)[:, None, None]

    def index(self, date):
        """Time index of 'YYYY-MM-DD' (must be a data date)."""
        i = np.searchsorted(self.dates, date)
        if i >= len(self.dates) or self.dates[i] != date:
            sys.exit(f"{date} is not a data date (5-day pentads, e.g. 2014-10-10)")
        return int(i)

    def state(self, i):
        return np.array(self.states[i], dtype=np.float32)

    def forc(self, i):
        return np.array(self.forcing[i], dtype=np.float32)

    def phys(self, a):
        """Normalized (..., 77, Y, X) -> physical, NaN on land."""
        return np.where(self.wet, a * self.sd + self.mu, np.nan).astype(np.float32)


# ---------------------------------------------------------------- Perlin ---------
def perlin(shape, periods, rng):
    """2D Perlin noise, `periods` lattice cells over the domain, zero-mean unit-variance."""
    ny, nx = shape
    angles = rng.uniform(0, 2 * np.pi, (periods + 1, periods + 1)).astype(np.float32)
    angles[-1, :], angles[:, -1] = angles[0, :], angles[:, 0]
    gx, gy = np.cos(angles), np.sin(angles)
    xg, yg = np.meshgrid(np.linspace(0, periods, nx, endpoint=False, dtype=np.float32),
                         np.linspace(0, periods, ny, endpoint=False, dtype=np.float32))
    y0, x0 = yg.astype(int), xg.astype(int)
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)  # noqa: E731
    uy, ux = fade(yg - y0), fade(xg - x0)
    dot = lambda iy, ix: gx[iy, ix] * (xg - ix) + gy[iy, ix] * (yg - iy)  # noqa: E731
    nx0 = dot(y0, x0) + uy * (dot(y0 + 1, x0) - dot(y0, x0))
    nx1 = dot(y0, x0 + 1) + uy * (dot(y0 + 1, x0 + 1) - dot(y0, x0 + 1))
    n = nx0 + ux * (nx1 - nx0)
    n -= n.mean()
    return n / n.std() if n.std() > 0 else n


def perlin_ic_noise(shape, index, base_seed):
    """3-octave Perlin field for perturbation index `index` (periods 6/12/24, weights
    0.2/0.1/0.05, seeds base_seed + 3*index + {0,1,2}); unit variance. Same rule as
    0-SamudraV2_long.py and the CRPS training (samudra_1deg.perlin_3octaves)."""
    noise = np.zeros(shape, np.float32)
    for per, w, k in zip((6, 12, 24), (0.2, 0.1, 0.05), range(3)):
        noise += w * perlin(shape, per, np.random.default_rng(base_seed + index * 3 + k))
    return (noise - noise.mean()) / noise.std()


# ---------------------------------------------------------------- SamudraV2 ------
def build_samudra(ckpt_path, device, config_path=SAMUDRA_CONFIG):
    """SamudraV2 1deg (one step: [x(t-1), x(t), f(t-1), f(t)] -> x(t+1)), strict load.

    A checkpoint with a `film_config` (CRPS stage 2) gets its FiLM heads first
    (make_stochastic with that config), then the strict load. Returns (model, ctx, ckpt dict).
    BatchNorm runs with PER-SAMPLE statistics, the only stable mode for this net
    (see 0-SamudraV2_long.py and src/CRPS/README.md).
    """
    sys.path.insert(0, str(SRC))
    from ocean_emulators.config import SamudraConfig
    wet = np.load(SI_DATA / "prog_mask.npy") > 0
    mcfg = copy.deepcopy(yaml.safe_load(open(config_path))["model"])
    model = SamudraConfig(**mcfg).build(
        in_channels=2 * (NP + NF), out_channels=NP, hist=1, static_data_for_corrector=None,
        srcs=[SimpleNamespace(grid_size=wet.shape[1:], spherical_area_weights=None)])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "film_config" in ck:
        sys.path.insert(0, str(SRC / "CRPS"))
        from stochastic_samudra import make_stochastic
        model = make_stochastic(model, **ck["film_config"])
    model.load_state_dict(OrderedDict((k.removeprefix("module."), v) for k, v in ck["model"].items()), strict=True)
    model = model.to(device).eval()
    for bn in [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]:
        bn.forward = types.MethodType(
            lambda self, x: torch.nn.functional.instance_norm(x, weight=self.weight, bias=self.bias, eps=self.eps), bn)
    ctx = SimpleNamespace(label_mask=torch.from_numpy(wet).to(device), input_resolution_cpu=None)
    return model, ctx, ck


# ---------------------------------------------------------------- output zarr ----
COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=4, shuffle=numcodecs.Blosc.SHUFFLE)
# zarr v2 stores in both envs: ocean_env has zarr 2; wdenv (zarr 3, used by 4-SI_long) needs it explicitly,
# otherwise a numcodecs compressor raises "Expected a BytesBytesCodec".
ZARR_KW = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}


def create_zarr(path, data, frame_dates, attrs, chunk_frames=5):
    """Empty store: prediction/target (time, var, lat, lon), lead_days and forecast_date
    relative to frame 0."""
    n = len(frame_dates)
    shape = (n, NP, len(data.lat), len(data.lon))
    empty = da.zeros(shape, chunks=(chunk_frames,) + shape[1:], dtype=np.float32)
    lead = (frame_dates.astype("datetime64[D]") - frame_dates[0].astype("datetime64[D]")).astype(float)
    xr.Dataset(
        {"prediction": (("time", "var", "lat", "lon"), empty),
         "target": (("time", "var", "lat", "lon"), empty)},
        coords={"time": np.arange(n), "var": PROG, "lat": data.lat, "lon": data.lon,
                "lead_days": ("time", lead), "forecast_date": ("time", frame_dates.astype("U10"))},
        attrs=attrs,
    ).to_zarr(path, mode="w", compute=False,
              encoding={v: {"compressor": COMPRESSOR, "_FillValue": np.float32(np.nan)}
                        for v in ("prediction", "target")}, **ZARR_KW)


def write_frames(path, data, pred, tgt, start):
    """Write normalized frames (n, 77, Y, X) at time index `start`; returns the next index."""
    xr.Dataset({"prediction": (("time", "var", "lat", "lon"), data.phys(np.stack(pred))),
                "target": (("time", "var", "lat", "lon"), data.phys(np.stack(tgt)))}).to_zarr(
        path, region={"time": slice(start, start + len(pred))}, safe_chunks=False, **ZARR_KW)
    return start + len(pred)


def sst_error(data, y, truth):
    """SST bias and RMSE (deg C) over the ocean, unweighted (as 0-SamudraV2_long.py)."""
    e = np.where(data.wet[SST], (y[SST] - truth[SST]) * data.sd[SST], np.nan)
    return np.nanmean(e), np.sqrt(np.nanmean(e ** 2))
