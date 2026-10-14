"""1-degree SamudraV2: single-IC long ensemble (Perlin IC perturbation), every state saved.

Same model / data / output layout as 0-SamudraV2_multiICs.py (see its docstring):
one-step model, 5 days per step, one zarr per MEMBER with all 77 channels,
frame 0-1 = the (perturbed) IC states, frame k+2 = prediction of step k.
`target` frames 0-1 hold the unperturbed truth.

Member 0 is the unperturbed control; member m >= 1 adds sigma * 3-octave Perlin noise
(seeds base_seed + 3m + {0,1,2}, broadcast over all channels of both IC states), the
same perturbation as the 0.25deg long_perturb runs. Default rollout = the whole window
(2014-10-10..2022-12-24 -> 598 steps = 2990 d).

    cd /home/nvidia/weidong/InMOS_1degree/src/Inference
    CUDA_VISIBLE_DEVICES=0 /data/weidong/ocean_env/bin/python 0-SamudraV2_long.py \
        --n-members 2 --rollout-len 20 --output-dir /tmp/smoke/long
"""
import argparse
import copy
import sys
import time
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

REPO_ROOT = Path(__file__).resolve().parents[2]                     # .../InMOS_1degree
sys.path.insert(0, str(REPO_ROOT / "src"))
from ocean_emulators.config import SamudraConfig  # noqa: E402

PROG = [f"{v}_{k}" for v in ("uo", "vo", "thetao", "so") for k in range(19)] + ["zos"]
FORC = ["tauuo", "tauvo", "hfds"]
NP, NF = len(PROG), len(FORC)                                       # 77, 3

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--config-path", type=Path, default=REPO_ROOT / "checkpoints/SamudraV2_1deg/config.yaml")
p.add_argument("--ckpt-path", type=Path, default=Path("/data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt"))
p.add_argument("--data-root", type=Path, default=Path("/data/weidong/InMOS_1deg/om4_onedeg_v3"))
p.add_argument("--data-source", choices=["zarr", "cache"], default="zarr")
p.add_argument("--inference-start", default="2014-10-10")
p.add_argument("--inference-end", default="2022-12-24")
p.add_argument("--rollout-len", type=int, default=None, help="steps (5 d each); default = whole window")
p.add_argument("--n-members", type=int, default=8, help="member 0 = unperturbed control")
p.add_argument("--perturbation-level", type=float, default=0.025)
p.add_argument("--base-seed", type=int, default=2025)
p.add_argument("--output-dir", type=Path, default=Path("/data/weidong/InMOS_1deg/predictions/SamudraV2_long"))
p.add_argument("--chunk-frames", type=int, default=5, help="frames per zarr chunk / RAM buffer")
p.add_argument("--log-every", type=int, default=50)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--rank", type=int, default=0, help="worker index; members are split [rank::world_size]")
p.add_argument("--world-size", type=int, default=1)
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.perf_counter()

# ---------------------------------------------------------------- data -----------
z = xr.open_zarr(args.data_root / "OM4.zarr")
lat, lon = z["y"].values, z["x"].values
level = [int(n.split("_")[1]) for n in PROG[:-1]] + [0]
wet = np.stack([z[f"mask_{k}"].values.astype(bool) for k in level])        # (77, Y, X)
wet0 = z["mask_0"].values.astype(bool)                                      # forcing mask
means = xr.open_zarr(args.data_root / "OM4_means.zarr").load()
stds = xr.open_zarr(args.data_root / "OM4_stds.zarr").load()
mu = np.array([float(means[v]) for v in PROG + FORC], np.float32)[:, None, None]
sd = np.array([float(stds[v]) for v in PROG + FORC], np.float32)[:, None, None]

if args.data_source == "zarr":
    dates = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in z.time.values])
else:
    files = {f.stem: f for split in ("train", "val") for f in (args.data_root / "cache" / split).glob("*.npz")}
    dates = np.array(sorted(files))
sel = np.where((dates >= args.inference_start) & (dates <= args.inference_end))[0]
dates = dates[sel]
gap_days = np.diff(dates.astype("datetime64[D]")).astype(int)
if not np.isin(gap_days, (5, 6)).all():          # 6 = leap-year Dec 28 -> Jan 3
    i = np.where(~np.isin(gap_days, (5, 6)))[0][0]
    sys.exit(f"time gap {dates[i]} -> {dates[i + 1]} inside the window; pick another window")


def read_frame(i):
    """Window index i -> normalized (state (77,Y,X), forcing (3,Y,X)) float32, land = 0."""
    if args.data_source == "cache":
        with np.load(files[dates[i]]) as f:
            return f["state"].astype(np.float32), f["forcing"].astype(np.float32)
    a = z[PROG + FORC].isel(time=int(sel[i])).to_array().values.astype(np.float32)
    a = np.nan_to_num((a - mu) / sd, nan=0.0)
    return np.where(wet, a[:NP], 0.0), np.where(wet0, a[NP:], 0.0)


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


# ---------------------------------------------------------------- model ----------
mcfg = copy.deepcopy(yaml.safe_load(open(args.config_path))["model"])
model = SamudraConfig(**mcfg).build(
    in_channels=2 * (NP + NF), out_channels=NP, hist=1, static_data_for_corrector=None,
    srcs=[SimpleNamespace(grid_size=wet0.shape, spherical_area_weights=None)]).to(device)
state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)["model"]
model.load_state_dict(OrderedDict((k.removeprefix("module."), v) for k, v in state_dict.items()), strict=True)
model.eval()
# BatchNorm with PER-SAMPLE statistics: the 1deg net was trained with BN in train mode at 1 sample
# per GPU, i.e. every field normalized by its own spatial mean/var. With the checkpoint running
# stats (plain eval mode) the rollout blows up after ~3.5 yr; per-sample it is stable for 8.2 yr
# (checked 2026-10-12). Module type and state dict are unchanged; running stats are not used.
for bn in [mod for mod in model.modules() if isinstance(mod, torch.nn.BatchNorm2d)]:
    bn.forward = types.MethodType(
        lambda self, x: torch.nn.functional.instance_norm(x, weight=self.weight, bias=self.bias, eps=self.eps), bn)
ctx = SimpleNamespace(label_mask=torch.from_numpy(wet).to(device), input_resolution_cpu=None)

L = args.rollout_len if args.rollout_len is not None else len(dates) - 2
if L + 2 > len(dates):
    sys.exit(f"rollout-len {L} exceeds the window ({len(dates) - 2} steps max)")
n_frames = L + 2
frame_dates = dates[:n_frames]
print(f"model loaded strictly: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M params, "
      f"data={args.data_source}, init {dates[0]}, {L} steps -> {frame_dates[-1]}", flush=True)

# truth for the whole rollout is shared by all members: read the IC once, the rest per step
s0, f0 = read_frame(0)
s1, f1 = read_frame(1)
args.output_dir.mkdir(parents=True, exist_ok=True)
compressor = numcodecs.Blosc(cname="zstd", clevel=4, shuffle=numcodecs.Blosc.SHUFFLE)

for m in list(range(args.n_members))[args.rank::args.world_size]:
    out = args.output_dir / (f"SamudraV2_1deg_long_init{dates[0]}_to{frame_dates[-1]}_steps{L}"
                             f"_sigma{args.perturbation_level}_m{m:02d}.zarr")
    if out.exists() and not args.overwrite:
        print(f"skip (exists) {out.name}", flush=True)
        continue
    t0 = time.perf_counter()
    empty = da.zeros((n_frames, NP, *wet0.shape), chunks=(args.chunk_frames, NP, *wet0.shape), dtype=np.float32)
    xr.Dataset(
        {"prediction": (("time", "var", "lat", "lon"), empty),
         "target": (("time", "var", "lat", "lon"), empty)},
        coords={"time": np.arange(n_frames), "var": PROG, "lat": lat, "lon": lon,
                "lead_days": ("time", (frame_dates.astype("datetime64[D]")
                                       - frame_dates[0].astype("datetime64[D]")).astype(float)),
                "forecast_date": ("time", frame_dates.astype("U10"))},
        attrs={"init_date": str(dates[0]), "final_date": str(frame_dates[-1]), "rollout_len": L,
               "step_days": 5, "n_ic_frames": 2,
               "frames": "0-1 = IC states (prediction: perturbed, target: truth); k+2 = prediction of step k",
               "member": m, "n_members": args.n_members, "perturbation_level": args.perturbation_level,
               "perturbation_kind": "perlin_3octaves" if m > 0 else "none", "base_seed": args.base_seed,
               "units": "physical, NaN on land", "checkpoint": str(args.ckpt_path),
               "config": str(args.config_path), "data_root": str(args.data_root),
               "data_source": args.data_source, "bn_stats": "per-sample", "model": "SamudraV2 1deg (one-step) + Perlin IC"},
    ).to_zarr(out, mode="w", compute=False,
              encoding={v: {"compressor": compressor, "_FillValue": np.float32(np.nan)}
                        for v in ("prediction", "target")})

    # IC (+ Perlin perturbation for m >= 1; periods 6/12/24, weights 0.2/0.1/0.05)
    state = torch.from_numpy(np.concatenate([s0, s1]))[None].to(device)        # (1, 154, Y, X)
    if m > 0:
        noise = np.zeros(wet0.shape, np.float32)
        for per, w, k in zip((6, 12, 24), (0.2, 0.1, 0.05), range(3)):
            noise += w * perlin(wet0.shape, per, np.random.default_rng(args.base_seed + m * 3 + k))
        noise = (noise - noise.mean()) / noise.std()
        state = state + args.perturbation_level * torch.from_numpy(noise).to(device)
    ic_np = state[0].cpu().numpy()
    buf_pred, buf_tgt = [ic_np[:NP], ic_np[NP:]], [s0, s1]
    written = 0
    with torch.no_grad():
        for k in range(L):
            f_prev, f_now = (f0, f1) if k == 0 else (f_now, read_frame(k + 1)[1])
            x = torch.cat([state, torch.from_numpy(np.concatenate([f_prev, f_now]))[None].to(device)], 1)
            y = model.forward_once(x, ctx)                                   # (1, 77, Y, X)
            state = torch.cat([state[:, NP:], y], 1)                         # drop x(t-1), append x(t+1)
            y_np, truth = y[0].float().cpu().numpy(), read_frame(k + 2)[0]
            buf_pred.append(y_np)
            buf_tgt.append(truth)

            if len(buf_pred) >= args.chunk_frames or k == L - 1:
                pred = np.where(wet, np.stack(buf_pred) * sd[:NP] + mu[:NP], np.nan).astype(np.float32)
                tgt = np.where(wet, np.stack(buf_tgt) * sd[:NP] + mu[:NP], np.nan).astype(np.float32)
                xr.Dataset({"prediction": (("time", "var", "lat", "lon"), pred),
                            "target": (("time", "var", "lat", "lon"), tgt)}).to_zarr(
                    out, region={"time": slice(written, written + len(pred))}, safe_chunks=False)
                written += len(pred)
                buf_pred, buf_tgt = [], []

            if (k + 1) % args.log_every == 0 or k == L - 1:
                i = PROG.index("thetao_0")
                e = np.where(wet[i], (y_np[i] - truth[i]) * sd[i], np.nan)       # degC
                print(f"m={m:02d} step {k + 1}/{L} valid={frame_dates[k + 2]}  SST bias={np.nanmean(e):+.2f}C "
                      f"rmse={np.sqrt(np.nanmean(e ** 2)):.2f}C  {(time.perf_counter() - t0) / (k + 1):.2f}s/step",
                      flush=True)

    print(f"member {m:02d} done in {time.perf_counter() - t0:.1f}s -> {out.name}", flush=True)

print(f"done in {time.perf_counter() - t_start:.1f}s -> {args.output_dir}")
