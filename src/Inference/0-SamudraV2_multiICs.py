"""1-degree SamudraV2: deterministic multi-IC rollouts, every state saved.

Model: /data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt, a ONE-STEP model
    input  [x(t-1), x(t), f(t-1), f(t)]   (2*77 prognostic + 2*3 forcing = 160 ch)
    output x(t+1)                          (77 ch, 5 days ahead)
x = uo_0..18, vo_0..18, thetao_0..18, so_0..18, zos (normalized, land = 0)
f = tauuo, tauvo, hfds                  (normalized, land = 0)

One zarr per IC, frames at a 5-day cadence, ALL 77 channels, physical units, NaN on land:
    frame 0, 1        : the IC states x(t0), x(t0+1)        (lead 0, 5 d)
    frame k+2         : prediction of step k, k=0..L-1       (lead 10, 15, ... d)
`prediction` and `target` share that axis (for frames 0-1 both hold the truth).

Data (--data-source):
    zarr  : <data-root>/OM4.zarr normalized here with OM4_means/stds (raw hfds)
    cache : <data-root>/cache/{train,val}/*.npz, already normalized (hfds clipped to +-8;
            ends 2014-10-05; train lacks 2003-04-13..2009-12-29)

    cd /home/nvidia/weidong/InMOS_1degree/src/Inference
    CUDA_VISIBLE_DEVICES=0 /data/weidong/ocean_env/bin/python 0-SamudraV2_multiICs.py \
        --max-ics 2 --rollout-len 6 --output-dir /tmp/smoke/multi
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
p.add_argument("--ic-stride", type=int, default=6, help="steps between ICs (1 step = 5 d)")
p.add_argument("--rollout-len", type=int, default=20, help="autoregressive steps per IC (5 d each)")
p.add_argument("--max-ics", type=int, default=None)
p.add_argument("--output-dir", type=Path, default=Path("/data/weidong/InMOS_1deg/predictions/SamudraV2_multiICs"))
p.add_argument("--chunk-frames", type=int, default=5, help="frames per zarr chunk / RAM buffer")
p.add_argument("--overwrite", action="store_true")
p.add_argument("--rank", type=int, default=0, help="worker index; ICs are split [rank::world_size]")
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
print(f"model loaded strictly: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M params, "
      f"data={args.data_source}, window {dates[0]}..{dates[-1]} ({len(dates)} frames)", flush=True)

# ---------------------------------------------------------------- ICs ------------
L = args.rollout_len
ic_list = list(range(0, len(dates) - (L + 2) + 1, args.ic_stride))[: args.max_ics]
if not ic_list:
    sys.exit(f"rollout-len {L} does not fit in {len(dates)} frames")
print(f"{len(ic_list)} ICs ({dates[ic_list[0]]} .. {dates[ic_list[-1]]}), "
      f"this rank: {len(ic_list[args.rank::args.world_size])}", flush=True)
args.output_dir.mkdir(parents=True, exist_ok=True)
compressor = numcodecs.Blosc(cname="zstd", clevel=4, shuffle=numcodecs.Blosc.SHUFFLE)
n_frames = L + 2

for ic in ic_list[args.rank::args.world_size]:
    out = args.output_dir / f"SamudraV2_1deg_init{dates[ic]}_steps{L}.zarr"
    if out.exists() and not args.overwrite:
        print(f"skip (exists) {out.name}", flush=True)
        continue
    t0 = time.perf_counter()
    frame_dates = dates[ic: ic + n_frames]
    empty = da.zeros((n_frames, NP, *wet0.shape), chunks=(args.chunk_frames, NP, *wet0.shape), dtype=np.float32)
    xr.Dataset(
        {"prediction": (("time", "var", "lat", "lon"), empty),
         "target": (("time", "var", "lat", "lon"), empty)},
        coords={"time": np.arange(n_frames), "var": PROG, "lat": lat, "lon": lon,
                "lead_days": ("time", (frame_dates.astype("datetime64[D]")
                                       - frame_dates[0].astype("datetime64[D]")).astype(float)),
                "forecast_date": ("time", frame_dates.astype("U10"))},
        attrs={"init_date": str(dates[ic]), "rollout_len": L, "step_days": 5, "n_ic_frames": 2,
               "frames": "0-1 = IC states (truth); k+2 = prediction of step k",
               "units": "physical, NaN on land", "checkpoint": str(args.ckpt_path),
               "config": str(args.config_path), "data_root": str(args.data_root),
               "data_source": args.data_source, "bn_stats": "per-sample", "model": "SamudraV2 1deg (one-step)"},
    ).to_zarr(out, mode="w", compute=False,
              encoding={v: {"compressor": compressor, "_FillValue": np.float32(np.nan)}
                        for v in ("prediction", "target")})

    # IC: the two input states and the first input forcing
    s0, f0 = read_frame(ic)
    s1, f1 = read_frame(ic + 1)
    state = torch.from_numpy(np.concatenate([s0, s1]))[None].to(device)        # (1, 154, Y, X)
    buf_pred, buf_tgt = [s0, s1], [s0, s1]
    written = 0
    with torch.no_grad():
        for k in range(L):
            # forcing at the two input times t0+k, t0+k+1; truth at t0+k+2
            f_prev, f_now = (f0, f1) if k == 0 else (f_now, read_frame(ic + k + 1)[1])
            x = torch.cat([state, torch.from_numpy(np.concatenate([f_prev, f_now]))[None].to(device)], 1)
            y = model.forward_once(x, ctx)                                   # (1, 77, Y, X)
            state = torch.cat([state[:, NP:], y], 1)                         # drop x(t-1), append x(t+1)
            buf_pred.append(y[0].float().cpu().numpy())
            buf_tgt.append(read_frame(ic + k + 2)[0])

            if len(buf_pred) >= args.chunk_frames or k == L - 1:
                pred = np.where(wet, np.stack(buf_pred) * sd[:NP] + mu[:NP], np.nan).astype(np.float32)
                tgt = np.where(wet, np.stack(buf_tgt) * sd[:NP] + mu[:NP], np.nan).astype(np.float32)
                xr.Dataset({"prediction": (("time", "var", "lat", "lon"), pred),
                            "target": (("time", "var", "lat", "lon"), tgt)}).to_zarr(
                    out, region={"time": slice(written, written + len(pred))}, safe_chunks=False)
                written += len(pred)
                buf_pred, buf_tgt = [], []

    sst = (pred[-1, PROG.index("thetao_0")] - tgt[-1, PROG.index("thetao_0")])
    print(f"IC {dates[ic]}  {L} steps  {time.perf_counter() - t0:.1f}s  "
          f"last-step SST bias={np.nanmean(sst):+.2f}C rmse={np.sqrt(np.nanmean(sst ** 2)):.2f}C  -> {out.name}",
          flush=True)

print(f"done in {time.perf_counter() - t_start:.1f}s -> {args.output_dir}")
