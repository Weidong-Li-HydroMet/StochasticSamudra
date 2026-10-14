"""Long-rollout stability: base 1deg Samudra vs the stage-1 CRPS checkpoint.

Reads the 0-SamudraV2_long.py stores under <root>/{base,crps}/ (members m00 control,
m01 Perlin), and for each model/member computes, per 5-day frame:
  * area-weighted global means of prediction and target (SST, SSS, SSH, deep T, |u| surf);
  * area-weighted RMSE vs target (SST, SSS, SSH);
  * NaN count on wet points and max |value| (blow-up check);
plus the m00-m01 member difference (spread proxy). Writes a PNG and a markdown summary.

    python 1-check_stability.py --root /data/weidong/predictions/samudra_1deg/stability \
        --out-dir /data/weidong/InMOS_1deg/ckpt/SamudraV2_1deg_CRPS
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("/data/weidong/predictions/samudra_1deg/stability"))
p.add_argument("--out-dir", type=Path, default=Path("/data/weidong/InMOS_1deg/ckpt/SamudraV2_1deg_CRPS"))
args = p.parse_args()

VARS = {"thetao_0": "SST (C)", "so_0": "SSS (psu)", "zos": "SSH (m)", "thetao_18": "T level 18 (C)"}
RMSE_VARS = ["thetao_0", "so_0", "zos"]

stats = {}
for model in ("base", "crps"):
    for f in sorted(glob.glob(str(args.root / model / "*_m0*.zarr"))):
        m = f.split("_m")[-1][:2]
        ds = xr.open_zarr(f)
        w = np.cos(np.deg2rad(ds.lat.values))[:, None]
        sel = list(VARS) + ["uo_0", "vo_0"]
        P = ds.prediction.sel(var=sel).values                       # (T, V, Y, X)
        T = ds.target.sel(var=sel).values
        wet = np.isfinite(T[0])                                      # (V, Y, X)
        ww = np.where(wet, w, 0.0)
        gm = lambda A, i: np.nansum(A[:, i] * ww[i], axis=(1, 2)) / ww[i].sum()
        rec = {"lead": ds.lead_days.values, "date": ds.forecast_date.values,
               "pred": P, "tgt": T, "ww": ww}
        for i, v in enumerate(VARS):
            rec[f"gm_pred_{v}"], rec[f"gm_tgt_{v}"] = gm(P, i), gm(T, i)
        for v in RMSE_VARS:
            i = sel.index(v)
            rec[f"rmse_{v}"] = np.sqrt(np.nansum((P[:, i] - T[:, i]) ** 2 * ww[i], axis=(1, 2)) / ww[i].sum())
        spd_p = np.sqrt(P[:, 4] ** 2 + P[:, 5] ** 2); spd_t = np.sqrt(T[:, 4] ** 2 + T[:, 5] ** 2)
        rec["gm_pred_speed"] = np.nansum(spd_p * ww[4], axis=(1, 2)) / ww[4].sum()
        rec["gm_tgt_speed"] = np.nansum(spd_t * ww[4], axis=(1, 2)) / ww[4].sum()
        rec["nan_wet"] = np.array([int(np.isnan(P[t][wet]).sum()) for t in range(len(P))])
        rec["maxabs_sst"] = np.nanmax(np.abs(P[:, 0]), axis=(1, 2))
        stats[(model, m)] = rec
        print(f"read {model} m{m}: {P.shape[0]} frames")

# member spread (m00 vs m01) for SST
for model in ("base", "crps"):
    a, b = stats[(model, "00")], stats[(model, "01")]
    ww = a["ww"][0]
    stats[(model, "spread")] = np.sqrt(np.nansum((a["pred"][:, 0] - b["pred"][:, 0]) ** 2 * ww, axis=(1, 2)) / ww.sum())

# ---------------------------------------------------------------- figure ----------
years = stats[("base", "00")]["lead"] / 365.25
fig, axs = plt.subplots(3, 3, figsize=(17, 11))
col = {"base": "tab:blue", "crps": "tab:red"}
panels = [(f"gm_{{}}_{v}", lab) for v, lab in VARS.items()] + [("gm_{}_speed", "|u| surface (m/s)")]
for ax, (key, lab) in zip(axs.ravel()[:5], panels):
    ax.plot(years, stats[("base", "00")][key.format("tgt")], "k", lw=1.2, label="OM4 truth")
    for model in ("base", "crps"):
        for m, ls in (("00", "-"), ("01", ":")):
            ax.plot(years, stats[(model, m)][key.format("pred")], color=col[model], ls=ls, lw=1,
                    label=f"{model} m{m}")
    ax.set_title(f"global mean {lab}"); ax.set_xlabel("lead (years)")
axs[0, 0].legend(fontsize=8)
for ax, v, lab in zip(axs.ravel()[5:8], RMSE_VARS, ["SST RMSE (C)", "SSS RMSE (psu)", "SSH RMSE (m)"]):
    for model in ("base", "crps"):
        ax.plot(years, stats[(model, "00")][f"rmse_{v}"], color=col[model], lw=1, label=f"{model} m00")
    ax.set_title(lab); ax.set_xlabel("lead (years)")
axs[1, 2].legend(fontsize=8)
ax = axs[2, 2]
for model in ("base", "crps"):
    ax.plot(years, stats[(model, "spread")], color=col[model], lw=1, label=model)
ax.set_title("SST member difference |m00 - m01| RMS (C)"); ax.set_xlabel("lead (years)"); ax.legend(fontsize=8)
plt.tight_layout()
png = args.out_dir / "stability_long_rollout.png"
fig.savefig(png, dpi=110)
print("wrote", png)

# ---------------------------------------------------------------- summary ---------
def at(rec_key, model, m, yrs):
    r = stats[(model, m)]; i = int(np.argmin(np.abs(years - yrs)))
    return r[rec_key][i]

lines = ["| model | member | NaN on wet pts | max |SST| (C) | SST RMSE 1/2/4/8 yr (C) | SSS RMSE 1/2/4/8 yr | SSH RMSE 1/2/4/8 yr | global-mean SST bias at 8 yr (C) |",
         "|---|---|---|---|---|---|---|---|"]
for model in ("base", "crps"):
    for m in ("00", "01"):
        r = stats[(model, m)]
        f3 = lambda k: " / ".join(f"{at(k, model, m, y):.3f}" for y in (1, 2, 4, 8))
        bias = r["gm_pred_thetao_0"][-1] - r["gm_tgt_thetao_0"][-1]
        lines.append(f"| {model} | m{m} | {int(r['nan_wet'].sum())} | {np.nanmax(r['maxabs_sst']):.2f} | "
                     f"{f3('rmse_thetao_0')} | {f3('rmse_so_0')} | {f3('rmse_zos')} | {bias:+.3f} |")
drift = ["", "| model (m00) | global-mean bias, mean over last year: SST (C) / SSS (psu) / SSH (m) / T18 (C) / |u| (m/s) |", "|---|---|"]
for model in ("base", "crps"):
    r = stats[(model, "00")]; last = years > years[-1] - 1
    b = [np.mean(r[f"gm_pred_{v}"][last] - r[f"gm_tgt_{v}"][last]) for v in VARS] + \
        [np.mean(r["gm_pred_speed"][last] - r["gm_tgt_speed"][last])]
    drift.append(f"| {model} | " + " / ".join(f"{x:+.4f}" for x in b) + " |")
sp = ["", "| model | SST member difference RMS at 0.25 / 1 / 4 / 8 yr (C) |", "|---|---|"]
for model in ("base", "crps"):
    s = stats[(model, "spread")]
    sp.append(f"| {model} | " + " / ".join(f"{s[int(np.argmin(np.abs(years - y)))]:.3f}" for y in (0.25, 1, 4, 8)) + " |")
txt = "\n".join(lines + drift + sp)
(args.out_dir / "stability_summary.md").write_text(txt + "\n")
print(txt)
