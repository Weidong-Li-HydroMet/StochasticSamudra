# InMOS_1degree: 1° Samudra (inference and CRPS)

| item | path |
|---|---|
| data | `/data/weidong/InMOS_1deg/om4_onedeg_v3`: OM4.zarr (4745 pentads, 1958–2022, 180×360) plus the npz cache |
| checkpoint | `/data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt` (shipped without a config) |
| config | `checkpoints/SamudraV2_1deg/config.yaml`: reconstructed; the architecture is verified by a strict load |
| inference | `src/Inference/0-SamudraV2_multiICs.sh`, `0-SamudraV2_long.sh` (GPU 0) → `/data/weidong/InMOS_1deg/predictions/` |
| CRPS | `src/CRPS/` (see its README) → `/data/weidong/InMOS_1deg/ckpt/CRPS/` |
| data overview | `0-data_overview.ipynb` |

## The 1° checkpoint

- **Architecture:** SamudraV2 widths, BatchNorm, forcing = `tauuo, tauvo, hfds`, no `hfds_anomalies`.
- **One step = 5 days:** input `[x(t-1), x(t), f(t-1), f(t)]` (160 channels) → `x(t+1)` (77 channels).
  The local `ocean_emulators` assumes 154 output channels, so the scripts read the output width from the checkpoint.
- **BatchNorm must use per-sample statistics** (instance norm with the BN weight/bias; the module and state dict are unchanged).
  This is how the network was trained. Over an 8.2-yr rollout, SST RMSE goes 0.31 → 0.64 °C and stays stable;
  with the running stats the model blows up after about 3.5 years.

## Inference output

- One zarr per IC (multiICs) or per member (long). Variables `prediction` and `target`, dims `(time, var, lat, lon)`,
  all 77 channels, physical units, NaN on land.
- Frames 0–1 are the IC states. Frame k+2 is step k. Coordinates `lead_days` and `forecast_date`.
- Settings are overridable through environment variables (see the header of each `.sh`). A full-window member is about 24 GB.

## Data caveats

- The npz cache ends 2014-10-05, is missing 2003-04..2009-12, and clips hfds to ±8σ. OM4.zarr is complete.
