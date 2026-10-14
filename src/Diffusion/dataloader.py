"""
Data loading for the 1-degree conditional diffusion model.

Same sample as weidong/Diffusion/train_diffusion_025deg.py (OceanDiffDataset):
    state    (77, 180, 360)  uo, vo, thetao, so x 19 levels + zos; (x - mean) / std, land = 0
    forcing  ( 3, 180, 360)  tauuo, tauvo, hfds;                  (x - mean) / std, land = 0, hfds clipped to [-8, 8]
    datefrac ()              day-of-year / 365

The 0.25-degree code cached one .npz per date. At 1 degree the whole record is already
normalized in one array by src/SI/0-prepare_data.py, so this reads that array directly
(np.load with mmap_mode: no copy, only the requested time steps are read from disk):
    /data/weidong/InMOS_1deg/SI_data/states.npy     (4745, 77, 180, 360) float32
    /data/weidong/InMOS_1deg/SI_data/forcing.npy    (4745,  3, 180, 360) float32
    /data/weidong/InMOS_1deg/SI_data/prog_mask.npy  (77, 180, 360)
    /data/weidong/InMOS_1deg/SI_data/times.npz      year, month, day, doy (5-day OM4 pentads, 1958-01-03 .. 2022-12-29)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = "/data/weidong/InMOS_1deg/SI_data"

# Prognostic variables (state): uo, vo, thetao, so x 19 depths + zos = 77 (channel order of states.npy)
DEPTH_I_LEVELS = [str(i) for i in range(19)]
PROGNOSTIC_VARS = [
    f"{v}_{d}" for v in ["uo", "vo", "thetao", "so"] for d in DEPTH_I_LEVELS
] + ["zos"]
N_PROG = len(PROGNOSTIC_VARS)    # 77
# Boundary variables (forcing / condition): same as ocean_emulators tau_hfds
BOUNDARY_VARS = ["tauuo", "tauvo", "hfds"]
N_BOUND = len(BOUNDARY_VARS)     # 3

H = 180
W = 360

# Same split as the 0.25-degree code / ocean_emulators samudra_om4 config (inclusive dates)
TRAIN_START = "1958-01-03"
TRAIN_END = "2013-10-05"
VAL_START = "2013-10-10"         # first pentad after TRAIN_END (the 0.25 code shared 2013-10-05 between train and val)
VAL_END = "2022-12-29"           # end of the 1-degree record


def date_indices(start_date, end_date):
    """Time indices whose date lies in [start_date, end_date] (strings 'YYYY-MM-DD')."""
    t = np.load(os.path.join(DATA_DIR, "times.npz"))
    key = t["year"] * 10000 + t["month"] * 100 + t["day"]
    lo = int(start_date.replace("-", ""))
    hi = int(end_date.replace("-", ""))
    return np.nonzero((key >= lo) & (key <= hi))[0]


class OceanDiffDataset(Dataset):
    """Returns (state [N_PROG, H, W], forcing [N_BOUND, H, W], date_frac)."""

    def __init__(self, start_date, end_date):
        self.indices = date_indices(start_date, end_date)
        t = np.load(os.path.join(DATA_DIR, "times.npz"))
        self.datefrac = (t["doy"] / 365.0).astype(np.float32)
        self.states = None       # memmaps are opened lazily, once per DataLoader worker
        self.forcing = None
        print(f"OceanDiffDataset: {len(self.indices)} dates, {start_date} .. {end_date}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.states is None:
            self.states = np.load(os.path.join(DATA_DIR, "states.npy"), mmap_mode="r")
            self.forcing = np.load(os.path.join(DATA_DIR, "forcing.npy"), mmap_mode="r")
        i = self.indices[idx]
        state = np.array(self.states[i], dtype=np.float32)
        forcing = np.array(self.forcing[i], dtype=np.float32)
        forcing[2] = np.clip(forcing[2], -8, 8)     # hfds, as in the 0.25-degree cache
        return (torch.from_numpy(state),
                torch.from_numpy(forcing),
                torch.tensor(self.datefrac[i], dtype=torch.float32))


def load_prog_mask():
    """Ocean mask per state channel, (N_PROG, H, W) float32."""
    return np.load(os.path.join(DATA_DIR, "prog_mask.npy")).astype(np.float32)
