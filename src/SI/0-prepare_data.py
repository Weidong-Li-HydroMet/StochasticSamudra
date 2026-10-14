"""One-time conversion of the 1deg OM4.zarr into the memmaps the SI training reads.

Run once (about 10-20 min, ~98 GB written):
    /data/weidong/wdenv/bin/python 0-prepare_data.py

Writes to $SI_DATA (default /data/weidong/InMOS_1deg/SI_data):

    states.npy    (T, 77, 180, 360) f32   Samudra-normalized FULL FIELD, land = 0
    forcing.npy   (T,  3, 180, 360) f32   [tauuo, tauvo, hfds], normalized, land = 0
    prog_mask.npy (77, 180, 360)    f32   per-channel wet mask: mask_k per level (mask_0 for zos)
    lat.npy       (180,)            f64   cell-centre latitudes (zarr `y`; not exactly uniform)
    times.npz     year, month, day, doy, jday  (Julian calendar; 5-day spacing, 6 d across leap days)
    stats.npz     s_inc (77,)  std of a_{t+1} - a_t (5-day increment) over TRAIN,
                               ocean-only, area-weighted per channel
                  s_inc_floored = max(s_inc, 0.05 * median(s_inc))
                  std_full (77,) area-weighted std of a over TRAIN (~1 by construction)
                  n_train  number of train time steps used

Normalization = what the 1deg Samudra / CRPS code uses (ocean_emulators Normalize,
normalize_before_mask=True, masked_fill_value=0): (x - mean) / std with the scalar
per-variable OM4_means / OM4_stds. No climatology is subtracted ("full field").
Forcing follows src/CRPS: boundary_vars_key tau_hfds, NO hfds_anomalies.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cftime
import numpy as np
import xarray as xr
import zarr

SRC = os.environ.get('OM4_1DEG', '/data/weidong/InMOS_1deg/om4_onedeg_v3')
OUT = os.environ.get('SI_DATA', '/data/weidong/InMOS_1deg/SI_data')
TRAIN_END = (2013, 10, 5)          # last train state (inclusive); val starts here too

PROG = [f'{v}_{k}' for v in ('uo', 'vo', 'thetao', 'so') for k in range(19)] + ['zos']
FORC = ['tauuo', 'tauvo', 'hfds']


def main():
    os.makedirs(OUT, exist_ok=True)
    root = zarr.open_group(os.path.join(SRC, 'OM4.zarr'), mode='r')
    means = xr.open_zarr(os.path.join(SRC, 'OM4_means.zarr'))
    stds = xr.open_zarr(os.path.join(SRC, 'OM4_stds.zarr'))

    # ---- time axis ---------------------------------------------------------
    ds = xr.open_zarr(os.path.join(SRC, 'OM4.zarr'))
    times = ds.time.values
    T = len(times)
    jday = np.array([cftime.date2num(t, 'days since 1958-01-01', calendar='julian')
                     for t in times])
    gaps = np.diff(jday)
    # 365-day model pentads stamped on a Julian axis: 6-day gaps at leap days only
    # (same rule as src/CRPS/samudra_1deg.py). Anything else is a hole in the data.
    assert np.all((gaps == 5) | (gaps == 6)), f'bad gaps at {np.nonzero((gaps != 5) & (gaps != 6))[0][:5]}'
    print(f'time gaps: {int(np.sum(gaps == 5))} x 5 d, {int(np.sum(gaps == 6))} x 6 d (leap days)')
    year = np.array([t.year for t in times]); month = np.array([t.month for t in times])
    day = np.array([t.day for t in times]); doy = np.array([t.dayofyr for t in times])
    np.savez(os.path.join(OUT, 'times.npz'), year=year, month=month, day=day,
             doy=doy, jday=jday)
    key = year * 10000 + month * 100 + day
    n_train = int(np.sum(key <= TRAIN_END[0] * 10000 + TRAIN_END[1] * 100 + TRAIN_END[2]))
    lat = np.asarray(ds.y.values, dtype=np.float64)
    np.save(os.path.join(OUT, 'lat.npy'), lat)
    H, W = len(lat), ds.sizes['x']
    print(f'T={T} ({times[0]} .. {times[-1]}), train steps {n_train}, grid {H}x{W}')

    # ---- masks -------------------------------------------------------------
    level_mask = {k: np.asarray(root[f'mask_{k}'][:], dtype=np.float32) for k in range(19)}
    mask = np.stack([level_mask[int(n.split('_')[1])] if n != 'zos' else level_mask[0]
                     for n in PROG]).astype(np.float32)
    cosw = np.cos(np.deg2rad(lat))[:, None]
    n_dropped = {}
    filled = {}

    states = np.lib.format.open_memmap(os.path.join(OUT, 'states.npy.tmp'), mode='w+',
                                       dtype=np.float32, shape=(T, len(PROG), H, W))
    forcing = np.lib.format.open_memmap(os.path.join(OUT, 'forcing.npy.tmp'), mode='w+',
                                        dtype=np.float32, shape=(T, len(FORC), H, W))

    s_inc = np.zeros(len(PROG)); std_full = np.zeros(len(PROG))

    def do_prog(c):
        name = PROG[c]
        a = np.asarray(root[name][:], dtype=np.float32)       # (T,H,W), NaN on land
        # Holes inside the wet mask: so_13 is NaN over the WHOLE mask at one step
        # (1999-08-01). Such steps are filled by linear interpolation between the
        # neighbouring steps; any cell still NaN after that is dropped from the
        # channel mask (none expected).
        m0 = mask[c] > 0
        bad_t = np.nonzero(np.isnan(a[:, m0]).any(axis=1))[0]
        for t in bad_t:
            if 0 < t < T - 1:
                hole = np.isnan(a[t])
                a[t][hole] = 0.5 * (a[t - 1][hole] + a[t + 1][hole])
        if len(bad_t):
            filled[name] = [str(times[t])[:10] for t in bad_t]
        m = m0 & np.isfinite(a).all(axis=0)
        n_dropped[name] = int(m0.sum() - m.sum())
        mask[c] = m
        a = (a - float(means[name])) / float(stds[name])
        a = np.where(m[None], a, 0.0).astype(np.float32)
        states[:, c] = a
        tr = a[:n_train].astype(np.float64)
        d = tr[1:] - tr[:-1]                                   # 5-day increments
        wc = m * cosw
        wc = wc / wc.sum()                                     # area weights, sum 1
        mu = (d * wc).sum(axis=(1, 2)).mean()
        s_inc[c] = np.sqrt(((d - mu) ** 2 * wc).sum(axis=(1, 2)).mean())
        muf = (tr * wc).sum(axis=(1, 2)).mean()
        std_full[c] = np.sqrt(((tr - muf) ** 2 * wc).sum(axis=(1, 2)).mean())
        return name

    def do_forc(c):
        name = FORC[c]
        a = np.asarray(root[name][:], dtype=np.float32)
        a = (a - float(means[name])) / float(stds[name])
        a = np.where(mask[0][None] > 0, np.nan_to_num(a, nan=0.0), 0.0).astype(np.float32)
        forcing[:, c] = a
        return name

    t0 = time.time()
    with ThreadPoolExecutor(12) as ex:
        for n in ex.map(do_forc, range(len(FORC))):
            print(f'  forcing {n} done ({time.time()-t0:.0f}s)', flush=True)
        for n in ex.map(do_prog, range(len(PROG))):
            print(f'  state {n} done ({time.time()-t0:.0f}s)', flush=True)
    np.save(os.path.join(OUT, 'prog_mask.npy'), mask)
    dropped = {k: v for k, v in n_dropped.items() if v}
    print(f'time steps with NaN inside the wet mask, filled by time interpolation: {filled or "none"}')
    print(f'wet-mask cells dropped (still NaN after filling): {dropped or "none"}')
    states.flush(); forcing.flush()
    del states, forcing
    os.replace(os.path.join(OUT, 'states.npy.tmp'), os.path.join(OUT, 'states.npy'))
    os.replace(os.path.join(OUT, 'forcing.npy.tmp'), os.path.join(OUT, 'forcing.npy'))

    floor = 0.05 * np.median(s_inc)
    s_inc_f = np.maximum(s_inc, floor)
    np.savez(os.path.join(OUT, 'stats.npz'), s_inc=s_inc, s_inc_floored=s_inc_f,
             std_full=std_full, n_train=n_train, names=np.array(PROG))
    print(f'\ns_inc floor {floor:.5f} (0.05 x median); per group (level 0 / 9 / 18):')
    for g, v in enumerate(('uo', 'vo', 'thetao', 'so')):
        sl = slice(19 * g, 19 * g + 19)
        print(f'  {v:7s} s_inc {s_inc[sl][0]:.4f} {s_inc[sl][9]:.4f} {s_inc[sl][18]:.4f}'
              f' | std_full {std_full[sl][0]:.3f} {std_full[sl][9]:.3f} {std_full[sl][18]:.3f}')
    print(f'  zos     s_inc {s_inc[76]:.4f} | std_full {std_full[76]:.3f}')
    print(f'done in {(time.time()-t0)/60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
