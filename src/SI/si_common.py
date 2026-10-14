"""Shared pieces of the 1deg SI forecaster (1-SI_step1.py and 2-SI_rollout.py).

Formulation (2026-10-13, user decisions; fork of Samudra-SI):

    state      a_t = Samudra-normalized FULL FIELD (77 ch), land = 0
    SI bridge  source x = a_t  ->  target y = a_{t+1}       (ONE 5-day step, 77 ch)
    cond       [a_t, da_t],  da_t = (a_t - a_{t-1}) / s_inc  (154 ch)
    forcing    [tauuo, tauvo, hfds] at t-1 and t           (6 ch, as src/CRPS; no hfds_anom)
    scalar     [0, doy(t)]                                 (calendar embedding)
    backbone   UViT, NO carry: y_hat = decoder output (full field, no residual on x)

Noise and loss scale: the interpolant still runs from a_t to a_{t+1}, so the
noise it injects must be on the scale of the 5-day increment, not of the field
(a unit-std noise would bury a deep-tracer increment of ~0.01). Hence, exactly
as in Samudra-SI: noise_scales = s_inc_floored, loss weight = cos(lat) * mask /
s_inc^2 (LOSS_W_MODE='inc', persistence scores ~1). LOSS_W_MODE='full' drops
the 1/s_inc^2 factor (plain area-weighted MSE of the normalized field).

s_delta (5-day difference std) and s_inc (one-step increment std) are the same
statistic now that one call = one 5-day step, so da_t is divided by s_inc.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SI_DATA = os.environ.get('SI_DATA', '/data/weidong/InMOS_1deg/SI_data')
OM4_1DEG = os.environ.get('OM4_1DEG', '/data/weidong/InMOS_1deg/om4_onedeg_v3')

H, W = 180, 360
N_STATE = 77
N_FORC = 3                       # tauuo, tauvo, hfds
C_GRID = 2 * N_FORC              # at t-1 and t
PROGNOSTIC_VARS = [f'{v}_{k}' for v in ('uo', 'vo', 'thetao', 'so') for k in range(19)] + ['zos']
SST_CHANNEL = PROGNOSTIC_VARS.index('thetao_0')
VAL_START = (2013, 10, 5)        # first val state (= last train state, as Samudra-SI)
VAL_END = (2022, 12, 29)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class SIData:
    """Memmaps + calendar + statistics from 0-prepare_data.py."""

    def __init__(self, root=SI_DATA):
        need = ['states.npy', 'forcing.npy', 'prog_mask.npy', 'lat.npy', 'times.npz', 'stats.npz']
        miss = [n for n in need if not os.path.exists(os.path.join(root, n))]
        if miss:
            raise SystemExit(f'missing {miss} under {root}: run 0-prepare_data.py first')
        self.root = root
        self.states = np.load(os.path.join(root, 'states.npy'), mmap_mode='r')
        self.forcing = np.load(os.path.join(root, 'forcing.npy'), mmap_mode='r')
        t = np.load(os.path.join(root, 'times.npz'))
        self.doy = t['doy'].astype(np.float32)
        self.key = t['year'] * 10000 + t['month'] * 100 + t['day']
        st = np.load(os.path.join(root, 'stats.npz'))
        self.s_inc = st['s_inc'].astype(np.float32)
        self.s_inc_floored = st['s_inc_floored'].astype(np.float32)
        self.std_full = st['std_full'].astype(np.float32)
        self.T = self.states.shape[0]
        k0 = VAL_START[0] * 10000 + VAL_START[1] * 100 + VAL_START[2]
        k1 = VAL_END[0] * 10000 + VAL_END[1] * 100 + VAL_END[2]
        # time-index ranges [lo, hi] (inclusive) of each split
        self.train_range = (0, int(np.searchsorted(self.key, k0, side='right')) - 1)
        self.val_range = (int(np.searchsorted(self.key, k0)), int(np.searchsorted(self.key, k1, side='right')) - 1)

    def date(self, i):
        k = int(self.key[i])
        return f'{k // 10000:04d}-{k // 100 % 100:02d}-{k % 100:02d}'

    def thetao0_std(self):
        """Physical std of thetao_0 (deg C per normalized unit) for SST RMSE in deg C."""
        import xarray as xr
        return float(xr.open_zarr(os.path.join(OM4_1DEG, 'OM4_stds.zarr'))['thetao_0'])


class SIWindowDataset(Dataset):
    """Windows for K consecutive model calls starting at input time t.

    Returns
        x_prev   (77,H,W)    a_{t-1}
        x_cur    (77,H,W)    a_t
        y        (K,77,H,W)  a_{t+1} .. a_{t+K}
        forcing  (K,6,H,W)   call k: [f_{t-1+k}, f_{t+k}]
        scalar   (K,2)       call k: [0, doy(t+k)]
    The whole window lies inside one split's [lo, hi] range.
    """

    def __init__(self, data, split, K=1, limit=None):
        self.d = data
        self.K = K
        lo, hi = data.train_range if split == 'train' else data.val_range
        self.ts = np.arange(lo + 1, hi - K + 1)
        if limit:
            self.ts = self.ts[:limit]

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, i):
        t, K = int(self.ts[i]), self.K
        s = np.array(self.d.states[t - 1:t + K + 1])          # (K+2,77,H,W)
        f = np.array(self.d.forcing[t - 1:t + K])             # (K+1,3,H,W)
        forcing = np.concatenate([f[:-1], f[1:]], axis=1)     # (K,6,H,W)
        scalar = np.stack([np.zeros(K, np.float32), self.d.doy[t:t + K]], axis=1)
        return (torch.from_numpy(s[0]), torch.from_numpy(s[1]), torch.from_numpy(s[2:]),
                torch.from_numpy(forcing), torch.from_numpy(scalar))


# ---------------------------------------------------------------------------
# Model call, SI loss, inference operator
# ---------------------------------------------------------------------------
def bind_call(model, c_scalar, x_prev, cell_mask, inv_s_da, amp_dtype=None, ckpt=False):
    """Model call with the signature DynamicInterpolant uses: f(X_t, x, t, c_grid).

    The interpolant passes the raw source x = a_t as `cond`; it is rewritten HERE
    into the cond view [a_t, (a_t - a_{t-1}) / s_inc], with a_{t-1} = x_prev bound
    per call. One code path for training, validation and every sampler.
    Output is fp32 and land-zeroed. ckpt=True wraps the whole evaluation in a
    non-reentrant activation checkpoint (used by the differentiable rollout).
    """
    def _fwd(x_noised, cond, t, c_grid):
        cond = torch.cat([cond, (cond - x_prev) * inv_s_da], dim=1)
        if amp_dtype is None:
            out = model(x_noised, cond, t, c_grid, c_scalar=c_scalar)
        else:
            with torch.autocast('cuda', dtype=amp_dtype):
                out = model(x_noised, cond, t, c_grid, c_scalar=c_scalar)
        return out.float() * cell_mask

    def call(x_noised, cond, t, c_grid):
        if ckpt and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(_fwd, x_noised, cond, t, c_grid,
                                                     use_reentrant=False)
        return _fwd(x_noised, cond, t, c_grid)
    return call


def masked_si_loss(si, call, x, c_grid, y, loss_w):
    """DynamicInterpolant.compute_loss with spherical masked noise * noise_scales,
    land mask and the area (+1/s_inc^2) weights. x-prediction; fp32 reduction."""
    noise = si.get_noise(x) * si.noise_scales
    t = si.wide(torch.rand(x.shape[0], device=x.device))
    W_t = torch.sqrt(t) * noise
    X_t = (1 - t) * x + t * y + (1 - t) * si.sigma_coef * W_t
    pred = call(X_t, x, t.squeeze(dim=[1, 2, 3]), c_grid)
    return (loss_w * (pred - y).pow(2)).sum() / (x.shape[0] * y.shape[1])


def rollout_step(call, si, x, c_grid, ddim_steps, inc_clamp, s_inc, cell_mask):
    """One 5-day call a_t -> a_{t+1}: ddim_steps-step DDIM (gamma from si), the
    increment bounded to +-inc_clamp*s_inc (0 = off), land re-zeroed."""
    y = si.sample(call, x, c_grid, num_steps=int(ddim_steps))
    if inc_clamp and inc_clamp > 0:
        lim = float(inc_clamp) * s_inc
        y = x + torch.maximum(torch.minimum(y - x, lim), -lim)
    return y * cell_mask


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_model(ocean, cfg):
    from uvit import UViT
    return UViT(
        keep_idx=ocean.keep_idx, n_tokens=ocean.n_tokens,
        surf_mask=ocean.cell_mask[0].amax(dim=0),
        n_out=N_STATE, cond_ch=2 * N_STATE, forcing_ch=C_GRID, nlat=H, nlon=W,
        widths=cfg['uvit_widths'], latent_dim=cfg['uvit_latent_dim'],
        dit_dim=cfg['dit_dim'], dit_heads=cfg['dit_heads'], dit_blocks=cfg['dit_blocks'],
        dit_ca_blocks=cfg['dit_ca_blocks'], dit_dropout=cfg.get('dit_dropout', 0.0),
        grad_checkpoint=cfg.get('grad_checkpoint', True))


def build_si(cfg, cell_mask, s_inc_f, device):
    """DynamicInterpolant with L0.5 spherical noise (masked, calibrated) and
    noise_scales = s_inc_floored registered as a buffer and moved to device."""
    from modules.diffusion.x_interpolant import DynamicInterpolant
    from ocean_noise import OceanSphereNoise
    si = DynamicInterpolant(
        num_steps=cfg['si_num_steps'], sigma_coef=cfg['si_sigma_coef'],
        train_sampler='uniform', integrator='ddim', noise='gaussian',
        gamma=cfg.get('si_gamma', 0.0), spectral_weight=0.0, loss_form='x')
    del si.noise_scales                     # plain None attribute from the constructor
    si.register_buffer('noise_scales', s_inc_f.detach().clone().view(N_STATE, 1, 1).float())
    si.generator = OceanSphereNoise(nlat=H, nlon=W, Cl=None, mask=cell_mask,
                                    lmax=cfg.get('si_lmax', (H + 1) // 2))
    si.to(device)
    scale = si.generator.calibrate(device)
    assert si.noise_scales.shape == (N_STATE, 1, 1)
    assert si.noise_scales.device.type == device.type
    return si, scale


def loss_weights(ocean, s_inc_f, mode):
    if mode == 'inc':
        return ocean.loss_w * (1.0 / s_inc_f ** 2).view(1, N_STATE, 1, 1)
    if mode == 'full':
        return ocean.loss_w.clone()
    raise ValueError(f'LOSS_W_MODE must be inc|full, got {mode!r}')


def load_weights(module, path):
    """Load raw weights (accelerate dir with model.safetensors, or a .safetensors
    file) or an EMA sidecar (*.pt, AveragedModel state: 'module.' prefix)."""
    if os.path.isdir(path):
        path = os.path.join(path, 'model.safetensors')
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        sd = {k[len('module.'):]: v for k, v in sd.items() if k.startswith('module.')}
    module.load_state_dict(sd)
    return path


# ---------------------------------------------------------------------------
# Rollout evaluation (hindcast with true forcing from the val split)
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_eval(module, si, data, ocean, s_inc_f, inv_s_da, start, n_calls,
                 ddim_steps, inc_clamp, seed, sst_std, device, tag=''):
    """n_calls 5-day calls from val time `start` (needs start-1 as a_{t-1}).
    fp32. Returns summary + per-call curves. rmse77 in normalized units
    (area-weighted, ocean only, mean over the 77 channels of the squared error),
    SST RMSE in deg C, persistence baseline, max|a|."""
    was_training = module.training
    module.eval()
    cm = ocean.cell_mask
    w = ocean.loss_w
    st = data.states

    def state(i):
        return torch.from_numpy(np.array(st[i])).unsqueeze(0).to(device)

    def forc(i):
        f = np.array(data.forcing[i - 1:i + 1]).reshape(1, C_GRID, H, W)
        return torch.from_numpy(f).to(device)

    x_prev, x = state(start - 1), state(start)
    persist = x.clone()
    maxabs0 = float(x.abs().max())
    rmse77, rmse_sst, rmse_pers, maxabs = [], [], [], []
    t0 = time.time()
    for k in range(n_calls):
        t = start + k
        scalar = torch.tensor([[0.0, float(data.doy[t])]], device=device)
        torch.manual_seed(seed + 1000 * k)
        call = bind_call(module, scalar, x_prev, cm, inv_s_da, amp_dtype=None)
        x_new = rollout_step(call, si, x, forc(t), ddim_steps, inc_clamp, s_inc_f.view(1, -1, 1, 1), cm)
        if not torch.isfinite(x_new).all():
            x_new = torch.nan_to_num(x_new, nan=0.0, posinf=0.0, neginf=0.0)
            maxabs.append(float('inf'))
        else:
            maxabs.append(float(x_new.abs().max()))
        x_prev, x = x, x_new
        truth = state(t + 1)
        e2 = (x - truth) ** 2
        rmse77.append(float(((w * e2).sum() / N_STATE).sqrt()))
        rmse_sst.append(float((w[:, SST_CHANNEL] * e2[:, SST_CHANNEL]).sum().sqrt()) * sst_std)
        rmse_pers.append(float(((w * (persist - truth) ** 2).sum() / N_STATE).sqrt()))
    if was_training:
        module.train()
    rmse77, rmse_sst, maxabs = np.array(rmse77), np.array(rmse_sst), np.array(maxabs)
    out = dict(rmse77_end=float(rmse77[-1]), rmse77_mean=float(rmse77.mean()),
               rmse_sst_end=float(rmse_sst[-1]), rmse_pers_end=float(rmse_pers[-1]),
               maxabs_ic=maxabs0, maxabs_peak=float(maxabs.max()), seconds=time.time() - t0,
               curves=dict(rmse77=rmse77, rmse_sst=rmse_sst, maxabs=maxabs,
                           rmse_pers=np.array(rmse_pers)))
    print(f'  [rollout{("/" + tag) if tag else ""}] {n_calls} calls from {data.date(start)} '
          f'ddim={ddim_steps} clamp={inc_clamp}: rmse77 end={out["rmse77_end"]:.4f} '
          f'mean={out["rmse77_mean"]:.4f} (persistence end {out["rmse_pers_end"]:.4f}) | '
          f'SST end {out["rmse_sst_end"]:.3f} C | max|a| {maxabs0:.1f} -> peak '
          f'{out["maxabs_peak"]:.1f} | {out["seconds"]:.0f}s', flush=True)
    return out


def rollout_png(results, n_calls, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    kx = np.arange(1, n_calls + 1) * 5 / 365.25
    for tag, r in results.items():
        c = r['curves']
        axes[0].plot(kx, c['rmse77'], label=tag)
        axes[1].plot(kx, c['rmse_sst'], label=tag)
        axes[2].semilogy(kx, c['maxabs'], label=tag)
    axes[0].plot(kx, next(iter(results.values()))['curves']['rmse_pers'], 'k--', label='persistence')
    axes[0].set_title('77-ch RMSE (normalized units)')
    axes[1].set_title('SST RMSE (deg C)')
    axes[2].set_title('max|a| of predicted state')
    for ax in axes:
        ax.set_xlabel('years')
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def set_precision(mode):
    """Global matmul/conv precision + the autocast dtype for training forwards.

    'fp32' (default): strict IEEE fp32 everywhere. REQUIRED by the no-skip
        full-field formulation: the network must read a_t (O(1)) and return
        a_{t+1} to ~1e-3, because deep-tracer 5-day increments are tiny
        (thetao_18 s_inc 0.0018, so_18 0.0013 in normalized units) and the loss
        weights them by 1/s_inc^2. Measured: rounding a ~N(0,0.9) field to bf16
        alone costs 1.14 x s_inc(so_18) of error; TF32 inputs (10-bit mantissa)
        ~0.2 x. Cost at 1deg, batch 8, one H100: 2.43 s/step (tf32 1.36, bf16 0.56).
    'tf32': fp32 tensors, TF32 matmul/conv kernels.
    'bf16': TF32 + bf16 autocast of the training forwards (Samudra-SI setting;
        fine only with a carry/residual head -- not here).
    Samplers / rollout evaluation always run without autocast.
    """
    if mode not in ('fp32', 'tf32', 'bf16'):
        raise ValueError(f'PRECISION must be fp32|tf32|bf16, got {mode!r}')
    k = 'ieee' if mode == 'fp32' else 'tf32'
    torch.backends.cuda.matmul.fp32_precision = k
    torch.backends.cudnn.conv.fp32_precision = k
    torch.backends.cudnn.benchmark = True
    return torch.bfloat16 if mode == 'bf16' else None


def make_onecycle(optimizer, max_lr, epochs, steps_per_epoch, pct_start, div_factor, min_lr):
    """OneCycleLR stepped once per OPTIMIZER step (user decision 2026-10-13).

    lr: max_lr/div_factor -> (cosine, pct_start of the run) max_lr -> (cosine)
    min_lr at the last step. Momentum is NOT cycled (AdamW betas stay (0.9, 0.95)).
    The schedule is always rebuilt from the current settings; a resume
    fast-forwards it with onecycle_fast_forward(), so edited LR / EPOCHS take effect.
    """
    from torch.optim.lr_scheduler import OneCycleLR
    initial = max_lr / div_factor
    return OneCycleLR(optimizer, max_lr=max_lr, total_steps=epochs * steps_per_epoch,
                      pct_start=pct_start, anneal_strategy='cos', cycle_momentum=False,
                      div_factor=div_factor, final_div_factor=max(initial / min_lr, 1.0))


def onecycle_step(sched):
    """One scheduler step; holds the last lr instead of raising past the end."""
    if sched.last_epoch + 1 < sched.total_steps:
        sched.step()


def onecycle_fast_forward(sched, n):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')        # "lr_scheduler.step() before optimizer.step()"
        for _ in range(n):
            onecycle_step(sched)


def env_int(name, default):
    v = os.environ.get(name)
    return default if v in (None, '') else int(v)


def env_float(name, default):
    v = os.environ.get(name)
    return default if v in (None, '') else float(v)


def env_bool(name, default):
    v = os.environ.get(name)
    return default if v in (None, '') else v not in ('0', 'false', 'False')
