# ===========================================================================
# 1deg SI forecaster, STAGE 1: single-step SI training (optimize the first call only).
#
# Fork of Samudra-SI/Samudra_SI_NVIDIA.py (2026-10-13). Formulation in
# si_common.py; what changed vs Samudra-SI:
#   * ONE output step: SI bridge a_t -> a_{t+1} (77 ch, 5 days), not the
#     two-in/two-out pair (154 ch, 10 days).
#   * NO big skip: the UViT outputs the full field (no carry(x) + s_inc*g).
#   * cond = [a_t, da_t], da_t = (a_t - a_{t-1}) / s_inc.
#   * 1deg data (OM4.zarr -> 0-prepare_data.py memmaps), Samudra normalization,
#     full field (no climatology / b0 subtraction).
#   * forcing as src/CRPS: [tauuo, tauvo, hfds] at t-1 and t (6 ch), no hfds_anom.
#   * UViT stride 4 (2 conv levels): 180x360 -> 45x90 latent.
# Kept: L0.5 spherical masked noise with noise_scales = s_inc, x-prediction,
# 1/s_inc^2 area-weighted loss (LOSS_W_MODE=inc), input-noise hedge 0.1*s_inc,
# EMA 0.999. Resume from <tag>_best by default (RESUME_FROM=best|latest).
# Changed: strict fp32, FORCED (no bf16 / TF32) -- without the
# carry the net must reproduce a_t to ~1e-3 (see si_common.set_precision).
#
# Launch: bash 1-SI_step1.sh   (env overrides listed there)
# ===========================================================================
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
import math
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from si_common import (  # noqa: E402
    H, W, N_STATE, C_GRID, PROGNOSTIC_VARS, SST_CHANNEL, SIData, SIWindowDataset,
    bind_call, masked_si_loss, build_model, build_si, loss_weights, rollout_eval,
    rollout_png, set_precision, make_onecycle, onecycle_step, onecycle_fast_forward, env_int, env_float, env_bool,
)
from ocean_mask import OceanMask  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


# ================== hyperparameters (env-overridable) ==================
epochs = env_int('EPOCHS', 1500)               # ~108k optimizer steps at 7 GPUs x 8 (0.25deg ckpt-A: ~90k at its best)
batch_size = env_int('BATCH_SIZE', 8)              # per GPU
grad_accum = env_int('GRAD_ACCUM', 1)
limit_windows = env_int('LIMIT_WINDOWS', 0) or None   # smoke-test knob
num_workers = env_int('NUM_WORKERS', 6)
lr = env_float('LR', 1e-4)
min_lr = env_float('MIN_LR', 1e-6)
# OneCycle (per optimizer step): LR/DIV_FACTOR -> LR over PCT_START of the run, cosine down to MIN_LR
pct_start = env_float('PCT_START', 0.3)
div_factor = env_float('DIV_FACTOR', 25.0)
weight_decay = 0.05
grad_clip = 1.0
ema_decay = env_float('EMA_DECAY', 0.999)          # <=0 disables
precision = 'fp32'   # FORCED strict fp32 (user decision 2026-10-13); see si_common.set_precision
amp_dtype = set_precision(precision)
loss_w_mode = os.environ.get('LOSS_W_MODE', 'inc')  # inc (1/s_inc^2) | full
input_noise_rel = env_float('INPUT_NOISE_REL', 0.1)

cfg = dict(
    uvit_widths=(256, 384), uvit_latent_dim=512,
    dit_dim=768, dit_heads=12, dit_blocks=12, dit_ca_blocks=2, dit_dropout=0.0,
    grad_checkpoint=env_bool('GRAD_CKPT', True),   # conv stages; B=8 fp32: 31 GiB vs 58, +10% time (fits beside CRPS on GPU 0)
    si_num_steps=10, si_sigma_coef=0.5, si_gamma=0.0, si_lmax=(H + 1) // 2,
)

# diagnostics
diag_every = env_int('DIAG_EVERY', 5)              # 10-step DDIM sample + calibration
rollout_every = env_int('ROLLOUT_EVERY', 20)        # 0 = off
rollout_calls = env_int('ROLLOUT_CALLS', 73)        # 73 x 5 d = 1 yr
rollout_ddim = env_int('ROLLOUT_DDIM_STEPS', 2)
inc_clamp = env_float('INC_CLAMP', 8.0)
rollout_seed = 0
VAL_SEED = 20261013

ckpt_tag = os.environ.get('CKPT_TAG', 'SI1deg_step1')
ckpt_dir = os.environ.get('CKPT_DIR', '/data/weidong/InMOS_1deg/ckpt/SI')
log_dir = os.environ.get('LOG_DIR', os.path.join(HERE, 'logs'))
use_wandb = env_bool('USE_WANDB', True)
wandb_project = 'InMOS-1deg-SI'
wandb_entity = 'beauwdli-the-university-of-chicago'
# ======================================================================

samples_dir = os.path.join(log_dir, 'samples', ckpt_tag)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, ckpt_tag)
ckpt_path_best = ckpt_path + '_best'

accelerator = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision='no',
                          log_with='wandb' if use_wandb else None)
device = accelerator.device
is_main = accelerator.is_main_process

# ---- data ----
data = SIData()
train_ds = SIWindowDataset(data, 'train', K=1, limit=limit_windows)
val_ds = SIWindowDataset(data, 'val', K=1, limit=limit_windows)
loader_kw = dict(num_workers=num_workers, pin_memory=True,
                 persistent_workers=num_workers > 0,
                 prefetch_factor=2 if num_workers > 0 else None)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kw)

# ---- mask, stats, model, interpolant ----
stride = 2 ** len(cfg['uvit_widths'])
ocean = OceanMask(patch_size=stride).to(device)
cell_mask = ocean.cell_mask                                  # (1,77,H,W)
s_inc_f = torch.from_numpy(data.s_inc_floored).to(device)    # (77,)
s_inc_b = s_inc_f.view(1, N_STATE, 1, 1)
inv_s_da = 1.0 / s_inc_b
loss_w = loss_weights(ocean, s_inc_f, loss_w_mode)
sst_std = data.thetao0_std()

model = build_model(ocean, cfg)
si, noise_scale = build_si(cfg, cell_mask, s_inc_f, device)
if is_main:
    n_dit = sum(p.numel() for p in model.dit.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f'Device {device}, world {accelerator.num_processes} | train windows {len(train_ds)}, '
          f'val {len(val_ds)} | train {data.date(data.train_range[0])}..{data.date(data.train_range[1])}, '
          f'val {data.date(data.val_range[0])}..{data.date(data.val_range[1])}')
    print(f'Land mask: {ocean.summary()}')
    print(f'UViT widths={cfg["uvit_widths"]} latent={cfg["uvit_latent_dim"]}@{H//stride}x{W//stride} '
          f'| DiT {n_dit/1e6:.1f}M | total {n_all/1e6:.1f}M | NO carry (full-field output)')
    print(f's_inc_floored min {s_inc_f.min():.5f} median {s_inc_f.median():.5f} max {s_inc_f.max():.5f} '
          f'| noise calib {noise_scale:.4g} | loss_w_mode={loss_w_mode}')

optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
steps_per_epoch = math.ceil(len(train_loader) / grad_accum)   # optimizer steps per epoch (per rank)
lr_sched = make_onecycle(optimizer, lr, epochs, steps_per_epoch, pct_start, div_factor, min_lr)
raw_module = accelerator.unwrap_model(model)
use_ema = ema_decay > 0
ema_model = AveragedModel(raw_module, avg_fn=get_ema_avg_fn(ema_decay)) if use_ema else None

# ---- resume ----
# RESUME_FROM=best (default, user decision 2026-10-13): continue from <tag>_best
# (selected by the SEEDED EMA val loss), rolling back any epochs after it; falls
# back to the latest checkpoint if no _best exists. RESUME_FROM=latest: continue
# from the last saved epoch. The epochs after the best one are re-trained and
# overwrite <tag>/; a rolled-back resume logs to a NEW wandb run (wandb drops
# steps lower than a run's last step).
resume_from = os.environ.get('RESUME_FROM', 'best')
assert resume_from in ('best', 'latest'), f'RESUME_FROM must be best|latest, got {resume_from}'
start_epoch, global_step, best_loss, wandb_run_id = 0, 0, float('inf'), None
wandb_name = ckpt_tag
load_path = None
if resume_from == 'best' and os.path.exists(os.path.join(ckpt_path_best, 'model.safetensors')):
    load_path = ckpt_path_best
elif os.path.exists(os.path.join(ckpt_path, 'model.safetensors')):
    load_path = ckpt_path
if load_path is not None:
    accelerator.load_state(load_path)
    if use_ema and os.path.exists(load_path + '_ema.pt'):
        ema_model.load_state_dict(torch.load(load_path + '_ema.pt', map_location='cpu', weights_only=False))
    meta = torch.load(load_path + '_meta.pt', map_location='cpu', weights_only=False)
    best_loss = meta.get('best_loss', best_loss)
    global_step = meta.get('global_step', 0)
    start_epoch = meta.get('epoch', -1) + 1
    wandb_run_id = meta.get('wandb_run_id')
    if load_path == ckpt_path_best and os.path.exists(ckpt_path + '_meta.pt'):
        last = torch.load(ckpt_path + '_meta.pt', map_location='cpu', weights_only=False).get('epoch', -1) + 1
        if last > start_epoch:
            wandb_run_id = None
            wandb_name = f'{ckpt_tag}_from_ep{start_epoch}'
            if is_main:
                print(f'RESUME_FROM=best: rolling back from epoch {last} to the best epoch {start_epoch}')
    # Rebuild the schedule from THIS file and fast-forward (a stored scheduler
    # state would silently override edited lr / epochs).
    onecycle_fast_forward(lr_sched, start_epoch * steps_per_epoch)
    if is_main:
        print(f'Resumed {load_path}: epoch {start_epoch}, step {global_step}, best {best_loss:.5f}')
elif is_main:
    print('No checkpoint; training from scratch.')

if use_wandb:
    init = dict(entity=wandb_entity, name=wandb_name)
    if wandb_run_id:
        init.update(id=wandb_run_id, resume='must')
    accelerator.init_trackers(wandb_project, config=dict(
        stage=1, epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
        world=accelerator.num_processes, lr=lr, min_lr=min_lr, scheduler='onecycle', pct_start=pct_start, div_factor=div_factor,
        ema_decay=ema_decay, precision=precision, loss_w_mode=loss_w_mode,
        input_noise_rel=input_noise_rel, **{k: (list(v) if isinstance(v, tuple) else v)
                                            for k, v in cfg.items()}),
        init_kwargs={'wandb': init})
    if is_main:
        wandb_run_id = accelerator.get_tracker('wandb', unwrap=True).id

# ---- pinned val example ----
xp_pin, xc_pin, y_pin, f_pin, sc_pin = val_ds[0]
xp_pin, xc_pin = xp_pin[None].to(device), xc_pin[None].to(device)
y_pin, f_pin, sc_pin = y_pin.to(device), f_pin.to(device), sc_pin.to(device)   # K=1 -> batch dim of 1
_pers = float((loss_w * (xc_pin - y_pin).pow(2)).sum() / N_STATE)
if is_main:
    print(f'persistence loss on the pinned val window: {_pers:.4f}'
          + (' (target ~1 in the inc weighting)' if loss_w_mode == 'inc' else ''))
if loss_w_mode == 'inc':
    assert 0.2 < _pers < 5.0, f'persistence loss {_pers:.3f} far from 1: s_inc / mask mis-wired'


def sample_and_diagnose(mod, epoch, tag):
    """10-step DDIM on the pinned window: increment calibration std(y-x)/s_inc
    (target 1, ocean-only, area-weighted), MSE in the training weighting vs
    persistence, thetao_0 figure."""
    call = bind_call(mod, sc_pin, xp_pin, cell_mask, inv_s_da, amp_dtype=None)
    with torch.no_grad():
        torch.manual_seed(VAL_SEED)
        yp = si.sample(call, xc_pin, f_pin, num_steps=cfg['si_num_steps'])
    w = ocean.loss_w
    d = yp - xc_pin
    mu = (w * d).sum(dim=(-2, -1), keepdim=True)
    calib = ((w * (d - mu) ** 2).sum(dim=(-2, -1)).sqrt()[0] / s_inc_f).cpu().numpy()
    mse = float((loss_w * (yp - y_pin) ** 2).sum() / N_STATE)
    mse_p = float((loss_w * (xc_pin - y_pin) ** 2).sum() / N_STATE)
    c = SST_CHANNEL
    a_per, a_tgt, a_prd = (z[0, c].cpu().numpy() for z in (xc_pin, y_pin, yp))
    wc = w[0, c].cpu().numpy()
    r = float(np.sqrt((wc * (a_prd - a_tgt) ** 2).sum())) * sst_std
    r_p = float(np.sqrt((wc * (a_per - a_tgt) ** 2).sum())) * sst_std
    inc_t, inc_p = a_tgt - a_per, a_prd - a_per
    lim = max(float(np.abs(inc_t).max()), 1e-6)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    for a, z, ttl, cm_, vv in ((ax[0, 0], a_tgt, 'truth thetao_0 (t+1)', 'RdYlBu_r', None),
                               (ax[0, 1], a_prd, f'sampled  RMSE {r:.3f} C (persist {r_p:.3f})', 'RdYlBu_r', None),
                               (ax[1, 0], inc_t, 'true 5-day increment', 'RdBu_r', lim),
                               (ax[1, 1], inc_p, 'sampled 5-day increment', 'RdBu_r', lim)):
        im = a.imshow(z, origin='lower', cmap=cm_, vmin=-vv if vv else None, vmax=vv)
        a.set_title(ttl); a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(f'{tag} epoch {epoch+1}  {data.date(int(val_ds.ts[0]) + 1)}')
    fig.savefig(os.path.join(samples_dir, f'epoch_{epoch+1:04d}_{tag}.png'), dpi=100)
    plt.close(fig)
    print(f'  [{tag}] calib std(y-x)/s_inc (target 1): mean {calib.mean():.3f} | '
          f'uo {calib[0:19].mean():.3f} vo {calib[19:38].mean():.3f} thetao {calib[38:57].mean():.3f} '
          f'so {calib[57:76].mean():.3f} zos {calib[76]:.3f} | MSE {mse:.4f} (persist {mse_p:.4f}) '
          f'| SST RMSE {r:.3f} C (persist {r_p:.3f})', flush=True)
    return dict(calib_mean=float(calib.mean()), mse=mse, sst_rmse=r)


for epoch in range(start_epoch, epochs):
    model.train()
    t_ep = time.time()
    run, n = 0.0, 0
    pbar = tqdm(train_loader, disable=not is_main, desc=f'ep {epoch+1}')
    for x_prev, x_cur, y, forcing, scalar in pbar:
        with accelerator.accumulate(model):
            y, forcing, scalar = y[:, 0], forcing[:, 0], scalar[:, 0]
            # Input-noise hedge (training only): both input states perturbed, so the
            # source AND da_t see an imperfect input, as in a rollout.
            if input_noise_rel > 0:
                with torch.no_grad():
                    x_prev = x_prev + input_noise_rel * si.noise_scales * si.get_noise(x_prev)
                    x_cur = x_cur + input_noise_rel * si.noise_scales * si.get_noise(x_cur)
            call = bind_call(model, scalar, x_prev, cell_mask, inv_s_da, amp_dtype=amp_dtype)
            loss = masked_si_loss(si, call, x_cur, forcing, y, loss_w)
            # Non-finite on ANY rank -> skip on ALL ranks (a lone skip would leave the
            # other ranks waiting in the gradient all-reduce).
            if accelerator.reduce(torch.tensor([float(not torch.isfinite(loss))], device=device), 'sum').item() > 0:
                if is_main:
                    print('  [warn] non-finite loss; step skipped on all ranks')
                optimizer.zero_grad()
                continue
            accelerator.backward(loss)
            if accelerator.sync_gradients and grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                onecycle_step(lr_sched)
            if use_ema and accelerator.sync_gradients:
                ema_model.update_parameters(raw_module)
        lv = float(loss.detach())
        run, n = run + lv, n + 1
        global_step += 1
        if is_main:
            pbar.set_postfix(loss=f'{lv:.4f}', run=f'{run/n:.4f}')
            if use_wandb:
                accelerator.log({'train/loss': lv, 'train/lr': optimizer.param_groups[0]['lr'],
                                 'epoch': epoch + 1}, step=global_step)

    # ---- val: raw and EMA, same t/noise per batch, summed over ranks ----
    model.eval()
    mods = [('raw', raw_module)] + ([('ema', ema_model.module)] if use_ema else [])
    sums = torch.zeros(len(mods), device=device)
    cnt = torch.zeros(1, device=device)
    with torch.no_grad(), torch.random.fork_rng(devices=[device]):
        for bi, (x_prev, x_cur, y, forcing, scalar) in enumerate(val_loader):
            for j, (_, mod) in enumerate(mods):
                mod.eval()
                torch.manual_seed(VAL_SEED + bi + 100000 * accelerator.process_index)
                call = bind_call(mod, scalar[:, 0], x_prev, cell_mask, inv_s_da, amp_dtype=amp_dtype)
                sums[j] += masked_si_loss(si, call, x_cur, forcing[:, 0], y[:, 0], loss_w) * x_cur.shape[0]
            cnt += x_cur.shape[0]
    sums, cnt = accelerator.reduce(sums, 'sum'), accelerator.reduce(cnt, 'sum')
    val = {name: float(sums[j] / cnt) for j, (name, _) in enumerate(mods)}
    val_loss = val['ema' if use_ema else 'raw']

    if is_main:
        dt = time.time() - t_ep
        print(f'Epoch {epoch+1}/{epochs} | train {run/max(n,1):.5f} | '
              + ' | '.join(f'val[{k}] {v:.5f}' for k, v in val.items())
              + f' | lr {optimizer.param_groups[0]["lr"]:.2e} | {dt/60:.1f} min | peak '
              f'{torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)
        log = {f'val/loss_{k}': v for k, v in val.items()}
        if diag_every and (epoch + 1) % diag_every == 0:
            for name, mod in mods:
                for k, v in sample_and_diagnose(mod, epoch, name).items():
                    log[f'diag/{k}_{name}'] = v
        if rollout_every and (epoch + 1) % rollout_every == 0:
            res = {}
            for name, mod in mods:
                res[name] = rollout_eval(mod, si, data, ocean, s_inc_f, inv_s_da,
                                         data.val_range[0] + 1, rollout_calls, rollout_ddim,
                                         inc_clamp, rollout_seed, sst_std, device, tag=name)
                log[f'rollout/rmse77_end_{name}'] = res[name]['rmse77_end']
                log[f'rollout/rmse77_mean_{name}'] = res[name]['rmse77_mean']
                log[f'rollout/sst_end_{name}'] = res[name]['rmse_sst_end']
                log[f'rollout/maxabs_peak_{name}'] = res[name]['maxabs_peak']
            rollout_png(res, rollout_calls, f'{ckpt_tag} epoch {epoch+1}',
                        os.path.join(samples_dir, f'rollout_epoch_{epoch+1:04d}.png'))
        if use_wandb:
            accelerator.log(log, step=global_step)

    # ---- save latest (+ best by EMA val loss) ----
    accelerator.wait_for_everyone()
    if is_main:
        meta = dict(best_loss=best_loss, global_step=global_step, epoch=epoch,
                    wandb_run_id=wandb_run_id, cfg=cfg, loss_w_mode=loss_w_mode)
        accelerator.save_state(ckpt_path)
        torch.save(meta, ckpt_path + '_meta.pt')
        if use_ema:
            torch.save(ema_model.state_dict(), ckpt_path + '_ema.pt')
        if val_loss < best_loss:
            best_loss = meta['best_loss'] = val_loss
            accelerator.save_state(ckpt_path_best)
            torch.save(meta, ckpt_path_best + '_meta.pt')
            if use_ema:
                torch.save(ema_model.state_dict(), ckpt_path_best + '_ema.pt')
            print(f'  best val {best_loss:.5f} -> {ckpt_path_best}')
    accelerator.wait_for_everyone()

if use_wandb:
    accelerator.end_training()
