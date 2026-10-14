# ===========================================================================
# 1deg SI forecaster, STAGE 3: long-horizon REPLAY fine-tune.
#
# Port of Samudra-SI's 2.x replay series (2-/2.1-/2.2-/2.3-Samudra_SI_rollout_
# NVIDIA.py + ocean_replay_dataset.py). Warm start from stage 2 (2-SI_rollout.py).
#
# Why (Samudra-SI ROLLOUT_STAGES.md): a K-call unroll from TRUTH inputs never
# shows the trainer the states its own long rollouts live in, so the rollout
# blew up after ~1 yr whatever K was. Replay fixed it: the onset moved
# 1.0 -> 2.2 -> 2.9 -> 6.7 yr -> none in 8 yr as the replay range grew, and
# tracked ~1.7-2 x the maximum replay length.
#
# What it does, per batch (same m on every rank and every sample of the batch):
#   1. draw m ~ U{REPLAY_MIN..REPLAY_MAX} (m = 0 with probability 1 - REPLAY_P);
#   2. REPLAY: roll the LIVE raw weights m calls WITHOUT gradient from the truth
#      pair (a_{t-1}, a_t), with the inference operator (ROLLOUT_DDIM_STEPS-step
#      DDIM + INC_CLAMP*s_inc clamp, fp32) and true lagged forcing;
#   3. the K graded calls (exactly stage 2: differentiable unroll, linear
#      per-call weights) start from that MODEL-GENERATED pair, with the TRUE
#      targets a_{t+m+1} .. a_{t+m+K}.
# It is not a stored buffer: the replayed states are regenerated with the
# current weights every batch, so they always come from the model being trained.
#
# Curriculum (REPLAY_SCHEDULE "epoch:min-max,..."; one call = 5 days here,
# 10 days in Samudra-SI, so the call counts are doubled for the same physical
# time). Default "0:0-80,3:60-220,6:120-400" with EPOCHS=9:
#   epochs 1-3  m in   0-80  (0-1.1 yr)     ~ Samudra-SI 2 / 2.1 (0-40 calls)
#   epochs 4-6  m in  60-220 (0.8-3.0 yr)   ~ 2.2 (30-110)
#   epochs 7-9  m in 120-400 (1.6-5.5 yr)   ~ 2.3 (60-200), the final recipe
# Train windows are the same for every stage (room for 400 + K calls).
#
# Loss weights stay cos(lat) * mask / s_inc^2 (user decision). Samudra-SI 2.2/2.3
# also boosted the surface uo/vo (levels 0-2) loss weights x4 because the
# blow-up mode lived there and 1/s_inc^2 gave them 0.3 % of the loss;
# LOSS_W_SURFV_BOOST (default 1 = off) reproduces that if needed.
# Selection: `_best` = lowest rmse77 at the end of a ROLLOUT_CALLS (8 yr) val
# hindcast of SELECT_WEIGHTS (default raw: Samudra-SI's final model used the raw
# weights; the EMA lagged).
#
# Launch: bash 3-replay_buffer.sh   (env overrides listed there)
# ===========================================================================
import contextlib
import math
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from si_common import (  # noqa: E402
    H, W, N_STATE, C_GRID, PROGNOSTIC_VARS, SIData, SIWindowDataset, bind_call,
    masked_si_loss, rollout_step, build_model, build_si, loss_weights, load_weights,
    rollout_eval, rollout_png, set_precision, make_onecycle, onecycle_step,
    onecycle_fast_forward, env_int, env_float, env_bool,
)
from ocean_mask import OceanMask  # noqa: E402


# ================== hyperparameters (env-overridable) ==================
K = env_int('K', 8)
epochs = env_int('EPOCHS', 9)
replay_schedule_spec = os.environ.get('REPLAY_SCHEDULE', '0:0-80,3:60-220,6:120-400')
replay_p = env_float('REPLAY_P', 1.0)
batch_size = env_int('BATCH_SIZE', 2)              # per GPU
grad_accum = env_int('GRAD_ACCUM', 1)
limit_windows = env_int('LIMIT_WINDOWS', 0) or None
lr = env_float('LR', 1e-5)
min_lr = env_float('MIN_LR', 1e-7)
pct_start = env_float('PCT_START', 0.3)
div_factor = env_float('DIV_FACTOR', 25.0)
weight_decay = 0.05
grad_clip = 1.0
ema_decay = env_float('EMA_DECAY', 0.99)           # Samudra-SI 2.3: 0.99, reseeded at stage start
precision = 'fp32'   # FORCED strict fp32 (user decision 2026-10-13); see si_common.set_precision
amp_dtype = set_precision(precision)
input_noise_rel = env_float('INPUT_NOISE_REL', 0.1)
full_grad = env_bool('FULL_GRAD', True)
step_weights_spec = os.environ.get('STEP_WEIGHTS', 'linear')
pf_ddim = env_int('ROLLOUT_DDIM_STEPS', 2)          # operator of the replay, the unroll AND the eval
inc_clamp = env_float('INC_CLAMP', 8.0)
surfv_boost = env_float('LOSS_W_SURFV_BOOST', 1.0)
seed = env_int('SEED', 20261014)

rollout_every = env_int('ROLLOUT_EVERY', 1)
rollout_calls = env_int('ROLLOUT_CALLS', 584)       # 584 x 5 d = 8 yr
select_weights = os.environ.get('SELECT_WEIGHTS', 'raw')   # raw | ema
rollout_seed = 0
VAL_SEED = 20261013

init_ckpt = os.environ.get('INIT_CKPT', '/data/weidong/InMOS_1deg/ckpt/SI/SI1deg_rollout_K8_best')
ckpt_tag = os.environ.get('CKPT_TAG', 'SI1deg_replay')
ckpt_dir = os.environ.get('CKPT_DIR', '/data/weidong/InMOS_1deg/ckpt/SI')
log_dir = os.environ.get('LOG_DIR', os.path.join(HERE, 'logs'))
use_wandb = env_bool('USE_WANDB', True)
wandb_project = 'InMOS-1deg-SI'
wandb_entity = 'beauwdli-the-university-of-chicago'
# ======================================================================


def parse_schedule(spec):
    """'0:0-80,3:60-220' -> [(start_epoch0, m_min, m_max), ...] sorted by epoch."""
    out = []
    for part in spec.split(','):
        ep, rng = part.split(':')
        lo, hi = rng.split('-')
        out.append((int(ep), int(lo), int(hi)))
    out.sort()
    assert out[0][0] == 0, f'REPLAY_SCHEDULE must start at epoch 0: {spec}'
    assert all(0 <= lo <= hi for _, lo, hi in out), spec
    return out


def replay_range(schedule, epoch):
    cur = schedule[0]
    for s in schedule:
        if epoch >= s[0]:
            cur = s
    return cur[1], cur[2]


schedule = parse_schedule(replay_schedule_spec)
m_max_all = max(hi for _, _, hi in schedule)

samples_dir = os.path.join(log_dir, 'samples', ckpt_tag)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, ckpt_tag)
ckpt_path_best = ckpt_path + '_best'

accelerator = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision='no',
                          log_with='wandb' if use_wandb else None)
device = accelerator.device
is_main = accelerator.is_main_process
world, rank = accelerator.num_processes, accelerator.process_index

# Model config + loss weighting from the run being fine-tuned (stage 2 meta).
_init_meta = None
for cand in (init_ckpt.replace('_ema.pt', '_meta.pt'), init_ckpt.rstrip('/') + '_meta.pt'):
    if os.path.exists(cand):
        _init_meta = torch.load(cand, map_location='cpu', weights_only=False)
        break
if _init_meta is None:
    raise SystemExit(f'no meta next to INIT_CKPT={init_ckpt}')
cfg = _init_meta['cfg']
loss_w_mode = os.environ.get('LOSS_W_MODE', _init_meta.get('loss_w_mode', 'inc'))
cfg['grad_checkpoint'] = env_bool('GRAD_CKPT', True)

# ---- data ----
data = SIData()
lo_tr, hi_tr = data.train_range
# start t needs a_{t-1}; the window reaches t + m_max_all + K for every stage
train_ts = np.arange(lo_tr + 1, hi_tr - (m_max_all + K) + 1)
if limit_windows:
    train_ts = train_ts[:limit_windows]
iters_per_epoch = len(train_ts) // (world * batch_size)
assert iters_per_epoch > 0, f'{len(train_ts)} windows < world x batch'
val_ds = SIWindowDataset(data, 'val', K=1, limit=limit_windows)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, drop_last=False, num_workers=4,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)

stride = 2 ** len(cfg['uvit_widths'])
ocean = OceanMask(patch_size=stride).to(device)
cell_mask = ocean.cell_mask
s_inc_f = torch.from_numpy(data.s_inc_floored).to(device)
s_inc_b = s_inc_f.view(1, N_STATE, 1, 1)
inv_s_da = 1.0 / s_inc_b
loss_w = loss_weights(ocean, s_inc_f, loss_w_mode)
if surfv_boost != 1.0:
    surf_v = [PROGNOSTIC_VARS.index(f'{v}_{k}') for v in ('uo', 'vo') for k in range(3)]
    loss_w[:, surf_v] *= surfv_boost
sst_std = data.thetao0_std()

model = build_model(ocean, cfg)
si, _ = build_si(cfg, cell_mask, s_inc_f, device)


def step_weights(k_eff):
    if step_weights_spec == 'linear':
        v = np.arange(1, k_eff + 1, dtype=np.float64)
    elif step_weights_spec == 'uniform':
        v = np.ones(k_eff)
    elif step_weights_spec == 'last':
        v = np.zeros(k_eff); v[-1] = 1.0
    else:
        v = np.array([float(s) for s in step_weights_spec.split(',')][:k_eff])
        assert len(v) == k_eff, f'STEP_WEIGHTS has fewer than K={k_eff} entries'
    return (v / v.sum()).tolist()


def states_at(idx):
    """(B,77,H,W) on device for a list of time indices (memmap, page-cached)."""
    return torch.from_numpy(np.stack([data.states[i] for i in idx])).to(device, non_blocking=True)


def forcing_at(idx):
    """(B,6,H,W) lagged forcing [f_{t-1}, f_t] for calls whose latest input is t."""
    f = np.stack([np.asarray(data.forcing[i - 1:i + 1]).reshape(C_GRID, H, W) for i in idx])
    return torch.from_numpy(f).to(device, non_blocking=True)


def scalar_at(idx):
    return torch.tensor([[0.0, float(data.doy[i])] for i in idx], device=device)


optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

# ---- weights: RESUME_FROM=best (default) | latest; else warm start from stage 2 ----
resume_from = os.environ.get('RESUME_FROM', 'best')
assert resume_from in ('best', 'latest'), f'RESUME_FROM must be best|latest, got {resume_from}'
load_path = None
if resume_from == 'best' and os.path.exists(os.path.join(ckpt_path_best, 'model.safetensors')):
    load_path = ckpt_path_best
elif os.path.exists(os.path.join(ckpt_path, 'model.safetensors')):
    load_path = ckpt_path
if load_path is None:
    src = load_weights(model, init_ckpt)
    if is_main:
        print(f'Warm start from {src} (fresh optimizer, EMA reseeded from these weights)')

model, optimizer, val_loader = accelerator.prepare(model, optimizer, val_loader)
steps_per_epoch = math.ceil(iters_per_epoch / grad_accum)
lr_sched = make_onecycle(optimizer, lr, epochs, steps_per_epoch, pct_start, div_factor, min_lr)
raw_module = accelerator.unwrap_model(model)
use_ema = ema_decay > 0
ema_model = AveragedModel(raw_module, avg_fn=get_ema_avg_fn(ema_decay)) if use_ema else None

start_epoch, global_step, best_metric, wandb_run_id = 0, 0, float('inf'), None
wandb_name = ckpt_tag
if load_path is not None:
    accelerator.load_state(load_path)
    if use_ema and os.path.exists(load_path + '_ema.pt'):
        ema_model.load_state_dict(torch.load(load_path + '_ema.pt', map_location='cpu', weights_only=False))
    meta = torch.load(load_path + '_meta.pt', map_location='cpu', weights_only=False)
    best_metric = meta.get('best_metric', best_metric)
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
    onecycle_fast_forward(lr_sched, start_epoch * steps_per_epoch)
    if is_main:
        print(f'Resumed {load_path}: epoch {start_epoch}, step {global_step}, best {best_metric:.4f}')

if is_main:
    print(f'Device {device}, world {world} | K={K} | {len(train_ts)} train windows, '
          f'{iters_per_epoch} it/rank/epoch | batch {batch_size}x{grad_accum}x{world} '
          f'| replay schedule {schedule} p={replay_p} | operator {pf_ddim}-step DDIM + clamp {inc_clamp} '
          f'| loss_w_mode={loss_w_mode} surfV boost {surfv_boost:g} | EMA {ema_decay} '
          f'| select {select_weights} @ {rollout_calls} calls')

if use_wandb:
    init = dict(entity=wandb_entity, name=wandb_name)
    if wandb_run_id:
        init.update(id=wandb_run_id, resume='must')
    accelerator.init_trackers(wandb_project, config=dict(
        stage=3, K=K, epochs=epochs, replay_schedule=replay_schedule_spec, replay_p=replay_p,
        batch_size=batch_size, grad_accum=grad_accum, world=world, lr=lr, min_lr=min_lr,
        scheduler='onecycle', pct_start=pct_start, div_factor=div_factor, ema_decay=ema_decay,
        full_grad=full_grad, step_weights=step_weights_spec, ddim=pf_ddim, inc_clamp=inc_clamp,
        init_ckpt=init_ckpt, loss_w_mode=loss_w_mode, surfv_boost=surfv_boost,
        input_noise_rel=input_noise_rel, select_weights=select_weights,
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}),
        init_kwargs={'wandb': init})
    if is_main:
        wandb_run_id = accelerator.get_tracker('wandb', unwrap=True).id


def run_rollouts(epoch):
    mods = [('raw', raw_module)] + ([('ema', ema_model.module)] if use_ema else [])
    res = {name: rollout_eval(mod, si, data, ocean, s_inc_f, inv_s_da, data.val_range[0] + 1,
                              rollout_calls, pf_ddim, inc_clamp, rollout_seed, sst_std, device,
                              tag=name) for name, mod in mods}
    rollout_png(res, rollout_calls, f'{ckpt_tag} epoch {epoch}',
                os.path.join(samples_dir, f'rollout_epoch_{epoch:04d}.png'))
    return res


if is_main and start_epoch == 0:
    print('Rollout of the warm-start weights (epoch 0 baseline):')
    run_rollouts(0)
accelerator.wait_for_everyone()

for epoch in range(start_epoch, epochs):
    m_lo, m_hi = replay_range(schedule, epoch)
    model.train()
    t_ep = time.time()
    run, n = 0.0, 0
    call_loss = np.zeros(K); call_n = np.zeros(K)
    rep_m, rep_err, n_rep_bad = [], [], 0
    # Same permutation on every rank (seeded by epoch), sharded by rank.
    g = torch.Generator().manual_seed(seed + epoch)
    perm = train_ts[torch.randperm(len(train_ts), generator=g).numpy()]
    mine = perm[rank::world][:iters_per_epoch * batch_size].reshape(iters_per_epoch, batch_size)
    pbar = tqdm(range(iters_per_epoch), disable=not is_main, desc=f'ep {epoch+1} m{m_lo}-{m_hi}')
    for it in pbar:
        ts = [int(t) for t in mine[it]]
        # m shared by all ranks (seeded by epoch, iteration): the slowest rank sets the
        # step time, so independent per-rank draws would cost the max of `world` draws.
        gm = torch.Generator().manual_seed(seed * 7 + epoch * 100003 + it)
        m = 0 if (m_hi == 0 or float(torch.rand((), generator=gm)) >= replay_p) else \
            int(torch.randint(m_lo, m_hi + 1, (1,), generator=gm))
        with accelerator.accumulate(model):
            x_prev, x_cur = states_at([t - 1 for t in ts]), states_at(ts)
            # ---- replay: (a_{t-1}, a_t) -> model^m, no grad, live raw weights ----
            if m > 0:
                with torch.no_grad():
                    xp, xc = x_prev, x_cur
                    for r in range(m):
                        tr = [t + r for t in ts]
                        call_r = bind_call(raw_module, scalar_at(tr), xp, cell_mask, inv_s_da)
                        xn = rollout_step(call_r, si, xc, forcing_at(tr), pf_ddim, inc_clamp,
                                          s_inc_b, cell_mask)
                        xp, xc = xc, xn
                    if torch.isfinite(xc).all() and torch.isfinite(xp).all():
                        truth = states_at([t + m for t in ts])
                        e = ((ocean.loss_w * (xc - truth) ** 2).sum(dim=(1, 2, 3)) / N_STATE).sqrt()
                        rep_m.append(m); rep_err.append(float(e.mean()))
                        x_prev, x_cur = xp, xc
                    else:
                        n_rep_bad += 1      # keep the truth start, but its targets are t+m+k below
                        x_prev, x_cur = states_at([t + m - 1 for t in ts]), states_at([t + m for t in ts])
            t0 = [t + m for t in ts]                          # latest input time of graded call 1
            if input_noise_rel > 0:
                with torch.no_grad():
                    x_prev = x_prev + input_noise_rel * si.noise_scales * si.get_noise(x_prev)
                    x_cur = x_cur + input_noise_rel * si.noise_scales * si.get_noise(x_cur)
            grad_ctx = contextlib.nullcontext() if full_grad else torch.no_grad()
            losses, bad = [], False
            for k in range(K):
                tk = [t + k for t in t0]
                sc_k, f_k = scalar_at(tk), forcing_at(tk)
                y_k = states_at([t + 1 for t in tk])
                call = bind_call(model, sc_k, x_prev, cell_mask, inv_s_da, amp_dtype=amp_dtype,
                                 ckpt=full_grad)
                l_k = masked_si_loss(si, call, x_cur, f_k, y_k, loss_w)
                if not torch.isfinite(l_k):
                    bad = True
                    break
                losses.append(l_k)
                call_loss[k] += float(l_k.detach()); call_n[k] += 1
                if k == K - 1:
                    break
                with grad_ctx:
                    op = bind_call(model if full_grad else raw_module, sc_k, x_prev, cell_mask,
                                   inv_s_da, amp_dtype=amp_dtype, ckpt=full_grad)
                    x_next = rollout_step(op, si, x_cur, f_k, pf_ddim, inc_clamp, s_inc_b, cell_mask)
                if not torch.isfinite(x_next).all():
                    bad = True
                    break
                x_prev, x_cur = x_cur, x_next
            # A non-finite loss/state on ANY rank skips the step on ALL ranks (a rank
            # that skipped alone would leave the others waiting in the all-reduce).
            any_bad = accelerator.reduce(torch.tensor([float(bad)], device=device), 'sum').item() > 0
            if any_bad:
                if is_main:
                    print(f'  [warn] non-finite loss/state at it {it} (m={m}); step skipped on all ranks')
                optimizer.zero_grad()
                losses = [torch.zeros((), device=device)]
                loss = losses[0]
            else:
                lam = step_weights(len(losses))
                loss = sum(w_ * l_ for w_, l_ in zip(lam, losses))
                accelerator.backward(loss)
                if accelerator.sync_gradients and grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                if use_ema and accelerator.sync_gradients:
                    ema_model.update_parameters(raw_module)
            if accelerator.sync_gradients:
                onecycle_step(lr_sched)
            bad = any_bad
        lv = float(loss.detach())
        if not bad:
            run, n = run + lv, n + 1
        global_step += 1
        if is_main:
            pbar.set_postfix(m=m, loss=f'{lv:.4f}', run=f'{run/max(n,1):.4f}',
                             rep_err=f'{rep_err[-1]:.3f}' if (m and rep_err) else '-')
            if use_wandb:
                accelerator.log({'train/loss': lv, 'train/lr': optimizer.param_groups[0]['lr'],
                                 'train/replay_m': m, 'epoch': epoch + 1}, step=global_step)

    # ---- single-step val (raw + EMA), for reference only ----
    model.eval()
    mods = [('raw', raw_module)] + ([('ema', ema_model.module)] if use_ema else [])
    sums = torch.zeros(len(mods), device=device)
    cnt = torch.zeros(1, device=device)
    with torch.no_grad(), torch.random.fork_rng(devices=[device]):
        for bi, (x_prev, x_cur, y, forcing, scalar) in enumerate(val_loader):
            for j, (_, mod) in enumerate(mods):
                mod.eval()
                torch.manual_seed(VAL_SEED + bi + 100000 * rank)
                call = bind_call(mod, scalar[:, 0], x_prev, cell_mask, inv_s_da, amp_dtype=amp_dtype)
                sums[j] += masked_si_loss(si, call, x_cur, forcing[:, 0], y[:, 0], loss_w) * x_cur.shape[0]
            cnt += x_cur.shape[0]
    sums, cnt = accelerator.reduce(sums, 'sum'), accelerator.reduce(cnt, 'sum')
    val = {name: float(sums[j] / cnt) for j, (name, _) in enumerate(mods)}

    metric = None
    if is_main:
        dt = time.time() - t_ep
        rm, re_ = np.array(rep_m), np.array(rep_err)
        bins = [(a, b) for a, b in ((0, 40), (41, 80), (81, 160), (161, 240), (241, 320), (321, 400),
                                    (401, 10 ** 6)) if a <= m_hi]
        rep_by = {f'{a}-{b}': float(re_[(rm >= a) & (rm <= b)].mean())
                  for a, b in bins if ((rm >= a) & (rm <= b)).any()}
        print(f'Epoch {epoch+1}/{epochs} (replay m {m_lo}-{m_hi}) | train {run/max(n,1):.5f} | per-call SI loss '
              + ' '.join(f'l{k+1}={call_loss[k]/max(1, call_n[k]):.4f}' for k in range(K))
              + ' | ' + ' | '.join(f'val1[{k}] {v:.5f}' for k, v in val.items())
              + f' | replay mean m {rm.mean() if len(rm) else 0:.0f}, non-finite {n_rep_bad}, '
              f'replayed-input rmse77 by m: ' + ' '.join(f'{k_}:{v_:.3f}' for k_, v_ in rep_by.items())
              + f' | {dt/60:.1f} min ({dt/max(iters_per_epoch,1):.1f} s/it) | peak '
              f'{torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)
        log = {f'val/loss_{k}': v for k, v in val.items()}
        log.update({f'train/si_loss_call{k+1}': call_loss[k] / max(1, call_n[k]) for k in range(K)})
        log.update({f'train/replay_err_{k_}': v_ for k_, v_ in rep_by.items()})
        log['train/replay_nonfinite'] = n_rep_bad
        if rollout_every and (epoch + 1) % rollout_every == 0:
            res = run_rollouts(epoch + 1)
            for name, r in res.items():
                log[f'rollout/rmse77_end_{name}'] = r['rmse77_end']
                log[f'rollout/rmse77_mean_{name}'] = r['rmse77_mean']
                log[f'rollout/sst_end_{name}'] = r['rmse_sst_end']
                log[f'rollout/maxabs_peak_{name}'] = r['maxabs_peak']
            metric = res[select_weights if use_ema else 'raw']['rmse77_end']
        if use_wandb:
            accelerator.log(log, step=global_step)

    accelerator.wait_for_everyone()
    if is_main:
        meta = dict(best_metric=best_metric, global_step=global_step, epoch=epoch,
                    wandb_run_id=wandb_run_id, cfg=cfg, loss_w_mode=loss_w_mode, K=K,
                    replay_schedule=replay_schedule_spec)
        accelerator.save_state(ckpt_path)
        torch.save(meta, ckpt_path + '_meta.pt')
        if use_ema:
            torch.save(ema_model.state_dict(), ckpt_path + '_ema.pt')
        if metric is not None and np.isfinite(metric) and metric < best_metric:
            best_metric = meta['best_metric'] = metric
            accelerator.save_state(ckpt_path_best)
            torch.save(meta, ckpt_path_best + '_meta.pt')
            if use_ema:
                torch.save(ema_model.state_dict(), ckpt_path_best + '_ema.pt')
            print(f'  best rollout rmse77 ({select_weights}) {best_metric:.4f} -> {ckpt_path_best}')
    accelerator.wait_for_everyone()

if use_wandb:
    accelerator.end_training()
