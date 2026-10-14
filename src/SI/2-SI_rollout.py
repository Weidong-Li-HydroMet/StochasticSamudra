# ===========================================================================
# 1deg SI forecaster, STAGE 2: rollout-loss fine-tune (K = 8 calls = 40 days).
#
# Warm start from stage 1 (1-SI_step1.py). Same model / interpolant / data /
# weights as stage 1 (si_common.py); only the loss changes:
#
#   L = sum_{k=1..K} lambda_k * SI_loss(call k | input x_hat_{k-1}, target a_{t+k})
#
#   x_hat_0 = truth (+ input-noise hedge); x_hat_k = the LIVE model's inference
#   operator applied to x_hat_{k-1}: ROLLOUT_DDIM_STEPS-step DDIM (default 2) +
#   increment clamp INC_CLAMP*s_inc (default 8) -- the same operator as the
#   rollout evaluation. Each call's cond uses the model's own previous state:
#   da_t = (x_hat_{k-1} - x_hat_{k-2}) / s_inc.
#   FULL_GRAD=1 (default): the gradient flows back through every intermediate
#   model evaluation (differentiable unroll, each evaluation activation-
#   checkpointed). FULL_GRAD=0: intermediate calls under no_grad (pushforward).
#   lambda: STEP_WEIGHTS = linear (lambda_k ~ k, default) | uniform | last |
#   comma list.
#
# Selection: `_best` = lowest rollout_eval rmse77 at the end of a ROLLOUT_CALLS
# hindcast (EMA weights, val split), NOT the single-step val loss.
#
# Launch: bash 2-SI_rollout.sh   (env overrides listed there)
# ===========================================================================
import contextlib
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
    H, W, N_STATE, SIData, SIWindowDataset, bind_call, masked_si_loss, rollout_step,
    build_model, build_si, loss_weights, load_weights, rollout_eval, rollout_png,
    set_precision, make_onecycle, onecycle_step, onecycle_fast_forward, env_int, env_float, env_bool,
)
from ocean_mask import OceanMask  # noqa: E402


# ================== hyperparameters (env-overridable) ==================
K = env_int('K', 8)
epochs = env_int('EPOCHS', 30)
batch_size = env_int('BATCH_SIZE', 4)              # per GPU (K=8 unroll, B=2: 8.7 GiB, 6.1 s/it)
grad_accum = env_int('GRAD_ACCUM', 1)
limit_windows = env_int('LIMIT_WINDOWS', 0) or None
num_workers = env_int('NUM_WORKERS', 6)
lr = env_float('LR', 1e-5)
min_lr = env_float('MIN_LR', 1e-7)
# OneCycle (per optimizer step): LR/DIV_FACTOR -> LR over PCT_START of the run, cosine down to MIN_LR
pct_start = env_float('PCT_START', 0.3)
div_factor = env_float('DIV_FACTOR', 25.0)
weight_decay = 0.05
grad_clip = 1.0
ema_decay = env_float('EMA_DECAY', 0.999)
precision = 'fp32'   # FORCED strict fp32 (user decision 2026-10-13); see si_common.set_precision
amp_dtype = set_precision(precision)
input_noise_rel = env_float('INPUT_NOISE_REL', 0.1)
full_grad = env_bool('FULL_GRAD', True)
step_weights_spec = os.environ.get('STEP_WEIGHTS', 'linear')
pf_ddim = env_int('ROLLOUT_DDIM_STEPS', 2)          # operator for the unroll AND the eval
inc_clamp = env_float('INC_CLAMP', 8.0)

# rollout evaluation / selection
rollout_every = env_int('ROLLOUT_EVERY', 1)
rollout_calls = env_int('ROLLOUT_CALLS', 146)       # 146 x 5 d = 2 yr
rollout_seed = 0
VAL_SEED = 20261013

init_ckpt = os.environ.get('INIT_CKPT', '/data/weidong/InMOS_1deg/ckpt/SI/SI1deg_step1_best_ema.pt')
ckpt_tag = os.environ.get('CKPT_TAG', f'SI1deg_rollout_K{K}')
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

# The model config and loss weighting come from the stage-1 run being fine-tuned.
_init_meta = None
for cand in (init_ckpt.replace('_ema.pt', '_meta.pt'), init_ckpt.rstrip('/') + '_meta.pt'):
    if os.path.exists(cand):
        _init_meta = torch.load(cand, map_location='cpu', weights_only=False)
        break
if _init_meta is None:
    raise SystemExit(f'no stage-1 meta next to INIT_CKPT={init_ckpt}')
cfg = _init_meta['cfg']
loss_w_mode = os.environ.get('LOSS_W_MODE', _init_meta.get('loss_w_mode', 'inc'))
cfg['grad_checkpoint'] = env_bool('GRAD_CKPT', True)

# ---- data ----
data = SIData()
train_ds = SIWindowDataset(data, 'train', K=K, limit=limit_windows)
val_ds = SIWindowDataset(data, 'val', K=1, limit=limit_windows)
loader_kw = dict(num_workers=num_workers, pin_memory=True,
                 persistent_workers=num_workers > 0,
                 prefetch_factor=2 if num_workers > 0 else None)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, drop_last=False, **loader_kw)

stride = 2 ** len(cfg['uvit_widths'])
ocean = OceanMask(patch_size=stride).to(device)
cell_mask = ocean.cell_mask
s_inc_f = torch.from_numpy(data.s_inc_floored).to(device)
s_inc_b = s_inc_f.view(1, N_STATE, 1, 1)
inv_s_da = 1.0 / s_inc_b
loss_w = loss_weights(ocean, s_inc_f, loss_w_mode)
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


optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

# ---- weights ----
# RESUME_FROM=best (default, user decision 2026-10-13): if this run has a _best
# (selected by the rollout rmse77), continue from it, rolling back later epochs
# (they are re-trained, and logged to a NEW wandb run); otherwise from the latest
# <tag>/; with no checkpoint of this run, warm start from stage 1 (INIT_CKPT).
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
        print(f'Warm start from {src} (fresh optimizer, EMA reseeded)')

model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
steps_per_epoch = math.ceil(len(train_loader) / grad_accum)   # optimizer steps per epoch (per rank)
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
    print(f'Device {device}, world {accelerator.num_processes} | K={K} windows {len(train_ds)} '
          f'| batch {batch_size}x{grad_accum}x{accelerator.num_processes} | full_grad={full_grad} '
          f'| weights {step_weights_spec} {np.round(step_weights(K), 3).tolist()} '
          f'| operator {pf_ddim}-step DDIM + clamp {inc_clamp} | loss_w_mode={loss_w_mode}')

if use_wandb:
    init = dict(entity=wandb_entity, name=wandb_name)
    if wandb_run_id:
        init.update(id=wandb_run_id, resume='must')
    accelerator.init_trackers(wandb_project, config=dict(
        stage=2, K=K, epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
        world=accelerator.num_processes, lr=lr, min_lr=min_lr, scheduler='onecycle', pct_start=pct_start, div_factor=div_factor, ema_decay=ema_decay,
        full_grad=full_grad, step_weights=step_weights_spec, ddim=pf_ddim,
        inc_clamp=inc_clamp, init_ckpt=init_ckpt, loss_w_mode=loss_w_mode,
        input_noise_rel=input_noise_rel,
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}),
        init_kwargs={'wandb': init})
    if is_main:
        wandb_run_id = accelerator.get_tracker('wandb', unwrap=True).id


def run_rollouts(epoch):
    mods = [('ema', ema_model.module)] if use_ema else []
    mods.append(('raw', raw_module))
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
    model.train()
    t_ep = time.time()
    run, n = 0.0, 0
    call_loss = np.zeros(K); call_n = np.zeros(K)
    pbar = tqdm(train_loader, disable=not is_main, desc=f'ep {epoch+1}')
    for x_prev, x_cur, ys, forcings, scalars in pbar:
        with accelerator.accumulate(model):
            if input_noise_rel > 0:
                with torch.no_grad():
                    x_prev = x_prev + input_noise_rel * si.noise_scales * si.get_noise(x_prev)
                    x_cur = x_cur + input_noise_rel * si.noise_scales * si.get_noise(x_cur)
            grad_ctx = contextlib.nullcontext() if full_grad else torch.no_grad()
            losses, bad = [], False
            for k in range(K):
                call = bind_call(model, scalars[:, k], x_prev, cell_mask, inv_s_da,
                                 amp_dtype=amp_dtype, ckpt=full_grad)
                l_k = masked_si_loss(si, call, x_cur, forcings[:, k], ys[:, k], loss_w)
                if not torch.isfinite(l_k):
                    bad = True
                    break
                losses.append(l_k)
                call_loss[k] += float(l_k.detach()); call_n[k] += 1
                if k == K - 1:
                    break
                with grad_ctx:
                    op = bind_call(model if full_grad else raw_module, scalars[:, k], x_prev,
                                   cell_mask, inv_s_da, amp_dtype=amp_dtype, ckpt=full_grad)
                    x_next = rollout_step(op, si, x_cur, forcings[:, k], pf_ddim, inc_clamp,
                                          s_inc_b, cell_mask)
                if not torch.isfinite(x_next).all():
                    bad = True
                    break
                x_prev, x_cur = x_cur, x_next
            # Non-finite on ANY rank -> skip on ALL ranks (see 1-SI_step1.py).
            if accelerator.reduce(torch.tensor([float(bad)], device=device), 'sum').item() > 0:
                if is_main:
                    print('  [warn] non-finite loss/state; step skipped on all ranks')
                optimizer.zero_grad()
                continue
            lam = step_weights(len(losses))
            loss = sum(w_ * l_ for w_, l_ in zip(lam, losses))
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
            pbar.set_postfix(loss=f'{lv:.4f}', run=f'{run/n:.4f}',
                             calls='/'.join(f'{float(l.detach()):.3f}' for l in losses))
            if use_wandb:
                accelerator.log({'train/loss': lv, 'train/lr': optimizer.param_groups[0]['lr'],
                                 'epoch': epoch + 1}, step=global_step)

    # ---- single-step val (raw + EMA), for reference only ----
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

    metric = None
    if is_main:
        dt = time.time() - t_ep
        print(f'Epoch {epoch+1}/{epochs} | train {run/max(n,1):.5f} | per-call SI loss '
              + ' '.join(f'l{k+1}={call_loss[k]/max(1, call_n[k]):.4f}' for k in range(K))
              + ' | ' + ' | '.join(f'val1[{k}] {v:.5f}' for k, v in val.items())
              + f' | {dt/60:.1f} min ({dt/max(n,1):.2f} s/it) | peak '
              f'{torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)
        log = {f'val/loss_{k}': v for k, v in val.items()}
        log.update({f'train/si_loss_call{k+1}': call_loss[k] / max(1, call_n[k]) for k in range(K)})
        if rollout_every and (epoch + 1) % rollout_every == 0:
            res = run_rollouts(epoch + 1)
            for name, r in res.items():
                log[f'rollout/rmse77_end_{name}'] = r['rmse77_end']
                log[f'rollout/rmse77_mean_{name}'] = r['rmse77_mean']
                log[f'rollout/sst_end_{name}'] = r['rmse_sst_end']
                log[f'rollout/maxabs_peak_{name}'] = r['maxabs_peak']
            metric = res['ema' if use_ema else 'raw']['rmse77_end']
        if use_wandb:
            accelerator.log(log, step=global_step)

    accelerator.wait_for_everyone()
    if is_main:
        meta = dict(best_metric=best_metric, global_step=global_step, epoch=epoch,
                    wandb_run_id=wandb_run_id, cfg=cfg, loss_w_mode=loss_w_mode, K=K)
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
            print(f'  best rollout rmse77 {best_metric:.4f} -> {ckpt_path_best}')
    accelerator.wait_for_everyone()

if use_wandb:
    accelerator.end_training()
