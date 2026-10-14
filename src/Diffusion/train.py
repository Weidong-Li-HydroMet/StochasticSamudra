"""
Training for the 1-degree conditional diffusion model  p(state_t | forcing_t, day-of-year).

1-degree port of weidong/Diffusion/train_diffusion_025deg.py. The training step is the same:
    u ~ U(0,1);  x_t = alpha(u) x + sigma(u) eps;  v = alpha eps - sigma x
    v_hat = UNet([x_t * mask, forcing], u, datefrac);  loss = MSE(v_hat * mask, v * mask)
Changes from the 0.25-degree script are marked "1deg:".

Launch with train.sh (accelerate, one process per GPU).
Environment overrides: RESUME_FROM (best = default | latest), EPOCHS, BATCH,
DEBUG_SAMPLES (use only the first N train/val dates), CKPT_DIR.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from accelerate import Accelerator

from dataloader import OceanDiffDataset, load_prog_mask, N_PROG, N_BOUND, H, W, \
    TRAIN_START, TRAIN_END, VAL_START, VAL_END
from model import UNet_1degV1, Diffusion_for_UNet

# =====================================================================================
# Config
# =====================================================================================
cc = N_PROG                  # 77 state channels (model output)
cc_in = N_PROG + N_BOUND     # 77 + 3 = 80 (noisy_state + forcing concatenated)
channel = 128
lrate = 1e-5
batch_size = int(os.environ.get("BATCH", 16))  # 1deg: per GPU (16 x 8 GPUs = 128; the 0.25 run used 2 x 16 GPUs = 32)
epochs = int(os.environ.get("EPOCHS", 2000))
num_worker = 4
debug_samples = int(os.environ.get("DEBUG_SAMPLES", 0))
sample_steps = 100           # DDIM steps for the per-epoch picture

# 1deg: 180x360 grid. c_mults=[1,2,2,4,8], scale=[3,2,2]
#   H path: 180 -> 60 -> 30 -> 15 -> conv15
#   W path: 360 -> 120 -> 60 -> 30 -> conv30
# (the same 15x30 bottleneck as the 0.25-degree net: 720 -> 240 -> 120 -> 60 -> 30 -> 15)
c_mults = [1, 2, 2, 4, 8]
scale = [3, 2, 2]

CKPT_DIR = os.environ.get("CKPT_DIR", "/data/weidong/InMOS_1deg/ckpt/Diffusion")
model_path = os.path.join(CKPT_DIR, "condiff_1deg_date")              # latest (every epoch)
model_path_best = os.path.join(CKPT_DIR, "condiff_1deg_date_best")    # best val loss
meta_path = model_path + "_meta.pt"
meta_path_best = model_path_best + "_meta.pt"
# 1deg: resume from the best checkpoint (default; the epochs after it are re-trained) or the latest
resume_from = os.environ.get("RESUME_FROM", "best")
assert resume_from in ("best", "latest"), f"RESUME_FROM must be best|latest, got {resume_from}"
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "img_1deg")


def draw_save(data, mask=None, save_name=None, save_dir=IMG_DIR):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    im = plt.imshow(np.squeeze(data), interpolation='none', origin='lower')   # 1deg: row 0 is the south pole
    if mask is not None:
        plt.contour(np.squeeze(mask), levels=[0.5], colors='k', linewidths=1)
    plt.colorbar(im)
    plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')
    plt.close()


# 1deg: gradient_accumulation_steps=4 is dropped; the 0.25 loop never used accelerator.accumulate,
# so it did not accumulate either.
accelerator = Accelerator(mixed_precision="bf16")
device = accelerator.device
os.makedirs(CKPT_DIR, exist_ok=True)

# =====================================================================================
# Data
# =====================================================================================
prog_mask = load_prog_mask()

train_dataset = OceanDiffDataset(TRAIN_START, TRAIN_END)
val_dataset = OceanDiffDataset(VAL_START, VAL_END)
if debug_samples > 0:
    train_dataset = Subset(train_dataset, range(debug_samples))
    val_dataset = Subset(val_dataset, range(debug_samples))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_worker, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_worker, pin_memory=True,
)

# =====================================================================================
# Model
# =====================================================================================
# dim_in = N_PROG + N_BOUND = 80 (noisy_state + forcing concatenated)
# dim_out = N_PROG = 77 (predict v for the state only)
size = [cc, H, W]
model = UNet_1degV1(cc_in, cc, channel, c_mults, scale=scale)
optimizer = optim.Adam(model.parameters(), lr=lrate)
mse = nn.MSELoss()

diffusion = Diffusion_for_UNet(size=size, lambda_range=20, steps=1000, device=device)

# 1deg: val_loader is prepared too, so each GPU evaluates its own share of the val dates
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader,
)
accelerator.print(f"UNet_1degV1: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
                  f"{accelerator.num_processes} GPUs x batch {batch_size}, "
                  f"{len(train_loader)} steps/epoch")

# Load checkpoint (best by default, see RESUME_FROM) and its training metadata
start_epoch = 0
global_step = 0
best_loss = float("inf")
if resume_from == "best" and os.path.exists(meta_path_best):
    load_path, load_meta = model_path_best, meta_path_best
elif os.path.exists(meta_path):
    load_path, load_meta = model_path, meta_path        # RESUME_FROM=latest, or no best yet
else:
    load_path = None
if load_path is not None:
    accelerator.print(f"Loading checkpoint from {load_path} ...")
    accelerator.load_state(load_path)
    # Override lr from checkpoint with current `lrate`
    for param_group in optimizer.param_groups:
        param_group["lr"] = lrate
    # 1deg: each checkpoint has its own metadata (the 0.25 code always read the _best one)
    meta = torch.load(load_meta, map_location="cpu", weights_only=True)
    start_epoch = meta["epoch"]
    global_step = meta["global_step"]
    best_loss = meta["best_loss"]
    accelerator.print(f"Restored meta: epoch={start_epoch}, global_step={global_step}, best_loss={best_loss:.6f}")

prog_mask_gpu = torch.from_numpy(prog_mask[np.newaxis, ]).to(device=device)

# =====================================================================================
# Training
# =====================================================================================
for epoch in range(start_epoch, epochs):
    accelerator.print(f"Epoch {epoch+1}, lr: {optimizer.param_groups[0]['lr']}")

    # ---------- Train ----------
    model.train()
    loss_running, n_used = 0.0, 0
    pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
    for i, (state, forcing, date_frac) in enumerate(pbar):
        # 1deg: state/forcing stay fp32 (the 0.25 code cast them to bf16 first); autocast
        # still runs the UNet in bf16, but x_t and the target v are exact.
        # Noise the state only
        t = diffusion.sample_timesteps(state.shape[0])
        x_t, noise, v = diffusion.noise_images(state, t)

        # Concatenate [noisy_state, forcing] -> model input
        x_t *= prog_mask_gpu
        x_in = torch.cat([x_t, forcing], dim=1)  # [B, N_PROG+N_BOUND, H, W]

        predicted_v = model(x_in, t, date_frac=date_frac)
        predicted_v *= prog_mask_gpu
        v *= prog_mask_gpu

        loss = mse(predicted_v, v)

        # 1deg: skip a bad batch on ALL GPUs together. In the 0.25 code each GPU decided alone,
        # and one GPU skipping backward while the others wait in the gradient all-reduce hangs DDP.
        bad = (~torch.isfinite(loss.detach())) | (loss.detach() >= 10.0)
        bad = accelerator.reduce(bad.float(), reduction="sum").item() > 0
        if not bad:
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            global_step += 1
            n_used += 1
            loss_running += (loss.item() - loss_running) / n_used
            pbar.set_postfix(
                Epoch=epoch,
                MSE=loss.item(),
                MSE_running=loss_running,
                step=global_step,
            )
        else:
            accelerator.print(f"  skipped step {i}: loss {loss.item():.4g} on this GPU")

    # ---------- Val ----------
    model.eval()
    val_loss_sum, val_cnt = 0.0, 0
    with torch.no_grad():
        for state, forcing, date_frac in val_loader:
            t = diffusion.sample_timesteps(state.shape[0])
            x_t, noise, v = diffusion.noise_images(state, t)

            x_t *= prog_mask_gpu
            x_in = torch.cat([x_t, forcing], dim=1)
            predicted_v = model(x_in, t, date_frac=date_frac)

            predicted_v *= prog_mask_gpu
            v *= prog_mask_gpu
            val_loss_sum += mse(predicted_v, v).item()
            val_cnt += 1
    # 1deg: average over all GPUs, so every GPU takes the same "best" decision below
    val_stats = accelerator.reduce(torch.tensor([val_loss_sum, val_cnt], device=device), reduction="sum")
    val_loss = (val_stats[0] / val_stats[1].clamp(min=1)).item()
    train_stats = accelerator.reduce(torch.tensor([loss_running], device=device), reduction="mean")
    loss_running = train_stats.item()

    # Use DDIM to generate one sample conditioned on the last val forcing (main GPU only)
    if accelerator.is_main_process:
        with torch.no_grad(), accelerator.autocast():
            raw_model = accelerator.unwrap_model(model)
            sample = diffusion.sample_ddim_conditional(
                model=raw_model, n=1, cond=forcing[:1],
                mask=prog_mask_gpu, steps=sample_steps, cc=cc,
                visual_bar=False, date_frac=date_frac[:1],
            )
        # channel 38 = thetao_0 (SST); truth next to the sample
        draw_save(sample.float().cpu().numpy()[0, 38], mask=prog_mask[38], save_name=f'SST_epoch_{epoch+1}.png')
        draw_save(state.float().cpu().numpy()[0, 38], mask=prog_mask[38], save_name=f'SST_epoch_{epoch+1}_truth.png')

    accelerator.print(f"  Epoch {epoch+1} | train_running: {loss_running:.6f} "
                      f"| val: {val_loss:.6f} | step {global_step}")

    # ---------- Save ----------
    # 1deg: best is chosen by the val loss (the 0.25 code used the train running loss)
    if val_loss < best_loss:
        best_loss = val_loss
        accelerator.save_state(model_path_best)
        if accelerator.is_main_process:
            torch.save({"epoch": epoch + 1, "global_step": global_step, "best_loss": best_loss}, meta_path_best)
        accelerator.print(f"Best val loss: {best_loss:.6f} (saved to {model_path_best})")

    accelerator.save_state(model_path)
    if accelerator.is_main_process:
        torch.save({"epoch": epoch + 1, "global_step": global_step, "best_loss": best_loss}, meta_path)
    accelerator.wait_for_everyone()
