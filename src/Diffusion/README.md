# 1° conditional diffusion: p(state_t | forcing_t, day-of-year)

This is a 1° port of `weidong/Diffusion` (the 0.25° `train_diffusion_025deg.py` and `DiffusionDate_noDropout.py`). The original files are unchanged.

It is **not a forecaster**. It generates the 77-channel ocean state from same-time forcing and the date. It is never given the previous ocean state.

| file | content |
|---|---|
| `dataloader.py` | **Data.** `OceanDiffDataset` returns `(state 77×180×360, forcing 3×180×360, doy/365)`. It reads `/data/weidong/InMOS_1deg/SI_data/{states,forcing}.npy` with mmap (written by `src/SI/0-prepare_data.py`: Samudra normalization, land 0), clips hfds to ±8 as the 0.25° cache did, and holds the train/val dates. |
| `model.py` | **Model.** `UNet_1degV1` is unchanged. `Diffusion_for_UNet` keeps only what training and the conditional DDIM sample use: the cosine logSNR schedule (λ ∈ ±20), v-prediction, and the DDIM update. |
| `train.py` | **Training.** Top-level script with no `main()`, divided into Config / Data / Model / Training. |
| `train.sh` | Launches with `accelerate`, log in `logs/train_<date>.log`. |

## Settings

- **Same as the original:**
  - Optimizer and loss: Adam, constant lr 1e-5, bf16 autocast, `MSE(v̂·mask, v·mask)`, batches with loss ≥ 10 are skipped.
  - Network: base width 128.
  - Picture: one 100-step DDIM SST sample per epoch.
- **Changed for 1°:**
  - Grid and network: 180×360, `c_mults=[1,2,2,4,8]`, `scale=[3,2,2]`, giving the same 15×30 bottleneck as the 0.25° net. 144.1M parameters.
  - Batch: 16 per GPU × 8 GPUs = 128 (the 0.25° run: 2 × 16 = 32).
  - Run length: 2000 epochs (the 0.25° run: 200).
  - Train dates: 1958-01-03 .. 2013-10-05, 4071 dates. The 0.25° script said 1975, but its cache check made it use 1958 too.
  - Val dates: 2013-10-10 .. 2022-12-29, 674 dates. The 0.25° split shared 2013-10-05 between train and val.
- **Fixes to the 0.25° training code (marked `1deg:` in `train.py`):**
  - A bad batch is skipped on all GPUs together. Before, one GPU skipping alone could hang DDP.
  - The best checkpoint is chosen by the val loss, averaged over all GPUs. Before, it used each GPU's own training loss, so the GPUs could disagree and hang inside `save_state`.
  - The val set is sharded across GPUs.
  - The DDIM picture is made on the main GPU only.
  - The resume metadata (epoch, step, best loss) is saved with the latest checkpoint, and the epoch counter resumes from it.
  - `gradient_accumulation_steps=4` is removed. It was never active, because the loop did not use `accelerator.accumulate`.
  - The input is kept in fp32, so x_t and the target v are exact; the UNet still runs in bf16.
  - `draw_save` passes the image to the colorbar.

## Run

```bash
cd /home/nvidia/weidong/InMOS_1degree/src/Diffusion
nohup setsid bash train.sh > /dev/null 2>&1 &               # GPUs 0-7
GPUS=0 EPOCHS=2 DEBUG_SAMPLES=16 CKPT_DIR=/tmp/dbg bash train.sh   # debug
```

Checkpoints go to `/data/weidong/InMOS_1deg/ckpt/Diffusion/condiff_1deg_date{,_best}`, each with its own `_meta.pt`. Running again resumes from **best** by default: it loads `_best` and continues from that epoch, re-training the epochs after it. `RESUME_FROM=latest` continues from the last epoch instead. The learning rate stays constant at 1e-5. Pictures go to `logs/img_1deg/`.

## Measured (GPU 0, shared with CRPS, 2026-10-14)

- Speed and memory per step: 0.23 s and 6.2 GiB at batch 4; 0.25 s and 10.8 GiB at batch 8; 0.42 s and 19.8 GiB at batch 16.
- At the defaults (batch 16, 8 GPUs) that is 31 steps per epoch, about 25 s per epoch with val and the DDIM picture. **2000 epochs ≈ 62k steps, about 14 h**, more when the GPUs are shared. Data reading may be the limit: about 320 MB per step per GPU.
