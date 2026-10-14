"""Bake a FiLM strength multiplier into a checkpoint, so inference needs no flag.

The FiLM heads' output layers are zero-initialized and the modulation is linear
in them (FiLMHead.forward: mod = mlp[-1](z_vec) + spatial(z_field), applied as
h*(1+gamma)+beta), so multiplying exactly those four tensors per head scales
gamma/beta by the same factor — this is an identity, not an approximation. It is
the same operation `scale_film()` performs in memory during a sweep, made
permanent so the shipped file already runs at the chosen strength.

Everything else in the state dict is copied bitwise, and the payload keeps
`film_config` / `z_field_hw` so the loading contract (see /home/nvidia/weidong/CRPS/README_FILM.md) is
unchanged. `film_scale_baked` records the cumulative factor; baking a second
time needs --force and multiplies on top of it (say so before you do).

    python 3-bake_film_scale.py --in-ckpt ckpt_latest_ema.pt --scale 1.25
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

# The zero-init output layers of every FiLM head — and only those.
OUT_LAYER = re.compile(r"\.heads\.\d+\.(mlp\.2|spatial)\.(weight|bias)$")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-ckpt", type=Path, required=True)
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--out-ckpt", type=Path, default=None)
    p.add_argument("--force", action="store_true",
                   help="allow baking on top of an already-baked checkpoint")
    args = p.parse_args()

    if args.scale <= 0:
        raise SystemExit("--scale must be > 0 (0 would erase the FiLM heads)")

    ck = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise SystemExit(f"{args.in_ckpt} has no 'model' payload")
    if ck.get("film_config") is None:
        raise SystemExit(f"{args.in_ckpt} has no film_config — not a FiLM checkpoint")
    prior = ck.get("film_scale_baked", 1.0)
    if prior != 1.0 and not args.force:
        raise SystemExit(f"{args.in_ckpt} is already baked at {prior}; pass --force "
                         f"to multiply again (result would be {prior * args.scale})")

    sd = ck["model"]
    keys = [k for k in sd if OUT_LAYER.search(k)]
    if not keys:
        raise SystemExit("no FiLM output-layer tensors matched — refusing to write "
                         "a checkpoint that would be identical to the input")
    out_sd = {}
    for k, v in sd.items():
        out_sd[k] = (v.to(torch.float32) * args.scale).to(v.dtype) \
            if k in set(keys) else v.clone()

    # 1deg: stage-3 output name, e.g. 3-FiLM_ship_it2199_s1.25.pt (input 2-FiLM_noise_it2199_ema.pt)
    src = args.in_ckpt.resolve() if args.in_ckpt.is_symlink() else args.in_ckpt
    tag = src.stem.replace("2-FiLM_noise_", "").replace("_ema", "")
    out = args.out_ckpt or args.in_ckpt.with_name(f"3-FiLM_ship_{tag}_s{args.scale:g}{args.in_ckpt.suffix}")
    payload = {**ck, "model": out_sd,
               "film_scale_baked": prior * args.scale,
               "film_scale_source": str(args.in_ckpt)}
    torch.save(payload, out)

    # Verify against what was written, not against what we intended to write.
    chk = torch.load(out, map_location="cpu", weights_only=False)["model"]
    bad = []
    for k, v in sd.items():
        w = chk[k]
        expect = (v.to(torch.float32) * args.scale).to(v.dtype) if k in set(keys) else v
        if not torch.equal(w, expect):
            bad.append(k)
    if bad:
        raise SystemExit(f"verification FAILED for {len(bad)} tensors, e.g. {bad[:3]}")

    n_scaled = len(keys)
    print(f"scaled {n_scaled} FiLM output-layer tensors by {args.scale:g}; "
          f"{len(sd) - n_scaled} other tensors copied bitwise")
    print(f"iter={ck.get('iter')}  film_config={ck['film_config']}  "
          f"cumulative film_scale_baked={payload['film_scale_baked']:g}")
    print(f"verified OK -> {out}")


if __name__ == "__main__":
    main()
