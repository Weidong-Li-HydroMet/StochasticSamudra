"""Find the FiLM strength that optimizes CRPS / RMSE / calibration.

Reads what `eval_film_candidates.py` already wrote — the JSON for the aggregate
metrics and the .log for the per-lead rows — and answers three separate
questions, because they have three different answers:

  * which scale minimizes CRPS        (probabilistic skill)
  * which scale minimizes RMSE        (ensemble-mean / deterministic skill)
  * which scale brings SSR to 1.0     (calibration), reported PER LEAD, since
                                      SSR falls with lead and one scalar cannot
                                      calibrate 10 d and 80 d at once

Grid values are reported as measured; the interpolated optimum is a quadratic
vertex through the three points bracketing the grid minimum, and the SSR=1
crossing is a linear interpolation between the bracketing pair. Both are
labelled as interpolated — they are not measured points.

No GPU, no torch: run it on a login node.

    python 3-analyze_scale_sweep.py --json .../_eval/eval_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROW = re.compile(
    r"\s+(?P<var>\S+)\s+lead=\s*(?P<lead>\d+) steps \(\s*(?P<days>\d+)d\)\s+"
    r"CRPS=(?P<crps>[\d.]+)\s+RMSE=(?P<rmse>[\d.]+)\s+spread=(?P<spread>[\d.]+)\s+"
    r"SSR=(?P<ssr>[\d.]+)"
)
HEADER = re.compile(r"\[val (?P<tag>.+?) @ iter \d+\]")


def tag_of(r: dict) -> str:
    """Reproduce the tag `eval_film_candidates.py` logged for this candidate."""
    s = r["film_scale"]
    return f"{r['label']} scale={s:g}" if s != 1.0 else r["label"]


def parse_log(path: Path) -> dict[str, dict[int, dict]]:
    """tag -> lead -> {'all': row, <var>: row}. Last block wins on duplicates."""
    out: dict[str, dict[int, dict]] = {}
    cur = None
    for line in path.read_text().splitlines():
        h = HEADER.search(line)
        if h:
            cur = h.group("tag")
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = ROW.search(line)
        if m:
            lead = int(m.group("lead"))
            out[cur].setdefault(lead, {})[m.group("var")] = {
                "days": int(m.group("days")),
                "crps": float(m.group("crps")), "rmse": float(m.group("rmse")),
                "spread": float(m.group("spread")), "ssr": float(m.group("ssr")),
            }
    return out


def vertex(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    """Quadratic vertex through 3 points; None unless it is a real minimum."""
    if len(xs) != 3:
        return None
    a, b, c = np.polyfit(xs, ys, 2)
    if a <= 0:
        return None
    xv = -b / (2 * a)
    if not (xs[0] <= xv <= xs[2]):
        return None
    return float(xv), float(a * xv * xv + b * xv + c)


def crossing(xs: list[float], ys: list[float], target: float = 1.0) -> float | None:
    """First x where y crosses `target`, linearly interpolated."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


def var_ssr_peak(blk: dict) -> tuple[float, str] | None:
    """Largest per-variable SSR in one candidate's block, and where.

    The pooled `all` SSR hides single-variable over-dispersion: run 1 sat at
    SSR_max 0.765 while thetao_0 had already overshot to 1.14, because averaging
    over 154 channels buried it. Only the 5 surface variables are logged, so
    this is a floor on the real worst case, not the worst case.
    """
    best = None
    for lead, row in blk.items():
        for var, m in row.items():
            if var == "all":
                continue
            if best is None or m["ssr"] > best[0]:
                best = (m["ssr"], f"{var}@{m['days']}d")
    return best


def report_metric(name: str, scales: list[float], vals: list[float],
                  lower_is_better: bool = True) -> None:
    arr = np.asarray(vals)
    i = int(arr.argmin() if lower_is_better else arr.argmax())
    print(f"  {name}: best on grid at s={scales[i]:g}  ({arr[i]:.5f})")
    if i in (0, len(scales) - 1):
        print(f"    !! the optimum sits at the EDGE of the grid — extend it "
              f"past s={scales[i]:g} before trusting this")
        return
    v = vertex(np.array(scales[i - 1:i + 2]), arr[i - 1:i + 2])
    if v:
        print(f"    interpolated optimum s={v[0]:.2f} ({v[1]:.5f}); the grid "
              f"neighbours are {scales[i-1]:g} and {scales[i+1]:g}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--log", type=Path, default=None,
                   help="defaults to the .log beside --json")
    p.add_argument("--label", default=None,
                   help="restrict to one candidate label (default: the one with "
                        "the most scale points)")
    args = p.parse_args()

    data = json.loads(args.json.read_text())
    log_path = args.log or args.json.with_suffix(".log")
    per_lead = parse_log(log_path) if log_path.exists() else {}
    if not per_lead:
        print(f"note: no per-lead rows ({log_path} missing) — SSR-by-lead skipped")

    rows = data["results"]
    ref = next((r for r in rows if r["path"] == "BASE"), None)
    sweep = [r for r in rows if r["path"] != "BASE"]
    if args.label:
        sweep = [r for r in sweep if r["label"] == args.label]
    else:
        counts: dict[str, int] = {}
        for r in sweep:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        if counts:
            pick = max(counts, key=lambda k: counts[k])
            if len(counts) > 1:
                print(f"note: {len(counts)} labels present, analyzing '{pick}' "
                      f"({counts[pick]} points); others: "
                      f"{', '.join(k for k in counts if k != pick)}")
            sweep = [r for r in sweep if r["label"] == pick]
    sweep.sort(key=lambda r: r["film_scale"])
    if len(sweep) < 2:
        raise SystemExit("need at least 2 scale points to sweep")

    label = sweep[0]["label"]
    print(f"\nFiLM strength sweep — {label}")
    print(f"  {data['n_ics']} ICs over {data['val_start']}..{data['val_end']}, "
          f"IC gap {data['ic_gap_windows']:.1f} windows, M={data['M']}, "
          f"leads to {data['val_steps']} steps, IC sigma={data['perturbation_level']}")
    if ref:
        print(f"  reference '{ref['label']}': CRPS={ref['crps']:.5f}  "
              f"RMSE={ref['rmse']:.5f}  SSRmax={ref['ssr_max']:.3f}")

    leads = sorted({ld for t in per_lead.values() for ld in t}) if per_lead else []
    head = f"\n  {'s':>5} {'CRPS':>9} {'RMSE':>9} {'SSRmax':>7}"
    for ld in leads:
        days = next(iter(per_lead.values()))[ld]["all"]["days"]
        head += f" {'SSR@' + str(days) + 'd':>8}"
    head += f" {'varSSRmax':>10}"
    if ref:
        head += f" {'dCRPS%':>8} {'dRMSE%':>8}"
    print(head)
    ssr_by_lead: dict[int, list[float]] = {ld: [] for ld in leads}
    var_peaks: list[tuple[float, str] | None] = []
    scales = [r["film_scale"] for r in sweep]
    for r in sweep:
        line = (f"  {r['film_scale']:>5g} {r['crps']:>9.5f} {r['rmse']:>9.5f} "
                f"{r['ssr_max']:>7.3f}")
        blk = per_lead.get(tag_of(r), {})
        for ld in leads:
            v = blk.get(ld, {}).get("all", {}).get("ssr")
            line += f" {v:>8.3f}" if v is not None else f" {'-':>8}"
            ssr_by_lead[ld].append(v if v is not None else float("nan"))
        peak = var_ssr_peak(blk) if blk else None
        var_peaks.append(peak)
        line += f" {peak[0]:>6.3f} {peak[1]:<12}" if peak else f" {'-':>10}"
        if ref:
            line += (f" {100 * (r['crps'] / ref['crps'] - 1):>+8.2f}"
                     f" {100 * (r['rmse'] / ref['rmse'] - 1):>+8.2f}")
        print(line)

    print("\noptima (interpolated values are fits, not measurements):")
    report_metric("CRPS", scales, [r["crps"] for r in sweep])
    report_metric("RMSE", scales, [r["rmse"] for r in sweep])

    print("\ncalibration — scale at which SSR reaches 1.0, per lead:")
    if not leads:
        print("  (no per-lead data)")
    for ld in leads:
        vals = ssr_by_lead[ld]
        days = next(iter(per_lead.values()))[ld]["all"]["days"]
        if any(np.isnan(vals)):
            print(f"  {days:>3}d: incomplete")
            continue
        x = crossing(scales, vals)
        if x is None:
            side = "above" if vals[-1] > 1.0 else "below"
            print(f"  {days:>3}d: never crosses 1.0 on this grid "
                  f"(stays {side}; SSR at s={scales[-1]:g} is {vals[-1]:.3f})")
        else:
            print(f"  {days:>3}d: s={x:.2f}")
    print("\nover-dispersion — first scale at which ANY logged variable "
          "passes SSR 1.0:")
    hit = next(((sc, pk) for sc, pk in zip(scales, var_peaks)
                if pk and pk[0] > 1.0), None)
    if hit:
        print(f"  s={hit[0]:g} ({hit[1][1]} at SSR {hit[1][0]:.3f}) — past this "
              f"the pooled CRPS can still improve\n  while individual variables "
              f"are already over-dispersed. Only 5 surface variables are logged.")
    elif any(var_peaks):
        worst = max((pk for pk in var_peaks if pk), key=lambda pk: pk[0])
        print(f"  none on this grid; the worst is {worst[1]} at SSR "
              f"{worst[0]:.3f}")
    else:
        print("  (no per-variable data)")

    print("\nOne scale cannot calibrate every lead: if the crossings spread out, "
          "SSR falls with lead\nand the missing ingredient is the TIME STRUCTURE "
          "of the noise, not its amplitude.\nPick the operating point from what "
          "the product is scored on, and state the RMSE cost.")


if __name__ == "__main__":
    main()
