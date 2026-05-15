#!/usr/bin/env python3
"""Parse /tmp/sim_bench_*.log and print per-mode stats."""
import re
import sys

modes = ("circle", "square", "figure8")

# Matches the LOG lines:
#   [LOG   2275] t=   4.9s | z=0.253m | dist=1.19m(ema=1.21)[FAR  ] | cx= 208 area= 2401px² conf=1.00 | vx=+0.387 vyaw=+0.000
# or the FELL variant:
#   [LOG   1625] t=  32.5s  *** FELL *** | z=0.063m | dist=2.09m(ema=1.99)[FAR  ] | cx= 288 area=  650px² conf=1.00 | vx=+0.150 vyaw=+0.000
LOG_RE = re.compile(
    r"\[LOG\s*\d+\]\s*t=\s*([\d.]+)s.*?dist=([\d.]+)m.*?conf=([\d.]+).*?vx=([-+][\d.]+)\s+vyaw=([-+][\d.]+)"
)
FELL_WARN = re.compile(r"\[WARN\]\s*Robot FELL")

for mode in modes:
    path = f"/tmp/sim_bench_{mode}.log"
    try:
        data = open(path, "r", errors="ignore").read()
    except FileNotFoundError:
        print(f"--- {mode}: LOG MISSING ---")
        continue

    # Count distinct FELL events via WARN lines (one per fall event)
    falls = len(FELL_WARN.findall(data))

    rows = LOG_RE.findall(data)
    dists = [float(r[1]) for r in rows]
    confs = [float(r[2]) for r in rows]
    vxs   = [float(r[3]) for r in rows]

    n = len(dists)
    if n == 0:
        print(f"--- {mode}: no log rows parsed ---")
        continue

    mean_d = sum(dists) / n
    near_goal = sum(1 for d in dists if 0.7 <= d <= 1.3) / n
    far       = sum(1 for d in dists if d > 1.3) / n
    very_near = sum(1 for d in dists if d < 0.7) / n
    vis       = sum(1 for c in confs if c > 0.5) / n
    vx_active = sum(1 for v in vxs if abs(v) > 0.01) / n

    print(f"--- {mode} ---")
    print(f"  falls:        {falls}")
    print(f"  n log rows:   {n}")
    print(f"  mean dist:    {mean_d:.2f}m")
    print(f"  near-goal %:  {near_goal*100:.1f}  (0.7≤d≤1.3)")
    print(f"  too-close %:  {very_near*100:.1f}  (d<0.7)")
    print(f"  too-far %:    {far*100:.1f}   (d>1.3)")
    print(f"  visibility %: {vis*100:.1f}")
    print(f"  vx-active %:  {vx_active*100:.1f}")
    print()
