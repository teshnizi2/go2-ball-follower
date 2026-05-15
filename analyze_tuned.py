#!/usr/bin/env python3
"""Analyze /tmp/sim_tuned_*.log for tuned 4x config."""
import re, os

modes = ("circle", "square", "figure8")

LOG_RE = re.compile(
    r"\[LOG\s*\d+\]\s*t=\s*([\d.]+)s.*?dist=([\d.]+)m.*?conf=([\d.]+).*?vx=([-+][\d.]+)\s+vyaw=([-+][\d.]+)"
)
FELL_WARN = re.compile(r"\[WARN\]\s*Robot FELL")

for mode in modes:
    path = f"/tmp/sim_tuned_{mode}.log"
    if not os.path.exists(path):
        print(f"  {mode}: MISSING"); continue
    data = open(path).read()
    falls = len(FELL_WARN.findall(data))
    rows = LOG_RE.findall(data)
    dists = [float(r[1]) for r in rows]
    confs = [float(r[2]) for r in rows]
    vxs   = [float(r[3]) for r in rows]
    n = len(dists)
    if n == 0:
        print(f"  {mode}: 0 log rows"); continue
    mean_d = sum(dists) / n
    near  = sum(1 for d in dists if 0.7 <= d <= 1.3) / n
    close = sum(1 for d in dists if d < 0.7) / n
    far   = sum(1 for d in dists if d > 1.3) / n
    vis   = sum(1 for c in confs if c > 0.5) / n
    vx_on = sum(1 for v in vxs if abs(v) > 0.01) / n
    print(f"  {mode:7s}  falls={falls}  n={n:4d}  mean_d={mean_d:.2f}  "
          f"near%={near*100:4.1f}  close%={close*100:4.1f}  "
          f"far%={far*100:4.1f}  vis%={vis*100:4.1f}  vx%={vx_on*100:4.1f}")
