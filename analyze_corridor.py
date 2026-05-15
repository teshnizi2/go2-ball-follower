#!/usr/bin/env python3
"""Analyze /tmp/sim_corridor.log — detect falls, freezes, obstacle hits."""
import re, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sim_corridor.log"
if not os.path.exists(path):
    print(f"MISSING {path}"); sys.exit(1)

data = open(path).read()
FELL = re.compile(r"\[WARN\]\s*Robot FELL")
falls = len(FELL.findall(data))

STEP_RE = re.compile(
    r"\[STEP\s*\d+\]\s*t=([\d.]+)s.*?pos=\(([+-]?[\d.]+),([+-]?[\d.]+)\).*?"
    r"ball=\(([+-]?[\d.]+),([+-]?[\d.]+)\)\s*dist_world=([\d.]+)m"
)
steps = STEP_RE.findall(data)

# Parse obstacle positions from the startup log (any "obstacles at: [...]" form)
OBS_RE = re.compile(
    r"obstacles at: \[(.*?)\]"
)
obs_match = OBS_RE.search(data)
obstacles = []
if obs_match:
    for m in re.finditer(r"\(([^,]+),\s*([^)]+)\)", obs_match.group(1)):
        obstacles.append((float(m.group(1)), float(m.group(2))))

# Obstacle collision check: robot box-body is ≈0.6×0.3m, obstacles are 0.4×1.0m
# at z=0.4.  Collision if |rx - ox| < 0.35 AND |ry - oy| < 0.55 (with margin).
def in_obstacle(rx, ry, ox, oy, margin=0.0):
    return abs(rx - ox) < (0.20 + 0.35 + margin) and \
           abs(ry - oy) < (0.50 + 0.15 + margin)

collision_steps = 0
collision_events = 0
in_coll = False
for s in steps:
    rx, ry = float(s[1]), float(s[2])
    collides = any(in_obstacle(rx, ry, ox, oy) for (ox, oy) in obstacles)
    if collides:
        collision_steps += 1
        if not in_coll:
            collision_events += 1
            in_coll = True
    else:
        in_coll = False

LOG_RE = re.compile(
    r"\[LOG\s*\d+\]\s*t=\s*([\d.]+)s.*?dist=([\d.]+)m.*?conf=([\d.]+)"
)
logs = LOG_RE.findall(data)

print(f"Obstacles: {obstacles}")
print(f"Falls: {falls}")
print(f"Obstacle-collision events: {collision_events}  (in_coll for {collision_steps}/{len(steps)} STEPs)")

if steps:
    rxs = [float(s[1]) for s in steps]
    bxs = [float(s[3]) for s in steps]
    ts  = [float(s[0]) for s in steps]
    max_rx = max(rxs)
    final_rx = rxs[-1]
    print(f"Robot: start_x={rxs[0]:.2f}  max_x={max_rx:.2f}  final_x={final_rx:.2f}  progressed={final_rx-rxs[0]:.2f} m")
    print(f"Ball : start_x={bxs[0]:.2f}  final_x={bxs[-1]:.2f}  progressed={bxs[-1]-bxs[0]:.2f} m")

    # Freeze: robot x didn't move > 0.1m for > 3s
    freeze_start = None
    last_moving_t = ts[0]
    last_x = rxs[0]
    for t, x in zip(ts, rxs):
        if abs(x - last_x) < 0.05:
            if t - last_moving_t > 3.0 and freeze_start is None:
                freeze_start = last_moving_t
        else:
            last_moving_t = t
            last_x = x
    if freeze_start is not None:
        print(f"Robot froze at t={freeze_start:.1f}s")
    else:
        print("Robot did NOT freeze")

if logs:
    dists = [float(l[1]) for l in logs]
    confs = [float(l[2]) for l in logs]
    n = len(dists)
    vis = sum(1 for c in confs if c > 0.5) / n
    near_band = sum(1 for d in dists if 0.7 <= d <= 1.3) / n
    within_15 = sum(1 for d in dists if d <= 1.5) / n
    print(f"vis%={vis*100:.1f}  near_band%={near_band*100:.1f}  within_1.5m%={within_15*100:.1f}")
