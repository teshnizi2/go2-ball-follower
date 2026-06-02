#!/usr/bin/env python3
"""
scan_seeds.py — headless seed scanner for corridor recording.
Tests N seeds in parallel; scores by stuck events, falls, rows passed.

Usage:
    python scan_seeds.py [--seeds 1 42 100 ...] [--parallel 8] [--sim-sec 350]
"""
import argparse
import subprocess
import os
import sys
import time
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON  = os.path.join(SIM_DIR, ".venv312", "bin", "python")


def run_seed(seed: int, sim_seconds: float) -> dict:
    env = {
        **os.environ,
        "GO2_HEADLESS": "1",
        "GO2_HEADLESS_SECONDS": str(sim_seconds),
        "GO2_SHADOWS": "0",
        "GO2_NO_TORCH": "1",          # skip Metal shader compilation — fast headless
        "BALL_PATH_MODE": "corridor",
        "COLLISION_STOPS_SIM": "0",
        "ACTION_SCALE_MULT": "2.0",
        "ROBOT_SAFETY": "1.125",
        "DETOUR_MARGIN": "0.75",
        "OBS_BRAKE_LATERAL": "1.275",
        "CORRIDOR_SEED": str(seed),
        "OBS_ROWS": "50",
        "OBS_MIN_GAP": "5.4",
        "OBS_MIN_X": "4.0",
        "OBS_MAX_X": "275.0",
        "BALL_CORRIDOR_SPEED": "0.111",
        "BALL_MAX_LEAD": "4.0",
        "BALL_LANE_DWELL": "40.0",
        "RENDER_SKIP": "13",
        "PYTHONUNBUFFERED": "1",
    }
    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, "main.py"],
            cwd=SIM_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=sim_seconds * 3 + 60,   # generous wall-clock budget
        )
        out = result.stdout + result.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
    elapsed = time.time() - t0

    stuck = len(re.findall(r"\[STUCK-FAIL\]", out))
    falls = len(re.findall(r"\[WARN\] Robot FELL", out))
    walls = len(re.findall(r"\[COLLISION-WALL\]", out))
    m = re.search(r"\[RESULT\].*?rows=(\d+)/(\d+)", out)
    rows_passed, rows_total = (int(m.group(1)), int(m.group(2))) if m else (0, 50)

    xm = re.findall(r"\[STEP\s+\d+\] t=.*?pos=\(\+?(-?[\d.]+)", out)
    final_x = float(xm[-1]) if xm else 0.0

    score = int(final_x * 5) + rows_passed * 10 - stuck * 5 - falls * 8 - walls * 10
    return {
        "seed": seed,
        "stuck": stuck,
        "falls": falls,
        "walls": walls,
        "final_x": final_x,
        "rows_passed": rows_passed,
        "rows_total": rows_total,
        "score": score,
        "elapsed_wall": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[1, 13, 42, 77, 100, 123, 150, 200, 250, 300,
                                 321, 400, 500, 555, 600, 700, 800, 900, 999, 1234])
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--sim-sec", type=float, default=350.0,
                        help="sim seconds to run per seed (~350 s ≈ 45 s of video)")
    args = parser.parse_args()

    print(f"Testing {len(args.seeds)} seeds × {args.sim_sec:.0f} sim-sec each, "
          f"{args.parallel} in parallel", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print("-" * 70)

    results = []
    with ProcessPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_seed, s, args.sim_sec): s for s in args.seeds}
        for fut in as_completed(futures):
            r = fut.result()
            tag = "CLEAN" if r["stuck"] == 0 and r["falls"] == 0 and r["walls"] == 0 else (
                  "OK" if r["stuck"] <= 1 and r["falls"] == 0 and r["walls"] == 0 else "BAD")
            print(f"[{tag}] seed={r['seed']:5d}  stuck={r['stuck']}  "
                  f"falls={r['falls']}  walls={r['walls']}  "
                  f"rows={r['rows_passed']}/{r['rows_total']}  "
                  f"x={r['final_x']:.1f}  score={r['score']:4d}  wall={r['elapsed_wall']:.0f}s",
                  flush=True)
            results.append(r)

    results.sort(key=lambda x: -x["score"])
    print("\n" + "=" * 70)
    print("RANKING (best first):")
    for i, r in enumerate(results[:10]):
        tag = "CLEAN" if r["stuck"] == 0 and r["falls"] == 0 and r["walls"] == 0 else (
              "OK" if r["stuck"] <= 1 and r["falls"] == 0 and r["walls"] == 0 else "BAD")
        print(f"  #{i+1:2d} [{tag}] seed={r['seed']:5d}  stuck={r['stuck']}  "
              f"falls={r['falls']}  walls={r['walls']}  "
              f"x={r['final_x']:.1f}  rows={r['rows_passed']}/{r['rows_total']}  "
              f"score={r['score']}")

    best = results[0]
    print(f"\nBEST SEED: {best['seed']}  "
          f"(stuck={best['stuck']}, falls={best['falls']}, rows={best['rows_passed']})")


if __name__ == "__main__":
    main()
