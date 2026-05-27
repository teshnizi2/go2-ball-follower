"""
self_tune.py — autonomous parameter tuner for the Go2 corridor sim.

Run once to find best config, or call from run.sh with --tune flag.
Saves result to best_config.json which run.sh and main.py load on startup.

Usage:
    python self_tune.py                    # tune from scratch
    python self_tune.py --seeds 3          # seeds per trial (default 3)
    python self_tune.py --rows 30          # obstacle rows to test
    python self_tune.py --quick            # fewer seeds for quick check
    python self_tune.py --from-collision   # called after in-sim collision

Parameter space:
    ACTION_SCALE_MULT   robot speed multiplier   [1.0 .. 2.5]
    ROBOT_SAFETY        clearance margin (m)      [0.15 .. 0.80]
    DETOUR_MARGIN       band-width margin (m)     [0.10 .. 0.60]
    OBS_BRAKE_LATERAL   lateral brake thresh (m)  [0.45 .. 1.00]
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time
from typing import Optional

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).parent.resolve()
PYTHON       = str(SCRIPT_DIR / ".venv312" / "bin" / "python3.12")
MAIN_PY      = str(SCRIPT_DIR / "main.py")
CONFIG_FILE  = SCRIPT_DIR / "best_config.json"

# ── search space bounds ───────────────────────────────────────────────────────
PARAM_BOUNDS = {
    "ACTION_SCALE_MULT":  (1.0, 2.5),
    "ROBOT_SAFETY":       (0.15, 0.80),
    "DETOUR_MARGIN":      (0.10, 0.60),
    "OBS_BRAKE_LATERAL":  (0.45, 1.00),
}

# Validated baselines from prior grid search
# Key: (A, S, D, L) → score note
KNOWN_WINNERS = [
    {"ACTION_SCALE_MULT": 2.0, "ROBOT_SAFETY": 0.75, "DETOUR_MARGIN": 0.50, "OBS_BRAKE_LATERAL": 0.85},  # fast, 3/3
    {"ACTION_SCALE_MULT": 1.6, "ROBOT_SAFETY": 0.45, "DETOUR_MARGIN": 0.30, "OBS_BRAKE_LATERAL": 0.65},  # safe, 3/3
]

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SEEDS  = 3
DEFAULT_ROWS   = 30
TIMEOUT_PER_RUN = 240   # seconds per headless trial (longer for bigger row counts)


def run_trial(cfg: dict, seed: int, rows: int, verbose: bool = False) -> dict:
    """Run one headless corridor trial. Returns {'pass': bool, 'rows': int, 'time': float}."""
    env = os.environ.copy()
    # Budget: roughly 5 s per row of obstacles at nominal speed (generous).
    budget_s = max(60, rows * 5)
    env.update({
        "GO2_HEADLESS":         "1",
        "GO2_HEADLESS_SECONDS": str(budget_s),
        "OBS_ROWS":             str(rows),
        "BALL_PATH_MODE":       "corridor",    # enable corridor obstacle mode
        "CORRIDOR_SEED":        str(seed),     # reproducible obstacle placement
        "ACTION_SCALE_MULT":    str(cfg["ACTION_SCALE_MULT"]),
        "ROBOT_SAFETY":         str(cfg["ROBOT_SAFETY"]),
        "DETOUR_MARGIN":        str(cfg["DETOUR_MARGIN"]),
        "OBS_BRAKE_LATERAL":    str(cfg["OBS_BRAKE_LATERAL"]),
        "COLLISION_STOPS_SIM":  "1",
    })
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [PYTHON, MAIN_PY],
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_RUN,
        )
        elapsed = time.perf_counter() - t0
        output  = result.stdout + result.stderr
        if verbose:
            print(output[-2000:])
        # parse [RESULT] line
        passed   = False
        rows_got = 0
        for line in output.splitlines():
            if line.startswith("[RESULT]"):
                passed   = "PASS" in line
                try:
                    rows_got = int(line.split("rows=")[1].split("/")[0])
                except Exception:
                    pass
                break
        return {"pass": passed, "rows": rows_got, "time": elapsed}
    except subprocess.TimeoutExpired:
        return {"pass": False, "rows": 0, "time": TIMEOUT_PER_RUN}
    except Exception as e:
        print(f"  [trial error] {e}")
        return {"pass": False, "rows": 0, "time": 0.0}


def evaluate(cfg: dict, seeds: list[int], rows: int) -> tuple[int, float]:
    """Returns (n_passed_seeds, avg_rows_when_passed)."""
    n_pass   = 0
    rows_sum = 0.0
    for s in seeds:
        r = run_trial(cfg, s, rows)
        tag = "PASS" if r["pass"] else f"FAIL(rows={r['rows']})"
        print(f"    seed={s:4d}  {tag}  t={r['time']:.1f}s")
        if r["pass"]:
            n_pass   += 1
            rows_sum += r["rows"]
    avg = rows_sum / max(n_pass, 1)
    return n_pass, avg


def perturb(cfg: dict, direction: str) -> dict:
    """Nudge config toward safer (direction='safe') or faster (direction='fast')."""
    c = dict(cfg)
    if direction == "safe":
        c["ACTION_SCALE_MULT"]  = max(PARAM_BOUNDS["ACTION_SCALE_MULT"][0],
                                       c["ACTION_SCALE_MULT"] - 0.2)
        c["ROBOT_SAFETY"]       = min(PARAM_BOUNDS["ROBOT_SAFETY"][1],
                                       c["ROBOT_SAFETY"] + 0.05)
        c["DETOUR_MARGIN"]      = min(PARAM_BOUNDS["DETOUR_MARGIN"][1],
                                       c["DETOUR_MARGIN"] + 0.05)
        c["OBS_BRAKE_LATERAL"]  = min(PARAM_BOUNDS["OBS_BRAKE_LATERAL"][1],
                                       c["OBS_BRAKE_LATERAL"] + 0.05)
    else:  # fast
        c["ACTION_SCALE_MULT"]  = min(PARAM_BOUNDS["ACTION_SCALE_MULT"][1],
                                       c["ACTION_SCALE_MULT"] + 0.2)
        c["ROBOT_SAFETY"]       = max(PARAM_BOUNDS["ROBOT_SAFETY"][0],
                                       c["ROBOT_SAFETY"] - 0.05)
        c["DETOUR_MARGIN"]      = max(PARAM_BOUNDS["DETOUR_MARGIN"][0],
                                       c["DETOUR_MARGIN"] - 0.05)
        c["OBS_BRAKE_LATERAL"]  = max(PARAM_BOUNDS["OBS_BRAKE_LATERAL"][0],
                                       c["OBS_BRAKE_LATERAL"] - 0.05)
    return c


def fmt(cfg: dict) -> str:
    return (f"A={cfg['ACTION_SCALE_MULT']:.2f} S={cfg['ROBOT_SAFETY']:.2f} "
            f"D={cfg['DETOUR_MARGIN']:.2f} L={cfg['OBS_BRAKE_LATERAL']:.2f}")


def load_saved() -> Optional[dict]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save(cfg: dict, meta: dict) -> None:
    payload = {"config": cfg, "meta": meta}
    with open(CONFIG_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[SAVED] {CONFIG_FILE}")


def tune(seeds: int = DEFAULT_SEEDS, rows: int = DEFAULT_ROWS,
         from_collision: bool = False) -> dict:
    rng = random.Random()
    seed_list = [rng.randint(0, 999999) for _ in range(seeds)]

    print(f"\n{'='*60}")
    print(f"  Go2 Self-Tuner  seeds={seeds}  rows={rows}  "
          f"{'[post-collision]' if from_collision else ''}")
    print(f"{'='*60}\n")

    # Start from saved best (if exists) or known winners
    saved = load_saved()
    candidates: list[dict] = []
    if saved:
        base = saved["config"]
        print(f"[RESUME] Starting from saved config: {fmt(base)}")
        candidates = [base] + KNOWN_WINNERS
    else:
        candidates = list(KNOWN_WINNERS)

    best_cfg   = None
    best_score = (-1, 0.0)   # (n_pass, avg_rows)

    for i, cfg in enumerate(candidates):
        print(f"\n-- Candidate {i+1}/{len(candidates)}: {fmt(cfg)}")
        n_pass, avg_rows = evaluate(cfg, seed_list, rows)
        score = (n_pass, avg_rows)
        print(f"   => {n_pass}/{seeds} passed  avg_rows={avg_rows:.1f}")
        if n_pass > best_score[0] or (n_pass == best_score[0] and avg_rows > best_score[1]):
            best_score = score
            best_cfg   = cfg

    # If best is not 3/3, try safer variations
    if best_cfg is not None and best_score[0] < seeds:
        print(f"\n[TUNE] Best so far {best_score[0]}/{seeds} — trying safer variants...")
        for _ in range(4):
            safer = perturb(best_cfg, "safe")
            if safer == best_cfg:
                break
            print(f"\n  Safer: {fmt(safer)}")
            n_pass, avg_rows = evaluate(safer, seed_list, rows)
            print(f"  => {n_pass}/{seeds} passed  avg_rows={avg_rows:.1f}")
            if n_pass >= best_score[0]:
                best_score = (n_pass, avg_rows)
                best_cfg   = safer
            if n_pass >= seeds:
                break

    # If best is 3/3, try faster variants
    if best_cfg is not None and best_score[0] >= seeds:
        print(f"\n[TUNE] All seeds pass — trying faster variants...")
        for _ in range(3):
            faster = perturb(best_cfg, "fast")
            if faster == best_cfg:
                break
            print(f"\n  Faster: {fmt(faster)}")
            n_pass, avg_rows = evaluate(faster, seed_list, rows)
            print(f"  => {n_pass}/{seeds} passed  avg_rows={avg_rows:.1f}")
            if n_pass >= seeds:
                best_cfg   = faster
                best_score = (n_pass, avg_rows)
            else:
                break   # don't go further — reliability dropped

    if best_cfg is None:
        best_cfg = KNOWN_WINNERS[1]   # safe fallback

    print(f"\n{'='*60}")
    print(f"  WINNER: {fmt(best_cfg)}")
    print(f"  Score:  {best_score[0]}/{seeds} seeds pass  avg_rows={best_score[1]:.1f}")
    print(f"{'='*60}\n")

    save(best_cfg, {
        "seeds_tested": seeds,
        "rows":         rows,
        "score":        f"{best_score[0]}/{seeds}",
        "avg_rows":     round(best_score[1], 1),
        "from_collision": from_collision,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return best_cfg


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",          type=int,  default=DEFAULT_SEEDS)
    ap.add_argument("--rows",           type=int,  default=DEFAULT_ROWS)
    ap.add_argument("--quick",          action="store_true",
                    help="Use 2 seeds and 20 rows for a quick check")
    ap.add_argument("--from-collision", action="store_true",
                    help="Called after in-sim collision — try safer configs first")
    args = ap.parse_args()

    if args.quick:
        args.seeds = 2
        args.rows  = 20

    tune(seeds=args.seeds, rows=args.rows, from_collision=args.from_collision)


if __name__ == "__main__":
    main()
