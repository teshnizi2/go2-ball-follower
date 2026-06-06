#!/usr/bin/env python3
"""Ablation stress test for the report.

At normal difficulty the system is collision-free even with single layers
removed, because the guaranteed lane + braking + yaw steering still carry it
over a short window (defense in depth). To isolate the value of the lateral +
gate channel we run at MAXIMUM difficulty (DIFFICULTY_BASE high, so every lane
sits at the 1.30 m floor from the start) and compare:

  full        - the validated stack
  no_lateral  - lateral vy strafe + hard gate + repulsion removed (yaw-only,
                like the original unicycle), brakes + planner still on
  no_avoid    - lateral + gate + brakes all removed (only the yaw chase remains)

For each config x seed we record the distance travelled before the first
obstacle/wall collision (from the printed [COLLISION-...] line); configs that
never collide run the whole window and are marked collision-free.

Writes report/data/ablation.json.
"""
import json, os, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
PY = os.path.join(SIM, ".venv312", "bin", "python3.12")
MAIN = os.path.join(SIM, "main.py")
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

SECONDS = "200"
SEEDS = [42, 7, 123]
CONFIGS = {
    "full":       {},
    "no_lateral": {"VY_LATERAL": "0"},
    "no_avoid":   {"VY_LATERAL": "0", "OBSTACLE_BRAKE": "0", "WALL_BRAKE": "0"},
}

# validated config (run.sh / best_config.json) + MAXIMUM difficulty from the start
BASE = dict(
    BALL_PATH_MODE="corridor", ACTION_SCALE_MULT="2.0", ROBOT_SAFETY="0.50",
    DETOUR_MARGIN="0.5", OBS_BRAKE_LATERAL="0.85",
    GO2_HEADLESS="1", GO2_HEADLESS_SECONDS=SECONDS, OBS_ROWS="50",
    DIFFICULTY_BASE="1.5",   # lanes at the 1.30 m floor, widest obstacles, from step 0
)

re_coll = re.compile(r"\[COLLISION-\w+\].*?sim_t=([\d.]+)s\s+rx=([-\d.]+)")
re_csv = re.compile(r"CSV saved to\s*:\s*(.+\.csv)\s*$", re.M)


def last_robot_x(csv_path):
    try:
        import csv as _csv
        last = None
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                last = row
        return float(last["robot_x"]) if last else float("nan")
    except Exception:
        return float("nan")


def main():
    results = []
    for cfg, over in CONFIGS.items():
        for seed in SEEDS:
            env = dict(os.environ)
            env.update(BASE)
            env.update(over)
            env["CORRIDOR_SEED"] = str(seed)
            print(f"[ablation] {cfg} seed={seed} ...", flush=True)
            p = subprocess.run([PY, MAIN], cwd=SIM, env=env,
                               capture_output=True, text=True)
            out = p.stdout + "\n" + p.stderr
            mc = re_coll.search(out)
            collided = mc is not None
            first_t = float(mc.group(1)) if collided else None
            first_x = float(mc.group(2)) if collided else None
            mcsv = re_csv.search(out)
            final_x = last_robot_x(mcsv.group(1)) if mcsv else float("nan")
            rec = dict(config=cfg, seed=seed, collided=collided,
                       first_t=first_t, first_x=first_x, final_x=final_x)
            results.append(rec)
            print("   ->", json.dumps(rec), flush=True)
    json.dump(results, open(os.path.join(DATA, "ablation.json"), "w"), indent=2)
    print("WROTE ablation.json", flush=True)


if __name__ == "__main__":
    main()
