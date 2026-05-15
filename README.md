# Go2 Object Tracking — MuJoCo Simulation

Vision-only object following on the **Unitree Go2** quadruped, running in **MuJoCo 3.6** on macOS Apple Silicon.

## What it does

| Component | File | Description |
|-----------|------|-------------|
| Scene | `scene.xml` | Unitree Go2 + checkerboard arena + red target ball |
| Tracker | `tracker.py` | HSV color detection of the red ball in camera frames |
| Controller | `controller.py` | PD+I vision → `vx`, `vyaw` |
| Low-level | `low_level.py` | Pluggable torques (`TrotPDGait` default; `RawTorqueLowLevel` for policies) |
| Main loop | `main.py` | Vision, logging, target motion, physics |
| Baselines | `docs/` | Menagerie alignment + joint order vs LocoMuJoCo |
| Policy spike | `scripts/spike_external_policy.py` | HF SAC / LocoMuJoCo probe (optional deps) |

## Architecture

```
MuJoCo physics ──► head camera render ──► tracker.py
                                                │
                                          cx, cy, area
                                                │
                                       controller.py
                                                │
                                          vx, vyaw
                                                │
                              low_level.TrotPDGait (or swap backend)
                                                │
                              12 motor torques ──► MuJoCo physics
```

## Setup

```bash
# One-time: install dependencies (Python 3.12 venv in sim/)
cd sim
python3.12 -m venv .venv312
.venv312/bin/pip install -r requirements.txt
```

## Run

```bash
cd sim
./run.sh
# or
.venv312/bin/python3.12 main.py
```

**Headless benchmark** (no GUI, CSV + terminal summaries; exits after ~N seconds of *nominal* sim time):

```bash
env GO2_HEADLESS=1 GO2_HEADLESS_SECONDS=40 .venv312/bin/python3.12 main.py
```

Logs are written under `sim/logs/run_*.csv` with `detected`, `dist_ema`, and other columns for tuning.

## Controls (click the camera window first)

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `R` | Reset robot to home pose |
| `P` | Pause / resume physics |

## What you see

* **3-D MuJoCo viewer** – full robot + arena view with free camera rotation.
* **Head Camera window** – robot's POV; green circle = detected ball, HUD shows `vx` / `vyaw`.

## Levels of complexity

| Level | Status |
|-------|--------|
| 1 – Setup / environment | ✅ done |
| 2 – External control (keyboard / phone) | next step: map WASD keys to `vx`/`vyaw` overrides |
| 3 – Vision-only object follow | ✅ done (this simulation) |
| Stretch – Obstacle avoidance | add obstacle geoms + avoidance layer in `controller.py` |

## Sim-to-real path

* `tracker.py` is camera-agnostic — swap MuJoCo render for a real camera frame.
* `controller.py` is robot-agnostic — feed output to `unitree_sdk2_python` sport client.
* Only `main.py`'s physics loop needs to be replaced with real-robot comms.
