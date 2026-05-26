# Go2 Ball-Follower with Reactive Obstacle Avoidance

> Final project for **Robotics 2026** at **LIACS, Leiden University**.
> A MuJoCo simulation in which a **Unitree Go2** quadruped uses **only its head-mounted
> RGB camera** to chase a moving red ball through a corridor cluttered with obstacles.

**Team Ava** — Mohammadreza AhmadiTeshnizi · Patrik Perčinić

[![status](https://img.shields.io/badge/status-demo--ready-success)]()  
**Demo video:** see the technical report or `~/Desktop/go2_demo_1min.mp4` after running `record_corridor.command`.  
**Technical report PDF:** `~/Desktop/go2_technical_report.pdf` (also re-buildable from `/tmp/report/report.html`).

---

## 1. What the system does

The robot follows a **moving red ball** through a 5-m-wide corridor full of obstacle rows.
It does this **without** GPS, lidar, depth sensors, or any pre-built map — only the head
camera.

We compose three layers around a **pre-trained RL locomotion policy**:

| Layer | What we built | Key files |
|------|---------------|-----------|
| 🎯 **Perception** | Dual-window HSV mask + CamShift adaptive ROI; confidence-based reset | `tracker.py` |
| 🧭 **Reactive planning** | Free-band passage picker with commit-and-hold hysteresis; pure-pursuit controller; 4-layer safety stack | `controller.py`, `main.py` (planner block ~L2400) |
| 🐾 **Locomotion** | **Pre-trained** PPO velocity-tracking policy (`model_500.pt`); we do NOT train it | `low_level.py`, `policy/model_500.pt` |
| 🖥️ **Visualisation** | Live 720×720 cv2 dashboard: chase + head-cam + 2D map + telemetry with status LED & sparkline | `main.py` (HUD ~L1100) |
| 🎬 **Curriculum** | Stage advances every 20 s of sim time; ball gains lateral/vertical motion; obstacles shift from orange → black | `main.py` (mover class) |

---

## 2. Quick start

### Prerequisites
- **macOS** (Apple Silicon recommended) or Linux.
- **Python 3.12** with `mujoco`, `torch`, `numpy`, `opencv-python`.
- **ffmpeg** in `PATH` for recording.

### Install
```bash
git clone https://github.com/teshnizi2/go2-ball-follower.git
cd go2-ball-follower
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

### Run (one-liner)
```bash
./run.sh
```

Or run the full demo configuration explicitly:
```bash
GO2_HEADLESS=0 GUI=1 GO2_GUI_VIEW=chase \
  SIM_SPEED=20.0 RENDER_SKIP=5 STAGE_SECONDS=20.0 \
  BALL_PATH_MODE=corridor \
  OBS_ROWS=80 OBS_MIN_GAP=8.0 OBS_MIN_X=4.0 OBS_MAX_X=650.0 \
  MAX_OBS_HALF=0.55 CORRIDOR_SEED=100 \
  ACTION_SCALE_MULT=1.4 ROBOT_SAFETY=0.45 DETOUR_MARGIN=0.30 OBS_BRAKE_LATERAL=0.65 \
  ADAPTIVE_SPEED=1 OBSTACLE_BRAKE=1 WALL_BRAKE=1 SEARCH_AFTER_STEPS=30 \
  ./.venv312/bin/python3.12 -u main.py
```

A 720×720 window will open showing the live dashboard.  
Keys: **Q / Esc** = quit · **R** = reset · **P** = pause.

### Record a video
```bash
./record_corridor.command       # 60-sec recording → ~/Desktop/go2_demo_1min.mp4
```

---

## 3. Configuration knobs

| Env var | Default | Purpose |
|---------|---------|---------|
| `SIM_SPEED` | 1.0 | Wall-clock scaling (e.g. 20 = 20× faster playback) |
| `RENDER_SKIP` | 1 | Render every Nth control step; lowers CPU at the cost of vision rate |
| `STAGE_SECONDS` | 20.0 | Seconds between stage advances |
| `CORRIDOR_SEED` | random | Fix the obstacle layout |
| `OBS_ROWS` | 80 | Number of obstacle rows (1 row = 2 obstacles) |
| `OBS_MIN_GAP` | 8.0 | Minimum spacing between rows (m) |
| `MAX_OBS_HALF` | 0.55 | Cap on obstacle half-width (m) — lower = more reliable passages |
| `ACTION_SCALE_MULT` | 1.4 | Policy action gain (>2.5 collapses the gait) |
| `ROBOT_SAFETY` | 0.45 | Halo (m) the planner adds around each obstacle |
| `DETOUR_LOOKAHEAD` | 4.5 | How far ahead (m) the planner looks for blockers |
| `OBSTACLE_BRAKE` / `WALL_BRAKE` | 1 | Toggle the reactive brake layers |

The full list lives in the env-var grep at the top of `main.py`.

---

## 4. Repository layout

```
sim/
├── main.py             ─ control loop, planner, dashboard, recording
├── controller.py       ─ high-level pure-pursuit, dead-zone caps, brakes
├── tracker.py          ─ HSV + CamShift ball tracker
├── low_level.py        ─ MuJoCo policy interface (loads model_500.pt)
├── logger.py           ─ CSV telemetry log
├── joint_order.py      ─ Menagerie ↔ training-checkpoint joint remap
├── scene.xml           ─ MuJoCo scene: corridor, walls, 200 mocap obstacles
├── assets/             ─ Go2 mesh + materials (Menagerie)
├── policy/
│   └── model_500.pt    ─ Pre-trained PPO checkpoint (RSL-RL)
├── scripts/            ─ Headless validation + benchmark scripts
├── docs/               ─ Joint-order notes & Menagerie alignment
└── tests/              ─ Unit tests for the controller
```

---

## 5. Results (TL;DR)

| Metric | Value |
|--------|-------|
| Best clean-sim time | ~386 s on `CORRIDOR_SEED=100` (24 obstacles cleared) |
| Avg forward speed | ~0.5 m/s (policy-limited) |
| Physics step rate | ~5,700 Hz (after scene trim to 200 mocaps) |
| Stuck-recovery success rate | ~60 % before fall-through to FAIL |

Failure mode at later rows: ~0.25 m **lateral tracking error** of the un-fine-tuned RL
policy — it can plan a passage cleanly but cannot hit the centreline closely enough on
later rows. See the technical report for a detailed discussion.

---

## 6. What we did NOT do

To be transparent:

- We **did not train** the locomotion policy. We use `model_500.pt`, a pre-trained PPO
  checkpoint from the [RSL-RL](https://github.com/leggedrobotics/rsl_rl) framework
  (`unitree-go2-velocity-flat` variant). Everything around it (perception, planner,
  safety, dashboard, curriculum) is ours.
- The vision is HSV + CamShift — not a CNN detector.
- Evaluation is single-seed-anecdotal, not a full 50-seed sweep.

These are explicit future-work items in the technical report.

---

## 7. Citation / acknowledgements

If you use this code, please cite the relevant background:

- Hwangbo et al., *Learning agile and dynamic motor skills for legged robots.* Science Robotics, 2019.
- Rudin et al., *Learning to walk in minutes using massively parallel deep RL.* CoRL, 2021. *(source of our policy)*
- Todorov, Erez, Tassa, *MuJoCo: A physics engine for model-based control.* IROS, 2012.

Mesh & robot URDF come from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) Go2 model.

---

## 8. License

MIT for our code. Pre-trained policy weights and Go2 meshes inherit upstream licenses.
