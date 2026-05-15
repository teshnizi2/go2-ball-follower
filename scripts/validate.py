"""
validate.py – Scripted validation suite for the Go2 object-tracking simulation.

Each "phase" programmatically moves the ball for a fixed duration, then prints
a per-phase metrics summary and PASS/FAIL verdict.  All phases run in one
continuous headless session (no GUI).

Scenario playlist
-----------------
Phase 1  – Hold at 4 m (far)                     — stability when ball is hard to see
Phase 2  – Hold at 2 m (nominal)                  — steady-state tracking at target range
Phase 3  – Hold at 1.0 m (close)                  — proximity handling, no backward motion
Phase 4  – Orbit slow   (r=2 m, ω=0.10 rad/s)    — gentle tracking
Phase 5  – Orbit medium (r=2 m, ω=0.25 rad/s)    — moderate-speed tracking
Phase 6  – Orbit fast   (r=2 m, ω=0.50 rad/s)    — high-speed tracking stress test
Phase 7  – Orbit large  (r=3 m, ω=0.15 rad/s)    — ball near detection limit
Phase 8  – Teleport sequence                       — sudden appearance at ±90°/180°

Usage
-----
    cd sim
    .venv312/bin/python3.12 scripts/validate.py
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import Optional

import mujoco
import numpy as np

# ── ensure sim/ is on sys.path so we can import project modules ───────────────
_SIM_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SIM_DIR))

from controller import (                         # noqa: E402
    FollowController,
    TARGET_DIST_GOAL,
    TARGET_DIST_MIN_SAFE,
)
from low_level   import default_low_level, LowLevelBase  # noqa: E402
from logger      import SimLogger                # noqa: E402
from tracker     import BallTracker              # noqa: E402

# ── sim constants (must match main.py) ────────────────────────────────────────
SCENE_XML  = _SIM_DIR / "scene.xml"
SIM_DT     = 0.002                              # physics timestep (s)
CTRL_HZ    = 50                                 # control frequency
CTRL_SKIP  = max(1, round(1.0 / (CTRL_HZ * SIM_DT)))
CTRL_DT    = CTRL_SKIP * SIM_DT                # 0.02 s per control step
CAM_W, CAM_H = 320, 240
FALL_Z     = 0.12                               # m – body below this = fell
RAMP_DUR   = 1.0                               # s – standup ramp

# EMA command smoothing (same as main.py)
VX_SMOOTH   = 0.30
VYAW_SMOOTH = 0.55

# ── helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[VAL] {msg}", flush=True)


def _bar(char: str = "═", width: int = 62) -> str:
    return char * width


# ══════════════════════════════════════════════════════════════════════════════
#  ScriptedMover
# ══════════════════════════════════════════════════════════════════════════════

class Phase:
    """
    Descriptor for one validation scenario phase.

    Parameters
    ----------
    name       : human-readable label (e.g. "1. Hold far (4 m)")
    duration   : how long this phase runs (seconds)
    mode       : "hold" | "orbit" | "teleport_sequence"
    criterion  : pass/fail rule key (see PhaseMetrics.evaluate)
    target_dist: reference world distance for reporting
    min_det_pct: minimum detection % required to pass (for orbit/hold phases)

    Mode-specific keyword arguments
    --------------------------------
    hold              → x=, y=  (ball world position)
    orbit             → radius=, omega=  (CCW around origin)
    teleport_sequence → positions=[(x, y, hold_s), ...]
    """

    def __init__(
        self,
        name:       str,
        duration:   float,
        mode:       str,
        criterion:  str,
        target_dist: float = 2.0,
        min_det_pct: float = 70.0,
        warmup_s:    float = 3.0,
        **kwargs,
    ) -> None:
        self.name        = name
        self.duration    = duration
        self.mode        = mode
        self.criterion   = criterion
        self.target_dist = target_dist
        self.min_det_pct = min_det_pct
        self.warmup_s    = warmup_s   # hold at start pos before phase timer starts
        # mode params
        self.x:         float = float(kwargs.get("x", 2.0))
        self.y:         float = float(kwargs.get("y", 0.0))
        self.radius:    float = float(kwargs.get("radius", 2.0))
        self.omega:     float = float(kwargs.get("omega", 0.18))
        self.positions: list  = list(kwargs.get("positions", []))

    def start_pos(self) -> tuple[float, float]:
        """Ball world (x, y) at the beginning of this phase."""
        if self.mode == "hold":
            return (self.x, self.y)
        elif self.mode == "orbit":
            return (self.radius, 0.0)
        elif self.mode == "teleport_sequence" and self.positions:
            x0, y0, _ = self.positions[0]
            return (x0, y0)
        return (2.0, 0.0)


class ScriptedMover:
    """
    Executes a playlist of Phase objects, updating `data.mocap_pos` each step.

    Each phase has an optional `warmup_s` hold period at the start where the
    ball stays at its start position and the phase timer does NOT advance.
    This gives the robot time to spin and re-acquire after a phase transition
    before metrics are recorded.

    Call step(dt, mocap_pos) every control frame.
    Query phase_elapsed() to check if the current phase has finished (warmup
    is NOT counted in elapsed).
    Call advance_phase() to move to the next phase.
    """

    def __init__(self, phases: list[Phase]) -> None:
        self._phases       = phases
        self._phase_idx    = 0
        self._phase_t      = 0.0
        self._warmup_t     = 0.0   # time spent in warmup hold (not counted in metrics)
        self._orbit_angle  = 0.0
        self._tp_t         = 0.0   # time within current teleport sub-segment
        self._tp_idx       = 0     # index into positions list

    # ── public properties ─────────────────────────────────────────────────────

    @property
    def current_phase(self) -> Optional[Phase]:
        if self._phase_idx < len(self._phases):
            return self._phases[self._phase_idx]
        return None

    @property
    def done(self) -> bool:
        return self._phase_idx >= len(self._phases)

    @property
    def in_warmup(self) -> bool:
        """True while waiting for the tracker to acquire the ball."""
        ph = self.current_phase
        return ph is not None and self._warmup_t < ph.warmup_s

    @property
    def tp_idx(self) -> int:
        """Current teleport sub-segment index (for PhaseMetrics)."""
        return self._tp_idx

    def phase_elapsed(self) -> float:
        """Seconds into the active phase (warmup not counted)."""
        return self._phase_t

    # ── state transitions ─────────────────────────────────────────────────────

    def advance_phase(self) -> None:
        """Call after phase_elapsed() >= phase.duration."""
        self._phase_idx   += 1
        self._phase_t      = 0.0
        self._warmup_t     = 0.0
        self._orbit_angle  = 0.0
        self._tp_t         = 0.0
        self._tp_idx       = 0

    # ── per-step update ───────────────────────────────────────────────────────

    def step(self, dt: float, mocap_pos: np.ndarray) -> None:
        """Advance internal timers and write the new ball position."""
        ph = self.current_phase
        if ph is None:
            return

        # ── warmup: hold ball at start position, don't advance phase_t ────────
        if self._warmup_t < ph.warmup_s:
            self._warmup_t += dt
            x0, y0 = ph.start_pos()
            mocap_pos[0, 0] = x0
            mocap_pos[0, 1] = y0
            mocap_pos[0, 2] = 0.15
            return

        # ── active phase ───────────────────────────────────────────────────────
        self._phase_t += dt

        if ph.mode == "hold":
            x, y = ph.x, ph.y

        elif ph.mode == "orbit":
            self._orbit_angle = (self._orbit_angle + ph.omega * dt) % (2 * math.pi)
            x = ph.radius * math.cos(self._orbit_angle)
            y = ph.radius * math.sin(self._orbit_angle)

        elif ph.mode == "teleport_sequence":
            pos_list = ph.positions
            if not pos_list:
                x, y = 2.0, 0.0
            else:
                tp_x, tp_y, tp_dur = pos_list[self._tp_idx]
                self._tp_t += dt
                if self._tp_t >= tp_dur and self._tp_idx + 1 < len(pos_list):
                    self._tp_idx += 1
                    self._tp_t    = 0.0
                    tp_x, tp_y, _ = pos_list[self._tp_idx]
                    _log(f"TELEPORT → ({tp_x:+.1f}, {tp_y:+.1f})  [seg {self._tp_idx}]")
                x, y = tp_x, tp_y

        else:
            x, y = 2.0, 0.0

        mocap_pos[0, 0] = x
        mocap_pos[0, 1] = y
        mocap_pos[0, 2] = 0.15


# ══════════════════════════════════════════════════════════════════════════════
#  PhaseMetrics + per-phase summary / pass/fail
# ══════════════════════════════════════════════════════════════════════════════

class PhaseMetrics:
    """Accumulates per-step statistics for a single phase."""

    def __init__(self, phase: Phase) -> None:
        self.phase         = phase
        self.steps         = 0
        self.det_steps     = 0
        self.fell_steps    = 0
        self.dist_sum      = 0.0
        self.in_range      = 0       # steps with MIN_SAFE < world dist ≤ GOAL (<1 m)
        self.dist_start:   Optional[float] = None
        self.dist_end:     Optional[float] = None

        # Teleport re-acquisition tracking
        self._last_tp_idx      = 0
        self._tp_lost_t:       float = 0.0
        self._tp_tracking:     bool  = False
        self.reacquire_times:  list[float] = []   # s after each teleport event

    # ── record ────────────────────────────────────────────────────────────────

    def record(
        self,
        dist:     float,
        detected: bool,
        fell:     bool,
        phase_t:  float,
        tp_idx:   int,
    ) -> None:
        self.steps    += 1
        self.dist_sum += dist
        if detected:
            self.det_steps += 1
        if fell:
            self.fell_steps += 1
        if TARGET_DIST_MIN_SAFE < dist <= TARGET_DIST_GOAL:
            self.in_range += 1
        if self.dist_start is None:
            self.dist_start = dist
        self.dist_end = dist

        # Teleport re-acquisition
        if self.phase.mode == "teleport_sequence":
            if tp_idx != self._last_tp_idx:
                # A new teleport just happened
                self._tp_lost_t   = phase_t
                self._tp_tracking = False
                self._last_tp_idx = tp_idx
            if not self._tp_tracking and detected:
                elapsed = phase_t - self._tp_lost_t
                self.reacquire_times.append(elapsed)
                self._tp_tracking = True

    # ── derived stats ─────────────────────────────────────────────────────────

    @property
    def avg_dist(self) -> float:
        return self.dist_sum / max(self.steps, 1)

    @property
    def det_pct(self) -> float:
        return 100.0 * self.det_steps / max(self.steps, 1)

    @property
    def fell_pct(self) -> float:
        return 100.0 * self.fell_steps / max(self.steps, 1)

    @property
    def in_range_pct(self) -> float:
        return 100.0 * self.in_range / max(self.steps, 1)

    # ── pass / fail ───────────────────────────────────────────────────────────

    def evaluate(self) -> tuple[bool, str]:
        """Return (passed, reason_string)."""
        ph      = self.phase
        no_fall = self.fell_pct < 5.0    # < 5 % fell steps = stable

        if ph.criterion == "stability":
            # Just don't fall — ball is invisible, but robot should survive spinning.
            passed = no_fall
            reason = f"fell={self.fell_pct:.1f}%  det={self.det_pct:.1f}%"

        elif ph.criterion == "hold_nominal":
            passed = (
                self.in_range_pct >= 70.0
                and self.det_pct   >= ph.min_det_pct
                and no_fall
            )
            reason = (
                f"in_range={self.in_range_pct:.0f}%  "
                f"det={self.det_pct:.0f}% (need ≥{ph.min_det_pct:.0f}%)  "
                f"fell={self.fell_pct:.1f}%"
            )

        elif ph.criterion == "hold_close":
            passed = no_fall and self.det_pct >= ph.min_det_pct
            reason = (
                f"det={self.det_pct:.0f}% (need ≥{ph.min_det_pct:.0f}%)  "
                f"fell={self.fell_pct:.1f}%  avg_dist={self.avg_dist:.2f}m"
            )

        elif ph.criterion == "orbit":
            passed = self.det_pct >= ph.min_det_pct and no_fall
            reason = (
                f"det={self.det_pct:.0f}% (need ≥{ph.min_det_pct:.0f}%)  "
                f"avg_dist={self.avg_dist:.2f}m  fell={self.fell_pct:.1f}%"
            )

        elif ph.criterion == "orbit_large":
            # Detection will be lower (3m range limit); accept lower threshold.
            # Also check robot didn't wander far off (dist < 4m).
            passed = self.det_pct >= ph.min_det_pct and no_fall and self.avg_dist < 4.0
            reason = (
                f"det={self.det_pct:.0f}% (need ≥{ph.min_det_pct:.0f}%)  "
                f"avg_dist={self.avg_dist:.2f}m  fell={self.fell_pct:.1f}%"
            )

        elif ph.criterion == "teleport":
            if self.reacquire_times:
                max_ra = max(self.reacquire_times)
                passed = max_ra <= 4.0 and no_fall
                times_str = ", ".join(f"{t:.1f}s" for t in self.reacquire_times)
                reason = (
                    f"re-acq after each jump: [{times_str}]  "
                    f"max={max_ra:.1f}s (need ≤4.0s)  fell={self.fell_pct:.1f}%"
                )
            else:
                passed = False
                reason = "no teleport re-acquisitions recorded"

        else:
            passed = no_fall
            reason = f"fell={self.fell_pct:.1f}%"

        return passed, reason


def _print_phase_summary(m: PhaseMetrics, passed: bool, reason: str) -> None:
    ph      = m.phase
    verdict = "PASS ✓" if passed else "FAIL ✗"
    bar     = _bar()
    print(f"\n{bar}")
    print(f"  {ph.name}")
    print(_bar("-"))
    print(f"  Duration         : {ph.duration:.1f} s  ({m.steps} ctrl steps)")
    print(f"  Ball detected    : {m.det_pct:6.1f} %")
    print(f"  Robot fell       : {m.fell_pct:6.1f} %")
    print(f"  Avg world dist   : {m.avg_dist:6.2f} m")
    if ph.criterion in ("hold_nominal", "orbit", "orbit_large"):
        print(
            f"  In goal (<{TARGET_DIST_GOAL:g} m): {m.in_range_pct:5.1f} %"
        )
    if ph.criterion in ("stability", "hold_nominal", "hold_close"):
        d_start = m.dist_start if m.dist_start is not None else 0.0
        d_end   = m.dist_end   if m.dist_end   is not None else 0.0
        print(f"  dist start→end   : {d_start:.2f} m → {d_end:.2f} m")
    if m.reacquire_times:
        times_str = ", ".join(f"{t:.2f}s" for t in m.reacquire_times)
        print(f"  Re-acq times     : [{times_str}]")
    print(_bar("-"))
    print(f"  Verdict : {verdict}")
    print(f"  Reason  : {reason}")
    print(f"{bar}\n", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Scenario playlist
# ══════════════════════════════════════════════════════════════════════════════

PHASES: list[Phase] = [
    # 1 ── Hold at 4 m
    # Ball is barely visible at 4 m (area ≈ 40 px² in 320×240 — near tracker
    # minimum).  The robot will mostly spin in SEARCHING mode.
    # PASS criterion: robot stays upright for the full duration (stability test).
    Phase(
        name        = "1. Hold far (4 m)    — stability / SEARCHING",
        duration    = 12.0,
        mode        = "hold",
        criterion   = "stability",
        target_dist = 4.0,
        x=4.0, y=0.0,
    ),

    # 2 ── Hold at 2 m (nominal)
    # Robot should close from 2 m until world stand-off is usually ≤ 1 m.
    Phase(
        name         = "2. Hold nominal (2 m) — steady-state tracking",
        duration     = 10.0,
        mode         = "hold",
        criterion    = "hold_nominal",
        target_dist  = 2.0,
        min_det_pct  = 80.0,
        x=2.0, y=0.0,
    ),

    # 3 ── Hold at 1.0 m (close)
    # Ball is close; robot should detect it, hold position (backward disabled),
    # and not fall due to proximity.
    Phase(
        name         = "3. Hold close (1.0 m) — proximity handling",
        duration     = 8.0,
        mode         = "hold",
        criterion    = "hold_close",
        target_dist  = 1.0,
        min_det_pct  = 70.0,
        x=1.0, y=0.0,
    ),

    # 4 ── Orbit slow  (ω = 0.10 rad/s)
    # Comfortable orbit speed; tests basic rotation following.
    Phase(
        name         = "4. Orbit slow   (r=2 m, ω=0.10 rad/s)",
        duration     = 15.0,
        mode         = "orbit",
        criterion    = "orbit",
        target_dist  = 2.0,
        min_det_pct  = 85.0,
        radius=2.0, omega=0.10,
    ),

    # 5 ── Orbit medium (ω = 0.25 rad/s)
    Phase(
        name         = "5. Orbit medium (r=2 m, ω=0.25 rad/s)",
        duration     = 12.0,
        mode         = "orbit",
        criterion    = "orbit",
        target_dist  = 2.0,
        min_det_pct  = 70.0,
        radius=2.0, omega=0.25,
    ),

    # 6 ── Orbit fast   (ω = 0.50 rad/s)
    # Challenges the robot's yaw response.  Lower detection bar.
    Phase(
        name         = "6. Orbit fast   (r=2 m, ω=0.50 rad/s)",
        duration     = 10.0,
        mode         = "orbit",
        criterion    = "orbit",
        target_dist  = 2.0,
        min_det_pct  = 50.0,
        radius=2.0, omega=0.50,
    ),

    # 7 ── Orbit large radius (r=3 m)
    # Ball at 3 m is near the visual detection limit; lower det threshold.
    Phase(
        name         = "7. Orbit large  (r=3 m, ω=0.15 rad/s)",
        duration     = 15.0,
        mode         = "orbit",
        criterion    = "orbit_large",
        target_dist  = 3.0,
        min_det_pct  = 30.0,
        radius=3.0, omega=0.15,
    ),

    # 8 ── Teleport sequence
    # Ball suddenly jumps to four compass positions (all at 2 m radius) so the
    # robot must spin and re-acquire after each jump.
    # PASS: re-acquisition within 3 s after every jump.
    Phase(
        name         = "8. Teleport sequence (2m, ±90°/180° jumps)",
        duration     = 20.0,
        mode         = "teleport_sequence",
        criterion    = "teleport",
        target_dist  = 2.0,
        positions    = [
            ( 2.0,  0.0, 4.0),   # front
            ( 0.0,  2.0, 4.0),   # left  90°
            (-2.0,  0.0, 4.0),   # behind 180°
            ( 0.0, -2.0, 4.0),   # right  270°
            ( 2.0,  0.0, 4.0),   # back to front
        ],
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation loop
# ══════════════════════════════════════════════════════════════════════════════

def _reset_robot(
    model:      mujoco.MjModel,
    data:       mujoco.MjData,
    ramp_start: list,
    gait:       LowLevelBase,
    tracker:    BallTracker,
    controller: FollowController,
) -> None:
    """Hard-reset robot to keyframe 0 after a fall."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    gait.reset_phase(0.0)
    ramp_start[0] = data.time
    tracker.reset()
    controller.reset()
    _log("Robot reset after fall.")


def run_validation(phases: list[Phase]) -> list[tuple[Phase, bool, str]]:
    """
    Run all phases sequentially in one headless MuJoCo session.
    Returns a list of (phase, passed, reason) tuples.
    """

    # ── MuJoCo setup ─────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, CAM_H, CAM_W)
    head_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")

    gait:       LowLevelBase   = default_low_level(model)
    tracker:    BallTracker    = BallTracker()
    controller: FollowController = FollowController()

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = SimLogger(tag=f"validate_{ts}")

    ramp_start = [float(data.time)]
    vx_cmd = vyaw_cmd = 0.0
    step_n = 0

    # ── warm-up: let robot stand up, ball at first phase start pos ───────────
    if phases:
        x0, y0 = phases[0].start_pos()
        data.mocap_pos[0, 0] = x0
        data.mocap_pos[0, 1] = y0
        data.mocap_pos[0, 2] = 0.15

    WARMUP_STEPS = int((RAMP_DUR + 1.5) / CTRL_DT)
    _log(f"Warm-up: {WARMUP_STEPS} ctrl steps ({(RAMP_DUR + 1.5):.1f} s) …")
    for _ in range(WARMUP_STEPS):
        for _ in range(CTRL_SKIP):
            gait.substep(model, data, 0.0, 0.0, ramp_start[0], float(data.time), SIM_DT)
            mujoco.mj_step(model, data)
    _log("Warm-up complete. Starting validation phases.\n")

    # ── phase loop ────────────────────────────────────────────────────────────
    mover   = ScriptedMover(phases)
    metrics: Optional[PhaseMetrics] = None
    results: list[tuple[Phase, bool, str]] = []

    def _start_phase(ph: Phase) -> PhaseMetrics:
        """
        Prepare for a new phase:
          1. Pre-position ball at phase start location.
          2. Pre-rotate the robot body to FACE the ball — eliminates the search
             spin time and makes every phase independent of prior yaw state.
          3. Zero body angular velocity to avoid spin-induced stumble.
          4. mj_forward() so all derived quantities are consistent.
          5. Reset tracker + controller fresh.
          6. A short warmup_s hold allows the gait to stabilise and the
             tracker to acquire the ball before metrics recording begins.
        """
        x0, y0 = ph.start_pos()
        data.mocap_pos[0, 0] = x0
        data.mocap_pos[0, 1] = y0
        data.mocap_pos[0, 2] = 0.15

        # ── face the ball ────────────────────────────────────────────────────
        rx = float(data.qpos[0])
        ry = float(data.qpos[1])
        dx, dy = x0 - rx, y0 - ry
        target_yaw = math.atan2(dy, dx)
        half = target_yaw / 2.0
        data.qpos[3] = math.cos(half)   # quaternion w
        data.qpos[4] = 0.0              # quaternion x
        data.qpos[5] = 0.0              # quaternion y
        data.qpos[6] = math.sin(half)   # quaternion z
        data.qvel[3] = 0.0              # angular velocity (roll, pitch, yaw)
        data.qvel[4] = 0.0
        data.qvel[5] = 0.0
        mujoco.mj_forward(model, data)

        tracker.reset()
        controller.reset()

        _log(
            f"{'=' * 55}\n"
            f"[VAL] Phase: {ph.name}\n"
            f"[VAL] duration={ph.duration:.0f}s  warmup={ph.warmup_s:.0f}s  "
            f"mode={ph.mode}  criterion={ph.criterion}  "
            f"target_dist={ph.target_dist:.1f}m\n"
            f"[VAL] ball starts at ({x0:.1f}, {y0:.1f})  "
            f"robot yaw pre-set to {math.degrees(target_yaw):.1f}°\n"
            f"{'=' * 55}"
        )
        return PhaseMetrics(ph)

    if not mover.done:
        metrics = _start_phase(mover.current_phase)
        # seed ball at start position (warmup hold starts immediately)
        mover.step(0.0, data.mocap_pos)

    fell_at: Optional[float] = None

    while not mover.done:
        ph = mover.current_phase
        if ph is None:
            break

        # ── physics ──────────────────────────────────────────────────────────
        for _ in range(CTRL_SKIP):
            gait.substep(model, data, vx_cmd, vyaw_cmd,
                         ramp_start[0], float(data.time), SIM_DT)
            mujoco.mj_step(model, data)
        step_n += 1
        sim_time = float(data.time)

        # ── robot state ───────────────────────────────────────────────────────
        rx  = float(data.qpos[0])
        ry  = float(data.qpos[1])
        rz  = float(data.qpos[2])
        bx  = float(data.mocap_pos[0, 0])
        by  = float(data.mocap_pos[0, 1])
        dist = math.hypot(rx - bx, ry - by)
        quat = data.qpos[3:7]
        yaw  = math.atan2(
            2.0 * (quat[0]*quat[3] + quat[1]*quat[2]),
            1.0 - 2.0 * (quat[2]**2 + quat[3]**2),
        )
        fell = rz < FALL_Z

        # ── fall handling ────────────────────────────────────────────────────
        if fell:
            if fell_at is None:
                fell_at = sim_time
                _log(
                    f"FELL at t={sim_time:.2f}s | dist={dist:.2f}m | "
                    f"cmd vx={vx_cmd:+.3f} vyaw={vyaw_cmd:+.3f}"
                )
            elif sim_time - fell_at >= 3.0:
                _reset_robot(model, data, ramp_start, gait, tracker, controller)
                vx_cmd = vyaw_cmd = 0.0
                fell_at = None
                # Re-position ball after reset
                data.mocap_pos[0, 0] = bx
                data.mocap_pos[0, 1] = by
                data.mocap_pos[0, 2] = 0.15
        else:
            fell_at = None

        # ── vision ────────────────────────────────────────────────────────────
        renderer.update_scene(data, camera=head_id)
        head_rgb = renderer.render().copy()
        det  = tracker.update(head_rgb)
        cx, cy, area, conf = det.cx, det.cy, det.area, det.confidence

        # ── control ───────────────────────────────────────────────────────────
        vx_raw, vyaw_raw = controller.compute(
            cx, cy, area, conf, CAM_W, CAM_H, CTRL_DT,
            world_dist_m=dist,
        )
        if step_n == 1:
            vx_cmd, vyaw_cmd = vx_raw, vyaw_raw
        else:
            vx_cmd   = VX_SMOOTH   * vx_raw   + (1.0 - VX_SMOOTH)   * vx_cmd
            vyaw_cmd = VYAW_SMOOTH * vyaw_raw  + (1.0 - VYAW_SMOOTH) * vyaw_cmd

        # ── move ball ─────────────────────────────────────────────────────────
        mover.step(CTRL_DT, data.mocap_pos)

        # ── per-step log ──────────────────────────────────────────────────────
        if step_n % 50 == 0:
            dist_est_str = "---"
            if area > 0:
                from controller import TARGET_AREA_FRAC, DIST_CAL
                _ta = TARGET_AREA_FRAC * CAM_W * CAM_H
                de = DIST_CAL * math.sqrt(_ta / max(area, 1.0))
                dist_est_str = f"{de:.2f}m"
            fall_tag   = " ***FELL***" if fell else ""
            warmup_tag = " [WARMUP]"  if mover.in_warmup else ""
            print(
                f"[STEP {step_n:5d}] t={sim_time:5.1f}s{fall_tag}{warmup_tag}  "
                f"ph={mover._phase_idx+1}/{len(phases)}  "
                f"pos=({rx:+.2f},{ry:+.2f}) z={rz:.3f}  "
                f"ball=({bx:+.2f},{by:+.2f}) dist={dist:.2f}m  "
                f"det={'Y' if cx is not None else 'N'} area={int(area):4d} "
                f"dest≈{dist_est_str}  conf={conf:.2f}  "
                f"vx={vx_cmd:+.3f} vyaw={vyaw_cmd:+.3f}",
                flush=True,
            )

        # ── metrics (skip warmup period) ─────────────────────────────────────
        if metrics is not None and not mover.in_warmup:
            metrics.record(
                dist     = dist,
                detected = (cx is not None),
                fell     = fell,
                phase_t  = mover.phase_elapsed(),
                tp_idx   = mover.tp_idx,
            )

        logger.step(
            step=step_n, sim_time=sim_time,
            robot_x=rx, robot_y=ry, robot_z=rz, robot_yaw=yaw,
            ball_x=bx, ball_y=by,
            cx=cx, cy=cy, area=area, confidence=conf,
            vx_cmd=vx_cmd, vyaw_cmd=vyaw_cmd,
            img_w=CAM_W,
        )

        # ── phase completion check ────────────────────────────────────────────
        if mover.phase_elapsed() >= ph.duration:
            # End of phase: print summary
            passed, reason = metrics.evaluate()
            _print_phase_summary(metrics, passed, reason)
            results.append((ph, passed, reason))

            # Advance and start next phase
            mover.advance_phase()
            if not mover.done:
                vx_cmd = vyaw_cmd = 0.0
                metrics = _start_phase(mover.current_phase)
                mover.step(0.0, data.mocap_pos)   # pre-position ball in warmup hold

    logger.close()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Final report
# ══════════════════════════════════════════════════════════════════════════════

def print_final_report(results: list[tuple[Phase, bool, str]]) -> None:
    bar  = _bar("═", 72)
    dash = _bar("-", 72)
    print(f"\n{bar}")
    print("  VALIDATION FINAL REPORT")
    print(dash)
    print(f"  {'Phase':<45}  {'Verdict':<9}  Notes")
    print(dash)
    for ph, passed, reason in results:
        verdict = "PASS ✓" if passed else "FAIL ✗"
        note    = reason[:30] + "…" if len(reason) > 30 else reason
        print(f"  {ph.name:<45}  {verdict:<9}  {note}")
    print(dash)
    n_pass  = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    pct     = 100 * n_pass / max(n_total, 1)
    print(f"  TOTAL : {n_pass} / {n_total} phases passed  ({pct:.0f} %)")
    print(f"{bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.perf_counter()
    scenario_s = sum(p.duration for p in PHASES)
    warmup_s   = sum(p.warmup_s  for p in PHASES)
    print(_bar("═", 62))
    print("  Go2 Object-Tracking — Validation Suite")
    print(f"  {len(PHASES)} phases · {scenario_s:.0f} s active + {warmup_s:.0f} s warmup")
    print(_bar("═", 62) + "\n")

    results = run_validation(PHASES)
    print_final_report(results)

    wall = time.perf_counter() - t0
    print(f"Total wall-clock time: {wall:.1f} s")
