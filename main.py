"""
main.py – Go2 Object Tracking Simulation (MuJoCo)

Run from the sim/ directory:
    bash run.sh
or:
    .venv312/bin/python3.12 main.py

Display: a 2×2 cv2 window showing
  TOP-LEFT     – top-down arena (trails, heading arrow)
  TOP-RIGHT    – third-person chase camera (track COM)
  BOTTOM-LEFT  – head camera (detection overlay, distance bar, HUD)
  BOTTOM-RIGHT – rear-facing body camera

Controls (cv2 window must be focused):
    Q / ESC  – quit
    R        – reset robot to home pose
    P        – pause / resume physics

Improvements in this version
-----------------------------
* BallTracker (HSV + histogram back-projection + CamShift, with coast on brief loss).
* PD+I controller (FollowController) replaces pure-P.
* SimLogger writes per-step CSV + terminal summaries.
* low_level.trot_targets: orientation stabilisation + standup ramp (first ~1 s).
* Visualisation: trail deques, heading arrow, distance bar, expanded HUD,
  fall detection + auto-reset.
* DemoTargetMover: radial distance → lateral sweep → orbit (multi-speed) demo.
* low_level.TrotPDGait: pluggable torque interface (Menagerie-aligned model).
"""

from __future__ import annotations

import collections
import math
import os
import time
import pathlib

import mujoco
import numpy as np
import cv2

from tracker import BallTracker, annotate_frame
from controller import FollowController, apply_stability_cap
from logger import SimLogger
from low_level import LowLevelBase, default_low_level, quat_to_pitch_roll

# ── paths ────────────────────────────────────────────────────────────────────
SCENE_XML = pathlib.Path(__file__).parent / "scene.xml"

# ── simulation constants ─────────────────────────────────────────────────────
SIM_DT    = 0.002   # physics timestep (s)
CTRL_HZ   = 50      # control / render frequency (Hz)
CTRL_SKIP = max(1, round(1.0 / (CTRL_HZ * SIM_DT)))   # physics steps per frame
CTRL_DT   = CTRL_SKIP * SIM_DT                        # actual time per ctrl step

CAM_W, CAM_H = 480, 360    # per-camera render resolution

# RENDER_SKIP: render+vision only every Nth control step.  Physics, gait and
# the ball mover still tick every step, so the SIM ADVANCES at full rate while
# the wall-clock cost of rendering is amortised across N steps.  Used to get
# more sim-time-per-wall-second during long recordings (e.g. the 5-min
# presentation video).  N=1 → render every step (default, unchanged behaviour).
RENDER_SKIP = max(1, int(os.environ.get("RENDER_SKIP", "1")))
DISP_W = CAM_W * 2 + 4     # one mosaic row width (two views + 4 px gap)

# Headless benchmark: GO2_HEADLESS=1 GO2_HEADLESS_SECONDS=90 python main.py
HEADLESS = os.environ.get("GO2_HEADLESS", "").lower() in ("1", "true", "yes")
HEADLESS_MAX_T = float(os.environ.get("GO2_HEADLESS_SECONDS", "90"))
# Optional runtime cap for GUI/headless parity checks.
# When > 0, simulation exits after this many seconds regardless of mode.
MAX_RUNTIME_T = float(os.environ.get("GO2_MAX_SECONDS", "0"))
# GUI modes:
#   GO2_GUI_VIEW=head  -> fast single head-camera view (default)
#   GO2_GUI_VIEW=chase -> single 3rd-person chase camera (fullscreen, presentation)
#   GO2_GUI_VIEW=full  -> 2x2 mosaic (top/chase/head/rear)
GUI_VIEW_MODE = os.environ.get("GO2_GUI_VIEW", "head").strip().lower()
# Forward-velocity multiplier applied to the controller's vx_cmd before it
# enters the gait.  Default 1.0 = normal; >1.0 = faster (stress test).
ROBOT_SPEED_SCALE = float(os.environ.get("ROBOT_SPEED_SCALE", "1.0"))
GUI_HEAD_ONLY = GUI_VIEW_MODE not in ("full",)
GUI_CHASE_ONLY = (GUI_VIEW_MODE == "chase")
# Use step budget (fall reset zeros data.time, so sim_time is not monotonic)
HEADLESS_MAX_STEPS = max(1, round(HEADLESS_MAX_T / CTRL_DT))
MAX_RUNTIME_STEPS = max(1, round(MAX_RUNTIME_T / CTRL_DT)) if MAX_RUNTIME_T > 0.0 else 0

# ── fall / incapacity detection ───────────────────────────────────────────────
# Root z alone misses “half down” poses: base can stay >0.12 m while pitch/roll
# are huge, so the RL gait keeps chasing commands and never recovers.
FALL_Z          = 0.12   # m – root below this height = fell
FALL_TILT_RAD   = 0.75   # ≈43° — |pitch| or |roll| beyond this = incapacitated
                          # Raised from 0.62: the RL trot gait naturally peaks at ~35°
                          # during certain stride phases; 0.62 fired false-positives that
                          # zeroed commands and caused real falls (observed in GUI logs).
FALL_RESET_S    = 2.0    # seconds after fall before auto-reset
TILT_GUARD_SOFT = 0.05   # rad — begin attenuation at ~3°
TILT_GUARD_HARD = 0.14   # rad — strong damping (~8°)


def _pose_incapped(rz: float, quat_wxyz: np.ndarray) -> bool:
    pitch, roll = quat_to_pitch_roll(quat_wxyz.astype(float))
    if rz < FALL_Z:
        return True
    if abs(pitch) > FALL_TILT_RAD or abs(roll) > FALL_TILT_RAD:
        return True
    if rz < 0.18 and (abs(pitch) > 0.52 or abs(roll) > 0.52):
        return True
    return False


def _tilt_guard_scale(quat_wxyz: np.ndarray) -> float:
    """
    Smooth scale factor [0.2, 1.0] from current tilt.
    1.0 means no attenuation, lower values damp commands to avoid falls.
    """
    pitch, roll = quat_to_pitch_roll(quat_wxyz.astype(float))
    tilt = max(abs(pitch), abs(roll))
    if tilt <= TILT_GUARD_SOFT:
        return 1.0
    if tilt >= TILT_GUARD_HARD:
        return 0.0
    frac = (tilt - TILT_GUARD_SOFT) / max(TILT_GUARD_HARD - TILT_GUARD_SOFT, 1e-6)
    return 1.0 - 1.0 * frac

# ── trail config ─────────────────────────────────────────────────────────────
TRAIL_LEN    = 80
ARENA_HALF   = 5.0     # arena is roughly ±5 m in X and Y

# ── target distance bar config ────────────────────────────────────────────────
# Green band ≈ apparent size at ~1 m (inverse-square vs 1215 px² @ 2 m ref).
TARGET_AREA  = int(1_215 * 4)   # px² — HUD “in goal” ≈ under 1 m standoff
MAX_AREA     = 30_000           # px² — very close / huge blob
BAR_X, BAR_Y = 10, CAM_H - 30
BAR_W, BAR_H = CAM_W - 20, 16
STANDOFF_LO  = TARGET_AREA * 0.7
STANDOFF_HI  = TARGET_AREA * 1.3


# Low-level locomotion (trot + PD) lives in low_level.py — swap `default_low_level`
# for another backend (e.g. RawTorqueLowLevel) without changing vision / logging.

# ── target mover ─────────────────────────────────────────────────────────────

class DemoTargetMover:
    """
    Configurable continuous target mover.

    - `BALL_PATH_MODE=square`  : smooth large square, wait at corners.
    - `BALL_PATH_MODE=circle`  : smooth circle.
    - `BALL_PATH_MODE=figure8` : smooth figure-8 (Gerono curve).
    - If ball leaves sight, it keeps moving for 3 seconds, then pauses.
    - Motion resumes once the ball is seen again.
    """

    def __init__(self) -> None:
        self._mode: str = os.environ.get("BALL_PATH_MODE", "square").strip().lower()
        if self._mode not in {"square", "circle", "figure8", "corridor"}:
            self._mode = "square"

        self._lost_grace_s: float = 3.0
        self._lost_time: float = 0.0
        self._param_t: float = 0.0

        # Square mode — arena shrunk from half=3.2 to 2.0: the robot's top actual
        # speed is ~0.15-0.20 m/s so it needs close edges to ever catch up.
        half = float(os.environ.get("BALL_SQUARE_HALF", "2.0"))
        # Classic square wrap-around origin: ball traces a 2·half × 2·half
        # perimeter centered on (0,0) so the robot has to rotate to follow it
        # through all 4 quadrants.
        self._corners: list[tuple[float, float]] = [
            ( half,  half),     # top-right
            (-half,  half),     # top-left
            (-half, -half),     # bottom-left
            ( half, -half),     # bottom-right
        ]
        # Ball speed tuned to robot's actual saturated velocity (~0.15-0.20 m/s under
        # the RL gait).  At the previous 0.55 m/s the robot could never close distance
        # and stayed orbiting at ~3 m.  Slower ball also means fewer bang-bang yaw
        # pulses per second, which directly reduces fall probability.
        # 2026-04-21: user asked for faster ball — bumped 0.08→0.10 m/s (1.25×).
        # (0.12/0.15/0.18 all pushed mean distance well above 1 m on straight
        # edges; 0.10 m/s is a visible-but-catchable compromise.)
        self._edge_speed: float = float(os.environ.get("BALL_SQUARE_SPEED", "0.10"))
        self._corner_wait_s: float = float(os.environ.get("BALL_CORNER_WAIT", "5.0"))

        self._corner_idx: int = 0
        self._edge_from_idx: int = 0
        self._edge_to_idx: int = 1
        self._edge_progress: float = 0.0
        self._corner_wait_rem: float = self._corner_wait_s

        # Circle mode
        # Circle mode — tangential speed = r * omega.
        # 2026-04-21: user asked for faster ball — bumped omega 0.05→0.065
        # (tangential ≈0.10 m/s, 1.3× original; at 0.075 the 60-s near-goal
        # dropped to 65%; at 0.065 tracking stays ≥80% near goal.)
        self._circle_r: float = float(os.environ.get("BALL_CIRCLE_RADIUS", "1.5"))
        self._circle_w: float = float(os.environ.get("BALL_CIRCLE_OMEGA", "0.065"))

        # Figure-8 mode — wider pattern shifted forward so ball is mostly in front
        # of the robot (it was passing through origin before, making visibility jittery).
        # Offset +1.5 m in X so ball ranges from x=0 to x=3, always reachable.
        # 2026-04-21: user asked for faster ball — bumped omega 0.05→0.055
        # (peak tangential ≈0.17 m/s, 1.1× original; fig8 is the most fragile
        # pattern because of the crossover — larger ω tanks visibility.)
        self._fig8_scale: float = float(os.environ.get("BALL_FIG8_SCALE", "1.5"))
        self._fig8_w: float = float(os.environ.get("BALL_FIG8_OMEGA", "0.055"))
        self._fig8_cx: float = float(os.environ.get("BALL_FIG8_CX", "1.8"))
        self._fig8_cy: float = float(os.environ.get("BALL_FIG8_CY", "0.0"))

        # Corridor mode — ball moves +X in a 3-m-wide corridor.  The corridor
        # has 3 LANES (y = -1, 0, +1).  The ball:
        #   1. picks one of the 3 lanes,
        #   2. switches lane periodically (every LANE_DWELL_S),
        #   3. forced-switches when an obstacle blocks the current lane.
        # Lateral motion between lanes is smooth (capped at lateral_speed).
        self._corridor_speed: float = float(os.environ.get("BALL_CORRIDOR_SPEED", "0.111"))
        self._corridor_half_w: float = 2.5   # 5 m wide corridor
        self._lanes: tuple[float, ...] = (-1.8, 0.0, +1.8)
        self._current_lane_idx: int = 1   # start in middle lane
        self._lane_dwell_s: float = float(os.environ.get("BALL_LANE_DWELL", "8.0"))
        self._lane_dwell_t: float = 0.0
        # Maximum allowed lead distance the ball is willing to open up over
        # the robot.  When the (ball − robot) gap exceeds this, the ball
        # holds its x-position (lateral lane motion still allowed) until the
        # robot catches up inside the limit again.
        self._max_lead: float = float(os.environ.get("BALL_MAX_LEAD", "5.0"))
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        # Adaptive-speed state: track previous robot x and the rolling estimate
        # of robot forward velocity so the ball can keep a constant target
        # lead distance (ball goes faster when robot goes faster, slower when
        # robot slows).
        self._prev_robot_x: float = 0.0
        self._robot_vx_ema: float = 0.0
        # Default obstacles (overwritten by set_corridor_obstacles).
        # Each entry is (x, y, y_half).
        self._obstacles = [
            (5.0,   0.75, 0.30),
            (9.0,  -0.75, 0.30),
            (13.0,  0.90, 0.30),
            (17.0, -0.90, 0.30),
        ]
        self._ball_radius: float = 0.13
        self._avoid_margin: float = 0.35
        self._lookahead: float = 2.5
        self._lateral_speed: float = 0.30
        self._corridor_y_target: float = 0.0

        self._x, self._y = self._corners[self._corner_idx]
        self.phase_label: str = "INIT"
        self.reset()

    def reset(self) -> None:
        self._lost_time = 0.0
        self._param_t = 0.0
        self._corner_idx = 0
        self._edge_from_idx = 0
        self._edge_to_idx = 1
        self._edge_progress = 0.0
        self._corner_wait_rem = self._corner_wait_s
        if self._mode == "square":
            self._x, self._y = self._corners[self._corner_idx]
            self.phase_label = "WAIT CORNER 1/4"
        elif self._mode == "circle":
            self._x, self._y = self._circle_r, 0.0
            self.phase_label = "CIRCLE"
        elif self._mode == "corridor":
            self._x, self._y = 2.0, 0.0   # start 2 m ahead of robot, centered
            self._corridor_y_target = 0.0
            self._current_lane_idx = 1     # middle lane
            self._lane_dwell_t = 0.0
            self.phase_label = "CORRIDOR"
        else:
            self._x, self._y = self._fig8_cx, self._fig8_cy
            self.phase_label = "FIGURE8"

    def _step_square(self, dt: float) -> None:
        n_corners = len(self._corners)
        if self._corner_wait_rem > 0.0:
            self._corner_wait_rem = max(0.0, self._corner_wait_rem - dt)
            self.phase_label = f"SQUARE WAIT CORNER {self._corner_idx + 1}/{n_corners}"
            if self._corner_wait_rem == 0.0:
                self._edge_from_idx = self._corner_idx
                self._edge_to_idx = (self._corner_idx + 1) % n_corners
                self._edge_progress = 0.0
            return

        x0, y0 = self._corners[self._edge_from_idx]
        x1, y1 = self._corners[self._edge_to_idx]
        edge_len = math.hypot(x1 - x0, y1 - y0)
        self._edge_progress += self._edge_speed * dt
        u = min(1.0, self._edge_progress / max(edge_len, 1e-6))
        self._x = x0 + (x1 - x0) * u
        self._y = y0 + (y1 - y0) * u
        self.phase_label = f"SQUARE EDGE {self._edge_from_idx + 1}->{self._edge_to_idx + 1}"

        if u >= 1.0:
            self._corner_idx = self._edge_to_idx
            self._x, self._y = self._corners[self._corner_idx]
            self._corner_wait_rem = self._corner_wait_s
            self.phase_label = f"SQUARE WAIT CORNER {self._corner_idx + 1}/{n_corners}"

    def _step_circle(self, dt: float) -> None:
        self._param_t += dt
        th = self._circle_w * self._param_t
        self._x = self._circle_r * math.cos(th)
        self._y = self._circle_r * math.sin(th)
        self.phase_label = "CIRCLE"

    def set_corridor_obstacles(self, obstacles: list) -> None:
        """Update obstacle list (called after randomization in main())."""
        self._obstacles = list(obstacles)

    def set_robot_position(self, rx: float, ry: float) -> None:
        """Tell the mover where the robot is so it can gate forward motion
        on the (ball − robot) gap (corridor mode)."""
        self._robot_x = float(rx)
        self._robot_y = float(ry)

    def get_corridor_target_y(self) -> float:
        """Return the y-coordinate of the ball's CURRENT target lane.
        Used by main loop so the robot can anticipate lane switches
        instead of lagging behind smooth lateral motion."""
        return float(self._corridor_y_target)

    def _step_figure8(self, dt: float) -> None:
        self._param_t += dt
        th = self._fig8_w * self._param_t
        # Shifted figure-8 so ball stays mostly in front of robot (x >= 0).
        self._x = self._fig8_cx + self._fig8_scale * math.sin(th)
        self._y = self._fig8_cy + 0.5 * self._fig8_scale * math.sin(2.0 * th)
        self.phase_label = "FIGURE8"

    def _step_corridor(self, dt: float) -> None:
        """Ball moves +X using 3 discrete lanes (y = -1, 0, +1).

        Lane logic:
          1. Periodically (every LANE_DWELL_S) the ball randomly considers
             switching to a different lane.
          2. If an obstacle blocks the current lane within LOOKAHEAD, the
             ball is FORCED to switch — picks any lane that is NOT blocked
             by an obstacle within the same look-ahead window.
          3. Lateral motion is smooth (capped at `_lateral_speed` m/s).
        """
        import random as _random
        self._param_t += dt

        clearance = self._ball_radius + self._avoid_margin
        current_lane_y = self._lanes[self._current_lane_idx]

        # Function: is `lane_y` blocked by ANY obstacle within look-ahead?
        # Each obstacle has its own y_half (lateral half-width).
        def lane_blocked(lane_y: float) -> bool:
            for entry in self._obstacles:
                ox, oy = entry[0], entry[1]
                oy_half = entry[2] if len(entry) > 2 else 0.30
                if 0.0 < (ox - self._x) < self._lookahead:
                    if abs(lane_y - oy) < oy_half + clearance:
                        return True
            return False

        # Decide whether to switch lane this step.
        self._lane_dwell_t += dt
        force_switch = lane_blocked(current_lane_y)
        time_switch  = (self._lane_dwell_t >= self._lane_dwell_s)

        if force_switch or time_switch:
            # Build candidate list: lanes NOT blocked by upcoming obstacles.
            free_lanes = [i for i in range(len(self._lanes))
                           if not lane_blocked(self._lanes[i])]
            # Prefer a NEW lane if possible.
            other_free = [i for i in free_lanes if i != self._current_lane_idx]
            if force_switch and other_free:
                self._current_lane_idx = _random.choice(other_free)
            elif force_switch and free_lanes:
                # All blocked except current — stay put (shouldn't happen
                # if we have ≥2 clear lanes always).
                self._current_lane_idx = free_lanes[0]
            elif time_switch and other_free:
                # Random lane swap for variety.
                self._current_lane_idx = _random.choice(other_free)
            self._lane_dwell_t = 0.0

        target_y = self._lanes[self._current_lane_idx]
        self._corridor_y_target = target_y

        # ── Leash logic ──────────────────────────────────────────────────
        # Ball must always stay AHEAD of the robot.  When the lead gap
        # exceeds `_max_lead` (default 3 m), the ball stops and waits for
        # the robot to catch up.  When the gap is within range, the ball
        # advances at its nominal corridor speed (faster than the robot's
        # achievable ground speed, so it naturally pulls ahead again).
        # If the robot somehow overtakes (lead < min_lead), snap the ball
        # forward to enforce a minimum lead.
        min_lead = float(os.environ.get("BALL_MIN_LEAD", "0.5"))
        if self._x < self._robot_x + min_lead:
            self._x = self._robot_x + min_lead
        lead = self._x - self._robot_x

        # Adaptive ball speed: track robot's forward velocity (EMA) and move
        # the ball at robot_vx + a small PD correction toward the target lead
        # distance.  This keeps the ball at a roughly constant gap ahead of
        # the robot regardless of how fast the robot is going.
        if dt > 1e-6:
            _instant_vx = (self._robot_x - self._prev_robot_x) / dt
            # Heavy EMA — smooths out per-step jitter.
            self._robot_vx_ema = (
                0.92 * self._robot_vx_ema + 0.08 * _instant_vx
            )
        self._prev_robot_x = self._robot_x

        _target_lead = 0.5 * (min_lead + self._max_lead)
        _lead_err = lead - _target_lead
        # If too far ahead → slow down; if too close → speed up.
        _correction = -0.5 * _lead_err
        _ball_vx = max(0.0, self._robot_vx_ema + _correction)
        ball_waiting = (lead >= self._max_lead)
        if not ball_waiting:
            self._x += _ball_vx * dt
        # Smooth lateral move toward target lane
        dy = target_y - self._y
        max_step = self._lateral_speed * dt
        if abs(dy) <= max_step:
            self._y = target_y
        else:
            self._y += math.copysign(max_step, dy)

        lane_label = ["LEFT", "MID", "RIGHT"][self._current_lane_idx]
        wait_tag = " [WAITING]" if ball_waiting else ""
        self.phase_label = (
            f"CORRIDOR x={self._x:.1f} lane={lane_label}{wait_tag}"
        )

    def step(
        self,
        dt: float,
        mocap_pos: np.ndarray,
        *,
        ball_visible: bool,
    ) -> None:
        if ball_visible:
            self._lost_time = 0.0
        else:
            self._lost_time += dt

        # The ball ALWAYS keeps moving on its path — previously it paused after
        # `_lost_grace_s` of being unseen, which caused the "robot freezes, ball
        # freezes, nothing happens" deadlock.  Keeping the ball in motion forces
        # the controller to actually search + re-acquire.
        if self._mode == "square":
            self._step_square(dt)
        elif self._mode == "circle":
            self._step_circle(dt)
        elif self._mode == "corridor":
            self._step_corridor(dt)
        else:
            self._step_figure8(dt)

        # ── Stage-based difficulty layers (corridor mode only) ───────────
        # Every 3 rows of obstacles the robot passes advances one stage,
        # which adds a new motion modifier to the ball trajectory:
        #   1: straight (baseline)
        #   2: lateral sine sway
        #   3: + faster ball
        #   4: + vertical bob (z height varies)
        #   5: + tight S-curves
        #   6: + spiral / orbiting motion
        #   7+: + wider sway + faster
        x_disp = self._x
        y_disp = self._y
        z_disp = 0.15
        stage_label = ""
        if self._mode == "corridor":
            ROW_GAP = float(os.environ.get("OBS_MIN_GAP", "5.4"))
            ROW_START_X = float(os.environ.get("OBS_MIN_X", "4.0"))
            n_rows_passed = max(0, int((self._robot_x - ROW_START_X) / ROW_GAP))
            stage = 1 + n_rows_passed // 3
            t = self._param_t
            # Stage 2+: lateral sine sway
            if stage >= 2:
                amp = 0.5 if stage < 7 else 1.0
                y_disp += amp * math.sin(0.6 * t)
            # Stage 3+: faster ball (handled in _step_corridor via speed mult)
            # Stage 4+: vertical bob ("upside and down")
            if stage >= 4:
                # Amplitude grows with stage so the path visibly climbs/dips.
                _z_amp = 0.20
                if stage >= 8:  _z_amp = 0.32
                if stage >= 12: _z_amp = 0.40
                z_disp = 0.15 + _z_amp * (0.5 + 0.5 * math.sin(0.9 * t))
            # Stage 5+: tight S-curves on top
            if stage >= 5:
                y_disp += 0.35 * math.sin(1.8 * t)
            # Stage 6+: spiral / orbit motion (extra y + z circle)
            if stage >= 6:
                y_disp += 0.30 * math.cos(1.2 * t)
                z_disp += 0.10 * math.sin(1.2 * t)
            # Cache stage so _step_corridor can speed up the ball
            self._stage = stage
            stage_label = f" stage={stage}"
            # Append to phase label for HUD readout
            if "stage=" not in self.phase_label:
                self.phase_label = f"{self.phase_label}{stage_label}"
        else:
            self._stage = 1

        # Clamp ball inside the corridor walls (y = ±2.58) and to a z range
        # the robot's head camera can see (floor to wall-top).
        y_disp = max(-2.2, min(2.2, y_disp))
        z_disp = max(0.10, min(0.45, z_disp))
        mocap_pos[0, 0] = x_disp
        mocap_pos[0, 1] = y_disp
        mocap_pos[0, 2] = z_disp


# Backwards-compatible name for imports / older notes
TargetMover = DemoTargetMover


# ── display helpers ──────────────────────────────────────────────────────────

def _world_to_px(wx: float, wy: float,
                 w: int = CAM_W, h: int = CAM_H,
                 half: float = ARENA_HALF) -> tuple[int, int]:
    """Map world (x, y) to pixel (px, py) in the top-down frame."""
    px = int((wx + half) / (2 * half) * w)
    py = int((half - wy) / (2 * half) * h)
    return (max(3, min(w - 4, px)), max(3, min(h - 4, py)))


def draw_trail(
    frame: np.ndarray,
    trail: collections.deque,
    color: tuple,
    radius: int = 2,
) -> None:
    """Draw position history as fading dots."""
    n = len(trail)
    for i, (wx, wy) in enumerate(trail):
        alpha = (i + 1) / n  # older = more transparent
        r = max(1, int(radius * alpha))
        px, py = _world_to_px(wx, wy)
        intensity = int(alpha * 180)
        faded = tuple(min(255, int(c * alpha + intensity * (1 - alpha)))
                      for c in color)
        cv2.circle(frame, (px, py), r, faded, -1)


def draw_heading_arrow(
    frame: np.ndarray,
    rx: float, ry: float,
    yaw: float,
    length: int = 22,
) -> None:
    x0, y0 = _world_to_px(rx, ry)
    x1 = x0 + int(length * math.cos(yaw))
    y1 = y0 - int(length * math.sin(yaw))    # y flipped in image
    cv2.arrowedLine(frame, (x0, y0), (x1, y1), (255, 220, 0), 2, tipLength=0.4)


def draw_distance_bar(
    frame: np.ndarray,
    area: float,
    conf: float,
    cx,
) -> None:
    """Draw a horizontal bar on the head-camera frame showing estimated distance."""
    # Background bar
    cv2.rectangle(frame, (BAR_X, BAR_Y),
                  (BAR_X + BAR_W, BAR_Y + BAR_H), (40, 40, 40), -1)
    cv2.rectangle(frame, (BAR_X, BAR_Y),
                  (BAR_X + BAR_W, BAR_Y + BAR_H), (120, 120, 120), 1)

    if cx is None or area == 0:
        cv2.putText(frame, "dist: --", (BAR_X, BAR_Y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
        return

    fill = int(BAR_W * min(area / MAX_AREA, 1.0))
    # Green zone marks the stand-off window
    green_lo = int(BAR_W * STANDOFF_LO / MAX_AREA)
    green_hi = int(BAR_W * STANDOFF_HI / MAX_AREA)

    bar_color = (0, 220, 50) if green_lo <= fill <= green_hi else (0, 140, 240)
    cv2.rectangle(frame, (BAR_X, BAR_Y),
                  (BAR_X + fill, BAR_Y + BAR_H), bar_color, -1)
    cv2.rectangle(frame, (BAR_X + green_lo, BAR_Y - 3),
                  (BAR_X + green_hi, BAR_Y + BAR_H + 3), (60, 230, 60), 1)

    # Approx distance from area (empirically calibrated)
    dist_m = 0.9 * math.sqrt(TARGET_AREA / max(area, 1))
    cv2.putText(frame, f"dist≈{dist_m:.2f}m", (BAR_X, BAR_Y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def draw_label(frame: np.ndarray, text: str, pos: tuple[int, int],
               color=(220, 240, 255)) -> None:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (20, 20, 20), 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1)


def hud(
    frame: np.ndarray,
    vx: float, vyaw: float,
    cx, area: float, conf: float,
    step: int, sim_time: float,
    rz: float, fell: bool,
    demo_phase: str = "",
) -> None:
    z_color = (0, 50, 255) if fell else (200, 240, 180)
    lines = [
        (f"step {step:06d}  t={sim_time:.1f}s",  (200, 240, 255)),
        (demo_phase[:52] if demo_phase else "(demo phase)", (180, 220, 255)),
        (f"vx={vx:+.2f}m/s  vyaw={vyaw:+.2f}r/s", (200, 240, 255)),
        (f"z={rz:.3f}m {'[FELL]' if fell else '      '}",  z_color),
        (f"conf={conf:.0%}  {'FOUND' if cx is not None else 'SEARCHING'}",
         (80, 220, 80) if cx is not None else (60, 80, 220)),
    ]
    y = 16
    for text, color in lines:
        draw_label(frame, text, (10, y), color)
        y += 18


def _inject_safety_zone_geoms(scene, obstacles, robot_safety: float) -> None:
    """For each obstacle in `obstacles` (list of (x, y, y_half)), draw a
    thin translucent yellow rectangle on the floor showing the no-go
    safety zone the robot's avoidance considers around it.  Conveys
    'detected — keep clear' visually."""
    OBS_X_HALF_VIZ = 0.20
    Z = 0.008  # below path strip (z=0.012) so the strip remains visible
    for (ox, oy, oy_half) in obstacles:
        if scene.ngeom >= scene.maxgeom:
            break
        size = np.array([
            OBS_X_HALF_VIZ + 0.10,    # +x margin
            oy_half + robot_safety,   # +y safety
            0.002,
        ], dtype=np.float64)
        center = np.array([ox, oy, Z], dtype=np.float64)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            int(mujoco.mjtGeom.mjGEOM_BOX),
            size,
            center,
            np.eye(3, dtype=np.float64).flatten(),
            np.array([1.0, 0.85, 0.15, 0.22], dtype=np.float64),
        )
        try:
            geom.emission = 0.55
            geom.specular = 0.0
            geom.shininess = 0.0
        except Exception:
            pass
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1


def _inject_safety_wall_geom(
    scene,
    ox: float,
    safe_y: float,
    obs_half_x: float = 0.40,
) -> None:
    """Thin green wall along an obstacle's safety boundary on the side
    facing the robot.  Marks 'the closest the robot can come to this
    obstacle without colliding'."""
    if scene.ngeom >= scene.maxgeom:
        return
    WALL_HALF_LEN = max(0.5, obs_half_x + 0.20)   # length along x
    WALL_HALF_W   = 0.02                          # very thin in y
    WALL_HALF_H   = 0.50
    geom = scene.geoms[scene.ngeom]
    size = np.array([WALL_HALF_LEN, WALL_HALF_W, WALL_HALF_H], dtype=np.float64)
    center = np.array([ox, safe_y, WALL_HALF_H], dtype=np.float64)
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_BOX),
        size,
        center,
        np.eye(3, dtype=np.float64).flatten(),
        np.array([0.2, 1.0, 0.3, 0.55], dtype=np.float64),   # translucent green
    )
    try:
        geom.emission = 0.5
        geom.specular = 0.0
        geom.shininess = 0.0
    except Exception:
        pass
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def _inject_passage_gate_geoms(
    scene,
    gate_x: float,
    band_lo: float,
    band_hi: float,
) -> None:
    """Two glowing green pillars planted on the EDGES of the free passage
    band the planner committed to.  The robot must stay strictly between
    them — the pillars are obstacles the robot cannot occupy."""
    if gate_x <= 0 or band_hi <= band_lo or scene.ngeom + 2 > scene.maxgeom:
        return
    PILLAR_R = 0.05
    PILLAR_HALF_H = 0.45
    for _y in (band_lo, band_hi):
        if scene.ngeom >= scene.maxgeom:
            break
        center = np.array([
            gate_x,
            _y,
            PILLAR_HALF_H,
        ], dtype=np.float64)
        # Cylinder size = (radius, half-length)
        size = np.array([PILLAR_R, PILLAR_HALF_H, 0.0], dtype=np.float64)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            size,
            center,
            np.eye(3, dtype=np.float64).flatten(),
            np.array([0.20, 0.95, 0.45, 0.85], dtype=np.float64),
        )
        try:
            geom.emission = 0.80
            geom.specular = 0.4
            geom.shininess = 0.3
        except Exception:
            pass
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1


def _inject_path_geoms(scene, waypoints, rgba=None) -> None:
    """Add box geoms to MuJoCo's scene to draw a glowing path strip on
    the floor along the given waypoints.  Because these geoms are part of
    the rendered scene, the z-buffer makes the robot correctly occlude
    the path where their projections overlap — the strip really *is* on
    the floor and disappears under the robot's body when in the way."""
    if len(waypoints) < 2:
        return
    PATH_HALF = 0.035  # 0.07 m wide strip — narrow + clean
    HEIGHT    = 0.004  # very thin slab
    Z         = 0.012  # above floor (no z-fighting)
    # Trim final 0.6 m so strip stops short of the ball.
    wps = [np.asarray(w, dtype=float) for w in waypoints]
    if len(wps) >= 2:
        d = wps[-1] - wps[-2]
        seg = float(np.linalg.norm(d[:2]))
        if seg > 0.7:
            wps[-1] = wps[-1] - (d / max(seg, 1e-6)) * 0.60
    # Densify for color gradient (~2 segments per metre).
    dense = []
    cum = []
    s = 0.0
    for i in range(1, len(wps)):
        p1, p2 = wps[i - 1], wps[i]
        seg_len = float(np.linalg.norm(p2[:2] - p1[:2]))
        steps = max(2, int(seg_len * 2))
        for j in range(steps):
            t = j / steps
            dense.append(p1 * (1 - t) + p2 * t)
            cum.append(s + t * seg_len)
        s += seg_len
    dense.append(wps[-1])
    cum.append(s)
    total = max(1e-3, cum[-1])
    n0 = scene.ngeom
    for i in range(1, len(dense)):
        if scene.ngeom >= scene.maxgeom:
            break
        p1, p2 = dense[i - 1], dense[i]
        dx = float(p2[0] - p1[0]); dy = float(p2[1] - p1[1])
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-3:
            continue
        # Box rotation: x along segment, z up.
        x_axis = np.array([dx / seg_len, dy / seg_len, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
        center = ((p1 + p2) / 2.0).astype(np.float64)
        center[2] = Z
        size = np.array([seg_len / 2.0, PATH_HALF, HEIGHT], dtype=np.float64)
        # Solid red path (matches the ball's accent color) — or override.
        if rgba is None:
            _rgba_use = np.array([0.95, 0.10, 0.12, 0.92], dtype=np.float64)
        else:
            _rgba_use = np.asarray(rgba, dtype=np.float64)
        rgba_local = _rgba_use; rgba = _rgba_use  # alias to keep below code working
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            int(mujoco.mjtGeom.mjGEOM_BOX),
            size,
            center,
            R.flatten(),
            rgba,
        )
        # Make it glow softly so it reads even on dark sections of floor.
        try:
            geom.emission = 0.75
            geom.specular = 0.0
            geom.shininess = 0.0
        except Exception:
            pass
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1
    # ── Arrowhead at the end of the strip (the actual arrow tip) ──
    # Two thin boxes forming a V, both pointing toward `tip`.  Sized so
    # the arrowhead is visibly wider than the shaft.
    if len(dense) >= 2 and scene.ngeom + 2 <= scene.maxgeom:
        tip = dense[-1]
        # Forward tangent of the final segment
        p_prev = dense[-2]
        dxf = float(tip[0] - p_prev[0])
        dyf = float(tip[1] - p_prev[1])
        seg = math.hypot(dxf, dyf)
        if seg > 1e-3:
            tx = dxf / seg
            ty = dyf / seg
            forward_angle = math.atan2(ty, tx)
            ARROW_LEN      = 0.36   # length of each leg of the V
            ARROW_LEG_HALF = 0.030  # leg thickness (perpendicular)
            HEAD_ANGLE_DEG = 38.0   # half-spread of the V from the back-tangent
            for sign in (+1, -1):
                # Direction the leg extends from the tip (going BACKWARD
                # along the path, splayed out by HEAD_ANGLE_DEG).
                leg_angle = (forward_angle + math.pi
                             + sign * math.radians(HEAD_ANGLE_DEG))
                lx = math.cos(leg_angle)
                ly = math.sin(leg_angle)
                # Box center sits halfway along the leg (so that one end
                # is at the tip and the other extends backward).
                center = np.array([
                    tip[0] + lx * ARROW_LEN / 2.0,
                    tip[1] + ly * ARROW_LEN / 2.0,
                    Z,
                ], dtype=np.float64)
                x_axis = np.array([lx, ly, 0.0])
                z_axis = np.array([0.0, 0.0, 1.0])
                y_axis = np.cross(z_axis, x_axis)
                R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
                size = np.array([ARROW_LEN / 2.0, ARROW_LEG_HALF, HEIGHT],
                                dtype=np.float64)
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    int(mujoco.mjtGeom.mjGEOM_BOX),
                    size,
                    center,
                    R.flatten(),
                    rgba_local,
                )
                try:
                    geom.emission = 0.85
                    geom.specular = 0.0
                    geom.shininess = 0.0
                except Exception:
                    pass
                geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
                scene.ngeom += 1


def _project_world_to_image(
    p_world, cam_pos, cam_mat, fovy_deg: float, img_w: int, img_h: int
):
    """Pinhole-project a 3-D world point onto the image plane of a MuJoCo
    camera whose pose is given by `cam_pos` (world) and `cam_mat` (3×3
    rotation, columns = camera right/up/back in world).  Returns (u, v)
    pixel coords, or None if the point is behind the camera or off-frame.
    """
    v = np.asarray(p_world, dtype=float) - np.asarray(cam_pos, dtype=float)
    cx = float(np.dot(v, cam_mat[:, 0]))
    cy = float(np.dot(v, cam_mat[:, 1]))
    cz = float(np.dot(v, cam_mat[:, 2]))
    if cz > -0.05:    # behind camera (MuJoCo cameras look along −z)
        return None
    f = (img_h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    u_px = img_w / 2.0 + f * cx / (-cz)
    v_px = img_h / 2.0 - f * cy / (-cz)
    if not (-50 <= u_px < img_w + 50 and -50 <= v_px < img_h + 50):
        return None
    return (int(round(u_px)), int(round(v_px)))


def _draw_path_overlay(
    frame: np.ndarray,
    waypoints: list,
    cam_pos, cam_mat,
    fovy_deg: float, img_w: int, img_h: int,
) -> None:
    """Render a glowing 3-D road-style strip on the floor connecting the
    waypoints.  The strip has real-world width that perspective-shrinks
    naturally with distance, with chevron arrowheads spaced at regular
    intervals along the path, and a soft target halo on the floor near
    (but not covering) the final ball position."""
    if len(waypoints) < 2:
        return

    # Trim the final segment so the path STOPS ≈0.6 m before the ball
    # (so the strip and chevrons don't visually cover the ball).
    wps = [np.asarray(w, dtype=float) for w in waypoints]
    last = wps[-1].copy()
    if len(wps) >= 2:
        d = wps[-1] - wps[-2]
        seg = float(np.linalg.norm(d))
        if seg > 0.7:
            wps[-1] = wps[-1] - (d / seg) * 0.60
    else:
        return

    # Densify (5 samples per meter) for smooth gradient + chevron spacing
    dense = []
    seg_t = []   # cumulative arc length per dense point
    cum = 0.0
    for i in range(1, len(wps)):
        p1, p2 = wps[i - 1], wps[i]
        seg_len = float(np.linalg.norm(p2 - p1))
        n_step = max(3, int(seg_len * 5))
        for j in range(n_step):
            t = j / max(1, n_step - 1)
            p = p1 * (1 - t) + p2 * t
            dense.append(p)
            seg_t.append(cum + t * seg_len)
        cum += seg_len
    total_len = max(0.01, cum)

    # Gradient: amber (near robot) → magenta → cyan (toward ball).
    def _grad(t: float):
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            f = t * 2.0
            return (int(60 + 60 * f), int(170 + 50 * f), int(255 - 30 * f))
        else:
            f = (t - 0.5) * 2.0
            return (int(120 + 135 * f), int(220 - 70 * f), int(225 - 60 * f))

    PATH_HALF = 0.09   # 0.18 m wide strip on the floor (narrower, cleaner)
    Z         = 0.012  # slightly above floor (no z-fighting)

    # Project a 3-D point to image coords (nullable).
    def _proj(p):
        return _project_world_to_image(p, cam_pos, cam_mat,
                                       fovy_deg, img_w, img_h)

    # Build strip quads on a separate overlay (semi-transparent blend).
    strip_layer = np.zeros_like(frame)
    edge_layer  = np.zeros_like(frame)
    for i in range(1, len(dense)):
        p1, p2 = dense[i - 1], dense[i]
        dxw, dyw = p2[0] - p1[0], p2[1] - p1[1]
        seg = math.hypot(dxw, dyw)
        if seg < 1e-4:
            continue
        nx, ny = -dyw / seg, dxw / seg
        corners = [
            np.array([p1[0] + nx * PATH_HALF, p1[1] + ny * PATH_HALF, Z]),
            np.array([p1[0] - nx * PATH_HALF, p1[1] - ny * PATH_HALF, Z]),
            np.array([p2[0] - nx * PATH_HALF, p2[1] - ny * PATH_HALF, Z]),
            np.array([p2[0] + nx * PATH_HALF, p2[1] + ny * PATH_HALF, Z]),
        ]
        proj = [_proj(c) for c in corners]
        if any(p is None for p in proj):
            continue
        poly = np.array(proj, dtype=np.int32)
        col = _grad(seg_t[i] / total_len)
        cv2.fillPoly(strip_layer, [poly], col, cv2.LINE_AA)
        # Bright edge lines along the strip's two long sides.
        cv2.line(edge_layer, proj[0], proj[3], (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(edge_layer, proj[1], proj[2], (255, 255, 255), 1, cv2.LINE_AA)

    # Composite: glow (blurred strip) + sharp strip + edges.
    glow = cv2.GaussianBlur(strip_layer, (15, 15), 0)
    cv2.addWeighted(glow, 0.55, frame, 1.0, 0, frame)
    cv2.addWeighted(strip_layer, 0.45, frame, 0.85, 0, frame)
    cv2.addWeighted(edge_layer, 0.35, frame, 1.0, 0, frame)

    # Chevron arrowheads every 1.2 m along the path.
    next_chev = 0.8
    for i in range(1, len(dense)):
        if seg_t[i] < next_chev:
            continue
        next_chev += 1.2
        if seg_t[i] > total_len - 0.5:   # don't drop a chevron at very end
            break
        p_here = dense[i]
        # Tangent (xy) at this point
        p_prev = dense[i - 1]
        dx_t, dy_t = p_here[0] - p_prev[0], p_here[1] - p_prev[1]
        seg = math.hypot(dx_t, dy_t) or 1.0
        tx, ty = dx_t / seg, dy_t / seg
        nx, ny = -ty, tx
        # Chevron in 3-D (so it scales with perspective)
        ah_len = 0.30
        ah_wid = 0.14
        tip   = np.array([p_here[0] + tx * ah_len, p_here[1] + ty * ah_len, Z])
        base_l = np.array([p_here[0] + nx * ah_wid, p_here[1] + ny * ah_wid, Z])
        base_r = np.array([p_here[0] - nx * ah_wid, p_here[1] - ny * ah_wid, Z])
        proj = [_proj(p) for p in (tip, base_l, base_r)]
        if any(p is None for p in proj):
            continue
        poly = np.array(proj, dtype=np.int32)
        col = _grad(seg_t[i] / total_len)
        # White-filled chevron with colored outline for high contrast on
        # any background.
        cv2.fillPoly(frame, [poly], (240, 245, 250), cv2.LINE_AA)
        cv2.polylines(frame, [poly], True, col, 2, cv2.LINE_AA)

    # Subtle target halo on the floor BESIDE the ball (not covering it).
    # Use the un-trimmed last waypoint to project the ball's true xy.
    ball_floor = np.array([last[0], last[1], Z])
    bp = _proj(ball_floor)
    if bp is not None:
        cu, cv_p = bp
        # Two soft rings, brighter outside, fading inward
        for r_pix, col, thk in [(34, (40, 230, 255), 2),
                                (24, (180, 255, 255), 1)]:
            cv2.circle(frame, (cu, cv_p), r_pix, col, thk, cv2.LINE_AA)


def _apply_vignette(frame: np.ndarray, strength: float = 0.55) -> None:
    """Multiply a soft radial darkening mask onto the frame edges for a
    cinematic 'spotlight' effect.  Cached on the function to avoid
    rebuilding per-frame."""
    h, w = frame.shape[:2]
    cache = _apply_vignette.__dict__.get("_cache")
    if cache is None or cache.shape[:2] != (h, w):
        Y, X = np.ogrid[:h, :w]
        cx_, cy_ = w / 2.0, h / 2.0
        # Ellipse-shaped falloff
        r2 = ((X - cx_) / (w * 0.55)) ** 2 + ((Y - cy_) / (h * 0.55)) ** 2
        m = np.clip(1.0 - strength * r2, 0.45, 1.0).astype(np.float32)
        cache = m[:, :, None]
        _apply_vignette._cache = cache
    np.multiply(frame, cache, out=frame, casting="unsafe")


def _draw_styled_hud(
    frame: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    """Sleek title bar at the top of the chase view.  No bottom pill bar
    — all live telemetry is shown in the dedicated dashboard panel, so
    the bottom of the chase view stays clean."""
    h, w = frame.shape[:2]

    # ── Top title bar — slightly taller (52 px) so two lines breathe ─
    top_h = 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.line(frame, (0, top_h), (w, top_h), (60, 220, 255), 2, cv2.LINE_AA)
    # Cyan accent strip on the left edge — gives the title a "panel" feel.
    cv2.rectangle(frame, (0, 0), (4, top_h), (60, 220, 255), -1)

    # Auto-fit font scale so text never clips, regardless of frame width.
    def _fit_scale(txt: str, base: float, max_px: int, min_scale: float) -> float:
        sc = base
        while sc > min_scale:
            tw, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0]
            if tw <= max_px:
                return sc
            sc -= 0.02
        return min_scale

    avail = w - 24
    t_sc = _fit_scale(title,    base=0.66, max_px=avail, min_scale=0.40)
    s_sc = _fit_scale(subtitle, base=0.42, max_px=avail, min_scale=0.30)

    # Title (bold, white).
    cv2.putText(frame, title, (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, t_sc, (20, 20, 25), 4, cv2.LINE_AA)
    cv2.putText(frame, title, (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, t_sc, (245, 250, 255), 1, cv2.LINE_AA)
    # Subtitle (smaller, dim cyan).
    cv2.putText(frame, subtitle, (16, 44),
                cv2.FONT_HERSHEY_SIMPLEX, s_sc, (20, 20, 25), 3, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (16, 44),
                cv2.FONT_HERSHEY_SIMPLEX, s_sc, (160, 220, 255), 1, cv2.LINE_AA)


def _overlay_robot_vision_pip(
    chase_bgr: np.ndarray,
    head_rgb: np.ndarray,
    det,
    sim_t: float,
) -> None:
    """Draw a 'ROBOT VISION' picture-in-picture in the top-right corner of
    the chase view, showing the robot's head-camera feed with the ball
    detection box overlaid.  Adds a label strip with a recording dot."""
    H, W = chase_bgr.shape[:2]

    # PIP size — ~30% of frame width, 4:3 aspect.
    pip_w = max(120, int(W * 0.30))
    pip_h = int(pip_w * 3 / 4)

    # Build clean PIP source (no debug HUD), with detection annotation.
    pip = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    try:
        pip = annotate_frame(pip, det)
    except Exception:
        pass
    pip = cv2.resize(pip, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

    # Placement: top-right corner, just below title bar (44 px) + 8 px gap.
    margin = 10
    title_h = 44
    label_h = 18
    x0 = W - pip_w - margin
    y0 = title_h + 8 + label_h        # leave room for label above the PIP
    x1 = x0 + pip_w
    y1 = y0 + pip_h

    # Label strip ("ROBOT VISION  ● REC") above the PIP.
    lx0, ly0 = x0, y0 - label_h
    lx1, ly1 = x1, y0
    overlay = chase_bgr.copy()
    cv2.rectangle(overlay, (lx0, ly0), (lx1, ly1), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.78, chase_bgr, 0.22, 0, chase_bgr)
    cv2.rectangle(chase_bgr, (lx0, ly0), (lx1, ly1), (60, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(chase_bgr, "ROBOT VISION",
                (lx0 + 8, ly1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 245, 255), 1, cv2.LINE_AA)
    # Pulsing red REC dot.
    rec_blink = (int(sim_t * 2) % 2 == 0)
    rec_col = (40, 50, 235) if rec_blink else (60, 80, 200)
    cv2.circle(chase_bgr, (lx1 - 38, ly0 + label_h // 2), 3, rec_col, -1, cv2.LINE_AA)
    cv2.putText(chase_bgr, "REC",
                (lx1 - 30, ly1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, rec_col, 1, cv2.LINE_AA)

    # PIP body.
    chase_bgr[y0:y1, x0:x1] = pip
    # Cyan accent border (matches the styled HUD borders).
    cv2.rectangle(chase_bgr, (x0 - 1, y0 - 1), (x1, y1), (60, 220, 255), 1, cv2.LINE_AA)
    # Subtle inner shadow for depth.
    cv2.rectangle(chase_bgr, (x0, y0), (x1 - 1, y1 - 1), (30, 30, 30), 1, cv2.LINE_AA)


# ── dashboard composite helpers ─────────────────────────────────────────────

_DARK = (12, 16, 22)
_PANEL = (18, 22, 28)
_CYAN = (60, 220, 255)
_TXT_HI = (235, 245, 255)
_TXT_LO = (160, 175, 195)


def _resize_to_tile(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fit a BGR image into (w, h) preserving aspect with letterbox bars
    in the HUD's dark panel color."""
    src_h, src_w = img_bgr.shape[:2]
    scale = min(w / src_w, h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tile = np.full((h, w, 3), _DARK, dtype=np.uint8)
    ox, oy = (w - new_w) // 2, (h - new_h) // 2
    tile[oy:oy + new_h, ox:ox + new_w] = resized
    return tile


def _draw_tile_label(tile: np.ndarray, label: str,
                     accent: tuple = (60, 220, 255)) -> None:
    """Bottom-left ribbon on a camera tile with the panel name."""
    h, w = tile.shape[:2]
    rh = 22
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, h - rh), (w, h), _DARK, -1)
    cv2.addWeighted(overlay, 0.78, tile, 0.22, 0, tile)
    cv2.line(tile, (0, h - rh), (w, h - rh), accent, 1, cv2.LINE_AA)
    cv2.putText(tile, label, (8, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, _TXT_HI, 1, cv2.LINE_AA)
    # Cyan accent border, full tile.
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), accent, 1, cv2.LINE_AA)


def _build_camera_tile(rgb: np.ndarray, label: str, w: int, h: int,
                       det=None, accent: tuple = (60, 220, 255)) -> np.ndarray:
    """Convert RGB → BGR → letterboxed tile of (w,h) with label ribbon
    and (optional) ball-detection box from the tracker."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if det is not None:
        try:
            bgr = annotate_frame(bgr, det)
        except Exception:
            pass
    tile = _resize_to_tile(bgr, w, h)
    _draw_tile_label(tile, label, accent=accent)
    return tile


def _build_map_tile(rgb: np.ndarray, label: str, w: int, h: int,
                    robot_xy: tuple, ball_xy: tuple, yaw: float,
                    accent: tuple = (60, 220, 255)) -> np.ndarray:
    """Top-down map tile.  Renders the top-down camera view (already
    centered on the robot) and overlays a heading triangle at center."""
    tile = _build_camera_tile(rgb, label, w, h, accent=accent)
    # Center of the view (where the robot is, since the map cam tracks it).
    cx, cy = w // 2, h // 2
    # Heading triangle (robot's forward direction in world frame).
    L = 14
    # World +x is "up" in the rendered map (camera azimuth=90, looking down).
    # Robot heading vector (yaw) maps to (cos yaw, sin yaw) in world; the
    # rendered map has x-world → image up, y-world → image left.
    #   image_dx = -sin(yaw), image_dy = -cos(yaw)
    hx = -math.sin(yaw)
    hy = -math.cos(yaw)
    px = cx + L * hx
    py = cy + L * hy
    rx = -hy   # right-perp
    ry = hx
    p_apex = (int(px), int(py))
    p_l = (int(cx + 0.5 * rx * 8 - 0.4 * hx * 8),
           int(cy + 0.5 * ry * 8 - 0.4 * hy * 8))
    p_r = (int(cx - 0.5 * rx * 8 - 0.4 * hx * 8),
           int(cy - 0.5 * ry * 8 - 0.4 * hy * 8))
    cv2.fillPoly(tile, [np.array([p_apex, p_l, p_r])],
                 (90, 230, 255), cv2.LINE_AA)
    cv2.circle(tile, (cx, cy), 3, (30, 30, 30), -1, cv2.LINE_AA)
    return tile


def _build_info_tile(w: int, h: int, *,
                     sim_t: float, n_passed: int, total_rows: int,
                     vx_cmd: float, vyaw_cmd: float,
                     det_conf: float, ball_dist: float,
                     lane_label: str, robot_yaw: float,
                     robot_x: float,
                     fell: bool) -> np.ndarray:
    """Bottom-right square info panel — telemetry + heading compass."""
    tile = np.full((h, w, 3), _PANEL, dtype=np.uint8)

    # Header bar.
    rh = 26
    cv2.rectangle(tile, (0, 0), (w, rh), _DARK, -1)
    cv2.line(tile, (0, rh), (w, rh), _CYAN, 1, cv2.LINE_AA)
    cv2.putText(tile, "TELEMETRY", (10, rh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, _TXT_HI, 1, cv2.LINE_AA)
    # Pulsing rec dot at top-right of header.
    rec_blink = (int(sim_t * 2) % 2 == 0)
    rec_col = (40, 50, 235) if rec_blink else (60, 80, 200)
    cv2.circle(tile, (w - 22, rh // 2), 3, rec_col, -1, cv2.LINE_AA)
    cv2.putText(tile, "REC", (w - 14, rh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, rec_col, 1, cv2.LINE_AA)

    # Two-column key:value rows.
    rows = [
        ("TIME",    f"{int(sim_t // 60):02d}:{int(sim_t % 60):02d}"),
        ("POS X",   f"{robot_x:.1f} m"),
        ("ROW",     f"{n_passed} / {total_rows}"),
        ("SPEED",   f"{vx_cmd:+.2f} m/s"),
        ("YAW RATE",f"{vyaw_cmd:+.2f} r/s"),
        ("DETECT",  f"{int(det_conf * 100)} %"),
        ("BALL",    f"{ball_dist:.2f} m"),
        ("LANE",    lane_label),
        ("STATE",   "FALLEN" if fell else "OK"),
    ]
    y = rh + 14
    line_h = 17
    for k, v in rows:
        cv2.putText(tile, k, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _TXT_LO, 1, cv2.LINE_AA)
        cv2.putText(tile, v, (88, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, _TXT_HI, 1, cv2.LINE_AA)
        y += line_h

    # Heading compass — small dial in the bottom-right corner.
    cx, cy = w - 36, h - 36
    R = 22
    cv2.circle(tile, (cx, cy), R, (40, 50, 65), 1, cv2.LINE_AA)
    cv2.circle(tile, (cx, cy), R - 1, (28, 32, 42), -1, cv2.LINE_AA)
    cv2.circle(tile, (cx, cy), R, _CYAN, 1, cv2.LINE_AA)
    # Cardinal letters.
    for txt, dx, dy in [("N", 0, -R + 6), ("S", 0, R - 2),
                        ("E", R - 6, 4),   ("W", -R + 4, 4)]:
        cv2.putText(tile, txt, (cx + dx - 3, cy + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, _TXT_LO, 1, cv2.LINE_AA)
    # Heading triangle (robot yaw — assume world +x = "up" = N).
    hx = -math.sin(robot_yaw)
    hy = -math.cos(robot_yaw)
    px = cx + (R - 4) * hx
    py = cy + (R - 4) * hy
    rxv, ryv = -hy, hx
    p_apex = (int(px), int(py))
    p_l = (int(cx + 0.4 * rxv * 6 - 0.3 * hx * 6),
           int(cy + 0.4 * ryv * 6 - 0.3 * hy * 6))
    p_r = (int(cx - 0.4 * rxv * 6 - 0.3 * hx * 6),
           int(cy - 0.4 * ryv * 6 - 0.3 * hy * 6))
    cv2.fillPoly(tile, [np.array([p_apex, p_l, p_r])],
                 (90, 230, 255), cv2.LINE_AA)

    # Border.
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), _CYAN, 1, cv2.LINE_AA)
    return tile


def _build_dashboard(chase_bgr: np.ndarray,
                     front_rgb: np.ndarray, front_det,
                     map_rgb: np.ndarray,
                     left_rgb: np.ndarray,
                     right_rgb: np.ndarray,
                     *, info_kwargs: dict) -> np.ndarray:
    """Compose the 720x720 dashboard:
        +---------+----+
        | CHASE   | F  |  F = front (head_cam)
        | 480x480 +----+  M = 2D map
        |         | M  |
        +---+-----+----+
        | L | R   | I  |  L=left, R=right, I=info
        +---+-----+----+
    Each small tile is 240x240; chase fills the top-left 2x2.
    """
    T = 720
    cell = 240
    canvas = np.full((T, T, 3), _DARK, dtype=np.uint8)

    # Chase: 480x480 (already squared by the dedicated chase_renderer).
    if chase_bgr.shape[0] != 2 * cell or chase_bgr.shape[1] != 2 * cell:
        chase_bgr = cv2.resize(chase_bgr, (2 * cell, 2 * cell),
                               interpolation=cv2.INTER_AREA)
    canvas[0:2 * cell, 0:2 * cell] = chase_bgr

    # Pop the layout-only fields so they don't leak into _build_info_tile.
    robot_xy = info_kwargs.pop("robot_xy")
    ball_xy  = info_kwargs.pop("ball_xy")

    # Top-right column: front + 2D map.
    canvas[0:cell, 2 * cell:T] = _build_camera_tile(
        front_rgb, "FRONT VISION (head_cam)", cell, cell,
        det=front_det, accent=(80, 220, 120))
    canvas[cell:2 * cell, 2 * cell:T] = _build_map_tile(
        map_rgb, "2D MAP (top-down)", cell, cell,
        robot_xy=robot_xy,
        ball_xy=ball_xy,
        yaw=info_kwargs["robot_yaw"],
        accent=(255, 180, 80))

    # Bottom row: full-width info/telemetry panel (3-camera layout).
    canvas[2 * cell:T, 0:T] = _build_info_tile(T, cell, **info_kwargs)

    return canvas


# ── main ─────────────────────────────────────────────────────────────────────

def _reset(
    model,
    data,
    ramp_start: list,
    gait: LowLevelBase,
    tracker: BallTracker,
    controller: FollowController,
    robot_trail: collections.deque,
    ball_trail: collections.deque,
    mover: DemoTargetMover,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    gait.reset_phase(0.0)
    ramp_start[0] = data.time
    tracker.reset()
    controller.reset()
    robot_trail.clear()
    ball_trail.clear()
    mover.reset()
    print("Reset.")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    cam_w, cam_h = CAM_W, CAM_H
    if HEADLESS:
        cam_w, cam_h = CAM_W, CAM_H  # same resolution as GUI for accurate benchmarks
    renderer = mujoco.Renderer(model, cam_h, cam_w)
    head_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    top_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
    chase_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "chase_cam")
    rear_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rear_cam")
    side_left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_left_cam")
    side_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_right_cam")

    # ── Dedicated square renderer for the chase view (480x480) ──
    # The composite dashboard places chase in a 480x480 cell.  Rendering
    # at native square aspect avoids letterbox bars or vertical squash.
    DASH_CHASE_PX = 480
    chase_renderer = mujoco.Renderer(model, DASH_CHASE_PX, DASH_CHASE_PX)
    # Smaller renderer for the dashboard side-tiles (240x240).
    DASH_TILE_PX = 240
    tile_renderer = mujoco.Renderer(model, DASH_TILE_PX, DASH_TILE_PX)
    # Tile + head renderers: shadows/reflections invisible at 240x240 / 480x360,
    # so disable them for ~15-25% render-speed gain with no visible quality loss.
    for _r in (tile_renderer, renderer):
        _r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        _r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    # 2D-map free camera that follows the robot from above.
    map_cam_free = mujoco.MjvCamera()
    map_cam_free.type = mujoco.mjtCamera.mjCAMERA_FREE
    map_cam_free.azimuth   = 90.0     # +x → image up
    map_cam_free.elevation = -90.0    # straight down
    map_cam_free.distance  = 14.0     # 14 m field of view
    map_cam_free.lookat[:] = (0.0, 0.0, 0.0)

    # ── Wall-mounted CCTV-style side cameras ──
    # Mounted on the corridor walls (y = ±2.5 m), height 1.0 m, looking
    # inward at the robot.  Their lookat tracks a heavy-LPF'd robot_x so
    # the view PANS along the wall as the robot walks past — but locks y
    # to the corridor centerline (y=0) and locks z so wall flexure /
    # gait bobbing never enter the frame.
    #
    # MuJoCo free-cam convention (verified via the chase cam):
    #   camera_position = lookat - distance * (cos(el)cos(az),
    #                                           cos(el)sin(az),
    #                                           sin(el))
    # For "camera on +y wall looking toward -y" we need camera at +y of
    # lookat, so the (cam→lookat) unit-vector has -y component → az=-90.
    # Symmetric for the -y wall (az=+90).
    left_wall_cam_free = mujoco.MjvCamera()
    left_wall_cam_free.type = mujoco.mjtCamera.mjCAMERA_FREE
    left_wall_cam_free.azimuth   = -90.0
    left_wall_cam_free.elevation = -15.0
    left_wall_cam_free.distance  = 2.7
    left_wall_cam_free.lookat[:] = (0.0, 0.0, 0.30)

    right_wall_cam_free = mujoco.MjvCamera()
    right_wall_cam_free.type = mujoco.mjtCamera.mjCAMERA_FREE
    right_wall_cam_free.azimuth   = 90.0
    right_wall_cam_free.elevation = -15.0
    right_wall_cam_free.distance  = 2.7
    right_wall_cam_free.lookat[:] = (0.0, 0.0, 0.30)

    # ── Stabilized chase camera ──
    # A free MjvCamera whose `lookat` is a heavily low-pass-filtered
    # version of the robot's body position.  The z-coordinate is locked
    # to a fixed value so the gait's vertical bobbing doesn't shake the
    # view.  Use this in place of the body-attached chase_cam for the
    # presentation render.
    _trunk_body_name = "base"
    _trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _trunk_body_name)
    if _trunk_id < 0:
        # Fall back: try common alternative names
        for _name in ("trunk", "torso", "body"):
            _trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _name)
            if _trunk_id >= 0:
                break
    chase_cam_free = mujoco.MjvCamera()
    chase_cam_free.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Match the original body-attached chase view:
    # camera 2.5 m behind robot in -x direction, 1.2 m above,
    # looking forward at the robot with slight downward tilt.
    # MuJoCo free-camera convention (verified empirically):
    #   pos = lookat + distance * (cos(el)cos(az), cos(el)sin(az), sin(el))
    # For camera at -x of lookat AND above: az=180, el=+25.6
    chase_cam_free.azimuth   = 0.0
    chase_cam_free.elevation = -25.6
    chase_cam_free.distance  = 2.77
    chase_cam_free.lookat[:] = (0.0, 0.0, 0.30)
    _chase_smooth_xy = None  # filled lazily on first frame

    tracker    = BallTracker()
    controller = FollowController()
    _run_tag   = os.environ.get("RUN_TAG", "default")
    logger     = SimLogger(tag=_run_tag)
    mover      = TargetMover()
    gait: LowLevelBase = default_low_level(model)

    # ── Randomize obstacle positions (corridor mode) ─────────────────────
    # Two obstacle groups:
    #   TALL (16 slots): 0.4 m tall.  Both ball and robot must avoid.
    #   SHORT (6 slots): 0.12 m tall.  Ball "passes over" (its avoidance
    #     logic ignores them), but the robot MUST go around since legs can't
    #     safely step over.
    # Discover all 100 varied-width obstacles + their per-obstacle
    # y_half (lateral half-width), z_half (vertical half-extent), and
    # category (tall vs short).  Avoidance uses per-obstacle y_half so
    # the detour distance is correct for each width.
    obstacle_mocap_ids: list[int] = []
    short_obstacle_mocap_ids: list[int] = []   # subset that's "short" (z=0.12)
    obstacle_y_half_by_mid: dict[int, float] = {}
    obstacle_is_tall_by_mid: dict[int, bool] = {}
    obstacle_geom_ids: set[int] = set()       # geom ids of all vobs_* boxes (for collision detection)
    # Wall geom IDs — any contact with these also counts as a collision/fail.
    wall_geom_ids: set[int] = set()
    for _wname in ("corridor_wall_north", "corridor_wall_south", "corridor_wall_back"):
        _wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _wname)
        if _wid >= 0:
            wall_geom_ids.add(_wid)
    for i in range(1, 201):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"varied_obs_{i}")
        if bid < 0:
            continue
        mid = int(model.body_mocapid[bid])
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"vobs_{i}")
        y_half = float(model.geom_size[gid, 1]) if gid >= 0 else 0.30
        z_half = float(model.geom_size[gid, 2]) if gid >= 0 else 0.40
        is_tall = (z_half > 0.20)
        obstacle_y_half_by_mid[mid] = y_half
        obstacle_is_tall_by_mid[mid] = is_tall
        obstacle_mocap_ids.append(mid)
        if gid >= 0:
            obstacle_geom_ids.add(gid)
        if not is_tall:
            short_obstacle_mocap_ids.append(mid)

    _ball_mode = os.environ.get("BALL_PATH_MODE", "").lower()
    if _ball_mode == "corridor" and obstacle_mocap_ids:
        import random
        seed_env = os.environ.get("CORRIDOR_SEED", "")
        if seed_env:
            random.seed(int(seed_env))
        n_rows      = int(os.environ.get("OBS_ROWS", "50"))   # 50 rows × 2 obstacles = 100
        n_rows      = max(1, min(len(obstacle_mocap_ids) // 2, n_rows))
        obs_min_x   = float(os.environ.get("OBS_MIN_X", "5.0"))
        obs_max_x   = float(os.environ.get("OBS_MAX_X", "275.0"))
        obs_min_gap = float(os.environ.get("OBS_MIN_GAP", "5.4"))

        # Each ROW has 2 obstacles at the same x.  Pair widest body with
        # narrowest body so each row's total y-extent stays manageable
        # (max wide-narrow pair: 1.88 + 0.31 = 2.19 m, fits in 5 m corridor).
        bodies_sorted = sorted(obstacle_mocap_ids,
                               key=lambda m: obstacle_y_half_by_mid[m])
        pairs = []
        n_total = len(bodies_sorted)
        for k in range(min(n_rows, n_total // 2)):
            wide_mid   = bodies_sorted[n_total - 1 - k]   # widest
            narrow_mid = bodies_sorted[k]                  # narrowest
            pairs.append((wide_mid, narrow_mid))

        # Even-spaced row x positions with small jitter.
        usable = obs_max_x - obs_min_x
        spacing = max(obs_min_gap, usable / max(n_rows, 1))
        jitter = min(0.30, spacing * 0.10)
        xs = []
        for i in range(n_rows):
            x_nominal = obs_min_x + (i + 0.5) * spacing
            x = x_nominal + random.uniform(-jitter, +jitter)
            if x > obs_max_x:
                xs.append(None)
            else:
                xs.append(round(x, 2))

        # Place each row with VARIED y positions for visual diversity:
        # obstacles can be on either side OR in the middle, and the
        # 2-obstacle pair must be SEPARATED enough that they look like
        # distinct objects (not connected/overlapping).  At least one
        # passage of body+safety width must remain for the robot.
        obstacle_world_positions = []  # (x, y, y_half) for ball mover
        all_robot_obstacles = []
        used_mids = set()

        def _largest_free_band(blocked_zones):
            """Given list of (lo, hi) blocked y-intervals, return the
            largest contiguous free segment within corridor [-2.5, +2.5]."""
            zones = sorted(blocked_zones)
            free = []
            prev = -2.5
            for lo, hi in zones:
                if lo > prev:
                    free.append(lo - prev)
                prev = max(prev, hi)
            if 2.5 > prev:
                free.append(2.5 - prev)
            return max(free) if free else 0.0

        # Curving-corridor offset: shift each row's center laterally along a
        # sine wave whose amplitude scales with the row's stage.  Builds a
        # serpentine passage that bends harder as the run progresses.
        _CURVE_X0 = float(os.environ.get("OBS_MIN_X", "4.0"))
        def _row_lane_center(row_idx: int, xp: float) -> float:
            row_stage = 1 + row_idx // 3
            # Two-tier curve fitting inside the 5 m corridor (walls at ±2.58):
            #  - high-freq S-curve (snake-like wobble)
            #  - low-freq path swing (gentle turn left/right over many rows)
            amp = 0.0      # high-freq amplitude
            rot_amp = 0.0  # low-freq path-rotation amplitude
            if row_stage >= 3:  amp = 0.4
            if row_stage >= 6:  amp = 0.6; rot_amp = 0.5
            if row_stage >= 9:  amp = 0.8; rot_amp = 0.7
            if row_stage >= 12: amp = 1.0; rot_amp = 0.8
            if row_stage >= 15: amp = 1.1; rot_amp = 1.0
            ds = (xp - _CURVE_X0)
            return amp * math.sin(0.18 * ds) + rot_amp * math.sin(0.04 * ds)

        for row_idx, (xp, (wide_mid, narrow_mid)) in enumerate(zip(xs, pairs)):
            if xp is None:
                continue
            h_w = obstacle_y_half_by_mid[wide_mid]
            h_n = obstacle_y_half_by_mid[narrow_mid]
            row_lane_center = _row_lane_center(row_idx, float(xp))

            # Allowed y-range for each (just inside the wall).
            y_w_max_abs = max(0.0, 2.30 - h_w)
            y_n_max_abs = max(0.0, 2.30 - h_n)

            # Try random placements until both feasibility constraints met:
            #   1. Obstacle edges separated by ≥ MIN_EDGE_GAP (0.40 m)
            #      so the pair looks like two distinct objects.
            #   2. At least one free band ≥ ROBOT_BODY (0.65 m) so the
            #      robot can pass.
            MIN_EDGE_GAP   = 0.40
            ROBOT_BODY     = 0.65   # body+small buffer
            PLACE_SAFETY   = 0.45   # mirror of ROBOT_SAFETY in avoidance
            wide_y = narrow_y = 0.0
            placed = False
            for _ in range(60):
                wide_y = random.uniform(-y_w_max_abs, y_w_max_abs)
                narrow_y = random.uniform(-y_n_max_abs, y_n_max_abs)
                # Edge separation: distance between centers minus their half-widths
                edge_gap = abs(wide_y - narrow_y) - (h_w + h_n)
                if edge_gap < MIN_EDGE_GAP:
                    continue
                # Feasibility: largest free band (with safety) ≥ robot body
                blocked = [
                    (wide_y   - h_w - PLACE_SAFETY, wide_y   + h_w + PLACE_SAFETY),
                    (narrow_y - h_n - PLACE_SAFETY, narrow_y + h_n + PLACE_SAFETY),
                ]
                if _largest_free_band(blocked) >= ROBOT_BODY:
                    placed = True
                    break
            if not placed:
                # Fallback: same-side guaranteed-clear-opposite layout.
                sign = +1 if (row_idx % 2 == 0) else -1
                wide_y   = sign * max(h_w + 0.10, 2.30 - h_w)
                narrow_y = sign * max(h_n + 0.20, 0.30)

            for mid, oy, oy_half in [
                (wide_mid,   wide_y,   h_w),
                (narrow_mid, narrow_y, h_n),
            ]:
                # Shift the obstacle by the row's lane-center, then clamp so
                # the obstacle still fits inside the corridor (walls at y = ±2.58).
                shifted_y = oy + row_lane_center
                _wall_lim = 2.30 - oy_half
                shifted_y = max(-_wall_lim, min(_wall_lim, shifted_y))
                is_tall = obstacle_is_tall_by_mid[mid]
                data.mocap_pos[mid, 0] = float(xp)
                data.mocap_pos[mid, 1] = shifted_y
                data.mocap_pos[mid, 2] = 0.40 if is_tall else 0.12
                obstacle_world_positions.append((float(xp), shifted_y, oy_half))
                all_robot_obstacles.append(("tall" if is_tall else "short",
                                            float(xp), shifted_y))
                used_mids.add(mid)

        # Push unused obstacles off-map.
        for mid in obstacle_mocap_ids:
            if mid not in used_mids:
                data.mocap_pos[mid, 0] = 300.0
                data.mocap_pos[mid, 1] = 300.0
                data.mocap_pos[mid, 2] = -5.0

        mover.set_corridor_obstacles(obstacle_world_positions)
        n_placed = len(all_robot_obstacles)
        widths   = [yh * 2 for _, _, yh in obstacle_world_positions]
        print(f"[CORRIDOR] placed {n_placed} obstacles in "
              f"{n_placed // 2} rows of 2  "
              f"widths: min={min(widths):.2f}m max={max(widths):.2f}m "
              f"mean={sum(widths)/len(widths):.2f}m",
              flush=True)
    else:
        all_robot_obstacles = []
        for mid in obstacle_mocap_ids:
            data.mocap_pos[mid, 0] = 200.0
            data.mocap_pos[mid, 1] = 200.0
            data.mocap_pos[mid, 2] = -5.0

    ramp_start  = [data.time]          # simulation time when ramp started
    RAMP_DUR    = 1.0                  # seconds for standup ramp (see TrotPDGait.ramp_dur)

    vx_cmd = vyaw_cmd = 0.0
    # ── EMA smoothing for velocity commands ───────────────────────────────────
    # Smoothing reduces rapid oscillations (e.g. vx toggling ±) that confuse the
    # RL gait and cause falls.  Alpha=1 → no smoothing; lower → heavier filter.
    # Both vx and vyaw use the same EMA alpha so that when the ball leaves the
    # deadband (ball suddenly off-centre after walking straight), vx drops
    # quickly as the turn-reduction formula reduces it.  The old VX_SMOOTH=0.30
    # let vx linger at ~0.38 m/s for 5+ steps while vyaw was already at -0.70,
    # causing the (fwd+spin) instability that produced all observed falls.
    VX_SMOOTH:   float = 0.60    # EMA alpha for forward velocity — higher = snappier response.
                                 #   Time constant ≈ CTRL_DT * (1-α)/α ≈ 13 ms at 50 Hz.
    # Yaw EMA is effectively disabled: the controller now emits bang-bang
    # (-YAW_HIGH, 0, +YAW_HIGH) values chosen specifically to avoid the 0.3–0.9
    # unstable dead zone.  Any smoothing would send transient commands INTO that
    # dead zone for several steps each time the yaw state flips.
    VYAW_SMOOTH: float = 0.35    # EMA: ~3-step time constant, smoothness without losing responsiveness
    LOG_EVERY = 50               # print a verbose line every N control steps

    paused   = False
    running  = True
    step_num = 0
    ball_visible_prev = True

    # Auto-reset tracker after prolonged no-detection so CamShift can re-acquire
    # after the ball leaves the frame or was too close (area overflow).
    _no_det_streak: int = 0
    _NO_DET_RESET_STEPS = 100   # ~2 s at 50 Hz → force fresh histogram init

    fell_at: float | None = None       # sim_time when robot fell
    robot_trail: collections.deque = collections.deque(maxlen=TRAIL_LEN)
    ball_trail:  collections.deque = collections.deque(maxlen=TRAIL_LEN)

    if GUI_CHASE_ONLY:
        disp_w = disp_h = 720         # square 4-panel dashboard
    elif GUI_HEAD_ONLY:
        disp_w, disp_h = cam_w, cam_h
    else:
        disp_w = cam_w * 2 + 4
        disp_h = cam_h * 2 + 4

    # Optional video recording.  Set RECORD_VIDEO=/path/to/file.mp4 to enable.
    # Output uses the FINAL composed frame (head OR mosaic) so what you see
    # in the GUI is exactly what gets saved.
    video_writer = None
    record_path = os.environ.get("RECORD_VIDEO", "")
    record_duration = float(os.environ.get("RECORD_DURATION_S", "0"))   # 0 = no auto-stop
    record_start_wall = None
    if record_path and not HEADLESS:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = float(os.environ.get("RECORD_FPS", "30.0"))
        video_writer = cv2.VideoWriter(record_path, fourcc, fps_out, (disp_w, disp_h))
        if not video_writer.isOpened():
            print(f"[REC] FAILED to open VideoWriter for {record_path}", flush=True)
            video_writer = None
        else:
            print(f"[REC] Writing video to {record_path} @ {fps_out} fps "
                  f"({disp_w}x{disp_h}) for {record_duration} s wall.", flush=True)
            record_start_wall = time.perf_counter()

    if not HEADLESS:
        cv2.namedWindow("Go2 Object Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Go2 Object Tracking", disp_w, disp_h)
        print("Simulation running — window: 'Go2 Object Tracking'")
        if GUI_HEAD_ONLY:
            print("  FAST GUI mode: head camera only (normal-speed preferred).")
            print("  Set GO2_GUI_VIEW=full for 2x2 mosaic view.")
        else:
            print("  LEFT = top-down view   RIGHT = robot head camera")
            print("  Ball demo: large square path (no teleport), smooth edge motion.")
            print("  Ball waits 5s at each corner.")
            print("  Ball keeps moving even when out of sight (forces active search).")
            print("  Phase label on overview + head HUD.")
        print("  [Q/ESC] quit   [R] reset   [P] pause")
    else:
        print(
            f"HEADLESS mode — ~{HEADLESS_MAX_T:.0f}s nominal ({HEADLESS_MAX_STEPS} ctrl steps), "
            "no GUI, then exit."
        )
        print(
            "Target demo: mode="
            f"{os.environ.get('BALL_PATH_MODE', 'square')} "
            "(3s loss grace, pause until re-acquired)."
        )
    if MAX_RUNTIME_STEPS > 0:
        print(
            f"Runtime cap active: ~{MAX_RUNTIME_T:.0f}s ({MAX_RUNTIME_STEPS} ctrl steps), then exit."
        )

    t_prev = time.perf_counter()
    rx = ry = rz = 0.0

    while running:
        if not paused:
            # ── physics + gait at 500 Hz ──────────────────────────────────
            # Tilt-aware command guard: damp both vx and vyaw as tilt grows.
            # This keeps recovery authority while preventing the (forward + turn)
            # combinations that trigger roll-over in long runs.
            quat_pre = np.asarray(data.qpos[3:7], dtype=float)
            rz_pre   = float(data.qpos[2])
            incapped = _pose_incapped(rz_pre, quat_pre)
            tilt_scale = _tilt_guard_scale(quat_pre)
            # Tilt guard behaviour:
            #   - If the robot is truly incapacitated (rz collapsed / extreme pitch
            #     or roll), zero vx so we stop driving into a toppled pose.
            #   - Otherwise DO NOT attenuate — trace analysis shows the previous
            #     (tilt → cut vx) behaviour put the policy in a low-vx regime where
            #     it couldn't self-stabilise, and roll grew without bound until a
            #     flip.  The RL trot gait was trained with moderate tilt as part
            #     of its disturbance curriculum, so letting it run at full command
            #     while tilted actually helps it catch the fall.
            # ── Obstacle-aware brake ───────────────────────────────────
            # Scan obstacles in a forward cone; brake vx_cmd proportional
            # to the gap to the nearest threatening obstacle.  Gives the
            # controller time to plan a detour at high action-scale gaits.
            if os.environ.get("OBSTACLE_BRAKE", "1") == "1":
                LOOKAHEAD = float(os.environ.get("OBS_BRAKE_LOOKAHEAD", "3.0"))
                LATERAL   = float(os.environ.get("OBS_BRAKE_LATERAL",   "0.55"))
                _nearest_ahead = LOOKAHEAD
                for _tag, _ox, _oy in all_robot_obstacles:
                    _dx = _ox - rx
                    if 0.0 < _dx < LOOKAHEAD and abs(_oy - ry) < LATERAL:
                        if _dx < _nearest_ahead:
                            _nearest_ahead = _dx
                # Brake factor: 1.0 (no obstacle) → 0.6 (right in front)
                _brake = 0.6 + 0.4 * (_nearest_ahead / LOOKAHEAD)
                if vx_cmd > 0.0:
                    vx_cmd = max(0.55, vx_cmd * _brake)

            # ── Wall-proximity brake ────────────────────────────────────
            # Widened margin so the robot slows earlier and has more time
            # to steer away from the wall before reaching it.
            if os.environ.get("WALL_BRAKE", "1") == "1":
                WALL_Y = 2.58
                WALL_MARGIN = 1.0
                _wall_gap = WALL_Y - abs(ry)
                if _wall_gap < WALL_MARGIN and vx_cmd > 0.0:
                    _wb = max(0.25, _wall_gap / WALL_MARGIN)
                    vx_cmd = max(0.55, vx_cmd * _wb)

            # ── Stuck-detection + spin-out ─────────────────────────────
            # If forward progress stalls for ~3 sim-seconds while the
            # controller wants to go forward, override yaw to spin in
            # place for 2 s — usually pops the gait out of a wedge.
            _now_t  = float(data.time)
            _now_rx = float(data.qpos[0])
            if "stuck_hist" not in locals():
                stuck_hist = collections.deque(maxlen=160)   # ~3.2 s at 50 Hz
                stuck_spin_until = -1.0
                stuck_spin_dir = 1.0
            stuck_hist.append((_now_t, _now_rx))
            if _now_t < stuck_spin_until:
                vyaw_cmd = stuck_spin_dir * 0.8
                vx_cmd   = 0.0
            elif len(stuck_hist) == stuck_hist.maxlen and vx_cmd > 0.2:
                _t0, _x0 = stuck_hist[0]
                _dt = _now_t - _t0
                if _dt > 2.8 and (_now_rx - _x0) < 0.30:
                    # Stuck: spin out for 2 s, alternating direction each event.
                    stuck_spin_until = _now_t + 2.0
                    stuck_spin_dir = -stuck_spin_dir
                    stuck_hist.clear()
                    print(f"[STUCK] spin-out at t={_now_t:.1f}s rx={_now_rx:.2f} "
                          f"(progress={_now_rx-_x0:.2f} m in {_dt:.1f}s)", flush=True)

            vx_g     = 0.0 if incapped else (vx_cmd * ROBOT_SPEED_SCALE)
            vyaw_g   = 0.0 if incapped else (vyaw_cmd * tilt_scale)
            _ = tilt_scale  # keep referenced
            # Adaptive speed: tie joint-target amplitude to vx_cmd so the
            # gait throttles down when the controller chooses to slow
            # (obstacle near, ball lost/close).
            if os.environ.get("ADAPTIVE_SPEED", "1") == "1":
                _speed_ratio = min(1.0, max(0.0, abs(vx_cmd) / 0.8))
                import low_level as _ll
                _ll.set_dynamic_action_mult(_speed_ratio)
            for _ in range(CTRL_SKIP):
                gait.substep(
                    model, data, vx_g, vyaw_g,
                    ramp_start[0], float(data.time), SIM_DT,
                )
                mujoco.mj_step(model, data)
            step_num += 1
            sim_time = float(data.time)

            # ── collision detection: any contact between robot and an obstacle ──
            # Initialise counters lazily so they persist across iterations.
            if "collision_count" not in locals():
                collision_count = 0
                collision_failed = False
                collision_stage_at_fail = 0
            for _ci in range(data.ncon):
                c = data.contact[_ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                _hit_obstacle = (g1 in obstacle_geom_ids) or (g2 in obstacle_geom_ids)
                _hit_wall     = (g1 in wall_geom_ids)     or (g2 in wall_geom_ids)
                if _hit_obstacle or _hit_wall:
                    collision_count += 1
                    if not collision_failed:
                        collision_failed = True
                        collision_stage_at_fail = getattr(mover, "_stage", 1)
                        _kind = "WALL" if _hit_wall else "OBSTACLE"
                        print(
                            f"[COLLISION-{_kind}] FAILED at stage {collision_stage_at_fail} — "
                            f"step={step_num} sim_t={sim_time:.2f}s rx={float(data.qpos[0]):.2f}",
                            flush=True,
                        )
                        if os.environ.get("COLLISION_STOPS_SIM", "1") == "1":
                            _collision_stop_at = step_num + 60
                    break
            # Apply the delayed stop (lets the red overlay render a few frames).
            if collision_failed and step_num >= locals().get("_collision_stop_at", 10**18):
                print("[COLLISION] terminating after fail tail", flush=True)
                running = False

            # ── robot state (before target step — freeze uses robot↔ball distance) ──
            rx, ry, rz = (float(data.qpos[0]),
                          float(data.qpos[1]),
                          float(data.qpos[2]))

            quat = data.qpos[3:7]
            yaw = math.atan2(
                2.0 * (quat[0]*quat[3] + quat[1]*quat[2]),
                1.0 - 2.0 * (quat[2]**2 + quat[3]**2),
            )

            # ── move target ──────────────────────────────────────────────
            # Target path follows the configured mode; on sustained loss it waits
            # in place until re-acquired (after a short grace period).
            mover.set_robot_position(rx, ry)
            mover.step(CTRL_DT, data.mocap_pos, ball_visible=ball_visible_prev)
            bx = float(data.mocap_pos[0, 0])
            by = float(data.mocap_pos[0, 1])

            robot_trail.append((rx, ry))
            ball_trail.append((bx, by))

            # ── fall detection (height + tilt) ────────────────────────────
            quat_now = np.asarray(data.qpos[3:7], dtype=float)
            pr, rr = quat_to_pitch_roll(quat_now.astype(float))
            fell = _pose_incapped(rz, quat_now)
            if fell:
                if fell_at is None:
                    fell_at = sim_time
                    dist_m = math.hypot(rx - bx, ry - by)
                    print(
                        f"[WARN] Robot FELL / incapped at t={sim_time:.2f}s | "
                        f"pos=({rx:+.2f},{ry:+.2f}) z={rz:.3f}m "
                        f"pitch={math.degrees(pr):+.1f}° roll={math.degrees(rr):+.1f}° | "
                        f"dist_to_ball={dist_m:.2f}m | "
                        f"last cmd: vx={vx_cmd:+.3f} vyaw={vyaw_cmd:+.3f} | "
                        f"auto-reset in {FALL_RESET_S:.0f}s",
                        flush=True,
                    )
                elif sim_time - fell_at >= FALL_RESET_S:
                    _reset(model, data, ramp_start, gait,
                           tracker, controller, robot_trail, ball_trail, mover)
                    fell_at = None
                    continue
            else:
                fell_at = None

            # ── standup ramp (for HUD only; actual ramp used inside physics loop) ──
            elapsed_since_start = sim_time - ramp_start[0]
            ramp = float(np.clip(elapsed_since_start / RAMP_DUR, 0.0, 1.0))

            # ── render (mosaic views when GUI; head always for vision) ───
            # RENDER_SKIP: only render+run vision every Nth ctrl step.  On
            # skipped steps, reuse last rendered images and last vision det.
            # Always render on step 1 so cached *_rgb / cx / conf exist before
            # any skip step uses them.
            do_render = (step_num == 1) or (step_num % RENDER_SKIP == 0)
            if do_render:
                # Helper: inject visualization geoms (safety zones,
                # passage gate, path arrow) into the scene before render.
                def _inject_chase_overlays(scene):
                    # Safety zones for obstacles in lookahead range
                    _viz_obs = []
                    for _mid in obstacle_mocap_ids:
                        _ox = float(data.mocap_pos[_mid, 0])
                        if _ox > 50.0:
                            continue
                        if rx - 0.3 < _ox < rx + 5.5:
                            _oy = float(data.mocap_pos[_mid, 1])
                            _yh = obstacle_y_half_by_mid.get(_mid, 0.30)
                            _viz_obs.append((_ox, _oy, _yh))
                    if _viz_obs:
                        # Halved safety-zone visualisation (was 0.45) so
                        # the yellow patch around each obstacle is more
                        # compact — matches a tighter pass distance.
                        _inject_safety_zone_geoms(scene, _viz_obs, 0.22)
                    # Green safety wall — drawn ONLY on the side of the
                    # obstacle the robot is currently passing.  Filter:
                    # (1) obstacle is just ahead (within OBS_X_HALF + 1m),
                    # (2) robot is laterally close enough that the obstacle
                    #     is the one being skirted (within obs_half + 1m).
                    # We keep only the single closest such side; other
                    # obstacles' margins are not relevant.
                    _SAFETY = 0.05   # very close to obstacle edge — robot's actual closest approach
                    _NEAR_AHEAD = 2.5
                    _best = None   # (lateral_clearance, ox, safe_y)
                    for (_ox, _oy, _oh) in obstacle_world_positions:
                        _dx = _ox - rx
                        if _dx < -0.3 or _dx > _NEAR_AHEAD:
                            continue
                        _lat = abs(_oy - ry) - _oh
                        # Robot must be physically beside this obstacle's
                        # safety zone, not far above/below it.
                        if _lat > 1.0:
                            continue
                        if ry > _oy:
                            _safe_y = _oy + _oh + _SAFETY
                        else:
                            _safe_y = _oy - _oh - _SAFETY
                        if _best is None or _lat < _best[0]:
                            _best = (_lat, _ox, _safe_y)
                    if _best is not None:
                        _, _bx_obs, _by_safe = _best
                        _inject_safety_wall_geom(scene, _bx_obs, _by_safe, obs_half_x=0.40)
                    # RED arrow on the floor — forecasted path the robot
                    # plans to follow to reach the ball.
                    if ("cached_plan_waypoints" in locals()
                            and cached_plan_waypoints):
                        _inject_path_geoms(scene, cached_plan_waypoints,
                                           rgba=(1.0, 0.0, 0.0, 0.95))

                # Update the stabilized chase camera's lookat (heavy
                # low-pass filter on the robot's xy + locked z) so the
                # view doesn't bob with the gait.
                if _trunk_id >= 0:
                    _robot_pos = data.xpos[_trunk_id]
                    if _chase_smooth_xy is None:
                        _chase_smooth_xy = np.array(
                            [_robot_pos[0], _robot_pos[1]], dtype=float)
                    else:
                        _alpha = 0.06   # heavy smoothing — view glides, no shake
                        _chase_smooth_xy = (
                            _alpha * np.array([_robot_pos[0], _robot_pos[1]])
                            + (1.0 - _alpha) * _chase_smooth_xy
                        )
                    chase_cam_free.lookat[0] = float(_chase_smooth_xy[0]) + 0.10
                    chase_cam_free.lookat[1] = float(_chase_smooth_xy[1])
                    chase_cam_free.lookat[2] = 0.30   # locked z

                if not HEADLESS and not GUI_HEAD_ONLY:
                    renderer.update_scene(data, camera=top_id)
                    top_rgb = renderer.render().copy()
                    renderer.update_scene(data, camera=chase_cam_free)
                    _inject_chase_overlays(renderer.scene)
                    chase_rgb = renderer.render().copy()
                    renderer.update_scene(data, camera=rear_id)
                    rear_rgb = renderer.render().copy()
                elif not HEADLESS and GUI_CHASE_ONLY:
                    # Dashboard mode: chase rendered at 480x480 with the
                    # dedicated square renderer; side / front / map views
                    # rendered at 240x240 with the tile renderer.
                    chase_renderer.update_scene(data, camera=chase_cam_free)
                    _inject_chase_overlays(chase_renderer.scene)
                    chase_rgb = chase_renderer.render().copy()

                    # Top-down map cam follows the robot.
                    if _trunk_id >= 0 and _chase_smooth_xy is not None:
                        map_cam_free.lookat[0] = float(_chase_smooth_xy[0])
                        map_cam_free.lookat[1] = float(_chase_smooth_xy[1])
                        map_cam_free.lookat[2] = 0.0

                    # Wall cams: pan along the wall (track robot's smoothed
                    # x), but lock y=0 (corridor centerline) and z=0.30
                    # so the view never bobs with the gait or with robot
                    # lateral wiggle.
                    if _chase_smooth_xy is not None:
                        _wx = float(_chase_smooth_xy[0])
                        left_wall_cam_free.lookat[0]  = _wx
                        left_wall_cam_free.lookat[1]  = 0.0
                        left_wall_cam_free.lookat[2]  = 0.30
                        right_wall_cam_free.lookat[0] = _wx
                        right_wall_cam_free.lookat[1] = 0.0
                        right_wall_cam_free.lookat[2] = 0.30

                    # 3-camera layout: chase + front + 2D map.  Wall cams
                    # are not rendered; bottom row is the info/telemetry panel.
                    left_rgb = None
                    right_rgb = None

                    tile_renderer.update_scene(data, camera=map_cam_free)
                    _inject_chase_overlays(tile_renderer.scene)
                    map_rgb = tile_renderer.render().copy()

                # Head camera: render CLEAN first — the ball tracker reads
                # this image and would lock onto the red path-arrow geoms
                # if they were burned in.  Tracker uses the clean copy.
                renderer.update_scene(data, camera=head_id)
                head_rgb = renderer.render().copy()

                # ── vision ────────────────────────────────────────────────
                det = tracker.update(head_rgb)
                cx, cy, area, conf = det.cx, det.cy, det.area, det.confidence
                ball_visible_prev = (cx is not None) and (conf >= 0.12)

                # FRONT VISION tile reuses the clean head_rgb (saves one
                # head-cam render per frame; ball circle is drawn in cv2 later).
                head_rgb_overlay = head_rgb
            # else: cx, cy, area, conf, ball_visible_prev, *_rgb retained
            #       from previous render — controller uses ZOH on vision.

            # ── tracker auto-reset on prolonged no-detection ──────────────
            if cx is None or conf < 0.12:
                _no_det_streak += 1
                if _no_det_streak == _NO_DET_RESET_STEPS:
                    tracker.reset()
                    print(f"[TRACKER] Auto-reset after {_no_det_streak} steps without detection "
                          f"(t={sim_time:.1f}s)", flush=True)
                    _no_det_streak = 0
            else:
                _no_det_streak = 0

            # ── ball-search behaviour when blind ─────────────────────────
            # When ball confidence stays low for >SEARCH_AFTER_STEPS, the
            # controller produces stale or wrong commands.  Override yaw
            # with a slow scan and zero forward speed so the robot doesn't
            # wedge into walls / obstacles.
            _SEARCH_AFTER = int(os.environ.get("SEARCH_AFTER_STEPS", "30"))
            _searching = (_no_det_streak >= _SEARCH_AFTER)
            if _searching:
                # Direction alternates every ~1.5 s so we sweep both sides.
                _search_dir = 1.0 if (_no_det_streak // 75) % 2 == 0 else -1.0
                vyaw_cmd = 0.45 * _search_dir   # slow yaw scan
                vx_cmd   = 0.0                  # don't blunder forward blind

            # ── control ───────────────────────────────────────────────────
            dist_world = math.hypot(rx - bx, ry - by)
            # Compute ball's WORLD angular velocity (for FF).  For fast-orbiting
            # balls we also LEAD the target: aim at where the ball WILL be,
            # not where it is, because our robot is ~4× slower than the ball
            # tangentially and can never catch the current position.
            world_angle = math.atan2(by - ry, bx - rx)
            if "prev_world_angle" not in locals() or prev_world_angle is None:
                prev_world_angle = world_angle
                world_angular_rate = 0.0
                prev_bx = bx
                prev_by = by
                ball_vx_est = 0.0
                ball_vy_est = 0.0
            else:
                d_angle = world_angle - prev_world_angle
                while d_angle > math.pi:  d_angle -= 2 * math.pi
                while d_angle < -math.pi: d_angle += 2 * math.pi
                rate_raw = d_angle / max(CTRL_DT, 1e-3)
                world_angular_rate = 0.90 * world_angular_rate + 0.10 * rate_raw
                prev_world_angle = world_angle
                # Ball velocity (world frame) via finite difference + LPF.
                bvx_raw = (bx - prev_bx) / max(CTRL_DT, 1e-3)
                bvy_raw = (by - prev_by) / max(CTRL_DT, 1e-3)
                ball_vx_est = 0.90 * ball_vx_est + 0.10 * bvx_raw
                ball_vy_est = 0.90 * ball_vy_est + 0.10 * bvy_raw
                prev_bx = bx
                prev_by = by

            # ── Intercept strategy (mode-aware, persistent) ─────────────
            # Robot's 0.148 m/s max speed is way too slow to chase a 0.18 m/s
            # ball across a 3 m arena.  Strategy per motion type:
            #   ORBITAL  (stable radius): ambush INSIDE orbit at radius
            #     (r_ball - 1.0) so ball's nearest pass lands at ~1.0 m
            #     dist from robot — in the near band (0.7–1.3 m).
            #   LINEAR   (moving along edges): iterate linear lead-intercept
            #     to predict where ball will be when robot arrives.
            # Mode is decided AFTER 2 s of data (persistent — doesn't flip
            # every frame as ω varies within a trajectory).
            V_ROBOT_MAX = 0.148
            ball_r_world = math.hypot(bx, by)
            if "mode_decision_t" not in locals():
                mode_decision_t = 6.0
                traj_mode = "pending"
                r_history = []
                bx_history = []
                by_history = []
            if traj_mode == "pending":
                r_history.append(ball_r_world)
                bx_history.append(bx)
                by_history.append(by)
                if sim_time >= mode_decision_t and len(r_history) > 100:
                    r_min = min(r_history)
                    r_max = max(r_history)
                    r_range = r_max - r_min
                    r_mean = sum(r_history) / len(r_history)
                    bx_mean = sum(bx_history) / len(bx_history)
                    by_mean = sum(by_history) / len(by_history)
                    # Classify:
                    # - Orbital around ORIGIN: r_range small, r_mean large
                    #   (circle mode)
                    # - Orbital around (cx, cy) NOT origin: r from origin varies
                    #   a lot, but r from (bx_mean, by_mean) is small and
                    #   stable (figure-8 mode)
                    # - Linear: ball moves along a line/edge (square)
                    r_centered = [math.hypot(x - bx_mean, y - by_mean)
                                   for x, y in zip(bx_history, by_history)]
                    rc_min = min(r_centered)
                    rc_max = max(r_centered)
                    rc_range = rc_max - rc_min
                    rc_mean = sum(r_centered) / len(r_centered)

                    if r_range < 0.2 and r_mean > 1.0:
                        traj_mode = "orbital"
                        orbit_r = r_mean
                        orbit_cx = 0.0
                        orbit_cy = 0.0
                    elif rc_range < 1.5 and rc_mean > 0.5:
                        # Ball moves around a NON-ORIGIN center (figure-8)
                        traj_mode = "orbital"
                        orbit_r = rc_mean
                        orbit_cx = bx_mean
                        orbit_cy = by_mean
                    else:
                        traj_mode = "linear"
                        orbit_cx = 0.0
                        orbit_cy = 0.0

            # CORRIDOR mode (env-configured): chase ball, but detour around
            # any obstacle that lies between robot and ball.
            ball_mode_env = os.environ.get("BALL_PATH_MODE", "").lower()

            if ball_mode_env == "corridor":
                target_x = bx
                # Track ball's CURRENT y position (best empirical result —
                # pure-target-lane and hybrids both lose tracking precision).
                target_y = by

                # ── Obstacle-aware steering ─────────────────────────────────
                # Only detour around an obstacle if it ACTUALLY lies between
                # robot and ball.  If ball is closer than the obstacle, just
                # chase the ball — no need to steer pre-emptively.
                OBS_X_HALF = 0.20
                OBS_Y_HALF = 0.30   # match scene.xml — obstacles are 0.6 m wide on y
                # Tightened for shortest-path mode: robot can graze obstacles
                # at ~25 cm clearance instead of 45 cm. Configurable via env.
                ROBOT_SAFETY  = float(os.environ.get("ROBOT_SAFETY",  "0.25"))
                DETOUR_MARGIN = float(os.environ.get("DETOUR_MARGIN", "0.15"))
                LOOKAHEAD_X = float(os.environ.get("DETOUR_LOOKAHEAD", "3.5"))
                corridor_half_w = 2.5   # 5 m wide corridor

                # Stuck-detection: if x-position hasn't advanced ≥0.05 m over
                # the last ~1 s of sim AND the robot isn't already mid-turn,
                # treat it as wedged.  The "not mid-turn" guard prevents a
                # false fire when the controller is legitimately turning
                # hard around an obstacle: a hard turn naturally slows
                # forward progress to ~0.08 m/s, just above the threshold,
                # but that's correct behaviour, not a wedge.  A real wedge
                # has <0.02 m of progress AND a saturated yaw with no actual
                # rotation (robot grinding in place).
                if "stuck_xs" not in locals():
                    from collections import deque as _deque
                    stuck_xs   = _deque(maxlen=50)    # 1 s window (Tier 1)
                    stuck_ys   = _deque(maxlen=50)
                    stuck_x3s  = _deque(maxlen=150)   # 3 s window (Tier 2)
                    stuck_y3s  = _deque(maxlen=150)
                    stuck_flip = False
                    stuck_until = 0
                    stuck_recovery_until = 0
                    last_burst_step = -1000
                stuck_xs.append(rx); stuck_ys.append(ry)
                stuck_x3s.append(rx); stuck_y3s.append(ry)
                # Track when the controller was commanding burst yaw
                # (>0.50 rad/s) — the gait is sluggish for ~1 s after a
                # burst, which can cause false stuck detection during the
                # post-turn momentum recovery.
                if abs(vyaw_cmd) > 0.50:
                    last_burst_step = step_num
                post_burst = (step_num - last_burst_step) < 50  # 1 s cooldown

                # Two-tier stuck detection.
                # Tier 1 (1 s, fast): VERY tight x AND y range AND not
                #   turning hard AND not in post-burst cooldown.  Tightened
                #   from 0.05→0.03 so brief slowdowns after sharp turns
                #   don't fire false positives.
                # Tier 2 (3 s, slow): 2-D motion < 0.15 m REGARDLESS of
                #   yaw — catches sustained pin-while-spinning.
                turning_hard = abs(vyaw_cmd) >= 0.11
                tier1 = (
                    len(stuck_xs) == stuck_xs.maxlen and
                    (max(stuck_xs) - min(stuck_xs)) < 0.03 and
                    (max(stuck_ys) - min(stuck_ys)) < 0.03 and
                    not turning_hard and
                    not post_burst
                )
                tier2 = False
                if len(stuck_x3s) == stuck_x3s.maxlen:
                    rng_x = max(stuck_x3s) - min(stuck_x3s)
                    rng_y = max(stuck_y3s) - min(stuck_y3s)
                    rng_2d = (rng_x * rng_x + rng_y * rng_y) ** 0.5
                    if rng_2d < 0.15:
                        tier2 = True
                if (tier1 or tier2) and step_num > stuck_until + 150:
                    stuck_flip = not stuck_flip
                    stuck_until = step_num + 100   # commit detour side ~2 s
                    stuck_recovery_until = step_num + 75   # 1.5 s reverse
                    tag = "T1" if tier1 else "T2"
                    print(f"[STUCK-{tag}] t={sim_time:.1f}s pos=({rx:+.2f},{ry:+.2f}) "
                          f"flip→{'+y' if stuck_flip else '-y'}",
                          flush=True)
                    stuck_x3s.clear(); stuck_y3s.clear()

                # Collect obstacles MEANINGFULLY ahead of the robot — i.e.
                # at least 0.30 m forward.  Obstacles beside or just behind
                # the robot must be excluded so the avoidance doesn't keep
                # detouring after the robot has already passed them.
                ahead_obstacles = []
                for mid in obstacle_mocap_ids:
                    ox = float(data.mocap_pos[mid, 0])
                    oy = float(data.mocap_pos[mid, 1])
                    if ox > 50.0 and (oy > 50.0 or oy < -50.0):
                        continue   # parked off-map
                    if (rx - OBS_X_HALF < ox < rx + LOOKAHEAD_X
                            and ox < bx + 0.5):
                        oy_half = obstacle_y_half_by_mid.get(mid, 0.30)
                        ahead_obstacles.append((ox, oy, oy_half))
                ahead_obstacles.sort()

                # ── ROW-AWARE band-finding avoidance ─────────────────────
                # Group ahead obstacles into ROWS (same x within 1.5 m of
                # each other), then for the nearest row find the largest
                # FREE y-band (after subtracting all obstacles' blocked
                # zones) and aim for the band closest to the ball.  This
                # correctly handles 2-obstacle rows where the robot must
                # thread BETWEEN obstacles (or pass on the side opposite
                # both of them).
                rows = []
                for (ox, oy, oy_half) in ahead_obstacles:
                    if ox < rx:
                        continue
                    placed = False
                    for row in rows:
                        if abs(ox - row[0][0]) < 1.5:
                            row.append((ox, oy, oy_half))
                            placed = True
                            break
                    if not placed:
                        rows.append([(ox, oy, oy_half)])
                rows.sort(key=lambda r: min(o[0] for o in r))

                # Persistent commitment: once we pick a free band for a
                # row, keep aiming at the same band-y until the robot is
                # past that row's back edge.
                if "detour_obs_x" not in locals():
                    detour_obs_x = -1e9
                    detour_side_y = 0.0
                # Clear commitment when robot has cleared the row.
                if rx > detour_obs_x + OBS_X_HALF + 0.30:
                    detour_obs_x = -1e9
                # ALSO release commitment if the ball has moved so that
                # the straight-line robot→ball path no longer crosses
                # any obstacle in the committed row.  Without this, the
                # robot keeps detouring through the chosen passage even
                # after the ball moved to a clear lane.
                if detour_obs_x > 0 and abs(bx - rx) > 0.01:
                    _path_blocked = False
                    _committed_row_obs = []
                    for (_ox, _oy, _oh) in ahead_obstacles:
                        if abs(_ox - detour_obs_x) > 1.5:
                            continue   # different row
                        if _ox < rx:
                            continue
                        _committed_row_obs.append((_ox, _oy, _oh))
                        _t = (_ox - rx) / (bx - rx)
                        if 0.0 < _t < 1.0:
                            _y_at = ry + _t * (by - ry)
                            # 0.15 m hysteresis so it doesn't flicker
                            if abs(_y_at - _oy) < (_oh + ROBOT_SAFETY + 0.15):
                                _path_blocked = True
                                break
                    if not _path_blocked:
                        detour_obs_x = -1e9
                    elif _committed_row_obs:
                        # Path still blocked → re-check whether the band on
                        # the OTHER SIDE of the obstacle is now closer to a
                        # straight line robot→ball.  If yes, release so the
                        # planner picks the new optimal band on the next step.
                        _blocked2 = []
                        for (_ox, _oy, _oh) in _committed_row_obs:
                            _blocked2.append((_oy - _oh - ROBOT_SAFETY,
                                              _oy + _oh + ROBOT_SAFETY))
                        _blocked2.sort()
                        _wall_lim2 = corridor_half_w - 0.30
                        _free2 = []
                        _prev2 = -_wall_lim2
                        for _lo, _hi in _blocked2:
                            if _lo > _prev2:
                                _free2.append((_prev2, _lo))
                            _prev2 = max(_prev2, _hi)
                        if _wall_lim2 > _prev2:
                            _free2.append((_prev2, _wall_lim2))
                        if _free2:
                            _row_x = _committed_row_obs[0][0]
                            def _tot_len_full(b):
                                by_in = max(b[0], min(b[1], by))
                                ry_in = max(b[0], min(b[1], ry))
                                yc = 0.5 * (by_in + ry_in)
                                d1 = math.hypot(_row_x - rx, yc - ry)
                                d2 = math.hypot(bx - _row_x, by - yc)
                                width = b[1] - b[0]
                                return d1 + d2 - 0.05 * width   # matches main planner
                            best_now = min(_free2, key=_tot_len_full)
                            best_now_y = 0.5 * (
                                max(best_now[0], min(best_now[1], by))
                                + max(best_now[0], min(best_now[1], ry)))
                            new_cost = _tot_len_full(best_now)
                            # Current commitment's "virtual band" — at least
                            # 0.4 m wide centred on detour_side_y.
                            cur_band = (detour_side_y - 0.2, detour_side_y + 0.2)
                            cur_cost = _tot_len_full(cur_band)
                            # Hysteresis: only switch if the new band saves
                            # at least 0.8 m of total path AND the y differs
                            # by ≥ 0.6 m.  Prevents per-step flapping.
                            if "detour_repick_cooldown" not in locals():
                                detour_repick_cooldown = 0
                            if detour_repick_cooldown > 0:
                                detour_repick_cooldown -= 1
                            elif (cur_cost - new_cost > 1.5    # need bigger savings to flip
                                    and abs(best_now_y - detour_side_y) > 0.8):
                                print(f"[DETOUR] re-pick: side {detour_side_y:.2f} → {best_now_y:.2f} "
                                      f"(saves {cur_cost-new_cost:.2f} m)", flush=True)
                                detour_obs_x = -1e9
                                detour_repick_cooldown = 120   # 2.4 s @ 50 Hz cooldown

                # If no active commitment, evaluate the nearest row.
                if detour_obs_x < 0 and rows:
                    nearest_row = rows[0]
                    nearest_ox = max(o[0] for o in nearest_row)

                    # Test if straight-line to ball would actually hit any
                    # obstacle in this row (path-cross check).  Otherwise
                    # the ball-path is already clear of the row → no detour.
                    must_detour = False
                    for (ox, oy, oy_half) in nearest_row:
                        if abs(target_x - rx) > 0.01:
                            t_at_ox = (ox - rx) / (target_x - rx)
                            if 0.0 < t_at_ox < 1.0:
                                y_at_ox = ry + t_at_ox * (target_y - ry)
                                if abs(y_at_ox - oy) < (oy_half + ROBOT_SAFETY):
                                    must_detour = True
                                    break
                        # Wedge fallback: robot in obstacle's y-band AND close
                        if (ox - rx) < 1.8 and abs(ry - oy) < (oy_half + 0.45):
                            must_detour = True
                            break

                    if must_detour:
                        # Compute blocked zones from ALL obstacles in this row
                        blocked = []
                        for (ox, oy, oy_half) in nearest_row:
                            blocked.append((oy - oy_half - ROBOT_SAFETY,
                                            oy + oy_half + ROBOT_SAFETY))
                        blocked.sort()
                        # Find free bands within corridor (with body half
                        # margin from walls).
                        wall_lim = corridor_half_w - 0.30
                        free_bands = []
                        prev = -wall_lim
                        for lo, hi in blocked:
                            if lo > prev:
                                free_bands.append((prev, lo))
                            prev = max(prev, hi)
                        if wall_lim > prev:
                            free_bands.append((prev, wall_lim))
                        if free_bands:
                            # Pick band whose closest y to ball is smallest.
                            # During stuck-recovery, force opposite side.
                            if step_num < stuck_until:
                                # Pick band on stuck_flip side
                                target_side = +1 if stuck_flip else -1
                                bands_on_side = [
                                    b for b in free_bands
                                    if (b[0] + b[1]) / 2 * target_side > 0
                                ]
                                if bands_on_side:
                                    best = max(bands_on_side,
                                               key=lambda b: b[1] - b[0])
                                else:
                                    best = max(free_bands,
                                               key=lambda b: b[1] - b[0])
                            else:
                                # Pick band that minimises the TOTAL detour
                                # length: (robot→passage point) + (passage→ball).
                                # This is the proper shortest-path heuristic
                                # for a single row of obstacles; the prior
                                # min-(passage-to-ball) heuristic often picked
                                # a band on the far side, forcing a big swing.
                                def _total_len(b):
                                    by_in = max(b[0], min(b[1], by))      # clamp ball y into band
                                    ry_in = max(b[0], min(b[1], ry))      # clamp robot y into band
                                    # Choose the band point closest to a line
                                    # connecting robot and ball — approximated
                                    # by averaging the two clamped y's.
                                    y_choice = 0.5 * (by_in + ry_in)
                                    d1 = math.hypot(nearest_ox - rx, y_choice - ry)
                                    d2 = math.hypot(bx - nearest_ox, by - y_choice)
                                    # Penalize narrow bands so we don't pick
                                    # one that barely fits.
                                    width = (b[1] - b[0])
                                    return d1 + d2 - 0.05 * width
                                best = min(free_bands, key=_total_len)
                            # Aim through the band at the point closest to
                                # the straight robot→ball line (shortest path).
                                _by_in = max(best[0], min(best[1], by))
                                _ry_in = max(best[0], min(best[1], ry))
                                target_band_y = 0.5 * (_by_in + _ry_in)
                            # Commit — store the band range so the aim-point
                            # can be RE-CENTERED each step on the current
                            # robot↔ball line (so a moving ball doesn't keep
                            # the robot heading at a stale aim-point).
                            detour_obs_x   = nearest_ox
                            detour_band_lo = best[0]
                            detour_band_hi = best[1]
                            detour_side_y  = target_band_y   # initial; updated each step below
                            _bands_str = ",".join(f"[{b[0]:+.2f},{b[1]:+.2f}]" for b in free_bands)
                            _costs = [(b, _total_len(b)) for b in free_bands]
                            _costs_str = ",".join(f"{c:.2f}" for _,c in _costs)
                            print(f"[PLAN] row x={nearest_ox:.1f} robot=({rx:.2f},{ry:.2f}) "
                                  f"ball=({bx:.2f},{by:.2f}) bands={_bands_str} costs={_costs_str} "
                                  f"→ band={best[0]:+.2f}..{best[1]:+.2f}  init target_y={target_band_y:.2f}",
                                  flush=True)

                # Apply commitment with phase-based target_x.
                # Re-center the aim-point within the committed band each step
                # so a moving ball doesn't leave the robot heading at a stale
                # target — picks the point closest to the current straight
                # robot↔ball line that's still inside the band.
                if detour_obs_x > 0 and "detour_band_lo" in locals():
                    _by_in = max(detour_band_lo, min(detour_band_hi, by))
                    _ry_in = max(detour_band_lo, min(detour_band_hi, ry))
                    detour_side_y = 0.5 * (_by_in + _ry_in)
                # Reset each step so blue doesn't show stale plan when
                # the planner is no longer in a detour.
                shared_path_pts = []
                if detour_obs_x > 0:
                    # 3-zone smooth path with pure-pursuit lookahead.  Robot's
                    # target is a point a FIXED DISTANCE ahead along the path
                    # (not a fraction of the segment) — this yields a constant
                    # lookahead and a smooth, stable heading command.
                    obs_back_x = detour_obs_x + OBS_X_HALF
                    passage = (detour_obs_x, detour_side_y)
                    ball_p  = (bx, by)
                    start_p = (rx, ry)
                    STRAIGHT_HALF = 0.9
                    in_x  = detour_obs_x - STRAIGHT_HALF
                    out_x = detour_obs_x + STRAIGHT_HALF
                    LOOKAHEAD_M = 1.8   # meters ahead (longer = smoother bearing)

                    # Densely sample the full planned path (in→passage_in→
                    # straight→passage_out→ball) into a list of points.
                    path_pts = [(rx, ry)]
                    # Pre-passage Bezier
                    Pa = (rx, ry); Pb = (in_x, detour_side_y)
                    Pc = (0.5 * (rx + in_x), detour_side_y)
                    if Pb[0] > Pa[0]:
                        for _i in range(1, 15):
                            _t = _i / 15.0; _u = 1.0 - _t
                            path_pts.append((
                                _u*_u*Pa[0] + 2*_u*_t*Pc[0] + _t*_t*Pb[0],
                                _u*_u*Pa[1] + 2*_u*_t*Pc[1] + _t*_t*Pb[1],
                            ))
                    # Straight zone
                    path_pts.append((in_x, detour_side_y))
                    path_pts.append((out_x, detour_side_y))
                    # Post-passage Bezier
                    Pa = (out_x, detour_side_y); Pb = (bx, by)
                    Pc = (out_x + 0.5, detour_side_y)
                    if Pb[0] > Pa[0]:
                        for _i in range(1, 15):
                            _t = _i / 15.0; _u = 1.0 - _t
                            path_pts.append((
                                _u*_u*Pa[0] + 2*_u*_t*Pc[0] + _t*_t*Pb[0],
                                _u*_u*Pa[1] + 2*_u*_t*Pc[1] + _t*_t*Pb[1],
                            ))

                    # Strip points strictly behind the robot in x so the
                    # path == the blue-arrow path == what we want robot to
                    # follow.  After filter, find closest and walk forward.
                    _fwd_pts = [(rx, ry)]
                    _lx = rx
                    for _px, _py in path_pts:
                        if _px > _lx + 0.05:
                            _fwd_pts.append((_px, _py))
                            _lx = _px
                    path_pts = _fwd_pts
                    # Stash the same path the controller follows so the
                    # blue arrow shows EXACTLY what robot is tracking.
                    shared_path_pts = list(path_pts)
                    # Find the point on path closest to robot.
                    _closest_i = 0
                    _closest_d = 1e9
                    for _i, (_px, _py) in enumerate(path_pts):
                        _d = math.hypot(_px - rx, _py - ry)
                        if _d < _closest_d:
                            _closest_d = _d
                            _closest_i = _i
                    # Walk forward arc-distance LOOKAHEAD_M.
                    _acc = 0.0
                    target_x, target_y = path_pts[-1]
                    for _i in range(_closest_i + 1, len(path_pts)):
                        _px0, _py0 = path_pts[_i - 1]
                        _px1, _py1 = path_pts[_i]
                        _seg = math.hypot(_px1 - _px0, _py1 - _py0)
                        if _acc + _seg >= LOOKAHEAD_M:
                            _f = (LOOKAHEAD_M - _acc) / max(_seg, 1e-6)
                            target_x = _px0 + _f * (_px1 - _px0)
                            target_y = _py0 + _f * (_py1 - _py0)
                            break
                        _acc += _seg
                    # Safety clamp in straight zone
                    if "detour_band_lo" in locals():
                        if in_x - 0.2 < rx < out_x + 0.2:
                            target_y = max(detour_band_lo, min(detour_band_hi, target_y))
            elif traj_mode == "orbital":
                # Orbital ambush: sit INSIDE ball's orbit at radius
                # (orbit_r - 1.0) measured from the orbit center.  Lead angle
                # = ω × (orbit_r_offset / v_r) — walk time from origin to
                # ambush_r is fixed, so lead is constant.
                ball_rel_x = bx - orbit_cx
                ball_rel_y = by - orbit_cy
                ball_rel_angle = math.atan2(ball_rel_y, ball_rel_x)
                ambush_r = max(0.5, orbit_r - 1.0)
                approx_walk_dist = math.hypot(orbit_cx + ambush_r, orbit_cy)
                T_lead = approx_walk_dist / V_ROBOT_MAX
                future_angle = ball_rel_angle + world_angular_rate * T_lead
                target_x = orbit_cx + ambush_r * math.cos(future_angle)
                target_y = orbit_cy + ambush_r * math.sin(future_angle)
            elif traj_mode == "linear":
                # Perp-foot intercept: closest point on ball's velocity line
                # to origin.  For a square edge, this is the midpoint.
                # Robot walks there; ball passes close by on its way through.
                # Cap target distance from origin at 2.0 m so robot doesn't
                # walk all the way to the far edge (where late-run falls happen).
                bvx, bvy = ball_vx_est, ball_vy_est
                bspeed2 = bvx * bvx + bvy * bvy
                if bspeed2 > 0.005:
                    t_perp = -(bx * bvx + by * bvy) / bspeed2
                    if t_perp > 0:
                        target_x = bx + bvx * t_perp
                        target_y = by + bvy * t_perp
                    else:
                        target_x, target_y = bx, by
                else:
                    target_x, target_y = bx, by
                # Dynamic clamp on target radius — based on ball's observed
                # min distance from origin (so we aim inside ball's path).
                # Prevents robot walking to far corners (square late-run fall).
                # Offset 1.0 m inside ball's nearest point so dist to ball
                # when ball passes lands in the near band (0.7-1.3 m).
                if len(r_history) >= 50:
                    ball_r_min = min(r_history)
                    clamp_r = max(1.0, ball_r_min - 0.80)
                else:
                    clamp_r = 2.0
                target_r = math.hypot(target_x, target_y)
                if target_r > clamp_r:
                    scale = clamp_r / target_r
                    target_x *= scale
                    target_y *= scale
            else:
                # During warm-up: walk straight toward ball (no prediction).
                target_x, target_y = bx, by
            dist_to_target = math.hypot(target_x - rx, target_y - ry)
            # Smooth target jumps caused by detour transitions / re-picks
            # so big y changes don't translate into sudden yaw kicks.
            if "smooth_target_y" not in locals():
                smooth_target_y = float(target_y)
                smooth_target_x = float(target_x)
            _ty_alpha = 0.30    # ~3-step EMA — fast enough to follow path updates
            smooth_target_y = (1.0 - _ty_alpha) * smooth_target_y + _ty_alpha * float(target_y)
            smooth_target_x = (1.0 - _ty_alpha) * smooth_target_x + _ty_alpha * float(target_x)
            world_bearing_rad = math.atan2(smooth_target_y - ry, smooth_target_x - rx) - yaw
            # Anticipatory yaw FF — rate of change of bearing → controller can
            # feed-forward the turn instead of reacting after the fact.
            if "prev_world_bearing" not in locals():
                prev_world_bearing = world_bearing_rad
                world_bearing_rate_ff = 0.0
            _db = world_bearing_rad - prev_world_bearing
            # unwrap
            while _db > math.pi: _db -= 2 * math.pi
            while _db < -math.pi: _db += 2 * math.pi
            _instant_rate = _db / max(CTRL_DT, 1e-3)
            world_bearing_rate_ff = 0.7 * world_bearing_rate_ff + 0.3 * _instant_rate
            prev_world_bearing = world_bearing_rad
            # Mild slowdown when bearing error is large — too aggressive a
            # cap chokes the gait and causes collisions.
            _be = abs(world_bearing_rad)
            if _be > 0.35:
                vx_cmd = min(vx_cmd, 0.65)
            elif _be > 0.22:
                vx_cmd = min(vx_cmd, 0.74)
            while world_bearing_rad > math.pi:  world_bearing_rad -= 2 * math.pi
            while world_bearing_rad < -math.pi: world_bearing_rad += 2 * math.pi

            # Report distance to target.  For orbital mode we want the robot
            # to travel all the way to the ambush point (controller's
            # TARGET_DIST_GOAL=1.0 would stop it short, so offset by +1.0).
            # For linear we stop at target.  For corridor we ALWAYS report
            # dist ≥ 1.5 so vx never goes to zero — robot must keep chasing.
            if ball_mode_env == "corridor":
                reported_dist = max(1.5, dist_to_target)
            elif traj_mode == "orbital":
                reported_dist = dist_to_target + 1.0
            else:
                reported_dist = dist_to_target      # stop at target
            vx_raw, vyaw_raw = controller.compute(
                cx, cy, area, conf, cam_w, cam_h, CTRL_DT,
                world_dist_m=reported_dist,
                world_bearing_rate=world_bearing_rate_ff,
                world_bearing_rad=world_bearing_rad,
            )
            # EMA smoothing: prevents rapid ± direction flips from destabilising
            # the RL gait.  First step: seed smoothed value directly.
            if step_num == 1:
                vx_cmd, vyaw_cmd = vx_raw, vyaw_raw
            else:
                vx_cmd   = VX_SMOOTH   * vx_raw   + (1.0 - VX_SMOOTH)   * vx_cmd
                vyaw_cmd = VYAW_SMOOTH * vyaw_raw  + (1.0 - VYAW_SMOOTH) * vyaw_cmd
            # Final safety cap — EMA can mix a previous-step high-vx-low-vyaw
            # sample with a new-step low-vx-high-vyaw one and briefly land in the
            # (vx≥0.35, vyaw≥0.34) unstable corner of the RL trot.  This re-cap
            # after smoothing guarantees the command actually sent to the gait
            # stays in the safe region.
            vx_cmd, vyaw_cmd = apply_stability_cap(vx_cmd, vyaw_cmd)

            # ── stuck-recovery: reverse + commit yaw to flip side ───────
            # When the stuck-detector fires, command REVERSE vx for ~2 s
            # AND override yaw to the flipped detour direction at near-max
            # rate.  Backing up creates clear space; the forced yaw means
            # by the time forward chase resumes, the robot is pointing at
            # the new detour lane instead of grinding back into the same
            # corner.  Without this forced yaw, the controller's normal
            # output keeps oscillating with stale-target bearings and the
            # robot just rocks in place.
            if "stuck_recovery_until" in locals() and step_num < stuck_recovery_until:
                vx_cmd = -0.30
                vyaw_cmd = +0.13 if stuck_flip else -0.13

            # While tipped, cap forward to a small nudge but DO NOT re-damp yaw —
            # yaw is how the robot re-orients, and the EMA + tilt-guard already shape
            # it.  Previously this halving stacked on top of both, which left the
            # robot with too little authority to recover before the 2 s auto-reset.
            if fell:
                vx_cmd   = min(vx_cmd, 0.15)

            # ── verbose step log ──────────────────────────────────────────
            if step_num % LOG_EVERY == 0:
                dist_m = math.hypot(rx - bx, ry - by)
                if area > 0:
                    # estimated distance from apparent area (inverse-square law)
                    from controller import (TARGET_AREA_FRAC, DIST_CAL,
                                            TARGET_DIST_GOAL, TARGET_DIST_MIN_SAFE)
                    _ta = TARGET_AREA_FRAC * cam_w * cam_h
                    dist_est = DIST_CAL * math.sqrt(_ta / max(area, 1.0))
                    dist_vis = f"area={int(area)}px²  dist_est≈{dist_est:.2f}m"
                    _tag = (
                        "CLOSE" if dist_m < TARGET_DIST_MIN_SAFE
                        else ("FAR" if dist_m > TARGET_DIST_GOAL else "IN-GOAL")
                    )
                    dist_vis += f"[{_tag}]"
                else:
                    dist_vis = "no detection"
                fall_tag = "  *** FELL ***" if _pose_incapped(rz, quat) else ""
                print(
                    f"[STEP {step_num:6d}] t={sim_time:.1f}s{fall_tag} | "
                    f"[{mover.phase_label}] "
                    f"pos=({rx:+.2f},{ry:+.2f}) z={rz:.3f}m | "
                    f"ball=({bx:+.2f},{by:+.2f}) dist_world={dist_m:.2f}m | "
                    f"cmd: vx={vx_cmd:+.3f} vyaw={vyaw_cmd:+.3f} | "
                    f"vision: {dist_vis}  conf={conf:.2f}",
                    flush=True,
                )

            # ── logging ───────────────────────────────────────────────────
            logger.step(
                step=step_num, sim_time=sim_time,
                robot_x=rx, robot_y=ry, robot_z=rz, robot_yaw=yaw,
                ball_x=bx, ball_y=by,
                cx=cx, cy=cy, area=area, confidence=conf,
                vx_cmd=vx_cmd, vyaw_cmd=vyaw_cmd,
                img_w=cam_w,
            )

            # Skip the entire visualisation block when this step did not
            # render — chase_rgb / head_rgb are stale, the HUD is duplicate
            # work, and the bash post-process retimes the video to wall-clock
            # length so dropped frames don't shorten the output.
            if not HEADLESS and do_render:
                # ── visualise: top-down ───────────────────────────────────
                head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
                head_bgr = annotate_frame(head_bgr, det)
                draw_distance_bar(head_bgr, area, conf, cx)
                hud(head_bgr, vx_cmd, vyaw_cmd, cx, area, conf,
                    step_num, sim_time, rz, fell, demo_phase=mover.phase_label)
                if GUI_CHASE_ONLY:
                    # 3rd-person chase camera + 4-panel dashboard composite.
                    chase_bgr = cv2.cvtColor(chase_rgb, cv2.COLOR_RGB2BGR)
                    # Cinematic vignette (spotlight darkening at edges).
                    _apply_vignette(chase_bgr, strength=0.45)

                    # Compute number of obstacle rows the robot is past
                    seen_x = set()
                    for k_, ox_, oy_ in all_robot_obstacles:
                        if ox_ + OBS_X_HALF < rx:
                            seen_x.add(round(ox_, 1))
                    n_passed = len(seen_x)        # 1 object = 1 row (user convention)

                    _stage_num = getattr(mover, "_stage", 1)
                    _collisions = locals().get("collision_count", 0)
                    _failed = locals().get("collision_failed", False)
                    _hud_title = f"GO2 STAGE {_stage_num}   ROWS {n_passed}"
                    _hud_sub = (
                        f"COLLISIONS: {_collisions}  STATUS: "
                        + ("FAILED" if _failed else "OK")
                    )
                    _draw_styled_hud(chase_bgr, title=_hud_title, subtitle=_hud_sub)
                    # Big stage banner across chase view's lower-left corner
                    _stage_text = f"STAGE {_stage_num}"
                    cv2.putText(chase_bgr, _stage_text, (16, chase_bgr.shape[0] - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (10, 10, 14), 6, cv2.LINE_AA)
                    cv2.putText(chase_bgr, _stage_text, (16, chase_bgr.shape[0] - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 235, 255), 2, cv2.LINE_AA)
                    if _failed:
                        # Red FAIL overlay
                        _ovr = chase_bgr.copy()
                        cv2.rectangle(_ovr, (0, 0), (chase_bgr.shape[1], chase_bgr.shape[0]),
                                      (0, 0, 200), -1)
                        cv2.addWeighted(_ovr, 0.25, chase_bgr, 0.75, 0, chase_bgr)
                        cv2.putText(chase_bgr, "COLLISION — FAILED",
                                    (40, chase_bgr.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (10,10,10), 8, cv2.LINE_AA)
                        cv2.putText(chase_bgr, "COLLISION — FAILED",
                                    (40, chase_bgr.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40,40,255), 3, cv2.LINE_AA)

                    # Build 720x720 dashboard: chase + 4 cam tiles + info.
                    _lane = (mover.phase_label.split("lane=")[-1][:6]
                             if "lane=" in mover.phase_label else "—")
                    info_kwargs = dict(
                        sim_t=sim_time,
                        n_passed=n_passed,
                        total_rows=2 * int(os.environ.get("OBS_ROWS", "50")),
                        vx_cmd=vx_cmd,
                        vyaw_cmd=vyaw_cmd,
                        det_conf=float(conf or 0.0),
                        ball_dist=math.hypot(bx - rx, by - ry),
                        lane_label=_lane.strip(),
                        robot_yaw=yaw,
                        robot_x=rx,
                        fell=fell,
                        robot_xy=(rx, ry),
                        ball_xy=(bx, by),
                    )
                    dashboard = _build_dashboard(
                        chase_bgr,
                        front_rgb=head_rgb_overlay, front_det=det,
                        map_rgb=map_rgb,
                        left_rgb=left_rgb,
                        right_rgb=right_rgb,
                        info_kwargs=info_kwargs,
                    )

                    cv2.imshow("Go2 Object Tracking", dashboard)
                    if "video_writer" in locals() and video_writer is not None:
                        video_writer.write(dashboard)

                    # Early-stop: once we've cleared all configured rows,
                    # there's no more obstacle interaction worth recording.
                    _stop_rows = 2 * int(os.environ.get("OBS_ROWS", "50"))   # 1 object = 1 row
                    # Robust stop: any of (a) n_passed reached target,
                    # (b) robot is past the last placed obstacle by ≥5 m.
                    _last_obs_x = max((ox_ for _t, ox_, _oy in all_robot_obstacles),
                                      default=0.0)
                    if n_passed >= _stop_rows or rx > _last_obs_x + 5.0:
                        print(f"[REC] Stages complete — n_passed={n_passed}/{_stop_rows} "
                              f"rx={rx:.1f} last_obs_x={_last_obs_x:.1f}; stopping.",
                              flush=True)
                        running = False
                elif GUI_HEAD_ONLY:
                    cv2.imshow("Go2 Object Tracking", head_bgr)
                    if "video_writer" in locals() and video_writer is not None:
                        video_writer.write(head_bgr)
                else:
                    top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
                    draw_trail(top_bgr, robot_trail, (255, 130, 0))   # orange
                    draw_trail(top_bgr, ball_trail,  (40,  40, 200))  # red-ish
                    draw_heading_arrow(top_bgr, rx, ry, yaw)
                    rpx, rpy = _world_to_px(rx, ry)
                    bpx, bpy = _world_to_px(bx, by)
                    cv2.circle(top_bgr, (rpx, rpy), 6, (255, 140, 0), -1)
                    cv2.circle(top_bgr, (rpx, rpy), 6, (255, 255, 255), 1)
                    cv2.circle(top_bgr, (bpx, bpy), 5, (30,  30, 220), -1)
                    cv2.circle(top_bgr, (bpx, bpy), 5, (255, 255, 255), 1)
                    draw_label(top_bgr, "OVERVIEW", (6, 18), (180, 230, 255))
                    draw_label(top_bgr, mover.phase_label[:56], (6, 40), (200, 255, 200))

                    chase_bgr = cv2.cvtColor(chase_rgb, cv2.COLOR_RGB2BGR)
                    draw_label(chase_bgr, "CHASE (3RD PERSON)", (6, 18), (220, 220, 255))

                    rear_bgr = cv2.cvtColor(rear_rgb, cv2.COLOR_RGB2BGR)
                    draw_label(rear_bgr, "REAR CAM", (6, 18), (200, 255, 220))

                    gap_h = np.zeros((cam_h, 4, 3), np.uint8)
                    top_row = np.concatenate([top_bgr, gap_h, chase_bgr], axis=1)
                    bot_row = np.concatenate([head_bgr, gap_h, rear_bgr], axis=1)
                    gap_v = np.zeros((4, top_row.shape[1], 3), np.uint8)
                    combined = np.concatenate([top_row, gap_v, bot_row], axis=0)
                    cv2.imshow("Go2 Object Tracking", combined)
                    if "video_writer" in locals() and video_writer is not None:
                        video_writer.write(combined)

            if (HEADLESS and step_num >= HEADLESS_MAX_STEPS) or (
                MAX_RUNTIME_STEPS > 0 and step_num >= MAX_RUNTIME_STEPS
            ):
                running = False
            # Auto-stop on recording duration (wall clock)
            if (video_writer is not None and record_duration > 0
                    and record_start_wall is not None
                    and (time.perf_counter() - record_start_wall) >= record_duration):
                print(f"[REC] Reached {record_duration:.0f} s wall — stopping.",
                      flush=True)
                running = False

        # ── cache the planned path so the next frame's chase render
        #     can inject it as 3-D scene geoms (proper z-buffer occlusion).
        if not paused:
            _Z_FLOOR_OVL = 0.012
            # Path overlay = the robot's ACTUAL trajectory:
            #   recent trail (where robot has been) + forward arc projection
            #   based on the current vx, vyaw — so what's drawn matches what
            #   the robot is actually doing.
            _wps = []
            # Past trail — sample a few recent positions for context.
            _trail_list = list(robot_trail)
            for _tx, _ty in _trail_list[-12:]:
                _wps.append((_tx, _ty, _Z_FLOOR_OVL))
            _wps.append((rx, ry, _Z_FLOOR_OVL))
            # Forward arc projection: assume robot keeps current vx, vyaw.
            # Project ~5 s ahead so the arrow visibly extends toward where
            # the robot is headed.  Heavy EMA on the projection inputs so
            # the arrow doesn't whip around on every controller jitter.
            if "_arrow_vx" not in locals():
                _arrow_vx = float(vx_cmd)
                _arrow_w  = float(vyaw_cmd)
            _arrow_alpha = 0.05   # ~20-step time constant (≈0.4 s @ 50 Hz)
            _arrow_vx = (1.0 - _arrow_alpha) * _arrow_vx + _arrow_alpha * float(vx_cmd)
            _arrow_w  = (1.0 - _arrow_alpha) * _arrow_w  + _arrow_alpha * float(vyaw_cmd)
            _proj_vx = max(0.05, abs(_arrow_vx))
            _proj_w  = _arrow_w
            _proj_T_total = 5.0
            _proj_steps = 28
            _proj_dt = _proj_T_total / _proj_steps
            _px, _py, _pyaw = rx, ry, yaw
            for _k in range(_proj_steps):
                _px += _proj_vx * _proj_dt * math.cos(_pyaw)
                _py += _proj_vx * _proj_dt * math.sin(_pyaw)
                _pyaw += _proj_w * _proj_dt
                _wps.append((_px, _py, _Z_FLOOR_OVL))
            cached_path_waypoints = _wps

            # ── Forecasted-path waypoints (red arrow on floor) ─────────
            # Forward-simulate the pure-pursuit controller through the
            # committed planner path.  This is the actual predicted
            # trajectory the robot will follow given its yaw and speed
            # caps — so the arrow matches what the robot does, not the
            # idealized planner path.  Truncate cleanly when we reach
            # the ball or run out of horizon (never connect to ball
            # through obstacles).
            _disp_path = None
            if ("shared_path_pts" in locals() and shared_path_pts
                    and len(shared_path_pts) >= 2):
                _disp_path = list(shared_path_pts)
            elif "ahead_obstacles" in locals() and ahead_obstacles:
                _blocker = None
                for (_ox, _oy, _oh) in ahead_obstacles:
                    if _ox <= rx + 0.05 or _ox >= bx - 0.05:
                        continue
                    if abs(bx - rx) < 0.05:
                        continue
                    _t = (_ox - rx) / (bx - rx)
                    if 0.0 < _t < 1.0:
                        _y_at = ry + _t * (by - ry)
                        if abs(_y_at - _oy) < (_oh + 0.30):
                            _blocker = (_ox, _oy, _oh)
                            break
                if _blocker is not None:
                    _ox, _oy, _oh = _blocker
                    _side_up = _oy + _oh + 0.35
                    _side_dn = _oy - _oh - 0.35
                    _yc = _side_up if abs(by - _side_up) < abs(by - _side_dn) else _side_dn
                    _disp_path = [
                        (rx, ry),
                        (_ox - 0.6, _yc),
                        (_ox + 0.6, _yc),
                        (bx, by),
                    ]
            if _disp_path is None:
                _disp_path = [(rx, ry), (bx, by)]

            _plan = [(rx, ry, _Z_FLOOR_OVL)]
            if len(_disp_path) >= 2:
                _sim_path = _disp_path
                _sx, _sy, _syaw = rx, ry, yaw
                _SIM_DT = 0.08
                _SIM_N  = 40   # 3.2 s horizon
                _SIM_LOOKAHEAD = 1.4
                # Stable nominal speed so arrow length doesn't jitter with
                # brake / detection-state command swings.
                _SIM_VX = 0.8
                _SIM_YAW_CAP = 0.42
                for _k in range(_SIM_N):
                    _ci = 0; _cd = 1e9
                    for _i, (_px, _py) in enumerate(_sim_path):
                        _d = math.hypot(_px - _sx, _py - _sy)
                        if _d < _cd: _cd = _d; _ci = _i
                    _acc = 0.0
                    _tx, _ty = _sim_path[-1]
                    for _i in range(_ci + 1, len(_sim_path)):
                        _p0 = _sim_path[_i - 1]; _p1 = _sim_path[_i]
                        _seg = math.hypot(_p1[0]-_p0[0], _p1[1]-_p0[1])
                        if _acc + _seg >= _SIM_LOOKAHEAD:
                            _f = (_SIM_LOOKAHEAD - _acc) / max(_seg, 1e-6)
                            _tx = _p0[0] + _f*(_p1[0]-_p0[0])
                            _ty = _p0[1] + _f*(_p1[1]-_p0[1])
                            break
                        _acc += _seg
                    _desired = math.atan2(_ty - _sy, _tx - _sx)
                    _yaw_err = _desired - _syaw
                    while _yaw_err > math.pi:  _yaw_err -= 2*math.pi
                    while _yaw_err < -math.pi: _yaw_err += 2*math.pi
                    _w = max(-_SIM_YAW_CAP, min(_SIM_YAW_CAP, _yaw_err * 2.5))
                    _syaw += _w * _SIM_DT
                    _sx += _SIM_VX * _SIM_DT * math.cos(_syaw)
                    _sy += _SIM_VX * _SIM_DT * math.sin(_syaw)
                    _plan.append((_sx, _sy, _Z_FLOOR_OVL))
                    if math.hypot(bx - _sx, by - _sy) < 0.6:
                        # Close enough to the ball — connect directly.
                        _plan.append((bx, by, _Z_FLOOR_OVL))
                        break
                # NOTE: if the loop exits without reaching the ball, we
                # leave the plan truncated at the simulation horizon —
                # never connect to the ball through obstacles.
            cached_plan_waypoints = _plan

        # ── keyboard ─────────────────────────────────────────────────────
        if HEADLESS:
            key = 0
        else:
            key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            running = False
        elif key == ord("r"):
            _reset(model, data, ramp_start, gait,
                   tracker, controller, robot_trail, ball_trail, mover)
            fell_at = None
        elif key == ord("p"):
            paused = not paused
            print("Paused." if paused else "Resumed.")

        # ── real-time pacing ──────────────────────────────────────────────
        # SPEED env var: scale the wall-clock budget so 3.0 → 3× faster playback.
        if not HEADLESS:
            t_now   = time.perf_counter()
            elapsed = t_now - t_prev
            speed_mult = float(os.environ.get("SIM_SPEED", "1.0"))
            budget  = CTRL_DT / max(speed_mult, 0.01)
            if elapsed < budget:
                time.sleep(budget - elapsed)
            t_prev = time.perf_counter()
        else:
            t_prev = time.perf_counter()

    if video_writer is not None:
        video_writer.release()
        print(f"[REC] Video saved: {record_path}", flush=True)
    if not HEADLESS:
        cv2.destroyAllWindows()
    logger.close()
    print("Simulation ended.")


if __name__ == "__main__":
    main()
