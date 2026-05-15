"""
logger.py – Structured run logging for the Go2 object-tracking simulation.

Each control step is logged to a CSV file in sim/logs/.
A human-readable summary is printed to the terminal every SUMMARY_EVERY steps.
A final report is printed on close().

CSV columns
-----------
step        : control-loop iteration counter
sim_time    : simulation time in seconds
robot_x/y/z : robot base world position (m)
robot_yaw   : robot yaw angle (rad)
ball_x/y    : target ball world position (m)
cx / cy     : detected ball pixel centre (-1 if not detected)
area        : detected blob area in px²  (0 if not detected)
confidence  : tracker confidence 0–1
vx_cmd      : commanded forward velocity (m/s)
vyaw_cmd    : commanded yaw rate (rad/s)
dist_to_ball: Euclidean distance robot→ball in the XY plane (m)
detected    : 1 if ball was detected this step (cx valid), else 0
dist_ema    : exponential moving average of dist (tuning / trend)
fell        : 1 if robot_z < FALL_Z_THRESH, else 0
"""

from __future__ import annotations

import csv
import math
import pathlib

from controller import TARGET_DIST_GOAL, TARGET_DIST_MIN_SAFE
import sys
import time
from datetime import datetime
from typing import Optional

# ── constants ─────────────────────────────────────────────────────────────────
SUMMARY_EVERY = 25         # print a terminal line every N control steps
FALL_Z_THRESH = 0.12       # metres – robot considered fallen below this height
DIST_EMA_ALPHA = 0.03      # EMA on dist for CSV trend column
LOG_DIR       = pathlib.Path(__file__).parent / "logs"

_CSV_FIELDS = [
    "step", "sim_time",
    "robot_x", "robot_y", "robot_z", "robot_yaw",
    "ball_x", "ball_y",
    "cx", "cy", "area", "confidence",
    "vx_cmd", "vyaw_cmd",
    "dist_to_ball", "detected", "dist_ema", "fell", "img_w",
]


class SimLogger:
    """
    Attach one instance to main() and call .step() every control cycle.
    """

    def __init__(self, tag: str = "default") -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = LOG_DIR / f"run_{ts}_{tag}.csv"
        self._fh = open(self._csv_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._fh, fieldnames=_CSV_FIELDS)
        self._writer.writeheader()

        # accumulators for the final report
        self._total_steps     = 0
        self._detected_steps  = 0
        self._fell_steps      = 0
        self._x_err_sum       = 0.0   # sum of |pixel x error| (when detected)
        self._dist_sum        = 0.0   # sum of robot→ball distances
        self._dist_in_range   = 0     # steps where MIN_SAFE < dist ≤ GOAL (under 1 m)
        self._dist_ema        = 0.0   # running EMA of distance

        self._tag     = tag
        self._t_start = time.perf_counter()
        print(f"[LOG:{tag}] Writing CSV to {self._csv_path}")
        print(f"[LOG:{tag}] Terminal summary every {SUMMARY_EVERY} steps  "
              f"(fall threshold z<{FALL_Z_THRESH} m)")

    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        step: int,
        sim_time: float,
        robot_x: float, robot_y: float, robot_z: float, robot_yaw: float,
        ball_x:  float, ball_y:  float,
        cx: Optional[int], cy: Optional[int],
        area: float, confidence: float,
        vx_cmd: float, vyaw_cmd: float,
        img_w: int,
        img_h: int = 0,
    ) -> None:
        """Log one control step. Call from the main loop."""

        dist = math.hypot(robot_x - ball_x, robot_y - ball_y)
        fell = 1 if robot_z < FALL_Z_THRESH else 0
        detected = 1 if cx is not None else 0

        if step <= 1:
            self._dist_ema = dist
        else:
            a = DIST_EMA_ALPHA
            self._dist_ema = a * dist + (1.0 - a) * self._dist_ema

        # accumulate for report
        self._total_steps += 1
        if cx is not None:
            self._detected_steps += 1
            x_err = abs(cx - img_w / 2)
            self._x_err_sum += x_err
        self._fell_steps  += fell
        self._dist_sum    += dist
        if TARGET_DIST_MIN_SAFE < dist <= TARGET_DIST_GOAL:
            self._dist_in_range += 1

        row = {
            "step":        step,
            "sim_time":    f"{sim_time:.4f}",
            "robot_x":     f"{robot_x:.4f}",
            "robot_y":     f"{robot_y:.4f}",
            "robot_z":     f"{robot_z:.4f}",
            "robot_yaw":   f"{robot_yaw:.4f}",
            "ball_x":      f"{ball_x:.4f}",
            "ball_y":      f"{ball_y:.4f}",
            "cx":          cx if cx is not None else -1,
            "cy":          cy if cy is not None else -1,
            "area":        f"{area:.1f}",
            "confidence":  f"{confidence:.3f}",
            "vx_cmd":      f"{vx_cmd:.4f}",
            "vyaw_cmd":    f"{vyaw_cmd:.4f}",
            "dist_to_ball": f"{dist:.4f}",
            "detected":    detected,
            "dist_ema":    f"{self._dist_ema:.4f}",
            "fell":        fell,
            "img_w":       img_w,
        }
        self._writer.writerow(row)

        # periodic terminal summary
        if step % SUMMARY_EVERY == 0:
            self._print_summary(step, sim_time, robot_x, robot_y, robot_z,
                                ball_x, ball_y, cx, area, confidence,
                                vx_cmd, vyaw_cmd, dist, fell, self._dist_ema)

    # ──────────────────────────────────────────────────────────────────────────

    def _print_summary(
        self,
        step: int, sim_time: float,
        rx: float, ry: float, rz: float,
        bx: float, by: float,
        cx, area: float, conf: float,
        vx: float, vyaw: float,
        dist: float, fell: int,
        dist_ema: float,
    ) -> None:
        det_str  = f"cx={cx:4d} area={int(area):5d}px²" if cx is not None else "NOT DETECTED      "
        fall_str = "  *** FELL ***" if fell else ""
        # Distance range tag
        if dist <= TARGET_DIST_MIN_SAFE:
            range_tag = "CLOSE"
        elif dist > TARGET_DIST_GOAL:
            range_tag = "FAR  "
        else:
            range_tag = "OK   "
        print(
            f"[LOG {step:6d}] t={sim_time:6.1f}s{fall_str} | "
            f"z={rz:.3f}m | "
            f"dist={dist:.2f}m(ema={dist_ema:.2f})[{range_tag}] | "
            f"{det_str} conf={conf:.2f} | "
            f"vx={vx:+.3f} vyaw={vyaw:+.3f}",
            flush=True,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush CSV, print final report, close file."""
        self._fh.close()
        wall = time.perf_counter() - self._t_start
        n    = max(self._total_steps, 1)

        det_pct      = 100.0 * self._detected_steps / n
        fall_pct     = 100.0 * self._fell_steps / n
        in_range_pct = 100.0 * self._dist_in_range / n
        avg_dist     = self._dist_sum / n
        avg_x_err    = self._x_err_sum / max(self._detected_steps, 1)

        print("\n" + "=" * 70)
        print("SIMULATION FINAL REPORT")
        print("=" * 70)
        print(f"  Total control steps  : {self._total_steps}")
        print(f"  Wall time            : {wall:.1f} s")
        print(f"  Ball detected        : {det_pct:.1f}% of steps")
        print(f"  Robot fell           : {fall_pct:.1f}% of steps")
        print(f"  Avg robot→ball dist  : {avg_dist:.3f} m")
        print(f"  In goal (≤{TARGET_DIST_GOAL:g} m, >{TARGET_DIST_MIN_SAFE:g} m) : "
              f"{in_range_pct:.1f}% of steps  (higher = better)")
        print(f"  Avg |pixel x error|  : {avg_x_err:.1f} px  (lower = better centering)")
        print(f"  CSV saved to         : {self._csv_path}")
        print("=" * 70 + "\n")
