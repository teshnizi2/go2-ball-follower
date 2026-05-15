"""
Joint / actuator ordering for Go2 MuJoCo models.

* **Project / XML order** ([assets/go2.xml](assets/go2.xml) actuators):  
  indices 0–11 = FL(hip,thigh,calf), FR(...), RL(...), RR(...).

* **LocoMuJoCo `UnitreeGo2` observation joint order** (documentation):  
  FR, FL, RR, RL for the 12 joint positions (and same for velocities).

Use `locomujoco_to_project_ctrl` / `project_to_locomujoco_ctrl` when feeding
12-D normalized torques from LocoMuJoCo-style policies into this simulation.
"""

from __future__ import annotations

import numpy as np

# MuJoCo actuator order in this repo (same as Menagerie go2.xml listing)
PROJECT_LEGS: tuple[str, ...] = ("FL", "FR", "RL", "RR")

# LocoMuJoCo default observation joint order (see loco-mujoco docs, unitreeGo2)
LOCOMUJOCO_LEGS: tuple[str, ...] = ("FR", "FL", "RR", "RL")


# Project ctrl order: FL(0:3), FR(3:6), RL(6:9), RR(9:12)
# LocoMuJoCo joint order: FR, FL, RR, RL  → indices [3:6],[0:3],[9:12],[6:9]
_PROJECT_TO_LOCO_IDX = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
_LOCO_TO_PROJECT_IDX = np.argsort(_PROJECT_TO_LOCO_IDX)


def project_to_locomujoco_ctrl(u_project: np.ndarray) -> np.ndarray:
    """Map 12-vector from project/Menagerie actuator order to LocoMuJoCo order."""
    u = np.asarray(u_project, dtype=float).reshape(12)
    return u[_PROJECT_TO_LOCO_IDX]


def locomujoco_to_project_ctrl(u_loco: np.ndarray) -> np.ndarray:
    """Map 12-vector from LocoMuJoCo order to project actuator order."""
    u = np.asarray(u_loco, dtype=float).reshape(12)
    return u[_LOCO_TO_PROJECT_IDX]


def denormalize_torques(
    u_norm: np.ndarray,
    ctrlrange_lo: np.ndarray,
    ctrlrange_hi: np.ndarray,
) -> np.ndarray:
    """Map [-1, 1]^12 actions to physical torques using per-actuator ctrlrange."""
    u = np.asarray(u_norm, dtype=float).reshape(12)
    lo = np.asarray(ctrlrange_lo, dtype=float).reshape(12)
    hi = np.asarray(ctrlrange_hi, dtype=float).reshape(12)
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + u * half
