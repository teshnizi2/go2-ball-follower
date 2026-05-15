"""
Pluggable low-level controllers: map high-level (vx, vyaw) to MuJoCo `data.ctrl`.

Default: RLVelocityLowLevel (RL-trained velocity-command policy for Go2).
Fallback: TrotPDGait (hand-tuned diagonal trot), used if the policy file is missing.
"""

from __future__ import annotations

import math
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    pass

# ── standing pose (matches keyframe in go2.xml) ─────────────────────────────
HIP_S, THIGH_S, CALF_S = 0.0, 0.9, -1.8
JSTART, VSTART = 7, 6
KP, KD = 150.0, 4.0

# gait_phase offsets: FL/RR in phase, FR/RL pi out (trot diagonal pairs)
LEG_DEFS: list[tuple[str, int, float, float]] = [
    ("FL", 0, 0.0, -1.0),
    ("RR", 9, 0.0, +1.0),
    ("FR", 3, math.pi, +1.0),
    ("RL", 6, math.pi, -1.0),
]


def quat_to_pitch_roll(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll


def trot_targets(
    phase: float,
    vx: float,
    vyaw: float,
    pitch: float,
    roll: float,
    ramp: float,
) -> np.ndarray:
    targets = np.array([HIP_S, THIGH_S, CALF_S] * 4, dtype=float)
    lift_amp = float(np.clip(0.18 + abs(vx) * 0.15, 0.18, 0.40)) * ramp
    sweep_amp = float(np.clip(vx * 2.2, -0.40, 0.40)) * ramp
    turn_amp = float(np.clip(abs(vyaw) * 0.12, 0.0, 0.20))
    pitch_corr = float(np.clip(-pitch * 0.40, -0.20, 0.20))
    roll_corr = float(np.clip(-roll * 0.25, -0.14, 0.14))

    for name, idx, base_ph, hip_sign in LEG_DEFS:
        lp = (phase + base_ph) % (2 * math.pi)
        sin_lp = math.sin(lp)
        cos_lp = math.cos(lp)
        targets[idx + 1] = THIGH_S + sweep_amp * cos_lp
        calf_lift = lift_amp * max(0.0, sin_lp)
        targets[idx + 2] = CALF_S - calf_lift
        targets[idx] = (
            HIP_S
            + hip_sign * turn_amp * math.sin(lp + math.pi / 2)
            + hip_sign * 0.10 * vyaw
        )
        targets[idx + 1] += pitch_corr
        is_left = name in ("FL", "RL")
        targets[idx + 1] += roll_corr * (1.0 if is_left else -1.0)
    return targets


class LowLevelBase(ABC):
    """Interface: update `data.ctrl` for one physics substep."""

    def reset_phase(self, value: float = 0.0) -> None:
        """Optional reset hook (e.g. for gait phase); default is a no-op."""

    @abstractmethod
    def substep(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        vx_cmd: float,
        vyaw_cmd: float,
        ramp_start_time: float,
        sim_time: float,
        sim_dt: float,
    ) -> None:
        ...


class TrotPDGait(LowLevelBase):
    """Diagonal trot reference + PD tracking at physics rate."""

    def __init__(self, model: mujoco.MjModel, phase_rate_hz: float = 2.2) -> None:
        self._lo = np.asarray(model.actuator_ctrlrange[:, 0], dtype=float)
        self._hi = np.asarray(model.actuator_ctrlrange[:, 1], dtype=float)
        self.phase = 0.0
        self.phase_rate = 2 * math.pi * phase_rate_hz
        self.ramp_dur = 1.0

    def reset_phase(self, value: float = 0.0) -> None:
        self.phase = value

    def substep(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        vx_cmd: float,
        vyaw_cmd: float,
        ramp_start_time: float,
        sim_time: float,
        sim_dt: float,
    ) -> None:
        del model
        ramp = float(np.clip((sim_time - ramp_start_time) / self.ramp_dur, 0.0, 1.0))
        eff_rate = self.phase_rate * (0.6 + 0.6 * min(abs(vx_cmd), 0.8))
        self.phase = (self.phase + eff_rate * sim_dt) % (2 * math.pi)
        pitch, roll = quat_to_pitch_roll(data.qpos[3:7])
        tgt = trot_targets(self.phase, vx_cmd, vyaw_cmd, pitch, roll, ramp)
        qpos = data.qpos[JSTART:]
        qvel = data.qvel[VSTART:]
        raw = KP * (tgt - qpos) - KD * qvel
        data.ctrl[:] = np.clip(raw, self._lo, self._hi)


class RawTorqueLowLevel(LowLevelBase):
    """
    Apply a precomputed 12-D torque vector each substep (e.g. external policy).

    Call `set_torques(tau)` from your policy thread / before mj_step.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self._lo = np.asarray(model.actuator_ctrlrange[:, 0], dtype=float)
        self._hi = np.asarray(model.actuator_ctrlrange[:, 1], dtype=float)
        self._tau = np.zeros(12, dtype=float)

    def set_torques(self, tau: np.ndarray) -> None:
        self._tau = np.asarray(tau, dtype=float).reshape(12).copy()

    def substep(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        vx_cmd: float,
        vyaw_cmd: float,
        ramp_start_time: float,
        sim_time: float,
        sim_dt: float,
    ) -> None:
        del model, vx_cmd, vyaw_cmd, ramp_start_time, sim_time, sim_dt
        data.ctrl[:] = np.clip(self._tau, self._lo, self._hi)


# ── RL velocity-command policy ────────────────────────────────────────────────

# Path to the pre-trained checkpoint (download with scripts/download_policy.py)
_POLICY_PATH = pathlib.Path(__file__).parent / "policy" / "model_500.pt"

# Training-matched default joint positions: FL, FR, RL, RR × (hip, thigh, calf)
# from env.yaml: .*R_hip=0.1, .*L_hip=-0.1, thigh=0.9, calf=-1.8
_RL_DEFAULT_POS = np.array(
    [-0.1, 0.9, -1.8,   # FL
      0.1, 0.9, -1.8,   # FR
     -0.1, 0.9, -1.8,   # RL
      0.1, 0.9, -1.8],  # RR
    dtype=np.float32,
)

# Observations are passed RAW to the normalizer (no pre-scaling).
# The running normalizer (mean/std) stored in the checkpoint handles all
# normalization.  Confirmed by inspecting normalizer._std:
#   ang_vel std ≈ 0.5–0.8 rad/s (raw body-frame gyro)
#   joint_vel std ≈ 1.8–2.6 rad/s (raw joint velocities)
#   command std ≈ 0.4–0.5 (raw m/s / rad/s)

# PD gains from training env.yaml (stiffness=20, damping=1)
_RL_KP = 20.0
_RL_KD = 1.0

# Action scale: policy output → joint position offsets from default
import os as _os
_ACTION_SCALE_BASE = 0.25
_ACTION_SCALE_MULT_BASE = float(_os.environ.get("ACTION_SCALE_MULT", "1.0"))
_dynamic_action_mult = [1.0]   # 1-element list = mutable handle for runtime updates

def set_dynamic_action_mult(m: float) -> None:
    """Runtime knob: scales joint-target amplitude in [0.0, 1.0].
    1.0 = full ACTION_SCALE_MULT; 0.0 = baseline (untrained-speed-safe)."""
    _dynamic_action_mult[0] = max(0.0, min(1.0, float(m)))

def _current_action_scale() -> float:
    # Lerp between 1.0× (safe baseline) and ACTION_SCALE_MULT_BASE (stress)
    # based on the runtime knob.
    m = _dynamic_action_mult[0]
    effective_mult = 1.0 + (_ACTION_SCALE_MULT_BASE - 1.0) * m
    return _ACTION_SCALE_BASE * effective_mult

# Backwards-compat constant for any code reading it at module level (legacy).
_ACTION_SCALE = _ACTION_SCALE_BASE * _ACTION_SCALE_MULT_BASE

# Policy runs at 50 Hz (training: decimation=4 at 200 Hz, equivalent to 50 Hz)
# Our physics: 500 Hz → run policy every 10 physics steps
_POLICY_DECIMATE = 10

# Velocity command limits: clip before building the obs so the normalised value
# stays within ≈±5 std (the RSL-RL clip window).  At vyaw=2.5 the normalised
# value is 2.5/0.531 ≈ 4.7 std — safe.  Experimentally the robot achieves
# ~0.7 rad/s effective yaw at cmd=2.5, comfortably ahead of the 0.22 rad/s orbit.
_CMD_VX_MAX   = 3.0   # m/s (raised; effective value clipped to ±5σ anyway)
_CMD_VYAW_MAX = 2.5   # rad/s  (≈4.7 σ, within the ±5 clip; effective ≈0.7 rad/s)

# Forward velocity observation scaling.
# Problem: vx_cmd=0.8 m/s normalises to only ~1.8σ while vyaw=2.5 normalises
# to ~4.7σ.  The policy barely responds to small normalised vx inputs.
# Fix: scale vx in the observation so moderate commands give meaningful σ values.
# With scale=1.8, vx_cmd=1.2 → obs=1.2*1.8/0.4251≈5.1σ (just saturated) and
# vx_cmd=0.6 → 2.54σ (unsaturated gradient).  This gives a gentler speed ramp
# at 1.5–2 m distances (typical orbit range) and prevents the gait instability
# that caused consistent falls after ~65 s of sustained full-speed running.
_VX_OBS_SCALE = 2.5   # raised from 1.8: at 1.8 vx=0.8 gave only ~3.4σ → ~0.03 m/s actual
                       # displacement in MuJoCo (sim-to-sim transfer gap). At 2.5 the
                       # normalised cmd is 0.8*2.5/0.4251≈4.7σ matching the vyaw regime.

# Yaw observation scaling.
# With MAX_VYAW capped at 1.2 rad/s (to prevent falls from sustained high yaw),
# the normalised vyaw = 1.2/0.531 ≈ 2.3σ — the policy achieves only ~0.34 rad/s
# effective yaw, too slow to track the ball during orbit phases (0.12-0.35 rad/s).
# Scaling by 2.0 raises the normalised cmd to ~4.6σ (effective ~0.65 rad/s) while
# keeping the HIGH-LEVEL command capped at 1.2 (proportional, not sustained).
_VYAW_OBS_SCALE = 2.0   # scale applied to vyaw before inserting into obs


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3×3 rotation matrix (body → world)."""
    w, x, y, z = q.astype(float)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


class RLVelocityLowLevel(LowLevelBase):
    """
    RSL-RL trained velocity-command policy for Unitree Go2.

    Observation (45-dim, matches diasAiMaster/unitree-go2-velocity-flat):
      [0:3]   angular velocity (body frame, raw rad/s)
      [3:6]   gravity vector projected into body frame (raw)
      [6:9]   velocity commands [vx, vy=0, vyaw] (raw m/s and rad/s)
      [9:21]  joint_pos − default_pos          (FL,FR,RL,RR × hip,thigh,calf)
      [21:33] joint_vel (raw rad/s)
      [33:45] last action (raw policy output, unscaled)

    All values are passed raw to the stored running normalizer (mean/std from the
    checkpoint).  No manual per-feature scaling is applied.

    Actions (12-dim): joint position offsets from default, scaled by 0.25.
    PD torques: KP=20·(target−qpos) − KD·qvel, clipped to actuator limits.
    Policy runs at 50 Hz (_POLICY_DECIMATE physics steps between updates).

    If the checkpoint is missing or torch fails, silently falls back to TrotPDGait.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self._lo = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float32)
        self._hi = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._target_pos  = _RL_DEFAULT_POS.copy()   # held between policy steps
        self._substep_ctr = 0
        self._loaded = False
        self._fallback: TrotPDGait | None = None

        # Load policy (may set _loaded=False and create fallback on error)
        self._actor = None
        self._norm_mean: np.ndarray | None = None
        self._norm_std:  np.ndarray | None = None
        self._load_policy(model)

    # ── policy loading ────────────────────────────────────────────────────────

    def _load_policy(self, model: mujoco.MjModel) -> None:
        if not _POLICY_PATH.exists():
            print(f"[RL] Policy file not found: {_POLICY_PATH}")
            print("[RL] Run:  python scripts/download_policy.py  then restart.")
            self._use_fallback(model)
            return
        try:
            import torch
            import torch.nn as nn

            ckpt = torch.load(str(_POLICY_PATH), map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"]

            # Reconstruct actor MLP: 45 → 512 → 256 → 128 → 12  with ELU
            actor = nn.Sequential(
                nn.Linear(45, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 12),
            )
            layer_keys = [("actor.0", 0), ("actor.2", 2), ("actor.4", 4), ("actor.6", 6)]
            for key, idx in layer_keys:
                actor[idx].weight.data.copy_(sd[f"{key}.weight"])
                actor[idx].bias.data.copy_(sd[f"{key}.bias"])
            actor.eval()
            self._actor = actor

            # Running normalizer stats (applied before the actor)
            self._norm_mean = sd["actor_obs_normalizer._mean"].numpy().squeeze()
            self._norm_std  = sd["actor_obs_normalizer._std"].numpy().squeeze()
            # Clamp std to avoid division by near-zero (shouldn't happen with trained model)
            self._norm_std = np.maximum(self._norm_std, 1e-6)

            self._loaded = True
            print(f"[RL] Policy loaded: {_POLICY_PATH.name}  "
                  f"(iter={ckpt.get('iter','?')}, 45→12, KP={_RL_KP}, KD={_RL_KD})")
            # Diagnostic: show normaliser scales for command dims so we can tune _VX_OBS_SCALE.
            print(f"[RL] norm std  cmd=[{self._norm_std[6]:.4f}, {self._norm_std[7]:.4f}, {self._norm_std[8]:.4f}]"
                  f"  (vx, vy, vyaw)  — vx_sat_cmd≈{5.0*self._norm_std[6]:.2f} m/s"
                  f"  → scale×{_VX_OBS_SCALE} effective sat≈{_VX_OBS_SCALE*0.8/self._norm_std[6]:.1f}σ")

        except Exception as exc:
            print(f"[RL] Failed to load policy ({exc}). Falling back to TrotPDGait.")
            self._use_fallback(model)

    def _use_fallback(self, model: mujoco.MjModel) -> None:
        self._loaded = False
        self._fallback = TrotPDGait(model)

    # ── LowLevelBase interface ────────────────────────────────────────────────

    def reset_phase(self, value: float = 0.0) -> None:
        self._last_action[:] = 0.0
        self._target_pos[:]  = _RL_DEFAULT_POS
        self._substep_ctr    = 0
        if self._fallback is not None:
            self._fallback.reset_phase(value)

    def substep(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        vx_cmd: float,
        vyaw_cmd: float,
        ramp_start_time: float,
        sim_time: float,
        sim_dt: float,
    ) -> None:
        if not self._loaded:
            assert self._fallback is not None
            self._fallback.substep(
                model, data, vx_cmd, vyaw_cmd, ramp_start_time, sim_time, sim_dt
            )
            return

        self._substep_ctr += 1
        joint_pos = data.qpos[JSTART:JSTART + 12].astype(np.float32)
        joint_vel = data.qvel[VSTART:VSTART + 12].astype(np.float32)

        # ── policy inference at 50 Hz (every _POLICY_DECIMATE physics steps) ──
        if self._substep_ctr % _POLICY_DECIMATE == 1:
            # Clamp commands to training distribution range
            vx   = float(np.clip(vx_cmd,   -_CMD_VX_MAX,   _CMD_VX_MAX))
            vyaw = float(np.clip(vyaw_cmd, -_CMD_VYAW_MAX, _CMD_VYAW_MAX))

            quat        = data.qpos[3:7]
            R           = _quat_to_rotmat(quat)     # body → world
            omega_world = np.array(data.qvel[3:6], dtype=np.float32)
            omega_body  = (R.T @ omega_world).astype(np.float32)
            g_body      = (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

            # Scale vx to push the normalised command closer to the ±5σ clip,
            # matching the effective saturation used for vyaw (≈4.7σ at cmd=2.5).
            # Without scaling, vx=0.8 normalises to only ~1.8σ → near-zero motion.
            obs = np.concatenate([
                omega_body,
                g_body,
                np.array([vx * _VX_OBS_SCALE, 0.0, vyaw * _VYAW_OBS_SCALE], dtype=np.float32),
                joint_pos - _RL_DEFAULT_POS,
                joint_vel,
                self._last_action,
            ])  # shape (45,), all raw (no manual scaling)

            # Running normalizer (z-score with training statistics)
            obs_norm = (obs - self._norm_mean) / self._norm_std
            obs_norm = np.clip(obs_norm, -5.0, 5.0).astype(np.float32)

            import torch
            with torch.no_grad():
                t_action = self._actor(
                    torch.from_numpy(obs_norm).unsqueeze(0)
                ).squeeze(0)
            action = t_action.numpy().astype(np.float32)
            self._last_action = action.copy()
            self._target_pos  = _RL_DEFAULT_POS + _current_action_scale() * action

        # ── PD control at full physics rate ──────────────────────────────────
        torque = _RL_KP * (self._target_pos - joint_pos) - _RL_KD * joint_vel
        data.ctrl[:] = np.clip(torque, self._lo, self._hi)


def default_low_level(model: mujoco.MjModel) -> LowLevelBase:
    """Return the best available low-level controller (RL policy, else trot)."""
    return RLVelocityLowLevel(model)
