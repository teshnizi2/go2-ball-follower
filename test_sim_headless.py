"""
Headless validation harness for the Go2 ball-follower controller.

No rendering (no OpenGL in this sandbox) — instead we project the ball into
the head camera analytically using MuJoCo's own cam_xpos / cam_xmat, and feed
the synthetic (cx, cy, area) into the real `controller.FollowController` while
the real `low_level` trot gait drives real MuJoCo physics.

This tests exactly what the user is complaining about:
  - does the robot fall?
  - does it "freeze" (no forward motion while ball visible)?
  - does it re-acquire when the ball goes out of view?
  - how close does it get to the 1.0 m standoff goal?

Runs for N seconds of sim time, then prints a summary.
"""
from __future__ import annotations
import os, math, sys, pathlib, collections
os.environ.setdefault("MUJOCO_GL", "disable")  # never try to init OpenGL
sys.path.insert(0, "/sessions/nice-quirky-brahmagupta/mnt/Project/robotics_project/sim")

import numpy as np
import mujoco

from controller import FollowController, apply_stability_cap
from low_level import default_low_level, quat_to_pitch_roll
# DemoTargetMover has no rendering dep, safe to import
from main import DemoTargetMover, _pose_incapped, _tilt_guard_scale, FALL_RESET_S

SIM_DT  = 0.002
CTRL_HZ = 50
CTRL_SKIP = max(1, round(1.0 / (CTRL_HZ * SIM_DT)))
CTRL_DT = CTRL_SKIP * SIM_DT
CAM_W, CAM_H = 480, 360
FOVY_DEG = 115.0
FOCAL_Y = (CAM_H * 0.5) / math.tan(math.radians(FOVY_DEG) * 0.5)
FOCAL_X = FOCAL_Y  # square pixels
BALL_RADIUS_M = 0.12  # approx — whatever the mocap sphere actually is; used only for area

SCENE_XML = "/sessions/nice-quirky-brahmagupta/mnt/Project/robotics_project/sim/scene.xml"


def project_ball(cam_pos, cam_mat, ball_pos):
    """
    Return (cx, cy, area, visible) for the ball in the head camera.

    MuJoCo camera convention (classical OpenGL): camera looks down -Z,
    +X right, +Y up. cam_mat is 3x3 rotation (world_R_cam).
    """
    rel = ball_pos - cam_pos
    # world -> camera-local
    local = cam_mat.T @ rel  # (x_right, y_up, -z_forward)
    x_c, y_c, z_c = float(local[0]), float(local[1]), float(local[2])
    # camera looks down -Z_cam → visible if z_c < 0 (point is in front)
    if z_c >= -0.05:
        return None, None, 0.0, False
    depth = -z_c  # positive distance forward
    u = FOCAL_X * (x_c / depth)
    v = FOCAL_Y * (y_c / depth)
    cx_px = CAM_W * 0.5 + u
    cy_px = CAM_H * 0.5 - v  # pixel y grows downward
    # FOV check
    half_fov_h = math.atan2(CAM_W * 0.5, FOCAL_X)
    half_fov_v = math.atan2(CAM_H * 0.5, FOCAL_Y)
    bearing_h = math.atan2(x_c, depth)
    bearing_v = math.atan2(y_c, depth)
    if abs(bearing_h) > half_fov_h or abs(bearing_v) > half_fov_v:
        return None, None, 0.0, False
    # Apparent area of a sphere (π r² in image) using pinhole
    r_px = FOCAL_X * BALL_RADIUS_M / depth
    area = math.pi * r_px * r_px
    return int(round(cx_px)), int(round(cy_px)), float(area), True


def run(seconds: float = 60.0, mode: str = "square", verbose: bool = False):
    os.environ["BALL_PATH_MODE"] = mode
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    ctrl = FollowController()
    mover = DemoTargetMover()
    gait = default_low_level(model)
    ramp_start = [data.time]

    vx_cmd = vyaw_cmd = 0.0
    VX_SMOOTH, VYAW_SMOOTH = 0.60, 0.80  # match main.py

    n_steps = int(seconds / CTRL_DT)
    falls = 0
    fell_at = None
    visible_steps = 0
    dist_samples = []
    vx_samples = []
    vyaw_samples = []
    state_hist = collections.Counter()
    confirmed_lost_events = 0
    was_incapped = False
    ball_visible_prev = True

    for step in range(n_steps):
        # read state
        rz_pre = float(data.qpos[2])
        quat_pre = np.asarray(data.qpos[3:7], dtype=float)
        incapped = _pose_incapped(rz_pre, quat_pre)
        tilt_scale = _tilt_guard_scale(quat_pre)
        vx_g = 0.0 if incapped else (vx_cmd * tilt_scale)
        vyaw_g = vyaw_cmd * max(0.55, tilt_scale)  # match main.py

        # physics
        for _ in range(CTRL_SKIP):
            gait.substep(model, data, vx_g, vyaw_g, ramp_start[0], float(data.time), SIM_DT)
            mujoco.mj_step(model, data)

        rx, ry, rz = float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
        quat = np.asarray(data.qpos[3:7], dtype=float)

        # move ball
        mover.step(CTRL_DT, data.mocap_pos, ball_visible=ball_visible_prev)
        bx, by = float(data.mocap_pos[0, 0]), float(data.mocap_pos[0, 1])
        bz = float(data.mocap_pos[0, 2])

        # fake vision from ground truth
        mujoco.mj_forward(model, data)
        cam_pos = np.asarray(data.cam_xpos[head_id], dtype=float)
        cam_mat = np.asarray(data.cam_xmat[head_id], dtype=float).reshape(3, 3)
        cx, cy, area, visible = project_ball(cam_pos, cam_mat, np.array([bx, by, bz]))
        conf = 0.9 if visible else 0.0
        ball_visible_prev = visible

        if visible:
            visible_steps += 1

        # controller
        dist_world = math.hypot(rx - bx, ry - by)
        vx_raw, vyaw_raw = ctrl.compute(
            cx, cy, area, conf, CAM_W, CAM_H, CTRL_DT, world_dist_m=dist_world
        )
        # EMA
        if step == 0:
            vx_cmd, vyaw_cmd = vx_raw, vyaw_raw
        else:
            vx_cmd = VX_SMOOTH * vx_raw + (1 - VX_SMOOTH) * vx_cmd
            vyaw_cmd = VYAW_SMOOTH * vyaw_raw + (1 - VYAW_SMOOTH) * vyaw_cmd
        # Re-apply the stability cap AFTER smoothing: EMA can mix a previous-step
        # high-vx-low-vyaw sample with a new-step low-vx-high-vyaw sample and
        # land inside the (vx≥0.35, vyaw≥0.34) unstable corner of the RL gait.
        vx_cmd, vyaw_cmd = apply_stability_cap(vx_cmd, vyaw_cmd)

        fell = _pose_incapped(rz, quat)
        if fell and fell_at is None:
            fell_at = data.time
            falls += 1
        elif fell and (data.time - fell_at >= FALL_RESET_S):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            gait.reset_phase(0.0)
            ramp_start[0] = data.time
            ctrl.reset()
            mover.reset()
            fell_at = None
        elif not fell:
            fell_at = None

        if fell:
            vx_cmd = min(vx_cmd, 0.15)

        dist_samples.append(dist_world)
        vx_samples.append(vx_cmd)
        vyaw_samples.append(vyaw_cmd)
        state_hist[ctrl._state] += 1

        if verbose and step % 100 == 0:
            pitch, roll = quat_to_pitch_roll(quat)
            # bearing from robot body to ball in world frame (atan2 of ball-robot delta)
            bearing_world = math.degrees(math.atan2(by - ry, bx - rx))
            # robot heading (yaw) from quaternion
            w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            robot_yaw = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
            rel_bearing = bearing_world - robot_yaw
            while rel_bearing > 180: rel_bearing -= 360
            while rel_bearing < -180: rel_bearing += 360
            print(f"t={data.time:5.1f}  robot=({rx:+.1f},{ry:+.1f})  yaw={robot_yaw:+5.0f}  "
                  f"pitch={math.degrees(pitch):+4.0f}  roll={math.degrees(roll):+4.0f}  "
                  f"ball=({bx:+.1f},{by:+.1f})  bearing={bearing_world:+5.0f}  rel={rel_bearing:+5.0f}  "
                  f"dist={dist_world:.2f}  vis={visible}  state={ctrl._state}")

    arr = np.array(dist_samples)
    vx_arr = np.abs(np.array(vx_samples))
    vyaw_arr = np.abs(np.array(vyaw_samples))
    print("="*72)
    print(f"Scenario: {mode}  |  sim time: {seconds:.0f}s  |  control steps: {n_steps}")
    print(f"Falls:                    {falls}  (auto-reset at {FALL_RESET_S}s each)")
    print(f"Ball visible fraction:    {visible_steps / n_steps:.1%}")
    print(f"Controller states:        {dict(state_hist)}")
    print(f"Distance to ball:         mean={arr.mean():.2f}m  median={np.median(arr):.2f}m  "
          f"p5={np.percentile(arr,5):.2f}  p95={np.percentile(arr,95):.2f}")
    print(f"Time near 1m goal (<=1.3): {(arr<=1.3).mean():.1%}")
    print(f"Time |vx_cmd|>0.05:       {(vx_arr>0.05).mean():.1%}  "
          f"mean={vx_arr.mean():.3f}  max={vx_arr.max():.3f}")
    print(f"Time |vyaw_cmd|>0.05:     {(vyaw_arr>0.05).mean():.1%}  "
          f"mean={vyaw_arr.mean():.3f}  max={vyaw_arr.max():.3f}")
    print("="*72)
    return dict(falls=falls, visible=visible_steps/n_steps,
                mean_dist=float(arr.mean()), states=dict(state_hist))


if __name__ == "__main__":
    dur = float(os.environ.get("TEST_SECONDS", "60"))
    run(seconds=dur, mode=os.environ.get("BALL_PATH_MODE", "square"),
        verbose=bool(int(os.environ.get("VERBOSE", "0"))))
