"""Sweep _ACTION_SCALE values and measure peak forward speed."""
import os
import sys
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
os.chdir(SIM_DIR)

import mujoco
import low_level

def body_vx(data):
    w, x, y, z = data.qpos[3:7]
    bx = np.array([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)])
    return float(np.dot(data.qvel[0:3], bx))

def probe(model, sim_dt, action_scale, vx_cmd):
    # Monkey-patch the module constant and rebuild controller
    low_level._ACTION_SCALE = action_scale
    ll = low_level.RLVelocityLowLevel(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ll.reset_phase()
    # warmup 1 s
    for _ in range(int(1.0 / sim_dt)):
        ll.substep(model, data, 0.0, 0.0, data.time, data.time, sim_dt)
        mujoco.mj_step(model, data)
    start = np.array(data.qpos[0:2])
    t0 = data.time
    samples = []
    fell = False
    for i in range(int(4.0 / sim_dt)):
        ll.substep(model, data, vx_cmd, 0.0, t0, data.time, sim_dt)
        mujoco.mj_step(model, data)
        if i > int(0.5 / sim_dt):
            samples.append(body_vx(data))
        if data.qpos[2] < 0.10:
            fell = True
            break
    samples = np.asarray(samples) if samples else np.array([0.0])
    disp = np.linalg.norm(np.array(data.qpos[0:2]) - start)
    elapsed = data.time - t0
    return samples.mean(), disp/max(elapsed,1e-6), data.qpos[2], fell

model = mujoco.MjModel.from_xml_path("scene.xml")
sim_dt = float(model.opt.timestep)

# Also lift the command clip so the scaled obs can saturate
low_level._CMD_VX_MAX = 3.0
low_level._VX_OBS_SCALE = 2.5

print(f"dt={sim_dt:.4f}  decimation={low_level._POLICY_DECIMATE}  policy_hz={1/(sim_dt*low_level._POLICY_DECIMATE):.1f}", flush=True)
print("action_scale  vx_cmd  meanV   disp/s   z   flag", flush=True)
for act in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    for vx in [0.8, 1.2, 1.5, 2.0]:
        mv, dps, z, fell = probe(model, sim_dt, act, vx)
        tag = "FELL" if fell else "ok"
        print(f"  {act:.2f}       {vx:.2f}   {mv:+.3f}  {dps:+.3f}  {z:.3f}  {tag}", flush=True)
