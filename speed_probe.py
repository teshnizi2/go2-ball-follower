"""Probe actual robot forward velocity at various vx commands under the RL policy."""
import os
import sys
import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
os.chdir(SIM_DIR)

import mujoco
from low_level import RLVelocityLowLevel

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
sim_dt = float(model.opt.timestep)

def body_vx(data):
    w, x, y, z = data.qpos[3:7]
    body_x = np.array([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)])
    return float(np.dot(data.qvel[0:3], body_x))

def reset_home():
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

ll = RLVelocityLowLevel(model)

print(f"physics dt={sim_dt:.4f}s  decimation 10 → policy at {1/(sim_dt*10):.1f} Hz", flush=True)

for vx_cmd in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
    reset_home()
    ll.reset_phase()
    # 1.5 s warmup with zero cmd
    t0 = data.time
    for i in range(int(1.5 / sim_dt)):
        ll.substep(model, data, 0.0, 0.0, t0, data.time, sim_dt)
        mujoco.mj_step(model, data)
    # 4 s at commanded vx
    t_cmd_start = data.time
    start_pos = np.array(data.qpos[0:2])
    samples = []
    fell = False
    for i in range(int(4.0 / sim_dt)):
        ll.substep(model, data, vx_cmd, 0.0, t_cmd_start, data.time, sim_dt)
        mujoco.mj_step(model, data)
        if i > int(0.5 / sim_dt):
            samples.append(body_vx(data))
        if data.qpos[2] < 0.10:
            fell = True
            break
    samples = np.asarray(samples) if samples else np.array([0.0])
    disp = np.linalg.norm(np.array(data.qpos[0:2]) - start_pos)
    elapsed = data.time - t_cmd_start
    tag = " FELL" if fell else ""
    print(f"vx_cmd={vx_cmd:.2f}  mean_bodyvx={samples.mean():+.3f}  "
          f"p90={np.percentile(samples,90):+.3f}  "
          f"disp/s={disp/max(elapsed,1e-6):.3f} m/s  z={data.qpos[2]:.3f}m{tag}",
          flush=True)
