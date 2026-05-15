"""Check stability of action_scale in {0.35, 0.40, 0.45} under vx+vyaw over longer runs."""
import os, sys, numpy as np
SIM = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SIM); os.chdir(SIM)
import mujoco, low_level

def body_vx(d):
    w,x,y,z = d.qpos[3:7]
    bx = np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
    return float(np.dot(d.qvel[0:3], bx))

model = mujoco.MjModel.from_xml_path("scene.xml")
dt = float(model.opt.timestep)
low_level._CMD_VX_MAX = 3.0
low_level._VX_OBS_SCALE = 2.5

cases = [
    # (action_scale, vx, vyaw, duration)
    (0.35, 0.80, 0.00, 20.0),
    (0.40, 0.80, 0.00, 20.0),
    (0.45, 0.80, 0.00, 20.0),
    (0.35, 0.80, 0.60, 20.0),
    (0.40, 0.80, 0.60, 20.0),
    (0.35, 0.80, 0.90, 20.0),
    (0.40, 0.80, 0.90, 20.0),
    (0.35, 0.50, 0.90, 20.0),
    (0.40, 0.50, 0.90, 20.0),
    (0.40, 1.20, 0.00, 20.0),
    (0.40, 1.20, 0.60, 20.0),
]

print("act  vx  vyaw  dur   mean_bodyvx  p90  final_z  t_fall  disp/s", flush=True)
for act, vx, vyaw, dur in cases:
    low_level._ACTION_SCALE = act
    ll = low_level.RLVelocityLowLevel(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ll.reset_phase()
    # warmup
    for _ in range(int(1.0/dt)):
        ll.substep(model, data, 0.0, 0.0, data.time, data.time, dt)
        mujoco.mj_step(model, data)
    start = np.array(data.qpos[0:2])
    t0 = data.time
    samples = []; t_fall = None
    for i in range(int(dur/dt)):
        ll.substep(model, data, vx, vyaw, t0, data.time, dt)
        mujoco.mj_step(model, data)
        if i > int(0.5/dt):
            samples.append(body_vx(data))
        if data.qpos[2] < 0.10 and t_fall is None:
            t_fall = data.time - t0
            break
    s = np.asarray(samples) if samples else np.array([0.0])
    disp = np.linalg.norm(np.array(data.qpos[0:2]) - start)
    el = (t_fall if t_fall else (data.time - t0))
    tf = f"{t_fall:.1f}s" if t_fall else "-"
    print(f"{act:.2f} {vx:.2f} {vyaw:+.2f} {dur:.0f}s  {s.mean():+.3f}  {np.percentile(s,90):+.3f}  {data.qpos[2]:.3f}  {tf}  {disp/max(el,1e-6):.3f}", flush=True)
