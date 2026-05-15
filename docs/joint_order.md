# Actuator / joint order (project vs LocoMuJoCo)

## Project / MuJoCo Menagerie XML

Actuators in [assets/go2.xml](../assets/go2.xml) are listed as:

`FL_hip`, `FL_thigh`, `FL_calf`, `FR_*`, `RL_*`, `RR_*`

→ **indices 0–11** in `data.ctrl` and in [joint_order.py](../joint_order.py) as **PROJECT** order.

## LocoMuJoCo `UnitreeGo2`

Default observation stacks joint positions as **FR, FL, RR, RL** (each leg 3 joints). See [LocoMuJoCo UnitreeGo2 docs](https://loco-mujoco.readthedocs.io/en/latest/source/quadrupeds/unitreego2.html).

## Remapping

Use [joint_order.py](../joint_order.py):

- `project_to_locomujoco_ctrl(x)` — before comparing to a LocoMuJoCo policy output.
- `locomujoco_to_project_ctrl(x)` — before writing torques to `data.ctrl`.
- `denormalize_torques(u_norm, lo, hi)` — map **[-1, 1]¹²** to Newton-metres using `model.actuator_ctrlrange`.
