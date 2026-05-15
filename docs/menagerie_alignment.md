# MuJoCo Menagerie alignment (`unitree_go2`)

Reference: [google-deepmind/mujoco_menagerie `unitree_go2`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2).

## Comparison summary

| Item | Menagerie | This project |
|------|-------------|--------------|
| Kinematics / meshes | `go2.xml` + `assets/*.obj` | Same structure; [assets/go2.xml](../assets/go2.xml) mirrors Menagerie body tree |
| Compiler | Menagerie uses `meshdir` on the robot file | **This repo:** [scene.xml](../scene.xml) sets `meshdir="assets/assets"` (paths from `sim/`) because MuJoCo resolves included meshes from the **main** XML directory |
| Scene wrapper | N/A | [scene.xml](../scene.xml) includes `go2.xml`; arena, ball, `top_cam` only |
| `head_cam` | Not in upstream | **Local addition** on `base` for vision demo (`fovy=90`) |
| Keyframe `home` | Varies by fork | **Local** standing pose + `ctrl` match in [assets/go2.xml](../assets/go2.xml) |
| Actuator block | FL, FR, RL, RR (hip, thigh, calf each) | **Same order** as Menagerie listing in XML |

## Required edits applied

1. **`meshdir` on the robot file** — Matches Menagerie so any external controller expecting standard paths can drop in the same XML layout.
2. **Parent scene** — Removed duplicate `meshdir` from [scene.xml](../scene.xml); mesh resolution is owned by the included `go2.xml`.

## Optional future sync

- Periodically `diff` [assets/go2.xml](../assets/go2.xml) against upstream `go2.xml` for physics/collision tweaks.
- If switching to Menagerie’s MJX variant, use their `go2_mjx.xml` pattern and test Apple Silicon separately.
