#!/usr/bin/env python3
"""
Spike: external low-level policies (HF SAC, LocoMuJoCo) vs this repo.

Run from `sim/`:
    .venv312/bin/python3.12 scripts/spike_external_policy.py

This script does not modify the simulation — it reports what would be needed
for full integration and optionally probes installed packages.

References
----------
* https://huggingface.co/cagataydev/sac-unitree-go2-mujoco  (SAC checkpoints + SB3)
* https://loco-mujoco.readthedocs.io/                       (UnitreeGo2 Gym env)
* ../joint_order.py                                        (actuator order remap)
"""

from __future__ import annotations

import sys


def _try_hf_sac() -> None:
    print("\n--- Hugging Face SAC (cagataydev/sac-unitree-go2-mujoco) ---")
    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        print("stable_baselines3 not installed. For a real load:")
        print("  pip install stable-baselines3 huggingface_hub")
        print("Policy zip lives under repo checkpoints/; training code must match")
        print("observation/action definitions (likely Gymnasium + custom Go2 env).")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed.")
        return
    print("Packages present. Full integration still requires:")
    print("  1) The exact Gymnasium environment class used at training time.")
    print("  2) Matching MuJoCo model (Menagerie Go2) and timestep.")
    print("  3) Mapping SB3 actions → data.ctrl (use joint_order.denormalize_torques).")
    try:
        p = hf_hub_download(
            repo_id="cagataydev/sac-unitree-go2-mujoco",
            filename="checkpoints/sac_go2_final.zip",
            local_files_only=False,
        )
        print(f"Downloaded / cached checkpoint path: {p}")
        from stable_baselines3 import SAC
        SAC.load(p, print_system_info=False)
        print("SAC.load() succeeded (policy weights readable).")
    except Exception as e:
        print(f"Optional HF download/load skipped or failed: {e}")


def _try_locomujoco() -> None:
    print("\n--- LocoMuJoCo UnitreeGo2 ---")
    try:
        import loco_mujoco  # noqa: F401
    except ImportError:
        print("loco_mujoco not installed. Install with: pip install loco-mujoco")
        print("Then: from loco_mujoco.environments.quadrupeds.unitreeGo2 import UnitreeGo2")
        print("Observations use joint order FR,FL,RR,RL — see docs/joint_order.md")
        return
    print("loco_mujoco import OK. Instantiate UnitreeGo2 in a separate script to")
    print("compare observations with this project's scene.xml + vision stack.")


def main() -> int:
    print("External policy spike (read-only assessment)\n")
    _try_hf_sac()
    _try_locomujoco()
    print("\nRecommendation: keep default_low_level() TrotPDGait in main; add a")
    print("RawTorqueLowLevel branch once a matching SB3 env reproduces training obs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
