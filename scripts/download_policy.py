"""
download_policy.py  –  Download the Go2 flat-terrain velocity policy from HuggingFace.

Usage (run once from the sim/ directory):
    .venv312/bin/python3.12 scripts/download_policy.py

The checkpoint is saved to:
    sim/policy/model_500.pt   ← RSL-RL checkpoint (actor + normalizer weights)

Source: diasAiMaster/unitree-go2-velocity-flat on HuggingFace Hub
  – 45-dim obs (ang_vel, gravity, cmd, joint_pos, joint_vel, last_action)
  – 12-dim actions (joint position offsets, action_scale=0.25)
  – PD gains: KP=20, KD=1.0  (stiffness/damping from training env.yaml)
  – Trained with RSL-RL / mjlab at 50 Hz (decimation=4 at 200 Hz physics)
  – Joint order: FL, FR, RL, RR  ×  (hip, thigh, calf)  [same as Menagerie go2.xml]
"""

from __future__ import annotations

import pathlib
import sys

DEST = pathlib.Path(__file__).resolve().parent.parent / "policy" / "model_500.pt"
HF_REPO = "diasAiMaster/unitree-go2-velocity-flat"
HF_FILE = "model_500.pt"


def download() -> pathlib.Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.  Run:  pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {HF_REPO}/{HF_FILE} …")
    cached = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)

    # Copy from HF cache to our policy/ directory so it works offline afterwards
    import shutil
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, DEST)
    print(f"Saved to: {DEST}")
    return DEST


def verify(path: pathlib.Path) -> None:
    try:
        import torch
    except ImportError:
        print("WARNING: torch not installed — skipping verification.")
        return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    actor_in  = sd["actor.0.weight"].shape[1]
    actor_out = sd["actor.6.weight"].shape[0]
    print(f"  actor shape : {actor_in} → ... → {actor_out}  ✓")
    print(f"  trained iter: {ckpt.get('iter', '?')}")
    mean = sd["actor_obs_normalizer._mean"]
    print(f"  normalizer  : mean.shape={tuple(mean.shape)}  ✓")


if __name__ == "__main__":
    if DEST.exists():
        print(f"Policy already present at {DEST}")
        verify(DEST)
    else:
        p = download()
        verify(p)
    print("Done.")
