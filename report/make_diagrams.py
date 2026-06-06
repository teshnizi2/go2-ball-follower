#!/usr/bin/env python3
"""Generate the two schematic figures for the report:
  fig_architecture.png  - the perception -> planner -> gate -> frozen policy pipeline
  fig_gate.png          - top-down geometry of the free-band passage gate (the novelty)

These are illustrations of the actual design (not measured data). Run:
  ./.venv312/bin/python3.12 report/make_diagrams.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D
import os

HERE = os.path.dirname(os.path.abspath(__file__))

OURS   = "#cfe3f7"   # boxes we wrote
OURS_E = "#2f6fb0"
FROZEN = "#e2e2e6"   # pre-trained / library blocks
FROZEN_E = "#888"
CAM    = "#ffe6b3"
CAM_E  = "#cc9a3a"


def _box(ax, x, y, w, h, text, fc, ec, fs=8.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=1.2, edgecolor=ec, facecolor=fc, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=5)


def _arrow(ax, x0, y0, x1, y1, label=None, fs=7.0, color="#333", rad=0.0, lx=None, ly=None):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                        lw=1.3, color=color,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        ax.text(lx if lx is not None else (x0 + x1) / 2,
                ly if ly is not None else (y0 + y1) / 2 + 0.018,
                label, ha="center", va="center", fontsize=fs, color="#222")


def architecture():
    fig, ax = plt.subplots(figsize=(10.0, 2.55))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.55); ax.axis("off")

    y = 1.45; h = 0.62
    # box x, width
    boxes = [
        (0.10, 1.05, "Head RGB\ncamera\n(480x480)", CAM, CAM_E),
        (1.45, 1.30, "HSV + CamShift\nball tracker\n(tracker.py)", OURS, OURS_E),
        (3.10, 1.55, "Reactive planner:\nfree-band select +\ncommit-and-hold\n(controller.py)", OURS, OURS_E),
        (4.95, 1.45, "Hard passage gate\n+ obstacle\nrepulsion (main.py)", OURS, OURS_E),
        (6.70, 1.40, "Frozen PPO\nvelocity policy\n(low_level.py)", FROZEN, FROZEN_E),
        (8.35, 1.45, "PD @ 500 Hz\n-> MuJoCo\nGo2 robot", FROZEN, FROZEN_E),
    ]
    for (x, w, t, fc, ec) in boxes:
        _box(ax, x, y, w, h, t, fc, ec)

    # forward arrows with labels
    _arrow(ax, 1.15, y + h / 2, 1.45, y + h / 2)
    _arrow(ax, 2.75, y + h / 2, 3.10, y + h / 2, "ball\nbearing", lx=2.92, ly=y + h / 2 + 0.16)
    _arrow(ax, 4.65, y + h / 2, 4.95, y + h / 2, "aim\npoint", lx=4.80, ly=y + h / 2 + 0.16)
    _arrow(ax, 6.40, y + h / 2, 6.70, y + h / 2, "vx, vy, vw", lx=6.55, ly=y + h / 2 + 0.15)
    _arrow(ax, 8.10, y + h / 2, 8.35, y + h / 2, "12 joint\ntargets", lx=8.22, ly=y + h / 2 + 0.16)

    # feedback arrows (below)
    yb = y - 0.30
    _arrow(ax, 9.10, y, 9.10, yb, color="#999")
    _arrow(ax, 9.10, yb, 3.88, yb, color="#999", label="robot pose  (base x, y, yaw)", fs=7.0,
           lx=6.4, ly=yb - 0.12)
    _arrow(ax, 3.88, yb, 3.88, y, color="#999")
    yb2 = y - 0.62
    _arrow(ax, 9.30, y, 9.30, yb2, color="#bbb")
    _arrow(ax, 9.30, yb2, 7.40, yb2, color="#bbb", label="45-D proprioceptive observation", fs=7.0,
           lx=8.0, ly=yb2 - 0.12)
    _arrow(ax, 7.40, yb2, 7.40, y, color="#bbb")

    # legend
    ax.add_patch(Rectangle((0.12, 0.18), 0.22, 0.12, fc=OURS, ec=OURS_E, lw=1.0))
    ax.text(0.40, 0.24, "our code", fontsize=7.4, va="center")
    ax.add_patch(Rectangle((1.45, 0.18), 0.22, 0.12, fc=FROZEN, ec=FROZEN_E, lw=1.0))
    ax.text(1.73, 0.24, "frozen / pre-trained (never modified)", fontsize=7.4, va="center")
    ax.add_patch(Rectangle((4.60, 0.18), 0.22, 0.12, fc=CAM, ec=CAM_E, lw=1.0))
    ax.text(4.88, 0.24, "sensor", fontsize=7.4, va="center")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    out = os.path.join(HERE, "fig_architecture.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def gate():
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    # down-corridor = +x (to the right); lateral = y
    x0, x1 = -3.4, 3.0
    W = 2.5  # corridor half-width
    ax.set_xlim(x0, x1); ax.set_ylim(-W - 0.55, W + 0.55)

    # corridor walls
    ax.add_patch(Rectangle((x0, W), x1 - x0, 0.5, fc="#d8d8dc", ec="none"))
    ax.add_patch(Rectangle((x0, -W - 0.5), x1 - x0, 0.5, fc="#d8d8dc", ec="none"))
    ax.plot([x0, x1], [W, W], color="#888", lw=1.0)
    ax.plot([x0, x1], [-W, -W], color="#888", lw=1.0)
    ax.text(x1 - 0.05, W + 0.22, "wall", ha="right", fontsize=7.5, color="#666")

    # obstacle row at x in [-0.55, 0.55]
    ox_lo, ox_hi = -0.55, 0.55
    SAFE = 0.5  # safety inflation (robot half + reserve)
    # wide obstacle: y in [0.55, 2.45]
    wlo, whi = 0.55, 2.45
    # narrow obstacle: y in [-2.45, -1.45]
    nlo, nhi = -2.45, -1.45
    for (lo, hi, lab) in [(wlo, whi, "wide"), (nlo, nhi, "narrow")]:
        ax.add_patch(Rectangle((ox_lo, lo), ox_hi - ox_lo, hi - lo,
                               fc="#101014", ec="#000", lw=1.0, zorder=4))
        # safety halo (dashed inflation)
        ax.add_patch(Rectangle((ox_lo - SAFE, lo - SAFE), (ox_hi - ox_lo) + 2 * SAFE,
                               (hi - lo) + 2 * SAFE, fill=False, ec="#c0392b",
                               lw=1.0, ls=(0, (4, 3)), zorder=3))
    ax.text(0.0, (wlo + whi) / 2, "obstacle", color="white", ha="center", va="center",
            fontsize=7.0, rotation=90, zorder=5)

    # free band between inflated obstacles: y in [nhi+SAFE, wlo-SAFE] = [-0.95, 0.05]
    fb_lo, fb_hi = nhi + SAFE, wlo - SAFE
    ax.add_patch(Rectangle((ox_lo - SAFE, fb_lo), (ox_hi - ox_lo) + 2 * SAFE, fb_hi - fb_lo,
                           fc="#79c879", ec="none", alpha=0.45, zorder=2))
    bc = 0.5 * (fb_lo + fb_hi)

    # gate / entry plane (vertical dashed)
    x_gate = ox_lo - SAFE
    ax.plot([x_gate, x_gate], [-W, W], color="#2f6fb0", lw=1.3, ls=(0, (5, 3)), zorder=3)
    ax.text(x_gate, W + 0.22, "entry plane (gate)", ha="center", fontsize=7.3, color="#2f6fb0")

    # aim point at band centre
    ax.plot([0.0], [bc], marker="*", ms=13, color="#1d6b1d", zorder=7)
    ax.text(0.0, -1.18, "aim point\n(band centre)", ha="center", va="top",
            fontsize=6.8, color="#1d6b1d", zorder=7)

    # robot path: approach misaligned -> creep to gate -> strafe to bc -> pass
    def robot(ax, cx, cy, c="#2f6fb0"):
        ax.add_patch(FancyBboxPatch((cx - 0.22, cy - 0.16), 0.44, 0.32,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     fc=c, ec="#16456f", lw=1.0, alpha=0.95, zorder=8))
    y_app = 1.25
    robot(ax, -3.0, y_app)
    # path line
    ax.plot([-3.0, x_gate], [y_app, y_app], color="#2f6fb0", lw=1.6, zorder=6)
    ax.plot([x_gate, x_gate], [y_app, bc], color="#2f6fb0", lw=1.6, ls=(0, (2, 1.5)), zorder=6)
    ax.plot([x_gate, 2.6], [bc, bc], color="#2f6fb0", lw=1.6, zorder=6)
    robot(ax, x_gate, bc, c="#3a86d0")
    robot(ax, 2.3, bc, c="#5fa0e0")
    # annotations for the two phases
    _arrow(ax, -2.4, y_app + 0.1, -1.5, y_app + 0.1, "creep, vy chase", fs=6.8, color="#2f6fb0",
           lx=-1.95, ly=y_app + 0.30)
    ax.annotate("vx -> 0,\nstrafe (vy)", xy=(x_gate, 0.45 * (y_app + bc)),
                xytext=(x_gate - 1.15, 0.45 * (y_app + bc)), fontsize=6.8, color="#16456f",
                va="center", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#16456f", lw=1.1))
    _arrow(ax, 1.0, bc + 0.12, 2.0, bc + 0.12, "pass aligned", fs=6.8, color="#2f6fb0",
           lx=1.5, ly=bc + 0.32)

    handles = [Patch(fc="#79c879", alpha=0.45, label="free band (safe lane)"),
               Line2D([0], [0], color="#c0392b", ls=(0, (4, 3)), label="safety margin")]
    ax.legend(handles=handles, fontsize=6.6, loc="lower right", framealpha=0.92)

    ax.set_xlabel("down-corridor  (m)", fontsize=8)
    ax.set_ylabel("lateral  y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.12)
    out = os.path.join(HERE, "fig_gate.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    architecture()
    gate()
