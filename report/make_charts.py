#!/usr/bin/env python3
"""Generate the data figures for the report from real run logs + the actual
difficulty formulas in main.py.

  fig_difficulty.png  - obstacle widths / guaranteed passage / stage vs distance
                        (plotted directly from the deterministic formulas)
  fig_trajectory.png  - top-down robot vs ball path, lateral offset, speed
                        (from a real headless run CSV)
  fig_ablation.png    - distance travelled before first collision per config
                        (from real ablation runs; report/data/ablation.json)

Usage:
  ./.venv312/bin/python3.12 report/make_charts.py <full_run.csv>
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, json, os, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


# ---------- deterministic difficulty curve (matches main.py exactly) ----------
def difficulty01(x, base=0.55, ramp=240.0, cap=1.6):
    return max(0.0, min(cap, base + (x - 4.0) / ramp))

def fig_difficulty():
    xs = np.linspace(0, 260, 600)
    d = np.array([difficulty01(x) for x in xs])
    wide = np.minimum(0.95, 0.30 + 0.50 * d) * 2.0
    narrow = np.minimum(0.68, 0.22 + 0.30 * d) * 2.0
    passage = np.maximum(1.30, 1.90 - 0.60 * d)
    stage = np.array([1 + min(16, int(max(0.0, x - 4.0) / 16.0)) for x in xs])

    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.plot(xs, passage, color="#1d6b1d", lw=2.0, label="guaranteed passage (lane)")
    ax.plot(xs, wide, color="#101014", lw=1.6, label="wide obstacle width")
    ax.plot(xs, narrow, color="#8a8a8a", lw=1.6, label="narrow obstacle width")
    ax.axhline(0.68, color="#c0392b", lw=1.2, ls=(0, (4, 3)),
               label="robot width (0.68 m)")
    ax.set_xlabel("distance down corridor  (m)", fontsize=8)
    ax.set_ylabel("width  (m)", fontsize=8)
    ax.set_ylim(0, 2.2)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.6, loc="upper center", ncol=2, framealpha=0.9)

    ax2 = ax.twinx()
    ax2.step(xs, stage, where="post", color="#2f6fb0", lw=1.2, alpha=0.55)
    ax2.set_ylabel("difficulty stage", fontsize=8, color="#2f6fb0")
    ax2.tick_params(labelsize=7, colors="#2f6fb0")
    ax2.set_ylim(0, 18)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.97, bottom=0.16)
    out = os.path.join(HERE, "fig_difficulty.png")
    fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


# ---------- read a run CSV ----------
def read_csv(path):
    cols = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    cols.setdefault(k, []).append(float("nan"))
    return {k: np.array(v) for k, v in cols.items()}


def fig_trajectory(csv_path):
    c = read_csv(csv_path)
    t = c["sim_time"]; rx = c["robot_x"]; ry = c["robot_y"]
    bx = c["ball_x"]; by = c["ball_y"]
    # forward speed from position deltas
    dt = np.diff(t); dt[dt <= 0] = np.nan
    spd = np.diff(rx) / dt  # forward (down-corridor) speed: x-progress rate, not lateral
    # smooth speed (rolling mean, ~1 s window @50 Hz)
    win = 25
    k = np.ones(win) / win
    spd_s = np.convolve(np.nan_to_num(spd, nan=0.0), k, mode="same")
    # lateral offset robot vs ball (interp ball_y onto robot timeline already aligned by row)
    lat = np.abs(ry - by)

    fig = plt.figure(figsize=(10.2, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # (a) top-down path
    ax = axs[0]
    ax.axhline(2.5, color="#888", lw=1.0); ax.axhline(-2.5, color="#888", lw=1.0)
    ax.fill_between([rx.min() - 1, max(rx.max(), bx.max()) + 1], 2.5, 3.1, color="#e3e3e7")
    ax.fill_between([rx.min() - 1, max(rx.max(), bx.max()) + 1], -3.1, -2.5, color="#e3e3e7")
    ax.plot(bx, by, color="#c0392b", lw=1.4, label="ball (goal) path", alpha=0.85)
    ax.plot(rx, ry, color="#2f6fb0", lw=1.6, label="robot path")
    ax.set_ylim(-3.1, 3.1)
    ax.set_xlabel("down-corridor  x (m)", fontsize=8)
    ax.set_ylabel("lateral y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.8, loc="upper left", ncol=2, framealpha=0.9)
    ax.set_title("(a) top-down path (robot vs ball)", fontsize=8.2)

    # (b) lateral offset
    ax = axs[1]
    ax.plot(rx, lat, color="#6a3d9a", lw=1.0)
    m = np.nanmean(lat)
    ax.axhline(m, color="#333", lw=1.0, ls=(0, (4, 3)), label=f"mean = {m:.2f} m")
    ax.set_xlabel("down-corridor  x (m)", fontsize=8)
    ax.set_ylabel("|y_robot - y_ball|\n(m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.8, loc="upper right", framealpha=0.9)
    ax.set_title("(b) lateral offset from ball", fontsize=8.2)

    # (c) speed
    ax = axs[2]
    ax.plot(rx[1:], spd_s, color="#1d6b1d", lw=1.0)
    ms = float(np.nanmean(spd))  # net forward speed over the whole run (incl. gate holds)
    ax.axhline(ms, color="#333", lw=1.0, ls=(0, (4, 3)), label=f"mean = {ms:.2f} m/s")
    ax.set_ylim(0, max(1.0, np.nanpercentile(spd_s, 99)))
    ax.set_xlabel("down-corridor  x (m)", fontsize=8)
    ax.set_ylabel("forward speed\n(m/s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.8, loc="upper right", framealpha=0.9)
    ax.set_title("(c) forward speed (gate holds = dips)", fontsize=8.2)

    fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.17, wspace=0.30)
    out = os.path.join(HERE, "fig_trajectory.png")
    fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)
    return dict(mean_lat=float(np.nanmean(lat)),
                max_lat=float(np.nanmax(lat)),
                mean_speed=float(ms),
                dist=float(rx.max() - rx.min()),
                dur=float(t.max()))


def fig_ablation():
    p = os.path.join(DATA, "ablation.json")
    if not os.path.exists(p):
        print("no ablation.json, skipping ablation chart"); return None
    runs = json.load(open(p))
    order = ["full", "no_lateral", "no_avoid"]
    labels = {"full": "Full\n(all on)",
              "no_lateral": "No lateral\n+ gate",
              "no_avoid": "No reactive\navoidance"}
    agg = {}
    for cfg in order:
        rs = [r for r in runs if r["config"] == cfg]
        if not rs:
            continue
        dists = [(r["first_x"] if r["collided"] else r["final_x"]) for r in rs]
        agg[cfg] = dict(mean=float(np.mean(dists)), dists=dists,
                        n_coll=sum(1 for r in rs if r["collided"]),
                        n=len(rs))
    cfgs = [c for c in order if c in agg]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    xpos = np.arange(len(cfgs))
    colors = {"full": "#1d6b1d", "no_lateral": "#c0392b", "no_avoid": "#e08e0b"}
    means = [agg[c]["mean"] for c in cfgs]
    ax.bar(xpos, means, width=0.6, color=[colors[c] for c in cfgs], alpha=0.85,
           edgecolor="#333", lw=0.8)
    # per-seed dots
    for i, c in enumerate(cfgs):
        ds = agg[c]["dists"]
        ax.scatter([i] * len(ds), ds, color="#222", s=14, zorder=5)
    for i, c in enumerate(cfgs):
        if agg[c]["n_coll"] == 0:
            ax.text(i, means[i] + 1.5, "0 collisions\n(full run)", ha="center",
                    fontsize=6.8, color="#1d6b1d")
        else:
            ax.text(i, means[i] + 1.5, f"collides\n({agg[c]['n_coll']}/{agg[c]['n']} seeds)",
                    ha="center", fontsize=6.8, color="#7a2018")
    ax.set_xticks(xpos); ax.set_xticklabels([labels[c] for c in cfgs], fontsize=7.4)
    ax.set_ylabel("distance before first\ncollision  (m)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, max(means) * 1.25 + 2)
    ax.set_title("Ablation at maximum difficulty (1.30 m lanes)", fontsize=8.2)
    fig.subplots_adjust(left=0.155, right=0.97, top=0.88, bottom=0.13)
    out = os.path.join(HERE, "fig_ablation.png")
    fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)
    return agg


if __name__ == "__main__":
    fig_difficulty()
    stats = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        stats = fig_trajectory(sys.argv[1])
        print("TRAJ_STATS", json.dumps(stats))
    ab = fig_ablation()
    if ab:
        print("ABLATION", json.dumps(ab))
