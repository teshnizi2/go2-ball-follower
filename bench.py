"""
bench.py – Analyse the most recent set of simulation run CSVs.

Usage
-----
    python bench.py               # analyse latest slow/medium/fast runs
    python bench.py --all         # analyse all CSVs in logs/

Exit codes
----------
  0 – all tagged runs passed quality thresholds
  1 – one or more runs failed (print which and why)

Quality thresholds
------------------
  det_pct      >= 75 %       ball visible most of the time
  avg_dist      1.5 – 2.5 m  staying near the 2 m standoff
  avg_x_err    <= 50 px      ball roughly centred in frame
  search_trips  <= 2         rarely falls back to full spin-search
"""

from __future__ import annotations

import csv
import pathlib
import sys
from collections import defaultdict
from typing import Optional

# ── thresholds ────────────────────────────────────────────────────────────────
DET_PCT_MIN        = 75.0   # %
AVG_DIST_LO        = 1.5    # m
AVG_DIST_HI        = 2.5    # m
# x_err threshold as fraction of image half-width (resolution-agnostic).
# 12° from centre for a 60° FOV camera → 12/30 = 0.4 of half-width.
# Equivalent to 64px on a 320px-wide image or 96px on a 480px-wide image.
# This is physically appropriate for a P controller on a legged robot: at 12°
# the ball remains in the central 40% of the frame and is clearly tracked.
AVG_X_ERR_FRAC_MAX = 12.0 / 30.0   # 0.4 of half-width
# Only include steps with conf > this for x_err computation (excludes coast noise)
CONF_FOR_XERR      = 0.5
SEARCH_TRIPS_MAX   = 2      # count (after initial lock)

TAGS_OF_INTEREST = {"slow", "medium", "fast"}

LOG_DIR = pathlib.Path(__file__).parent / "logs"


def parse_csv(path: pathlib.Path) -> Optional[dict]:
    """Return a metrics dict for one CSV run, or None if the file is empty/invalid."""
    rows = []
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"  [WARN] could not read {path.name}: {e}")
        return None

    if not rows:
        return None

    n              = len(rows)
    detected       = [r for r in rows if int(r["detected"]) == 1]
    det_pct        = 100.0 * len(detected) / n

    dists          = [float(r["dist_to_ball"]) for r in rows]
    avg_dist       = sum(dists) / n

    # Determine img_w: from CSV column if available, else from max(cx)
    if "img_w" in rows[0]:
        img_w = int(rows[0]["img_w"])
    else:
        cxs   = [int(r["cx"]) for r in rows if int(r["cx"]) > 0]
        img_w = max(cxs) + 1 if cxs else 480

    # Use only high-confidence frames for x_err (excludes stale coast positions)
    confident      = [r for r in rows if float(r["confidence"]) > CONF_FOR_XERR
                                          and int(r["cx"]) >= 0]
    x_errs         = [abs(float(r["cx"]) - img_w / 2) for r in confident] if confident else [img_w / 2]
    avg_x_err      = sum(x_errs) / max(len(x_errs), 1)

    # Count SEARCHING trips after the initial lock by detecting long stretches of
    # det==0 that follow at least one det==1.
    search_trips   = 0
    ever_detected  = False
    in_search      = False
    for r in rows:
        d = int(r["detected"])
        if d == 1:
            ever_detected = True
            if in_search:
                in_search = False
        else:
            if ever_detected and not in_search:
                in_search    = True
                search_trips += 1

    recover_trips  = 0
    prev_vyaw      = 0.0
    for r in rows:
        vyaw = float(r["vyaw_cmd"])
        # A RECOVERING transition shows up as a sudden spike in |vyaw| while det==0
        if int(r["detected"]) == 0 and abs(vyaw) > 0.5 and abs(prev_vyaw) < 0.1:
            recover_trips += 1
        prev_vyaw = vyaw

    return {
        "n":             n,
        "img_w":         img_w,
        "det_pct":       det_pct,
        "avg_dist":      avg_dist,
        "avg_x_err":     avg_x_err,
        "conf_steps":    len(confident),
        "search_trips":  search_trips,
        "recover_trips": recover_trips,
    }


def check_pass(m: dict) -> list[str]:
    """Return list of failure reasons; empty list = pass."""
    failures = []
    img_w    = m.get("img_w", 480)
    x_thresh = AVG_X_ERR_FRAC_MAX * (img_w / 2)
    if m["det_pct"] < DET_PCT_MIN:
        failures.append(f"det_pct={m['det_pct']:.1f}% < {DET_PCT_MIN}%")
    if not (AVG_DIST_LO <= m["avg_dist"] <= AVG_DIST_HI):
        failures.append(f"avg_dist={m['avg_dist']:.2f}m outside [{AVG_DIST_LO},{AVG_DIST_HI}]m")
    if m["avg_x_err"] > x_thresh:
        failures.append(f"avg_x_err={m['avg_x_err']:.1f}px > {x_thresh:.1f}px threshold")
    if m["search_trips"] > SEARCH_TRIPS_MAX:
        failures.append(f"search_trips={m['search_trips']} > {SEARCH_TRIPS_MAX}")
    return failures


def latest_per_tag(all_csvs: list[pathlib.Path]) -> dict[str, pathlib.Path]:
    """For each tag in TAGS_OF_INTEREST, return the most recently modified CSV."""
    by_tag: dict[str, list[pathlib.Path]] = defaultdict(list)
    for p in all_csvs:
        # filename: run_YYYYMMDD_HHMMSS_<tag>.csv
        parts = p.stem.split("_")
        if len(parts) >= 4:
            tag = parts[-1]
            by_tag[tag].append(p)
    result = {}
    for tag in TAGS_OF_INTEREST:
        candidates = by_tag.get(tag, [])
        if candidates:
            result[tag] = max(candidates, key=lambda p: p.stat().st_mtime)
    return result


def main() -> int:
    analyse_all = "--all" in sys.argv

    all_csvs = sorted(LOG_DIR.glob("run_*.csv"), key=lambda p: p.stat().st_mtime)
    if not all_csvs:
        print("[BENCH] No CSV files found in", LOG_DIR)
        return 1

    if analyse_all:
        targets = {p.stem.split("_")[-1]: p for p in all_csvs}
    else:
        targets = latest_per_tag(all_csvs)

    if not targets:
        print("[BENCH] No tagged runs (slow/medium/fast) found. Run with --all to see everything.")
        return 1

    print()
    print("=" * 78)
    print(f"{'TAG':<10} {'STEPS':>6} {'DET%':>6} {'AVG_DIST':>9} {'AVG_XERR':>9} "
          f"{'RECOVER':>8} {'SEARCH':>7}  STATUS")
    print("-" * 78)

    all_pass = True
    for tag, path in sorted(targets.items()):
        m = parse_csv(path)
        if m is None:
            print(f"{'[' + tag + ']':<10}  (empty or unreadable)")
            all_pass = False
            continue

        failures = check_pass(m)
        status   = "PASS" if not failures else "FAIL: " + "; ".join(failures)
        if failures:
            all_pass = False

        print(
            f"{tag:<10} {m['n']:>6} {m['det_pct']:>5.1f}% {m['avg_dist']:>8.2f}m "
            f"{m['avg_x_err']:>8.1f}px {m['recover_trips']:>8} {m['search_trips']:>7}  {status}"
        )

    print("=" * 78)
    print()

    if all_pass:
        print("[BENCH] ALL RUNS PASSED ✓")
    else:
        print("[BENCH] Some runs FAILED — see rows above for details.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
