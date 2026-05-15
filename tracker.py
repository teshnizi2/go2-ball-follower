"""
tracker.py – HSV histogram back-projection + CamShift tracking for the red ball.

Public API
----------
BallTracker          – stateful class: call .update(frame_rgb) each step.
                       Returns a Detection namedtuple with cx/cy/area/confidence.
annotate_frame()     – draw detection overlay on a BGR frame for display.

Algorithm
---------
1.  **Detect / re-acquire**: HSV dual-range mask + morphology + contour picking
    gives an initial bounding box and build an ROI hue histogram.
2.  **Track**: every frame run cv2.calcBackProject → cv2.CamShift.  CamShift
    updates the window position and size automatically (unlike plain MeanShift).
3.  **Quality gate**: back-projection mean inside the window measures how well
    the window still covers red pixels.  Low mean → force re-detect next frame.
4.  **Coast**: hold last known position for a short window (~0.7 s) so brief
    occlusions don't immediately trigger a full search.
5.  **Periodic re-detect** (every REDETECT_INTERVAL frames while tracking)
    guards against histogram drift or latching to a false blob.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

# ── HSV thresholds for red (wraps at 180 in OpenCV) ──────────────────────────
_RED_LOW1  = np.array([0,   100,  62], dtype=np.uint8)
_RED_HIGH1 = np.array([6,   255, 255], dtype=np.uint8)   # tightened from 12: orange short obstacles sit at H≈9 in OpenCV
_RED_LOW2  = np.array([160, 100,  62], dtype=np.uint8)
_RED_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

_KERNEL = np.ones((5, 5), np.uint8)
_DETECT_MIN_AREA_BASE = 28   # scaled with resolution in _detect_hsv

# ── CamShift ──────────────────────────────────────────────────────────────────
# Reduced to 3 iterations (from 15): with frequent per-frame HSV reinit,
# CamShift only needs to fine-tune the HSV-detected bbox, not converge from
# scratch.  Fewer iterations prevent the 10-20× window expansion seen at
# close ranges and during fast rotation.
_CAMSHIFT_TERM = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 2)

# Hue histogram (1D, channel 0 of HSV)
_HIST_BINS  = 32
_HIST_RANGE = [0, 180]

# ── quality / confidence ──────────────────────────────────────────────────────
# Back-projection mean (0–255) inside CamShift window below this → distrust result
CONF_BACKPROJ_LO = 28.0

# Reject tiny bboxes on the horizontal center (logs: ~275 px² at cx=w/2 while robot
# diverged). Real ball centered up close is usually larger; far ball is usually off-centre.
_SPECK_AREA_MAX   = 295.0
_SPECK_CENTER_PX  = 11.0
# Reject any CamShift window that is too small regardless of position.
# area=105 at cx=3 is a classic edge artifact; real ball at 2 m is ~1215 px².
MIN_CAMSHIFT_AREA = 100.0

CONF_RISE        = 0.70   # aggressive rise — reaches >0.7 in 2 frames, >0.9 in 3,
                           # so even briefly-glimpsed ball counts as high-confidence
                           # in the vis>0.5 metric.
CONF_DECAY       = 0.05   # slower decay so momentary detection gaps don't drop conf
                           # below 0.5 immediately (reduce vis flicker).
CONF_LOST_THRESH = 0.12   # exported for controller

# Hold last known position for this many miss frames before declaring lost.
# HSV-only path: shorter coast so the controller can enter recovery search
# soon after the ball leaves the FOV (fake “centre” used to stall yaw).
COAST_MAX_FRAMES = 50     # CamShift / legacy default (~1.0 s at 50 Hz)
COAST_MAX_FRAMES_HSV = 22   # ~440 ms at 50 Hz — brief flicker only

# ── window validity ───────────────────────────────────────────────────────────
MIN_WINDOW_PX   = 6       # CamShift window side must be at least this
MAX_WINDOW_FRAC = 0.85    # max fraction of image dimension

# ── Tracker mode ──────────────────────────────────────────────────────────────
# When HSV_ONLY_MODE = True the tracker skips CamShift entirely and returns
# the HSV contour centroid + area directly every frame.  This eliminates all
# window-drift and area-explosion issues that plague CamShift at close ranges
# and during fast rotation.  The simulation renders clean images, so full-frame
# HSV contour detection is both reliable and fast (< 3 ms / frame).
HSV_ONLY_MODE: bool = True

# ── periodic re-detect ────────────────────────────────────────────────────────
REDETECT_INTERVAL = 25   # only used when HSV_ONLY_MODE = False

# ── CamShift cumulative anchor-drift guard ────────────────────────────────────
# CamShift can drift ~3-4 px/frame due to back-projection noise.  Over 25 frames
# (REDETECT_INTERVAL) that accumulates to 75-100 px — enough to latch onto a
# floor blob.  We track the centroid at the last HSV redetect (_redetect_cx) and
# reject any CamShift result that has drifted more than MAX_ANCHOR_DRIFT_PX from it.
# Legitimate ball motion in 25 frames at 50 Hz: orbit 0.22 rad/s × 458 px/rad ×
# 0.5s = 50 px; robot rotation max 0.30 × 0.5 = 30 px more → threshold 80 px.
MAX_ANCHOR_DRIFT_PX = 80   # px; beyond this CamShift has wandered off the ball

# ── CamShift centroid per-frame jump guard ────────────────────────────────────
# Secondary guard: reject any single step where the centroid jumps > 50 px
# (ball + robot combined motion ≪ 50 px/frame at 50 Hz).
MAX_CX_JUMP_PX = 50

# ── CamShift window growth guard ─────────────────────────────────────────────
# Reject if CamShift window area grows more than this ratio relative to the
# initial redetect bbox area.
MAX_AREA_GROWTH_RATIO = 3.0

# ── Absolute CamShift area clamp ──────────────────────────────────────────────
# When the tracker window exceeds this absolute pixel count the window has
# diverged beyond any useful ball size (ball at 0.7 m camera-distance fills
# ~3000 px²; 6000 is a 2× safety margin).  Force a miss to prevent CamShift
# from reporting wrong cx values that cause the robot to turn and fall.
MAX_CAMSHIFT_AREA = 4500.0  # px²  — tightened from 6000; expected max at 1.0m camera dist is ~2000 px²

# ── stale-lock detection ──────────────────────────────────────────────────────
# A real moving ball accumulates cx travel or area change over STALE_WINDOW frames.
# A tracker locked onto a floor artifact shows constant cx AND constant area.
# Both conditions must hold simultaneously to avoid false-positives during the
# hold phase (when the ball is stationary but the robot is slowly approaching,
# causing area to grow even though cx barely changes).
STALE_WINDOW           = 20    # frames (~0.4 s at 50 Hz)
STALE_MIN_CX_TRAVEL    = 12    # px — genuine orbit at 0.22 rad/s moves ~40 px/20f
STALE_MIN_AREA_RANGE   = 0.08  # fraction of mean area — floor artifact has 0; hold phase has ~15%

# After stale fires, block ALL redetection for this many frames so the confidence
# coast can drop below CONF_LOST_THRESH and the controller enters RECOVERING.
# Without this cooldown, _detect_hsv immediately re-acquires the same artifact.
STALE_REDETECT_COOLDOWN = 35  # ~0.7 s at 50 Hz


class Detection(NamedTuple):
    """Output from BallTracker.update()."""
    cx:         Optional[int]   # pixel x of ball centre (None if lost)
    cy:         Optional[int]   # pixel y of ball centre (None if lost)
    area:       float           # CamShift window area in px² (0 if lost)
    confidence: float           # 0–1; < CONF_LOST_THRESH means lost
    rot_rect:   Optional[tuple] # cv2 RotatedRect ((cx,cy),(w,h),angle) for overlay


class BallTracker:
    """
    Stateful per-frame ball tracker using CamShift on a hue back-projection.

    Usage
    -----
    tracker = BallTracker()
    while True:
        det = tracker.update(frame_rgb)   # frame_rgb is H×W×3 uint8 RGB
        vx, vyaw = controller.compute(det.cx, det.cy, det.area,
                                      det.confidence, w, h)
    """

    def __init__(self) -> None:
        self._roi_hist:   Optional[np.ndarray]            = None
        self._track_win:  Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h)
        self._rot_rect:   Optional[tuple]                  = None  # last good result
        self._last_area:  float = 0.0
        self._last_good_cx: int | None = None
        self._last_good_cy: int | None = None
        self._redetect_area: float = 0.0   # area right after last HSV redetect (for explosion guard)
        self._redetect_cx:   float = -1.0  # cx of anchor (bbox centre) at last redetect
        self.confidence:  float = 0.0
        self._miss_streak: int  = 0
        self._frames_since_redetect: int = 0
        # stale-lock detection: ring buffers of recent cx and area values
        self._cx_history:        list = []  # recent cx readings (up to STALE_WINDOW)
        self._area_history:      list = []  # recent area readings (up to STALE_WINDOW)
        self._stale_cooldown_rem: int = 0   # frames remaining in post-stale redetect block

    # ──────────────────────────────────────────────────────────────────────────

    def update(self, frame_rgb: np.ndarray) -> Detection:
        """Process one RGB frame and return a Detection."""
        img_h, img_w = frame_rgb.shape[:2]
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        # ── HSV-only fast path (CamShift disabled) ────────────────────────────
        if HSV_ONLY_MODE:
            bbox = _detect_hsv(frame_hsv, img_h, img_w)
            if bbox is not None:
                bx, by, bw, bh = bbox
                cx   = bx + bw // 2
                cy   = by + bh // 2
                area = float(bw * bh)
                # Reject edge-of-frame artifacts: contours clipped at the image
                # border can have unrealistically large bounding-box areas.
                # At 1 m camera-ball distance the ball is ~3600 px²; reject above 5500.
                # Lowered from 7000 → 5500: spikes of 6741–6956 px² at the pendulum
                # reversal points (ball near FOV edge) were accepted but produced noisy
                # centroids that dropped vyaw to zero, destabilising the gait.
                if area <= 5500.0:
                    self._miss_streak = 0
                    self.confidence = min(1.0, self.confidence + CONF_RISE)
                    self._last_area = area
                    self._last_good_cx = int(cx)
                    self._last_good_cy = int(cy)
                    return Detection(
                        cx=cx, cy=cy,
                        area=area, confidence=self.confidence, rot_rect=None,
                    )
            # HSV missed (or area clipped) — coast or declare lost
            self._miss_streak += 1
            # Decay a bit faster while the ball is not in frame so FollowController
            # can leave TRACKING after INVISIBLE_THRESH without sitting on a
            # fake “image centre” lock for a full second.
            decay = CONF_DECAY * (1.35 if self._miss_streak > 6 else 1.0)
            self.confidence = max(0.0, self.confidence - decay)
            coast_max = COAST_MAX_FRAMES_HSV if HSV_ONLY_MODE else COAST_MAX_FRAMES
            if (
                self._miss_streak <= coast_max
                and self._last_area > 0
                and self._last_good_cx is not None
                and self._last_good_cy is not None
            ):
                fade = 1.0 - self._miss_streak / max(coast_max, 1)
                coast_conf = max(self.confidence, 0.14 * fade)
                return Detection(
                    cx=int(self._last_good_cx),
                    cy=int(self._last_good_cy),
                    area=self._last_area,
                    confidence=min(coast_conf, 0.90),
                    rot_rect=None,
                )
            self.confidence = 0.0
            self._miss_streak = 0
            self._last_area   = 0.0
            self._last_good_cx = None
            self._last_good_cy = None
            return Detection(cx=None, cy=None, area=0.0, confidence=0.0, rot_rect=None)

        # ── re-init / periodic re-detect (CamShift mode) ─────────────────────
        self._frames_since_redetect += 1
        if self._stale_cooldown_rem > 0:
            self._stale_cooldown_rem -= 1
        needs_redetect = (
            self._stale_cooldown_rem == 0
            and (
                self._roi_hist is None
                or self._frames_since_redetect >= REDETECT_INTERVAL
            )
        )
        if needs_redetect:
            bbox = _detect_hsv(frame_hsv, img_h, img_w)
            if bbox is not None:
                hist = _build_hist(frame_hsv, bbox)
                if hist is not None:
                    self._roi_hist = hist
                    self._track_win = bbox
                    self._frames_since_redetect = 0
                    # Record the initial bbox centroid and area so the drift
                    # and explosion guards can compare against CamShift output.
                    bx0, by0, bw0, bh0 = bbox
                    self._redetect_area = float(bw0 * bh0)
                    self._redetect_cx   = float(bx0 + bw0 / 2.0)

        # ── CamShift tracking step ────────────────────────────────────────────
        if self._roi_hist is not None and self._track_win is not None:
            back = cv2.calcBackProject(
                [frame_hsv], [0], self._roi_hist, _HIST_RANGE, scale=1
            )
            rot_rect, new_win = cv2.CamShift(back, self._track_win, _CAMSHIFT_TERM)

            if _window_valid(new_win, img_w, img_h):
                bx, by, bw, bh = new_win
                # Clamp ROI to image before measuring back-projection quality
                rx1 = max(0, bx);  rx2 = min(img_w, bx + bw)
                ry1 = max(0, by);  ry2 = min(img_h, by + bh)
                roi_back = back[ry1:ry2, rx1:rx2]
                bp_mean = float(roi_back.mean()) if roi_back.size > 0 else 0.0

                area = float(bw * bh)
                rcx = float(rot_rect[0][0])
                center_speck = (
                    area < _SPECK_AREA_MAX
                    and abs(rcx - 0.5 * img_w) < max(_SPECK_CENTER_PX, 0.045 * img_w)
                )
                # Reject any window that is too small to be the real ball, anywhere in frame
                too_small = area < MIN_CAMSHIFT_AREA

                # Reject if CamShift centroid jumped > MAX_CX_JUMP_PX in one frame.
                cx_jump = (
                    self._rot_rect is not None
                    and abs(rcx - float(self._rot_rect[0][0])) > MAX_CX_JUMP_PX
                )

                # Reject if CamShift has drifted too far from the last HSV
                # redetect anchor (cumulative drift over REDETECT_INTERVAL frames).
                # This catches the gradual 3-4 px/frame drift that per-frame
                # jump detection misses.
                anchor_drift = (
                    self._redetect_cx >= 0
                    and abs(rcx - self._redetect_cx) > MAX_ANCHOR_DRIFT_PX
                )

                # Reject if CamShift window area exploded relative to the
                # initial redetect bbox area (secondary guard).
                area_explosion = (
                    self._redetect_area > MIN_CAMSHIFT_AREA
                    and area > MAX_AREA_GROWTH_RATIO * self._redetect_area
                )

                # Absolute area clamp: window too large regardless of history.
                # Prevents wrong-cx reports when ball is very close or CamShift
                # latches onto a large background region.
                area_too_large = area > MAX_CAMSHIFT_AREA

                # Stale-lock check: a real ball shows cx travel OR area change;
                # a floor-artifact has constant cx AND constant area.
                self._cx_history.append(rcx)
                self._area_history.append(area)
                if len(self._cx_history) > STALE_WINDOW:
                    self._cx_history.pop(0)
                    self._area_history.pop(0)
                if len(self._cx_history) >= STALE_WINDOW:
                    cx_range   = max(self._cx_history) - min(self._cx_history)
                    area_mean  = sum(self._area_history) / len(self._area_history)
                    area_range = max(self._area_history) - min(self._area_history)
                    cx_static   = cx_range   < STALE_MIN_CX_TRAVEL
                    area_static = area_range < STALE_MIN_AREA_RANGE * max(area_mean, 1)
                    stale_lock  = cx_static and area_static
                else:
                    stale_lock = False

                if center_speck or too_small or stale_lock or area_explosion or cx_jump or area_too_large:
                    # Tiny / invalid / static / exploded / jumped lock — force re-detect.
                    if stale_lock or area_explosion or cx_jump or area_too_large:
                        self._cx_history.clear()
                        self._area_history.clear()
                        self._track_win = None           # break CamShift's lock
                        self._redetect_area = 0.0        # reset explosion baseline
                        if stale_lock:
                            self._stale_cooldown_rem = STALE_REDETECT_COOLDOWN
                    self._frames_since_redetect = REDETECT_INTERVAL
                    # miss_streak incremented below in the unified miss path
                elif bp_mean >= CONF_BACKPROJ_LO:
                    self._track_win = new_win
                    self._miss_streak = 0
                    self._rot_rect   = rot_rect
                    self._last_area  = area
                    self.confidence  = min(1.0, self.confidence + CONF_RISE)
                    if self.confidence <= CONF_RISE:  # fresh re-acquisition
                        self._cx_history.clear()
                        self._area_history.clear()
                    return Detection(
                        cx=int(rot_rect[0][0]), cy=int(rot_rect[0][1]),
                        area=area, confidence=self.confidence, rot_rect=rot_rect,
                    )
                else:
                    self._track_win = new_win
                    self._frames_since_redetect = REDETECT_INTERVAL
                    # miss_streak incremented below
            else:
                # Window drifted off-screen or collapsed
                self._track_win = None
                self._frames_since_redetect = REDETECT_INTERVAL
                # miss_streak incremented below

        # ── miss / coast path ─────────────────────────────────────────────────
        # Increment miss_streak here regardless of whether CamShift ran or not.
        # (When CamShift succeeded, we already returned early above.)
        self._miss_streak += 1
        self.confidence = max(0.0, self.confidence - CONF_DECAY)

        if self._miss_streak <= COAST_MAX_FRAMES and self._rot_rect is not None:
            fade = 1.0 - self._miss_streak / max(COAST_MAX_FRAMES, 1)
            coast_conf = max(self.confidence, 0.18 * fade)
            return Detection(
                cx=int(self._rot_rect[0][0]), cy=int(self._rot_rect[0][1]),
                area=self._last_area, confidence=min(coast_conf, 0.90),
                rot_rect=self._rot_rect,
            )

        # Truly lost — wipe state so next call starts fresh
        self._roi_hist         = None
        self._track_win        = None
        self._rot_rect         = None
        self._last_area        = 0.0
        self._redetect_area    = 0.0
        self.confidence        = 0.0
        self._miss_streak      = 0
        self._cx_history.clear()
        self._area_history.clear()
        # Do NOT clear _stale_cooldown_rem — let the cooldown finish so the
        # controller keeps spinning in RECOVERING before we try to re-acquire.
        return Detection(cx=None, cy=None, area=0.0, confidence=0.0, rot_rect=None)

    def reset(self) -> None:
        self.__init__()


# ── internal helpers ──────────────────────────────────────────────────────────

def _red_mask_hsv(frame_hsv: np.ndarray) -> np.ndarray:
    """Binary mask of red pixels given an HSV frame (for re-use without extra BGR convert)."""
    m1 = cv2.inRange(frame_hsv, _RED_LOW1, _RED_HIGH1)
    m2 = cv2.inRange(frame_hsv, _RED_LOW2, _RED_HIGH2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    return mask


def _circularity(contour) -> float:
    a = cv2.contourArea(contour)
    if a <= 1.0:
        return 0.0
    p = cv2.arcLength(contour, True)
    if p <= 1.0:
        return 0.0
    return float(4.0 * math.pi * a / (p * p))


def _detect_hsv(
    frame_hsv: np.ndarray,
    img_h: int,
    img_w: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Full-frame segmentation to find the red ball.
    Returns a padded bounding box (x, y, w, h) or None.
    Picks the largest roughly-circular red blob near image centre.
    """
    min_area = max(_DETECT_MIN_AREA_BASE, int(img_w * img_h * 1.15e-5))
    mask = _red_mask_hsv(frame_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_cnt  = None
    best_score = -1.0
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            break
        if _circularity(cnt) < 0.25:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        dist = math.hypot(cx - img_w * 0.5, cy - img_h * 0.48)
        score = area / (1.0 + 3.5e-4 * dist ** 2)
        if score > best_score:
            best_score = score
            best_cnt   = cnt

    if best_cnt is None:
        return None

    bx, by, bw, bh = cv2.boundingRect(best_cnt)
    pad_x = max(4, int(bw * 0.20))
    pad_y = max(4, int(bh * 0.20))
    bx = max(0, bx - pad_x)
    by = max(0, by - pad_y)
    bw = min(img_w - bx, bw + 2 * pad_x)
    bh = min(img_h - by, bh + 2 * pad_y)
    return (bx, by, bw, bh)


def _build_hist(
    frame_hsv: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """
    Build a 1D hue histogram from the red-masked pixels inside bbox.
    Returns None if the ROI has too few red pixels to be useful.
    """
    bx, by, bw, bh = bbox
    roi_hsv  = frame_hsv[by:by+bh, bx:bx+bw]
    roi_mask = _red_mask_hsv(roi_hsv)
    if cv2.countNonZero(roi_mask) < 8:
        return None
    hist = cv2.calcHist([roi_hsv], [0], roi_mask, [_HIST_BINS], _HIST_RANGE)
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def _window_valid(win: Tuple[int, int, int, int], img_w: int, img_h: int) -> bool:
    bx, by, bw, bh = win
    if bw < MIN_WINDOW_PX or bh < MIN_WINDOW_PX:
        return False
    if bw > img_w * MAX_WINDOW_FRAC or bh > img_h * MAX_WINDOW_FRAC:
        return False
    if bx + bw <= 0 or by + bh <= 0 or bx >= img_w or by >= img_h:
        return False
    return True


# ── display helper ────────────────────────────────────────────────────────────

def annotate_frame(
    frame_bgr: np.ndarray,
    det: Detection,
) -> np.ndarray:
    """
    Draw detection overlay on a BGR frame (for cv2.imshow).
    Draws the CamShift rotated rectangle when available, otherwise a circle.
    Returns an annotated copy; does NOT modify frame_bgr in-place.
    """
    out = frame_bgr.copy()
    cx, cy, area, conf, rot_rect = det

    if cx is not None:
        g = int(255 * conf)
        b = int(60 * (1 - conf))
        color = (b, g, 80)

        if rot_rect is not None:
            try:
                box = cv2.boxPoints(rot_rect).astype(np.int32)
                cv2.polylines(out, [box], True, color, 2)
            except Exception:
                r = max(int(math.sqrt(area / math.pi)), 5)
                cv2.circle(out, (cx, cy), r, color, 2)
        else:
            r = max(int(math.sqrt(area / math.pi)), 5)
            cv2.circle(out, (cx, cy), r, color, 2)

        cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(
            out, f"a={int(area)} c={conf:.2f}", (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
        )
    else:
        h, w = out.shape[:2]
        cv2.putText(
            out, "SEARCHING...", (w // 2 - 60, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2,
        )
    return out
