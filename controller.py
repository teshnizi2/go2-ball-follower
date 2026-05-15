"""
controller.py – Simple state-machine controller for Go2 ball following.

States
------
SEARCHING  – No ball visible. Rotate at constant speed until ball found.
TRACKING   – Ball visible. Centre it with proportional yaw and try to close
             range to **under 1.0 m** (world distance in sim when provided).
RECOVERING – Ball not seen for > INVISIBLE_THRESH frames. Rotate toward
             the side it was last seen on.

Key design choices
------------------
• Distance goal **< 1.0 m**: forward `vx` while range exceeds 1.0 m; hold when
  inside (0.35 m, 1.0 m] to avoid fighting the gait; below 0.35 m hold (no retreat).
• TRACKING hysteresis: brief tracker failures (head-bob, momentary occlusion) do
  NOT immediately trigger RECOVERING.  The robot must have INVISIBLE_THRESH
  consecutive invisible frames before the state changes.  During brief loss the
  last valid commands are repeated.
• Dead-band yaw: absorbs tracker jitter; kept small so the ball stays near the
  optical centre.  Slightly softer vx-vs-yaw coupling when error is moderate so
  the robot can still walk while fine-centring.
• Distance calibration: empirically, at 2.0 m world distance the ball covers
  ≈ 160 px² at 320×240 (≈ 360 px² at 480×360).  DIST_CAL = 1.1 corrects the
  previous over-estimate of 2.0 so the dead-band matches world distances.

Distance formula (inverse-square law):
    target_area = TARGET_AREA_FRAC × img_w × img_h   (area at 2 m ≈ 360 px² @ 480×360)
    dist_est    = DIST_CAL × √(target_area / area)
→  When area = target_area → dist_est = DIST_CAL = 1.1 m   (calibration anchor)
   At world 2 m → area≈160px²(320×240) → dist_est=1.1×√(540/160)=2.02 m ✓
"""

from __future__ import annotations
import math

# ── States ────────────────────────────────────────────────────────────────────
SEARCHING  = "searching"
TRACKING   = "tracking"
RECOVERING = "recovering"

# Print a tracking status line every this many TRACKING compute() calls
_LOG_EVERY_TRACK: int = 25


def _log(msg: str) -> None:
    import sys
    print(f"[CTRL] {msg}", flush=True, file=sys.stdout)


# ── Search / recovery speeds ──────────────────────────────────────────────────
# Single yaw-rate ceiling for TRACKING + RECOVERING + SEARCHING.  CSV logs
# (e.g. run_20260416_130411, 125720) show low robot_z with |vyaw_cmd| pegged
# ~1.1–1.2 rad/s during search/recovery; capping at ~0.70 rad/s stabilises the RL trot.
MAX_ROTATION_CMD: float = 0.90   # rad/s COMMAND cap.  Sustained cmd=1.2 caused
                                   # roll accumulation (falls after ~20s).  At cmd=0.9
                                   # the RL policy achieves ~0.35 rad/s actual, which is
                                   # enough to track ball at ω=0.18 rad/s with FF assist.
                                  # pulses.  Sustained 0.90 is unstable (RECOVERING roll-up),
                                  # so SEARCH_VYAW is lowered separately to 0.5 below.
SEARCH_VX:       float = 0.10   # forward nudge while searching / recovering — MUST be nonzero
                                  # whenever |vyaw|>0, or the policy topples within ~13 s.
                                  # 2026-04-22: dropped 0.30 → 0.10 because the shrunken-arena
                                  # (r=0.95) ball orbits near origin; at 0.30 the robot walked
                                  # forward during search, ended up 1.3m past origin with the
                                  # ball BEHIND it, and never saw the ball again.  0.10 keeps
                                  # the gait alive (SEARCH_VYAW=0.5 is below the fall corner at
                                  # vyaw≈0.9) while drift stays under ~0.5m over 5s search.
SEARCH_VYAW:     float = 0.25   # CONTINUOUS-mode search yaw cap (kept under the 0.3-0.9
                                  # dead-zone for stability).  Used as fallback only — the
                                  # ACTUAL search yaw is applied via bang-bang pulses below
                                  # (SEARCH_BANG_*) so we average ≈0.36 rad/s effective yaw
                                  # without sustaining a dead-zone command.  Continuous 1.0
                                  # toppled the policy via the EMA dragging through the dead
                                  # zone (vyaw=0.9 sustained → roll-up).
SEARCH_BANG_ON_S:  float = 0.30   # search bang-bang ON pulse — shorter than tracking pulse
                                   # to limit roll accumulation.
SEARCH_BANG_OFF_S: float = 0.50   # search bang-bang OFF rest — longer than ON so duty cycle
                                   # stays ~38% → effective yaw ≈ YAW_HIGH × 0.38 = 0.34 rad/s.
                                   # Beats the ball's 0.26 rad/s orbit but with enough rest to
                                   # let the policy null any accumulated roll between pulses.
SEARCH_DIR_FLIP_S: float = 7.0    # 2026-04-22: raised 1.6→7.0 for continuous sub-dead-zone
                                   # search (0.22 rad/s).  Ball orbits at 0.26 rad/s; at
                                   # opposing direction relative speed = 0.48 rad/s → full
                                   # 360° sweep takes 13s.  7s is enough to cover a half-orbit
                                   # and catch the ball in one direction before flipping.  Roll
                                   # accumulation is no longer an issue since 0.22 is below the
                                   # dead zone (sustained stable per probe).
# Gait keep-alive: RL policy is trained for locomotion, not static balance.
# When both vx and vyaw are near zero (ball centred and at goal), sustained
# zero commands allow the robot to drift into an unstable pose and fall.
# CSV logs show every fall preceded by 19-43 steps of exactly vx=0, vyaw=0
# while detected=1 at dist≈2.45 m.  A small forward nudge keeps the policy
# in its trained regime without significantly affecting tracking accuracy.
MIN_VX_KEEPALIVE: float = 0.08  # m/s — keep gait alive with less drift when centered
SEARCH_SWEEP_S: float = 4.5   # s — full sweep before reversing (halved so the robot tries
                              #     the other direction sooner when a short spin fails)
RECOVER_MAX_S: float = 4.0    # s — recovery search timeout.  With bang-bang search
                              # (~0.51 rad/s effective yaw) 4 s = 2.0 rad = 117° sweep —
                              # wide enough to catch the ball.  Roll-up risk low because
                              # bang-bang has built-in rest periods (no sustained dead-zone).

# ── TRACKING hysteresis ───────────────────────────────────────────────────────
# Require this many consecutive invisible frames before leaving TRACKING.
# The RL trot gait causes periodic head-bob at ~2.2 Hz (≈23 ctrl steps/cycle)
# which makes the tracker periodically lose confidence for 5–10 frames.
# Setting threshold = 20 steps (400 ms) absorbs this without masking real loss.
INVISIBLE_THRESH: int = 45    # × 20 ms/step = 900 ms hysteresis.  Raised 15→45 because
                              # analysis of circle run (vis=10.8%) showed the robot exits
                              # TRACKING after only 15 invisible frames, even though ball
                              # often re-enters FOV within 0.5-0.8s if robot just keeps
                              # its last yaw.  Going to SEARCH triggers bang-bang that then
                              # misses the ball entirely.  0.9s hysteresis lets the ball
                              # re-acquire without losing tracker state.

# ── Centering (yaw) ───────────────────────────────────────────────────────────
# ── Bang-bang yaw with hysteresis ─────────────────────────────────────────────
# A 30-second fine-grained stability sweep (see /tmp/yaw_probe3.py) showed the RL
# gait has a DEAD-ZONE of unstable yaw commands between roughly 0.30 and 0.90 rad/s:
#     (vx=0.5, vyaw=0.2) OK 60s   (vx=0.5, vyaw=0.4) falls 21s
#     (vx=0.6, vyaw=0.2) falls 36s (vx=0.6, vyaw=0.3) falls 21s
#     (vx=0.8, vyaw=0.5) falls 11s  …   (vx=0.8, vyaw=1.0) stable-ish
# Implication: proportional yaw like 0.008 × err dwells in the 0.3–0.9 fall zone
# whenever the cx error sits between 40–110 px — exactly the regime we spent most
# of the tracking time in.  We replace the proportional controller with a
# three-state (−HIGH, 0, +HIGH) bang-bang that hysteretically flips on/off, and
# we couple forward speed to the yaw state so we never hold (high vx, high vyaw).
X_TURN_ON_PX:    float = 35.0   # enter a turn when |cx_err| > this
X_TURN_OFF_PX:   float = 14.0   # leave the turn when |cx_err| < this (hysteresis)
YAW_HIGH:        float = 0.90   # command emitted while turning — bang-bang pulses only
                                 # hold for TURN_MAX_S=0.4s so roll doesn't accumulate.
                                 # Needed this high to outpace ball ω=0.26 rad/s at the
                                 # bang-bang duty cycle (~40%, so effective yaw 0.36 rad/s).
# Dead-band from the old proportional controller, still used for distance/logging
X_DEADBAND_PX: float = 20.0
# Beyond this fraction, zero vx — ball almost at edge, rotate-only is correct.
X_STOP_FWD_FRAC: float = 0.55   # |x_err| > 264 px at 480 → rotate only

# Yaw-coupled forward speed:  While turning HARD we reduce vx to stay out of the
# (high-vx, high-vyaw) fall corner that short-duration probes sometimes hit.
VX_WHILE_TURNING:  float = 0.40   # vx cap while |vyaw| == YAW_HIGH
                                  # 2026-04-21: bumped 0.30→0.40 so robot closes
                                  # distance during yaw pulses when ball is faster;
                                  # the fall corner starts much higher so 0.40 stays safe.
VX_WHILE_STRAIGHT: float = 0.60   # vx cap while vyaw == 0  (probe-stable at 60s)
# ── Distance control — close inside 1.0 m ────────────────────────────────────
# Calibration: the formula dist = DIST_CAL × √(target_area / area) … (see above).
DIST_CAL:         float = 1.1
TARGET_AREA_FRAC: float = 1_215.0 / (480 * 360)   # area-fraction at 2 m (480×360)

# Approach while range exceeds this; hold vx when inside (MIN_SAFE, GOAL].
TARGET_DIST_GOAL:     float = 1.0    # m — target stand-off
TARGET_DIST_MIN_SAFE: float = 0.35   # m — do not demand closer (no backward walk)

# When ball area exceeds this in the camera frame, the ball is too close
# for reliable CamShift tracking (area>5000 px² → visual dist < ~0.9 m).
# Stop advancing to prevent tracker runaway; the _min_approach guard in
# DemoTargetMover provides the primary protection; this is the belt-and-suspenders.
AREA_TOO_CLOSE: float = 5000.0   # px²

KP_VX:       float = 2.00   # command-units per metre beyond GOAL.  At 0.4 m over goal → cmd 0.8
                             # (saturates MAX_VX).  At 0.1 m over goal → cmd 0.2 (below SEARCH_VX
                             # floor, which floors to SEARCH_VX).
MAX_VX:      float = 0.80   # COMMAND cap — 60-s stability probe showed (0.8, 1.0) runs indefinitely
                             # while (0.8, 0.5) falls in ~10 s and (1.0, 0.0) eventually falls at
                             # ~59 s.  The RL policy has a non-monotone stability manifold: paradoxically
                             # it's more stable at FULL yaw than at half yaw, likely because the policy
                             # was trained with higher-magnitude yaw commands in its replay.
                             # 0.8 gives ~0.13 m/s actual forward velocity, plenty for closing.
# Backward walking is disabled: the RL policy was trained primarily for forward
# locomotion, and backward motion in our MuJoCo sim produces an unstable gait
# that causes falls within seconds.  When the tracker reports the ball as CLOSE
# (often a CamShift area-inflation artefact at ~2 m true distance), the robot
# simply holds position (vx=0) rather than reversing.
# For true proximity (ball actually < 0.5 m), the proximity guard below limits
# yaw and keeps vx=0 anyway.
MAX_VX_BACK: float = 0.0    # retreat disabled

# Cap aggressive yaw only when physically very close (separate from 1.0 m goal).
PROXIMITY_YAW_CAP_M: float = 0.55
PROXIMITY_YAW_LIMIT: float = 0.6    # rad/s — cap (not floor) when physically close

# ── Speed limits ──────────────────────────────────────────────────────────────
MAX_VYAW: float = MAX_ROTATION_CMD   # TRACKING clamp (same as search/recovery spin cap)

# Exported for tracker / logger
CONF_LOST_THRESH: float = 0.12


def apply_stability_cap(vx: float, vyaw: float) -> tuple[float, float]:
    """
    Final-stage (vx, vyaw) safety cap.  At the current command scaling (vx up to
    1.2, vyaw up to 1.0) the old "(vx≥0.35, vyaw≥0.34) unstable corner" no longer
    applies — velocity probes showed the policy operates stably at cmd (1.0, 0.8)
    with actual velocities ≈ (0.17 m/s, 0.55 rad/s).  This function now just clamps
    to the command caps so a too-hot raw command (should never happen) can't escape.
    """
    vx   = max(-MAX_VX,   min(MAX_VX,   vx))
    vyaw = max(-MAX_VYAW, min(MAX_VYAW, vyaw))
    return vx, vyaw


class FollowController:
    """
    3-state ball-follow controller with TRACKING hysteresis.

    Call .compute(cx, cy, area, conf, img_w, img_h, dt, world_dist_m=...) every
    control step.  Returns (vx, vyaw).

    If ``world_dist_m`` is set (simulation: true robot↔ball distance in metres),
    distance control uses it instead of vision-only ``dist_est``.  Omit on a
    real robot with no ground truth.
    """

    def __init__(self) -> None:
        self._state:          str   = SEARCHING
        self._last_turn_dir:  float = 1.0
        self._recover_time:   float = 0.0
        self._search_sweep_time: float = 0.0  # time accumulator for SEARCHING direction reversal
        self._invisible_ctr:  int   = 0     # consecutive invisible frames
        self._track_log_ctr:    int   = 0     # periodic log counter
        self._last_dist_cat:    str   = ""   # FAR / OK / CLOSE — only log transitions
        self._last_vx:        float = 0.0   # last valid commands (repeated during brief loss)
        self._last_vyaw:      float = 0.0
        # Bang-bang yaw state: -1 = turning right, 0 = straight, +1 = turning left.
        # Hysteresis-gated by X_TURN_ON_PX / X_TURN_OFF_PX to prevent chatter.
        self._yaw_bb:         int   = 0
        self._yaw_bb_hold_t:  float = 0.0   # time remaining in current bb state before flip allowed
        self._yaw_cooldown_t: float = 0.0   # time remaining after a turn during which new turns blocked
        self._turn_active_t:  float = 0.0   # time spent continuously in a turn (forces break)
        # Roll-accumulation defence: N consecutive same-direction pulses force a longer rest
        # period with vyaw=0 to let the policy regain zero-lean equilibrium.
        self._same_dir_pulses: int   = 0    # count of consecutive same-direction pulses
        self._forced_rest_t:  float = 0.0   # remaining time in a forced straight-walk rest
        # Bang-bang search state — pulses YAW_HIGH ON for SEARCH_BANG_ON_S then
        # rests at vyaw=0 for SEARCH_BANG_OFF_S.  Avoids sustaining a dead-zone
        # yaw command while still averaging > 0.26 rad/s effective yaw.
        self._search_bang_phase_t: float = 0.0
        self._search_bang_on:      bool  = True   # start in ON pulse
        self._search_dir_t:        float = 0.0    # time since last direction flip (search)
        # Feed-forward tracking: estimate ball angular velocity in FOV (pixels/sec)
        # and command matching yaw, so robot holds ball at consistent cx without lag.
        # Solves the fundamental lag of pure P: ball drifts at 0.26 rad/s, proportional
        # reacts only AFTER ball has moved, so ball continuously escapes toward edge.
        self._x_err_prev:   float = 0.0
        self._vyaw_ff_ema:  float = 0.0   # EMA-filtered feedforward to absorb cx jitter
        # Soft hysteresis for the yaw-cap policy in TRACKING — once the
        # controller picks low-yaw or burst-yaw mode, hold for >=5 steps
        # before allowing a flip.  This kills single-step chatter (which
        # produced the visible jerk) but doesn't lock the controller in
        # burst mode forever (sustained burst = robot rotates in place
        # without forward progress).  The burst mode is also force-exited
        # after _yaw_burst_max_steps to guarantee periodic forward bursts.
        self._yaw_mode:           str   = "low"   # "low" or "burst"
        self._yaw_mode_hold_t:    int   = 0       # ctrl steps left in current hold
        self._yaw_burst_dur_t:    int   = 0       # consecutive ctrl steps in burst
        # Roll-accumulation guard: exponential moving average of signed yaw
        # commands.  When this crosses a magnitude threshold, the controller
        # forces a rest period with vyaw=0 to let the RL gait null any
        # accumulated lateral roll before the next yaw command.
        self._yaw_accum_ema: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        cx,
        cy,
        area:  float,
        conf:  float,
        img_w: int,
        img_h: int,
        dt:    float = 0.02,
        world_dist_m: float | None = None,
        world_bearing_rate: float | None = None,
        world_bearing_rad: float | None = None,
    ) -> tuple[float, float]:
        ball_visible = (cx is not None) and (conf >= CONF_LOST_THRESH)

        # ── Sim-mode override: face the LEAD point (intercept) ─────────────
        # main.py supplies world_bearing_rad = bearing to intercept point
        # (where ball will be when we arrive).  We override cx to drive P-yaw
        # toward that point regardless of where the camera-detected ball is.
        # When the lead point is outside FOV (|b|>1.0), we clip the synthetic
        # pixel error to the max so the P-term saturates in the right
        # direction, spinning the robot toward the target.
        FOV_HALF_RAD = 1.00
        bearing_mag_rad = 0.0
        if world_bearing_rad is not None:
            bearing_mag_rad = abs(world_bearing_rad)
            b_clip = max(-FOV_HALF_RAD * 1.5, min(FOV_HALF_RAD * 1.5, world_bearing_rad))
            synthetic_x_err = -b_clip * (img_w / 2.0) / FOV_HALF_RAD
            synthetic_cx = img_w / 2.0 + synthetic_x_err
            cx = synthetic_cx
            ball_visible = True
            if conf < 1.0:
                conf = 1.0
        # ── TRACKING ──────────────────────────────────────────────────────────
        if ball_visible:
            self._invisible_ctr = 0

            if self._state != TRACKING:
                _log(f"→ TRACKING  (cx={cx}, conf={conf:.2f})")
                # Reset FF state so first frame doesn't see a huge err_rate jump
                # (x_err jumps from stale value to current on acquisition).
                self._x_err_prev  = cx - img_w / 2.0
                self._vyaw_ff_ema = 0.0
            self._state             = TRACKING
            self._recover_time      = 0.0
            self._search_sweep_time = 0.0

            # ── Yaw (smooth proportional, dead-zone-aware) ─────────────────────
            # 2026-04-22: replaced bang-bang with proportional yaw because
            # bang-bang pulses overshot the ball at the shrunken-arena radius
            # (0.95m): ball orbits at 0.26 rad/s, a 0.4s pulse at 0.9 rad/s
            # rotates 21° while ball only drifts 6° — robot overshoots by 15°
            # sending ball to opposite FOV edge, then cooldown lets ball slip
            # back out.  Proportional yaw capped at 0.25 rad/s (just below the
            # 0.3 rad/s dead-zone start) smoothly tracks orbiting ball without
            # triggering fall conditions.  Vis%=10.8 observed with bang-bang
            # tracking → we target >50% with smooth proportional tracking.
            x_err     = cx - img_w / 2.0
            turn_on   = X_TURN_ON_PX  * (img_w / 480.0)
            turn_off  = X_TURN_OFF_PX * (img_w / 480.0)
            deadband  = X_DEADBAND_PX * (img_w / 480.0)
            # Proportional yaw: kp chosen so that a ball 1/3 from center triggers
            # full tracking yaw of 0.25 rad/s.  Below 0.3 rad/s = outside the
            # dead zone.  When ball is near FOV edge (|err|>180px ~ 43°) we
            # escalate to a bang-bang pulse that briefly crosses the dead zone
            # fast enough to recover.
            # 2026-04-22 v2: raised 0.25 → 0.28 because ball orbits at 0.26 rad/s
            # (circle) / effective 0.22 (fig8).  At 0.25 the robot cannot catch up
            # with the ball's angular velocity, so ball drifts out of FOV
            # monotonically → visibility ~19%.  0.28 is still below the 0.30-0.90
            # dead zone (probe: vyaw=0.3 falls at 21s, but 0.28 is untested; we
            # rely on vx being ≈0 in TRACKING since ball mostly at d<1.0m in the
            # shrunken arenas, so (vx=0, vyaw=0.28) is outside all probed fall
            # corners which had vx ≥ 0.5).
            # The RL policy scales vyaw obs by 2.0; effective yaw saturates around
            # cmd=1.0.  Sub-dead-zone commands (<0.3) produce only ~20% of the
            # effective yaw, while super-dead-zone commands (>0.9) are stable
            # and strong.  Using YAW_TRACK_MAX=0.9 lets proportional ramp
            # through the 0.3-0.9 "dead zone" quickly (error proportional), so
            # we rarely dwell there — the instability is about SUSTAINED
            # mid-band commands, not transient passes.
            YAW_TRACK_MAX  = 0.50   # rad/s — P component cap; combined with FF
                                     # gives total commanded yaw up to MAX_VYAW
            YAW_PROP_SAT_PX    = 80.0 * (img_w / 480.0)   # saturates at err=80px
            YAW_EDGE_THRESH_PX = 180.0 * (img_w / 480.0)  # (unused — proportional IS the max)
            YAW_DEADBAND_PX    = 15.0 * (img_w / 480.0)   # moderate deadband
            # Maintain _last_turn_dir for search-direction memory (where was ball last)
            if x_err < -YAW_DEADBAND_PX:
                self._last_turn_dir = +1.0   # ball to LEFT → remember LEFT
            elif x_err > YAW_DEADBAND_PX:
                self._last_turn_dir = -1.0   # ball to RIGHT → remember RIGHT

            # Proportional yaw with ball-angular-velocity feed-forward.
            if abs(x_err) < YAW_DEADBAND_PX:
                vyaw_p = 0.0
            else:
                vyaw_p = -YAW_TRACK_MAX * (x_err / YAW_PROP_SAT_PX)
                vyaw_p = max(-YAW_TRACK_MAX, min(YAW_TRACK_MAX, vyaw_p))

            # FF: match ball's world angular rate.  The RL policy achieves
            # ~30% of commanded yaw at vx=0.2-0.3, so KFF≈3 makes actual yaw
            # approximately match ball's rate.  Cap the combined yaw at 0.5
            # to avoid sustained high-yaw roll accumulation over long runs.
            if world_bearing_rate is not None:
                vyaw_ff = 3.0 * world_bearing_rate
                vyaw_ff = max(-MAX_VYAW, min(MAX_VYAW, vyaw_ff))
            else:
                vyaw_ff = 0.0

            vyaw = vyaw_ff + vyaw_p
            # Binary yaw policy with SOFT hysteresis (5-step hold) and a
            # burst-mode duration cap (50 steps = 1 s).  Snap vyaw to
            # either tiny (≤0.13) or full (≥0.9) to stay out of the
            # probed-unstable 0.15-0.85 dead zone.  The hold prevents
            # single-step chatter between the two caps when the bearing
            # jitters around the trigger; the duration cap forces the
            # controller back into low-yaw mode after 1 s of burst so
            # the robot can actually translate forward (sustained burst
            # caps vx at 0.65 → robot rotates in place).
            # Tightened for shortest-path mode at high action_scale:
            # engage sharp turn sooner so the robot doesn't arc around
            # bearing errors > 10° while driving forward at 0.7 m/s.
            BURST_TRIGGER_RAD  = 0.18   # was 0.30 → engage burst at ~10°
            BURST_MAX_STEPS    = 75     # 1.5 s @ 50 Hz (was 1.0 s)
            HOLD_MIN_STEPS     = 5
            want_burst = (bearing_mag_rad > BURST_TRIGGER_RAD)
            if self._yaw_mode_hold_t > 0:
                self._yaw_mode_hold_t -= 1
            else:
                new_mode = "burst" if want_burst else "low"
                if (new_mode == "burst"
                        and self._yaw_burst_dur_t >= BURST_MAX_STEPS):
                    # Force-exit to low so robot can drive forward
                    new_mode = "low"
                    self._yaw_burst_dur_t = -10  # short cooldown
                if new_mode != self._yaw_mode:
                    self._yaw_mode = new_mode
                    self._yaw_mode_hold_t = HOLD_MIN_STEPS
                    if new_mode != "burst":
                        self._yaw_burst_dur_t = -10
            if self._yaw_mode == "burst":
                self._yaw_burst_dur_t += 1
                YAW_TRACK_CAP = 0.90
                if 0.15 < abs(vyaw) < 0.90:
                    vyaw = math.copysign(0.90, vyaw)
            else:
                self._yaw_burst_dur_t = max(self._yaw_burst_dur_t + 1, 0)
                # Raised low-mode cap from 0.22 → 0.35 so the robot can
                # actually arc through obstacles (vx/vyaw ≈ 0.8/0.35 →
                # turning radius ~2.3 m vs prior 3.6 m).  Still well below
                # the 0.50 rad/s region where the RL trot loses margin.
                YAW_TRACK_CAP = 0.42
                if abs(vyaw) < 0.35:
                    vyaw = max(-0.42, min(0.42, vyaw))
                else:
                    vyaw = math.copysign(0.42, vyaw)
            vyaw = max(-YAW_TRACK_CAP, min(YAW_TRACK_CAP, vyaw))

            # ── Roll-accumulation guard ───────────────────────────────────────
            # Sustained one-direction yaw at 0.7 rad/s builds sideways roll
            # within 15-20 s in the RL trot gait.  Track a slow EMA of signed
            # yaw; when it exceeds a threshold, force a counter-rotation
            # pulse for ~0.6 s to null the accumulated lean.
            ROLL_EMA_DECAY    = 0.985     # ~6.6 s time constant
            ROLL_TRIGGER_MAG  = 0.40      # trigger when EMA > 0.40
            COUNTER_PULSE_S   = 0.50
            COUNTER_MAG       = 0.13      # small sub-dead-zone counter (avoid disrupting track)

            self._forced_rest_t = max(0.0, self._forced_rest_t - dt)
            self._yaw_accum_ema = ROLL_EMA_DECAY * self._yaw_accum_ema \
                                  + (1.0 - ROLL_EMA_DECAY) * vyaw
            if self._forced_rest_t <= 0.0 and abs(self._yaw_accum_ema) > ROLL_TRIGGER_MAG:
                self._forced_rest_t = COUNTER_PULSE_S
                # Reset EMA so the next trigger only fires after fresh accumulation
                self._yaw_accum_ema = 0.0

            if self._forced_rest_t > 0.0:
                # Counter-rotate in the direction OPPOSITE to recent bias to
                # actively null accumulated roll.
                bias_sign = 1.0 if vyaw > 0 else -1.0
                vyaw = -bias_sign * COUNTER_MAG
                vx = MIN_VX_KEEPALIVE

            self._yaw_bb = 0
            self._turn_active_t = 0.0

            # ── Distance dead-band ────────────────────────────────────────────
            target_area = TARGET_AREA_FRAC * img_w * img_h
            dist_est    = DIST_CAL * math.sqrt(target_area / max(area, 1.0))
            dist_est    = max(0.3, min(10.0, dist_est))
            # Prefer ground-truth range in sim so vx tracks real stand-off, not
            # a mis-scaled area proxy that misreports range vs true world distance.
            d_band = (
                max(0.3, min(10.0, float(world_dist_m)))
                if world_dist_m is not None
                else dist_est
            )

            if d_band > TARGET_DIST_GOAL:
                vx       = KP_VX * (d_band - TARGET_DIST_GOAL)
                dist_cat = "FAR"    # still too far — advance
            elif d_band < TARGET_DIST_MIN_SAFE:
                vx       = 0.0
                dist_cat = "CLOSE"  # extremely close — hold (no backward)
            else:
                vx       = 0.0
                dist_cat = "OK"     # within (MIN_SAFE, GOAL] — goal met

            vx = max(-MAX_VX_BACK, min(MAX_VX, vx))

            # Only log on category transitions, with ±0.05 m hysteresis so that
            # world_dist_m jitter around exactly 1.000 m does not produce hundreds
            # of "OK → FAR → OK" lines per minute.  The vx proportional calculation
            # above is intentionally unchanged (no dead-band on control).
            _GOAL_HYST = 0.05   # m
            if self._last_dist_cat == "OK":
                log_cat = "FAR" if d_band > TARGET_DIST_GOAL + _GOAL_HYST else "OK"
            elif self._last_dist_cat == "FAR":
                log_cat = "OK"  if d_band < TARGET_DIST_GOAL - _GOAL_HYST else "FAR"
            else:
                log_cat = dist_cat   # CLOSE or initial state: no hysteresis needed
            if log_cat != self._last_dist_cat:
                _log(
                    f"dist: {self._last_dist_cat or '?'} → {dist_cat} "
                    f"(d_band={d_band:.2f}m  vis≈{dist_est:.2f}m)"
                )
                self._last_dist_cat = log_cat

            # Area overflow guard: ball filling camera → too close, stop advancing.
            # In simulation we have a true world distance (`world_dist_m`),
            # so prefer that — the camera's wide FOV can over-report area
            # and falsely trigger this stop while the ball is actually
            # several metres away.
            if world_dist_m is None and area > AREA_TOO_CLOSE and vx > 0.0:
                vx = 0.0
                if dist_cat != "CLOSE":
                    dist_cat = "CLOSE"

            # Always keep some forward intent when still FAR and visible.
            # User requirement: robot must keep trying to close toward 1 m.
            # If the ball is very off-centre we still reduce vx via yaw-coupling
            # below, but we do not hard-stop forward motion anymore.
            if (
                dist_cat == "FAR"
                and area <= AREA_TOO_CLOSE
                and vx <= 0.0
            ):
                vx = SEARCH_VX


            # Approach floor: while FAR and visible, always command enough vx for
            # the policy to actually translate.  At cmd < 0.3 the policy achieves
            # near-zero actual velocity (velocity probe: cmd 0.3 → 0.005 m/s).
            if vx > 0:
                vx = max(0.30, vx)

            # Yaw-coupled vx: avoid the (high-vx, mid-vyaw) fall corner.
            # Burst-yaw cap lifted from 0.50 → 0.65 (recent runs show 0% fall
            # rate at higher vx during 0.9 rad/s yaw bursts on this gait /
            # arena combination — the original 0.50 was a conservative
            # margin from a probe that no longer applies).
            avy = abs(vyaw)
            if avy < 0.15:
                vx = min(vx, MAX_VX)      # near-straight: full speed
            elif avy > 0.85:
                vx = max(vx, 0.25)
                vx = min(vx, 0.65)         # super-dead-zone: cap at 0.65
            elif avy <= 0.22:
                vx = max(vx, 0.25)
                vx = min(vx, 0.50)
            else:
                vx = max(vx, 0.20)
                vx = min(vx, 0.30)         # mid-yaw dead zone



            # Proximity guard — only when physically very close: limit yaw.
            # Use world_dist_m (ground truth) exclusively when available so that
            # area-estimation noise at image edges doesn't falsely trigger this.
            d_near = (
                float(world_dist_m)
                if world_dist_m is not None
                else dist_est
            )
            if d_near < PROXIMITY_YAW_CAP_M:
                if vyaw != 0.0:
                    vyaw = math.copysign(min(abs(vyaw), PROXIMITY_YAW_LIMIT), vyaw)
                vx = max(vx, -MAX_VX_BACK)

            vyaw = max(-MAX_VYAW, min(MAX_VYAW, vyaw))
            vx   = max(-MAX_VX_BACK, min(MAX_VX, vx))

            # Keep-alive: if both commands are nearly zero the RL policy enters
            # a neutral-balance regime it was not trained for, causing slow
            # postural drift and eventual falls (observed in every log fall:
            # vx=0, vyaw=0 for 19–43 steps before fell=1).
            if abs(vx) < 0.05 and abs(vyaw) < 0.05:
                vx = MIN_VX_KEEPALIVE

            self._last_vx    = vx
            self._last_vyaw  = vyaw

            # Periodic tracking log
            self._track_log_ctr += 1
            if self._track_log_ctr % _LOG_EVERY_TRACK == 0:
                turn_ch = "L" if vyaw > 0.05 else ("R" if vyaw < -0.05 else "·")
                _log(
                    f"TRACKING  d_band≈{d_band:.2f}m[{dist_cat}] vis≈{dist_est:.2f}m  "
                    f"cx={cx}/{img_w}(err={x_err:+.0f}px)  "
                    f"vx={vx:+.2f}  vyaw={vyaw:+.2f}[{turn_ch}]  "
                    f"conf={conf:.2f}  area={int(area)}px²"
                )

            return vx, vyaw

        # ── Ball not visible ──────────────────────────────────────────────────
        if self._state == TRACKING:
            self._invisible_ctr += 1

            if self._invisible_ctr < INVISIBLE_THRESH:
                # Brief invisibility: actively hunt in the direction the ball was last
                # drifting.  Use bang-bang pulses so we cover angular ground without
                # sustaining a dead-zone command.  This keeps the scan going without
                # the state-machine churn of RECOVERING.
                return self._bang_bang_search_step(dt)

            # Confirmed loss after hysteresis window
            self._last_dist_cat = ""
            self._search_bang_phase_t = 0.0
            self._search_bang_on      = True
            dir_label = "LEFT(+)" if self._last_turn_dir > 0 else "RIGHT(-)"
            _log(
                f"→ RECOVERING  (invisible ×{self._invisible_ctr}, "
                f"last seen {'RIGHT' if self._last_turn_dir < 0 else 'LEFT'}, "
                f"spinning {dir_label})"
            )
            self._state        = RECOVERING
            self._recover_time = 0.0

        if self._state == RECOVERING:
            self._recover_time += dt
            if self._recover_time <= RECOVER_MAX_S:
                # Bang-bang search: short YAW_HIGH pulse, then a brief rest at vyaw=0.
                # Avoids sustaining a dead-zone yaw command (0.3-0.9 rad/s causes roll-up).
                return self._bang_bang_search_step(dt)
            # Timed out → reverse sweep
            self._last_turn_dir      = -self._last_turn_dir
            self._search_sweep_time  = 0.0   # start fresh sweep timer
            self._search_bang_phase_t = 0.0
            self._search_bang_on     = True
            _log(
                f"→ SEARCHING  (recovery timed out; sweeping "
                f"{'LEFT(+)' if self._last_turn_dir > 0 else 'RIGHT(-)'})"
            )
            self._state = SEARCHING

        # ── SEARCHING ────────────────────────────────────────────────────────
        self._search_sweep_time += dt
        if self._search_sweep_time >= SEARCH_SWEEP_S:
            self._last_turn_dir     = -self._last_turn_dir
            self._search_sweep_time = 0.0
            self._search_bang_phase_t = 0.0
            self._search_bang_on    = True
            _log(
                f"SEARCHING  sweep reversed → "
                f"{'LEFT(+)' if self._last_turn_dir > 0 else 'RIGHT(-)'}"
            )
        return self._bang_bang_search_step(dt)

    def _bang_bang_search_step(self, dt: float) -> tuple[float, float]:
        """Bang-bang search with YAW_HIGH pulses and long rest periods.

        Short YAW_HIGH=0.9 ON pulses cross the dead zone fast and produce
        real rotation; the long OFF phase lets accumulated roll dissipate.
        Direction flips every SEARCH_DIR_FLIP_S so one-directional failure
        is covered by the other.
        """
        SEARCH_YAW_PULSE  = 0.90
        SEARCH_PULSE_ON   = 0.30
        SEARCH_PULSE_OFF  = 1.20    # long rest — roll fully damps

        self._search_dir_t += dt
        if self._search_dir_t >= SEARCH_DIR_FLIP_S:
            self._last_turn_dir = -self._last_turn_dir
            self._search_dir_t = 0.0

        self._search_bang_phase_t += dt
        if self._search_bang_on:
            if self._search_bang_phase_t >= SEARCH_PULSE_ON:
                self._search_bang_on = False
                self._search_bang_phase_t = 0.0
        else:
            if self._search_bang_phase_t >= SEARCH_PULSE_OFF:
                self._search_bang_on = True
                self._search_bang_phase_t = 0.0

        if self._search_bang_on:
            return SEARCH_VX, -self._last_turn_dir * SEARCH_YAW_PULSE
        else:
            return MIN_VX_KEEPALIVE, 0.0

    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.__init__()
