# Based on MuJoCo Playground's Go1 joystick task:
# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Modified for the red panda robot; every change is marked "RED PANDA:".
"""Walking (joystick) environment for the red panda.

Ported from MuJoCo Playground's Go1 joystick task
(mujoco_playground/_src/locomotion/go1/joystick.py).
Every change from Go1 is marked "RED PANDA:".

Reward pass 1: Go1's 16 terms and weights, with only size fixes (command
speeds, sigma, foot height, torque/energy weights). Ones still undecided are
marked "TODO(size fix)".
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

import terrain  # sim/terrain.py (same folder)

XML_PATH = Path(__file__).resolve().parent / "red_panda.xml"

# RED PANDA: names instead of hard-coded index ranges, because qpos and ctrl
# use different orders (tail and neck are swapped between them).
LEGS = ["fl", "fr", "bl", "br"]
LEG_JOINTS = [f"{leg}_{j}" for leg in LEGS for j in ["hip_side", "hip", "knee"]]
TAIL_JOINTS = ["tail_wag", "tail_lift", "tail_wag2"]
NECK_JOINTS = ["neck_turn", "neck_nod", "head_tilt"]
FEET = [f"{leg}_foot" for leg in LEGS]


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      action_scale=0.5,
      soft_joint_pos_limit_factor=0.95,
      # RED PANDA: no Kp/Kd here. Go1 overwrites the XML's servo gains with its
      # own (Kp 35); we keep the XML values (placeholders until BAM).
      # RED PANDA: policy input switch for the blind vs. sees-speed comparison.
      policy_sees_body_speed=False,
      # RED PANDA (run 14): robot shape, applied when the model loads (build_spec).
      # leg_scale shortens both leg segments (Pupper = 12.35 + 11.5 cm); the foot
      # ball keeps its size. stance_knee_deg > 0 rewrites the home pose: that knee
      # bend with each foot straight under its hip (0 = the XML's home, knee 80).
      # Run 15 default = run 14b: legs x0.7 (8.64 + 8.05 cm), home knee 45 deg with feet
      # under the hips (torso 16.3 cm, red panda height). Knee RMS current 108% -> 69%
      # of rated vs run 13 (Pupper legs, knee ~93 deg crouch). CAD may adjust lengths.
      model_config=config_dict.create(
          leg_scale=0.7,
          stance_knee_deg=45.0,   # runs 14-19: level torso, every knee 45 deg
          # Run 20: home pose = the walking posture. Torso centre stance_height up,
          # tilted stance_pitch_deg nose-down, each foot on the floor under its hip;
          # the knees follow (front ~61, hind ~48 deg). Replaces stance_knee_deg when
          # stance_height > 0. Runs 14-19's level 45/45 home fought the 3 deg slant
          # target, and the hind legs ended up nearly straight (30 deg) and skimmed.
          stance_height=0.158,
          stance_pitch_deg=2.0,
      ),
      # RED PANDA (run 10): domain randomization. Each training robot gets its own
      # world, so the policy can't rely on exact numbers. Used by tools/train_walk.py
      # through make_domain_randomizer(). The stress test (run 9's policy) showed
      # friction, mass, and center of mass barely matter; they're cheap insurance.
      randomize_config=config_dict.create(
          enable=True,
          friction=[0.3, 1.0],     # floor AND feet (MuJoCo uses the higher of the two)
          stiffness=[0.7, 1.3],    # servo kp scale, all servos together (real kp unknown until BAM)
          mass_scale=[0.9, 1.1],   # whole robot: 1.62-1.98 kg
          com_shift=0.02,          # torso center of mass +/- 2 cm in x and y
      ),
      # RED PANDA: end the episode if the torso center drops below this (m).
      min_torso_height=0.10,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      # RED PANDA: the tail is animated while walking (not the policy), with
      # random sway each episode so the policy learns to balance against it.
      tail_config=config_dict.create(
          # Run 2: calmer tail (run 1 used 0-30 for both; the tip whipped +/-31 deg).
          wag_amp_deg=[0.0, 15.0],     # side-to-side size
          wag_freq_hz=[0.3, 1.5],      # sways per second
          lift_center_deg=[-10.0, 40.0],
          lift_amp_deg=[0.0, 15.0],
          lift_min_deg=-25.0,          # below ~-40 deg the tail hits the floor
          wag2_amp_deg=[0.0, 10.0],    # second joint, lags behind (follow-through)
          wag2_lag_rad=[0.3, 1.5],
          # Run 5 (gait clock on): one sway per stride, opposite the body's roll
          # (counterbalance), tail held about level. Size randomized for robustness.
          gait_wag_amp_deg=[4.0, 10.0],
          gait_lift_deg=[-8.0, 2.0],
          # Run 6: +1 = tail swings toward the side whose feet are in the air, i.e.
          # opposite the body's new lean toward its supporting feet (counterbalance).
          # Run 5 used -1 (opposite the passive roll, before the body had a lean target).
          gait_wag_sign=1.0,
      ),
      # RED PANDA: the head "looks around" while walking (animation, not the
      # policy): each neck joint drifts slowly on its own random rhythm.
      # Set enable=False to hold the head still (e.g. to debug a bad run).
      neck_config=config_dict.create(
          enable=True,
          amp_deg=[30.0, 20.0, 20.0],  # max swing: turn, nod, tilt
          # Run 12: up to 1.5 Hz (quick glances). Was 0.1-0.5 Hz (one look every 2-10 s).
          freq_hz=[0.1, 1.5],
          # Run 5 (gait clock on): head low and forward, mostly steady, with a slight
          # nod each time a front foot lands; small random turn/tilt drift.
          gait_nod_base_deg=8.0,       # + = head down
          # Run 6: no scripted nod (run 5's 3 deg nod, 2.5 times/s, plus servo overshoot
          # and torso pitch made the head bob 16 deg). Was 3.0.
          gait_nod_amp_deg=0.0,
          # Run 6: head stabilization. Keep the head's pitch and roll steady in the
          # WORLD by countering the torso's tilt (measured like the real IMU: gravity).
          gait_head_stabilize=True,
          # Run 12: personality-sized head motion while walking (was 10, 5) plus random
          # nods on top of the steady 8 deg. Run 11a's head test: turn 30 / tilt 20 at
          # 2 Hz with 10 deg nods didn't disturb the walk at all, so this is cheap insurance.
          gait_turn_tilt_amp_deg=[30.0, 20.0],
          gait_nod_drift_deg=10.0,
      ),
      # RED PANDA (run 5): the user's red panda walk spec. A clock ticks through each
      # stride; each foot should be down for `duty` of the cycle, starting at its
      # touchdown offset. Lateral-sequence walk with lateral couplets: LH -> LF -> RH -> RF.
      gait_config=config_dict.create(
          enable=True,
          # Run 26: 1.25 -> 1.0 (longer steps: +/-4.5 cm at 0.12 m/s; knees stay >= 31 deg bent).
          freq_hz=1.0,                     # strides per second (spec: 1.0-1.6, never 3-4)
          # Run 24a: 0.70 -> 0.60. More swing time (0.32 s) lets the hind feet physically
          # reach ~2.4 cm (leg-in-air test); at 0.70 the servos top out at ~1.6 cm.
          # Run 26: 0.60 -> 0.75 with even offsets: 3 feet down at every moment, so no
          # front lift-off stands on a same-side pair (24a/25b propped a hind foot mid-swing
          # at ~41% of body weight there). 0.25 s swings: servos lift a hind foot 1.5-1.8 cm.
          duty=0.75,                       # each foot down 75% of the cycle
          # touchdown offsets in FEET order (fl, fr, bl, br) = (LF, RF, LH, RH);
          # spec: LH 0.00, LF 0.20, RH 0.50, RF 0.70 (runs 5-25). Run 26: evenly spaced, so
          # duty 0.75 never leaves fewer than 3 feet down (tools/eval/schedule_table.py).
          offsets=[0.25, 0.75, 0.00, 0.50],
          base_height=0.158,  # torso target = the home stance (run 20; 0.163 for runs 14b-19)
          # Run 6: torso choreography. Lean (roll) a few degrees toward the side whose
          # feet are down (weight shift onto the supporting pair = the waddle), once per
          # stride, and carry the head end slightly low. No twist: a rigid torso can't
          # yaw without turning (that would need the side-bend spine).
          roll_amp_deg=3.0,
          # Run 26: phase where the lean target's sine crosses zero going left (lean left
          # peaks a quarter stride later). 0.20 fits the offsets above (right feet in the
          # air around 0.45); even offsets [0.25, 0.75, 0, 0.5] need 0.25 (peak at 0.5).
          roll_phase=0.25,
          pitch_deg=2.0,                   # + = nose down (run 20: = the home pose's slant; was 3.0)
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base reward.
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              # Other.
              dof_pos_limits=-1.0,
              pose=0.5,
              # Other.
              termination=-1.0,
              stand_still=-1.0,
              # Regularization.
              # RED PANDA: Go1 x17. Standing torque cost measured in sim: Go1 45.5,
              # ours 2.6, so x17 keeps the same "bill" (-0.009 per step).
              # Run 13: x3 (was -0.0035). Run 12's knee ran at 112% of rated current
              # (heat ~ current^2), yet torque cost only -38 per episode vs +809 for
              # speed tracking. (Later found: half the knee load is jitter, not effort.)
              torques=-0.0105,
              # Run 16: x15 (was -0.01). Smoothness was nearly free (-1.9/episode vs +874
              # tracking), so the policy twitched its targets ~3 deg every 20 ms (knee
              # jitter, double touchdowns). Run 15b's -0.5 fixed that but drifted the
              # footfall timing and stopped sideways steps; -0.15 is the gentler middle.
              action_rate=-0.15,
              # RED PANDA: Go1 x7. Standing-matched would be x17 (-0.017), but under
              # early-training jitter our light legs make energy 2.4x Go1's share;
              # too high risks "stand still" early, so match that. If run 1 freezes,
              # suspect this first. Compare reward/energy's share with Go1's after run 1.
              energy=-0.007,
              # Feet.
              # Run 18: off (was -2.0, Go1's). It charges any foot moving horizontally below
              # max_foot_height, so it rewarded "hover, then drop straight down" (a stomp)
              # and duplicated the swing arc, which already says how high the foot goes.
              feet_clearance=0.0,
              feet_height=-0.2,
              feet_slip=-0.1,
              # RED PANDA (run 4): Go1 0.1. At 0.1 this term was ~0.1% of the total
              # (-1 per episode vs. pose +480), too small to change the gait. 2.0 makes
              # it about the size of torques now that pose no longer fights long steps.
              feet_air_time=0.0,  # run 5: off. The gait clock sets step timing now
              # (spec swing time 0.3/1.25 = 0.24 s would be fined by the 0.25 s threshold).
              # RED PANDA (run 5): the walk spec.
              # Run 8: 1.0 -> 1.5. At 1.0 the right front foot landing 0.2 stride late cost only
              # ~1 point per episode, so the lean won (front duty 0.57 instead of 0.70).
              gait_schedule=1.5,   # fraction of feet matching the footfall chart (0-1), per step
              # Run 6: exp(-(lean error^2)/sigma) toward the roll/pitch targets (0-1).
              # Same size as gait_schedule's pull; orientation (-5) still fines big tilts.
              body_sway=0.5,
              # Run 7: while a foot is scheduled to be in the air, its height should
              # follow an arc (lift off -> peak mid-swing -> land). (h - target)^2 summed
              # over swinging feet. Closes the loophole where a foot hovering 2 mm counts
              # as "up", and prices skimming hind feet (~-126/episode for run 6's gait).
              # Run 12: -300 -> -450. Under randomization the hind feet skimmed again at
              # -300 (run 10); -600 brought the lift back but bounced more and pushed the
              # knee peak to 2.31 N*m (run 11a). Rarer pushes need less push to lift.
              # Run 16: -450 -> -930. The cost is (height - arc)^2, so shrinking the lift
              # 3.6 -> 2.5 cm (run 14b legs) halved the price of skimming: x(3.6/2.5)^2.
              # Hind feet lifted only 0.1-0.4 cm in runs 14b-15c.
              swing_height=-930.0,
              # Run 16: x25 (was -20). Run 14b sat 2 cm below its 16.3 cm target for -9 per
              # episode; at -500 that's -200. Run 15c: 15.8 cm, front double touchdowns gone.
              base_height=-500.0,  # (torso height - target)^2
              joint_acc=-2.5e-7,   # sum of leg joint acceleration^2: anti-jitter (legged_gym's value)
              # Run 15a: planted feet under their thigh joints, (fore/aft^2 + sideways^2).
              # 0 = off (set per experiment). Run 14b's gait would pay ~-250/episode at
              # -30 (front feet 6.6 cm ahead in stance, hind 2.9 cm out); a centred gait ~-32.
              foot_under_hip=-30.0,  # run 16 (run 15a: front feet +7.8 -> +4.1 cm, knee 62% of rated)
              # Run 17: downward foot speed^2 in the last cm before touchdown (0 = off).
              # Run 16's hind feet landed hard enough to bounce the rear end: the other
              # hind foot hopped 1 mm for ~50 ms at every hind landing (21+15 taps in 10 s).
              # Run 17: -15 -> run 16's gait would pay ~-190/episode (hind feet land at
              # 0.18-0.21 m/s median, 1.7x the front); landing at ~0.1 m/s pays ~-40.
              # Run 18: -15 -> -5. With the bell arc, following the arc already lands
              # gently; at -15 even a perfect bell paid ~-180/episode and skimming beat it
              # by only ~50 (reward audit). At -5: bell -60, skim -206, stomp -511.
              soft_landing=-5.0,
              # Run 22b: knee should fold during swing (see _swing_knee_err2). 0 = off.
              swing_knee=0.0,
              # Run 23c: clearance floor, once per swing (see swing_floor_lift). 0 = off.
              # Run 24a default: sized so 20b's skimming hind feet pay what swing_height
              # charges them. With bumps and no knee term: hind 1.5-1.7 cm, knee 72% of rated.
              # Run 26: -330 with swing_floor_shape "linear" (0.3 cm short = 1.0 per swing).
              swing_floor=-330.0,
              # Run 25b: weight still on a foot around its scheduled lift-off (see
              # unload_before_s / unload_after_s). 24a's front feet held ~50% of the body
              # at lift-off and 37-41% 20 ms later, so the knee hit its torque limit and
              # the foot left ~50-60 ms late (hind feet unload in time). 24a would pay
              # ~37/episode before the weight (front 18.5 each, hind ~0). 0 = off.
              unload=0.0,
          ),
          # RED PANDA: two sigmas instead of Go1's one (0.25). Our walk speeds are
          # 4x slower than Go1's (error^2 16x smaller) -> 0.25/16; turn speeds are
          # 2.4x slower -> 0.25/5.8. Grades "off by a bit" like Go1 does.
          tracking_sigma_lin=0.016,
          tracking_sigma_ang=0.043,
          # RED PANDA: foot-center height target mid-step (Go1: 0.1). 4.0 cm =
          # 3 cm of lift + 0.97 cm foot radius: a low, long red panda step that's
          # quick for slow servos. (Go1-proportional would be 4.7: 84% of max lift.)
          # Run 5: spec says lift ~15-20% of leg length (24 cm) = 3.6-4.8 cm -> 3.6 cm
          # lift + 0.97 cm foot radius. (Runs 1-4 used 0.04.)
          max_foot_height=0.0277,  # run 26: 1.8 cm lift + 0.97 cm foot radius (14b-25: 0.035)
          body_sway_sigma=0.0012,  # (2 deg)^2 in rad^2: being flat when asked for 3 deg scores 0.1
          swing_lift=0.018,         # arc peak above resting. Run 26: what the servos reach in a 0.25 s swing (14b-25: 0.025)
          # Run 18: swing target shape. "bell" = lift x sin^2: zero vertical speed at
          # touchdown, so following the arc IS a soft landing (the sine arc demanded
          # ~0.32 m/s and fought soft_landing; run 17's hind feet skimmed to escape).
          swing_arc="bell",
          # Run 21: extra swing_height price for the hind feet only. A hind lift needs
          # 47 deg of knee fold vs 40 front, so at x1 skimming was the cheaper deal.
          swing_hind_mult=1.0,
          # Run 22b: extra knee bend at mid-swing over the home pose, (front, hind) deg.
          # IK: 28 / 32 deg lifts the foot 2.5 cm straight up from the home stance.
          swing_knee_fold_deg=[28.0, 32.0],
          # Run 23: "home" (22b: target = home knee + fold) or "liftoff" (target = the knee's
          # angle when this swing began + fold; crouching earns nothing). IK at 0.12 m/s:
          # lift-off (foot 4.8 cm behind the hip) to 2.5 cm up under the hip = 40 / 47 deg.
          swing_knee_ref="home",
          swing_knee_fold_liftoff_deg=[40.0, 47.0],
          # Run 23c: every swing should peak at least this high above resting (0 cost above).
          swing_floor_lift=0.012,   # run 26: below the servo max (1.5-1.8 cm) so it's reachable (23c-25: 0.015)
          # Run 26: "squared" (runs 23c-25: (floor - peak)^2, so a small miss is nearly
          # free; 25a's hind feet settled 0.5 cm under) or "linear" (floor - peak: the
          # first mm under costs as much as any other; a hard floor). Linear -330: 0.3 cm
          # short = 1.0 per swing, the price squared -22000 charged for 0.7 cm.
          swing_floor_shape="linear",
          # Run 26: where a planted foot "should" be fore/aft for foot_under_hip.
          # "hip" (runs 15a-25): under its thigh joint, so every step's travel costs and
          # long steps cost more. "clock": where the gait clock puts it, sliding from
          # +travel/2 at touchdown to -travel/2 at lift-off (travel = commanded forward
          # speed x stance time). Long centred steps are free; a stride shifted forward
          # (run 14b's front feet landing 7.8 cm ahead) still costs. Sideways: same.
          foot_under_hip_ref="clock",
          # Run 25b: unload window by the gait clock. The price ramps 0 -> 1 over the last
          # unload_before_s of stance, then stays 1 for the first unload_after_s of swing
          # (a foot that waits keeps paying; the rest of swing is free, so early hard
          # landings stay soft_landing's job).
          unload_before_s=0.06,
          unload_after_s=0.10,
          foot_rest_height=0.0097,  # foot-center height when planted (1 cm ball, sunk 0.3 mm)
          # RED PANDA (run 2): Go1 rewards air time above 0.1 s per step. Run 1's quick
          # steps (~0.19 s in the air) already beat that, so nothing pushed for longer
          # steps. 0.25 s makes quick little steps cost and longer swings pay.
          air_time_threshold=0.25,
          # RED PANDA (run 3): pose's per-joint weights (hip_side, hip, knee). Go1 uses
          # 1.0, 1.0, 0.1. A +/-25 deg hip swing cost ~135 pose points per episode, so
          # the policy kept steps short. Hip 0.2 makes long steps cheap; hip_side stays
          # 1.0 so the legs don't splay sideways.
          pose_weights=[1.0, 0.2, 0.1],
      ),
      # RED PANDA (run 10): pushes on. The push force uses the whole robot's mass, so a
      # kick of k changes the robot's speed by k / pi m/s (checked in sim). Go1's max
      # (3.0) would be 0.95 m/s. Run 9 already survives 0.45 m/s and starts falling
      # around 0.6-0.7 m/s; bigger shoves are the get-up policy's job.
      pert_config=config_dict.create(
          enable=True,
          velocity_kick=[0.0, 0.6 * 3.14159],  # up to 0.6 m/s speed change
          kick_durations=[0.05, 0.2],
          # Run 12: every 1-10 s (Go1: 1-3 s). Shoves every 1-3 s made training a
          # constantly shaking bus and the hind feet skimmed (run 10); 1-10 s tripled the
          # lift and was the most robust (run 11d). Short gaps still happen sometimes.
          kick_wait_times=[1.0, 10.0],
      ),
      # RED PANDA (run 9): servo model, step 1. A real servo twists less hard the
      # faster it turns (its spinning motor makes a voltage that fights the battery).
      # Straight line: full stall torque (the actuator forcerange, 2.9 N*m) at 0 rad/s,
      # zero at the no-load speed (STS3215 C018 at 12 V: 4.7 rad/s). Braking (twisting
      # against the motion) is allowed up to stall. BAM replaces this with the measured
      # curve. Configs saved before run 9 have no servo_config, so old runs replay as-is.
      servo_config=config_dict.create(
          torque_speed_limit=True,
          no_load_speed=4.7,    # rad/s at the output horn
          # Run 10: random motor delay per episode (sensors read -> servos act).
          # Estimate: ~5-10 ms for a careful build, ~25-30 ms for a slow one; the
          # winter-break bus timing test will narrow this.
          delay_ms=[0.0, 40.0],
          # Run 13: each leg servo's "0" is off by a fixed random angle per episode
          # (horn teeth + calibration error). The policy reads angle + offset and the
          # servo drives the joint to target - offset, like a real miscalibrated servo.
          # Run 12 untrained: +/-2 deg -> 0/16 falls, 0.19 -> 0.18 m/s.
          joint_offset_deg=2.0,
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          # RED PANDA: forward / sideways / turn max (Go1: 1.5, 0.8, 1.2).
          # Forward 0.4 m/s: servo-speed estimate (0.3-0.5). Sideways 0.2:
          # Go1's sideways/forward ratio (53%). Turn 0.5 rad/s: Go1's turn as a
          # fraction of its fastest possible turn (18%), scaled to our feet being
          # 0.14 m from the center. Measure the real top speeds after the first run.
          # Run 5: speed = stride x strides/s. At 1.25 strides/s a moderate ~16 cm
          # stride gives 0.2 m/s (0.4 would need 32 cm; the legs max out ~20 cm).
          # Sideways 0.1: sidestepping doesn't suit a lateral-sequence walk. (Runs 1-4: 0.4, 0.2, 0.5.)
          # Run 14b: x0.7 legs -> x0.7 forward/sideways (shorter strides); turning
          # depends on the body width, which didn't change. (Runs 5-14a: 0.2, 0.1, 0.5.)
          a=[0.14, 0.07, 0.5],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      # Terrain (Tier 2): lumpy rug-like bumps instead of the flat plane. See
      # sim/terrain.py for the tile layout. Off = the flat plane, exactly as runs 1-21.
      terrain_config=config_dict.create(
          enable=True,             # run 24a default (False = the flat plane of runs 1-21)
          tile_size=3.0,           # m; a 20 s episode walks <= 2.8 m
          n_levels=10,             # rows along x: level i has lumps up to max_height*(i+1)/10
          n_cols=8,                # columns along y: terrain types
          flat_cols=[3, 7],        # 2 of 8 columns flat -> 75% lumpy / 25% flat
          border=3.0,              # m of flat ground around the tiles
          resolution=0.01,         # m per grid cell
          max_height=0.015,        # m: hardest row's tallest lumps (hind feet reach ~2.4 cm at duty 0.60)
          lump_radius=[0.02, 0.04],  # m: lumps 4-8 cm wide
          lump_cover=0.6,          # lump discs cover ~60% of a tile (overlaps counted)
          seed=0,                  # same map every run
          spawn_level=-1,          # -1 = random row per robot; >= 0 = always this row (evals)
          spawn_col=-1,            # -1 = random column; >= 0 = always this column (evals)
      ),
      # RED PANDA: "jax" runs on the Mac; use "warp" on an NVIDIA GPU.
      impl="jax",
      naconmax=4 * 8192,
      njmax=40,
  )


def _bisect(f, lo, hi, iters=60):
  flo = f(lo)
  for _ in range(iters):
    mid = 0.5 * (lo + hi)
    fm = f(mid)
    if (fm > 0) == (flo > 0):
      lo, flo = mid, fm
    else:
      hi = mid
  return 0.5 * (lo + hi)


def build_spec(config) -> mujoco.MjSpec:
  """RED PANDA (run 14): the robot model with model_config (and terrain) applied.

  Returns an MjSpec (the env compiles it; the viewer adds a floor texture first).
  """
  spec = _robot_spec(config)
  tc = config.get("terrain_config", None)
  if tc is not None and tc.enable:
    add_terrain(spec, tc)
  if config.get("solid_body", False):
    make_body_solid(spec)
  return spec


SOLID_BODIES = ["head", "neck_1", "neck_2"] + [
    f"{leg}_{part}" for leg in ["fl", "fr", "bl", "br"] for part in ["hip", "upper", "lower"]]


def make_body_solid(spec: mujoco.MjSpec) -> list:
  """Head, neck, hip blocks, thighs and shins collide with the floor (not with each other).

  The base model only lets the foot balls, torso and tail touch the floor; everything else
  passed through it, so a face-plant sank the head into the floor for free (rr_dn1 era).
  Floor contype 1 + conaffinity 1 here = touches the floor only (all robot geoms have
  contype 0). Unnamed geoms get names "<body>_g<i>" so contact sensors can find them.
  Returns the names of the geoms made solid (feet excluded: already solid)."""
  names = []
  for b in SOLID_BODIES:
    body = spec.body(b)
    for i, g in enumerate(list(body.geoms)):
      if g.name.endswith("_foot"):
        continue
      if b.endswith("_lower") and g.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # The shin capsule ends at the foot ball's centre with the same radius, so it
        # would touch whenever the foot does. Keep it visual only and add an invisible
        # collision copy that stops SHIN_GAP above the foot: "shin on the floor" then
        # means the shin itself is lying down.
        g.name = g.name or f"{b}_visual"
        top, bottom = np.array(g.fromto[:3]), np.array(g.fromto[3:])
        length = np.linalg.norm(bottom - top)
        end = bottom + (top - bottom) * (SHIN_GAP / length)
        c = body.add_geom(name=f"{b}_g{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=np.concatenate([top, end]), size=g.size, mass=0.0,
                          contype=0, conaffinity=1, group=3, rgba=[0, 0, 0, 0])
        names.append(c.name)
        continue
      if not g.name:
        g.name = f"{b}_g{i}"
      g.conaffinity = 1
      names.append(g.name)
  return names


SHIN_GAP = 0.03  # m: the collision shin stops this far above the foot ball's centre


def solid_geom_names(spec: mujoco.MjSpec) -> list:
  """Names of the geoms make_body_solid made solid (to hang contact sensors on)."""
  return [g.name for b in SOLID_BODIES for g in spec.body(b).geoms
          if g.conaffinity == 1 and not g.name.endswith("_foot")]


def add_terrain(spec: mujoco.MjSpec, tc) -> None:
  """Terrain: swaps the flat floor plane for a lumpy height field (sim/terrain.py).

  Keeps the geom's name "floor", so the foot contact sensors and friction
  randomization work unchanged.
  """
  heights, half_x, half_y = terrain.make_heights(tc)
  top = max(float(heights.max()), 1e-3)
  nrow, ncol = heights.shape
  spec.add_hfield(
      name="terrain", nrow=nrow, ncol=ncol,
      size=[half_x, half_y, top, 0.05],    # x/y half-size, max height, 5 cm of solid base
      userdata=(heights / top).flatten().tolist())
  floor = spec.geom("floor")
  floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
  floor.hfieldname = "terrain"


def _robot_spec(config) -> mujoco.MjSpec:
  """The robot with model_config applied, standing on the flat floor (run 14)."""
  spec = mujoco.MjSpec.from_file(XML_PATH.as_posix())
  mc = config.get("model_config", None)
  if mc is None:  # configs from runs 1-13
    return spec
  k = mc.leg_scale
  if k != 1.0:
    for leg in LEGS:
      for body in (f"{leg}_upper", f"{leg}_lower"):
        for g in spec.body(body).geoms:
          if not np.isnan(g.fromto[0]):        # leg capsule: scale both ends
            g.fromto = np.array(g.fromto) * k
          elif g.name == f"{leg}_foot":        # foot ball: move it, same size
            g.pos = np.array(g.pos) * k
      spec.body(f"{leg}_lower").pos = np.array(spec.body(f"{leg}_lower").pos) * k
      spec.site(f"{leg}_foot").pos = np.array(spec.site(f"{leg}_foot").pos) * k
  if mc.stance_knee_deg > 0:
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    d.qpos[2], d.qpos[3:7] = 1.0, [1.0, 0.0, 0.0, 0.0]   # upright, in the air
    ctrl = np.array(m.key_ctrl[0])
    for leg in LEGS:
      knee_sign = -1.0 if leg.startswith("f") else 1.0     # front knees bend negative
      adr = {j: m.jnt_qposadr[m.joint(f"{leg}_{j}").id] for j in ("hip_side", "hip", "knee")}
      d.qpos[adr["hip_side"]] = 0.0
      d.qpos[adr["knee"]] = knee_sign * np.deg2rad(mc.stance_knee_deg)
      foot, hip = m.site(f"{leg}_foot").id, m.body(f"{leg}_hip").id

      def foot_ahead(h, adr=adr, foot=foot, hip=hip):
        d.qpos[adr["hip"]] = h
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[foot][0] - d.xpos[hip][0]

      d.qpos[adr["hip"]] = _bisect(foot_ahead, -1.6, 1.6)  # foot straight under the hip
    mujoco.mj_kinematics(m, d)
    radius = m.geom_size[m.geom("fl_foot").id][0]
    lowest = min(d.site_xpos[m.site(f"{leg}_foot").id][2] for leg in LEGS)
    d.qpos[2] = 1.0 - lowest + radius                       # feet just touching
    for leg in LEGS:
      for j in ("hip_side", "hip", "knee"):
        ctrl[m.actuator(f"{leg}_{j}").id] = d.qpos[m.jnt_qposadr[m.joint(f"{leg}_{j}").id]]
    home = next(key for key in spec.keys if key.name == "home")
    home.qpos, home.ctrl = np.array(d.qpos), ctrl
  if mc.get("stance_height", 0.0) > 0:  # run 20: height + nose-down slant, knees follow
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    pitch = np.deg2rad(mc.stance_pitch_deg)
    d.qpos[2] = mc.stance_height
    d.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]  # about y: + = nose down
    radius = m.geom_size[m.geom("fl_foot").id][0]
    ctrl = np.array(m.key_ctrl[0])
    for leg in LEGS:
      knee_sign = -1.0 if leg.startswith("f") else 1.0
      adr = {j: m.jnt_qposadr[m.joint(f"{leg}_{j}").id] for j in ("hip_side", "hip", "knee")}
      d.qpos[adr["hip_side"]] = 0.0
      foot, hip = m.site(f"{leg}_foot").id, m.body(f"{leg}_hip").id

      def place(h, kb, adr=adr):
        d.qpos[adr["hip"]], d.qpos[adr["knee"]] = h, knee_sign * kb
        mujoco.mj_kinematics(m, d)

      def hip_under(kb, foot=foot, hip=hip):
        def ahead(h):
          place(h, kb)
          return d.site_xpos[foot][0] - d.xpos[hip][0]
        grid = np.linspace(-1.5, 1.5, 61)
        vals = [ahead(h) for h in grid]
        i = next(i for i in range(60) if (vals[i] > 0) != (vals[i + 1] > 0))
        return _bisect(ahead, grid[i], grid[i + 1])

      def above_floor(kb, foot=foot):  # foot height above touching, with the foot under the hip
        place(hip_under(kb), kb)
        return d.site_xpos[foot][2] - radius

      kb = _bisect(above_floor, 0.0, np.deg2rad(120.0))  # straight = lowest foot, bent = higher
      place(hip_under(kb), kb)
      for j in ("hip_side", "hip", "knee"):
        ctrl[m.actuator(f"{leg}_{j}").id] = d.qpos[adr[j]]
    home = next(key for key in spec.keys if key.name == "home")
    home.qpos, home.ctrl = np.array(d.qpos), ctrl
  return spec


def use_flat(config) -> None:
  """Evals: walk on the flat floor even if the run trained on terrain.

  Standard checks (lift, knee margin, hardware, stress, GIF) run on flat so runs
  compare fairly; terrain gets its own checks (terrain_eval.py, --terrain).
  """
  if "terrain_config" in config:
    config.terrain_config.enable = False


def use_terrain(config, level: int, col: int = 1) -> None:
  """Evals: put any run's config (even one from before terrain existed) on one tile.

  Column 1 is lumpy with lumpy neighbors on both sides; the flat columns are 3 and 7.
  Keeps a newer run's own map settings if it has them.
  """
  if "terrain_config" not in config:
    config.terrain_config = default_config().terrain_config
  tc = config.terrain_config
  tc.enable, tc.spawn_level, tc.spawn_col = True, level, col


def make_domain_randomizer(env):
  """RED PANDA (run 10): Go1's randomization idea with our own ranges and fields.

  Returns randomize(model, rng) for Brax PPO: one random world per training robot.
  """
  rc = env._config.randomize_config
  m = env.mj_model
  contact_geoms = np.array([m.geom("floor").id] + [m.geom(f).id for f in FEET])
  torso = m.body("torso").id

  def randomize(model, rng):
    @jax.vmap
    def one_world(rng):
      k1, k2, k3, k4 = jax.random.split(rng, 4)
      # Friction: floor and feet get the same value, so it really changes.
      mu = jax.random.uniform(k1, minval=rc.friction[0], maxval=rc.friction[1])
      friction = model.geom_friction.at[contact_geoms, 0].set(mu)
      # Servo stiffness: kp (twist per radian of error) scaled for every servo.
      s = jax.random.uniform(k2, minval=rc.stiffness[0], maxval=rc.stiffness[1])
      gain = model.actuator_gainprm.at[:, 0].multiply(s)
      bias = model.actuator_biasprm.at[:, 1].multiply(s)
      # Mass: the whole robot heavier or lighter together (inertia scales with it).
      f = jax.random.uniform(k3, minval=rc.mass_scale[0], maxval=rc.mass_scale[1])
      mass, inertia = model.body_mass * f, model.body_inertia * f
      # Center of mass: shift the torso's balance point sideways / front-back.
      d = jax.random.uniform(k4, (2,), minval=-rc.com_shift, maxval=rc.com_shift)
      ipos = model.body_ipos.at[torso, :2].add(d)
      return friction, gain, bias, mass, inertia, ipos

    friction, gain, bias, mass, inertia, ipos = one_world(rng)
    fields = dict(geom_friction=friction, actuator_gainprm=gain, actuator_biasprm=bias,
                  body_mass=mass, body_inertia=inertia, body_ipos=ipos)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({k: 0 for k in fields})
    return model.tree_replace(fields), in_axes

  return randomize


class RedPandaWalk(mjx_env.MjxEnv):
  """Track a joystick command with the legs; tail and neck animated."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._xml_path = XML_PATH.as_posix()
    self._mj_model = build_spec(self._config).compile()  # run 14: + model_config
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    m = self._mj_model
    home = m.keyframe("home")
    self._init_q = jp.array(home.qpos)
    self._init_ctrl = jp.array(home.ctrl)

    # RED PANDA: look up where each joint lives in qpos, qvel, and ctrl.
    def qpos_adr(names):
      return np.array([m.joint(n).qposadr[0] for n in names])

    def qvel_adr(names):
      return np.array([m.joint(n).dofadr[0] for n in names])

    def act_ids(names):
      return np.array([m.actuator(n).id for n in names])

    self._leg_qpos = qpos_adr(LEG_JOINTS)
    self._leg_qvel = qvel_adr(LEG_JOINTS)
    self._leg_act = act_ids(LEG_JOINTS)
    self._tail_qpos = qpos_adr(TAIL_JOINTS)
    self._tail_qvel = qvel_adr(TAIL_JOINTS)
    self._tail_act = act_ids(TAIL_JOINTS)
    self._neck_qpos = qpos_adr(NECK_JOINTS)
    self._neck_qvel = qvel_adr(NECK_JOINTS)
    self._neck_act = act_ids(NECK_JOINTS)

    self._default_pose = jp.array(home.qpos[self._leg_qpos])  # legs only

    # Terrain (Tier 2): the height grid's shape, for looking up the ground under
    # a foot. The grid itself is read from the model (mjx_model.hfield_data).
    tc = self._config.get("terrain_config", None)
    self._terrain_on = tc is not None and tc.enable
    if self._terrain_on:
      hid = m.hfield("terrain").id
      assert m.nhfield == 1 and m.hfield_adr[hid] == 0
      self._hf = (int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid]),
                  *(float(v) for v in m.hfield_size[hid][:3]))  # nrow, ncol, half_x, half_y, max_z
      # Sample points under a foot ball for _feet_height: the center plus two rings
      # of 8 (at 0.5 r and 0.9 r). Columns: x offset, y offset, sqrt(r^2 - d^2).
      r = float(m.geom(FEET[0]).size[0])
      ring = [(0.0, 0.0)] + [(f * r * np.cos(a), f * r * np.sin(a))
                             for f in (0.5, 0.9) for a in np.arange(8) * np.pi / 4]
      self._foot_radius = r
      self._foot_ring = jp.array([(x, y, np.sqrt(r * r - x * x - y * y)) for x, y in ring])

    # RED PANDA (run 9): servo model. Each actuator's joint angle/speed, in ctrl order.
    act_joint = m.actuator_trnid[:, 0]
    self._act_qpos = jp.array(m.jnt_qposadr[act_joint])
    self._act_qvel = jp.array(m.jnt_dofadr[act_joint])
    self._stall = jp.array(m.actuator_forcerange[:, 1])
    self._servo = self._config.get("servo_config", None)
    self._servo_on = self._servo is not None and (
        self._servo.torque_speed_limit or max(self._servo.delay_ms) > 0)
    self._offset_deg = self._servo.get("joint_offset_deg", 0.0) if self._servo else 0.0

    # Soft limits (legs only). Go1 quirk kept for pass 1: x0.95 shrinks
    # toward zero, so our knees (ranges ending at 0) get no warning zone.
    leg_ranges = np.array([m.joint(n).range for n in LEG_JOINTS])
    self._lowers, self._uppers = leg_ranges.T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    tail_ranges = np.array([m.joint(n).range for n in TAIL_JOINTS])
    self._tail_lowers = jp.array(tail_ranges[:, 0])
    self._tail_uppers = jp.array(tail_ranges[:, 1])

    self._torso_body_id = m.body("torso").id
    self._torso_mass = m.body_subtreemass[self._torso_body_id]
    self._imu_site_id = m.site("imu").id

    self._feet_site_id = np.array([m.site(f).id for f in FEET])
    self._thigh_body_id = np.array([m.body(f"{leg}_upper").id for leg in LEGS])  # run 15a
    self._feet_floor_found_sensor = [
        m.sensor(f"{f}_floor_found").id for f in FEET
    ]

    foot_linvel_sensor_adr = []
    for f in FEET:
      sensor_id = m.sensor(f"{f}_global_linvel").id
      sensor_adr = m.sensor_adr[sensor_id]
      sensor_dim = m.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)
    # Run 25b: floor push on each foot (rewards only). Each sensor holds 4 contact points
    # x (normal, tangent1, tangent2) in N; [4 feet, 4 points] addresses of the normal push.
    self._foot_force_adr = jp.array([
        m.sensor_adr[m.sensor(f"{f}_floor_force").id] + 3 * np.arange(4) for f in FEET])
    self._weight = float(self._torso_mass * 9.81)  # nominal, N (torso = root: whole robot)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    # RED PANDA (run 5): gait clock. Old runs' configs have no gait_config -> off.
    self._gait = self._config.get("gait_config", None)
    self._gait_on = bool(self._gait is not None and self._gait.enable)
    if self._gait_on:
      self._gait_offsets = jp.array(self._gait.offsets)

  # Sensor readings (from Go1's base.py).

  def get_upvector(self, data):
    return mjx_env.get_sensor_data(self.mj_model, data, "upvector")

  def get_gravity(self, data):
    return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

  def get_global_linvel(self, data):
    return mjx_env.get_sensor_data(self.mj_model, data, "global_linvel")

  def get_global_angvel(self, data):
    return mjx_env.get_sensor_data(self.mj_model, data, "global_angvel")

  def get_local_linvel(self, data):
    return mjx_env.get_sensor_data(self.mj_model, data, "local_linvel")

  def get_accelerometer(self, data):
    return mjx_env.get_sensor_data(self.mj_model, data, "accelerometer")

  def get_gyro(self, data):
    return mjx_env.get_sensor_data(self.mj_model, data, "gyro")

  # RED PANDA: tail animation.

  def _sample_tail_params(self, rng: jax.Array) -> Dict[str, jax.Array]:
    c = self._config.tail_config
    keys = jax.random.split(rng, 8)

    def u(key, lo_hi):
      return jax.random.uniform(key, minval=lo_hi[0], maxval=lo_hi[1])

    return {
        "wag_amp": jp.deg2rad(u(keys[0], c.wag_amp_deg)),
        "freq": u(keys[1], c.wag_freq_hz),
        "phase": u(keys[2], [0.0, 2 * jp.pi]),
        "lift_center": jp.deg2rad(u(keys[3], c.lift_center_deg)),
        "lift_amp": jp.deg2rad(u(keys[4], c.lift_amp_deg)),
        "wag2_amp": jp.deg2rad(u(keys[5], c.wag2_amp_deg)),
        "wag2_lag": u(keys[6], c.wag2_lag_rad),
        # used only when the gait clock is on
        "gait_wag_amp": jp.deg2rad(u(keys[7], c.get("gait_wag_amp_deg", [0.0, 0.0]))),
        "gait_lift": jp.deg2rad(u(keys[3], c.get("gait_lift_deg", [0.0, 0.0]))),
    }

  def _tail_targets(self, t, p, phase) -> jax.Array:
    """Tail motor targets at time t: wag, lift, wag2 (radians)."""
    if self._gait_on:
      # One sway per stride, opposite the body's roll. Only the left feet are down
      # around phase 0.45 (body rolls right -> tail left, wag < 0); only the right
      # feet around 0.95 (tail right, wag > 0). + wag = tail toward the robot's right.
      angle = 2 * jp.pi * (phase - 0.2)
      sign = self._config.tail_config.get("gait_wag_sign", -1.0)  # -1 = run 5
      wag = sign * p["gait_wag_amp"] * jp.sin(angle)
      wag2 = sign * p["wag2_amp"] * jp.sin(angle - p["wag2_lag"])  # follow-through
      targets = jp.array([wag, p["gait_lift"], wag2])
      return jp.clip(targets, self._tail_lowers, self._tail_uppers)
    angle = 2 * jp.pi * p["freq"] * t + p["phase"]
    wag = p["wag_amp"] * jp.sin(angle)
    # Lift bobs twice per wag (once per side), like a swinging tail.
    lift = p["lift_center"] + p["lift_amp"] * jp.sin(2 * angle)
    lift = jp.maximum(lift, jp.deg2rad(self._config.tail_config.lift_min_deg))
    wag2 = p["wag2_amp"] * jp.sin(angle - p["wag2_lag"])  # follow-through
    targets = jp.array([wag, lift, wag2])
    return jp.clip(targets, self._tail_lowers, self._tail_uppers)

  # RED PANDA: neck animation.

  def _sample_neck_params(self, rng: jax.Array) -> Dict[str, jax.Array]:
    c = self._config.neck_config
    k1, k2, k3 = jax.random.split(rng, 3)
    on = 1.0 if c.enable else 0.0
    return {
        # each joint: random size (up to its max), speed, and starting point
        "amp": on * jp.deg2rad(jp.array(c.amp_deg)) * jax.random.uniform(k1, (3,)),
        "freq": jax.random.uniform(k2, (3,), minval=c.freq_hz[0], maxval=c.freq_hz[1]),
        "phase": jax.random.uniform(k3, (3,), minval=0.0, maxval=2 * jp.pi),
    }

  def _neck_targets(self, t, p, phase, torso_tilt) -> jax.Array:
    """Neck motor targets at time t: turn, nod, tilt (radians).
    torso_tilt = (pitch, roll) of the torso, + pitch = nose down, + roll = right side down."""
    drift = p["amp"] * jp.sin(2 * jp.pi * p["freq"] * t + p["phase"])
    if self._gait_on:
      c = self._config.neck_config
      turn_max, tilt_max = c.gait_turn_tilt_amp_deg
      nod_max = c.get("gait_nod_drift_deg", 0.0)  # run 12; 0 in older configs
      scale = jp.deg2rad(jp.array([turn_max, nod_max, tilt_max])) / jp.maximum(
          jp.deg2rad(jp.array(c.amp_deg)), 1e-6)
      drift = drift * scale  # random turn / nod / tilt drift
      # Front feet land at phase 0.2 (LF) and 0.7 (RF): nod down (+) at each.
      nod = jp.deg2rad(c.gait_nod_base_deg) + jp.deg2rad(c.gait_nod_amp_deg) * jp.cos(
          2 * jp.pi * 2 * (phase - 0.2))
      drift = drift.at[1].add(nod)  # steady nod + random nod drift (0 before run 12)
      if c.get("gait_head_stabilize", False):
        # Head pitch in the world ~= torso pitch + nod, so subtract the torso's pitch;
        # same for roll with the head-tilt joint (axis x, like the torso's roll).
        pitch, roll = torso_tilt
        drift = drift.at[1].add(-pitch)
        drift = drift.at[2].add(-roll)
    return drift

  def _motor_targets(self, leg_targets, t, tail_params, neck_params, phase,
                     torso_tilt=(0.0, 0.0)) -> jax.Array:
    """RED PANDA: all 18 motor targets, in actuator (ctrl) order."""
    ctrl = self._init_ctrl
    ctrl = ctrl.at[self._leg_act].set(leg_targets)
    ctrl = ctrl.at[self._tail_act].set(self._tail_targets(t, tail_params, phase))
    ctrl = ctrl.at[self._neck_act].set(
        self._neck_targets(t, neck_params, phase, torso_tilt))
    return ctrl

  def _torso_tilt(self, data):
    """RED PANDA (run 6): torso pitch and roll from the gravity direction in the body
    frame (what the real IMU gives). g_body = (sin p, -sin r cos p, -cos r cos p).
    + pitch = nose down, + roll = right side down (rotation about the forward axis)."""
    g = self.get_gravity(data)
    pitch = jp.arcsin(jp.clip(g[0], -1.0, 1.0))
    roll = jp.arctan2(-g[1], -g[2])
    return pitch, roll

  def _lean_target(self, phase, command):
    """RED PANDA (run 6): target (pitch, roll) for the torso at this point in the stride.
    Only the left feet are down around phase 0.45 -> lean left (left side down, roll < 0);
    only the right feet around 0.95 -> lean right. Standing still -> no sway."""
    g = self._gait
    sway = jp.sin(2 * jp.pi * (phase - g.get("roll_phase", 0.2)))  # default: +1 at 0.45 (left pair down), -1 at 0.95
    walking = jp.linalg.norm(command) >= 0.01
    roll = -jp.deg2rad(g.get("roll_amp_deg", 0.0)) * sway * walking
    pitch = jp.deg2rad(g.get("pitch_deg", 0.0))
    return pitch, roll

  def _desired_contact(self, phase, command) -> jax.Array:
    """RED PANDA (run 5): which feet should be down now, from the footfall chart.
    A foot is down from its touchdown offset until `duty` later (wrapping around).
    Joystick centered -> all four down (stand still)."""
    in_stance = jp.mod(phase - self._gait_offsets, 1.0) < self._gait.duty
    standing = jp.linalg.norm(command) < 0.01
    return jp.where(standing, True, in_stance)

  def reset(self, rng: jax.Array, tile: Optional[jax.Array] = None) -> mjx_env.State:
    """tile: optional [level, col] to start on (evals; overrides terrain_config's spawn)."""
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    if self._terrain_on:  # Terrain: start on a random tile, standing on top of its lumps
      tc = self._config.terrain_config
      rng, k1, k2 = jax.random.split(rng, 3)
      level = jax.random.randint(k1, (), 0, tc.n_levels)  # (curriculum: picked per robot later)
      col = jax.random.randint(k2, (), 0, tc.n_cols)
      if tc.get("spawn_level", -1) >= 0:  # evals: one fixed tile
        level = jp.array(tc.spawn_level)
      if tc.get("spawn_col", -1) >= 0:
        col = jp.array(tc.spawn_col)
      if tile is not None:  # evals: one batch can cover several tiles (one compile)
        level, col = tile[0], tile[1]
      cx, cy = terrain.tile_center(tc, level, col)
      qpos = qpos.at[0:2].add(jp.array([cx, cy]))
      # Raise the robot by the tallest ground under its footprint (1 cm grid over
      # 40 x 40 cm: covers the feet at any yaw), so no foot starts inside a lump.
      gx = qpos[0] + jp.linspace(-0.2, 0.2, 41)[:, None]
      gy = qpos[1] + jp.linspace(-0.2, 0.2, 41)[None, :]
      qpos = qpos.at[2].add(jp.max(self._ground(gx, gy)))

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    # RED PANDA: random tail sway for this episode.
    rng, key = jax.random.split(rng)
    tail_params = self._sample_tail_params(key)
    rng, key = jax.random.split(rng)
    neck_params = self._sample_neck_params(key)
    rng, key = jax.random.split(rng)
    gait_phase = jax.random.uniform(key)  # RED PANDA (run 5): start anywhere in the stride

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=self._motor_targets(self._default_pose, 0.0, tail_params, neck_params,
                                 gait_phase),
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * 5.0
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )
    cmd = jax.random.uniform(
        key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )

    info = {
        "rng": rng,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        "time": jp.zeros(()),        # RED PANDA: clock for the tail animation
        "tail_params": tail_params,  # RED PANDA
        "neck_params": neck_params,  # RED PANDA
        "gait_phase": gait_phase,  # RED PANDA (run 5): 0..1 through the stride
        "last_leg_qvel": jp.zeros(len(LEG_JOINTS)),  # RED PANDA (run 5): for joint_acc
        # Run 23 (_track_swings): per leg, by the gait clock
        "was_swing": jp.zeros(4, dtype=bool),        # scheduled in the air last step
        "liftoff_knee": self._default_pose[2::3],    # knee angle when the current swing began
        "sched_peak": jp.zeros(4),                   # highest foot height in the current swing
        "swing_end": jp.zeros(4, dtype=bool),        # a swing ended this step
        "swing_end_peak": jp.zeros(4),               # ...and how high that swing got
    }
    if self._terrain_on:
      info["terrain_level"] = level  # which row (difficulty) this robot started on
    if self._servo_on:  # RED PANDA (run 9)
      # Motor delay for this episode, in physics substeps (sim_dt each).
      rng, key = jax.random.split(rng)
      lo, hi = (jp.array(self._servo.delay_ms) / 1000.0 / self._config.sim_dt)
      n = self.n_substeps
      info["delay_substeps"] = jp.clip(jp.round(
          jax.random.uniform(key, minval=lo, maxval=hi)), 0, 2 * n).astype(jp.int32)
      info["last_motor_targets"] = jp.stack([data.ctrl, data.ctrl])  # 1 and 2 steps ago
      info["rng"] = rng
    if self._offset_deg > 0:  # RED PANDA (run 13): calibration error per leg servo
      rng, key = jax.random.split(rng)
      info["joint_offset"] = jp.deg2rad(self._offset_deg) * jax.random.uniform(
          key, (len(LEG_JOINTS),), minval=-1.0, maxval=1.0)
      info["rng"] = rng

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)

    # RED PANDA: 12 leg targets from the policy + 3 tail + 3 neck = 18.
    state.info["time"] += self.dt
    if self._gait_on:  # RED PANDA (run 5): advance the stride clock
      state.info["gait_phase"] = jp.mod(
          state.info["gait_phase"] + self._gait.freq_hz * self.dt, 1.0)
    leg_targets = self._default_pose + action * self._config.action_scale
    if self._offset_deg > 0:  # run 13: the servo's "target" lands offset in reality
      leg_targets = leg_targets - state.info["joint_offset"]
    motor_targets = self._motor_targets(
        leg_targets,
        state.info["time"],
        state.info["tail_params"],
        state.info["neck_params"],
        state.info["gait_phase"],
        self._torso_tilt(state.data),  # run 6: for head stabilization
    )
    if self._servo_on:  # RED PANDA (run 9)
      data = self._servo_substeps(state.data, motor_targets, state.info)
      state.info["last_motor_targets"] = jp.stack(
          [motor_targets, state.info["last_motor_targets"][0]])
    else:
      data = mjx_env.step(
          self.mjx_model, state.data, motor_targets, self.n_substeps
      )

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_fz = self._feet_height(data)  # terrain: above the ground under each foot
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
    if self._gait_on:  # run 23: lift-off knee angle and per-swing peak, by the gait clock
      self._track_swings(data, state.info, p_fz)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_leg_qvel"] = data.qvel[self._leg_qvel]
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["command"]),
        state.info["command"],
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _servo_substeps(self, data, motor_targets, info):
    """RED PANDA (run 9): physics substeps with the servo torque-speed limit (+ delay).

    MuJoCo's position actuator twists with kp * (target - angle), clipped to +/-stall.
    Each substep we work out that torque ourselves, clip it to what the servo can give
    at its current speed, then hand MuJoCo the target that produces exactly that torque.
    """
    model = self.mjx_model
    kp = model.actuator_gainprm[:, 0]  # read from the model so randomized gains count
    n = self.n_substeps
    prev, prev2 = info["last_motor_targets"]

    def substep(data, i):
      # Motor delay: substep i runs the targets sent delay_substeps substeps ago.
      back = i - info["delay_substeps"]
      target = jp.where(back >= 0, motor_targets, jp.where(back >= -n, prev, prev2))
      if self._servo.torque_speed_limit:
        q = data.qpos[self._act_qpos]
        w = data.qvel[self._act_qvel]
        tau = kp * (target - q)
        # Twisting the way it's already turning: less available the faster it turns.
        # Twisting against the motion (braking): up to stall.
        tau_max = self._stall * jp.clip(1.0 - w / self._servo.no_load_speed, 0.0, 1.0)
        tau_min = -self._stall * jp.clip(1.0 + w / self._servo.no_load_speed, 0.0, 1.0)
        target = q + jp.clip(tau, tau_min, tau_max) / kp
      return mjx.step(model, data.replace(ctrl=target)), None

    return jax.lax.scan(substep, data, jp.arange(n))[0]

  # --- Terrain (Tier 2): every height is measured from the ground under it. ---
  # With terrain off, the ground is 0 everywhere, so these give the same numbers
  # as the old world-floor heights (x - 0 = x exactly).

  def _ground(self, x, y):
    """Ground height under world point(s) (x, y): 0 on the flat floor."""
    if not self._terrain_on:
      return jp.zeros_like(x)
    return terrain.ground_height(jp, self.mjx_model.hfield_data, *self._hf, x, y)

  def _feet_height(self, data):
    """Each foot ball's center height above the ground it would touch (fl, fr, bl, br).

    Reads the ball radius whenever the foot touches, on flat ground or a lump's
    side. (Just "ground straight below" reads up to ~0.6 cm too high on a 50 deg
    slope, because the ball touches the slope off-center.) At each sample point
    in the ball's footprint, the center can't go lower than ground + sqrt(r^2 - d^2);
    the highest of those is where the center sits when touching.
    """
    p = data.site_xpos[self._feet_site_id]
    if not self._terrain_on:
      return p[:, 2]
    gx = p[:, 0:1] + self._foot_ring[:, 0]          # (4 feet, 17 points)
    gy = p[:, 1:2] + self._foot_ring[:, 1]
    touching_z = jp.max(self._ground(gx, gy) + self._foot_ring[:, 2], axis=1)
    return p[:, 2] - touching_z + self._foot_radius

  def _torso_height(self, data):
    """Torso height above the ground the feet stand on (mean under the 4 feet).

    Not the ground under the torso's center: a lump under the belly isn't
    something the legs stand on, and it would jerk the height target around.
    """
    p = data.site_xpos[self._feet_site_id]
    return data.qpos[2] - jp.mean(self._ground(p[:, 0], p[:, 1]))

  def _track_swings(self, data, info, feet_h):
    """Run 23: per-swing bookkeeping on the gait clock (not on contact: a dragging foot
    never leaves the ground, so contact-based bookkeeping would never see its swing).

    Remembers each knee's angle when its scheduled swing starts (for the lift-off
    knee fold) and the highest the foot got during the swing (for the clearance floor).
    """
    prog = jp.mod(info["gait_phase"] - self._gait_offsets, 1.0)
    swing = prog >= self._gait.duty
    started = swing & ~info["was_swing"]
    ended = ~swing & info["was_swing"]
    knee = data.qpos[self._leg_qpos[2::3]]
    info["liftoff_knee"] = jp.where(started, knee, info["liftoff_knee"])
    info["swing_end"] = ended
    info["swing_end_peak"] = info["sched_peak"]  # the swing that just ended (used where swing_end)
    info["sched_peak"] = jp.where(
        started, feet_h, jp.where(swing, jp.maximum(info["sched_peak"], feet_h), info["sched_peak"]))
    info["was_swing"] = swing

  def _swing_knee_err2(self, data, info):
    """Run 22b: (knee angle - fold target)^2 per leg while its foot should be in the air.

    Target = the home-pose knee, bent further by swing_knee_fold_deg (front, hind) times
    the bell shape (0 at lift-off and touchdown, full at mid-swing). 28 / 32 deg lifts
    the foot 2.5 cm straight up from the home stance (IK). Targets the "stiff pendulum"
    swing of skimming legs: thigh swings forward, knee stays straight, foot slides.
    """
    prog = jp.mod(info["gait_phase"] - self._gait_offsets, 1.0)  # 0..1 since touchdown
    swing = prog >= self._gait.duty
    s_prog = jp.clip((prog - self._gait.duty) / (1.0 - self._gait.duty), 0.0, 1.0)
    shape = jp.square(jp.sin(jp.pi * s_prog))
    rc = self._config.reward_config
    if rc.get("swing_knee_ref", "home") == "liftoff":
      # Run 23: fold measured from the knee's own angle at lift-off, so a knee that is
      # bent all the time (22b's crouch) gets no credit; only bending during the swing counts.
      front, hind = np.deg2rad(rc.swing_knee_fold_liftoff_deg)
      base = info["liftoff_knee"]
    else:  # run 22b: from the home pose
      front, hind = np.deg2rad(rc.swing_knee_fold_deg)
      base = self._default_pose[2::3]
    fold = jp.array([-front, -front, hind, hind])  # front knees bend negative, hind positive
    knee = data.qpos[self._leg_qpos[2::3]]           # fl, fr, bl, br knees
    target = base + fold * shape
    return jp.where(swing, jp.square(knee - target), 0.0)

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    # RED PANDA: also end if the torso sinks too low (belly-flop).
    too_low = self._torso_height(data) < self._config.min_torso_height
    return fall_termination | too_low

  def _noisy(self, info, x, scale):
    info["rng"], noise_rng = jax.random.split(info["rng"])
    return x + (
        (2 * jax.random.uniform(noise_rng, shape=x.shape) - 1)
        * self._config.noise_config.level
        * scale
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    scales = self._config.noise_config.scales
    gyro = self.get_gyro(data)
    gravity = self.get_gravity(data)
    linvel = self.get_local_linvel(data)
    leg_angles = data.qpos[self._leg_qpos]
    if self._offset_deg > 0:  # run 13: what a miscalibrated servo reads (policy only)
      leg_angles = leg_angles + info["joint_offset"]
    leg_vel = data.qvel[self._leg_qvel]
    tail_angles = data.qpos[self._tail_qpos]
    tail_vel = data.qvel[self._tail_qvel]
    neck_angles = data.qpos[self._neck_qpos]
    neck_vel = data.qvel[self._neck_qvel]

    # RED PANDA: no body speed by default (no real sensor measures it);
    # sees the tail and neck (A+). 3+3+12+12+12+3+3+3+3+3 = 57 numbers.
    parts = []
    if self._config.policy_sees_body_speed:
      parts.append(self._noisy(info, linvel, scales.linvel))  # 3
    parts += [
        self._noisy(info, gyro, scales.gyro),  # 3
        self._noisy(info, gravity, scales.gravity),  # 3
        self._noisy(info, leg_angles, scales.joint_pos) - self._default_pose,  # 12
        self._noisy(info, leg_vel, scales.joint_vel),  # 12
        info["last_act"],  # 12
        info["command"],  # 3
        self._noisy(info, tail_angles, scales.joint_pos),  # 3
        self._noisy(info, tail_vel, scales.joint_vel),  # 3
        self._noisy(info, neck_angles, scales.joint_pos),  # 3
        self._noisy(info, neck_vel, scales.joint_vel),  # 3
    ]
    if self._gait_on:  # RED PANDA (run 5): the stride clock, as a point on a circle
      angle = 2 * jp.pi * info["gait_phase"]
      parts.append(jp.array([jp.sin(angle), jp.cos(angle)]))  # 2 -> 59 inputs
    state = jp.hstack(parts)

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    # Critic only (training): perfect values, including body speed.
    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        angvel,  # 3
        data.qpos[self._leg_qpos] - self._default_pose,  # 12 (true angles, run 13)
        leg_vel,  # 12
        tail_angles,  # 3
        tail_vel,  # 3
        neck_angles,  # 3
        neck_vel,  # 3
        data.actuator_force,  # 18
        info["last_contact"],  # 4
        feet_vel,  # 4*3
        info["feet_air_time"],  # 4
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    # RED PANDA: posture and motor terms use the 12 leg joints only; the
    # policy doesn't control the tail or neck, so their motion isn't its fault.
    leg_q = data.qpos[self._leg_qpos]
    leg_qvel = data.qvel[self._leg_qvel]
    leg_force = data.actuator_force[self._leg_act]
    terms = {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation_target(data, info),
        "stand_still": self._cost_stand_still(info["command"], leg_q),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(leg_q),
        "torques": self._cost_torques(leg_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(leg_qvel, leg_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(leg_q),
    }
    if self._gait_on:  # RED PANDA (run 5): the walk spec
      pitch, roll = self._torso_tilt(data)
      pitch_t, roll_t = self._lean_target(info["gait_phase"], info["command"])
      err2 = jp.square(pitch - pitch_t) + jp.square(roll - roll_t)
      terms["body_sway"] = jp.exp(-err2 / self._config.reward_config.get("body_sway_sigma", 1.0))
      desired = self._desired_contact(info["gait_phase"], info["command"])
      terms["gait_schedule"] = jp.mean(desired == contact)
      rc = self._config.reward_config
      if "swing_height" in rc.scales:  # run 7
        prog = jp.mod(info["gait_phase"] - self._gait_offsets, 1.0)  # 0..1 since touchdown
        swing = prog >= self._gait.duty
        s_prog = jp.clip((prog - self._gait.duty) / (1.0 - self._gait.duty), 0.0, 1.0)
        if rc.get("swing_arc", "sine") == "bell":  # run 18: eases in AND out
          shape = jp.square(jp.sin(jp.pi * s_prog))  # zero speed at lift-off and touchdown
        else:                                        # runs 7-17: comes down at ~0.32 m/s
          shape = jp.sin(jp.pi * s_prog)
        target = rc.foot_rest_height + rc.swing_lift * shape
        foot_z = self._feet_height(data)  # terrain: above the ground under each foot
        walking = jp.linalg.norm(info["command"]) >= 0.01
        # Run 21: hind feet (bl, br) can cost more to skim; 1.0 = same as the front.
        hind = rc.get("swing_hind_mult", 1.0)
        foot_w = jp.array([1.0, 1.0, hind, hind])  # FEET order: fl, fr, bl, br
        terms["swing_height"] = jp.sum(
            jp.where(swing, foot_w * jp.square(foot_z - target), 0.0)) * walking
      if "swing_knee" in rc.scales:  # run 22b (0 = off; still logged)
        walking = jp.linalg.norm(info["command"]) >= 0.01
        terms["swing_knee"] = jp.sum(self._swing_knee_err2(data, info)) * walking
      if "swing_floor" in rc.scales:  # run 23c (0 = off; still logged)
        # Once per swing, when it ends: (floor - how high the foot got)^2, 0 if it cleared.
        walking = jp.linalg.norm(info["command"]) >= 0.01
        lift = info["swing_end_peak"] - rc.foot_rest_height
        short = jp.maximum(rc.swing_floor_lift - lift, 0.0)
        if rc.get("swing_floor_shape", "squared") == "linear":  # run 26: a hard floor
          miss = short
        else:
          miss = jp.square(short)
        terms["swing_floor"] = jp.sum(jp.where(info["swing_end"], miss, 0.0)) * walking
      if "unload" in rc.scales:  # run 25b (0 = off; still logged)
        # Sum over feet of w x (floor push / body weight)^2. w follows the gait clock:
        # 0 most of stance, ramps to 1 over the last unload_before_s, 1 for the first
        # unload_after_s of swing, 0 after. Lifting late doesn't help: the clock doesn't wait.
        walking = jp.linalg.norm(info["command"]) >= 0.01
        period = 1.0 / self._gait.freq_hz
        prog = jp.mod(info["gait_phase"] - self._gait_offsets, 1.0)  # 0..1 since touchdown
        to_lift = (self._gait.duty - prog) * period  # s until lift-off (stance)
        since_lift = (prog - self._gait.duty) * period  # s since lift-off (swing)
        w = jp.where(prog < self._gait.duty,
                     jp.clip(1.0 - to_lift / rc.unload_before_s, 0.0, 1.0),
                     (since_lift < rc.unload_after_s).astype(jp.float32))
        push = jp.sum(data.sensordata[self._foot_force_adr], axis=1) / self._weight
        terms["unload"] = jp.sum(w * jp.square(push)) * walking
      terms["base_height"] = jp.square(self._torso_height(data) - self._gait.base_height)
      terms["joint_acc"] = jp.sum(
          jp.square((leg_qvel - info["last_leg_qvel"]) / self.dt))
      if "soft_landing" in rc.scales:  # run 17
        # Feet should be set down, not stomped: downward foot speed^2 while the foot
        # is within 1 cm of its resting height (measured just before contact; once the
        # contact flag is on, the foot has already stopped). Terrain: measured from
        # the ground under the foot, so landing on top of a lump counts too.
        foot_vz = data.sensordata[self._foot_linvel_sensor_adr][:, 2]
        foot_h = self._feet_height(data) - rc.foot_rest_height
        near = foot_h < 0.01
        terms["soft_landing"] = jp.sum(jp.where(near, jp.square(jp.minimum(foot_vz, 0.0)), 0.0))
      if "foot_under_hip" in rc.scales:  # run 15a
        # Feet on the ground should sit under their thigh joints (where a straight leg
        # hangs; the hip body itself is 3 cm further in), in the torso's frame:
        # (fore/aft)^2 + (sideways)^2 per planted foot. A stride still needs about
        # +/-3.4 cm of fore/aft travel at 0.12 m/s, so a good gait pays a little.
        R = data.xmat[self._torso_body_id]
        rel = (data.site_xpos[self._feet_site_id] - data.xpos[self._thigh_body_id]) @ R
        if rc.get("foot_under_hip_ref", "hip") == "clock":  # run 26: long centred steps are free
          prog = jp.mod(info["gait_phase"] - self._gait_offsets, 1.0)  # 0..1 since touchdown
          # A foot down while scheduled in the air is judged against the nearer end of its
          # stance (early landing -> touchdown spot, late lift-off -> lift-off spot):
          # timing is gait_schedule's job, this term only judges position.
          duty = self._gait.duty
          prog = jp.where(prog > (1.0 + duty) / 2, prog - 1.0, prog)
          travel = info["command"][0] * duty / self._gait.freq_hz
          expected_x = travel * jp.clip(0.5 - prog / duty, -0.5, 0.5)  # +travel/2 -> -travel/2
          rel = rel.at[:, 0].add(-expected_x)
        terms["foot_under_hip"] = jp.sum(jp.where(contact, jp.sum(jp.square(rel[:, :2]), axis=1), 0.0))
    # only terms this run's config has a weight for (old runs replay unchanged)
    return {k: v for k, v in terms.items() if k in self._config.reward_config.scales}

  # Reward terms: unchanged from Go1.

  def _reward_tracking_lin_vel(self, commands, local_vel):
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma_lin)

  def _reward_tracking_ang_vel(self, commands, ang_vel):
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma_ang)

  def _cost_lin_vel_z(self, global_linvel):
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel):
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation_target(self, data, info):
    """RED PANDA (run 6): Go1's orientation cost, but around the lean target instead of
    perfectly flat. With zero targets it equals Go1's exactly: |g_body xy|^2 = sin^2(tilt)."""
    g = self.get_gravity(data)
    if not self._gait_on:
      return jp.sum(jp.square(g[:2]))
    pitch_t, roll_t = self._lean_target(info["gait_phase"], info["command"])
    g_t = jp.array([jp.sin(pitch_t), -jp.sin(roll_t) * jp.cos(pitch_t)])
    return jp.sum(jp.square(g[:2] - g_t))

  def _cost_orientation(self, torso_zaxis):
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_torques(self, torques):
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(self, qvel, qfrc_actuator):
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(self, act, last_act, last_last_act):
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _reward_pose(self, qpos):
    per_leg = self._config.reward_config.get("pose_weights", [1.0, 1.0, 0.1])  # Go1 = runs 1-2
    weight = jp.array(list(per_leg) * 4)  # hip_side, hip, knee per leg
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_stand_still(self, commands, qpos):
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

  def _cost_termination(self, done):
    return done

  def _cost_joint_pos_limits(self, qpos):
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_feet_slip(self, data, contact, info):
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

  def _cost_feet_clearance(self, data):
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_z = self._feet_height(data)  # terrain: above the ground under each foot
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(self, swing_peak, first_contact, info):
    cmd_norm = jp.linalg.norm(info["command"])
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

  def _reward_feet_air_time(self, air_time, first_contact, commands):
    cmd_norm = jp.linalg.norm(commands)
    threshold = self._config.reward_config.get("air_time_threshold", 0.1)  # 0.1 = Go1 (run 1)
    rew_air_time = jp.sum((air_time - threshold) * first_contact)
    rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
    return rew_air_time

  # Perturbation and command sampling: unchanged from Go1.

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      force = (
          u_t
          * self._torso_mass
          * state.info["pert_mag"]
          / state.info["pert_duration_seconds"]
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    return x_kp1

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return len(LEG_JOINTS)  # RED PANDA: policy moves the 12 leg joints only

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
