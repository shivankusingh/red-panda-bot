"""Coming back down (Tier 1): from the tail tripod onto all four feet, gently, ending in the
walking stance so the walking policy can take over.

Same robot, actions (12 legs + 3 tail), servo model, randomization, pushes, symmetry and
roll terms as rearing (red_panda_rear.py, rr_run11 defaults). What changes:
  - Starts: tripod states the rearing policy actually holds (tools/down_starts.py).
  - Target ("landed"): all four feet down, tail tip off the floor, tilt and hip height of
    the walking stance (tilt -2 deg = 2 deg nose-down, hind hips 16 cm), held 0.2 s.
  - Rewards (user's choices, 2026-11-16): stance bells (tilt, height), `lower` (progress
    down, x height bell), `feet_down` (feet down / 4; dn_run1-2 used an all-or-nothing
    `four_feet`), `front_landing` (each front-foot touchdown costs its
    downward speed squared: a passive topple lands at 1.77 m/s; walking feet land at
    0.05-0.30), a 1.5 s deadline with the same linear late bill (no curriculum).
    2026-11-17: `front_symmetry` (front legs mirror each other), stiffer `over_peak` and
    `backdriven` (a servo forced backwards while braking at stall) for the hind-hip strain.
Separate policy from rearing (CLAUDE.md: separate policies + a switcher).
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict

import red_panda_rear as rr

FRONT = [0, 1]  # fl, fr in the feet-contact order


def default_config() -> config_dict.ConfigDict:
  c = rr.default_config()
  c.episode_length = 200  # 4 s: come down, then stand
  rc = c.rear_config
  rc.starts = "data/down_starts.npz"
  rc.moving_fraction = 0.0
  # dn_run1 (no ladder) froze in the tripod: any random move flipped it or face-planted,
  # so standing still was all that ever worked. Reverse ladder (tools/down_ladder.py):
  # start at 10 deg (front feet just above the floor), one rung higher per success.
  rc.ladder = "data/down_ladder.npz"
  rc.ladder_fraction = 1.0
  rc.ladder_replay = 0.5
  rc.deadline_by_distance = False
  rc.deadline_start = rc.deadline_final = 1.5   # fixed deadline, no curriculum
  rc.deadline_step = 0.0
  rw = c.reward_config
  rw.target_pitch_deg = -2.0    # walking stance: 2 deg nose-down
  rw.target_hip_height = 0.16   # walking stance hind hips (home: 16.2 cm)
  s = rw.scales
  for k in ["tripod_contact", "tail_first", "lean"]:  # rearing-only terms
    del s[k]
  s.lower = 3.0          # (70 - tilt) / 72 in 0..1, x height bell: every degree down pays
  # dn_run2's all-or-nothing four_feet paid 3 feet the same as 0: every robot kept a
  # front paw up and never landed. Graded: each foot down is worth +1/s.
  s.feet_down = 4.0      # feet down / 4
  s.front_landing = -10.0  # per front-foot touchdown: (downward speed, m/s)^2. 1.77 m/s -> -31
  # dn_run3a/b (restored from dn_run2) stood on a lopsided triangle: left front hip-side
  # at its -20 deg limit under the body, right front swung out 24 deg and folded 4.4 cm up.
  # Four feet scored +11/s more, but no single-leg move gets there. Like rearing's
  # leg_symmetry, for the front legs. dn_run4a (without it) landed but lopsided and slower.
  s.front_symmetry = -5.0  # (fl - fr hip_side)^2 + (fl - fr knee)^2 (rad; mirrored axes)
  # dn_run4b (tools/eval/down_load.py, every 4 ms): both hind hips at stall for 56-80 ms
  # while the body tips forward, and the load wins: braking at full torque, the hip still
  # turns backwards at up to 4.7 rad/s (the gear-tooth case). Walking never does this.
  # over_peak x5 alone (dn_run5a) roughly halved it; with backdriven (dn_run5b) the
  # back-driving is ~gone (max 8-12 ms) and the hips are over 65% of stall ~30 ms per landing.
  s.over_peak = -15.0      # rearing: -3
  s.backdriven = -2.0      # sum over servos of |speed| (rad/s) while braking at >= 90% of stall
  # Kept policy dn_run5b = two steps (user's choices, 2026-11-17):
  #   dn_run4b: from scratch, --set reward_config.cost_start=0.3 (costs ramp 0.3 -> 1),
  #             front_symmetry -5, over_peak -3, backdriven 0.
  #   dn_run5b: --restore dn_run4b with these defaults (full costs from the start).
  # From scratch with the dn_run2 habit restored, the policy kept one paw up (dn_run3a/b).
  return c


class RedPandaDown(rr.RedPandaRear):
  """Come down from the tail tripod onto four feet."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

  def reset(self, rng: jax.Array, tile: Optional[jax.Array] = None):
    state = super().reset(rng, tile)
    state.info["last_front_vz"] = jp.zeros(2)  # front feet vertical speed, previous step
    return state

  def _target_contacts(self, data, contact):
    """Landed contacts: all four feet down, tail tip lifted off the floor."""
    return jp.all(contact) & ~self._found(data, "tail_tip")

  def _backdriven(self, data):
    """Sum of |joint speed| (rad/s) over the policy servos that are braking (torque against
    the motion) at >= 90% of stall and still being turned: the load is winning."""
    f = data.actuator_force[self._lt_act]
    w = data.qvel[self._lt_qvel]
    stall = self._stall[self._lt_act]
    back = (f * w < 0) & (jp.abs(f) >= 0.9 * stall) & (jp.abs(w) > 0.2)
    return jp.sum(jp.where(back, jp.abs(w), 0.0))

  def _rear_rewards(self, data, action, info, done, contact, late):
    terms = super()._rear_rewards(data, action, info, done, contact, late)
    rw = self._config.reward_config
    pitch, _ = self._torso_tilt(data)
    tilt = -jp.rad2deg(pitch)
    hip_h = jp.mean(data.xpos[self._hind_thigh, 2])
    height_bell = jp.exp(-jp.square((hip_h - rw.target_hip_height) / rw.height_sigma))
    # Front-foot touchdowns: the speed just before contact (this step's foot speed is
    # already braked by the floor). Remembered here for the next step.
    vz = data.sensordata[self._foot_linvel_sensor_adr][jp.array(FRONT), 2]
    touchdown = contact[jp.array(FRONT)] & ~info["last_contact"][jp.array(FRONT)]
    landing = jp.sum(jp.where(touchdown, jp.square(jp.minimum(info["last_front_vz"], 0.0)), 0.0))
    info["last_front_vz"] = vz
    extra = {
        "lower": jp.clip((70.0 - tilt) / (70.0 - rw.target_pitch_deg), 0.0, 1.0) * height_bell,
        "feet_down": jp.mean(contact.astype(jp.float32)),
        "front_landing": landing / self.dt,  # x dt (every term is) = speed^2 per touchdown
        "front_symmetry": sum(jp.square(data.qpos[self._leg_qpos[rr.LEG_JOINTS.index(f"fl_{j}")]]
                                        - data.qpos[self._leg_qpos[rr.LEG_JOINTS.index(f"fr_{j}")]])
                              for j in ("hip_side", "knee")),
        "backdriven": self._backdriven(data),
    }
    terms.update({k: v for k, v in extra.items() if k in rw.scales})
    return terms
