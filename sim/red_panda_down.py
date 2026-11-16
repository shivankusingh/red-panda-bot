"""Coming back down (Tier 1): from the tail tripod onto all four feet, gently, ending in the
walking stance so the walking policy can take over.

Same robot, actions (12 legs + 3 tail), servo model, randomization, pushes, symmetry and
roll terms as rearing (red_panda_rear.py, rr_run11 defaults). What changes:
  - Starts: tripod states the rearing policy actually holds (tools/down_starts.py).
  - Target ("landed"): all four feet down, tail tip off the floor, tilt and hip height of
    the walking stance (tilt -2 deg = 2 deg nose-down, hind hips 16 cm), held 0.2 s.
  - Rewards (user's choices, 2026-11-16): stance bells (tilt, height), `lower` (progress
    down, x height bell), `four_feet`, `front_landing` (each front-foot touchdown costs its
    downward speed squared: a passive topple lands at 1.77 m/s; walking feet land at
    0.05-0.30), a 1.5 s deadline with the same linear late bill (no curriculum).
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
  s.four_feet = 1.0      # all four feet down and the tail tip off the floor
  s.front_landing = -10.0  # per front-foot touchdown: (downward speed, m/s)^2. 1.77 m/s -> -31
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
        "four_feet": self._target_contacts(data, contact).astype(jp.float32),
        "front_landing": landing / self.dt,  # x dt (every term is) = speed^2 per touchdown
    }
    terms.update({k: v for k, v in extra.items() if k in rw.scales})
    return terms
