"""Rearing into the tail tripod (Tier 1): from wherever walking left the robot, plant the
tail, move the weight back, lift the front feet, and hold ~70 deg nose-up on both hind
feet + the tail tip (option A: tools/eval/tripod_static.py).

Built on the walking env (same robot, servo model, randomization, sensors) with:
  - 15 actions: 12 leg servos + 3 tail servos (the policy owns the tail here).
  - Tail and torso can touch the floor (contact sensors say which part).
  - Starts from a bank of states the walking policy produced (tools/rear_starts.py):
    standing still or mid-stride. No stride clock, no joystick, no pushes (v1).
  - Its own rewards (starter set, chosen 2026-11-15) and fall checks.

Separate policy from walking (CLAUDE.md: separate policies + a switcher), so training
this can't change the walking policy.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

import red_panda_walk as rpw
from red_panda_walk import LEG_JOINTS, TAIL_JOINTS

REPO = Path(__file__).resolve().parents[1]
# Parts that may touch the floor while rearing (named here; the walking XML leaves them
# unnamed and non-colliding). tail_tip is the good one; the rest mean "sitting".
TAIL_GEOMS = {"tail_base": ["tail_base_box"], "tail_1": ["tail_1_rod", "tail_1_box"],
              "tail_2": ["tail_tip"]}
BAD_GEOMS = ["torso", "tail_base_box", "tail_1_rod", "tail_1_box"]


def default_config() -> config_dict.ConfigDict:
  c = rpw.default_config()
  c.episode_length = 300  # 6 s: rear up, then hold
  c.terrain_config.enable = False  # flat floor first
  c.gait_config.enable = False     # no stride clock (and no clock input)
  c.pert_config.enable = False     # v1: no pushes
  c.naconmax = 16 * 8192           # tail + torso contacts on top of the feet
  c.njmax = 80
  c.rear_config = config_dict.create(
      starts="data/rear_starts.npz",
      moving_fraction=0.3,  # share of starts taken from mid-walk (rest: standing still)
      # Reverse ladder (rr_run4a; Florensa et al. 2017, "Reverse Curriculum Generation"):
      # this share of robots starts partway up instead (tools/rear_ladder.py: rung 0 = the
      # 70 deg tripod, then 60, 50 ... 10 deg). Each robot keeps its own rung: an episode
      # that arrives and stays up -> one rung lower next time (harder); a fall or no
      # arrival -> one rung higher. Past the last rung it starts from its walking state.
      ladder="data/rear_ladder.npz",
      ladder_fraction=0.0,  # 0 = off
      # Replay (rr_run6b): this share of a ladder robot's episodes starts at a random
      # rung it has already passed (0 .. its rung - 1) instead of its own rung, so
      # earlier rungs aren't forgotten (rr_run5b lost 40-50 deg). Replays don't move it.
      ladder_replay=0.0,
      # Tripod pose (option A, 70 deg / hips 16 cm), from tripod_static.pose: joint angles
      # in degrees for the joints that move a lot. Each one's action range is centred
      # halfway between home and here and stretched to reach both, plus action_scale
      # of slack each way. Other joints: home +/- action_scale, like walking.
      tripod_deg=config_dict.create(bl_hip=38.0, br_hip=38.0, bl_knee=15.0, br_knee=15.0,
                                    tail_lift=55.0),
      # Other angles a joint's range must also reach (rr_run1 bug: tail_lift only went
      # down to -29 deg, but planting the tail from standing needs -35 deg).
      plant_deg=config_dict.create(tail_lift=-45.0),
      max_roll_deg=30.0,      # end the episode past this roll
      max_nose_down_deg=30.0, # ...or this far nose-down (fell forward)
      # Speed (user, 2026-11-15). "Arrived" = the full tripod at once (both hind feet +
      # tail tip down, both front feet up, tilt and hip height in tolerance) held for
      # arrive_hold_s in a row; the arrival time is when that streak began. Hips that
      # sag to be quick never arrive, so speed can't be bought with height.
      arrive_tilt_tol_deg=5.0,
      arrive_height_tol=0.015,
      arrive_hold_s=0.2,      # so swinging through the pose on the way to a fall doesn't count
      # Per-robot deadline curriculum (like Rudin 2021's terrain levels): every robot
      # starts at deadline_start; each episode on time and still standing -> deadline_step
      # tighter; fell or never arrived -> deadline_step looser; never below deadline_final.
      # Physics floor (stick model, 65% stall cap): 0.44 s at 1.8 kg, 0.47 s at 1.98 kg.
      deadline_start=2.0,
      deadline_final=0.8,     # later try 0.6
      deadline_step=0.1,
      # rr_run7b: scale each episode's deadline by how far its start is from the tripod:
      # deadline = full budget x (70 deg - start tilt) / 70 deg (0..1, at least 0.1 s), so a
      # 70 deg ladder start and a walking start are judged fairly and the curriculum moves
      # one number (the full budget, rear_config.deadline) for all of them. Start it near
      # what the policy already does (rr_run6b: median 4.1 s from walking starts).
      deadline_by_distance=False,
  )
  c.reward_config = config_dict.create(
      scales=config_dict.create(
          # Starter set (user, 2026-11-15). Run first, then take apart.
          tilt=1.0,            # 0.5 bell(35 deg) + 0.5 bell(8 deg) around 70 deg nose-up
          height=0.5,          # bell(2 cm) around hind-hip height 16 cm (stops "sitting")
          tripod_contact=1.0,  # both hind feet + tail tip down, both front feet up
          lean=3.0,            # rr_run3 (user): nose-up tilt / 70 deg (0..1) x the height bell, so
                               # every degree of lean pays (runs 1-2 had no path: "a bit up" paid
                               # the same full late bill as "down"). x height bell: sitting back
                               # with low hips can't earn it. A reward, not a "still down" penalty:
                               # falling would stop a penalty.
          tail_first=-1.0,     # a front foot up while the tail tip isn't down
          bad_contact=-1.0,    # torso or tail root on the floor ("sitting")
          # Speed and servo limits (user, 2026-11-15):
          late=-10.0,          # per second past the deadline without having arrived (linear: a
                               # little late still costs). A fall before arriving pays every
                               # remaining second of the episode, so falling can't stop the bill.
          servo_pinned=-1.0,   # per second, per servo using >= 95% of the torque available at
                               # its current speed (asking for more than it can do; run 23c)
          over_peak=-3.0,      # per second, per N*m above 65% of stall (our hardware peak rule)
          # Regularizers, as walking:
          action_rate=-0.15,
          torques=-0.0105,
          energy=-0.007,
          dof_pos_limits=-1.0,
          termination=-1.0,
          joint_acc=-2.5e-7,
          feet_slip=-0.1,      # walking's only counted with a joystick command; always on here
          roll_rate=-0.05,     # walking's ang_vel_xy without pitch (rearing must pitch)
          # Symmetry (rr_run8a, user): rr_run7a held the tripod crooked: tail bent 20 deg to the
          # right (wag +27, wag2 +15 deg), left hind leg splayed 16 deg wider, right hind hip
          # working 21% harder. Off here; rr_run8a: tail_straight -2, leg_symmetry -5 (costs: negative).
          tail_straight=0.0,   # wag^2 + wag2^2 (rad): the tail's two sideways joints at 0
          leg_symmetry=0.0,    # (left hind hip_side - right hind hip_side)^2 (rad; + = outward on both)
      ),
      target_pitch_deg=70.0,
      tilt_sigmas_deg=[35.0, 8.0],
      target_hip_height=0.16,
      height_sigma=0.02,
      pin_frac=0.95,
      # Cost curriculum (rr_run4b; Hwangbo et al. 2019): these costs start at x
      # cost_start and grow linearly to x1 over each robot's first cost_ramp_steps steps,
      # so trying to get up is cheap early and the full servo rules apply later.
      cost_start=1.0,       # 1 = off
      cost_ramp_steps=15000,  # ~60% of a 200M-step run (24k steps per robot at 8192 robots)
      cost_terms=["servo_pinned", "over_peak", "torques", "energy", "action_rate"],
      peak_frac=0.65,
  )
  return c


def rear_spec(config) -> mujoco.MjSpec:
  spec = rpw.build_spec(config)
  for body, names in TAIL_GEOMS.items():
    for g, n in zip(spec.body(body).geoms, names):
      g.name = n
      g.conaffinity = 1  # the floor has contype 1
  spec.geom("torso").conaffinity = 1
  for n in ["tail_tip"] + BAD_GEOMS:
    spec.add_sensor(name=f"{n}_floor_found", type=mujoco.mjtSensor.mjSENS_CONTACT,
                    objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=n,
                    reftype=mujoco.mjtObj.mjOBJ_GEOM, refname="floor",
                    intprm=[1, 1, 1])  # data "found", reduce "mindist", 1 slot
  return spec


class RedPandaRear(rpw.RedPandaWalk):
  """Rear up into the tail tripod and hold it."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    self._xml_path = rpw.XML_PATH.as_posix()
    self._mj_model = rear_spec(self._config).compile()
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()
    m = self._mj_model
    self._found_adr = {n: int(m.sensor_adr[m.sensor(f"{n}_floor_found").id])
                       for n in ["tail_tip"] + BAD_GEOMS}
    self._hind_thigh = np.array([m.body("bl_upper").id, m.body("br_upper").id])
    self._lt_act = jp.concatenate([self._leg_act, self._tail_act])     # the 15 policy servos
    self._lt_qvel = jp.concatenate([self._leg_qvel, self._tail_qvel])

    # Action -> target: centre + half_span * action (action in [-1, 1] after tanh).
    rc = self._config.rear_config
    names = LEG_JOINTS + TAIL_JOINTS
    home = np.array(m.keyframe("home").qpos)[np.concatenate([self._leg_qpos, self._tail_qpos])]
    centre, half = home.copy(), np.full(len(names), self._config.action_scale)
    # Each listed joint's range covers home, its tripod angle and any plant angle.
    reach = {}
    for table in (rc.tripod_deg, rc.get("plant_deg", {})):
      for j, deg in table.items():
        reach.setdefault(j, [home[names.index(j)]]).append(np.deg2rad(deg))
    for j, angles in reach.items():
      i = names.index(j)
      lo, hi = min(angles), max(angles)
      centre[i] = (lo + hi) / 2
      half[i] = (hi - lo) / 2 + self._config.action_scale
    self._act_centre, self._act_half = jp.array(centre), jp.array(half)

    # Start states (hand-off from walking).
    bank = np.load(REPO / rc.starts)
    self._bank_qpos, self._bank_qvel = jp.array(bank["qpos"]), jp.array(bank["qvel"])
    moving = bank["moving"].astype(bool)
    self._bank_move, self._bank_stand = jp.array(np.where(moving)[0]), jp.array(np.where(~moving)[0])
    self._ladder_on = rc.get("ladder_fraction", 0.0) > 0
    if self._ladder_on:
      ladder = np.load(REPO / rc.ladder)
      self._ladder_qpos = jp.array(ladder["qpos"])
      self._n_rungs = len(ladder["qpos"])  # rung n_rungs = no override (walking start)

  @property
  def action_size(self) -> int:
    return len(LEG_JOINTS) + len(TAIL_JOINTS)

  def _targets(self, action, info):
    t = self._act_centre + self._act_half * action
    legs, tail = t[:12], t[12:]
    if self._offset_deg > 0:
      legs = legs - info["joint_offset"]
    ctrl = self._init_ctrl.at[self._leg_act].set(legs)
    ctrl = ctrl.at[self._tail_act].set(jp.clip(tail, self._tail_lowers, self._tail_uppers))
    # Stride clock off -> the neck ignores phase and torso tilt (just the random drift).
    neck = self._neck_targets(info["time"], info["neck_params"], 0.0, (0.0, 0.0))
    return ctrl.at[self._neck_act].set(neck)

  def _found(self, data, name):
    return data.sensordata[self._found_adr[name]] > 0

  def reset(self, rng: jax.Array, tile: Optional[jax.Array] = None) -> mjx_env.State:
    del tile
    rc = self._config.rear_config
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    use_move = jax.random.uniform(k1) < rc.moving_fraction
    i_move = self._bank_move[jax.random.randint(k2, (), 0, len(self._bank_move))]
    i_stand = self._bank_stand[jax.random.randint(k3, (), 0, len(self._bank_stand))]
    i = jp.where(use_move, i_move, i_stand)
    qpos = self._bank_qpos[i].at[0:2].set(0.0)  # same pose and heading, moved to the origin
    qvel = self._bank_qvel[i]

    rng, key = jax.random.split(rng)
    neck_params = self._sample_neck_params(key)
    info = {"rng": rng, "time": jp.zeros(()), "neck_params": neck_params,
            "tail_params": self._sample_tail_params(key),  # unused (policy owns the tail)
            "command": jp.zeros(3), "gait_phase": jp.zeros(()),
            "last_act": jp.zeros(self.action_size), "last_last_act": jp.zeros(self.action_size),
            "feet_air_time": jp.zeros(4), "last_contact": jp.zeros(4, dtype=bool),
            "steps_until_next_pert": jp.array(10**6, dtype=jp.int32), "steps_since_last_pert": 0,
            "last_leg_qvel": qvel[self._leg_qvel],
            # Per episode (step() clears them when a new episode starts: Playground's
            # auto-reset restores the physics state but keeps info):
            "ep_time": jp.zeros(()), "arrived": jp.array(False),
            "box_since": jp.array(-1.0), "arrive_time": jp.array(-1.0),
            # Kept across episodes (the curriculums):
            "deadline": jp.array(rc.deadline_start),
            "ep_deadline": jp.array(rc.deadline_start),  # this episode's deadline (scaled or not)
            "life_steps": jp.array(0, dtype=jp.int32),  # steps this robot has lived (cost curriculum)
            "rung": jp.array(0, dtype=jp.int32),
            "play_rung": jp.array(0, dtype=jp.int32)}  # the rung this episode started on
    rng, key = jax.random.split(rng)
    info["on_ladder"] = jax.random.uniform(key) < rc.get("ladder_fraction", 0.0)
    if self._offset_deg > 0:
      rng, key = jax.random.split(rng)
      info["joint_offset"] = jp.deg2rad(self._offset_deg) * jax.random.uniform(
          key, (len(LEG_JOINTS),), minval=-1.0, maxval=1.0)
    # Servos start holding the angles they are at (no jump at the hand-off).
    ctrl = self._init_ctrl.at[self._leg_act].set(qpos[self._leg_qpos])
    ctrl = ctrl.at[self._tail_act].set(qpos[self._tail_qpos])
    ctrl = ctrl.at[self._neck_act].set(qpos[self._neck_qpos])
    data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl,
                             impl=self.mjx_model.impl.value,
                             naconmax=self._config.naconmax, njmax=self._config.njmax)
    data = mjx.forward(self.mjx_model, data)
    if self._servo_on:
      rng, key = jax.random.split(rng)
      lo, hi = (jp.array(self._servo.delay_ms) / 1000.0 / self._config.sim_dt)
      info["delay_substeps"] = jp.clip(jp.round(jax.random.uniform(key, minval=lo, maxval=hi)),
                                       0, 2 * self.n_substeps).astype(jp.int32)
      info["last_motor_targets"] = jp.stack([data.ctrl, data.ctrl])
    info["rng"] = rng
    info["last_contact"] = self._feet_contact(data)
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales}
    metrics.update({k: jp.zeros(()) for k in self._METRICS})
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def _feet_contact(self, data):
    return jp.array([data.sensordata[self._mj_model.sensor_adr[s]] > 0
                     for s in self._feet_floor_found_sensor])

  # Per-step metrics. Brax adds each one up over an episode, so the speed ones are
  # non-zero on one step only: summed, "arrived" = 1 if it arrived, "arrive_time_s" =
  # when, "deadline_s" = this robot's deadline (logged on the episode's last step).
  _METRICS = ["pitch_up_deg", "hip_height_cm", "in_tripod", "arrived", "arrive_time_s",
              "late_s", "deadline_s", "pinned_servos", "ladder_rung", "ladder_ends", "cost_factor"]

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = state.info
    info["time"] += self.dt
    # New episode? Playground's auto-reset puts its step counter back to 0 (episode
    # end or fall) but keeps our info, so clear the per-episode entries here.
    new_ep = (info["steps"] == 0) if "steps" in info else (info["ep_time"] == 0)
    info["arrived"] = info["arrived"] & ~new_ep
    info["box_since"] = jp.where(new_ep, -1.0, info["box_since"])
    info["arrive_time"] = jp.where(new_ep, -1.0, info["arrive_time"])
    info["ep_time"] = jp.where(new_ep, 0.0, info["ep_time"]) + self.dt
    info["life_steps"] = info["life_steps"] + 1

    data0 = state.data
    if self._ladder_on:  # move ladder robots to their rung's pose, still, at episode start
      rng, k1, k2 = jax.random.split(info["rng"], 3)
      info["rng"] = rng
      replay = (jax.random.uniform(k1) < self._config.rear_config.get("ladder_replay", 0.0)) & (info["rung"] > 0)
      old = jax.random.randint(k2, (), 0, jp.maximum(info["rung"], 1))
      info["play_rung"] = jp.where(new_ep, jp.where(replay, old, info["rung"]), info["play_rung"])
      use = new_ep & info["on_ladder"] & (info["play_rung"] < self._n_rungs)
      # (jax.lax.cond crashed on the GPU / Warp, rr_run4a; under vmap a cond runs both
      # branches anyway. Data.where, not jax.tree.map: Warp's contact buffers are shared
      # by all robots, not one per robot.)
      data0 = data0.where(use, self._ladder_data(data0, info))
      if self._servo_on:  # the delayed servo targets should be the new pose too, not the old one
        info["last_motor_targets"] = jp.where(use, jp.stack([data0.ctrl, data0.ctrl]),
                                              info["last_motor_targets"])

    # This episode's deadline, set once from where the episode starts.
    ep_dl = info["deadline"]
    if self._config.rear_config.get("deadline_by_distance", False):
      tilt0 = -jp.rad2deg(self._torso_tilt(data0)[0])
      target = self._config.reward_config.target_pitch_deg
      ep_dl = jp.maximum(info["deadline"] * jp.clip((target - tilt0) / target, 0.0, 1.0), 0.1)
    info["ep_deadline"] = jp.where(new_ep, ep_dl, info["ep_deadline"])

    targets = self._targets(action, info)
    if self._servo_on:
      data = self._servo_substeps(data0, targets, info)
      info["last_motor_targets"] = jp.stack([targets, info["last_motor_targets"][0]])
    else:
      data = mjx_env.step(self.mjx_model, data0, targets, self.n_substeps)

    contact = self._feet_contact(data)
    info["feet_air_time"] = (info["feet_air_time"] + self.dt) * ~contact
    obs = self._get_obs(data, info)
    done = self._get_termination(data)
    newly, late = self._track_arrival(data, info, contact, done)
    rewards = self._rear_rewards(data, action, info, done, contact, late)
    rw = self._config.reward_config
    cost_f = rw.get("cost_start", 1.0) + (1.0 - rw.get("cost_start", 1.0)) * jp.minimum(
        info["life_steps"] / rw.get("cost_ramp_steps", 1), 1.0)
    rewards = {k: v * rw.scales[k] * (cost_f if k in rw.get("cost_terms", []) else 1.0)
               for k, v in rewards.items()}
    # No clip at 0 (walking clips each step's reward at 0): with a clip, a -10 late cost
    # would just turn the step into 0, no worse than a -1.
    reward = sum(rewards.values()) * self.dt

    # Curriculum, on the episode's last step (fell, or ran out of time).
    rc = self._config.rear_config
    ep_end = done | (info["ep_time"] >= self._config.episode_length * self.dt - self.dt / 2)
    on_time = info["arrived"] & (info["arrive_time"] <= info["ep_deadline"])
    tighten = ep_end & on_time & ~done
    loosen = ep_end & (done | ~info["arrived"])
    deadline_now = info["deadline"]
    info["deadline"] = jp.clip(
        info["deadline"] - rc.deadline_step * tighten + rc.deadline_step * loosen,
        rc.deadline_final, rc.deadline_start)
    rung_now = info["rung"]
    if self._ladder_on:
      own = ep_end & info["on_ladder"] & (info["play_rung"] == info["rung"])  # not a replay
      up = own & info["arrived"] & ~done   # made it: start lower next time
      down = own & (done | ~info["arrived"])
      info["rung"] = jp.clip(info["rung"] + up.astype(jp.int32) - down.astype(jp.int32),
                             0, self._n_rungs)

    info["last_leg_qvel"] = data.qvel[self._leg_qvel]
    info["last_last_act"] = info["last_act"]
    info["last_act"] = action
    info["last_contact"] = contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    pitch, _ = self._torso_tilt(data)
    m = state.metrics
    m["pitch_up_deg"] = -jp.rad2deg(pitch)
    m["hip_height_cm"] = 100 * jp.mean(data.xpos[self._hind_thigh, 2])
    m["in_tripod"] = self._in_tripod(data, contact).astype(jp.float32)
    m["arrived"] = newly.astype(jp.float32)
    m["arrive_time_s"] = jp.where(newly, info["arrive_time"], 0.0)
    m["late_s"] = late * self.dt
    m["deadline_s"] = jp.where(ep_end, deadline_now, 0.0)
    m["pinned_servos"] = self._servo_limits(data)[0]
    ladder_end = ep_end & info["on_ladder"]
    m["ladder_rung"] = jp.where(ladder_end, rung_now, 0).astype(jp.float32)  # rung this episode used
    m["ladder_ends"] = ladder_end.astype(jp.float32)
    m["cost_factor"] = jp.where(ep_end, cost_f, 0.0)
    return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

  def _ladder_data(self, data, info):
    """The physics state at this robot's ladder rung: that pose, zero speed, servos holding it."""
    qpos = self._ladder_qpos[jp.minimum(info["play_rung"], self._n_rungs - 1)]
    ctrl = data.ctrl.at[self._leg_act].set(qpos[self._leg_qpos])
    ctrl = ctrl.at[self._tail_act].set(qpos[self._tail_qpos])
    d = data.replace(qpos=qpos, qvel=jp.zeros_like(data.qvel), ctrl=ctrl,
                     qacc=jp.zeros_like(data.qacc), qacc_warmstart=jp.zeros_like(data.qacc_warmstart))
    return mjx.forward(self.mjx_model, d)

  def _track_arrival(self, data, info, contact, done):
    """Updates info's arrival bookkeeping. Returns (arrived this step, late seconds / dt).

    late is 1 on a step past the deadline without having arrived (so x dt = seconds),
    except while holding a streak that began before the deadline (it may still count).
    On a fall before arriving it is also charged every remaining second of the episode,
    as if it stayed late to the end: falling can't stop the bill.
    """
    rc = self._config.rear_config
    rw = self._config.reward_config
    t = info["ep_time"]
    pitch, _ = self._torso_tilt(data)
    hip_h = jp.mean(data.xpos[self._hind_thigh, 2])
    in_box = (self._in_tripod(data, contact)
              & (jp.abs(-jp.rad2deg(pitch) - rw.target_pitch_deg) < rc.arrive_tilt_tol_deg)
              & (jp.abs(hip_h - rw.target_hip_height) < rc.arrive_height_tol))
    info["box_since"] = jp.where(in_box, jp.where(info["box_since"] < 0, t, info["box_since"]), -1.0)
    newly = (~info["arrived"] & in_box & ~done
             & (t - info["box_since"] >= rc.arrive_hold_s - self.dt / 2))
    info["arrive_time"] = jp.where(newly, info["box_since"], info["arrive_time"])
    info["arrived"] = info["arrived"] | newly

    deadline = info["ep_deadline"]
    may_count = in_box & (info["box_since"] <= deadline)
    late = (~info["arrived"] & (t > deadline) & ~may_count).astype(jp.float32)
    remaining = jp.maximum(self._config.episode_length * self.dt - jp.maximum(t, deadline), 0.0)
    late = late + jp.where(done & ~info["arrived"], remaining / self.dt, 0.0)
    return newly, late

  def _servo_limits(self, data):
    """(servos pinned, N*m above the peak rule), over the 15 policy servos.

    Pinned = using >= pin_frac of the torque available at its current speed (the servo
    model's straight torque-speed line; braking can use up to stall). Measured on the
    last physics substep of the step."""
    rw = self._config.reward_config
    f = data.actuator_force[self._lt_act]
    w = data.qvel[self._lt_qvel]
    stall = self._stall[self._lt_act]
    avail = stall * jp.clip(1.0 - jp.sign(f) * w / self._servo.no_load_speed, 0.0, 1.0)
    pinned = (jp.abs(f) > 1e-3) & (jp.abs(f) >= rw.pin_frac * avail)
    over = jp.maximum(jp.abs(f) - rw.peak_frac * stall, 0.0)
    return jp.sum(pinned.astype(jp.float32)), jp.sum(over)

  def _in_tripod(self, data, contact):
    return contact[2] & contact[3] & self._found(data, "tail_tip") & ~contact[0] & ~contact[1]

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    rc = self._config.rear_config
    pitch, roll = self._torso_tilt(data)  # + pitch = nose down
    flipped = self.get_upvector(data)[-1] < 0.0  # tipped over backwards (or upside down)
    rolled = jp.abs(roll) > jp.deg2rad(rc.max_roll_deg)
    fell_forward = pitch > jp.deg2rad(rc.max_nose_down_deg)
    return flipped | rolled | fell_forward

  def _rear_rewards(self, data, action, info, done, contact, late):
    rc = self._config.reward_config
    pitch, _ = self._torso_tilt(data)
    err = -pitch - jp.deg2rad(rc.target_pitch_deg)  # nose-up angle minus target
    sw, sn = jp.deg2rad(jp.array(rc.tilt_sigmas_deg))
    hip_h = jp.mean(data.xpos[self._hind_thigh, 2])
    tail_down = self._found(data, "tail_tip")
    front_up = ~contact[0] | ~contact[1]
    bad = jp.any(jp.array([self._found(data, n) for n in BAD_GEOMS]))
    leg_q, leg_qvel = data.qpos[self._leg_qpos], data.qvel[self._leg_qvel]
    force, qvel = data.actuator_force[self._lt_act], data.qvel[self._lt_qvel]
    feet_v = data.sensordata[self._foot_linvel_sensor_adr][:, :2]
    pinned, over = self._servo_limits(data)
    height_bell = jp.exp(-jp.square((hip_h - rc.target_hip_height) / rc.height_sigma))
    terms = {
        "lean": jp.clip(-pitch / jp.deg2rad(rc.target_pitch_deg), 0.0, 1.0) * height_bell,
        "tilt": 0.5 * jp.exp(-jp.square(err / sw)) + 0.5 * jp.exp(-jp.square(err / sn)),
        "height": height_bell,
        "tripod_contact": self._in_tripod(data, contact).astype(jp.float32),
        "tail_first": (front_up & ~tail_down).astype(jp.float32),
        "bad_contact": bad.astype(jp.float32),
        "late": late,
        "servo_pinned": pinned,
        "over_peak": over,
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "torques": self._cost_torques(force),
        "energy": self._cost_energy(qvel, force),
        "dof_pos_limits": self._cost_joint_pos_limits(leg_q),
        "termination": done,
        "joint_acc": jp.sum(jp.square((leg_qvel - info["last_leg_qvel"]) / self.dt)),
        "feet_slip": jp.sum(jp.sum(jp.square(feet_v), axis=-1) * contact),
        "roll_rate": jp.square(self.get_gyro(data)[0]),
        "tail_straight": jp.square(data.qpos[self._tail_qpos[0]]) + jp.square(data.qpos[self._tail_qpos[2]]),
        "leg_symmetry": jp.square(leg_q[LEG_JOINTS.index("bl_hip_side")] - leg_q[LEG_JOINTS.index("br_hip_side")]),
    }
    return {k: v for k, v in terms.items() if k in rc.scales}
