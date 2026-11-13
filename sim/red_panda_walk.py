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
          wag_amp_deg=[0.0, 30.0],     # side-to-side size
          wag_freq_hz=[0.3, 1.5],      # sways per second
          lift_center_deg=[-10.0, 40.0],
          lift_amp_deg=[0.0, 15.0],
          lift_min_deg=-25.0,          # below ~-40 deg the tail hits the floor
          wag2_amp_deg=[0.0, 30.0],    # second joint, lags behind (follow-through)
          wag2_lag_rad=[0.3, 1.5],
      ),
      # RED PANDA: the head "looks around" while walking (animation, not the
      # policy): each neck joint drifts slowly on its own random rhythm.
      # Set enable=False to hold the head still (e.g. to debug a bad run).
      neck_config=config_dict.create(
          enable=True,
          amp_deg=[30.0, 20.0, 15.0],  # max swing: turn, nod, tilt
          freq_hz=[0.1, 0.5],          # slow: one look every 2-10 s
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
              torques=-0.0035,
              action_rate=-0.01,
              # RED PANDA: Go1 x7. Standing-matched would be x17 (-0.017), but under
              # early-training jitter our light legs make energy 2.4x Go1's share;
              # too high risks "stand still" early, so match that. If run 1 freezes,
              # suspect this first. Compare reward/energy's share with Go1's after run 1.
              energy=-0.007,
              # Feet.
              feet_clearance=-2.0,
              feet_height=-0.2,
              feet_slip=-0.1,
              feet_air_time=0.1,
          ),
          # RED PANDA: two sigmas instead of Go1's one (0.25). Our walk speeds are
          # 4x slower than Go1's (error^2 16x smaller) -> 0.25/16; turn speeds are
          # 2.4x slower -> 0.25/5.8. Grades "off by a bit" like Go1 does.
          tracking_sigma_lin=0.016,
          tracking_sigma_ang=0.043,
          # RED PANDA: foot-center height target mid-step (Go1: 0.1). 4.0 cm =
          # 3 cm of lift + 0.97 cm foot radius: a low, long red panda step that's
          # quick for slow servos. (Go1-proportional would be 4.7: 84% of max lift.)
          max_foot_height=0.04,
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          # RED PANDA: forward / sideways / turn max (Go1: 1.5, 0.8, 1.2).
          # Forward 0.4 m/s: servo-speed estimate (0.3-0.5). Sideways 0.2:
          # Go1's sideways/forward ratio (53%). Turn 0.5 rad/s: Go1's turn as a
          # fraction of its fastest possible turn (18%), scaled to our feet being
          # 0.14 m from the center. Measure the real top speeds after the first run.
          a=[0.4, 0.2, 0.5],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      # RED PANDA: "jax" runs on the Mac; use "warp" on an NVIDIA GPU.
      impl="jax",
      naconmax=4 * 8192,
      njmax=40,
  )


class RedPandaWalk(mjx_env.MjxEnv):
  """Track a joystick command with the legs; tail and neck animated."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._xml_path = XML_PATH.as_posix()
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
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

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

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
    }

  def _tail_targets(self, t: jax.Array, p: Dict[str, jax.Array]) -> jax.Array:
    """Tail motor targets at time t: wag, lift, wag2 (radians)."""
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

  def _neck_targets(self, t: jax.Array, p: Dict[str, jax.Array]) -> jax.Array:
    """Neck motor targets at time t: turn, nod, tilt (radians)."""
    return p["amp"] * jp.sin(2 * jp.pi * p["freq"] * t + p["phase"])

  def _motor_targets(self, leg_targets, t, tail_params, neck_params) -> jax.Array:
    """RED PANDA: all 18 motor targets, in actuator (ctrl) order."""
    ctrl = self._init_ctrl
    ctrl = ctrl.at[self._leg_act].set(leg_targets)
    ctrl = ctrl.at[self._tail_act].set(self._tail_targets(t, tail_params))
    ctrl = ctrl.at[self._neck_act].set(self._neck_targets(t, neck_params))
    return ctrl

  def reset(self, rng: jax.Array) -> mjx_env.State:
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

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=self._motor_targets(self._default_pose, 0.0, tail_params, neck_params),
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
    }

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
    leg_targets = self._default_pose + action * self._config.action_scale
    motor_targets = self._motor_targets(
        leg_targets,
        state.info["time"],
        state.info["tail_params"],
        state.info["neck_params"],
    )
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
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

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

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    # RED PANDA: also end if the torso sinks too low (belly-flop).
    too_low = data.qpos[2] < self._config.min_torso_height
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
        leg_angles - self._default_pose,  # 12
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
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
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
    weight = jp.array([1.0, 1.0, 0.1] * 4)  # hip_side, hip, knee per leg
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
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(self, swing_peak, first_contact, info):
    cmd_norm = jp.linalg.norm(info["command"])
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

  def _reward_feet_air_time(self, air_time, first_contact, commands):
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
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
