"""Stress-test a trained walking policy: one real-world difference at a time.

Every condition runs S robots (different start phase / push timing) walking forward
at 0.2 m/s for 10 s. Prints fall rate and measured forward speed per condition.
Usage: .venv/bin/python stress_test.py rp_run8 [seeds]
"""
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

REPO = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk  # noqa: E402

run = REPO / "runs" / sys.argv[1]
S = int(sys.argv[2]) if len(sys.argv) > 2 else 8
cfg = json.load(open(run / "config.json"))
env_cfg = config_dict.ConfigDict(cfg["env"])
env_cfg.impl = "jax"
red_panda_walk.use_flat(env_cfg)  # standard checks on flat
env_cfg.pert_config.enable = True  # pushes on; size set per robot below (0 = no push)
if "servo_config" in env_cfg:  # delay is added per condition below, not sampled
  env_cfg.servo_config.delay_ms = [0.0, 0.0]
env = red_panda_walk.RedPandaWalk(config=env_cfg)
mj = env.mj_model

FOOT_AND_FLOOR = [0] + [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot")
                        for l in ["fl", "fr", "bl", "br"]]
TORSO = 1
M_TORSO, M_TOTAL = mj.body_mass[TORSO], mj.body_subtreemass[0]
KICK_PER_DV = np.pi * M_TOTAL / M_TORSO  # Go1 push: whole-robot dv = M_torso*mag/(pi*M_total)

# (label, changes)
CONDITIONS = [("mass x1.0 (1.80 kg)", {}), ("mass x1.1 (1.98 kg)", dict(mass=1.1))]
labels = [c for c, _ in CONDITIONS for _ in range(S)]
specs = [s for _, s in CONDITIONS for _ in range(S)]
N = len(specs)


def randomize(model):
  fr = np.tile(np.array(model.geom_friction), (N, 1, 1))
  mass = np.tile(np.array(model.body_mass), (N, 1))
  inertia = np.tile(np.array(model.body_inertia), (N, 1, 1))
  ipos = np.tile(np.array(model.body_ipos), (N, 1, 1))
  gain = np.tile(np.array(model.actuator_gainprm), (N, 1, 1))
  bias = np.tile(np.array(model.actuator_biasprm), (N, 1, 1))
  for i, s in enumerate(specs):
    if "mu" in s:
      fr[i, [0] if s.get("floor_only") else FOOT_AND_FLOOR, 0] = s["mu"]
    if "mass" in s:
      mass[i, 1:] *= s["mass"]
      inertia[i, 1:] *= s["mass"]
    if "com" in s:
      ipos[i, TORSO] += np.array(s["com"])
    if "kp" in s:
      gain[i, :, 0] *= s["kp"]
      bias[i, :, 1] *= s["kp"]
  new = dict(geom_friction=fr, body_mass=mass, body_inertia=inertia, body_ipos=ipos,
             actuator_gainprm=gain, actuator_biasprm=bias)
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({k: 0 for k in new})
  model = model.tree_replace({k: jp.array(v) for k, v in new.items()})
  return model, in_axes


venv = BraxDomainRandomizationVmapWrapper(env, randomize)

net = cfg["ppo"]["network_factory"]
networks = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
    value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
params = pickle.load(open(run / "params.pkl", "rb"))
policy = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))
reset, step = jax.jit(venv.reset), jax.jit(venv.step)


cmd = jp.tile(jp.array([0.2, 0.0, 0.0]), (N, 1))
state = reset(jax.random.split(jax.random.PRNGKey(0), N))
state.info["pert_mag"] = jp.zeros(N)
key = jax.random.PRNGKey(1)
leg_qvel = np.array(env._leg_qvel)
leg_qpos = np.array(env._leg_qpos)
tau, w, err = [], [], []
for t in range(500):
  state.info["command"] = cmd
  state.info["steps_until_next_cmd"] = jp.full((N,), 1000, dtype=jp.int32)
  key, k = jax.random.split(key)
  a, _ = policy(state.obs, k)
  state = step(state, a)
  if t >= 100:
    tau.append(np.array(state.data.actuator_force)[:, :12])
    w.append(np.array(state.data.qvel)[:, leg_qvel])
    err.append(np.array(state.data.ctrl)[:, :12] - np.array(state.data.qpos)[:, leg_qpos])
tau, w, err = np.abs(np.array(tau)), np.abs(np.array(w)), np.abs(np.array(err))
STALL, W0, RATED = 2.94, 4.7, 0.98
cap = STALL * np.clip(1 - w / W0, 0, 1)        # linear torque-speed curve
names = ["hip_side", "hip", "knee"]
for c, _ in CONDITIONS:
  idx = [i for i, l in enumerate(labels) if l == c]
  print(f"\n{c}: walking 0.2 m/s, 16 robots x 8 s")
  print(f"  {'joint':9s} {'avg':>5s} {'%rated':>7s} {'99th pct':>9s} {'% time maxed':>13s} {'top speed':>10s} {'% time over real-servo curve':>29s}  {'avg target error':>16s}")
  for j, n in enumerate(names):
    cols = [3 * leg + j for leg in range(4)]
    x = tau[:, idx][:, :, cols]; v = w[:, idx][:, :, cols]; cp = cap[:, idx][:, :, cols]
    e = err[:, idx][:, :, cols]
    print(f"  {n:9s} {x.mean():5.2f} {100*x.mean()/RATED:6.0f}% {np.percentile(x, 99):9.2f} "
          f"{100*np.mean(x > 2.85):12.1f}% {np.percentile(v, 99):7.1f} r/s {100*np.mean(x > cp):27.1f}%  {np.degrees(e.mean()):13.1f} deg")
