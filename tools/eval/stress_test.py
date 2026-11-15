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
red_panda_walk.use_flat(env_cfg)  # standard checks on flat (terrain: terrain_eval.py)
env_cfg.pert_config.enable = True  # pushes on; size set per robot below (0 = no push)
if "servo_config" in env_cfg:  # delay is added per condition below, not sampled
  env_cfg.servo_config.delay_ms = [0.0, 0.0]
env = red_panda_walk.RedPandaWalk(config=env_cfg)
mj = env.mj_model

FOOT_AND_FLOOR = [0] + [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot")
                        for l in ["fl", "fr", "bl", "br"]]
TORSO = 1
M_TORSO, M_TOTAL = mj.body_mass[TORSO], mj.body_subtreemass[0]
KICK_PER_DV = np.pi * M_TOTAL / float(env._torso_mass)  # env push: dv = env._torso_mass*mag/(pi*M_total) (checked in sim)

# (label, changes)
CONDITIONS = [
    ("baseline", {}),
    ("friction 0.5 (floor+feet)", dict(mu=0.5)),
    ("friction 0.3", dict(mu=0.3)),
    ("friction 0.2", dict(mu=0.2)),
    ("friction 0.1 (icy)", dict(mu=0.1)),
    ("friction 0.2, floor only", dict(mu=0.2, floor_only=True)),
    ("mass x0.9 (1.62 kg)", dict(mass=0.9)),
    ("mass x1.1 (1.98 kg)", dict(mass=1.1)),
    ("mass x1.2 (2.16 kg)", dict(mass=1.2)),
    ("COM 2 cm forward", dict(com=(0.02, 0, 0))),
    ("COM 2 cm back", dict(com=(-0.02, 0, 0))),
    ("COM 2 cm sideways", dict(com=(0, 0.02, 0))),
    ("COM 5 cm forward (Go1 size)", dict(com=(0.05, 0, 0))),
    ("servo stiffness x0.7", dict(kp=0.7)),
    ("servo stiffness x0.5", dict(kp=0.5)),
    ("servo stiffness x1.3", dict(kp=1.3)),
    ("delay 20 ms (1 step)", dict(delay=1)),
    ("delay 40 ms (2 steps)", dict(delay=2)),
    ("pushes 0.2 m/s", dict(push=0.2)),
    ("pushes 0.3 m/s", dict(push=0.3)),
    ("pushes 0.45 m/s", dict(push=0.45)),
    ("pushes 0.6 m/s", dict(push=0.6)),
    ("pushes 0.7 m/s", dict(push=0.7)),
    ("pushes 0.9 m/s", dict(push=0.9)),
    ("ALL: mu .4, m x1.1, kp x.8, 20ms, push .3",
     dict(mu=0.4, mass=1.1, kp=0.8, delay=1, push=0.3)),
]
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

delay = jp.array([s.get("delay", 0) for s in specs])
kick = jp.array([s.get("push", 0.0) * KICK_PER_DV for s in specs])
VX = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
cmd = jp.tile(jp.array([VX, 0.0, 0.0]), (N, 1))

t0 = time.time()
state = reset(jax.random.split(jax.random.PRNGKey(0), N))
state.info["pert_mag"] = kick
hist = jp.zeros((3, N, env.action_size))  # last 3 policy outputs, for the delay
fell = np.zeros(N, bool)
speed_sum, speed_n = np.zeros(N), np.zeros(N)
key = jax.random.PRNGKey(1)
STEPS = 500
for t in range(STEPS):
  state.info["command"] = cmd
  state.info["steps_until_next_cmd"] = jp.full((N,), 1000, dtype=jp.int32)
  key, k = jax.random.split(key)
  a, _ = policy(state.obs, k)
  hist = jp.concatenate([a[None], hist[:-1]])
  applied = hist[delay, jp.arange(N)]
  state = step(state, applied)
  fell |= np.array(state.done) > 0
  if t >= 100:
    vx = np.array(jax.vmap(env.get_local_linvel)(state.data))[:, 0]
    ok = ~fell
    speed_sum[ok] += vx[ok]
    speed_n[ok] += 1
  if t == 0:
    print(f"compiled + first step in {time.time() - t0:.0f} s", flush=True)
print(f"{N} robots x {STEPS} steps in {time.time() - t0:.0f} s\n")

print(f"{'condition':32s} {'fell':>8s} {'speed (target ' + str(VX) + ')':>20s}")
for c, _ in CONDITIONS:
  idx = [i for i, l in enumerate(labels) if l == c]
  f = fell[idx]
  sp = speed_sum[idx].sum() / max(speed_n[idx].sum(), 1)
  print(f"{c:32s} {f.sum():>3d} / {len(idx):<3d} {sp:>14.2f} m/s")
