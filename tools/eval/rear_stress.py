"""Stress-test a trained rearing policy: one real-world difference at a time.

Training already randomizes mass, friction, servo stiffness, centre of mass and servo
delay (a random mix per robot). This is the exam: each condition on its own, at or past
the training range, plus things training never saw (pushes, a sagging battery). Every
condition runs S robots from the walking start bank (standing / mid-walk) for 6 s.

Reports per condition: reached the tripod (the env's test: full tripod held 0.2 s),
median arrival time, falls (the env's termination: rolled, tipped over backwards, or
fell 30 deg nose-down), and still in the tripod at the end.

  python tools/eval/rear_stress.py rr_run7a [robots per condition]   (Mac, ~10-20 min)
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_rear as rr  # noqa: E402

RUN = sys.argv[1]
S = int(sys.argv[2]) if len(sys.argv) > 2 else 16
run = REPO / "runs" / RUN
cfg = json.load(open(run / "config.json"))

# (label, changes). Training ranges: friction 0.3-1.0, mass x0.9-1.1, stiffness x0.7-1.3,
# COM +/-2 cm, delay 0-40 ms; no pushes, no battery sag.
CONDITIONS = [
    ("baseline", {}),
    ("friction 0.4 (floor, feet, tail)", dict(mu=0.4)),
    ("friction 0.3 (training min)", dict(mu=0.3)),
    ("friction 0.2 (past training)", dict(mu=0.2)),
    ("mass x0.9 (1.62 kg)", dict(mass=0.9)),
    ("mass x1.1 (1.98 kg)", dict(mass=1.1)),
    ("mass x1.2 (2.16 kg, past)", dict(mass=1.2)),
    ("COM 2 cm forward", dict(com=(0.02, 0, 0))),
    ("COM 2 cm back", dict(com=(-0.02, 0, 0))),
    ("servo stiffness x0.7", dict(kp=0.7)),
    ("servo stiffness x1.3", dict(kp=1.3)),
    ("delay 20 ms (1 step)", dict(delay=1)),
    ("delay 40 ms (2 steps)", dict(delay=2)),
    ("push 0.2 m/s fwd while holding", dict(push=0.2, push_t=2.0)),
    ("push 0.4 m/s fwd while holding", dict(push=0.4, push_t=2.0)),
    ("push 0.2 m/s back while holding", dict(push=-0.2, push_t=2.0)),
    ("push 0.4 m/s back while holding", dict(push=-0.4, push_t=2.0)),
    ("push 0.2 m/s back while getting up", dict(push=-0.2, push_t=0.4)),
    ("ALL: mu .4, m x1.1, kp x.8, 20 ms", dict(mu=0.4, mass=1.1, kp=0.8, delay=1)),
]


def make_env(sag=1.0):
  c = config_dict.ConfigDict(cfg["env"])
  c.impl = "jax"
  c.rear_config.ladder_fraction = 0.0      # real starts only
  c.pert_config.enable = False             # only this test's own pushes
  if "servo_config" in c:                  # delay applied per condition below, not sampled
    c.servo_config.delay_ms = [0.0, 0.0]
    c.servo_config.no_load_speed *= sag    # battery sag: slower...
  env = rr.RedPandaRear(config=c)
  env._stall = env._stall * sag            # ...and weaker
  return env


def load_policy(env):
  net = cfg["ppo"]["network_factory"]
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
      value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
      policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
  return jax.jit(ppo_networks.make_inference_fn(nets)(
      pickle.load(open(run / "params.pkl", "rb")), deterministic=True))


def run_batch(env, specs, seed):
  mj = env.mj_model
  N = len(specs)
  touch = [0] + [mj.geom(n).id for n in ["fl_foot", "fr_foot", "bl_foot", "br_foot", "tail_tip"]]
  torso = mj.body("torso").id

  def randomize(model):
    fr = np.tile(np.array(model.geom_friction), (N, 1, 1))
    mass = np.tile(np.array(model.body_mass), (N, 1))
    inertia = np.tile(np.array(model.body_inertia), (N, 1, 1))
    ipos = np.tile(np.array(model.body_ipos), (N, 1, 1))
    gain = np.tile(np.array(model.actuator_gainprm), (N, 1, 1))
    bias = np.tile(np.array(model.actuator_biasprm), (N, 1, 1))
    for i, s in enumerate(specs):
      if "mu" in s:
        fr[i, touch, 0] = s["mu"]
      if "mass" in s:
        mass[i, 1:] *= s["mass"]
        inertia[i, 1:] *= s["mass"]
      if "com" in s:
        ipos[i, torso] += np.array(s["com"])
      if "kp" in s:
        gain[i, :, 0] *= s["kp"]
        bias[i, :, 1] *= s["kp"]
    new = dict(geom_friction=fr, body_mass=mass, body_inertia=inertia, body_ipos=ipos,
               actuator_gainprm=gain, actuator_biasprm=bias)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({k: 0 for k in new})
    return model.tree_replace({k: jp.array(v) for k, v in new.items()}), in_axes

  venv = BraxDomainRandomizationVmapWrapper(env, randomize)
  policy = load_policy(env)
  reset, step = jax.jit(venv.reset), jax.jit(venv.step)
  delay = jp.array([s.get("delay", 0) for s in specs])
  push = np.array([s.get("push", 0.0) for s in specs])
  push_step = np.array([int(round(s.get("push_t", -1) / env.dt)) for s in specs])

  state = reset(jax.random.split(jax.random.PRNGKey(seed), N))
  hist = jp.zeros((3, N, env.action_size))
  key = jax.random.PRNGKey(seed + 1)
  T = env._config.episode_length
  arrive = np.full(N, np.nan)
  fell = np.zeros(N, bool)
  for t in range(T):
    hit = (push_step == t) & (push != 0)
    if hit.any():  # kick the torso along its heading (+ = toward the nose / front feet)
      R = np.array(state.data.xmat)[:, torso].reshape(N, 3, 3)
      fwd = R[:, :, 0] * np.array([1, 1, 0])
      fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
      dv = np.where(hit[:, None], fwd * push[:, None], 0.0)
      qvel = np.array(state.data.qvel)
      qvel[:, 0:3] += dv
      state = state.replace(data=state.data.replace(qvel=jp.array(qvel)))
    key, k = jax.random.split(key)
    a, _ = policy(state.obs, jax.random.split(k, N))
    hist = jp.concatenate([a[None], hist[:-1]])
    state = step(state, hist[delay, jp.arange(N)])
    newly = (np.array(state.metrics["arrived"]) > 0) & np.isnan(arrive) & ~fell
    arrive[newly] = np.array(state.metrics["arrive_time_s"])[newly]
    fell |= np.array(state.done) > 0
  feet = np.array(jax.vmap(env._feet_contact)(state.data))
  tail = np.array(jax.vmap(lambda d: env._found(d, "tail_tip"))(state.data))
  held = ~fell & feet[:, 2] & feet[:, 3] & tail & ~feet[:, 0] & ~feet[:, 1]
  return arrive, fell, held


t0 = time.time()
specs = [s for _, s in CONDITIONS for _ in range(S)]
arrive, fell, held = run_batch(make_env(), specs, seed=0)
sag = run_batch(make_env(sag=0.9), [{}] * S, seed=0)
print(f"{RUN}: {S} robots per condition from walking starts (standing / mid-walk), 6 s each "
      f"({time.time() - t0:.0f} s)\n")
print(f"{'condition':38s} {'reached':>9s} {'median time':>11s} {'fell':>6s} {'held at 6 s':>11s}")
rows = [(lab, slice(i * S, (i + 1) * S)) for i, (lab, _) in enumerate(CONDITIONS)]
results = {}
for lab, sl in rows + [("battery sag (torque + speed x0.9)", None)]:
  a, f, h = (arrive[sl], fell[sl], held[sl]) if sl is not None else sag
  ok = ~np.isnan(a)
  med = f"{np.median(a[ok]):.2f} s" if ok.any() else "-"
  flag = "" if ok.sum() >= 0.9 * len(a) and f.sum() == 0 else "  <--"
  print(f"{lab:38s} {ok.sum():3d}/{len(a):<3d}  {med:>9s}  {f.sum():3d}   {h.sum():4d}/{len(a):<3d}{flag}")
  results[lab] = dict(reached=int(ok.sum()), n=len(a), fell=int(f.sum()), held=int(h.sum()),
                      median_s=float(np.median(a[ok])) if ok.any() else None)
json.dump(results, open(run / "rear_stress.json", "w"), indent=1)
print(f"\n<-- = under 90% reached or any fall.  -> {run / 'rear_stress.json'}")
