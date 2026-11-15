"""Hardware feasibility from sim: torque -> current -> heat / battery, all 18 servos.

STS3215 C018 (reseller specs, 12 V): stall 2.94 N*m @ 2.7 A, rated 0.98 N*m @ 0.9 A,
no-load 0.18 A, no-load speed 4.7 rad/s. Lumped motor model at the output shaft:
  current I = I0 + |torque| / kt,  kt = 2.94 / (2.7 - 0.18) N*m/A
  heat ~ I_rms^2 * R,  R = 12 V / 2.7 A
Braking moments are counted like motoring (conservative).
"""
import json, pickle, sys
import jax, jax.numpy as jp, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper
REPO = __import__("pathlib").Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as r

RUN = sys.argv[1]
run = f"{REPO}/runs/{RUN}"
cfg = json.load(open(f"{run}/config.json"))
net = cfg["ppo"]["network_factory"]
STALL, I_STALL, I0, RATED, I_RATED, W0 = 2.94, 2.7, 0.18, 0.98, 0.9, 4.7
KT = STALL / (I_STALL - I0)
R = 12.0 / I_STALL
N, STEPS = 16, 1000  # 16 robots x 20 s, the env's own random commands (incl. stops)


def make_env(volts):
  c = config_dict.ConfigDict(cfg["env"]); c.impl = "jax"; r.use_flat(c)
  c.pert_config.enable = False
  c.servo_config.delay_ms = [0.0, 0.0]
  c.servo_config.no_load_speed = W0 * volts / 12.0     # motor speed ~ voltage
  env = r.RedPandaWalk(config=c)
  env._stall = env._stall * volts / 12.0                  # stall torque ~ voltage
  return env


def rollout(env, mass_scale):
  def rand(model):
    mass = jp.tile(model.body_mass * mass_scale, (N, 1))
    inertia = jp.tile(model.body_inertia * mass_scale, (N, 1, 1))
    in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace(
        {"body_mass": 0, "body_inertia": 0})
    return model.tree_replace({"body_mass": mass, "body_inertia": inertia}), in_axes
  venv = BraxDomainRandomizationVmapWrapper(env, rand)
  nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
      value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
      policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
  policy = jax.jit(ppo_networks.make_inference_fn(nets)(pickle.load(open(f"{run}/params.pkl", "rb")), deterministic=True))
  reset, step = jax.jit(venv.reset), jax.jit(venv.step)
  s = reset(jax.random.split(jax.random.PRNGKey(0), N))
  key = jax.random.PRNGKey(1)
  tau, fell, speed_err = [], np.zeros(N, bool), []
  for t in range(STEPS):
    key, k = jax.random.split(key)
    a, _ = policy(s.obs, k)
    s = step(s, a)
    fell |= np.array(s.done) > 0
    if t >= 50:
      tau.append(np.abs(np.array(s.data.actuator_force)))
      cmd = np.array(s.info["command"])[:, 0]
      vx = np.array(jax.vmap(env.get_local_linvel)(s.data))[:, 0]
      speed_err.append(np.where(fell, np.nan, np.abs(vx - cmd)))
  return np.array(tau), fell, np.nanmean(speed_err)


env12 = make_env(12.0)
m = env12.mj_model
names = [m.actuator(i).name for i in range(m.nu)]
groups = {"hip_side": [i for i, n in enumerate(names) if n.endswith("hip_side")],
          "hip": [i for i, n in enumerate(names) if n.endswith("_hip")],
          "knee": [i for i, n in enumerate(names) if n.endswith("knee")],
          "tail_wag": [names.index("tail_wag")], "tail_lift": [names.index("tail_lift")],
          "tail_wag2": [names.index("tail_wag2")], "neck_turn": [names.index("neck_turn")],
          "neck_nod": [names.index("neck_nod")], "head_tilt": [names.index("head_tilt")]}

for mass_scale, label in [(1.0, "1.80 kg")]:
  tau, fell, err = rollout(env12, mass_scale)       # (T, N, 18)
  cur = I0 + tau / KT                                # amps per servo
  print(f"\n=== {RUN} at {label}, 12 V, random commands + head motion, 16 robots x 20 s "
        f"(fell {fell.sum()}/16) ===")
  print(f"{'servo':10s} {'avg N*m':>8s} {'RMS N*m':>8s} {'peak99':>7s} {'max':>6s} "
        f"{'RMS amps':>9s} {'vs rated 0.9 A':>15s} {'heat W':>7s}")
  for g, idx in groups.items():
    x = tau[:, :, idx]; i_rms = np.sqrt(np.mean(cur[:, :, idx] ** 2, axis=(0, 1))).max()
    print(f"{g:10s} {x.mean():8.2f} {np.sqrt((x ** 2).mean()):8.2f} {np.percentile(x, 99):7.2f} "
          f"{x.max():6.2f} {i_rms:9.2f} {100 * i_rms / I_RATED:13.0f}% {i_rms ** 2 * R:7.1f}")
  total = cur.sum(axis=2)                            # all 18 servos, per robot per step
  print(f"all 18 servos together: avg {total.mean():.1f} A, 99th pct {np.percentile(total, 99):.1f} A, "
        f"max {total.max():.1f} A (sampled every 20 ms)")
  pi_amps = 10.0 / 11.1  # Pi 5 ~10 W through a ~90% 5 V regulator
  avg_all = total.mean() + pi_amps
  print(f"with Pi 5 (~{pi_amps:.1f} A): avg {avg_all:.1f} A -> a 3S 2200 mAh pack (80% usable) "
        f"lasts ~{0.8 * 2.2 / avg_all * 60:.0f} min of walking")

