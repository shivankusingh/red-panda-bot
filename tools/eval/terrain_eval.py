"""How does a walking policy cope with lumps? Falls, speed and toe catches per terrain row.

S robots walk forward at VX for 20 s on one tile of each row (plus a flat tile), with
the nominal robot (no pushes, no delay). A "toe catch" = a foot touching the ground
while the gait clock says it should be in the air (the flat row shows the baseline:
scuffs that happen even with no lumps).

Usage: .venv/bin/python tools/eval/terrain_eval.py rp_run21a [robots=16] [vx=0.12] [rows=0,3,6,9]
"""
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as rpw  # noqa: E402

RUN = sys.argv[1]
S = int(sys.argv[2]) if len(sys.argv) > 2 else 16
VX = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
ROWS = [int(r) for r in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 3, 6, 9]
STEPS = 1000  # 20 s
run = REPO / "runs" / RUN
cfg = json.load(open(run / "config.json"))
params = pickle.load(open(run / "params.pkl", "rb"))


def evaluate(level, col):
  c = config_dict.ConfigDict(cfg["env"])
  c.impl = "jax"
  c.pert_config.enable = False
  if "servo_config" in c:
    c.servo_config.delay_ms = [0.0, 0.0]
  rpw.use_terrain(c, level, col)
  env = rpw.RedPandaWalk(config=c)
  net = cfg["ppo"]["network_factory"]
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
      value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
      policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
  policy = jax.jit(jax.vmap(ppo_networks.make_inference_fn(nets)(params, deterministic=True)))
  reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

  @jax.jit
  def measure(state):
    desired = jax.vmap(env._desired_contact)(state.info["gait_phase"], state.info["command"])
    touching = state.info["last_contact"]
    vx = jax.vmap(env.get_local_linvel)(state.data)[:, 0]
    return (~desired) & touching, ~desired, vx, jax.vmap(env._feet_height)(state.data)

  state = reset(jax.random.split(jax.random.PRNGKey(0), S))
  key = jax.random.PRNGKey(1)
  fell = np.zeros(S, bool)
  catch, up = np.zeros((S, 4)), np.zeros((S, 4))
  speed, n_speed = 0.0, 0
  peak = [[] for _ in range(4)]
  cur_peak = np.zeros((S, 4))
  for t in range(STEPS):
    state.info["command"] = jp.tile(jp.array([VX, 0.0, 0.0]), (S, 1))
    state.info["steps_until_next_cmd"] = jp.full((S,), 1000, dtype=jp.int32)
    key, k = jax.random.split(key)
    a, _ = policy(state.obs, jax.random.split(k, S))
    state = step(state, a)
    fell |= np.array(state.done) > 0
    if t < 100:  # skip the first 2 s (settling from the spawn drop)
      continue
    c_, u_, vx, fh = (np.array(x) for x in measure(state))
    ok = ~fell
    catch[ok] += c_[ok]
    up[ok] += u_[ok]
    speed += vx[ok].sum()
    n_speed += ok.sum()
    # swing peak height above the ground, per completed swing
    for f in range(4):
      for i in np.where(ok)[0]:
        if u_[i, f]:
          cur_peak[i, f] = max(cur_peak[i, f], fh[i, f])
        elif cur_peak[i, f] > 0:
          peak[f].append(cur_peak[i, f])
          cur_peak[i, f] = 0.0
  rate = catch.sum(0) / np.maximum(up.sum(0), 1)
  lift = [(np.median(p) - env._config.reward_config.foot_rest_height) * 100 if p else np.nan
          for p in peak]
  return fell.sum(), speed / max(n_speed, 1), rate, lift


t0 = time.time()
tc = rpw.default_config().terrain_config
print(f"{RUN}: {S} robots x 20 s per row, forward {VX} m/s, nominal robot (no pushes / delay)")
print("toe catch = % of 'should be in the air' time that the foot is touching; "
      "lift = median swing peak above the ground under the foot\n")
print(f"{'tile':24s} {'fell':>7s} {'speed':>7s} | {'toe catch % LF  RF  LH  RH':>27s} | lift cm  LF   RF   LH   RH")
for level, col in [(0, tc.flat_cols[0])] + [(r, 1) for r in ROWS]:
  if col in tc.flat_cols:
    name = "flat"
  else:
    name = f"row {level} (lumps <= {tc.max_height * (level + 1) / tc.n_levels * 100:.2f} cm)"
  f, sp, rate, lift = evaluate(level, col)
  print(f"{name:24s} {f:>3d}/{S:<3d} {sp:>6.3f} | {'':6s}" + " ".join(f"{x * 100:4.0f}" for x in rate)
        + " | " + " ".join(f"{x:4.1f}" for x in lift), flush=True)
print(f"\n({time.time() - t0:.0f} s)")
