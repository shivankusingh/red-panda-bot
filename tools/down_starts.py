"""Bank of start states for coming back down, made by the rearing policy.

Coming-down episodes start where rearing actually leaves the robot: run the rearing
policy from walking starts and save the full state of every robot that is holding the
tail tripod, at several moments (so poses and small sways differ). Nominal robot, flat.

  python tools/down_starts.py rr_run11 [robots] [out]      (Mac, ~1 min)

Saves data/down_starts.npz: qpos [N, nq], qvel [N, nv], moving [N] (all 0: the
rearing env's start-bank format, so it can load it unchanged).
"""
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_rear as rr  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "rr_run11"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "data" / "down_starts.npz"
SAVE_AT = [100, 150, 200, 250, 299]  # steps (x 20 ms); arrival is ~0.9 s = 45 steps

run = REPO / "runs" / RUN
cfg = json.load(open(run / "config.json"))
c = config_dict.ConfigDict(cfg["env"])
c.impl = "jax"
c.naconmax = 16 * N
c.rear_config.ladder_fraction = 0.0  # real (walking) starts
c.pert_config.enable = False
env = rr.RedPandaRear(config=c)
net = cfg["ppo"]["network_factory"]
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
    value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
policy = jax.jit(jax.vmap(ppo_networks.make_inference_fn(nets)(
    pickle.load(open(run / "params.pkl", "rb")), deterministic=True)))
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

key = jax.random.PRNGKey(0)
state = reset(jax.random.split(key, N))
qpos, qvel = [], []
for i in range(max(SAVE_AT) + 1):
  key, k = jax.random.split(key)
  a, _ = policy(state.obs, jax.random.split(k, N))
  state = step(state, a)
  if i in SAVE_AT:
    contact = np.array(jax.vmap(env._feet_contact)(state.data))
    ok = np.array(jax.vmap(env._in_tripod)(state.data, contact)) & (np.array(state.done) == 0)
    qpos.append(np.array(state.data.qpos)[ok])
    qvel.append(np.array(state.data.qvel)[ok])
    print(f"step {i}: {ok.sum()} of {N} holding the tripod, saved", flush=True)
qpos, qvel = np.concatenate(qpos), np.concatenate(qvel)
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, qpos=qpos, qvel=qvel, moving=np.zeros(len(qpos), np.int8), source=np.array(RUN))
print(f"saved {len(qpos)} tripod states -> {OUT}")
