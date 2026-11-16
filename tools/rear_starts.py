"""Bank of start states for rearing, made by the walking policy.

Rearing episodes start where walking actually leaves the robot (the hand-off), not
from a perfect pose: half the robots stand still (joystick centred), half walk with a
random forward / sideways / turn command, and we save their full state at several
random moments (mid-stride with a foot in the air included). On flat ground.

  python tools/rear_starts.py rp_run26 [robots] [out]     (Mac, MJX jax; a few minutes)

Saves data/rear_starts.npz: qpos [N, nq], qvel [N, nv], moving [N] (1 = was walking).
Regenerate whenever the body or the walking policy changes.
"""
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as rpw  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "rp_run26"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "data" / "rear_starts.npz"
SAVE_AT = [150, 190, 230, 270, 299]  # steps (x 20 ms) at which every robot's state is saved

run = REPO / "runs" / RUN
cfg = json.load(open(run / "config.json"))
c = config_dict.ConfigDict(cfg["env"])
c.impl = "jax"
c.pert_config.enable = False
rpw.use_flat(c)
env = rpw.RedPandaWalk(config=c)
net = cfg["ppo"]["network_factory"]
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
    value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
policy = jax.jit(ppo_networks.make_inference_fn(nets)(
    pickle.load(open(run / "params.pkl", "rb")), deterministic=True))

# Commands: first half stand still, second half walk (random within the training ranges).
key = jax.random.PRNGKey(0)
key, k1 = jax.random.split(key)
a = jp.array(c.command_config.a)
walk_cmd = jax.random.uniform(k1, (N, 3), minval=-a, maxval=a)
walk_cmd = walk_cmd.at[:, 0].set(jp.abs(walk_cmd[:, 0]))  # mostly forward, like real use
moving = np.arange(N) >= N // 2
cmd = jp.where(jp.array(moving)[:, None], walk_cmd, 0.0)

reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))
state = reset(jax.random.split(key, N))
qpos, qvel, mov = [], [], []
for i in range(max(SAVE_AT) + 1):
    state.info["command"] = cmd
    state.info["steps_until_next_cmd"] = jp.full((N,), 10_000, dtype=jp.int32)
    key, k = jax.random.split(key)
    act, _ = policy(state.obs, k)
    state = step(state, act)
    if i in SAVE_AT:
        ok = np.array(state.done) == 0
        qpos.append(np.array(state.data.qpos)[ok])
        qvel.append(np.array(state.data.qvel)[ok])
        mov.append(moving[ok])
        print(f"step {i}: saved {ok.sum()} of {N} (fallen: {N - ok.sum()})", flush=True)
qpos, qvel, mov = np.concatenate(qpos), np.concatenate(qvel), np.concatenate(mov)
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, qpos=qpos, qvel=qvel, moving=mov.astype(np.int8),
                    source=np.array(RUN))
print(f"saved {len(qpos)} start states ({mov.sum()} walking, {(~mov.astype(bool)).sum()} standing) -> {OUT}")
