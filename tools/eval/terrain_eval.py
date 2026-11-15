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


tc = rpw.default_config().terrain_config
TILES = [(0, tc.flat_cols[0])] + [(r, 1) for r in ROWS]   # flat first, then lumpy rows
N = S * len(TILES)                                         # all tiles in one batch: one compile


def evaluate():
  """Walks S robots on each tile, all at once. Returns one log per tile."""
  c = config_dict.ConfigDict(cfg["env"])
  c.impl = "jax"
  c.pert_config.enable = False
  if "servo_config" in c:
    c.servo_config.delay_ms = [0.0, 0.0]
  rpw.use_terrain(c, 0, 1)  # terrain on; each robot's tile is passed to reset below
  env = rpw.RedPandaWalk(config=c)
  net = cfg["ppo"]["network_factory"]
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
      value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
      policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
  policy = jax.vmap(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
  tiles = jp.array([t for t in TILES for _ in range(S)])   # (N, 2): robot i's [level, col]

  # The whole 20 s walk runs on the device in one lax.scan (run 25: the old per-step
  # Python loop pulled data off the GPU every step and was CPU-bound: 226 s on a pod).
  @jax.jit
  def rollout(rng, key):
    state = jax.vmap(env.reset)(jax.random.split(rng, N), tiles)

    def one_step(carry, _):
      state, key = carry
      state.info["command"] = jp.tile(jp.array([VX, 0.0, 0.0]), (N, 1))
      state.info["steps_until_next_cmd"] = jp.full((N,), 1000, dtype=jp.int32)
      key, k = jax.random.split(key)
      a, _ = policy(state.obs, jax.random.split(k, N))
      state = jax.vmap(env.step)(state, a)
      up = ~jax.vmap(env._desired_contact)(state.info["gait_phase"], state.info["command"])
      out = dict(done=state.done > 0, up=up, catch=up & state.info["last_contact"],
                 vx=jax.vmap(env.get_local_linvel)(state.data)[:, 0],
                 swing_end=state.info["swing_end"], swing_peak=state.info["swing_end_peak"])
      return (state, key), out
    return jax.lax.scan(one_step, (state, key), None, length=STEPS)[1]

  log = {k: np.array(v) for k, v in rollout(jax.random.PRNGKey(0), jax.random.PRNGKey(1)).items()}
  rest = env._config.reward_config.foot_rest_height
  results = []
  for i in range(len(TILES)):
    lg = {k: v[:, i * S:(i + 1) * S] for k, v in log.items()}   # this tile's robots
    fell_by = np.maximum.accumulate(lg["done"], axis=0)        # (T, S): fell at or before t
    ok = ~fell_by[100:]                                         # skip 2 s settling; drop fallen
    catch = (lg["catch"][100:] & ok[..., None]).sum((0, 1))
    up = (lg["up"][100:] & ok[..., None]).sum((0, 1))
    speed = lg["vx"][100:][ok].mean() if ok.any() else 0.0
    # swing peak above the ground, per completed swing (by the gait clock, like swing_floor)
    ended = lg["swing_end"][100:] & ok[..., None]
    peaks = lg["swing_peak"][100:]
    lift = [(np.median(peaks[..., f][ended[..., f]]) - rest) * 100 if ended[..., f].any() else np.nan
            for f in range(4)]
    results.append((fell_by[-1].sum(), speed, catch / np.maximum(up, 1), lift))
  return results


t0 = time.time()
print(f"{RUN}: {S} robots x 20 s per row, forward {VX} m/s, nominal robot (no pushes / delay)")
print("toe catch = % of 'should be in the air' time that the foot is touching; "
      "lift = median swing peak above the ground under the foot\n")
print(f"{'tile':24s} {'fell':>7s} {'speed':>7s} | {'toe catch % LF  RF  LH  RH':>27s} | lift cm  LF   RF   LH   RH")
for (level, col), (f, sp, rate, lift) in zip(TILES, evaluate()):
  if col in tc.flat_cols:
    name = "flat"
  else:
    name = f"row {level} (lumps <= {tc.max_height * (level + 1) / tc.n_levels * 100:.2f} cm)"
  print(f"{name:24s} {f:>3d}/{S:<3d} {sp:>6.3f} | {'':6s}" + " ".join(f"{x * 100:4.0f}" for x in rate)
        + " | " + " ".join(f"{x:4.1f}" for x in lift), flush=True)
print(f"\n({time.time() - t0:.0f} s)")
