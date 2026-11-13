"""Train a walking policy with Brax PPO (the same setup MuJoCo Playground uses for Go1).

Examples (run from the repo root):
  # setup check: Playground's own Go1 example, ~200M steps
  python tools/train_walk.py --env go1 --name go1_check
  # red panda run 1 (blind: policy doesn't see body speed)
  python tools/train_walk.py --env redpanda --name rp_run1
  # comparison run: policy sees body speed
  python tools/train_walk.py --env redpanda --name rp_sees_speed --sees_speed
  # tiny test on the Mac (CPU, a few seconds of training)
  python tools/train_walk.py --env redpanda --name smoke --tiny

Writes to runs/<name>/: progress.csv (one row per eval), config.json,
params.pkl (the trained policy, ~1 MB), and orbax checkpoints.
"""
import argparse
import csv
import functools
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Compatibility patch: Brax 0.14.2 (the newest) still calls
# jax.device_put_replicated, which JAX removed in 0.10. This is JAX's suggested
# replacement (docs.jax.dev/en/latest/migrate_pmap.html): one copy per device.
# Delete this block once Brax is updated.
def _device_put_replicated(x, devices):
  mesh = jax.sharding.Mesh(np.array(devices), ("d",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
  copies = lambda a: jnp.stack([jnp.asarray(a)] * len(devices))
  return jax.tree_util.tree_map(lambda a: jax.device_put(copies(a), sharding), x)


jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk  # noqa: E402

GO1 = "Go1JoystickFlatTerrain"

p = argparse.ArgumentParser()
p.add_argument("--env", choices=["go1", "redpanda"], required=True)
p.add_argument("--name", required=True, help="output folder name under runs/")
p.add_argument("--timesteps", type=int, default=None, help="default: Go1's 200M")
p.add_argument("--sees_speed", action="store_true", help="red panda policy sees body speed")
p.add_argument("--impl", default="warp", help="warp (NVIDIA GPU) or jax (Mac/CPU)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tiny", action="store_true", help="tiny CPU run to test the script")
args = p.parse_args()

out = REPO / "runs" / args.name
out.mkdir(parents=True, exist_ok=True)

# PPO settings: Playground's tuned Go1 settings for both robots (reward pass 1).
ppo_cfg = locomotion_params.brax_ppo_config(GO1)
if args.timesteps:
  ppo_cfg.num_timesteps = args.timesteps

# Environment.
if args.env == "go1":
  env_cfg = registry.get_default_config(GO1)
  env_cfg.impl = args.impl
  env = registry.load(GO1, config=env_cfg)
  eval_env = registry.load(GO1, config=env_cfg)
  randomizer = registry.get_domain_randomizer(GO1)  # friction, mass, etc.
else:
  env_cfg = red_panda_walk.default_config()
  env_cfg.impl = args.impl
  env_cfg.policy_sees_body_speed = args.sees_speed
  env = red_panda_walk.RedPandaWalk(config=env_cfg)
  eval_env = red_panda_walk.RedPandaWalk(config=env_cfg)
  randomizer = None  # no domain randomization yet (run 1)

extra = {}
if args.tiny:
  ppo_cfg.num_timesteps = 20_000
  ppo_cfg.num_envs = 8
  ppo_cfg.batch_size = 8
  ppo_cfg.num_minibatches = 4
  ppo_cfg.num_evals = 2
  ppo_cfg.num_resets_per_eval = 1
  extra["num_eval_envs"] = 8
  env_cfg.naconmax = 64  # buffers sized for 8 robots, not 8192

with open(out / "config.json", "w") as f:
  json.dump({"args": vars(args), "ppo": ppo_cfg.to_dict(), "env": env_cfg.to_dict()},
            f, indent=2, default=str)

# Progress: print and save one row per evaluation.
start = time.time()
rows = []


def progress(step, metrics):
  row = {"step": step, "minutes": round((time.time() - start) / 60, 2)}
  row.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/")})
  rows.append(row)
  with open(out / "progress.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
    w.writeheader()
    w.writerows(rows)
  print(f"step {step:>11,}  |  {row['minutes']:6.1f} min  |  "
        f"episode reward {row.get('eval/episode_reward', float('nan')):8.3f}  |  "
        f"episode length {row.get('eval/avg_episode_length', float('nan')):6.1f}",
        flush=True)


params = ppo_cfg.to_dict()
net_cfg = params.pop("network_factory")
network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)

print(f"devices: {jax.devices()}")
print(f"env: {args.env}  policy inputs: {env.observation_size}  outputs: {env.action_size}")
print(f"timesteps: {ppo_cfg.num_timesteps:,}  robots in parallel: {ppo_cfg.num_envs}", flush=True)

make_inference_fn, policy_params, _ = ppo.train(
    environment=env,
    eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    randomization_fn=randomizer,
    network_factory=network_factory,
    progress_fn=progress,
    seed=args.seed,
    save_checkpoint_path=str(out / "checkpoints"),
    **params,
    **extra,
)

with open(out / "params.pkl", "wb") as f:
  pickle.dump(jax.device_get(policy_params), f)
print(f"done in {(time.time() - start) / 60:.1f} min -> {out}")
