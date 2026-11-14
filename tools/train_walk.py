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
params.pkl (the trained policy, ~1 MB), orbax checkpoints, and timing.json
(where the wall-clock time went: imports, env build, setup/compile, each
training chunk, each evaluation, saving).
"""
import time
T0 = time.time()  # before the heavy imports, so their cost is measured too

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
p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
               help="override one red panda env setting (repeatable), e.g. "
                    "reward_config.scales.swing_height=-600.0")
p.add_argument("--ppo", action="append", default=[], metavar="KEY=VALUE",
               help="override one PPO setting (repeatable), e.g. num_evals=3")
p.add_argument("--no_checkpoints", action="store_true",
               help="skip the ~85 MB orbax checkpoints (speed experiments)")
args = p.parse_args()
T_IMPORTS = time.time()

out = REPO / "runs" / args.name
out.mkdir(parents=True, exist_ok=True)

# PPO settings: Playground's tuned Go1 settings for both robots (reward pass 1).
ppo_cfg = locomotion_params.brax_ppo_config(GO1)
if args.timesteps:
  ppo_cfg.num_timesteps = args.timesteps
for item in args.ppo:  # e.g. num_evals=3, num_envs=16384
  key, value = item.split("=", 1)
  if key not in ppo_cfg:
    raise SystemExit(f"--ppo: no PPO setting named {key}")
  ppo_cfg[key] = json.loads(value)
  print(f"ppo override: {key} = {ppo_cfg[key]}")

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
  for item in args.set:  # one-setting experiments without editing the env file
    key, value = item.split("=", 1)
    *path, last = key.split(".")
    node = env_cfg
    for k in path:
      node = node[k]
    if last not in node:
      raise SystemExit(f"--set: no setting named {key}")
    node[last] = json.loads(value)
    print(f"override: {key} = {node[last]}")
  env = red_panda_walk.RedPandaWalk(config=env_cfg)
  eval_env = red_panda_walk.RedPandaWalk(config=env_cfg)
  rc = env_cfg.get("randomize_config", None)  # run 10: friction, mass, servo stiffness...
  randomizer = red_panda_walk.make_domain_randomizer(env) if rc and rc.enable else None

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

T_ENV = time.time()
with open(out / "config.json", "w") as f:
  json.dump({"args": vars(args), "ppo": ppo_cfg.to_dict(), "env": env_cfg.to_dict()},
            f, indent=2, default=str)

# Progress: print and save one row per evaluation.
start = time.time()
rows = []


evals = []  # (seconds since training started, eval seconds) per progress call


def progress(step, metrics):
  now = time.time() - start
  evals.append((now, float(metrics.get("eval/epoch_eval_time", 0.0)), int(step)))
  row = {"step": step, "minutes": round(now / 60, 2)}
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
    save_checkpoint_path=None if args.no_checkpoints else str(out / "checkpoints"),
    **params,
    **extra,
)

T_TRAIN = time.time()
with open(out / "params.pkl", "wb") as f:
  pickle.dump(jax.device_get(policy_params), f)
T_END = time.time()

# Where the time went. Brax runs one evaluation before training (step 0), then
# alternates a training chunk and an evaluation; checkpoints save after each eval.
# So: setup = time to the first progress call minus its eval; each chunk = gap
# between progress calls minus that call's eval (includes the checkpoint save,
# and the first chunk also includes compiling the training step).
setup_s = evals[0][0] - evals[0][1] if evals else 0.0
chunks = [(evals[i][2], evals[i][0] - evals[i - 1][0] - evals[i][1]) for i in range(1, len(evals))]
timing = {
    "imports_s": round(T_IMPORTS - T0, 1),
    "env_build_s": round(T_ENV - T_IMPORTS, 1),
    "setup_and_first_compile_s": round(setup_s, 1),
    "eval_s": [round(e[1], 1) for e in evals],
    "eval_total_s": round(sum(e[1] for e in evals), 1),
    "train_chunk_s": [round(c[1], 1) for c in chunks],
    "train_total_s": round(sum(c[1] for c in chunks), 1),
    "first_chunk_extra_s": round(chunks[0][1] - np.median([c[1] for c in chunks[1:]]), 1) if len(chunks) > 2 else None,
    "after_training_s": round(T_END - T_TRAIN + (T_TRAIN - start - (evals[-1][0] if evals else 0)), 1),
    "total_s": round(T_END - T0, 1),
    "steps": int(ppo_cfg.num_timesteps),
    "train_steps_per_s": round(ppo_cfg.num_timesteps / max(sum(c[1] for c in chunks), 1e-9)),
    "checkpoints": not args.no_checkpoints,
    "ppo_overrides": args.ppo,
}
json.dump(timing, open(out / "timing.json", "w"), indent=2)
print("\nwhere the time went (seconds):")
for k in ["imports_s", "env_build_s", "setup_and_first_compile_s", "eval_total_s",
          "train_total_s", "first_chunk_extra_s", "after_training_s", "total_s"]:
  print(f"  {k:28s} {timing[k]}")
print(f"  training throughput: {timing['train_steps_per_s']:,} steps/s")
print(f"done in {(time.time() - start) / 60:.1f} min -> {out}")
