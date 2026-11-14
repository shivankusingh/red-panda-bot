"""Throughput benchmark: how fast does this machine run the red panda env?

For several robot counts it measures:
  - physics: env steps per second through the same wrappers training uses
    (domain randomization + auto-reset), with zero actions
  - the policy network (512-256-128): forward passes per second
and optionally saves a JAX profiler trace of a few steps at 8192 robots
(open trace/*.json.gz at https://ui.perfetto.dev to see which GPU kernels
take the time).

Compare with a training run's timing.json: training throughput below the
physics-only number = time spent on the network and the PPO update.

Run from the repo root (on the GPU pod):
  python tools/bench_env.py --name bench_run16 --trace
Writes runs/<name>/bench.json (and runs/<name>/trace/ with --trace).
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", default="bench")
p.add_argument("--impl", default="warp", help="warp (NVIDIA GPU) or jax (Mac/CPU)")
p.add_argument("--counts", default="1024,2048,4096,8192,16384", help="robot counts")
p.add_argument("--steps", type=int, default=100, help="timed steps per count")
p.add_argument("--trace", action="store_true", help="save a profiler trace at 8192 robots")
args = p.parse_args()
out = REPO / "runs" / args.name
out.mkdir(parents=True, exist_ok=True)
counts = [int(c) for c in args.counts.split(",")]

print(f"devices: {jax.devices()}", flush=True)
results = []
for n in counts:
  cfg = red_panda_walk.default_config()
  cfg.impl = args.impl
  cfg.naconmax = 4 * n  # contact buffer: 4 feet per robot (training uses 4 x 8192)
  env = red_panda_walk.RedPandaWalk(config=cfg)
  key = jax.random.PRNGKey(0)
  randomizer = functools.partial(red_panda_walk.make_domain_randomizer(env),
                                 rng=jax.random.split(key, n))
  wenv = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1,
                                        randomization_fn=randomizer)
  reset, step = jax.jit(wenv.reset), jax.jit(wenv.step)
  action = jp.zeros((n, env.action_size))

  # Training runs many steps inside ONE compiled loop (lax.scan); calling a jitted
  # step from Python once per step mostly measures per-call overhead (~24 ms on the
  # 4090, run 16's first benchmark). Time it both ways.
  def roll(s):
    return jax.lax.scan(lambda s, _: (wenv.step(s, action), None), s, None, length=args.steps)[0]
  roll = jax.jit(roll)

  t = time.time()
  state = reset(jax.random.split(key, n))
  jax.block_until_ready(state.reward)
  reset_s = time.time() - t
  t = time.time()
  state = step(state, action)          # first call compiles
  jax.block_until_ready(state.reward)
  compile_s = time.time() - t
  t = time.time()
  for _ in range(args.steps):
    state = step(state, action)
  jax.block_until_ready(state.reward)
  python_loop_s = time.time() - t
  jax.block_until_ready(roll(state).reward)  # compile the scanned loop
  t = time.time()
  state = roll(state)
  jax.block_until_ready(state.reward)
  physics_s = time.time() - t
  sps = n * args.steps / physics_s

  # Policy network forward pass on the same batch of observations.
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
      policy_obs_key="state", value_obs_key="privileged_state")
  obs_spec = jax.tree_util.tree_map(lambda x: specs.Array(x.shape[-1:], jp.dtype("float32")),
                                    state.obs)
  norm = running_statistics.init_state(obs_spec)
  pparams = nets.policy_network.init(key)
  fwd = jax.jit(lambda o: nets.policy_network.apply(norm, pparams, o))
  jax.block_until_ready(fwd(state.obs))
  t = time.time()
  for _ in range(args.steps):
    logits = fwd(state.obs)
  jax.block_until_ready(logits)
  net_s = time.time() - t

  row = {"robots": n, "reset_s": round(reset_s, 2), "compile_s": round(compile_s, 2),
         "physics_steps_per_s": round(sps), "ms_per_env_step": round(1000 * physics_s / args.steps, 2),
         "python_loop_ms_per_step": round(1000 * python_loop_s / args.steps, 2),
         "policy_forward_ms": round(1000 * net_s / args.steps, 3)}
  results.append(row)
  print(f"{n:6d} robots | compile {compile_s:5.1f} s | physics {sps:>10,.0f} steps/s "
        f"({row['ms_per_env_step']:6.2f} ms per step in one compiled loop; "
        f"{row['python_loop_ms_per_step']:6.2f} ms one call per step) | "
        f"policy forward {row['policy_forward_ms']:.3f} ms", flush=True)

  if args.trace and n == 8192:
    with jax.profiler.trace(str(out / "trace"), create_perfetto_trace=True):
      for _ in range(20):
        state = step(state, action)
        logits = fwd(state.obs)
      jax.block_until_ready((state.reward, logits))
    print(f"trace -> {out / 'trace'}", flush=True)

json.dump({"device": str(jax.devices()[0]), "impl": args.impl, "results": results},
          open(out / "bench.json", "w"), indent=2)
print(f"saved {out / 'bench.json'}")
