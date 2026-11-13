"""Smoke test for sim/red_panda_walk.py: does the env run and give sensible numbers?

Checks: input sizes, standing still with zero actions (tail and head moving), random
actions (no NaNs), the height cutoff, and the sees-body-speed switch.

Run from the repo root:  .venv/bin/python tools/smoke_test_walk.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
import red_panda_walk as rpw  # noqa: E402

env = rpw.RedPandaWalk()
reset = jax.jit(env.reset)
step = jax.jit(env.step)

state = reset(jax.random.PRNGKey(0))
print(f"policy inputs: {state.obs['state'].shape[0]}   critic inputs: "
      f"{state.obs['privileged_state'].shape[0]}   policy outputs: {env.action_size}")

m = env.mj_model
lift_adr = m.joint("tail_lift").qposadr[0]
wag_adr = m.joint("tail_wag").qposadr[0]
turn_adr = m.joint("neck_turn").qposadr[0]

# 1. Zero actions = "hold the home pose" for 5 s while the tail sways.
print("\nzero actions (hold home), 5 s:")
print("   t |  height | done | reward | tail wag  tail lift | head turn | feet down")
zero = jp.zeros(env.action_size)
for i in range(250):
  state = step(state, zero)
  if (i + 1) % 50 == 0:
    d = state.data
    contacts = int(sum(np.array(state.info["last_contact"])))
    print(f" {(i+1)*env.dt:3.0f}s | {float(d.qpos[2])*100:5.1f} cm |  {int(state.done)}   |"
          f" {float(state.reward):.4f} | {np.degrees(float(d.qpos[wag_adr])):+6.1f}°   "
          f"{np.degrees(float(d.qpos[lift_adr])):+6.1f}°   |  {np.degrees(float(d.qpos[turn_adr])):+6.1f}°  |   {contacts}/4")

# 2. Random actions: anything blow up?
print("\nrandom actions, 200 steps:")
state = reset(jax.random.PRNGKey(1))
key = jax.random.PRNGKey(2)
dones, rewards = 0, []
for i in range(200):
  key, k = jax.random.split(key)
  state = step(state, jax.random.uniform(k, (env.action_size,), minval=-1, maxval=1))
  rewards.append(float(state.reward))
  dones += int(state.done)
  if state.done:
    state = reset(k)
obs_ok = bool(jp.all(jp.isfinite(state.obs["state"])))
print(f"  reward min/mean/max: {min(rewards):.4f} / {np.mean(rewards):.4f} / {max(rewards):.4f}")
print(f"  episodes ended: {dones}   inputs all finite: {obs_ok}")

# 3. Height cutoff: check the rule directly. (Teleporting the torso low and
#    stepping doesn't work: the feet end up inside the floor and get launched back up.)
state = reset(jax.random.PRNGKey(3))
print()
for h, want in [(0.12, False), (0.09, True)]:
  d = mjx.forward(env.mjx_model, state.data.replace(qpos=state.data.qpos.at[2].set(h)))
  print(f"torso at {h*100:.0f} cm -> episode ends: {bool(env._get_termination(d))} (want {want})")

# 4. The comparison switch.
cfg = rpw.default_config()
cfg.policy_sees_body_speed = True
env2 = rpw.RedPandaWalk(config=cfg)
print(f"policy_sees_body_speed=True -> policy inputs: "
      f"{env2.reset(jax.random.PRNGKey(0)).obs['state'].shape[0]} (want 60)")
