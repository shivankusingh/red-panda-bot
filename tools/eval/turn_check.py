"""Does the walking policy turn on command? Tier 1: "walk and turn on gamepad command".

Each joystick command is held for SECONDS on N robots at once (no shoves); measured over the
last MEASURE_S (after speeding up). Per command: turn rate vs commanded (from the torso's
heading, unwrapped), forward and sideways speed (body frame), falls, and the share of time
with at least 3 feet down (the gait's design: always 3 feet down). Plus a top-down plot:
every robot's path, rotated into its own starting frame, over the ideal path.

  python tools/eval/turn_check.py [walk run] [--random]      (default rp_run26)

--random: each robot gets a random world from training's randomizer (friction, servo
stiffness, mass, torso balance point) plus the run's random servo delay and calibration error.
"""
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as rpw  # noqa: E402

RANDOM = "--random" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RUN = ARGS[0] if ARGS else "rp_run26"
N, SECONDS, MEASURE_S = 32, 6.0, 4.0
# (label, forward m/s, sideways m/s, turn rad/s); trained range: +/-0.14, +/-0.07, +/-0.5
COMMANDS = [
    ("straight", 0.12, 0.0, 0.0),
    ("turn in place left", 0.0, 0.0, 0.5),
    ("turn in place right", 0.0, 0.0, -0.5),
    ("slow turn in place", 0.0, 0.0, 0.25),
    ("curve left", 0.12, 0.0, 0.3),
    ("curve right", 0.12, 0.0, -0.3),
    ("backward", -0.10, 0.0, 0.0),
    ("stop", 0.0, 0.0, 0.0),
]

run = REPO / "runs" / RUN
cfg = json.load(open(run / "config.json"))
c = config_dict.ConfigDict(cfg["env"])
c.impl = "jax"
c.pert_config.enable = False
rpw.use_flat(c)
if not RANDOM:
  c.servo_config.delay_ms = [20.0, 20.0]
  c.servo_config.joint_offset_deg = 0.0
env = rpw.RedPandaWalk(config=c)
net = cfg["ppo"]["network_factory"]
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
    value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
policy = jax.jit(jax.vmap(ppo_networks.make_inference_fn(nets)(
    pickle.load(open(run / "params.pkl", "rb")), deterministic=True)))
if RANDOM:
  randomize = rpw.make_domain_randomizer(env)
  venv = BraxDomainRandomizationVmapWrapper(env, lambda m: randomize(m, jax.random.split(jax.random.PRNGKey(7), N)))
  reset, step = jax.jit(venv.reset), jax.jit(venv.step)
else:
  reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))


def yaw_of(qpos):
  w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


local_vel = jax.jit(jax.vmap(env.get_local_linvel))
contacts = jax.jit(jax.vmap(lambda d: jp.array(
    [d.sensordata[env.mj_model.sensor_adr[s]] > 0 for s in env._feet_floor_found_sensor])))

steps = int(SECONDS / env.dt)
m0 = steps - int(MEASURE_S / env.dt)
results = []
for ci, (label, vx, vy, wz) in enumerate(COMMANDS):
  s = reset(jax.random.split(jax.random.PRNGKey(100 + ci), N))
  # Start at rest: walking's reset gives robots a random initial speed (training practice
  # for recovering), which once tipped a robot over during "stop" before it had settled.
  s = s.replace(data=s.data.replace(qvel=jp.zeros_like(s.data.qvel)))
  cmd = jp.tile(jp.array([vx, vy, wz]), (N, 1))
  key = jax.random.PRNGKey(ci)
  xy, yaw, vel, feet = [], [], [], []
  fell = np.zeros(N, bool)
  for t in range(steps):
    s.info["command"] = cmd
    s.info["steps_until_next_cmd"] = jp.full((N,), 10**6, dtype=jp.int32)
    key, k = jax.random.split(key)
    a, _ = policy(s.obs, jax.random.split(k, N))
    s = step(s, a)
    fell |= np.array(s.done) > 0
    q = np.array(s.data.qpos)
    xy.append(q[:, :2]); yaw.append(yaw_of(q))
    vel.append(np.array(local_vel(s.data))[:, :2]); feet.append(np.array(contacts(s.data)).sum(1))
  xy, yaw, vel, feet = map(np.array, (xy, yaw, vel, feet))   # (steps, robots, ...)
  yaw_u = np.unwrap(yaw, axis=0)
  turn = (yaw_u[-1] - yaw_u[m0]) / MEASURE_S
  # Paths in each robot's own starting frame (start at 0, facing +x).
  rel = xy - xy[0]
  c0, s0 = np.cos(-yaw[0]), np.sin(-yaw[0])
  path = np.stack([rel[..., 0] * c0 - rel[..., 1] * s0, rel[..., 0] * s0 + rel[..., 1] * c0], -1)
  results.append(dict(label=label, cmd=(vx, vy, wz), turn=turn, fwd=vel[m0:, :, 0].mean(0),
                      side=vel[m0:, :, 1].mean(0), total=yaw_u[-1] - yaw_u[0], three=(feet[m0:] >= 3).mean(0), fell=fell,
                      path=path))

print(f"{RUN} turning check, {N} robots, {'RANDOMIZED robots' if RANDOM else 'nominal robot, delay 20 ms'}, "
      f"no shoves; measured over the last {MEASURE_S:.0f} of {SECONDS:.0f} s (median over robots)\n")
print(f"{'command':20s} {'asked (fwd, turn)':>18s} | {'turn rad/s':>10s} {'% of asked':>10s} "
      f"{'(worst)':>8s} | {'fwd cm/s':>8s} {'side cm/s':>9s} | {'>=3 feet':>8s} | fell")
for r in results:
  vx, _, wz = r["cmd"]
  pct = f"{100 * np.median(r['turn']) / wz:9.0f}%" if wz else f"{'-':>10s}"
  worst = f"{100 * (r['turn'] / wz).min():7.0f}%" if wz else f"{'-':>8s}"
  print(f"{r['label']:20s} ({100 * vx:+5.0f}, {wz:+5.2f})   | {np.median(r['turn']):+10.2f} {pct} {worst} | "
        f"{100 * np.median(r['fwd']):+8.1f} {100 * np.median(r['side']):+9.1f} | "
        f"{100 * np.median(r['three']):7.0f}% | {int(r['fell'].sum())}/{N}")
for r in results:
  if not r["cmd"][2] and r["label"] != "stop":
    print(f"  {r['label']}: heading drift {np.median(np.abs(r['turn'])):.3f} rad/s "
          f"({np.degrees(np.median(np.abs(r['turn']))) * SECONDS:.0f} deg over {SECONDS:.0f} s)")

# --- Top-down plot: one small panel per command, all robots' paths over the ideal one.
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e0"
fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), facecolor="#fcfcfb")
t = np.arange(1, steps + 1) * env.dt
for ax, r in zip(axes.flat, results):
  vx, _, wz = r["cmd"]
  if abs(wz) > 1e-6:
    ideal = np.stack([vx / wz * np.sin(wz * t), vx / wz * (1 - np.cos(wz * t))], -1)
  else:
    ideal = np.stack([vx * t, 0 * t], -1)
  for i in range(N):
    ax.plot(100 * r["path"][:, i, 0], 100 * r["path"][:, i, 1], color=BLUE, lw=1.0, alpha=0.35)
  ax.plot(100 * ideal[:, 0], 100 * ideal[:, 1], color=MUTED, lw=2, ls="--")
  ax.plot([0], [0], "o", color=INK, ms=4)
  if abs(wz) > 1e-6 and abs(vx) < 1e-6:   # turning in place: show the heading turned instead
    ax.text(0.03, 0.05, f"turned {np.degrees(np.median(r['total'])):.0f}° "
            f"(asked {np.degrees(wz * SECONDS):.0f}°)", transform=ax.transAxes, color=MUTED, fontsize=9)
  ax.set_title(f"{r['label']}  ({100 * vx:+.0f} cm/s, {wz:+.2f} rad/s)", fontsize=10, color=INK)
  ax.set_aspect("equal", adjustable="datalim")
  ax.grid(color=GRID, lw=0.6)
  ax.tick_params(colors=MUTED, labelsize=8)
  for sp in ax.spines.values():
    sp.set_color(GRID)
  ax.set_facecolor("#fcfcfb")
fig.suptitle(f"{RUN}: top-down paths of {N} robots over {SECONDS:.0f} s (blue), ideal path dashed, "
             f"start = black dot facing right (cm)" + ("; randomized robots" if RANDOM else ""),
             fontsize=11, color=INK)
fig.tight_layout()
out = run / ("turn_check_random.png" if RANDOM else "turn_check.png")
fig.savefig(out, dpi=110)
print(f"\n-> {out}")
