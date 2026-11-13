"""Watch a trained walking policy: scripted joystick -> GIF video + speed table.

Runs the policy on the Mac (CPU) in the same env it trained in, with a fixed
joystick script (forward, curve, stop, backward, sideways), saves a GIF that
follows the robot, and prints commanded vs. measured speed for each segment.

Run from the repo root:
  .venv/bin/python tools/watch_policy.py rp_run1
  open runs/rp_run1/walk.gif
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from ml_collections import config_dict
from brax.training.agents.ppo import networks as ppo_networks
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk  # noqa: E402

# Joystick script: (seconds, forward m/s, sideways m/s, turn rad/s, label)
SCRIPT = [
    (4.0, 0.30, 0.0, 0.0, "forward"),
    (3.0, 0.20, 0.0, 0.4, "curve left"),
    (2.0, 0.00, 0.0, 0.0, "stop"),
    (3.0, -0.20, 0.0, 0.0, "backward"),
    (3.0, 0.00, 0.15, 0.0, "sideways"),
    (3.0, 0.00, 0.0, -0.5, "turn in place right"),
]

p = argparse.ArgumentParser()
p.add_argument("run", help="folder name under runs/")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--pushes", action="store_true", help="keep the run's random pushes on")
p.add_argument("--delay_ms", type=float, default=0.0,
               help="fixed motor delay (runs 9+ with the servo model); default 0")
args = p.parse_args()
run = REPO / "runs" / args.run
cfg = json.load(open(run / "config.json"))

# Same env as training, but on the CPU.
env_cfg = config_dict.ConfigDict(cfg["env"])  # the settings this run trained with
env_cfg.impl = "jax"
# Show the walk itself: no random shoves unless asked, one fixed delay, nominal robot.
if "pert_config" in env_cfg:
  env_cfg.pert_config.enable = env_cfg.pert_config.enable and args.pushes
if "servo_config" in env_cfg:
  env_cfg.servo_config.delay_ms = [args.delay_ms, args.delay_ms]
env = red_panda_walk.RedPandaWalk(config=env_cfg)

# Rebuild the network shape, then load the trained weights into it.
net = cfg["ppo"]["network_factory"]
networks = ppo_networks.make_ppo_networks(
    env.observation_size,
    env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
    value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"],
    value_obs_key=net["value_obs_key"],
)
params = pickle.load(open(run / "params.pkl", "rb"))
policy = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))
reset, step = jax.jit(env.reset), jax.jit(env.step)

# Run the script.
state = reset(jax.random.PRNGKey(args.seed))
key = jax.random.PRNGKey(1)
qpos_log, rows = [], []
gait_log = []  # (phase or step, contacts) during the first segment (forward)
# Keep the script inside this run's trained command range (e.g. run 5+ walks at <= 0.2 m/s).
cmd_max = np.array(env_cfg.command_config.a)
script_max = np.abs(np.array([[vx, vy, wz] for _, vx, vy, wz, _ in SCRIPT])).max(axis=0)
shrink = np.minimum(1.0, cmd_max / script_max)
SCRIPT = [(sec, vx * shrink[0], vy * shrink[1], wz * shrink[2], label)
          for sec, vx, vy, wz, label in SCRIPT]
for seconds, vx, vy, wz, label in SCRIPT:
  cmd = jp.array([vx, vy, wz])
  speeds, fell = [], False
  for _ in range(int(seconds / env.dt)):
    state.info["command"] = cmd
    state.info["steps_until_next_cmd"] = jp.array(1000, dtype=jp.int32)  # no random new commands
    key, k = jax.random.split(key)
    action, _ = policy(state.obs, k)
    state = step(state, action)
    qpos_log.append(np.array(state.data.qpos))
    lin = np.array(env.get_local_linvel(state.data))
    gyro = np.array(env.get_gyro(state.data))
    speeds.append([lin[0], lin[1], gyro[2]])
    fell |= bool(state.done)
    if label == SCRIPT[0][4]:
      gait_log.append((float(state.info["gait_phase"]),
                       np.array(state.info["last_contact"])))
  s = np.array(speeds[len(speeds) // 3:])  # skip the first third (speeding up)
  rows.append((label, (vx, vy, wz), s.mean(axis=0), fell))

print(f"\n{'segment':22s} {'commanded (fwd, side, turn)':30s} {'measured':30s} fell?")
for label, c, m, fell in rows:
  print(f"{label:22s} ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})          "
        f"({m[0]:+.2f}, {m[1]:+.2f}, {m[2]:+.2f})          {'YES' if fell else 'no'}")

# Footfall chart while walking forward (skip the first second: speeding up).
# Feet in the spec's order: LH, LF, RH, RF  (FEET order is fl, fr, bl, br).
order = [("LH", 2), ("LF", 0), ("RH", 3), ("RF", 1)]
log = gait_log[50:]
if env._gait_on:  # fold by the stride clock into 20 bins of 5%
  print("\nfootfall chart, folded by the stride clock (█ = down most of that 5%):")
  bins = [[] for _ in range(20)]
  for phase, c in log:
    bins[min(int(phase * 20), 19)].append(c)
  for name, i in order:
    row = "".join("█" if np.mean([c[i] for c in b]) > 0.5 else "·" for b in bins)
    print(f"  {name}  {row}")
else:  # no clock: show 1 second of raw timeline (1 char = 0.04 s)
  print("\nfootfall timeline, 1 s (1 char = 0.04 s):")
  for name, i in order:
    print(f"  {name}  " + "".join("█" if log[k][1][i] else "·" for k in range(0, 50, 2)))

# Render: a copy of the model with a checkered floor (looks only, same physics).
spec = mujoco.MjSpec.from_file(str(red_panda_walk.XML_PATH))
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                 builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.85, 0.85, 0.8], rgb2=[0.7, 0.7, 0.65], width=300, height=300)
mat = spec.add_material(name="grid", texrepeat=[20, 20])
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
spec.geom("floor").material = "grid"
vis_model = spec.compile()
vis_data = mujoco.MjData(vis_model)
renderer = mujoco.Renderer(vis_model, height=360, width=480)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 1.0, -20, 120

frames = []
for i, q in enumerate(qpos_log):
  if i % 2:  # 25 frames per second
    continue
  vis_data.qpos[:] = q
  mujoco.mj_forward(vis_model, vis_data)
  cam.lookat[:] = vis_data.qpos[:3]  # follow the torso
  renderer.update_scene(vis_data, cam)
  frames.append(Image.fromarray(renderer.render()))

renderer.close()
out = run / "walk.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, loop=0)
print(f"\nvideo: {out}  ({len(frames)} frames, {len(frames) / 25:.0f} s)")
