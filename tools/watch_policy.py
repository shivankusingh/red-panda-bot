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
from brax.training.agents.ppo import networks as ppo_networks
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk  # noqa: E402

# Joystick script: (seconds, forward m/s, sideways m/s, turn rad/s, label)
SCRIPT = [
    (4.0, 0.30, 0.0, 0.0, "forward 0.3"),
    (3.0, 0.20, 0.0, 0.4, "curve left"),
    (2.0, 0.00, 0.0, 0.0, "stop"),
    (3.0, -0.20, 0.0, 0.0, "backward 0.2"),
    (3.0, 0.00, 0.15, 0.0, "sideways 0.15"),
    (3.0, 0.00, 0.0, -0.5, "turn in place right"),
]

p = argparse.ArgumentParser()
p.add_argument("run", help="folder name under runs/")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
run = REPO / "runs" / args.run
cfg = json.load(open(run / "config.json"))

# Same env as training, but on the CPU.
env_cfg = red_panda_walk.default_config()
env_cfg.impl = "jax"
env_cfg.policy_sees_body_speed = cfg["args"].get("sees_speed", False)
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
  s = np.array(speeds[len(speeds) // 3:])  # skip the first third (speeding up)
  rows.append((label, (vx, vy, wz), s.mean(axis=0), fell))

print(f"\n{'segment':22s} {'commanded (fwd, side, turn)':30s} {'measured':30s} fell?")
for label, c, m, fell in rows:
  print(f"{label:22s} ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})          "
        f"({m[0]:+.2f}, {m[1]:+.2f}, {m[2]:+.2f})          {'YES' if fell else 'no'}")

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
