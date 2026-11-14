"""Watch a trained walking policy: speed table, posture numbers, side-view GIF.

Runs the policy on the Mac (CPU) in the same env it trained in:
  1. a scripted joystick (forward, curve, stop, backward, sideways, turn) ->
     commanded vs. measured speed per segment;
  2. a fixed command (default 0.15 m/s straight ahead) -> mean torso height, each
     foot's touchdown / liftoff position relative to its own hip (stride and crouch
     as numbers, saved to runs/<run>/posture.json), the footfall chart, and a GIF
     from a side-view camera that tracks the robot.

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
p.add_argument("--vx", type=float, default=0.12,
               help="fixed forward command for the GIF (inside every run's range since run 14b)")
p.add_argument("--seconds", type=float, default=10.0, help="length of the fixed-command walk")
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

# 1) Scripted joystick -> speed table: does the policy follow commands?
state = reset(jax.random.PRNGKey(args.seed))
key = jax.random.PRNGKey(1)
rows = []
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

# 2) Fixed command -> posture numbers, footfall chart, side-view GIF.
FEET = ["fl", "fr", "bl", "br"]
LABEL = {"fl": "LF", "fr": "RF", "bl": "LH", "br": "RH"}
OUTWARD = {"fl": 1.0, "fr": -1.0, "bl": 1.0, "br": -1.0}  # +y is the robot's left
mm = env.mj_model
torso_id = mm.body("torso").id
hip_ids = np.array([mm.body(f"{l}_hip").id for l in FEET])
foot_sites = np.array(env._feet_site_id)
state = reset(jax.random.PRNGKey(args.seed))
cmd = jp.array([args.vx, 0.0, 0.0])
settle, n = int(2.0 / env.dt), int(args.seconds / env.dt)
qpos_log, heading_log, heights, gait_log = [], [], [], []
touch = {l: [] for l in FEET}   # foot - hip in the torso frame at touchdown (m)
lift = {l: [] for l in FEET}    # ... at liftoff
last = None
fell = False
for t in range(settle + n):
  state.info["command"] = cmd
  state.info["steps_until_next_cmd"] = jp.array(1000, dtype=jp.int32)
  key, k = jax.random.split(key)
  action, _ = policy(state.obs, k)
  state = step(state, action)
  fell |= bool(state.done)
  if t < settle:
    continue
  d = state.data
  R = np.array(d.xmat[torso_id]).reshape(3, 3)            # torso frame: x forward, y left
  rel = (np.array(d.site_xpos[foot_sites]) - np.array(d.xpos[hip_ids])) @ R
  contact = np.array(state.info["last_contact"]).astype(bool)
  if last is not None:
    for i, l in enumerate(FEET):
      if contact[i] and not last[i]:
        touch[l].append(rel[i])
      if last[i] and not contact[i]:
        lift[l].append(rel[i])
  last = contact
  qpos_log.append(np.array(d.qpos))
  heading_log.append(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
  heights.append(float(d.qpos[2]))
  gait_log.append((float(state.info["gait_phase"]) if env._gait_on else 0.0, contact))

h = 100 * np.array(heights)
target = 100 * env._gait.base_height if env._gait_on else float("nan")
print(f"\nposture, walking {args.vx:.2f} m/s straight for {args.seconds:.0f} s"
      f"{'  (FELL)' if fell else ''}:")
print(f"  torso height: mean {h.mean():.1f} cm (min {h.min():.1f}, max {h.max():.1f}); "
      f"target base_height {target:.1f} cm")
print("  each foot relative to its own hip, in the robot's frame (cm; + = ahead / outward):")
print(f"  {'foot':4s} {'touchdown ahead':>16s} {'liftoff ahead':>14s} {'stride':>7s} "
      f"{'touchdown outward':>18s} {'steps':>6s}")
posture = {"run": args.run, "vx": args.vx, "seconds": args.seconds, "fell": fell,
           "torso_height_cm": round(float(h.mean()), 2), "base_height_target_cm": target,
           "feet": {}}
for l in FEET:
  td, lo = np.array(touch[l]) * 100, np.array(lift[l]) * 100
  if len(td) == 0 or len(lo) == 0:
    print(f"  {LABEL[l]:4s} no full steps")
    continue
  out = OUTWARD[l] * td[:, 1]
  print(f"  {LABEL[l]:4s} {td[:, 0].mean():+8.1f} +/- {td[:, 0].std():3.1f} {lo[:, 0].mean():+8.1f} +/- "
        f"{lo[:, 0].std():3.1f} {td[:, 0].mean() - lo[:, 0].mean():6.1f} {out.mean():+12.1f} +/- "
        f"{out.std():3.1f} {len(td):5d}")
  posture["feet"][LABEL[l]] = {
      "touchdown_ahead_cm": round(float(td[:, 0].mean()), 2),
      "touchdown_ahead_std_cm": round(float(td[:, 0].std()), 2),
      "liftoff_ahead_cm": round(float(lo[:, 0].mean()), 2),
      "touchdown_outward_cm": round(float(out.mean()), 2), "steps": int(len(td))}
json.dump(posture, open(run / "posture.json", "w"), indent=2)

order = [("LH", 2), ("LF", 0), ("RH", 3), ("RF", 1)]
log = gait_log
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

spec = red_panda_walk.build_spec(env_cfg)  # same robot shape as the run
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
cam.distance, cam.elevation = 0.8, -8  # low side view: legs, crouch and stride visible

frames = []
for i, q in enumerate(qpos_log):
  if i % 2:  # 25 frames per second
    continue
  vis_data.qpos[:] = q
  mujoco.mj_forward(vis_model, vis_data)
  cam.lookat[:] = vis_data.qpos[:3]          # follow the torso
  cam.azimuth = heading_log[i] + 90.0        # stay side-on as the robot turns
  renderer.update_scene(vis_data, cam)
  frames.append(Image.fromarray(renderer.render()))

renderer.close()
out = run / "walk.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, loop=0)
print(f"\nvideo (side view, {args.vx:.2f} m/s): {out}  ({len(frames)} frames, {len(frames) / 25:.0f} s)")
print(f"posture numbers: {run / 'posture.json'}")
