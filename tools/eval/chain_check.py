"""The whole Tier 1 chain in sim: walk -> rear up -> hold -> come down -> walk.

Three separate policies (walking, rearing, coming down), switched like the robot's switcher
will: each hand-off copies the robot's exact pose and speeds into the next policy's env, with
every servo holding the angle it is at (no jump). Timeline per robot:
  1. walk forward at 0.12 m/s for WALK_S
  2. "rear" command while walking -> rearing policy; it arrives (full tripod held 0.2 s) and
     keeps holding until REAR_S after the command
  3. "down" command (all robots together) -> coming-down policy
  4. the moment a robot lands (4 feet, tail up, walking stance, held 0.2 s) it switches to
     the walking policy, which walks it forward at 0.12 m/s for WALK2_S
A robot that hasn't landed DOWN_MAX_S after the command is handed to walking anyway (counted).

Nominal robot so the switching is what's tested: flat floor, no pushes, no joint offsets,
servo delay fixed at 20 ms (the middle of the trained 0-40 ms) in every policy.

  python tools/eval/chain_check.py [walk run] [rear run] [down run] [robots]
     (default rp_run26 rr_run11 dn_run5b 32)
"""
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
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_down as rd  # noqa: E402
import red_panda_rear as rr  # noqa: E402
import red_panda_walk as rpw  # noqa: E402

A = sys.argv[1:]
WALK = A[0] if len(A) > 0 else "rp_run26"
REAR = A[1] if len(A) > 1 else "rr_run11"
DOWN = A[2] if len(A) > 2 else "dn_run5b"
N = int(A[3]) if len(A) > 3 else 32
WALK_S, REAR_S, DOWN_MAX_S, WALK2_S = 3.0, 4.0, 3.0, 3.0
DT = 0.02
CMD = jp.array([0.12, 0.0, 0.0])
OUT = REPO / "runs" / DOWN / "chain"


def make_env(kind):
  run = {"walk": WALK, "rear": REAR, "down": DOWN}[kind]
  c = config_dict.ConfigDict(json.load(open(REPO / "runs" / run / "config.json"))["env"])
  c.impl = "jax"
  c.pert_config.enable = False
  c.servo_config.delay_ms = [20.0, 20.0]
  c.servo_config.joint_offset_deg = 0.0
  if kind == "walk":
    rpw.use_flat(c)
    env = rpw.RedPandaWalk(config=c)
  else:
    c.naconmax = 24 * N
    c.rear_config.ladder_fraction = 0.0
    env = (rr.RedPandaRear if kind == "rear" else rd.RedPandaDown)(config=c)
  net = json.load(open(REPO / "runs" / run / "config.json"))["ppo"]["network_factory"]
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
      value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
      policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
  policy = jax.jit(jax.vmap(ppo_networks.make_inference_fn(nets)(
      pickle.load(open(REPO / "runs" / run / "params.pkl", "rb")), deterministic=True)))
  return env, policy, jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))


def contact_of(env, data):
  return jp.array([data.sensordata[env.mj_model.sensor_adr[s]] > 0
                   for s in env._feet_floor_found_sensor])


def handoff_fn(env):
  """fresh (a reset state of `env`) + src physics -> the same robot, now run by `env`."""
  def one(fresh, src):
    q, v = src.qpos, src.qvel
    ctrl = (env._init_ctrl.at[env._leg_act].set(q[env._leg_qpos])
            .at[env._tail_act].set(q[env._tail_qpos]).at[env._neck_act].set(q[env._neck_qpos]))
    d = mjx_env.make_data(env.mj_model, qpos=q, qvel=v, ctrl=ctrl, impl=env.mjx_model.impl.value,
                          naconmax=env._config.naconmax, njmax=env._config.njmax)
    d = mjx.forward(env.mjx_model, d)
    info = dict(fresh.info)
    if "last_motor_targets" in info:
      info["last_motor_targets"] = jp.stack([d.ctrl, d.ctrl])
    info["last_contact"] = contact_of(env, d)
    info["last_leg_qvel"] = v[env._leg_qvel]
    if "command" in info and "steps_until_next_cmd" in info:   # walking: fixed command
      info["command"] = CMD
      info["steps_until_next_cmd"] = jp.array(10**6, dtype=jp.int32)
    return fresh.replace(data=d, obs=env._get_obs(d, info), info=info,
                         done=jp.zeros_like(fresh.done))
  return jax.jit(jax.vmap(one))


def select(mask, a, b):
  """Per robot: a where mask, else b (any batched pytree)."""
  def pick(x, y):
    m = mask.reshape(mask.shape + (1,) * (x.ndim - 1))
    return jp.where(m, x, y)
  return jax.tree.map(pick, a, b)


def tilt_of(env):
  return jax.jit(jax.vmap(lambda d: -jp.rad2deg(env._torso_tilt(d)[0])))


wenv, wpol, wreset, wstep = make_env("walk")
renv, rpol, rreset, rstep = make_env("rear")
denv, dpol, dreset, dstep = make_env("down")
to_rear, to_down, to_walk = handoff_fn(renv), handoff_fn(denv), handoff_fn(wenv)
tilt = tilt_of(renv)
hips = np.array([renv.mj_model.actuator(n).id for n in ("bl_hip", "br_hip")])
key = jax.random.PRNGKey(0)
keys = lambda k: jax.random.split(k, N)  # noqa: E731

log_q, phase_log = [], []          # robot 0, for the GIF
tilt_log, hip_log = [], []         # (steps, robots)


def record(state, phase):
  log_q.append(np.array(state.data.qpos[0]))
  phase_log.append(phase)
  tilt_log.append(np.array(tilt(state.data)))
  hip_log.append(np.abs(np.array(state.data.actuator_force)[:, hips]).max(1))


# 1. Walk
ws = to_walk(wreset(keys(jax.random.PRNGKey(1))), wreset(keys(jax.random.PRNGKey(1))).data)
walk1_fell = np.zeros(N, bool)
x0 = np.array(ws.data.qpos[:, :2])
for t in range(int(WALK_S / DT)):
  key, k = jax.random.split(key)
  a, _ = wpol(ws.obs, keys(k))
  ws = wstep(ws, a)
  ws.info["command"] = jp.tile(CMD, (N, 1))
  ws.info["steps_until_next_cmd"] = jp.full((N,), 10**6, dtype=jp.int32)
  walk1_fell |= np.array(ws.done) > 0
  record(ws, "walk")
walk1_dist = np.linalg.norm(np.array(ws.data.qpos[:, :2]) - x0, axis=1)
speed_at_cmd = np.linalg.norm(np.array(ws.data.qvel[:, :2]), axis=1)

# 2. Rear + hold
rs = to_rear(rreset(keys(jax.random.PRNGKey(2))), ws.data)
arrive = np.full(N, np.nan)
rear_fell = np.zeros(N, bool)
for t in range(int(REAR_S / DT)):
  key, k = jax.random.split(key)
  a, _ = rpol(rs.obs, keys(k))
  rs = rstep(rs, a)
  arr = np.array(rs.info["arrived"])
  new = arr & np.isnan(arrive)
  arrive[new] = np.array(rs.info["arrive_time"])[new]
  rear_fell |= np.array(rs.done) > 0
  record(rs, "rear" if not arr[0] else "hold")
in_tripod_at_cmd = np.array(jax.vmap(renv._in_tripod)(rs.data, jax.vmap(renv._feet_contact)(rs.data)))
tilt_at_cmd = np.array(tilt(rs.data))

# 3 + 4. Come down; each robot switches to walking the moment it lands
ds = to_down(dreset(keys(jax.random.PRNGKey(3))), rs.data)
wfresh = wreset(keys(jax.random.PRNGKey(4)))
ws = to_walk(wfresh, ds.data)
switched = np.zeros(N, bool)
land = np.full(N, np.nan)
switch_xy = np.zeros((N, 2))
steps_walked = np.zeros(N, int)
end_xy = np.full((N, 2), np.nan)
down_fell, walk2_fell = np.zeros(N, bool), np.zeros(N, bool)
hip_after_switch = np.zeros(N)
t = 0
while True:
  key, k1, k2 = jax.random.split(key, 3)
  ad, _ = dpol(ds.obs, keys(k1))
  aw, _ = wpol(ws.obs, keys(k2))
  ds = dstep(ds, ad)
  ws = wstep(ws, aw)
  ws.info["command"] = jp.tile(CMD, (N, 1))
  ws.info["steps_until_next_cmd"] = jp.full((N,), 10**6, dtype=jp.int32)
  t += 1
  steps_walked[switched] += 1
  walk2_fell |= switched & (np.array(ws.done) > 0)
  down_fell |= ~switched & (np.array(ds.done) > 0)
  hip_now = np.abs(np.array(ws.data.actuator_force)[:, hips]).max(1)
  early = switched & (steps_walked <= int(0.5 / DT))         # first 0.5 s of walking
  hip_after_switch = np.where(early, np.maximum(hip_after_switch, hip_now), hip_after_switch)
  landed_now = np.array(ds.info["arrived"]) & ~switched
  timeout = ~switched & (t * DT >= DOWN_MAX_S)
  go = landed_now | timeout
  if go.any():
    land[landed_now] = t * DT
    ws = select(jp.array(go), to_walk(wfresh, ds.data), ws)
    switched |= go
    switch_xy[go] = np.array(ds.data.qpos[go, :2])
  done_walking = switched & (steps_walked == int(WALK2_S / DT))
  end_xy[done_walking] = np.array(ws.data.qpos[done_walking, :2])
  shown = ws if switched[0] else ds
  record(shown, "walk" if switched[0] else "down")
  if (steps_walked >= int(WALK2_S / DT)).all() or t > 1000:
    break

walk2_dist = np.linalg.norm(end_xy - switch_xy, axis=1)
landed = ~np.isnan(land)

# --- Report
pct = lambda m: f"{int(m.sum())}/{N}"  # noqa: E731
print(f"chain: {WALK} -> {REAR} -> {DOWN} -> {WALK}, {N} robots, nominal robot, delay 20 ms\n")
print(f"1. walk {WALK_S:.0f} s at 0.12 m/s:   fell {pct(walk1_fell)}, distance median "
      f"{100 * np.median(walk1_dist):.0f} cm (ideal {100 * 0.12 * WALK_S:.0f}), speed at 'rear' "
      f"{100 * np.median(speed_at_cmd):.0f} cm/s")
ok = ~np.isnan(arrive)
print(f"2. rear (from walking):      arrived {pct(ok)}, fell {pct(rear_fell)}, arrival median "
      f"{np.nanmedian(arrive):.2f} s (slowest {np.nanmax(arrive):.2f}); at 'down': in tripod "
      f"{pct(in_tripod_at_cmd)}, tilt median {np.median(tilt_at_cmd):.0f} deg, held "
      f"{REAR_S - np.nanmax(arrive):.1f}-{REAR_S - np.nanmin(arrive):.1f} s")
print(f"3. come down:                landed {pct(landed)}, fell {pct(down_fell)}, landing median "
      f"{np.nanmedian(land):.2f} s (slowest {np.nanmax(land):.2f}); not landed by "
      f"{DOWN_MAX_S:.0f} s -> handed over anyway: {pct(switched & ~landed)}")
print(f"4. walk off at landing:      fell {pct(walk2_fell)}, distance in {WALK2_S:.0f} s median "
      f"{100 * np.nanmedian(walk2_dist):.0f} cm (ideal {100 * 0.12 * WALK2_S:.0f}), slowest "
      f"{100 * np.nanmin(walk2_dist):.0f} cm; hind-hip peak in the first 0.5 s median "
      f"{np.median(hip_after_switch):.2f} N*m (max {hip_after_switch.max():.2f})")
all_ok = (~walk1_fell & ok & ~rear_fell & landed & ~down_fell & ~walk2_fell)
print(f"\nwhole chain without a fall or a missed step: {pct(all_ok)}")
OUT.mkdir(parents=True, exist_ok=True)
json.dump(dict(robots=N, walk1_fell=int(walk1_fell.sum()), rear_arrived=int(ok.sum()),
               rear_fell=int(rear_fell.sum()), arrive_median=float(np.nanmedian(arrive)),
               landed=int(landed.sum()), down_fell=int(down_fell.sum()),
               land_median=float(np.nanmedian(land)), walk2_fell=int(walk2_fell.sum()),
               walk2_dist_cm=float(100 * np.nanmedian(walk2_dist)), all_ok=int(all_ok.sum())),
          open(OUT / "chain.json", "w"), indent=1)

# --- GIF of robot 0 (side view, follows the robot), phase written on each frame
spec = rr.rear_spec(renv._config)
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.85, 0.85, 0.8], rgb2=[0.7, 0.7, 0.65], width=300, height=300)
mat = spec.add_material(name="grid", texrepeat=[20, 20])
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
spec.geom("floor").material = "grid"
vm = spec.compile()
vd = mujoco.MjData(vm)
ren = mujoco.Renderer(vm, height=360, width=480)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation = 1.0, -8
bl, br = vm.site("bl_foot").id, vm.site("br_foot").id
frames, heading0 = [], None
for i, (q, ph) in enumerate(zip(log_q, phase_log)):
  if i % 2:
    continue
  vd.qpos[:] = q
  mujoco.mj_forward(vm, vd)
  if heading0 is None:
    left = vd.site_xpos[bl] - vd.site_xpos[br]
    heading0 = np.degrees(np.arctan2(-left[0], left[1]))
  cam.azimuth = heading0 + 90
  cam.lookat[:] = [q[0], q[1], 0.12]
  ren.update_scene(vd, cam)
  img = Image.fromarray(ren.render())
  ImageDraw.Draw(img).text((10, 10), f"{i * DT:4.1f} s  {ph}", fill=(255, 255, 0))
  frames.append(img)
frames[0].save(OUT / "chain.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
ren.close()
print(f"-> {OUT / 'chain.json'}, {OUT / 'chain.gif'} (robot 0, side)")
