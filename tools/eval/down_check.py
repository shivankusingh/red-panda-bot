"""Measure a trained coming-down policy, then hand off to the walking policy.

1. Come down: N robots from the tripod start bank (nominal robot, no random pushes),
   4 s. Per robot: landed (the env's test: four feet down, tail up, walking stance, held
   0.2 s), when, falls, front-foot touchdown speeds, final tilt and hip height, servo peaks.
2. Hand-off: each robot's final state goes to the walking policy (rp_run26 by default),
   which walks forward at 0.12 m/s for 2 s. Counts falls. A landing walking can't start
   from is no use.
3. GIFs of robot 0 (side + straight behind), come-down then walking.

  python tools/eval/down_check.py dn_run1 [robots] [walking run]
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
from mujoco_playground._src import mjx_env
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_down as rd  # noqa: E402
import red_panda_walk as rpw  # noqa: E402

RUN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
WALK = sys.argv[3] if len(sys.argv) > 3 else "rp_run26"
run = REPO / "runs" / RUN


def load(run_dir, env):
  cfg = json.load(open(run_dir / "config.json"))
  net = cfg["ppo"]["network_factory"]
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
      value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
      policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
  return jax.jit(jax.vmap(ppo_networks.make_inference_fn(nets)(
      pickle.load(open(run_dir / "params.pkl", "rb")), deterministic=True)))


# --- 1. Come down ---
cfg = json.load(open(run / "config.json"))
c = config_dict.ConfigDict(cfg["env"])
c.impl, c.naconmax = "jax", 16 * N
c.pert_config.enable = False
c.rear_config.ladder_fraction = 0.0  # real tripod starts only, never ladder rungs
env = rd.RedPandaDown(config=c)
policy = load(run, env)
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
m = env.mj_model
lt = np.array(env._lt_act)
names = [m.actuator(int(i)).name for i in lt]
front_v = np.array(env._foot_linvel_sensor_adr)[:2, 2]

key = jax.random.PRNGKey(0)
state = reset(jax.random.split(key, N))
landed = np.full(N, np.nan)
fell = np.zeros(N, bool)
touch_speeds = [[] for _ in range(N)]
last_contact = np.array(jax.vmap(env._feet_contact)(state.data))[:, :2]
last_vz = np.zeros((N, 2))
force, qlog = [], []
for t in range(c.episode_length):
  key, k = jax.random.split(key)
  a, _ = policy(state.obs, jax.random.split(k, N))
  state = step(state, a)
  contact = np.array(jax.vmap(env._feet_contact)(state.data))[:, :2]
  vz = np.array(state.data.sensordata)[:, front_v]
  for i, j in zip(*np.where(contact & ~last_contact)):
    touch_speeds[i].append(max(-last_vz[i, j], 0.0))
  last_contact, last_vz = contact, vz
  new = (np.array(state.metrics["arrived"]) > 0) & np.isnan(landed) & ~fell
  landed[new] = np.array(state.metrics["arrive_time_s"])[new]
  fell |= np.array(state.done) > 0
  force.append(np.where(~fell[:, None], np.abs(np.array(state.data.actuator_force))[:, lt], np.nan))
  qlog.append(np.array(state.data.qpos[0]))
ok = ~np.isnan(landed)
speeds = np.array([s for r in touch_speeds for s in r])
first = np.array([r[0] for r in touch_speeds if r])
F = np.array(force)
print(f"{RUN}: {N} robots from the tripod, {c.episode_length * env.dt:.0f} s, nominal robot\n")
print(f"landed (4 feet, tail up, walking stance, held 0.2 s): {ok.sum()}/{N}   fell: {fell.sum()}/{N}")
if ok.any():
  print(f"landing time: median {np.median(landed[ok]):.2f} s, fastest {landed[ok].min():.2f}, slowest {landed[ok].max():.2f} (deadline 1.5 s)")
if len(first):
  print(f"front-foot touchdowns: {len(speeds)} ({len(speeds) / N:.1f} per robot); first touchdown speed median {np.median(first):.2f} m/s, "
        f"worst {first.max():.2f}; any touchdown worst {speeds.max():.2f} m/s  (goal <= 0.3; passive topple 1.77; walking 0.05-0.30)")
pitch, _ = jax.vmap(env._torso_tilt)(state.data)
hip = 100 * np.array(state.data.xpos)[:, env._hind_thigh, 2].mean(1)
print(f"end (robots up): tilt {np.median(-np.degrees(np.array(pitch))[~fell]):.1f} deg (stance -2), hips {np.median(hip[~fell]):.1f} cm (16)")
print("servo peaks (99th pct |torque|; 60-70% of stall = 1.76-2.06 N*m):")
for nm in ["fl_hip", "fl_knee", "fr_hip", "fr_knee", "bl_hip", "br_hip", "bl_knee", "br_knee", "tail_lift"]:
  j = names.index(nm)
  print(f"  {nm:10s} {np.nanpercentile(F[:, :, j], 99):5.2f} N*m")

# --- 2. Hand-off to walking ---
wrun = REPO / "runs" / WALK
wc = config_dict.ConfigDict(json.load(open(wrun / "config.json"))["env"])
wc.impl = "jax"
wc.pert_config.enable = False
rpw.use_flat(wc)
wenv = rpw.RedPandaWalk(config=wc)
wpolicy = load(wrun, wenv)
wreset, wstep = jax.jit(jax.vmap(wenv.reset)), jax.jit(jax.vmap(wenv.step))
ws = wreset(jax.random.split(jax.random.PRNGKey(1), N))
qpos, qvel = state.data.qpos, state.data.qvel


def handoff(q, v, info):
  d = mjx_env.make_data(wenv.mj_model, qpos=q, qvel=v, ctrl=wenv._init_ctrl,
                        impl=wenv.mjx_model.impl.value, naconmax=wc.naconmax, njmax=wc.njmax)
  return mujoco.mjx.forward(wenv.mjx_model, d)


wdata = jax.vmap(handoff)(qpos, qvel, ws.info)
ws = ws.replace(data=wdata, obs=jax.vmap(wenv._get_obs)(wdata, ws.info))
cmd = jp.tile(jp.array([0.12, 0.0, 0.0]), (N, 1))
wfell = np.zeros(N, bool)
x0 = np.array(wdata.qpos[:, :2])
for t in range(100):
  ws.info["command"] = cmd
  ws.info["steps_until_next_cmd"] = jp.full((N,), 1000, dtype=jp.int32)
  key, k = jax.random.split(key)
  a, _ = wpolicy(ws.obs, jax.random.split(k, N))
  ws = wstep(ws, a)
  wfell |= np.array(ws.done) > 0
  qlog.append(np.array(ws.data.qpos[0]))
dist = np.linalg.norm(np.array(ws.data.qpos[:, :2]) - x0, axis=1)
up = ~fell
print(f"\nhand-off to {WALK} (walk 0.12 m/s for 2 s): fell {wfell[up].sum()}/{up.sum()}, "
      f"distance median {100 * np.median(dist[up & ~wfell]):.0f} cm (ideal 24)")
json.dump(dict(robots=N, landed=int(ok.sum()), fell=int(fell.sum()),
               landing_median_s=float(np.median(landed[ok])) if ok.any() else None,
               first_touch_median=float(np.median(first)) if len(first) else None,
               first_touch_worst=float(first.max()) if len(first) else None,
               handoff_fell=int(wfell[up].sum()), handoff_robots=int(up.sum())),
          open(run / "down_check.json", "w"), indent=1)

# --- 3. GIFs (robot 0: come down, then walk) ---
spec = rd.rr.rear_spec(c)
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.85, 0.85, 0.8], rgb2=[0.7, 0.7, 0.65], width=300, height=300)
mat = spec.add_material(name="grid", texrepeat=[20, 20])
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
spec.geom("floor").material = "grid"
vm = spec.compile()
vd = mujoco.MjData(vm)
ren = mujoco.Renderer(vm, height=360, width=480)
bl, br = vm.site("bl_foot").id, vm.site("br_foot").id
for view, name in [("side", "down.gif"), ("back", "down_back.gif")]:
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation = (0.9, -5) if view == "side" else (0.75, -2)
  frames, heading0 = [], None
  for i, q in enumerate(qlog):
    if i % 2:
      continue
    vd.qpos[:] = q
    mujoco.mj_forward(vm, vd)
    if heading0 is None:  # keep the camera fixed to the starting heading (it walks off forward)
      left = vd.site_xpos[bl] - vd.site_xpos[br]
      heading0 = np.degrees(np.arctan2(-left[0], left[1]))
    cam.azimuth = heading0 + (90 if view == "side" else 0)
    cam.lookat[:] = [q[0], q[1], 0.12]
    ren.update_scene(vd, cam)
    frames.append(Image.fromarray(ren.render()))
  frames[0].save(run / name, save_all=True, append_images=frames[1:], duration=40, loop=0)
ren.close()
print(f"\n-> {run / 'down_check.json'}, {run / 'down.gif'} (side), {run / 'down_back.gif'} (behind)")
