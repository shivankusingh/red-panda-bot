"""Measure a trained rearing policy: does it arrive, how fast, and what does it cost the servos?

Runs N robots for one episode (6 s) from the start bank, nominal robot (no domain
randomization), deterministic policy. Per robot: arrived (the env's own test: full tripod
held 0.2 s), arrival time, fell, final tilt and hip height; per servo: peak torque and
% of time pinned / over 65% of stall. Also a contact timeline and a side-view GIF of
robot 0.

  python tools/eval/rear_check.py rr_run1 [robots] [rung]
  -> runs/<run>/rear_check.json, runs/<run>/rear.gif
  With a ladder rung (0 = 70 deg ... 6 = 10 deg, tools/rear_ladder.py) every robot starts
  from that pose instead -> rear_check_rung<r>.json, rear_rung<r>.gif.
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
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_rear as rr  # noqa: E402

RUN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
RUNG = int(sys.argv[3]) if len(sys.argv) > 3 else None
TAG = "" if RUNG is None else f"_rung{RUNG}"
run = REPO / "runs" / RUN
cfg = json.load(open(run / "config.json"))
c = config_dict.ConfigDict(cfg["env"])
c.impl = "jax"
c.naconmax = 16 * N
if "rear_config" in c:  # real starts only (standing / mid-walk), unless a rung was asked for
  c.rear_config.ladder_fraction = 0.0 if RUNG is None else 1.0
env = rr.RedPandaRear(config=c)
net = cfg["ppo"]["network_factory"]
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
    value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
policy = jax.jit(jax.vmap(ppo_networks.make_inference_fn(nets)(
    pickle.load(open(run / "params.pkl", "rb")), deterministic=True)))
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

m = env.mj_model
names = [m.actuator(int(i)).name for i in np.array(env._lt_act)]
stall = np.array(env._stall)[np.array(env._lt_act)]
T = c.episode_length
key = jax.random.PRNGKey(0)
state = reset(jax.random.split(key, N))
if RUNG is not None:
  state.info["rung"] = jp.full((N,), RUNG, dtype=jp.int32)
moving = np.linalg.norm(np.array(state.data.qvel[:, :2]), axis=1) > 0.03  # started mid-walk
fell_at = np.full(N, np.nan)
arrive = np.full(N, np.nan)
force, pinned, qlog, contacts = [], [], [], []
# Cheat checks (things the sim allows but the real robot can't do):
#   knees/hips below the floor (only feet, torso and tail collide with it),
#   tail forward of the hind hips under the body (no part-to-part collisions),
#   tail tip slamming down (vertical speed just before it touches).
LEG_BODIES = [m.body(f"{l}_{p}").id for l in ["fl", "fr", "bl", "br"] for p in ["upper", "lower"]]
HIND_HIP = [m.body("bl_upper").id, m.body("br_upper").id]
TORSO = m.body("torso").id
TAIL_PTS = [m.body("tail_1").id, m.body("tail_2").id]
TIP = np.array([-0.165, 0.0, 0.0])  # tail tip in tail_2's frame (tools/eval/tripod_static.py)
low_joint, tail_fwd, tip_z, tail_on = [], [], [], []
for i in range(T):
  key, k = jax.random.split(key)
  act, _ = policy(state.obs, jax.random.split(k, N))
  state = step(state, act)
  t = (i + 1) * env.dt
  done = np.array(state.done) > 0
  fell_at[np.isnan(fell_at) & done] = t
  alive = np.isnan(fell_at) | (fell_at == t)
  newly = np.array(state.metrics["arrived"]) > 0
  arrive[newly & alive & np.isnan(arrive)] = np.array(state.metrics["arrive_time_s"])[newly & alive & np.isnan(arrive)]
  f = np.abs(np.array(state.data.actuator_force)[:, np.array(env._lt_act)])
  force.append(np.where(alive[:, None], f, np.nan))
  pinned.append(np.where(alive, np.array(state.metrics["pinned_servos"]), np.nan))
  qlog.append(np.array(state.data.qpos[0]))
  feet = np.array(jax.vmap(env._feet_contact)(state.data))[0]
  tail = bool(np.array(jax.vmap(lambda d: env._found(d, "tail_tip"))(state.data))[0])
  contacts.append((feet, tail))
  xpos, xmat = np.array(state.data.xpos), np.array(state.data.xmat).reshape(N, -1, 3, 3)
  low_joint.append(np.where(alive, xpos[:, LEG_BODIES, 2].min(1), np.nan))
  R = xmat[:, TORSO]                                   # torso frame: +x = forward
  hip_x = np.einsum("nji,nj->ni", R, xpos[:, HIND_HIP].mean(1) - xpos[:, TORSO])[:, 0]
  tip = xpos[:, TAIL_PTS[1]] + np.einsum("nij,j->ni", xmat[:, TAIL_PTS[1]], TIP)
  pts = np.stack([xpos[:, TAIL_PTS[0]], xpos[:, TAIL_PTS[1]], tip], 1)
  fwd = np.einsum("nji,nkj->nki", R, pts - xpos[:, TORSO][:, None])[:, :, 0].max(1) - hip_x
  tail_fwd.append(np.where(alive, fwd, np.nan))
  tip_z.append(tip[:, 2])
  tail_on.append(np.array(jax.vmap(lambda d: env._found(d, "tail_tip"))(state.data)))
fell = ~np.isnan(fell_at)
pitch = np.array(state.metrics["pitch_up_deg"])
hip = np.array(state.metrics["hip_height_cm"])
F = np.array(force)  # [T, N, 15]

print(f"{RUN}: {N} robots, {T * env.dt:.0f} s, nominal robot ({moving.sum()} started mid-walk)\n")
ok = ~np.isnan(arrive)
print(f"arrived (full tripod held 0.2 s): {ok.sum()}/{N}   fell: {fell.sum()}/{N}")
if ok.any():
  a = arrive[ok]
  print(f"arrival time: median {np.median(a):.2f} s, fastest {a.min():.2f}, slowest {a.max():.2f}  "
        f"(on time for 0.8 s: {(a <= 0.8).sum()}/{N}; physics floor ~0.44-0.47 s)")
  for lab, sel in [("standing starts", ~moving), ("mid-walk starts", moving)]:
    s = ok & sel
    print(f"  {lab:16s}: arrived {s.sum()}/{sel.sum()}" + (f", median {np.median(arrive[s]):.2f} s" if s.any() else ""))
still = ~fell
print(f"end of episode (robots still up): tilt {np.nanmedian(pitch[still]):.0f} deg (target 70), "
      f"hips {np.nanmedian(hip[still]):.1f} cm (target 16)")

print(f"\nservos (robots while up): peak = 99th percentile |torque|; rated 0.98, peak rule 65% stall = {0.65 * 2.94:.2f}, stall 2.94 N*m")
print(f"  {'servo':10s} {'peak':>6s} {'% time > 65% stall':>19s}")
rows = {}
for j, n in enumerate(names):
  if n.endswith("hip_side") or n == "tail_wag2":
    continue
  fj = F[:, :, j]
  pk = np.nanpercentile(fj, 99)
  over = np.nanmean(fj > 0.65 * stall[j]) * 100
  rows[n] = dict(peak=round(float(pk), 2), over_pct=round(float(over), 1))
  print(f"  {n:10s} {pk:6.2f} {over:18.1f}%")
print(f"servos pinned (>= 95% of available torque), average over time: {np.nanmean(np.array(pinned)):.2f} of 15")

LJ, TF, TZ, TO = np.array(low_joint), np.array(tail_fwd), np.array(tip_z), np.array(tail_on)
vz = np.diff(TZ, axis=0, prepend=TZ[:1]) / env.dt          # tail tip vertical speed
lands = TO & ~np.roll(TO, 1, axis=0)
lands[0] = False
land_v = -vz[np.roll(lands, -1, axis=0)]                   # speed on the step before touching
ghost = np.nanmin(LJ, 0) < 0.0
under = np.nanmax(TF, 0) > 0.0
print("\ncheat checks:")
print(f"  #2 ghost joints (a hip or knee below the floor): {ghost.sum()}/{N} robots, lowest {100 * np.nanmin(LJ):.1f} cm")
print(f"  #3 tail under the body (forward of the hind hips): {under.sum()}/{N} robots, furthest {100 * np.nanmax(TF):+.1f} cm")
if land_v.size:
  print(f"  #5 tail touchdowns: {land_v.size}, downward speed median {np.median(land_v):.2f} m/s, max {land_v.max():.2f} m/s "
        f"(walking feet landed at 0.05-0.30 m/s)")
else:
  print("  #5 tail touchdowns: none")

print("\nrobot 0 contact timeline, 1 char = 0.1 s (█ = touching):")
for lab, idx in [("LF", 0), ("RF", 1), ("LH", 2), ("RH", 3), ("tail", None)]:
  row = ""
  for k in range(0, T, 5):
    feet, tail = contacts[k]
    row += "█" if (tail if idx is None else feet[idx]) else "·"
  print(f"  {lab:4s} {row}")
print(f"  robot 0: arrived {arrive[0]:.2f} s" if ok[0] else "  robot 0: did not arrive", "| fell" if fell[0] else "")

json.dump(dict(robots=N, arrived=int(ok.sum()), fell=int(fell.sum()),
               arrive_median_s=float(np.median(arrive[ok])) if ok.any() else None,
               arrive_times=[None if np.isnan(x) else round(float(x), 2) for x in arrive],
               end_tilt_deg=float(np.nanmedian(pitch[still])) if still.any() else None,
               end_hip_cm=float(np.nanmedian(hip[still])) if still.any() else None,
               rung=RUNG, servos=rows, ghost_robots=int(ghost.sum()), lowest_joint_cm=float(100 * np.nanmin(LJ)),
               tail_under_robots=int(under.sum()),
               tail_land_speed_max=float(land_v.max()) if land_v.size else None), open(run / f"rear_check{TAG}.json", "w"), indent=1)

# Side-view GIF of robot 0.
spec = rr.rear_spec(c)
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.85, 0.85, 0.8], rgb2=[0.7, 0.7, 0.65], width=300, height=300)
mat = spec.add_material(name="grid", texrepeat=[20, 20])
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
spec.geom("floor").material = "grid"
vm = spec.compile()
vd = mujoco.MjData(vm)
ren = mujoco.Renderer(vm, height=360, width=480)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 0.9, -5, 90
frames = []
for i, q in enumerate(qlog):
  if i % 2:
    continue
  vd.qpos[:] = q
  mujoco.mj_forward(vm, vd)
  cam.lookat[:] = [q[0], q[1], 0.12]
  ren.update_scene(vd, cam)
  frames.append(Image.fromarray(ren.render()))
ren.close()
frames[0].save(run / f"rear{TAG}.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
print(f"\n-> {run / f'rear_check{TAG}.json'}, {run / f'rear{TAG}.gif'}")
