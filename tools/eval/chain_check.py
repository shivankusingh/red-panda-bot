"""The whole Tier 1 chain in sim: walk -> rear up -> hold -> come down -> walk.

Three separate policies (walking, rearing, coming down), switched like the robot's switcher
will: each hand-off copies the robot's exact pose and speeds into the next policy's env, with
every servo holding the angle it is at (no jump). Timeline per robot:
  1. walk forward at 0.12 m/s for WALK_S
  2. "rear" command while walking -> rearing policy; it arrives (full tripod held 0.2 s) and
     keeps holding until REAR_S after the command
  3. "down" command (all robots together) -> coming-down policy
  4. the moment a robot is "ready to walk" it switches to the walking policy, which walks it
     forward at 0.12 m/s for WALK2_S. Ready to walk (the switcher's rule, user's choice
     2026-11-18) = all 4 feet down + tail tip up + body still (torso < 5 cm/s and < 0.5 rad/s)
     for READY_HOLD_S in a row. No tilt or height limits: the training "landed" box (tilt -2
     +/-5 deg, hips 16 +/-1.5 cm) is stricter than walking needs; it is still reported.
A robot not ready DOWN_MAX_S after the command is handed to walking anyway (counted).

Nominal robot so the switching is what's tested: flat floor, no pushes, no joint offsets,
servo delay fixed at 20 ms (the middle of the trained 0-40 ms) in every policy.

  python tools/eval/chain_check.py [walk run] [rear run] [down run] [robots] [--random] [--legs]
     (default rp_run26 rr_run11 dn_run5b 32)

--legs: also log every robot's hip/knee angles, speeds and torques through the whole chain and
report what a rod-linkage leg (knee servo coaxial with the hip, parallelogram rod to the shin)
would need: the knee servo sets the shin's direction (hip + knee), so its angle, speed and the
hip servo's torque (hip - knee, by virtual work) differ from the sim's joints. -> legs.json

--random: a realistic robot instead. Each robot gets one random world from training's own
randomizer (friction 0.3-1.0, servo stiffness x0.7-1.3, mass x0.9-1.1, torso balance point
+/-2 cm), its own servo delay (0-40 ms) and calibration error (+/-2 deg per leg servo), all
kept the same through every policy, and each policy's training shoves are on.
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
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_down as rd  # noqa: E402
import red_panda_rear as rr  # noqa: E402
import red_panda_walk as rpw  # noqa: E402

RANDOM = "--random" in sys.argv
LEGS = "--legs" in sys.argv
A = [a for a in sys.argv[1:] if not a.startswith("--")]
WALK = A[0] if len(A) > 0 else "rp_run26"
REAR = A[1] if len(A) > 1 else "rr_run11"
DOWN = A[2] if len(A) > 2 else "dn_run5b"
N = int(A[3]) if len(A) > 3 else 32
WALK_S, REAR_S, DOWN_MAX_S, WALK2_S = 3.0, 4.0, 3.0, 3.0
READY_HOLD_S, STILL_LIN, STILL_ANG = 0.2, 0.05, 0.5   # switcher: s, m/s, rad/s
DT = 0.02
CMD = jp.array([0.12, 0.0, 0.0])
OUT = REPO / "runs" / DOWN / ("chain_random" if RANDOM else "chain")
WORLD_KEYS = jax.random.split(jax.random.PRNGKey(7), N)          # one random world per robot
_k1, _k2 = jax.random.split(jax.random.PRNGKey(8))
DELAY_SUB = jax.random.randint(_k1, (N,), 0, 11) if RANDOM else None  # 0-40 ms in 4 ms
OFFSET = (jax.random.uniform(_k2, (N, 12), minval=-1, maxval=1) * jp.deg2rad(2.0)) if RANDOM else None


def make_env(kind):
  run = {"walk": WALK, "rear": REAR, "down": DOWN}[kind]
  c = config_dict.ConfigDict(json.load(open(REPO / "runs" / run / "config.json"))["env"])
  c.impl = "jax"
  if not RANDOM:
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
  if not RANDOM:
    return env, policy, jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
  randomize = rpw.make_domain_randomizer(env)
  venv = BraxDomainRandomizationVmapWrapper(env, lambda m: randomize(m, WORLD_KEYS))
  env._venv = venv
  return env, policy, jax.jit(venv.reset), jax.jit(venv.step)


def contact_of(env, data):
  return jp.array([data.sensordata[env.mj_model.sensor_adr[s]] > 0
                   for s in env._feet_floor_found_sensor])


def handoff_fn(env):
  """fresh (a reset state of `env`) + src physics -> the same robot, now run by `env`."""
  def one(fresh, src, i):
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
    if RANDOM:  # this robot's own servo delay and calibration error, the same in every policy
      info["delay_substeps"] = DELAY_SUB[i]
      if "joint_offset" in info:
        info["joint_offset"] = OFFSET[i]
    if "command" in info and "steps_until_next_cmd" in info:   # walking: fixed command
      info["command"] = CMD
      info["steps_until_next_cmd"] = jp.array(10**6, dtype=jp.int32)
    return fresh.replace(data=d, obs=env._get_obs(d, info), info=info,
                         done=jp.zeros_like(fresh.done))
  f = jax.jit(jax.vmap(one))
  return lambda fresh, src: f(fresh, src, jp.arange(N))


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


leg_log = []                       # (qpos, qvel, force, phase) for all robots, if --legs
PHASES = ["walk", "rear", "hold", "down", "walk off"]


def leg_arrays(env, d):
  return (np.array(d.qpos)[:, env._leg_qpos], np.array(d.qvel)[:, env._leg_qvel],
          np.array(d.actuator_force)[:, env._leg_act])


def record_legs(parts, phase):
  if LEGS:
    leg_log.append((*parts, np.broadcast_to(phase, (N,)).copy()))


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
  record_legs(leg_arrays(wenv, ws.data), 0)
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
  record_legs(leg_arrays(renv, rs.data), np.where(arr, 2, 1))
in_tripod_at_cmd = np.array(jax.vmap(renv._in_tripod)(rs.data, jax.vmap(renv._feet_contact)(rs.data)))
tilt_at_cmd = np.array(tilt(rs.data))

# 3 + 4. Come down; each robot switches to walking the moment it is ready to walk
ds = to_down(dreset(keys(jax.random.PRNGKey(3))), rs.data)
wfresh = wreset(keys(jax.random.PRNGKey(4)))
ws = to_walk(wfresh, ds.data)
switched = np.zeros(N, bool)
land = np.full(N, np.nan)          # strict training "landed" (reported, not used to switch)
ready_t = np.full(N, np.nan)       # when the switcher handed over (ready to walk)
ready_streak = np.zeros(N, int)
ready_fn = jax.jit(jax.vmap(lambda d: jp.all(denv._feet_contact(d)) & ~denv._found(d, "tail_tip")
                            & (jp.linalg.norm(d.qvel[0:3]) < STILL_LIN)
                            & (jp.linalg.norm(d.qvel[3:6]) < STILL_ANG)))
switch_xy = np.zeros((N, 2))
steps_walked = np.zeros(N, int)
end_xy = np.full((N, 2), np.nan)
down_fell, walk2_fell = np.zeros(N, bool), np.zeros(N, bool)
hip_after_switch = np.zeros(N)
why_not = {}
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
  if LEGS:
    sw = switched[:, None]
    record_legs(tuple(np.where(sw, w_, d_) for w_, d_ in
                      zip(leg_arrays(wenv, ws.data), leg_arrays(denv, ds.data))), np.where(switched, 4, 3))
  hip_now = np.abs(np.array(ws.data.actuator_force)[:, hips]).max(1)
  early = switched & (steps_walked <= int(0.5 / DT))         # first 0.5 s of walking
  hip_after_switch = np.where(early, np.maximum(hip_after_switch, hip_now), hip_after_switch)
  landed_now = np.array(ds.info["arrived"]) & ~switched & np.isnan(land)
  land[landed_now] = t * DT
  ready_streak = np.where(np.array(ready_fn(ds.data)), ready_streak + 1, 0)
  ready_now = ~switched & (ready_streak >= int(round(READY_HOLD_S / DT)))
  timeout = ~switched & ~ready_now & (t * DT >= DOWN_MAX_S)
  go = ready_now | timeout
  if timeout.any():  # why not ready? each part of the "landed" test at the moment of hand-over
    c_ = np.array(jax.vmap(denv._feet_contact)(ds.data))
    tail_ = np.array(jax.vmap(lambda d: denv._found(d, "tail_tip"))(ds.data))
    tilt_ = np.array(tilt(ds.data))
    hip_ = 100 * np.array(ds.data.xpos)[:, denv._hind_thigh, 2].mean(1)
    for i in np.where(timeout)[0]:
      why_not[i] = (f"feet FL/FR/BL/BR {''.join('1' if x else '0' for x in c_[i])}, tail tip "
                    f"{'DOWN' if tail_[i] else 'up'}, tilt {tilt_[i]:.1f} (box -2+/-5), "
                    f"hips {hip_[i]:.1f} cm (box 16+/-1.5)")
  if go.any():
    ready_t[ready_now] = t * DT
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
ready = ~np.isnan(ready_t)

# --- Report
pct = lambda m: f"{int(m.sum())}/{N}"  # noqa: E731
print(f"chain: {WALK} -> {REAR} -> {DOWN} -> {WALK}, {N} robots, "
      + ("RANDOMIZED robots + training shoves\n" if RANDOM else "nominal robot, delay 20 ms\n"))
print(f"1. walk {WALK_S:.0f} s at 0.12 m/s:   fell {pct(walk1_fell)}, distance median "
      f"{100 * np.median(walk1_dist):.0f} cm (ideal {100 * 0.12 * WALK_S:.0f}), speed at 'rear' "
      f"{100 * np.median(speed_at_cmd):.0f} cm/s")
ok = ~np.isnan(arrive)
print(f"2. rear (from walking):      arrived {pct(ok)}, fell {pct(rear_fell)}, arrival median "
      f"{np.nanmedian(arrive):.2f} s (slowest {np.nanmax(arrive):.2f}); at 'down': in tripod "
      f"{pct(in_tripod_at_cmd)}, tilt median {np.median(tilt_at_cmd):.0f} deg, held "
      f"{REAR_S - np.nanmax(arrive):.1f}-{REAR_S - np.nanmin(arrive):.1f} s")
print(f"3. come down:                ready to walk {pct(ready)}, fell {pct(down_fell)}, handed over "
      f"at median {np.nanmedian(ready_t):.2f} s (slowest {np.nanmax(ready_t):.2f}); not ready by "
      f"{DOWN_MAX_S:.0f} s -> handed over anyway: {pct(switched & ~ready)}")
print(f"   (strict training 'landed' box met before the hand-over: {pct(landed)})")
print(f"4. walk off at landing:      fell {pct(walk2_fell)}, distance in {WALK2_S:.0f} s median "
      f"{100 * np.nanmedian(walk2_dist):.0f} cm (ideal {100 * 0.12 * WALK2_S:.0f}), slowest "
      f"{100 * np.nanmin(walk2_dist):.0f} cm; hind-hip peak in the first 0.5 s median "
      f"{np.median(hip_after_switch):.2f} N*m (max {hip_after_switch.max():.2f})")
all_ok = (~walk1_fell & ok & ~rear_fell & ready & ~down_fell & ~walk2_fell)
print(f"\nwhole chain without a fall or a missed step: {pct(all_ok)}")
if RANDOM:
  mv = wenv._venv._mjx_model_v
  mass = np.array(mv.body_mass).sum(1)
  mu = np.array(mv.geom_friction)[:, wenv.mj_model.geom("floor").id, 0]
  kp = np.array(mv.actuator_gainprm)[:, 0, 0] / wenv.mj_model.actuator_gainprm[0, 0]
  stage = np.where(walk1_fell, "walk", np.where(rear_fell | np.isnan(arrive), "rear",
                   np.where(down_fell | ~ready, "down", np.where(walk2_fell, "walk off", ""))))
  print(f"\nrobots that failed (mass kg, friction, servo stiffness, delay ms):")
  for i in np.where(~all_ok)[0]:
    print(f"  robot {i:2d}: failed at {stage[i]:8s}  {mass[i]:.2f} kg, mu {mu[i]:.2f}, "
          f"kp x{kp[i]:.2f}, delay {4 * int(DELAY_SUB[i])} ms; in tripod at 'down': "
          f"{'yes' if in_tripod_at_cmd[i] else 'NO'} (tilt {tilt_at_cmd[i]:.0f})")
    if i in why_not:
      print(f"            not ready at {DOWN_MAX_S:.0f} s: {why_not[i]}")
  for i in np.where(~in_tripod_at_cmd)[0]:
    print(f"  robot {i:2d}: not in the tripod at 'down' (tilt {tilt_at_cmd[i]:.0f} deg), "
          f"{'came down fine' if ready[i] else 'not ready to walk'}")
  print(f"range tested: mass {mass.min():.2f}-{mass.max():.2f} kg, friction {mu.min():.2f}-{mu.max():.2f}, "
        f"stiffness x{kp.min():.2f}-{kp.max():.2f}, delay 0-40 ms")
OUT.mkdir(parents=True, exist_ok=True)
json.dump(dict(robots=N, walk1_fell=int(walk1_fell.sum()), rear_arrived=int(ok.sum()),
               rear_fell=int(rear_fell.sum()), arrive_median=float(np.nanmedian(arrive)),
               ready=int(ready.sum()), landed_strict=int(landed.sum()), down_fell=int(down_fell.sum()),
               ready_median=float(np.nanmedian(ready_t)), walk2_fell=int(walk2_fell.sum()),
               walk2_dist_cm=float(100 * np.nanmedian(walk2_dist)), all_ok=int(all_ok.sum())),
          open(OUT / "chain.json", "w"), indent=1)

if LEGS:
  # Leg layout per leg: [hip_side, hip, knee]; hip and knee both turn about the body's y axis,
  # so the shin's direction relative to the torso = hip + knee. MuJoCo +hip swings the foot
  # backward; reported here with FORWARD = + (as in the CAD sketch).
  Q = np.stack([x[0] for x in leg_log])            # (steps, robots, 12)
  V = np.stack([x[1] for x in leg_log])
  F = np.stack([x[2] for x in leg_log])
  PH = np.stack([x[3] for x in leg_log])           # (steps, robots)
  STALL, RATED, VMAX = 2.94, 0.98, 4.7
  groups = {"front": [0, 1], "hind": [2, 3]}
  phase_sets = {"walk": [0, 4], "rear": [1], "hold": [2], "down": [3], "ALL": [0, 1, 2, 3, 4]}
  out = {}
  rng = lambda x: f"{np.percentile(x, 1):6.0f} to {np.percentile(x, 99):4.0f}  (min {x.min():4.0f}, max {x.max():4.0f})"  # noqa: E731
  print("\n--- legs: what a rod-linkage leg needs (forward = +, degrees; 1st-99th pct, then min/max)")
  for g, legs in groups.items():
    hip_side = np.concatenate([Q[:, :, 3 * l] for l in legs], 1)
    hip = np.concatenate([Q[:, :, 3 * l + 1] for l in legs], 1)
    knee = np.concatenate([Q[:, :, 3 * l + 2] for l in legs], 1)
    hv = np.concatenate([V[:, :, 3 * l + 1] for l in legs], 1)
    kv = np.concatenate([V[:, :, 3 * l + 2] for l in legs], 1)
    hf = np.concatenate([F[:, :, 3 * l + 1] for l in legs], 1)
    kf = np.concatenate([F[:, :, 3 * l + 2] for l in legs], 1)
    ph = np.concatenate([PH] * len(legs), 1)
    thigh = -np.degrees(hip)                       # thigh direction, forward +
    bend = np.degrees(np.abs(knee))                # knee bend (gap thigh-shin)
    shin = -np.degrees(hip + knee)                 # shin direction = knee servo angle (linkage)
    ksv = np.abs(hv + kv)                          # knee servo speed with the linkage
    hsf = np.abs(hf - kf)                          # hip servo torque with the linkage
    out[g] = {}
    print(f"\n{g.upper()} legs")
    for pn, ids in phase_sets.items():
      m = np.isin(ph, ids)
      if not m.any():
        continue
      r = dict(thigh=thigh[m], bend=bend[m], shin=shin[m], hip_side=np.degrees(hip_side[m]))
      out[g][pn] = {k: [float(np.percentile(v, 1)), float(np.percentile(v, 99)), float(v.min()), float(v.max())]
                    for k, v in r.items()}
      print(f"  {pn:5s} thigh {rng(r['thigh'])} | bend {rng(r['bend'])} | shin (knee servo) {rng(r['shin'])}")
    m = np.ones_like(ph, bool)
    tor = dict(knee_joint=np.abs(kf), hip_joint_sim=np.abs(hf), hip_servo_linkage=hsf)
    spd = dict(knee_joint_sim=np.abs(kv), knee_servo_linkage=ksv, hip=np.abs(hv))
    print(f"  hip_side (all phases, deg): {rng(np.degrees(hip_side))}")
    print("  torque N*m        peak   p99   time > rated 0.98 / > 65% stall 1.91")
    for k, v in tor.items():
      print(f"    {k:18s} {v.max():5.2f} {np.percentile(v, 99):5.2f}   {100 * (v > RATED).mean():5.1f}% / {100 * (v > 0.65 * STALL).mean():4.1f}%")
    print(f"  speed rad/s       peak   p99   time > {VMAX} (no-load max)")
    for k, v in spd.items():
      print(f"    {k:18s} {v.max():5.2f} {np.percentile(v, 99):5.2f}   {100 * (v > VMAX).mean():5.1f}%")
    out[g]["torque"] = {k: dict(peak=float(v.max()), p99=float(np.percentile(v, 99)),
                                over_rated_pct=float(100 * (v > RATED).mean())) for k, v in tor.items()}
    out[g]["speed"] = {k: dict(peak=float(v.max()), p99=float(np.percentile(v, 99)),
                               over_max_pct=float(100 * (v > VMAX).mean())) for k, v in spd.items()}
  print("\n(sampled every 20 ms: short torque spikes between samples are missed; down_load.py checks those)")
  json.dump(out, open(OUT / "legs.json", "w"), indent=1)
  print(f"-> {OUT / 'legs.json'}")
  # Raw knee samples for cad/knee_linkage.py: knee bend (deg) and knee torque in the
  # bend-MORE direction (+ = servo folds the knee, - = servo straightens it / holds weight).
  # Front knees bend negative in the sim, hind positive.
  raw = {}
  for g, legs in groups.items():
    sign = [-1.0 if g == "front" else 1.0]
    raw[f"{g}_bend"] = np.concatenate([np.degrees(np.abs(Q[:, :, 3 * l + 2])) for l in legs], 1)
    raw[f"{g}_tau"] = np.concatenate([sign[0] * F[:, :, 3 * l + 2] for l in legs], 1)
    raw[f"{g}_phase"] = np.concatenate([PH] * len(legs), 1)
  np.savez_compressed(OUT / "legs_raw.npz", **raw)
  print(f"-> {OUT / 'legs_raw.npz'}")

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
