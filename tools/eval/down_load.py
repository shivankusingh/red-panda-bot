"""Hardware strain while coming down from the tripod, measured every physics substep (4 ms).

The other checks sample once per policy step (20 ms) and can miss short impact spikes. This
one records every substep of the servo model: torque per servo, joint speed, and each foot's
floor force. Same numbers for walking (rp_run26) as the baseline we already accept.

  python tools/eval/down_load.py [down run] [walking run]      (default dn_run4b rp_run26)

STS3215 C018 at 12 V (reseller specs): stall 2.94 N*m @ 2.7 A, rated 0.98 N*m @ 0.9 A,
no-load speed 4.7 rad/s. Current I = 0.18 A + |torque| / kt (hardware_check.py's model).
"Back-driven": the servo is at >= 90% of the torque it can give and the joint still turns the
other way (the load wins and turns the gears backwards: the gear-tooth case).
"""
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_down as rd  # noqa: E402
import red_panda_walk as rpw  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DOWN = ARGS[0] if ARGS else "dn_run4b"
WALK = ARGS[1] if len(ARGS) > 1 else "rp_run26"
STALL, I_STALL, I0, RATED, I_RATED = 2.94, 2.7, 0.18, 0.98, 0.9
KT = STALL / (I_STALL - I0)
N = 32
STEPS = 100            # 2 s: dn_run4b lands by 1.1 s
MASS_KG = 1.8


def instrument(env):
  """Replace env._servo_substeps with a copy that also keeps every substep's torque, joint
  speed and foot force in info (the physics is unchanged: same delay, same limits)."""
  n = env.n_substeps
  servo = env._servo

  def substeps(data, motor_targets, info):
    model = env.mjx_model
    kp = model.actuator_gainprm[:, 0]
    prev, prev2 = info["last_motor_targets"]

    def one(data, i):
      back = i - info["delay_substeps"]
      target = jp.where(back >= 0, motor_targets, jp.where(back >= -n, prev, prev2))
      if servo.torque_speed_limit:
        q, w = data.qpos[env._act_qpos], data.qvel[env._act_qvel]
        tau = kp * (target - q)
        tau_max = env._stall * jp.clip(1.0 - w / servo.no_load_speed, 0.0, 1.0)
        tau_min = -env._stall * jp.clip(1.0 + w / servo.no_load_speed, 0.0, 1.0)
        target = q + jp.clip(tau, tau_min, tau_max) / kp
      data = mjx.step(model, data.replace(ctrl=target))
      out = (data.actuator_force, data.qvel[env._act_qvel],
             jp.sum(data.sensordata[env._foot_force_adr], axis=1))
      return data, out

    data, (tau, w, feet) = jax.lax.scan(one, data, jp.arange(n))
    info["x_tau"], info["x_w"], info["x_feet"] = tau, w, feet  # (n, 18), (n, 18), (n, 4)
    return data

  env._servo_substeps = substeps
  return env


def load_policy(run_dir, env):
  cfg = json.load(open(run_dir / "config.json"))
  net = cfg["ppo"]["network_factory"]
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
      value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
      policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
  return jax.jit(ppo_networks.make_inference_fn(nets)(
      pickle.load(open(run_dir / "params.pkl", "rb")), deterministic=True))


def with_mass(env, scale):
  def rand(model):
    mass = jp.tile(model.body_mass * scale, (N, 1))
    inertia = jp.tile(model.body_inertia * scale, (N, 1, 1))
    in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace(
        {"body_mass": 0, "body_inertia": 0})
    return model.tree_replace({"body_mass": mass, "body_inertia": inertia}), in_axes
  return BraxDomainRandomizationVmapWrapper(env, rand)


def rollout(kind, scale):
  """kind 'down' (tripod starts) or 'walk' (0.12 m/s forward). Returns per-substep logs."""
  run_dir = REPO / "runs" / (DOWN if kind == "down" else WALK)
  c = config_dict.ConfigDict(json.load(open(run_dir / "config.json"))["env"])
  c.impl = "jax"
  c.pert_config.enable = False
  if kind == "down":
    c.naconmax = 24 * N
    c.rear_config.ladder_fraction = 0.0   # real tripod starts only
    env = instrument(rd.RedPandaDown(config=c))
  else:
    rpw.use_flat(c)
    env = instrument(rpw.RedPandaWalk(config=c))
  policy = load_policy(run_dir, env)
  venv = with_mass(env, scale)
  reset, step = jax.jit(venv.reset), jax.jit(venv.step)
  s = reset(jax.random.split(jax.random.PRNGKey(0), N))
  key = jax.random.PRNGKey(1)
  tau, w, feet, tilt, front_down = [], [], [], [], []
  fell = np.zeros(N, bool)
  tilt_fn = jax.jit(jax.vmap(lambda d: -jp.rad2deg(env._torso_tilt(d)[0])))
  contact_fn = jax.jit(jax.vmap(env._feet_contact))
  for _ in range(STEPS):
    if kind == "walk":
      s.info["command"] = jp.tile(jp.array([0.12, 0.0, 0.0]), (N, 1))
      s.info["steps_until_next_cmd"] = jp.full((N,), 1000, dtype=jp.int32)
    key, k = jax.random.split(key)
    a, _ = policy(s.obs, jax.random.split(k, N))
    s = step(s, a)
    fell |= np.array(s.done) > 0
    tau.append(np.array(s.info["x_tau"]))
    w.append(np.array(s.info["x_w"]))
    feet.append(np.array(s.info["x_feet"]))
    if kind == "down":
      tilt.append(np.array(tilt_fn(s.data)))
      front_down.append(np.array(contact_fn(s.data))[:, :2].any(1))
  lt = np.array(env._lt_act if kind == "down" else env._leg_act)
  names = [env.mj_model.actuator(int(i)).name for i in lt]
  stall = np.array(env._stall)[lt]
  # (robots, time in substeps, servos)
  T = np.concatenate(tau, axis=1)[:, :, lt]
  W = np.concatenate(w, axis=1)[:, :, lt]
  F = np.concatenate(feet, axis=1)
  return dict(names=names, tau=T, w=W, feet=F, stall=stall, fell=fell,
              dt=env.dt / env.n_substeps, weight=MASS_KG * scale * 9.81,
              tilt=np.array(tilt).T if tilt else None,                # (robots, policy steps)
              front_down=np.array(front_down).T if front_down else None)


def timeline(r, bin_s=0.1, until_s=1.5):
  """When does the strain happen? Per time bin: tilt, share of robots with a front paw down,
  and ms per robot that the hind hips / tail lift spend over 65% of stall or back-driven."""
  sub = int(round(0.02 / r["dt"]))            # substeps per policy step
  per_bin = int(round(bin_s / 0.02))          # policy steps per bin
  A, T, W = np.abs(r["tau"]), r["tau"], r["w"]
  back = (A >= 0.9 * r["stall"]) & (T * W < 0) & (np.abs(W) > 0.2)
  over = A > 0.65 * STALL
  idx = {nm: r["names"].index(nm) for nm in ("bl_hip", "br_hip", "tail_lift")}
  dt_ms = r["dt"] * 1000
  print(f"\ntimeline (median over robots; ms = average per robot in that {bin_s:.1f} s window)")
  print(f"{'time s':>9s} {'tilt':>6s} {'front paw':>9s} | {'hind hips >65%':>14s} {'back-driven':>11s} | {'tail >65%':>9s}")
  for b in range(int(until_s / bin_s)):
    s0, s1 = b * per_bin, (b + 1) * per_bin
    u0, u1 = s0 * sub, s1 * sub
    hips = [idx["bl_hip"], idx["br_hip"]]
    print(f"{s0 * 0.02:4.1f}-{s1 * 0.02:3.1f} {np.median(r['tilt'][:, s1 - 1]):6.1f} "
          f"{100 * r['front_down'][:, s0:s1].any(1).mean():8.0f}% | "
          f"{over[:, u0:u1][:, :, hips].sum((1, 2)).mean() * dt_ms / 2:11.1f} ms "
          f"{back[:, u0:u1][:, :, hips].sum((1, 2)).mean() * dt_ms / 2:8.1f} ms | "
          f"{over[:, u0:u1, idx['tail_lift']].sum(1).mean() * dt_ms:6.1f} ms")


def servo_table(r):
  dt_ms = r["dt"] * 1000
  T, W = r["tau"], r["w"]
  A = np.abs(T)
  current = I0 + A / KT
  limit = np.where(np.sign(T) == np.sign(W), r["stall"] * np.clip(1 - np.abs(W) / 4.7, 0, 1), r["stall"])
  backdriven = (A >= 0.9 * limit) & (T * W < 0) & (np.abs(W) > 0.2)
  rows = {}
  for j, nm in enumerate(r["names"]):
    rows[nm] = dict(
        peak=A[:, :, j].max(),
        p99=np.percentile(A[:, :, j], 99),
        over_rated_ms=np.median((A[:, :, j] > RATED).sum(1)) * dt_ms,
        over_rated_ms_max=(A[:, :, j] > RATED).sum(1).max() * dt_ms,
        over_rule_ms=np.median((A[:, :, j] > 0.65 * STALL).sum(1)) * dt_ms,
        over_rule_ms_max=(A[:, :, j] > 0.65 * STALL).sum(1).max() * dt_ms,
        rms_amps=np.sqrt(np.mean(current[:, :, j] ** 2)),
        back_ms_max=backdriven[:, :, j].sum(1).max() * dt_ms,
        back_speed=np.abs(np.where(backdriven[:, :, j], W[:, :, j], 0)).max(),
        speed=np.percentile(np.abs(W[:, :, j]), 99.9),
    )
  return rows


def report(label, r, show):
  rows = servo_table(r)
  print(f"\n=== {label}: {N} robots, {STEPS * r['dt'] * 5:.0f} s, fell {r['fell'].sum()}/{N} ===")
  print(f"{'servo':10s} {'peak':>6s} {'p99':>6s} | {'>rated ms':>13s} | {'>65% stall ms':>13s} | "
        f"{'RMS A':>6s} | {'back-driven':>17s} | {'speed':>5s}")
  print(f"{'':10s} {'N*m':>6s} {'N*m':>6s} | {'median / max':>13s} | {'median / max':>13s} | "
        f"{'(0.9)':>6s} | {'max ms / rad/s':>17s} | {'rad/s':>5s}")
  for nm in show:
    x = rows[nm]
    print(f"{nm:10s} {x['peak']:6.2f} {x['p99']:6.2f} | {x['over_rated_ms']:5.0f} / {x['over_rated_ms_max']:5.0f} | "
          f"{x['over_rule_ms']:5.0f} / {x['over_rule_ms_max']:5.0f} | {x['rms_amps']:6.2f} | "
          f"{x['back_ms_max']:7.0f} / {x['back_speed']:5.2f} | {x['speed']:5.2f}")
  F = r["feet"] / r["weight"]
  print("foot floor force, x body weight (standing still = 0.25 each):  peak / 99.9th pct")
  print("   " + "   ".join(f"{f}: {F[:, :, i].max():.2f} / {np.percentile(F[:, :, i], 99.9):.2f}"
                         for i, f in enumerate(["FL", "FR", "BL", "BR"])))
  return rows


SHOW = ["fl_hip", "fl_knee", "fr_hip", "fr_knee", "bl_hip", "bl_knee", "br_hip", "br_knee",
        "fl_hip_side", "bl_hip_side"]
print(f"{DOWN} (coming down) vs {WALK} (walking 0.12 m/s), every 4 ms substep")
print(f"STS3215: rated {RATED} N*m / {I_RATED} A, our peak rule 65% stall = {0.65 * STALL:.2f}, "
      f"stall {STALL} N*m, no-load 4.7 rad/s")
if "--timeline" in sys.argv:  # just the come-down at 1.8 kg, with the when-does-it-happen table
  r = rollout("down", 1.0)
  report(f"coming down {DOWN} at {MASS_KG:.2f} kg", r, ["bl_hip", "br_hip", "tail_lift"])
  timeline(r)
  sys.exit()
results = {}
for kind, label in [("walk", f"walking {WALK}"), ("down", f"coming down {DOWN}")]:
  for scale in (1.0, 1.1):
    r = rollout(kind, scale)
    show = SHOW + (["tail_lift", "tail_wag"] if kind == "down" else [])
    results[f"{kind}_{scale}"] = report(f"{label} at {MASS_KG * scale:.2f} kg", r, show)
out = REPO / "runs" / DOWN / "down_load.json"
json.dump({k: {nm: {m: float(v) for m, v in x.items()} for nm, x in rows.items()}
           for k, rows in results.items()}, open(out, "w"), indent=1)
print(f"\n-> {out}")
