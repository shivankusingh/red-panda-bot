"""Is the rearing policy crooked? Left/right symmetry while it holds the tail tripod.

Runs N robots from walking starts (nominal robot) for 6 s and measures every robot that
is in the tripod, over the last 3 s:
  - body roll (+ = right side down), from gravity like the real IMU
  - vertical load on each hind foot and on the tail (= weight - hind feet; front feet up)
  - hind leg joint angles left vs right (hip_side + = outward on both legs)
  - average |torque| per hind joint and tail joint, left vs right
  - tail sideways angles (wag, wag2) and the tail tip's sideways offset
  - the centre of mass's sideways offset from the midpoint of the hind feet
Sideways = across the robot's heading; + = to the robot's left.

  python tools/eval/rear_symmetry.py rr_run7a [robots]
"""
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_rear as rr  # noqa: E402

RUN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
run = REPO / "runs" / RUN
cfg = json.load(open(run / "config.json"))
c = config_dict.ConfigDict(cfg["env"])
c.impl = "jax"
c.naconmax = 16 * N
c.rear_config.ladder_fraction = 0.0
env = rr.RedPandaRear(config=c)
m = env.mj_model
net = cfg["ppo"]["network_factory"]
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
    value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
policy = jax.jit(jax.vmap(ppo_networks.make_inference_fn(nets)(
    pickle.load(open(run / "params.pkl", "rb")), deterministic=True)))
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

JOINTS = ["bl_hip_side", "br_hip_side", "bl_hip", "br_hip", "bl_knee", "br_knee",
          "tail_wag", "tail_lift", "tail_wag2"]
qadr = {j: m.jnt_qposadr[m.joint(j).id] for j in JOINTS}
act = {j: m.actuator(j).id for j in JOINTS}
force_adr = {f: m.sensor_adr[m.sensor(f"{f}_floor_force").id] + 3 * np.arange(4) for f in ["bl_foot", "br_foot"]}
TORSO = m.body("torso").id
TAIL2 = m.body("tail_2").id
TIP = np.array([-0.165, 0.0, 0.0])  # tail tip in tail_2's frame
FEET = [m.site("bl_foot").id, m.site("br_foot").id]
weight = float(m.body_subtreemass[0]) * 9.81

key = jax.random.PRNGKey(0)
state = reset(jax.random.split(key, N))
T = c.episode_length
rec = {k: [] for k in ["roll", "fl", "fr", "tail", "lat_tip", "lat_com", "tilt"] + [f"q_{j}" for j in JOINTS] + [f"t_{j}" for j in JOINTS]}
ok_mask = []
for t in range(T):
  key, k = jax.random.split(key)
  a, _ = policy(state.obs, jax.random.split(k, N))
  state = step(state, a)
  if t < T // 2:
    continue
  d = state.data
  pitch, roll = jax.vmap(env._torso_tilt)(d)
  feet = np.array(jax.vmap(env._feet_contact)(d))
  tail_on = np.array(jax.vmap(lambda x: env._found(x, "tail_tip"))(d))
  ok_mask.append(feet[:, 2] & feet[:, 3] & tail_on & ~feet[:, 0] & ~feet[:, 1] & (np.array(state.done) == 0))
  sd = np.array(d.sensordata)
  fl, fr = sd[:, force_adr["bl_foot"]].sum(1), sd[:, force_adr["br_foot"]].sum(1)
  xpos, xmat = np.array(d.xpos), np.array(d.xmat).reshape(N, -1, 3, 3)
  fwd = xmat[:, TORSO, :, 0] * np.array([1, 1, 0])
  fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
  left = np.stack([-fwd[:, 1], fwd[:, 0], np.zeros(N)], 1)
  sites = np.array(d.site_xpos)
  mid = sites[:, FEET].mean(1)
  tip = xpos[:, TAIL2] + np.einsum("nij,j->ni", xmat[:, TAIL2], TIP)
  com = np.array(d.subtree_com)[:, 0]
  rec["roll"].append(np.degrees(np.array(roll)))
  rec["tilt"].append(-np.degrees(np.array(pitch)))
  rec["fl"].append(fl)
  rec["fr"].append(fr)
  rec["tail"].append(weight - fl - fr)
  rec["lat_tip"].append(np.einsum("ni,ni->n", tip - mid, left) * 100)
  rec["lat_com"].append(np.einsum("ni,ni->n", com - mid, left) * 100)
  q = np.array(d.qpos)
  f = np.abs(np.array(d.actuator_force))
  for j in JOINTS:
    rec[f"q_{j}"].append(np.degrees(q[:, qadr[j]]))
    rec[f"t_{j}"].append(f[:, act[j]])
ok = np.array(ok_mask)  # [T/2, N]
R = {k: np.array(v)[ok] for k, v in rec.items()}  # every (time, robot) sample in the tripod
per_robot_roll = np.array([np.array(rec["roll"])[ok[:, i], i].mean() for i in range(N) if ok[:, i].any()])


def lr(name, unit, l, r, fmt="{:6.2f}"):
  print(f"  {name:24s} left {fmt.format(l)} {unit:5s} right {fmt.format(r)} {unit:5s}  (right - left {fmt.format(r - l)})")


print(f"{RUN}: {N} robots, {ok.any(0).sum()} in the tripod during the last 3 s ({ok.mean() * 100:.0f}% of samples)\n")
print("BODY")
print(f"  roll (+ = right side down)   mean {R['roll'].mean():+.1f} deg, spread across robots {per_robot_roll.std():.1f} deg "
      f"(robots: {np.percentile(per_robot_roll, 10):+.1f} .. {np.percentile(per_robot_roll, 90):+.1f})")
print(f"  tilt                         mean {R['tilt'].mean():.1f} deg")
print(f"  centre of mass sideways      {R['lat_com'].mean():+.2f} cm from the hind feet midpoint (+ = left)")
print("LOAD (vertical, N; body weight {:.1f} N)".format(weight))
lr("hind foot", "N", R["fl"].mean(), R["fr"].mean(), "{:6.1f}")
print(f"  tail (weight - hind feet)    {R['tail'].mean():.1f} N ({R['tail'].mean() / weight * 100:.0f}% of the weight)")
print("HIND LEG ANGLES (deg)")
for jl, jr, nm in [("bl_hip_side", "br_hip_side", "hip_side (+ outward)"), ("bl_hip", "br_hip", "hip"), ("bl_knee", "br_knee", "knee")]:
  lr(nm, "deg", R[f"q_{jl}"].mean(), R[f"q_{jr}"].mean(), "{:6.1f}")
print("AVERAGE |TORQUE| (N*m; rated 0.98)")
for jl, jr, nm in [("bl_hip_side", "br_hip_side", "hip_side"), ("bl_hip", "br_hip", "hip"), ("bl_knee", "br_knee", "knee")]:
  lr(nm, "N*m", R[f"t_{jl}"].mean(), R[f"t_{jr}"].mean())
print("TAIL")
print(f"  wag {R['q_tail_wag'].mean():+.1f} deg, wag2 {R['q_tail_wag2'].mean():+.1f} deg, lift {R['q_tail_lift'].mean():.1f} deg "
      f"(sideways joints: 0 = straight)")
print(f"  tip sideways                 {R['lat_tip'].mean():+.1f} cm from the hind feet midpoint (+ = left)")
print(f"  |torque| wag {R['t_tail_wag'].mean():.2f}, lift {R['t_tail_lift'].mean():.2f}, wag2 {R['t_tail_wag2'].mean():.2f} N*m")
json.dump({k: float(v.mean()) for k, v in R.items()} | {"roll_robot_std": float(per_robot_roll.std())},
          open(run / "rear_symmetry.json", "w"), indent=1)
