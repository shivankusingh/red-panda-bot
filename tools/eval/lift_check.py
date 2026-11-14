import json, pickle, sys
import sys as _s; VX = float(_s.argv[2]) if len(_s.argv) > 2 else 0.2
from pathlib import Path
import jax, jax.numpy as jp, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict
REPO = Path(__file__).resolve().parents[2]  # repo root; sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as rpw
run = REPO / "runs" / sys.argv[1]; cfg = json.load(open(run / "config.json"))
c = config_dict.ConfigDict(cfg["env"]); c.impl = "jax"; c.pert_config.enable = False
LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else -1  # terrain row to walk on (-1 = the run's own floor)
if LEVEL >= 0: rpw.use_terrain(c, LEVEL)
if "servo_config" in c: c.servo_config.delay_ms = [0.0, 0.0]
env = rpw.RedPandaWalk(config=c)
net = cfg["ppo"]["network_factory"]
nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]), value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
policy = jax.jit(ppo_networks.make_inference_fn(nets)(pickle.load(open(run / "params.pkl", "rb")), deterministic=True))
reset, step = jax.jit(env.reset), jax.jit(env.step)
state = reset(jax.random.PRNGKey(0)); key = jax.random.PRNGKey(1)
target = c.reward_config.max_foot_height
rows = []
for i in range(300):
    state.info["command"] = jp.array([VX, 0., 0.]); state.info["steps_until_next_cmd"] = jp.array(1000, dtype=jp.int32)
    key, k = jax.random.split(key); a, _ = policy(state.obs, k); state = step(state, a)
    d = state.data
    h = np.array(env._feet_height(d))  # above the ground under each foot (= world height on flat)
    v = np.linalg.norm(np.array(d.sensordata[env._foot_linvel_sensor_adr])[:, :2], axis=1)
    con = np.array(state.info["last_contact"])
    desired = np.array(env._desired_contact(state.info["gait_phase"], state.info["command"]))
    rows.append((h, v, con, desired))
rows = rows[100:]
H = np.array([r[0] for r in rows]); V = np.array([r[1] for r in rows]); C = np.array([r[2] for r in rows]); D = np.array([r[3] for r in rows])
names = ["LF (fl)", "RF (fr)", "LH (bl)", "RH (br)"]
print(f"foot     | lift when scheduled UP | 'not touching' when scheduled UP | clearance cost/step x weight 2")
for i, n in enumerate(names):
    up = ~D[:, i]
    lift = (H[up, i].max() - H[:, i].min()) * 100
    not_touch = (~C[up, i]).mean()
    cost = (np.abs(H[:, i] - target) * np.sqrt(V[:, i])).mean() * 2
    print(f"{n:8s} |        {lift:4.1f} cm         |              {not_touch*100:3.0f}%               |   {cost:.4f}")
print(f"\nfoot height when 'not touching' but barely up: lowest swing heights (cm above resting) per foot: "
      f"{np.round([(np.percentile(H[~C[:, i], i], 10) - H[:, i].min())*100 for i in range(4)], 2).tolist()}")

# What would a "swing height follows the clock" term charge? target = rest + 3.6 cm * sin(pi * swing progress)
duty = c.gait_config.duty; offs = np.array(c.gait_config.offsets); lift = 0.036
P = []
state = reset(jax.random.PRNGKey(0)); key = jax.random.PRNGKey(1)
for i in range(300):
    state.info["command"] = jp.array([VX, 0., 0.]); state.info["steps_until_next_cmd"] = jp.array(1000, dtype=jp.int32)
    key, k = jax.random.split(key); a, _ = policy(state.obs, k); state = step(state, a)
    P.append((float(state.info["gait_phase"]), np.array(state.data.site_xpos[env._feet_site_id])[:, 2]))
P = P[100:]
rest = 0.0097
err = np.zeros(4)
for ph, h in P:
    prog = np.mod(ph - offs, 1.0)
    swing = prog >= duty
    s = np.clip((prog - duty) / (1 - duty), 0, 1)
    tgt = rest + lift * np.sin(np.pi * s)
    err += np.where(swing, (h - tgt) ** 2, 0.0)
err /= len(P)
print("\nswing-height error^2 per step (front LF, RF, hind LH, RH):", np.round(err * 1e4, 2).tolist(), "x 1e-4")
for w in [100, 300, 500]:
    print(f"  weight -{w}: costs {err.sum()*w:.3f} per step = {err.sum()*w*1000:.0f} per 20 s episode "
          f"(hind feet's share {err[2:].sum()/err.sum()*100:.0f}%)")
