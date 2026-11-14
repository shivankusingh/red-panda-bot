import json, pickle, sys
from pathlib import Path
import jax, jax.numpy as jp, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict
REPO = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as rpw
run = REPO / "runs" / sys.argv[1]; cfg = json.load(open(run / "config.json"))
c = config_dict.ConfigDict(cfg["env"]); c.impl = "jax"; c.pert_config.enable = False
if "servo_config" in c: c.servo_config.delay_ms = [0.0, 0.0]
env = rpw.RedPandaWalk(config=c)
net = cfg["ppo"]["network_factory"]
nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]), value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
    policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
policy = jax.jit(ppo_networks.make_inference_fn(nets)(pickle.load(open(run / "params.pkl", "rb")), deterministic=True))
reset, step = jax.jit(env.reset), jax.jit(env.step)
state = reset(jax.random.PRNGKey(0)); key = jax.random.PRNGKey(1)
m = env.mj_model
nod_q = m.joint("neck_nod").qposadr[0]; nod_a = m.actuator("neck_nod").id
head = m.body("head").id; torso = m.body("torso").id
L = []
for i in range(250):
    state.info["command"] = jp.array([0.2, 0., 0.]); state.info["steps_until_next_cmd"] = jp.array(1000, dtype=jp.int32)
    key, k = jax.random.split(key); a, _ = policy(state.obs, k); state = step(state, a)
    d = state.data
    R = np.array(d.xmat[torso]).reshape(3, 3)
    pitch = np.degrees(np.arcsin(-R[2, 0]))  # nose up/down of the torso
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    Rh = np.array(d.xmat[head]).reshape(3, 3)
    head_pitch = np.degrees(np.arcsin(-Rh[2, 0]))
    pt, rt = env._lean_target(state.info["gait_phase"], state.info["command"]) if env._gait_on else (0.0, 0.0)
    L.append([np.degrees(float(d.ctrl[nod_a])), np.degrees(float(d.qpos[nod_q])), pitch, roll, head_pitch, float(d.xpos[head][2]) * 100, np.degrees(float(rt))])
L = np.array(L[100:])  # skip speeding up
names = ["nod TARGET (deg)", "nod ACTUAL (deg)", "torso pitch (deg, +down)", "torso roll (deg, +right dn)", "head pitch in world (deg)", "head height (cm)", "roll TARGET (deg)"]
print(f"{'':28s}{'min':>8s}{'max':>8s}{'swing':>8s}")
for n, col in zip(names, L.T):
    print(f"{n:28s}{col.min():8.1f}{col.max():8.1f}{col.max()-col.min():8.1f}")
# how fast does the head pitch change? count up-down cycles per second
hp = L[:, 4] - L[:, 4].mean()
crossings = np.sum((hp[1:] > 0) & (hp[:-1] <= 0)) / (len(hp) * 0.02)
print(f"\nhead pitch up-down cycles per second: {crossings:.1f}")
if np.ptp(L[:, 6]) > 0.1:
    c = np.corrcoef(L[:, 3], L[:, 6])[0, 1]
    print(f"torso roll follows its target? correlation {c:+.2f} (1 = perfectly in step)")
print(f"\nsummary: head tilt swing {np.ptp(L[:, 4]):.1f} deg, head bob {np.ptp(L[:, 5]):.1f} cm, torso roll swing {np.ptp(L[:, 3]):.1f} deg, mean torso pitch {L[:, 2].mean():+.1f} deg")
