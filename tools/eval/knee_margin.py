"""Knee servo margin: how close to their limit do the knees run, and what if the battery sags?

For each run: walk forward for 12 s (16 robots, flat, no pushes/delay) with the normal
servos and with servos 10% weaker AND slower (about a 3S pack sagging under 12-14 A).
Reports falls, % of time each knee uses >= 95% of the torque available at its current
speed (the torque-speed line), the 99th-percentile knee speed as % of no-load speed, and
the 90th-percentile swing height. Run 23c: hind knees pinned 33-36% (fails); 24a: < 1%.

Usage: python tools/eval/knee_margin.py rp_run24a [rp_run20b ...] [--vx 0.12]
"""
import json, pickle, sys
from pathlib import Path
import jax, jax.numpy as jp, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict
REPO = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as rpw
S, T = 16, 600
VX = 0.12
if "--vx" in sys.argv:
    i = sys.argv.index("--vx"); VX = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
def rollout(RUN, weaken):
    run = REPO / "runs" / RUN; cfg = json.load(open(run / "config.json"))
    c = config_dict.ConfigDict(cfg["env"]); c.impl = "jax"; c.pert_config.enable = False; c.servo_config.delay_ms = [0.0, 0.0]
    rpw.use_flat(c)
    c.servo_config.no_load_speed *= weaken
    env = rpw.RedPandaWalk(config=c); env._stall = env._stall * weaken   # weaker servo: less torque AND less speed
    net = cfg["ppo"]["network_factory"]
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]), value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
        policy_obs_key=net["policy_obs_key"], value_obs_key=net["value_obs_key"])
    policy = jax.jit(jax.vmap(ppo_networks.make_inference_fn(nets)(pickle.load(open(run / "params.pkl", "rb")), deterministic=True)))
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    knee_act = np.array(env._leg_act[2::3]); w0 = c.servo_config.no_load_speed; stall = np.array(env._stall)[knee_act]
    st = reset(jax.random.split(jax.random.PRNGKey(0), S)); key = jax.random.PRNGKey(1)
    R, Wr, H, U = [], [], [], []; fell = np.zeros(S, bool)
    for t in range(T):
        st.info["command"] = jp.tile(jp.array([VX, 0., 0.]), (S, 1)); st.info["steps_until_next_cmd"] = jp.full((S,), 1000, dtype=jp.int32)
        key, k = jax.random.split(key); a, _ = policy(st.obs, jax.random.split(k, S)); st = step(st, a)
        fell |= np.array(st.done) > 0
        if t < 100: continue
        f = np.array(st.data.actuator_force)[:, knee_act]
        w = np.array(st.data.qvel)[:, np.array(env._act_qvel)[knee_act]]
        avail = stall * np.clip(1.0 - np.sign(f) * w / w0, 0.0, 1.0)   # torque available pushing the way f pushes
        R.append(np.abs(f) / np.maximum(avail, 1e-6)); Wr.append(np.abs(w) / w0)
        H.append(np.array(jax.vmap(env._feet_height)(st.data))); U.append(~np.array(jax.vmap(env._desired_contact)(st.info["gait_phase"], st.info["command"])))
    R, Wr, H, U = map(np.array, (R, Wr, H, U))
    ok = ~fell
    pinned = (R[:, ok] > 0.95).mean((0, 1)) * 100
    speed = np.percentile(Wr[:, ok], 99, axis=(0, 1)) * 100
    lift = [np.percentile(H[:, ok, f][U[:, ok, f]], 90) * 100 - 0.97 for f in range(4)]
    return fell.sum(), pinned, speed, lift
print(f"knee servos, walking {VX} m/s on flat, 16 robots x 12 s. legs LF RF LH RH")
print(f"{'run':10s} {'servos':8s} {'fell':>5s} | {'% of time knee pinned at limit':>31s} | {'99th pct knee speed, % of max':>30s} | {'swing height (90th pct) cm':>27s}")
for RUN in sys.argv[1:]:
    for weaken, name in [(1.0, "12 V"), (0.9, "sag x0.9")]:
        f, p, s, l = rollout(RUN, weaken)
        print(f"{RUN:10s} {name:8s} {f:>2d}/{S} | " + " ".join(f"{x:6.1f}" for x in p) + "    | " + " ".join(f"{x:6.0f}" for x in s) + "   | " + " ".join(f"{x:5.1f}" for x in l), flush=True)
