import json, pickle, sys
from pathlib import Path
import jax, jax.numpy as jp, mujoco, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from PIL import Image
REPO = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as rpw
CMD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
run = REPO / "runs" / sys.argv[1]; cfg = json.load(open(run / "config.json"))
from ml_collections import config_dict
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
J = {n: m.joint(n).qposadr[0] for n in rpw.LEG_JOINTS + rpw.TAIL_JOINTS}
log = []
for i in range(300):  # 6 s forward at 0.3
    state.info["command"] = jp.array([CMD, 0.0, 0.0]); state.info["steps_until_next_cmd"] = jp.array(1000, dtype=jp.int32)
    key, k = jax.random.split(key); a, _ = policy(state.obs, k); state = step(state, a)
    d = state.data
    log.append(dict(q=np.array(d.qpos), feet=np.array(d.site_xpos[env._feet_site_id]), contact=np.array(state.info["last_contact"])))
log = log[100:]  # skip first 2 s
q = np.array([l["q"] for l in log]); feet = np.array([l["feet"] for l in log]); con = np.array([l["contact"] for l in log])
home = np.array(m.keyframe("home").qpos)
print("joint swing during forward walking at 0.3 m/s (min..max relative to home, degrees):")
print(f"== {sys.argv[1]} ==")
for leg in ["fl", "bl"]:
    for j in ["hip", "knee"]:
        n = f"{leg}_{j}"; v = np.degrees(q[:, J[n]] - home[J[n]])
        print(f"  {n:8s} {v.min():+6.1f} .. {v.max():+6.1f}   (total {v.max()-v.min():5.1f} deg; leash allows +/-28.6)")
for n in rpw.TAIL_JOINTS:
    v = np.degrees(q[:, J[n]]); print(f"  {n:10s} {v.min():+6.1f} .. {v.max():+6.1f}  (total {v.max()-v.min():5.1f} deg)")
# steps: count touchdowns per foot
touch = (con[1:] & ~con[:-1]).sum(axis=0); secs = len(log) * env.dt
freq = touch / secs
print(f"\nsteps per second per foot: {np.round(freq, 2)}  -> stride length = speed / freq = {0.3/np.mean(freq)*100:.1f} cm")
lift = feet[:, :, 2].max(axis=0) - feet[:, :, 2].min(axis=0)
print(f"foot lift (cm): {np.round(lift*100, 1)}   (target 3.0)")
print(f"time on ground (duty): {np.round(con.mean(axis=0), 2)}")
# side-view frames
spec = mujoco.MjSpec.from_file(str(rpw.XML_PATH))
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[.85,.85,.8], rgb2=[.7,.7,.65], width=300, height=300)
mat = spec.add_material(name="grid", texrepeat=[20, 20]); mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
spec.geom("floor").material = "grid"
vm = spec.compile(); vd = mujoco.MjData(vm); r = mujoco.Renderer(vm, 300, 400)
cam = mujoco.MjvCamera(); cam.distance, cam.elevation = 0.75, -5
tiles = []
for i in range(0, 32, 4):  # 8 frames, 0.08 s apart
    vd.qpos[:] = q[i]; mujoco.mj_forward(vm, vd)
    # face the robot's side: camera azimuth = robot heading + 90
    yaw = np.degrees(np.arctan2(vd.xmat[1].reshape(3,3)[1,0], vd.xmat[1].reshape(3,3)[0,0]))
    cam.azimuth = yaw + 90; cam.lookat[:] = vd.qpos[:3]
    r.update_scene(vd, cam); tiles.append(Image.fromarray(r.render()))
r.close()
sheet = Image.new("RGB", (400 * 4, 300 * 2))
for k, t in enumerate(tiles): sheet.paste(t, ((k % 4) * 400, (k // 4) * 300))
sheet.save(str(Path(__file__).parent / f"side_{sys.argv[1]}.png"))
