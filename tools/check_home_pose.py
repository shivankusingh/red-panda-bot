"""Load the home keyframe, let the robot settle 2 s, and measure the stance checks.

Run:  .venv/bin/python tools/check_home_pose.py  [optional/path/to/model.xml]
"""
import sys, mujoco, numpy as np
from pathlib import Path
path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "sim" / "red_panda.xml")
m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
for _ in range(int(2 / m.opt.timestep)): mujoco.mj_step(m, d)

up = d.xmat[m.body("torso").id].reshape(3, 3)[:, 2]
print(f"Torso height: {d.qpos[2]*100:.1f} cm    Body tilt: {np.degrees(np.arccos(up[2])):.2f}°  (want ~0)\n")
print("leg | knee points      | foot ahead of hip | hip torque | knee torque")
for leg, want in [("fl", "back"), ("fr", "back"), ("bl", "forward"), ("br", "forward")]:
    hip = d.xpos[m.body(f"{leg}_upper").id]
    knee = d.xpos[m.body(f"{leg}_lower").id]
    lo = m.body(f"{leg}_lower").id
    foot = d.xpos[lo] + d.xmat[lo].reshape(3, 3) @ np.array([0, 0, -0.115])
    side = knee[0] - (hip[0] + foot[0]) / 2          # + = knee is in front of the hip-foot line
    got = "forward" if side > 0 else "back"
    ok = "✓" if got == want else "✗ WRONG"
    print(f" {leg} | {got:7s} {ok:8s} | {(foot[0]-hip[0])*100:+6.2f} cm         |"
          f" {d.actuator_force[m.actuator(leg+'_hip').id]:+.3f} N·m | {d.actuator_force[m.actuator(leg+'_knee').id]:+.3f} N·m")
