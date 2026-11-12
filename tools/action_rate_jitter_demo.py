"""Smooth swing vs. the same swing with a zigzag on top (action_rate's job).

The heavy, slow leg filters out the zigzag, so both runs move the hip almost
identically, but the jittery run's motor torque jumps around and its
action_rate is ~37x larger. Without action_rate, the reward can't tell them apart.

Run from anywhere:  .venv/bin/python tools/action_rate_jitter_demo.py
"""
from pathlib import Path

import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "sim" / "red_panda.xml"
ACTION_SCALE = 0.5
N_STEPS = 15
JITTER = 0.15  # zigzag size added on top of the smooth ramp

m = mujoco.MjModel.from_xml_path(str(MODEL))
steps_per_ctrl = round(0.02 / m.opt.timestep)
a = m.actuator("fl_hip").id
j = m.joint("fl_hip").qposadr[0]


def run(actions):
    d = mujoco.MjData(m)
    for _ in range(int(2.0 / m.opt.timestep)):  # settle 2 s
        mujoco.mj_step(m, d)
    out = []
    for act in actions:
        d.ctrl[a] = act * ACTION_SCALE
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(m, d)
        out.append((np.degrees(d.qpos[j]), d.actuator_force[a]))
    return out


def action_rate(acts):
    return sum((acts[i] - acts[i - 1]) ** 2 for i in range(1, len(acts)))


smooth = [0.05 * i for i in range(N_STEPS)]
jitter = [s + JITTER * (-1) ** i for i, s in enumerate(smooth)]
S, J = run(smooth), run(jitter)

print(" step | action smooth  jittery | actual angle smooth  jittery | torque smooth  jittery")
for i in range(N_STEPS):
    print(f" {i+1:4d} |   {smooth[i]:+.2f}    {jitter[i]:+.2f}  |"
          f"      {S[i][0]:+6.2f}°   {J[i][0]:+6.2f}°  |   {S[i][1]:+.2f}    {J[i][1]:+.2f}")
print(f"\naction_rate total:  smooth {action_rate(smooth):.3f}   jittery {action_rate(jitter):.3f}")
