"""Smooth vs. flipping commands on one hip (why action_rate exists).

Sends the front-left hip a smooth ramp, then a target that flips +/-8.6 deg
every 0.02 s, and prints target, actual angle, and motor torque each step.
Flipping: the leg barely moves, but the motor pushes back and forth.

Run from anywhere:  .venv/bin/python tools/action_rate_flip_demo.py
"""
from pathlib import Path

import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "sim" / "red_panda.xml"
ACTION_SCALE = 0.5  # target = home (0) + action * 0.5 rad, like Go1

m = mujoco.MjModel.from_xml_path(str(MODEL))
steps_per_ctrl = round(0.02 / m.opt.timestep)  # physics steps per 50 Hz policy step
a = m.actuator("fl_hip").id
j = m.joint("fl_hip").qposadr[0]


def run(name, actions):
    d = mujoco.MjData(m)
    for _ in range(int(2.0 / m.opt.timestep)):  # let the robot settle for 2 s
        mujoco.mj_step(m, d)
    print(f"\n{name}  (timestep {m.opt.timestep}s, {steps_per_ctrl} physics steps per 0.02 s)")
    print(" step  action  target(deg)  actual(deg)  motor torque (N·m)")
    for i, act in enumerate(actions):
        d.ctrl[a] = act * ACTION_SCALE
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(m, d)
        print(f" {i+1:4d}  {act:+.2f}   {np.degrees(act * ACTION_SCALE):+8.1f}"
              f"    {np.degrees(d.qpos[j]):+8.2f}      {d.actuator_force[a]:+.2f}")


run("SMOOTH ramp", [0.05 * i for i in range(7)])
run("FLIPPING", [0.3 * (-1) ** i for i in range(7)])
