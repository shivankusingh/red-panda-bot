"""How fast could the robot possibly rear up? A best-case bound, from the sim model.

1. Joint travel: each joint's move from the walking stance to the tripod pose (option A,
   tools/eval/tripod_static.py) at the servo's no-load speed. Ignores the body's weight.
2. Body rotation: treat the robot as one stiff body pivoting 72 deg (2 deg nose-down ->
   70 deg nose-up) about its hind feet. Drive = 2 hind hips + tail_lift on the servo
   torque-speed line (joint speed = body speed x how far each joint turns per body degree);
   gravity pulls the nose down. Full drive, then full braking, switched at the best moment
   so it stops exactly at 72 deg. Masses and inertias come from the MuJoCo model.

Both are lower bounds: a real rear-up also has to plant the tail, keep its feet from
sliding, and cope with servo delay and randomization. Use it to pick deadlines that
leave room (a deadline under the bound forces the servos to pin, like run 23c).

  python tools/eval/rear_speed_limit.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tripod_static as ts  # noqa: E402

STALL, W0, G = 2.9, 4.7, 9.81  # model stall torque (N*m), no-load speed (rad/s)
TURN = np.radians(72)


def tripod_pose():
  best = None
  for dx in np.arange(0.0, 0.12, 0.005):
    p = ts.pose(np.radians(70), 0.16, dx)
    if p is not None and (best is None or p[1] > best[0]):
      best = (p[1], p[0])  # (CoM margin, joint angles)
  return best[1]


def stick(mass_scale, cap):
  """Fastest 72 deg rotation (s) with every servo's torque capped at `cap` N*m."""
  m = ts.build(mass_scale)
  d = mujoco.MjData(m)
  d.qpos[:] = ts.HOME
  mujoco.mj_forward(m, d)
  M = m.body_mass.sum()
  feet = (d.site_xpos[m.site("bl_foot").id] + d.site_xpos[m.site("br_foot").id]) / 2
  com = (m.body_mass[:, None] * d.xipos).sum(0) / M
  inertia = 0.0  # about the y axis through the hind feet
  for b in range(1, m.nbody):
    R = d.ximat[b].reshape(3, 3)
    r = d.xipos[b] - feet
    inertia += (R @ np.diag(m.body_inertia[b]) @ R.T)[1, 1] + m.body_mass[b] * (r[0] ** 2 + r[2] ** 2)
  rx, rz = com[0] - feet[0], com[2] - feet[2]
  ang0, L = np.arctan2(rz, rx), np.hypot(rx, rz)
  q = tripod_pose()
  ratios = [(2, abs(q[ts.QADR["bl_hip"]] - ts.HOME[ts.QADR["bl_hip"]]) / TURN),
            (1, abs(q[ts.QADR["tail_lift"]] - ts.HOME[ts.QADR["tail_lift"]]) / TURN)]

  def drive(w, speeding_up):
    s = -1 if speeding_up else 1  # braking can use up to stall
    return sum(n * r * min(cap, STALL * np.clip(1 + s * w * r / W0, 0, 1)) for n, r in ratios)

  def run(switch, dt=1e-4):
    th = w = t = 0.0
    while t < 3:
      tq = drive(w, True) if th < switch else -drive(w, False)
      w += (tq - M * G * L * np.cos(ang0 + th)) / inertia * dt
      th += w * dt
      t += dt
      if th >= switch and w <= 0:
        return t, th
      if t > 0.05 and th < 0:
        return None
    return None

  for sw in np.linspace(0.2, 1.4, 481):
    r = run(sw)
    if r and abs(r[1] - TURN) < np.radians(1):
      return r[0], M, 100 * rx
  return None, M, 100 * rx


def main():
  q = tripod_pose()
  m = ts.m
  print("1. Joint travel at no-load speed (4.7 rad/s):")
  for i in range(m.nu):
    a = m.jnt_qposadr[m.actuator_trnid[i, 0]]
    dq = q[a] - ts.HOME[a]
    if abs(dq) > 1e-3:
      print(f"   {m.actuator(i).name:10s} {np.degrees(dq):+6.1f} deg  {1000 * abs(dq) / W0:4.0f} ms")
  print("\n2. Stiff body pivoting 72 deg about the hind feet (best-case timing):")
  for scale in [1.0, 1.1]:
    for name, cap in [("stall", STALL), ("65% of stall (peak rule)", 0.65 * 2.94)]:
      t, M, rx = stick(scale, cap)
      print(f"   {M:.2f} kg, torque up to {name:<25s}: {1000 * t:4.0f} ms   (CoM {rx:.1f} cm ahead of the hind feet)")


if __name__ == "__main__":
  main()
