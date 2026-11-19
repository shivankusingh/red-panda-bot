"""Animate the A2 knee linkage (hind leg, side view) bending 0 -> 110 -> 0 deg.

Shows the servo arm at the hip, the rod, the arm on the shin, and how hard the rod has to
push for the servo's full (stall) torque at each angle. Same geometry as cad/knee_linkage.py.

  python cad/knee_linkage_anim.py [--L 20] [--psi0 130]   -> cad/knee_linkage_anim.gif
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

THIGH, SHIN, STALL = 86.45, 80.5, 2.94
ap = argparse.ArgumentParser()
ap.add_argument("--L", type=float, default=20.0)
ap.add_argument("--psi0", type=float, default=130.0)
args = ap.parse_args()
L, PSI0 = args.L, args.psi0

bends = np.concatenate([np.linspace(0, 110, 45), np.linspace(110, 0, 45)])
u, o = np.array([0.0, -1.0]), np.array([1.0, 0.0])   # thigh straight down, knee points forward (+x)
knee = THIGH * u

fig, (ax, axf) = plt.subplots(1, 2, figsize=(9, 5.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax.set_xlim(-85, 45)
ax.set_ylim(-180, 30)
ax.set_aspect("equal")
ax.grid(alpha=0.25)
ax.set_title("Hind leg, side view (front = right)", fontsize=10)
ax.plot([0, knee[0]], [0, knee[1]], color="k", lw=7, solid_capstyle="round", zorder=1)
ax.text(-38, 6, "hip: knee servo\nturns this arm", fontsize=8, color="tab:blue")
ax.text(-30, knee[1] - 4, "knee", fontsize=8)
ax.text(-48, -40, "thigh", fontsize=9)
shin_ln, = ax.plot([], [], color="tab:brown", lw=6, solid_capstyle="round")
crank_ln, = ax.plot([], [], color="tab:blue", lw=4)
lever_ln, = ax.plot([], [], color="tab:blue", lw=4)
rod_ln, = ax.plot([], [], color="tab:red", lw=2.5)
txt = ax.text(-80, -172, "", fontsize=9, family="monospace")

bb = np.linspace(0, 110, 221)
force = STALL / (L / 1000 * np.sin(np.radians(PSI0 - bb)))
axf.plot(bb, force, color="0.4")
axf.axvspan(0, 78, color="tab:green", alpha=0.08)
axf.text(39, 0.93 * force.max(), "range the policies use", ha="center", fontsize=8, color="tab:green")
dot, = axf.plot([], [], "o", color="tab:red", ms=9)
axf.set_xlim(0, 110)
axf.set_ylim(0, force.max() * 1.05)
axf.set_xlabel("knee bend (deg)")
axf.set_ylabel("rod push at full servo torque (N)")
axf.set_title(f"Rod force (arm length {L:g} mm)", fontsize=10)
axf.grid(alpha=0.25)


def draw(i):
  b = bends[i]
  rb, pr = np.radians(b), np.radians(PSI0 - b)
  shin_dir = np.cos(rb) * u - np.sin(rb) * o
  arm = np.cos(pr) * u + np.sin(pr) * o            # both short arms point the same way
  foot = knee + SHIN * shin_dir
  c, lv = L * arm, knee + L * arm
  shin_ln.set_data([knee[0], foot[0]], [knee[1], foot[1]])
  crank_ln.set_data([0, c[0]], [0, c[1]])
  lever_ln.set_data([knee[0], lv[0]], [knee[1], lv[1]])
  rod_ln.set_data([c[0], lv[0]], [c[1], lv[1]])
  f = STALL / (L / 1000 * np.sin(pr))
  dot.set_data([b], [f])
  txt.set_text(f"knee bend {b:5.1f} deg\nrod vs arm {np.degrees(pr):5.1f} deg\nrod push  {f:5.0f} N")
  return shin_ln, crank_ln, lever_ln, rod_ln, dot, txt


fig.tight_layout()
out = Path(__file__).resolve().parent / "knee_linkage_anim.gif"
FuncAnimation(fig, draw, frames=len(bends), blit=True).save(out, writer=PillowWriter(fps=15), dpi=80)
print(f"-> {out}")
