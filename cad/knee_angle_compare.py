"""Why the knee arm sits 50 deg off the thigh: three legs side by side, same knee motion,
the arm fixed at three different angles. A bar under each shows how hard the rod must push
(servo at full torque, L = 20 mm). Near "JAM" the arm lines up with the rod.

  python cad/knee_angle_compare.py   -> cad/knee_angle_compare.gif
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

THIGH, SHIN, L, STALL = 86.45, 80.5, 20.0, 2.94
OFFS = [15, 50, 80]                       # arm angle off the thigh line at a straight knee (deg)
NAMES = ["15° : too close to the rod", "50° : chosen", "80° : too far"]
FMAX = 600.0                              # N, bar full scale
bends = np.concatenate([np.linspace(0, 110, 50), [110] * 6, np.linspace(110, 0, 50), [0] * 6])
u, o = np.array([0.0, -1.0]), np.array([1.0, 0.0])
knee = THIGH * u

fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), gridspec_kw=dict(height_ratios=[4, 1]))
arts = []
for k, (off, name) in enumerate(zip(OFFS, NAMES)):
  ax, bx = axes[0, k], axes[1, k]
  ax.set_xlim(-90, 45)
  ax.set_ylim(-180, 30)
  ax.set_aspect("equal")
  ax.axis("off")
  ax.set_title(name, fontsize=11, weight="bold", color="tab:green" if off == 50 else "k")
  ax.plot([0, knee[0]], [0, knee[1]], color="k", lw=7, solid_capstyle="round")
  a = dict(shin=ax.plot([], [], color="tab:brown", lw=6, solid_capstyle="round")[0],
           crank=ax.plot([], [], color="tab:blue", lw=4)[0],
           lever=ax.plot([], [], color="tab:blue", lw=4)[0],
           rod=ax.plot([], [], color="tab:red", lw=2.5)[0],
           jam=ax.text(-85, -20, "", fontsize=16, weight="bold", color="tab:red"),
           bend=ax.text(-85, -175, "", fontsize=10, family="monospace"))
  bx.set_xlim(0, FMAX)
  bx.set_ylim(0, 1)
  bx.set_yticks([])
  bx.set_xlabel("rod push needed (N)", fontsize=9)
  bx.axvspan(400, FMAX, color="tab:red", alpha=0.1)
  a["bar"] = bx.barh([0.5], [0], height=0.6, color="tab:green")[0]
  a["num"] = bx.text(5, 0.5, "", va="center", fontsize=10, weight="bold")
  arts.append(a)
fig.text(0.5, 0.01, "Same knee motion in all three (0 -> 110° -> 0). Only the arm's fixed angle differs. "
         "Blue arms always point the same way; red = rod.", ha="center", fontsize=9)


def draw(i):
  b = bends[i]
  for off, a in zip(OFFS, arts):
    psi = 180 - off - b                              # angle between the arm and the thigh line past the knee
    rb, pr = np.radians(b), np.radians(psi)
    shin_dir = np.cos(rb) * u - np.sin(rb) * o
    arm = np.cos(pr) * u + np.sin(pr) * o
    foot = knee + SHIN * shin_dir
    a["shin"].set_data([knee[0], foot[0]], [knee[1], foot[1]])
    a["crank"].set_data([0, L * arm[0]], [0, L * arm[1]])
    a["lever"].set_data([knee[0], knee[0] + L * arm[0]], [knee[1], knee[1] + L * arm[1]])
    a["rod"].set_data([L * arm[0], knee[0] + L * arm[0]], [L * arm[1], knee[1] + L * arm[1]])
    s = abs(np.sin(pr))
    f = STALL / (L / 1000 * max(s, 1e-3))
    jammed = s < 0.17                                # within ~10 deg of lining up
    a["jam"].set_text("JAM!" if jammed else "")
    a["bar"].set_width(min(f, FMAX))
    a["bar"].set_color("tab:red" if f > 400 else ("tab:orange" if f > 250 else "tab:green"))
    a["num"].set_text(f"{f:.0f} N" if f < 5000 else "∞ (stuck)")
    a["bend"].set_text(f"knee bend {b:5.1f}°")
  return []


fig.tight_layout(rect=(0, 0.03, 1, 1))
out = Path(__file__).resolve().parent / "knee_angle_compare.gif"
FuncAnimation(fig, draw, frames=len(bends)).save(out, writer=PillowWriter(fps=12), dpi=70)
print(f"-> {out}")
