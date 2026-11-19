"""Animate one hind leg with all three servos: hip-side (leg outward), hip (swing), knee (rod).

Side view (left) and front view (right) of the left hind leg, same sizes as the sim.
Four parts, one servo at a time, then all together:
  1. knee servo turns   -> the arms + rod bend the knee
  2. hip servo turns    -> the whole thigh swings; the knee servo rides along, the knee
                           bend doesn't change
  3. hip-side servo     -> the whole leg (hip servo, thigh, knee servo, rod, shin) tilts
                           outward together; nothing in the rod mechanism changes
  4. all three together (a walking-like wiggle)

  python cad/leg_anim.py [--L 20] [--psi0 130]   -> cad/leg_anim.gif
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

THIGH, SHIN, FOOT_R = 86.45, 80.5, 10.0
FLOOR = 158.0   # torso centre (= hip-side axis height) above the floor when standing, mm
SIDE_AXIS_Y, LEG_Y, ROD_Y = 40.0, 70.0, 82.0   # mm out from the centre line (rod beside the thigh)
ap = argparse.ArgumentParser()
ap.add_argument("--L", type=float, default=20.0)
ap.add_argument("--psi0", type=float, default=130.0)
args = ap.parse_args()
L, PSI0 = args.L, args.psi0
HOME = dict(side=0.0, hip=25.4, bend=48.6)   # hind leg standing pose (cad/variables.md)


def ramp(a, b, n):
  return list(np.linspace(a, b, n))


# --- timeline: (hip_side, hip, bend, caption, which servo moves)
steps = []
def add(key, values, caption):
  for v in values:
    s = dict(steps[-1][0]) if steps else dict(HOME)
    s[key] = v
    steps.append((s, caption, key))

pause = lambda n, cap: [steps.append((dict(steps[-1][0]), cap, None)) for _ in range(n)]  # noqa: E731
add("bend", [HOME["bend"]] * 10, "standing pose")
cap = "1. KNEE servo turns: arms + rod bend the knee"
add("bend", ramp(48.6, 95, 15) + ramp(95, 10, 22) + ramp(10, 48.6, 12), cap)
cap = "2. HIP servo turns: thigh swings, knee servo rides along,\n   knee bend stays the same"
add("hip", ramp(25.4, -20, 16) + ramp(-20, 45, 22) + ramp(45, 25.4, 8), cap)
cap = "3. HIP-SIDE servo turns: the whole leg tilts outward together,\n   rod mechanism unchanged"
add("side", ramp(0, 28, 16) + ramp(28, -7, 20) + ramp(-7, 0, 8), cap)
cap = "4. all three servos together"
t = np.linspace(0, 2 * np.pi, 50)
for k in range(len(t)):
  s = dict(side=12 * np.sin(t[k]), hip=25.4 + 15 * np.sin(t[k] + 1.0), bend=48.6 + 20 * np.sin(2 * t[k]))
  steps.append((s, cap, "all"))


def leg_points(side, hip, bend):
  """3D points (body frame, mm, origin = hip-side axis; x forward, y outward, z up)."""
  th, rb, pr = np.radians(hip), np.radians(bend), np.radians(PSI0 - bend)
  u = np.array([np.sin(th), 0, -np.cos(th)])          # thigh direction
  o = np.array([np.cos(th), 0, np.sin(th)])           # the side the hind knee points to (front)
  s = np.cos(rb) * u - np.sin(rb) * o                 # shin direction
  a = np.cos(pr) * u + np.sin(pr) * o                 # both short arms
  hip_c = np.array([0, LEG_Y - SIDE_AXIS_Y, 0])
  knee = hip_c + THIGH * u
  foot = knee + SHIN * s
  dy = np.array([0, ROD_Y - LEG_Y, 0])                # arms + rod sit beside the thigh
  P = dict(side_axis=np.zeros(3), hip=hip_c, knee=knee, foot=foot,
           crank0=hip_c + dy, crank=hip_c + dy + L * a, lever0=knee + dy, lever=knee + dy + L * a)
  ph = np.radians(side)                               # tilt about x: + = foot outward
  R = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
  return {k: R @ v for k, v in P.items()}


fig, (axs, axf) = plt.subplots(1, 2, figsize=(10, 5.6))
for ax, title in ((axs, "SIDE view (front = right)"), (axf, "FRONT view (looking at the robot's face)")):
  ax.set_aspect("equal")
  ax.grid(alpha=0.2)
  ax.set_title(title, fontsize=10)
  ax.set_ylim(-205, 45)
axs.set_xlim(-130, 120)
axf.set_xlim(-100, 150)
# torso: side view = the box behind/ahead of the hip, front view = cross-section
axs.add_patch(plt.Rectangle((-40, -25), 320, 50, color="0.85", zorder=0))  # hind hip is 4 cm from the back
axs.text(60, 5, "torso", fontsize=9, color="0.4")
axf.add_patch(plt.Rectangle((-SIDE_AXIS_Y - 50, -25), 100, 50, color="0.85", zorder=0))
axf.axvline(-SIDE_AXIS_Y, color="0.6", ls=":", lw=1)
axf.text(-SIDE_AXIS_Y + 2, -200, "robot's\ncentre line", fontsize=8, color="0.5")
axf.text(80, -200, "outward ->", fontsize=8, color="0.5")
for ax in (axs, axf):
  ax.axhline(-FLOOR, color="0.5", lw=1)
  ax.text(ax.get_xlim()[0] + 3, -FLOOR - 12, "floor (standing)", fontsize=7, color="0.5")

def mk(ax):
  return dict(thigh=ax.plot([], [], color="k", lw=6, solid_capstyle="round")[0],
              shin=ax.plot([], [], color="tab:brown", lw=5, solid_capstyle="round")[0],
              crank=ax.plot([], [], color="tab:blue", lw=3.5)[0],
              lever=ax.plot([], [], color="tab:blue", lw=3.5)[0],
              rod=ax.plot([], [], color="tab:red", lw=2.2)[0],
              link=ax.plot([], [], color="0.5", lw=3)[0],
              foot=ax.plot([], [], "o", color="tab:brown", ms=7)[0],
              servos=ax.plot([], [], "s", ms=9, mfc="white", mec="k", mew=1.5)[0],
              active=ax.plot([], [], "s", ms=13, mfc="gold", mec="k", mew=1.5)[0])
A, B = mk(axs), mk(axf)
labels = [axf.text(0, 0, t, ha="center", fontsize=7.5, weight="bold") for t in ("hip-side", "hip", "knee")]
axs.text(5, 12, "all 3 servos line up here\n(see the front view)", fontsize=7, color="0.3")
caption = fig.text(0.5, 0.955, "", ha="center", va="top", fontsize=11, weight="bold")
readout = axf.text(55, -15, "", fontsize=8, family="monospace", va="top")
fig.text(0.5, 0.02, "white squares = servos (gold = the one moving).  black thigh, brown shin, "
         "blue arms, red rod", ha="center", fontsize=8.5)


def show(art, P, view, mover):
  ix = (0, 2) if view == "side" else (1, 2)
  xy = lambda *ks: ([P[k][ix[0]] for k in ks], [P[k][ix[1]] for k in ks])  # noqa: E731
  art["thigh"].set_data(*xy("hip", "knee"))
  art["shin"].set_data(*xy("knee", "foot"))
  art["crank"].set_data(*xy("crank0", "crank"))
  art["lever"].set_data(*xy("lever0", "lever"))
  art["rod"].set_data(*xy("crank", "lever"))
  art["link"].set_data(*xy("side_axis", "hip"))
  art["foot"].set_data(*xy("foot"))
  art["servos"].set_data(*xy("side_axis", "hip", "crank0"))
  key = {"side": ["side_axis"], "hip": ["hip"], "bend": ["crank0"], "all": ["side_axis", "hip", "crank0"]}
  art["active"].set_data(*xy(*key.get(mover, [])) if mover in key else ([], []))


def draw(i):
  s, cap, mover = steps[i]
  P = leg_points(s["side"], s["hip"], s["bend"])
  show(A, P, "side", mover)
  show(B, P, "front", mover)
  for lab, k, dy in zip(labels, ("side_axis", "hip", "crank0"), (14, 26, 14)):
    lab.set_position((P[k][1], P[k][2] + dy))
  caption.set_text(cap)
  readout.set_text(f"hip-side {s['side']:+5.1f} deg\n"
                   f"hip      {s['hip']:+5.1f} deg\n"
                   f"knee     {s['bend']:5.1f} deg")
  return []


fig.subplots_adjust(top=0.84, bottom=0.08, left=0.05, right=0.98, wspace=0.1)
out = Path(__file__).resolve().parent / "leg_anim.gif"
FuncAnimation(fig, draw, frames=len(steps)).save(out, writer=PillowWriter(fps=15), dpi=72)
print(f"-> {out}  ({len(steps)} frames)")
