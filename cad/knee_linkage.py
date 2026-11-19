"""Knee rod linkage (A2): where to put the lever so the rod never jams, and how hard the rod works.

The A2 knee: the knee servo rides on the thigh at the hip axis. A crank on its horn and a
lever on the shin (same length L) are joined by a rod as long as the thigh, so the four
pivots make a parallelogram: the rod stays parallel to the thigh and the shin turns exactly
as much as the crank (1:1, the servo angle = the sim's knee joint).

Side view of one leg, looking along the hip axis:
  u   = thigh direction, hip -> knee
  psi = lever angle, measured from u (the line continuing past the knee). The crank at the
        hip is always parallel to the lever, so it has the same psi.
  "outside" = the side the knee points to (front for hind legs, back for front legs), like a
        kneecap. "inside" = the side the shin folds toward.
As the knee bends by b, the lever turns with the shin toward the inside:
  outside lever: psi = psi0 - b        inside lever: psi = psi0 + b
Dead point: psi = 0 or 180 deg (crank, rod and lever in one line). There the rod pushes
straight through the pivots and can't turn the shin at all; the parallelogram can even flip
into a crossed shape. Keep psi well away from 0 and 180 over the whole knee range.

Rod force for a knee torque tau (the torque arm is L * sin psi):
  rod tension T = tau / (L sin psi)     (inside lever)
  rod tension T = -tau / (L sin psi)    (outside lever)
with tau > 0 = folding the knee. Holding the body up means tau < 0 (straightening), so an
outside lever puts the rod in TENSION when the leg carries weight (like a quadriceps tendon),
and an inside lever puts it in COMPRESSION. A thin rod under compression can buckle.

Knee torques come from the sim: tools/eval/chain_check.py --legs saves every knee's bend and
torque through walk -> rear -> hold -> come down -> walk (sampled every 20 ms; short spikes in
between are missed). Torques are the same for any linkage (1:1), only the rod force changes.

  python cad/knee_linkage.py [--L 20] [--rod M4] [--side outside] [--psi0 130]
  (no --psi0: picks the one with the lowest peak rod force, >= MIN_MARGIN from dead points
   over the design range 0-110 deg) -> cad/knee_linkage.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
THIGH = 86.45            # mm, = rod length (cad/variables.md)
SHIN = 80.5              # mm, knee to foot ball centre
DESIGN = (0.0, 110.0)    # knee bend range the parts must allow (CLAUDE.md)
MIN_MARGIN = 20.0        # deg, closest the lever may come to a dead point over DESIGN
STALL, RATED = 2.94, 0.98   # N*m, STS3215 12 V
# Threaded steel rod (stainless, E = 193 GPa), sized by the thread's root (minor) diameter.
# Ball links ~ pins, so Euler buckling with K = 1 over the pivot-to-pivot length (= thigh).
D_ROOT = {"M3": 2.387, "M4": 3.141}   # mm
E_STEEL = 193e3                       # N/mm^2

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=float, default=20.0, help="crank = lever length, mm (decided 2026-11-19: 20)")
ap.add_argument("--rod", choices=list(D_ROOT), default="M4", help="threaded rod size (decided: M4)")
ap.add_argument("--side", choices=["outside", "inside"], default="outside")
ap.add_argument("--psi0", type=float, default=None, help="lever angle at a straight knee, deg")
ap.add_argument("--down", default="dn_run5b", help="run folder holding chain*/legs_raw.npz")
args = ap.parse_args()
P_BUCKLE = np.pi ** 2 * E_STEEL * (np.pi * D_ROOT[args.rod] ** 4 / 64) / THIGH ** 2   # N


def psi_of(bend, psi0, side):
  return psi0 - bend if side == "outside" else psi0 + bend


def tension(tau, bend, psi0, side, L):
  """Rod tension in N (negative = compression). tau in N*m, + = folding; L in mm."""
  s = np.sin(np.radians(psi_of(bend, psi0, side)))
  t = tau / (L / 1000 * s)
  return -t if side == "outside" else t


def margin(psi0, side):
  """Closest approach to a dead point (deg) over the design range."""
  p = psi_of(np.linspace(*DESIGN, 111), psi0, side)
  return float(np.min(np.minimum(p, 180 - p)))


# --- sim knee samples (front and hind, nominal chain and randomized chain with shoves)
samples = []
for sub, label in (("chain", "nominal"), ("chain_random", "randomized + shoves")):
  f = REPO / "runs" / args.down / sub / "legs_raw.npz"
  if f.exists():
    z = np.load(f)
    for g in ("front", "hind"):
      samples.append((f"{g}, {label}", z[f"{g}_bend"].ravel(), z[f"{g}_tau"].ravel()))
if not samples:
  print("no sim samples found: run  python tools/eval/chain_check.py --legs  (and --random) first")
bend_all = np.concatenate([s[1] for s in samples]) if samples else np.array([0.0])
tau_all = np.concatenate([s[2] for s in samples]) if samples else np.array([0.0])


def rod_stats(psi0, side, L):
  t = tension(tau_all, bend_all, psi0, side, L)
  return dict(peak_tension=max(t.max(), 0.0), peak_compression=max(-t.min(), 0.0),
              p99=float(np.percentile(np.abs(t), 99)))


# --- pick psi0
lo, hi = (DESIGN[1], 180.0) if args.side == "outside" else (0.0, 180.0 - DESIGN[1])
grid = np.arange(lo, hi + 0.1, 1.0)
ok = [p for p in grid if margin(p, args.side) >= MIN_MARGIN]
if args.psi0 is None:
  args.psi0 = min(ok, key=lambda p: max(rod_stats(p, args.side, args.L)["peak_tension"],
                                        rod_stats(p, args.side, args.L)["peak_compression"]))
psi0, side, L = args.psi0, args.side, args.L

print(f"knee linkage A2: thigh = rod = {THIGH} mm, design knee range {DESIGN[0]:.0f}-{DESIGN[1]:.0f} deg, "
      f"{side} lever")
print(f"sim samples: {len(tau_all)} knee readings; bends {bend_all.min():.0f}-{bend_all.max():.0f} deg, "
      f"|torque| peak {np.abs(tau_all).max():.2f} N*m, p99 {np.percentile(np.abs(tau_all), 99):.2f}")
print(f"{args.rod} stainless rod, {THIGH} mm between ball centres: buckles at ~{P_BUCKLE:.0f} N in compression "
      "(ideal straight rod; real threaded rod: count on less)\n")

print(f"lever angle at a straight knee (psi0), L = {L} mm. Forces in N; compression can buckle the rod.")
print(f"{'psi0':>5s} {'psi range':>12s} {'dead-pt margin':>15s} {'peak tension':>13s} "
      f"{'peak compr.':>12s} {'p99 |F|':>8s} {'stall worst':>12s}")
for p in sorted(set(list(np.arange(lo + 5, hi, 5.0)) + [psi0])):
  if margin(p, side) <= 0:
    continue
  r = rod_stats(p, side, L)
  smin = np.min(np.abs(np.sin(np.radians(psi_of(np.linspace(*DESIGN, 111), p, side)))))
  ps = psi_of(np.array(DESIGN), p, side)
  mark = "  <- chosen" if p == psi0 else ""
  print(f"{p:5.0f} {ps.min():5.0f} to {ps.max():4.0f} {margin(p, side):13.0f} deg {r['peak_tension']:12.0f} "
        f"{r['peak_compression']:12.0f} {r['p99']:8.0f} {STALL / (L / 1000 * smin):11.0f}{mark}")

print(f"\nlever length, at psi0 = {psi0:.0f} ({side}):")
print(f"{'L mm':>5s} {'peak tension':>13s} {'peak compr.':>12s} {'compr. vs buckling':>19s} {'p99 |F|':>8s}")
for Lx in (12, 15, 20, 25):
  r = rod_stats(psi0, side, Lx)
  sf = P_BUCKLE / r["peak_compression"] if r["peak_compression"] > 0 else np.inf
  print(f"{Lx:5d} {r['peak_tension']:12.0f} {r['peak_compression']:12.0f} {sf:15.1f}x safe {r['p99']:8.0f}")
print("\nThe knee servo's output shaft takes the same force sideways (the rod pulls/pushes the crank pin),")
print("plus the twist. Ball links, the printed lever and the crank must carry it too.")

# --- figure: (a) the linkage at three bends, (b) rod force vs knee bend with the sim samples
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw=dict(width_ratios=[1, 1.4]))
u, o = np.array([0.0, -1.0]), np.array([1.0, 0.0])  # thigh straight down; knee points +x (hind leg)
if side == "inside":
  o_lever = -o
else:
  o_lever = o
knee = THIGH * u
for b, a in ((0, 1.0), (55, 0.55), (110, 0.25)):
  rb = np.radians(b)
  shin_dir = np.cos(rb) * u - np.sin(rb) * o          # shin folds to -x (inside)
  pr = np.radians(psi_of(b, psi0, side))
  lev = np.cos(pr) * u + np.sin(pr) * o_lever
  crank_tip, lever_tip = L * lev, knee + L * lev
  foot = knee + SHIN * shin_dir
  ax1.plot([knee[0], foot[0]], [knee[1], foot[1]], color="tab:brown", lw=4, alpha=a)
  ax1.plot([0, crank_tip[0]], [0, crank_tip[1]], color="tab:blue", lw=3, alpha=a)
  ax1.plot([knee[0], lever_tip[0]], [knee[1], lever_tip[1]], color="tab:blue", lw=3, alpha=a)
  ax1.plot([crank_tip[0], lever_tip[0]], [crank_tip[1], lever_tip[1]], color="tab:red", lw=1.5, alpha=a)
  ax1.annotate(f"bend {b}°", foot, textcoords="offset points", xytext=(4, -10), fontsize=9, alpha=max(a, 0.5))
ax1.plot([0, knee[0]], [0, knee[1]], color="k", lw=6, solid_capstyle="round")
ax1.plot([0, knee[0]], [0, knee[1]], color="0.6", lw=0.8, ls="--", zorder=0)
ax1.text(1, 3, "hip axis\n(hip + knee servo)", fontsize=8)
ax1.text(3, knee[1], "knee", fontsize=8)
ax1.set_title(f"Hind leg, side view ({side} lever, L = {L:g} mm, psi0 = {psi0:.0f}°)\n"
              "black thigh, brown shin, blue crank + lever, red rod. Front = right", fontsize=10)
ax1.set_aspect("equal")
ax1.grid(alpha=0.3)
ax1.set_xlabel("mm")

bb = np.linspace(*DESIGN, 221)
for tau, ls, lab in ((STALL, "-", "stall 2.94 N·m"), (RATED, ":", "rated 0.98 N·m")):
  for sgn in (1, -1):
    ax2.plot(bb, tension(sgn * tau, bb, psi0, side, L), color="0.4", ls=ls, lw=1,
             label=lab if sgn == 1 else None)
colors = {"front": "tab:orange", "hind": "tab:green"}
for lab, b_, t_ in samples:
  k = np.random.default_rng(0).choice(len(b_), size=min(len(b_), 4000), replace=False)
  ax2.scatter(b_[k], tension(t_[k], b_[k], psi0, side, L), s=3, alpha=0.25,
              color=colors[lab.split(",")[0]], marker="o" if "nominal" in lab else "x", label=lab)
ax2.axhline(-P_BUCKLE, color="tab:red", lw=1.5, label=f"{args.rod} rod buckles (~{P_BUCKLE:.0f} N)")
ax2.axhspan(-1e4, 0, color="tab:red", alpha=0.05)
ax2.text(DESIGN[1] - 2, -20, "compression", ha="right", va="top", color="tab:red", fontsize=9)
ax2.text(DESIGN[1] - 2, 20, "tension", ha="right", va="bottom", fontsize=9)
lim = 1.15 * np.abs(tension(STALL, bb, psi0, side, L)).max()
ax2.set_ylim(-lim, lim)
if P_BUCKLE > lim:
  ax2.text(1, -0.97 * lim, f"{args.rod} rod buckles at ~{P_BUCKLE:.0f} N: below this chart", fontsize=8,
           color="tab:red", va="bottom")
ax2.set_xlim(*DESIGN)
ax2.set_xlabel("knee bend (deg, 0 = straight)")
ax2.set_ylabel("rod force (N), + = tension")
ax2.set_title(f"Rod force, L = {L:g} mm: sim knee torques (dots), servo limits (lines)", fontsize=10)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, loc="upper left", markerscale=3)
fig.tight_layout()
out = REPO / "cad" / "knee_linkage.png"
fig.savefig(out, dpi=110)
print(f"-> {out}")
