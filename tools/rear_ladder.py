"""Start poses for the reverse ladder (rr_run4a+): partway up into the tail tripod.

Rungs 0-6: the tripod itself (70 deg nose-up), then 10 deg less each, down to 10 deg.
Every one: hind hips 16 cm up, hind feet placed for the most balance margin
(tools/eval/tripod_static.py), tail tip lowered onto the floor, front legs at home, zero
speed. At 60-70 deg the balance point is inside the hind feet + tail triangle (it can
stand still there); at 50 deg and below (with the hips at 16 cm) it isn't, so the robot
has to push up or fall forward onto its front feet: that's the getting-up practice.

Bridge rungs (rr_run6, the gap between 10 deg and standing was the hardest step):
  7: 5 deg, hind feet half as far ahead as at 10 deg, tail planted (front feet up)
  8: the walking stance (home) with the tail lowered onto the floor, all four feet down
     (the "kickstand" pose earlier policies found by themselves)
Past the last rung a robot starts from its walking state (data/rear_starts.npz).

  python tools/rear_ladder.py      -> data/rear_ladder.npz (qpos [rungs, nq], tilt_deg)
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "eval"))
import tripod_static as ts  # noqa: E402

qs, tilts, names = [], [], []
dx10 = None
for th in [70, 60, 50, 40, 30, 20, 10]:
  best = None
  for dx in np.arange(-0.04, 0.181, 0.005):
    p = ts.pose(np.radians(th), 0.16, dx)
    if p is not None and (best is None or p[1] > best[0]):
      best = (p[1], p[0], dx)
  if best is None:
    raise SystemExit(f"no valid pose at {th} deg")
  margin, q, dx = best
  dx10 = dx
  qs.append(q)
  tilts.append(th)
  names.append(f"{th} deg, hind feet {100 * dx:+.1f} cm ahead, margin {100 * margin:+.1f} cm")

# Bridge 1: 5 deg, hind feet halfway between standing (0) and the 10 deg rung.
p = ts.pose(np.radians(5), 0.16, dx10 / 2)
if p is None:
  raise SystemExit("no valid 5 deg bridge pose")
qs.append(p[0])
tilts.append(5)
names.append(f"5 deg bridge, hind feet {100 * dx10 / 2:+.1f} cm ahead, tail planted")

# Bridge 2: walking stance with the tail lowered onto the floor.
q = ts.lower_tail(ts.HOME.copy())
if q is None:
  raise SystemExit("tail can't reach the floor from the walking stance")
qs.append(q)
tilts.append(0)
names.append(f"standing + tail planted (tail_lift {np.degrees(q[ts.QADR['tail_lift']]):.0f} deg)")

for i, n in enumerate(names):
  print(f"rung {i}: {n}")
out = REPO / "data" / "rear_ladder.npz"
np.savez_compressed(out, qpos=np.array(qs), tilt_deg=np.array(tilts))
print(f"-> {out}  (rung {len(qs)} = walking starts)")
