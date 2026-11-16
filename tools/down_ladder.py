"""Start poses for the coming-down ladder (dn_run2): the rearing ladder in reverse.

Rung 0 = 10 deg (front feet just above the floor: only "catch yourself gently" to learn),
then 20, 30 ... 70 deg; past the last rung a robot starts from the real tripod states
(data/down_starts.npz). Same poses as tools/rear_ladder.py rungs 6..0 (tail planted,
hind hips 16 cm, zero speed).

  python tools/down_ladder.py      -> data/down_ladder.npz
"""
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
rear = np.load(REPO / "data" / "rear_ladder.npz")
pick = [i for i, t in enumerate(rear["tilt_deg"]) if t in (10, 20, 30, 40, 50, 60, 70)][::-1]
qpos, tilt = rear["qpos"][pick], rear["tilt_deg"][pick]
np.savez_compressed(REPO / "data" / "down_ladder.npz", qpos=qpos, tilt_deg=tilt)
print("rungs (deg):", list(tilt), "-> data/down_ladder.npz; past the last rung: real tripod states")
