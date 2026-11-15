"""Compare gait schedules (duty x stride frequency) before training any of them.

For each schedule, over one stride at a steady walk:
  - support: how much of the stride the body stands on 3+ feet, 2 diagonal feet, or
    2 same-side feet (the shape that made 24a/25b prop a hind foot mid-swing)
  - balance margin: is the centre of mass (CoM) inside the shape the planted feet make?
    Distance to the nearest edge, + inside / - outside (a 2-foot line is always -)
  - at each front lift-off: margin and the prop a hind foot would need (torque balance)
  - swing time, stance travel, and how high the servos lift a hind foot in that time
    (leg_in_air.py: torso held up, bell path through the servo model)

Simplifications: CoM at the middle, feet at their home spots +/- half the stance travel,
no body sway. Good for comparing schedules, not for exact numbers.

  python tools/eval/schedule_table.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import leg_in_air  # noqa: E402  (servo-limited swing test)

FEET = ["LF", "RF", "LH", "RH"]
HOME = np.array([[0.118, 0.07], [0.118, -0.07], [-0.122, 0.07], [-0.122, -0.07]])  # x, y vs CoM (m)
SPEED = 0.14  # m/s: the fastest forward command (worst case for stance travel)
OFFSETS = {"current": [0.2, 0.7, 0.0, 0.5],  # touchdown phase per foot, FEET order (LH -> LF -> RH -> RF)
           "even": [0.25, 0.75, 0.0, 0.5]}


def margin(pts):
    """Signed distance (m) from the CoM (origin) to the support shape: + inside."""
    if len(pts) < 2:
        return -1.0
    if len(pts) == 2:  # a line: never statically stable
        a, b = pts
        t = np.clip(np.dot(-a, b - a) / np.dot(b - a, b - a), 0, 1)
        return -np.linalg.norm(a + t * (b - a))
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # counter-clockwise
    d = []
    for a, b in zip(pts, np.roll(pts, -1, 0)):
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)  # inward normal (CCW order)
        d.append(np.dot(-a, n))
    return min(d)


def analyse(duty, freq, offsets, n=400):
    off = np.array(offsets)
    travel = SPEED * duty / freq  # how far a planted foot moves back relative to the body
    counts = {"3+": 0, "diag": 0, "side": 0}
    margins, liftoff = [], []
    for k in range(n):
        ph = k / n
        prog = np.mod(ph - off, 1.0)
        down = prog < duty
        x = HOME[:, 0] + travel * (0.5 - prog / duty)  # +travel/2 at touchdown -> -travel/2 at lift-off
        pts = np.stack([x, HOME[:, 1]], 1)[down]
        if down.sum() >= 3:
            counts["3+"] += 1
        elif down.sum() == 2:
            i, j = np.where(down)[0]
            counts["side" if HOME[i, 1] * HOME[j, 1] > 0 else "diag"] += 1
        margins.append(margin(pts))
    # each front foot's scheduled lift-off: who's down, margin, prop needed from the lifting side's hind foot
    for f in (0, 1):
        ph = (off[f] + duty) % 1.0 + 1e-6
        prog = np.mod(ph - off, 1.0)
        down = prog < duty
        x = HOME[:, 0] + travel * (0.5 - prog / duty)
        pts = np.stack([x, HOME[:, 1]], 1)[down]
        names = [FEET[i] for i in np.where(down)[0]]
        prop = 0.0
        if down.sum() == 2 and all(HOME[i, 1] * HOME[1 - f, 1] > 0 for i in np.where(down)[0]):
            prop = 0.5  # same-side pair: lever 7 cm vs 14 cm -> half the weight (see 25b notes)
        liftoff.append((names, margin(pts), prop))
    m = np.array(margins)
    return dict(travel=travel, swing=(1 - duty) / freq,
                pct={k: 100 * v / n for k, v in counts.items()},
                stable=100 * np.mean(m >= 0.01), worst=m.min(), liftoff=liftoff)


def main():
    duties = [0.60, 0.65, 0.70, 0.75, 0.80]
    freqs = [1.25, 1.0, 0.8]
    lift_cache = {}
    print(f"walking {SPEED} m/s (fastest command). CoM at the middle, no sway. margin: + = CoM inside the feet's shape\n")
    for oname, offsets in OFFSETS.items():
        print(f"=== offsets '{oname}' (touchdown phase LF RF LH RH = {offsets}) ===")
        print(f"{'duty':>4} {'Hz':>5} | {'swing':>6} {'stance travel':>13} | {'3+ feet':>7} {'2 diag':>6} {'2 same-side':>11} |"
              f" {'stable (>=1cm)':>14} | {'at front lift-off: feet down, margin, hind prop needed':<52} | {'hind lift (servo)':>17}")
        for freq in freqs:
            for duty in duties:
                r = analyse(duty, freq, offsets)
                key = (round(r["swing"], 3), round(r["travel"], 4))
                if key not in lift_cache:
                    lift_cache[key] = leg_in_air.trial("bl", r["swing"], r["travel"])[0]
                lo = r["liftoff"][0]  # LF (RF is the mirror image)
                lo_s = f"{'+'.join(lo[0]):<11} {100 * lo[1]:+5.1f} cm  prop {100 * lo[2]:.0f}%"
                p = r["pct"]
                print(f"{duty:4.2f} {freq:5.2f} | {r['swing']:5.2f}s {'+/-' + format(50 * r['travel'], '.1f') + ' cm':>13} |"
                      f" {p['3+']:6.0f}% {p['diag']:5.0f}% {p['side']:10.0f}% | {r['stable']:13.0f}% |"
                      f" {lo_s:<52} | {lift_cache[key]:14.1f} cm")
            print()


if __name__ == "__main__":
    main()
