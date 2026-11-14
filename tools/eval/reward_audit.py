"""Score idealized swing-foot paths against every foot-height reward term together.
Per-episode costs in progress.csv per-term units (weight x sum over 1000 control steps)."""
import numpy as np
REST, LIFT, MAXH = 0.0097, 0.025, 0.035          # foot-ball centre at rest, arc lift, max_foot_height
DUTY, FREQ, DT = 0.70, 1.25, 0.02
T = (1 - DUTY) / FREQ                              # swing time 0.24 s
V = 0.12                                           # walking speed (m/s)
STRIDE = V / FREQ                                  # foot travel per swing, 9.6 cm
SWINGS = 4 * 20.0 * FREQ                           # 4 feet x 20 s x 1.25 = 100 swings per episode
s = np.linspace(0, 1, 2001); t = s * T
u = STRIDE / T * 2 * np.sin(np.pi * s) ** 2        # forward speed of the foot during swing (integrates to STRIDE)
arcs = {"sine": REST + LIFT * np.sin(np.pi * s), "bell": REST + LIFT * np.sin(np.pi * s) ** 2}
paths = {
    "follow sine arc": arcs["sine"],
    "follow bell arc": arcs["bell"],
    "skim (0.3 cm)": REST + 0.003 * np.sin(np.pi * s),
    "hover then drop": REST + LIFT * np.clip(np.minimum(s, 1 - s) / 0.08, 0, 1),   # up/down in ~20 ms each
}
def per_episode(cost_rate):   # integral over a swing / DT = expected sum over control steps; x swings per episode
    return np.trapezoid(cost_rate, t) / DT * SWINGS
def audit(arc, w_swing, w_soft, w_clear, soft_zone=0.01, label=""):
    print(f"\n{label}")
    print(f"  {'foot path':18s} {'swing_height':>13s} {'soft_landing':>13s} {'feet_clearance':>15s} {'feet_height':>12s} {'TOTAL':>8s}")
    best = None
    for name, h in paths.items():
        vz = np.gradient(h, t)
        c_swing = w_swing * per_episode((h - arcs[arc]) ** 2)
        c_soft = w_soft * per_episode(np.where(h - REST < soft_zone, np.minimum(vz, 0) ** 2, 0))
        c_clear = w_clear * per_episode(np.abs(h - MAXH) * np.sqrt(u))
        c_fh = -0.2 * SWINGS * (h.max() / MAXH - 1) ** 2
        tot = c_swing + c_soft + c_clear + c_fh
        best = name if best is None or tot > best[1] else best
        best = (name, tot) if not isinstance(best, tuple) or tot > best[1] else best
        print(f"  {name:18s} {c_swing:13.1f} {c_soft:13.1f} {c_clear:15.1f} {c_fh:12.1f} {tot:8.1f}")
    print(f"  -> cheapest: {best[0]}")
audit("sine", -930, 0, -2, label="RUN 16 (sine arc, no soft landing)")
audit("sine", -930, -15, -2, label="RUN 17 (sine arc + soft landing -15)")
audit("bell", -930, -15, -2, label="PROPOSED A (bell arc + soft landing -15, clearance as is)")
audit("bell", -930, -15, 0, label="PROPOSED B (bell arc + soft landing -15, feet_clearance off)")
audit("bell", -930, -5, 0, label="PROPOSED C (bell arc + soft landing -5, feet_clearance off)")
# landing speed each path has 20 ms before touchdown (what the soft-landing check sees last)
print("\nfoot speed 20 ms before touchdown:")
for name, h in paths.items():
    vz = np.gradient(h, t); i = np.searchsorted(t, T - 0.02)
    print(f"  {name:18s} {-vz[i]:.2f} m/s")
