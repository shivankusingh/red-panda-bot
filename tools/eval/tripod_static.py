"""Can the rough model hold a tail tripod, and what does it cost the servos?

1. Pose search: tilt the body nose-up by theta, put the hind thigh joints at height H,
   plant the hind feet d ahead of them (IK on hip + knee), lower the tail until its tip
   touches the floor (tail_lift only), leave the front legs, neck and head at home.
   Keep poses where every joint is inside its range and the torso clears the floor.
   Score: how far the centre of mass (CoM) is inside the triangle hind feet + tail tip
   (seen from above; + inside, cm).
2. Hold test: the best pose per tilt, in full physics with tail and torso floor contact
   on, every servo holding its pose angle (kp 10, +/-2.9 N*m like the walking model)
   for 3 s, at 1.80 kg and 1.98 kg. Reports drift, what touches the floor, and torque
   per servo vs rated (0.98 N*m) and stall (2.94 N*m).

Static only: no motion, no rearing up. A pose that fails here can't work dynamically
either; one that passes still has to be reached and balanced by a policy.

  python tools/eval/tripod_static.py
"""
import sys
from pathlib import Path
import numpy as np
import mujoco
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import red_panda_walk as r  # noqa: E402

RATED, STALL = 0.98, 2.94
TAIL_TIP = np.array([-0.165, 0.0, 0.0])  # end of the tail_2 capsule (radius 1.5 cm)
TAIL_R = 0.015


def build(mass_scale=1.0):
    cfg = r.default_config()
    r.use_flat(cfg)  # the default has terrain on; this check assumes the flat floor at z = 0
    m = r.build_spec(cfg).compile()
    touch = {m.body(b).id for b in ["torso", "tail_base", "tail_1", "tail_2"]}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in touch:
            m.geom_conaffinity[g] = 1  # the floor has contype 1
    for b in touch:  # MuJoCo also keeps a per-body summary (set at compile) to skip body pairs early
        m.body_conaffinity[b] |= 1
    m.body_mass[:] *= mass_scale
    m.body_inertia[:] *= mass_scale
    return m


m = build()
d = mujoco.MjData(m)
HOME = m.key_qpos[0].copy()
TORSO = m.body("torso").id
QADR = {j: m.jnt_qposadr[m.joint(j).id] for j in ["bl_hip", "bl_knee", "br_hip", "br_knee", "tail_lift"]}
DADR = {j: m.jnt_dofadr[m.joint(j).id] for j in QADR}


def place(q, theta, H):
    """Torso pitched nose-up by theta, hind thigh joints at height H above x = 0."""
    q[3:7] = [np.cos(-theta / 2), 0, np.sin(-theta / 2), 0]  # rotation about +y by -theta
    q[0:3] = 0
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    hip = d.xpos[m.body("bl_upper").id]
    q[0] = -hip[0]
    q[2] = H - hip[2]
    return q


def ik_foot(q, leg, target_xz):
    site = m.site(f"{leg}_foot").id
    for _ in range(60):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        err = target_xz - d.site_xpos[site][[0, 2]]
        if np.linalg.norm(err) < 1e-5:
            break
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, site)
        J = jacp[[0, 2]][:, [DADR[f"{leg}_hip"], DADR[f"{leg}_knee"]]]
        dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(2), J.T @ err)
        q[QADR[f"{leg}_hip"]] += dq[0]
        q[QADR[f"{leg}_knee"]] += dq[1]
    return q, np.linalg.norm(err)


def tail_tip(q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    b = m.body("tail_2").id
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ TAIL_TIP


def lower_tail(q):
    """tail_lift angle that puts the tail tip on the floor (None if it can't reach)."""
    lo, hi = m.jnt_range[m.joint("tail_lift").id]
    a = QADR["tail_lift"]
    q[a] = lo
    if tail_tip(q)[2] - TAIL_R > 0:
        return None
    for _ in range(50):  # bisection: higher angle = tail higher
        mid = (lo + hi) / 2
        q[a] = mid
        if tail_tip(q)[2] - TAIL_R > 0:
            hi = mid
        else:
            lo = mid
    q[a] = lo
    return q


def margin(pts, p):
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    out = []
    for a, b in zip(pts, np.roll(pts, -1, 0)):
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        out.append(np.dot(p - a, n))
    return min(out)


def in_range(q):
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            lo, hi = m.jnt_range[j]
            if not lo - 1e-6 <= q[m.jnt_qposadr[j]] <= hi + 1e-6:
                return False
    return True


def torso_clear(q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    R = d.xmat[TORSO].reshape(3, 3)
    corners = [d.xpos[TORSO] + R @ np.array([sx * 0.16, sy * 0.05, sz * 0.025])
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return min(c[2] for c in corners) > 0.003


def pose(theta, H, dx):
    q = place(HOME.copy(), theta, H)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    for leg in ["bl", "br"]:
        hip = d.xpos[m.body(f"{leg}_upper").id]
        q, err = ik_foot(q, leg, np.array([hip[0] + dx, 0.01]))
        if err > 1e-3:
            return None
    q = lower_tail(q)
    if q is None or not in_range(q) or not torso_clear(q):
        return None
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    pts = np.array([d.site_xpos[m.site("bl_foot").id][:2], d.site_xpos[m.site("br_foot").id][:2], tail_tip(q)[:2]])
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    com = d.subtree_com[TORSO][:2].copy()
    return q, margin(pts, com), pts, com


def hold(q, mass_scale, seconds=3.0):
    mm = build(mass_scale)
    dd = mujoco.MjData(mm)
    dd.qpos[:] = q
    for i in range(mm.nu):
        dd.ctrl[i] = q[mm.jnt_qposadr[mm.actuator_trnid[i, 0]]]
    mujoco.mj_forward(mm, dd)
    pitch0 = np.degrees(np.arctan2(-dd.xmat[TORSO][6], dd.xmat[TORSO][8]))
    z0 = dd.xpos[TORSO][2]
    n = int(seconds / mm.opt.timestep)
    forces = []
    for k in range(n):
        mujoco.mj_step(mm, dd)
        if k > n * 2 // 3:
            forces.append(dd.actuator_force.copy())
    pitch1 = np.degrees(np.arctan2(-dd.xmat[TORSO][6], dd.xmat[TORSO][8]))
    touching = set()
    for c in dd.contact[:dd.ncon]:
        for g in (c.geom1, c.geom2):
            name = mm.body(mm.geom_bodyid[g]).name
            if name != "world":
                touching.add(name)
    f = np.abs(np.array(forces)).mean(0)
    names = [mm.actuator(i).name for i in range(mm.nu)]
    return dict(drift_deg=pitch1 - pitch0, sag_cm=100 * (z0 - dd.xpos[TORSO][2]),
                touching=sorted(touching), torque=dict(zip(names, f)))


def kind(touching):
    """What the pose really is, from what touches the floor."""
    t = set(touching)
    if t & {"fl_lower", "fr_lower"}:
        return "front feet down"
    if t & {"torso", "tail_base", "tail_1"}:
        return "sitting (tail root down)"
    if t == {"bl_lower", "br_lower", "tail_2"}:
        return "TRIPOD"
    return "other: " + ",".join(sorted(t))


def main():
    tilts = range(30, 85, 10)
    heights = [0.10, 0.12, 0.14, 0.16]
    print("For each tilt and hind-hip height: hind feet placed for the most CoM margin, then held 3 s.")
    print("Torques: the worse of left/right, at 1.98 kg (heavy case); rated 0.98, stall 2.94 N*m.\n")
    print(f"{'tilt':>4} {'hip H':>5} | {'feet ahead':>10} {'margin':>7} | {'what it really is':<26} {'drift':>6} | "
          f"{'hind hip':>8} {'hind knee':>9} {'tail_lift':>9}")
    for th in tilts:
        for H in heights:
            best = None
            for dx in np.arange(-0.04, 0.181, 0.005):
                p = pose(np.radians(th), H, dx)
                if p is not None and (best is None or p[1] > best[1]):
                    best = (dx, p[1], p[0])
            if best is None:
                print(f"{th:3d}° {100*H:4.0f}cm | no valid pose (joint ranges / tail can't reach the floor / torso hits it)")
                continue
            dx, mg, q = best
            if mg < 0.01:
                print(f"{th:3d}° {100*H:4.0f}cm | {100*dx:+8.1f}cm {100*mg:+6.1f}cm | CoM not inside the triangle by 1 cm")
                continue
            h = hold(q, 1.1)
            t = h["torque"]
            print(f"{th:3d}° {100*H:4.0f}cm | {100*dx:+8.1f}cm {100*mg:+6.1f}cm | {kind(h['touching']):<26} {h['drift_deg']:+5.1f}° | "
                  f"{max(t['bl_hip'], t['br_hip']):8.2f} {max(t['bl_knee'], t['br_knee']):9.2f} {t['tail_lift']:9.2f}")


if __name__ == "__main__":
    main()
