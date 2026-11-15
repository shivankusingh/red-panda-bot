"""Hold the torso in the air; make one leg trace the bell swing path through the servo
model (kp 10, torque-speed limit 2.9 N*m / 4.7 rad/s). Can it physically follow?"""
import sys
REPO = __import__("pathlib").Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(REPO / "sim"))
import numpy as np, mujoco
import red_panda_walk as r
m = r.build_spec(r.default_config()).compile(); d = mujoco.MjData(m)
home = m.key_qpos[0].copy(); home[2] += 0.15                     # 15 cm up: feet never touch
dt = m.opt.timestep = 0.004
LIFT, STRIDE = 0.025, 0.096                                        # bell peak; foot travel at 0.12 m/s
def ik(leg, tgt_rel, q0):
    """hip/knee angles that put the foot at tgt_rel (torso frame, relative to the home foot)."""
    hq, kq = m.jnt_qposadr[m.joint(f"{leg}_hip").id], m.jnt_qposadr[m.joint(f"{leg}_knee").id]
    hd, kd = m.jnt_dofadr[m.joint(f"{leg}_hip").id], m.jnt_dofadr[m.joint(f"{leg}_knee").id]
    site = m.site(f"{leg}_foot").id
    d.qpos[:] = home; mujoco.mj_kinematics(m, d); base = d.site_xpos[site].copy()
    q = q0.copy()
    for _ in range(50):
        d.qpos[:] = home; d.qpos[hq], d.qpos[kq] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        err = (base + tgt_rel) - d.site_xpos[site]
        jacp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jacp, None, site)
        J = jacp[[0, 2]][:, [hd, kd]]
        q = q + np.linalg.solve(J.T @ J + 1e-6 * np.eye(2), J.T @ err[[0, 2]])
    return q
def trial(leg, swing_t, stride=STRIDE):
    """(peak lift cm, landing speed m/s, knee fold deg) for one swing of swing_t s moving the foot stride m forward."""
    hq, kq = m.jnt_qposadr[m.joint(f"{leg}_hip").id], m.jnt_qposadr[m.joint(f"{leg}_knee").id]
    hd, kd = m.jnt_dofadr[m.joint(f"{leg}_hip").id], m.jnt_dofadr[m.joint(f"{leg}_knee").id]
    ah, ak = m.actuator(f"{leg}_hip").id, m.actuator(f"{leg}_knee").id
    site = m.site(f"{leg}_foot").id
    n = int(round(swing_t / dt)); q0 = np.array([home[hq], home[kq]])
    path = []                                                       # joint targets along the bell, from behind to ahead of the hip
    q = q0.copy()
    for i in range(n + 1):
        s = i / n
        q = ik(leg, np.array([stride * (s - 0.5), 0, LIFT * np.sin(np.pi * s) ** 2]), q); path.append(q.copy())
    start = ik(leg, np.array([-stride / 2, 0, 0]), q0)
    mujoco.mj_resetData(m, d); d.qpos[:] = home; d.qpos[hq], d.qpos[kq] = start; d.ctrl[:] = m.key_ctrl[0]
    mujoco.mj_forward(m, d); z0 = d.site_xpos[site][2]; zs, vzs = [], []
    kp = m.actuator_gainprm[ah, 0]; stall = 2.9; w0 = 4.7
    for i in range(n + 10):
        tgt = path[min(i, n)]
        for a, j, t in [(ah, hd, tgt[0]), (ak, kd, tgt[1])]:
            qj = d.qpos[m.jnt_qposadr[m.dof_jntid[j]]]; w = d.qvel[j]
            tau = np.clip(kp * (t - qj), -stall * np.clip(1 + w / w0, 0, 1), stall * np.clip(1 - w / w0, 0, 1))
            d.ctrl[a] = qj + tau / kp
        d.qpos[:7] = home[:7]; d.qvel[:6] = 0                         # torso held still in the air
        mujoco.mj_step(m, d)
        zs.append(d.site_xpos[site][2] - z0)
        vzs.append(np.zeros(6)); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, site, vzs[-1], 0)
    zs = np.array(zs); vz = np.array([v[5] for v in vzs])
    td = np.argmax(np.arange(len(zs)) > n * 0.5) + np.argmax(zs[int(n * 0.5):] < 0.002)  # first return near start height
    kneespan = np.degrees(max(abs(p[1] - path[0][1]) for p in path))
    return 100 * zs.max(), -vz[max(td - 5, 0)], kneespan
if __name__ == "__main__":
    print(f"{'leg':4s} {'swing':>6s} {'asked peak':>11s} {'got peak':>9s} {'knee fold':>10s} {'foot speed 20 ms before return':>31s}")
    for leg, name in [("fl", "LF"), ("bl", "LH")]:
        for T in [0.24, 0.32]:
            peak, vland, kf = trial(leg, T)
            print(f"{name:4s} {T:5.2f}s {100*LIFT:9.1f}cm {peak:7.1f}cm {kf:8.0f} deg {vland:24.2f} m/s")
