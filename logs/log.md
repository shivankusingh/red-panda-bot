# Daily log

Format: what I did · what I learned · mistakes fixed · decisions and why · next steps

## 2026-11-11
- **Did:** Set up the git repo, `.gitignore`, folder layout; pushed to GitHub.
- **Learned:** Commits store the author email permanently; `.gitignore` patterns like `logs/*/` skip subfolders but keep files.
- **Decisions:** Kept the proposed repo layout from CLAUDE.md.
- **Next:** ~~Week 1 simulator bake-off~~ (replaced, see below).

### Later the same day: rough robot model
- **Did:** Connected RunPod and added my SSH public key (no pod created, $0 spent). Installed MuJoCo 4.0.1 in `.venv`. Built `sim/red_panda.xml`: torso, 4 Pupper-style legs, 3-joint tail, 3-joint neck + head, 18 position motors, 1.8 kg. It stands still at 0.246 m.
- **Learned:**
  - The model is a tree of bodies; each `pos` is measured from its parent, so world positions add up down the chain (knee = 0.3 − 0.1235 = 0.1765 m up).
  - `size` (how big) is not `pos` (where). MuJoCo box sizes are half-sizes: 0.16 → 32 cm.
  - A joint's axis is the direction of its hinge pin, the line that stays still: wag/turn = z, lift/nod = y, tilt = x.
  - Two hinges moving together with a quarter-swing delay trace a circle, which is how 2 servos fake a shoulder.
  - Position motor: torque = kp × (target − angle). With kp = 10 and a 0.5 N·m load, the joint sags 0.05 rad (~3°). The sim's kp must match the real servo, not be as high as possible; the policy learns to handle the sag.
  - Rearing works by shifting the balance point over the hind feet before lifting the front feet. It stays up while the balance point is above the footprint (the area inside the contact points). The tail turns the 2-foot line into a tripod triangle.
  - Weight far from a joint costs a lot: load = weight × distance, and swing effort grows with distance², which is why the knee servo sits at the hip.
  - Sim beats hand math: tail-lift hold torque was 0.115 N·m (hand guess ~0.13), and the balance point was +0.22 cm (hand guess −0.1 cm).
- **Mistakes fixed:**
  - The legs jittered at ~18 rad/s forever: no damping, so the motor "spring" never settled. Added `damping="0.1"` to every joint, and now it settles to 0.
  - The MuJoCo viewer window won't open on this Mac (same error with 4.0.1 and 3.17). Using `mujoco-studio --gfx web` in the browser instead; firewall on, since it's open to the network.
- **Decisions and why:**
  - Skipped the simulator bake-off and picked Playground. Instead, run the Go1 example ~10 min on the first GPU session as a setup check.
  - Body 32 × 10 × 5 cm (red panda shape, room for electronics), hips ±12 cm, Pupper legs.
  - Tail 27.5 cm with 3 servos: wag + lift at the base (lift must be at the base for the tripod and counterweight) and a wag at 11 cm for an S-curve.
  - Neck with 3 servos (turn, nod, tilt), because the head tilt is cute. Held still in the sim for now.
  - Model mass 1.8 kg (parts estimate 1.64–1.74 kg), randomized ±10%. Not 1.9: a too-heavy sim gives fake torque fails and a policy tuned for the wrong robot.
  - Battery in the center of the torso. kp = 10 and damping = 0.1 are placeholders until the BAM servo rig.
- **Next:** ~~Add joint limits~~ (done, see below), then set up Playground on the Mac and read the Go1 example's environment and reward code.

### Joint limits
- **Did:** Added a range to all 18 joints, motors copy their joint's range (`inheritrange`), mirrored the right legs' hip-side axes, added armature. Tested: still stands still (0.248 m); motors asked for impossible angles stop at the limit; a 2 N·m shove gets 0.1° past a knee limit.
- **Learned:**
  - Joint limits keep the policy honest. RL will use any impossible motion the sim allows.
  - Right-hand rule: thumb along the axis, fingers curl = positive. Checked every joint in the sim (e.g. knee + = foot swings back, tail lift + = up, neck nod + = look down).
  - Ranges in the XML are in **degrees** (MuJoCo default), but joint angles and motor targets in the sim are **radians**.
  - Armature = the extra "heaviness" of a joint from the servo's fast motor spinning through 1:345 gears (≈ 0.024 kg·m² estimated, ~200× the lower leg itself). It only matters when speed changes, not when holding still or moving steadily.
- **Mistakes fixed:**
  - With the same hip-side axis on both sides, + was outward on left legs but inward on right legs, so I mirrored the right axes.
  - Knee limits were soft (10.7° past the limit under a 2 N·m shove) because the joints had no armature and felt as light as a 20 g leg piece. Armature 0.02 fixed it (0.1°).
  - I had been told moving the knee servo to the hip makes the leg ~2× easier to swing. Overstated: armature dominates. It still helps for balance, impacts, and protecting the servo and cables.
- **Decisions and why:**
  - Hind knees bend forward, front knees backward (animal-like). Backward hind knees would point at the ground and tail when the body tilts up for the tripod. All-backward (Go1-style) is a later sim test.
  - Right hip-side axes mirrored so + = outward on every leg, which avoids left/right sign bugs (flip those servo signs on the real robot).
  - Ranges (deg): front knees −150..0, hind knees 0..150, hips ±90, hip-side −20..35, tail lift −70..90, tail wags ±50, neck turn ±70, nod ±45, head tilt ±30. These are rough; the CAD sets the real ones.
  - Armature 0.02 is a placeholder (like kp and damping) until the BAM rig.
- **Next:** Install MuJoCo Playground on the Mac and read the Go1 example's environment and reward code.
