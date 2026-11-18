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
- **Next:** ~~Install MuJoCo Playground on the Mac and read the Go1 example~~ (started, see below).

## 2026-11-12
### Playground + how RL works (reading the Go1 example)
- **Did:** Installed MuJoCo Playground 0.2.0 (with JAX, Brax, MJX 4.0.1) in `.venv`. Ran `Go1JoystickFlatTerrain` on the CPU (100 steps OK). Read `joystick.py`: observations (`_get_obs`, line 357) and actions (`step`, line 253).
- **Learned:**
  - The RL loop, every 0.02 s (50 Hz): sense → policy → motor targets → physics → reward. Like training my dog: observation = what it notices, action = what it does, reward = treat.
  - The policy is a formula: numbers in, numbers out. Go1: 48 in → layers of 512, 256, 128 neurons → 12 out, ~190,000 weights (~1 MB). My red panda: 57 in (3+3+3 + 15+15+15 + 3), 15 out.
  - Observation = body speed, spin (gyro), which way is down (projected gravity), joint angles and speeds relative to the home pose, last action, joystick command. `info` is a dictionary: `info["command"]` is the label, the 3 numbers are what's stored. `jp.hstack` glues the lists into one.
  - Sensor noise: the policy sees true value + a small random error (gyro: ±0.2); the reward uses the true values. Domain randomization is different: it changes the robot itself (mass, friction, kp) each episode.
  - Action: target = home pose + action × action_scale (0.5). Actions are −1..+1 (the output squash), so Go1 can only reach ±0.5 rad ≈ ±29° from home.
  - Two limits: the XML joint range is what the body can do (it must match the real robot); action_scale is how far the policy may reach from home (a training choice). The tighter one wins. My hind hips need ~70° for the tripod, which needs a bigger scale, a different home pose, or full-range mapping, not a different XML.
  - Joystick command = 3 numbers (forward speed, sideways speed, turn rate): the 3 body motions worth steering. The other 3 (up/down, roll, pitch) should stay calm and are penalized. Training picks random commands every few seconds. Tier 1 may only need 2 (walk + turn).
  - Neural network: each neuron = weighted sum of the layer before → squash (ReLU: negative → 0). Without the squash, all the layers collapse into one straight line. Weights live in grids (row = input, column = neuron).
  - Learning: start random, try actions with bell-curve jitter, and nudge every weight by step size × (better/worse) × (jitter) × (its input). Backprop sends blame backward from the outputs in one sweep; each weight's nudge = blame at its receiving neuron × the signal through it (like pre × post in STDP, but the post signal is backpropagated blame). Step size 3e-4, 8,192 robots, 200M steps.
  - It's fast because it's all matrix multiplication; a GPU does trillions per second. The policy runs on the Pi in well under 1 ms (to measure).
  - Go1 has a critic that sees 123 numbers (privileged state), which is asymmetric actor-critic. Also found `Go1Getup` (fall recovery) and `Go1Footstand` (hind-leg stand), useful references.
- **Mistakes fixed:** I was told the policy has "millions" of weights; it's ~190,000.
- **Decisions and why:** Not adding an SNN version of the policy: the servos dominate power use and the Pi is already fast enough, so the robot doesn't need it.
- **Next:** ~~Part 3: the Go1 reward terms, a couple at a time~~ (groups A–D done, see below); then decide the red panda's rewards.

### Go1 rewards, part 3 (groups A–D: 12 of 16 terms)
- **Did:** Read `_get_reward` in `joystick.py` (weights at line 53, sum at 278, terms at 433–508). Traced where the network and PPO live in Brax. Ran two small sim tests on `sim/red_panda.xml` (front-left hip): flipping commands vs. smooth commands.
- **Learned:**
  - **Network vs. PPO vs. reward.** The network is the formula (Brax `MLP`, `brax/training/networks.py:137`); no file holds weight values, they start random (`lecun_uniform`) and live in the checkpoint. PPO (Schulman 2017) is the fixed rule that edits weights: reward → advantage (better/worse than the critic expected) → loss (`losses.py:245`) → `jax.grad` does backprop automatically → Adam nudges weights. The reward is the only part I write; it decides *what* gets learned. Same PPO for walking and the tripod, different reward.
  - **How the reward is added up:** 16 raw numbers × weights, summed, × dt (0.02), floored at 0. The floor stops "fall on purpose to end the penalties", but every step below 0 looks the same (0), so early training needs always-on treats (tracking, pose) to show differences.
  - **A. Tracking** (+1.0 speed, +0.5 turn): `exp(−error²/sigma)`, a hill from 1 down to 0. error² = (wanted − actual)² for forward + sideways. Crab-walking sideways at the commanded speed scores 0.0003. Sigma = how forgiving the grader is; reward = 0.37 when error² = sigma. Too small → random policy scores ~0 everywhere → nothing to learn. A slower robot needs a smaller sigma.
  - **B. Body calm:** `lin_vel_z` (−0.5, bouncing), `ang_vel_xy` (−0.05, rocking; small because trots naturally rock), `orientation` (−5.0, leaning; = x² + y² of the body's up arrow = sin²(tilt): 30° costs −1.25, more than all of tracking). Roll tips the up arrow toward **y**, pitch toward **x**: the arrow leans perpendicular to the pin.
  - **Weights are prices, not fences.** Joint limits and termination are fences. Lowering a penalty gives permission, not style; natural-looking motion comes from the personality layer and energy terms, not from cheaper wobble.
  - **C. Gentle on motors:** `torques` (−0.0002, total pushing, even when still), `energy` (−0.001, push × speed = watts, 0 when still), `action_rate` (−0.01, (change in command)²). Weights carry hidden units: red panda torques are ~10× smaller than Go1's, so the torque weight needs ~10× bigger. Heat ∝ torque², so `torques` is what matters for the tripod hold.
  - **action_rate test:** a hip told to flip ±8.6° every step only wiggled 0.4–0.7°, but the motor pushed ±1.5 N·m back and forth (half of max). Smooth ramp + zigzag on top: same leg path (23.68° vs 23.41°) but action_rate 0.035 vs 1.295. The leg is slow, so one 0.02 s step barely moves it; zigzags cancel out and the servo pays for nothing. Too big a weight makes "don't move" the best strategy.
  - **My idea: penalize sign flips of the change** = (change now − change before)². It doesn't punish fast-but-smooth moves, only sharp reversals. Go1 passes `last_last_act` but doesn't use it. Usually paired with a small action_rate.
  - **D. Posture and safety:** `pose` (+0.5, exp hill around the home keyframe; knees count 0.1× because walking bends them), `stand_still` (−1.0, Σ|q − home|, only when the joystick is centered), `dof_pos_limits` (−1.0, past the soft limit = range × 0.95), `termination` (−1.0, up arrow z < 0 = tilted past 90°). Falling hurts because the episode ends and all future reward is lost (falling at step 200 of 1000 loses ~24 vs. 0.02 from the term).
  - `lowers, uppers = jnt_range[1:].T`: the XML `range` loaded into a table (radians), skip row 0 (the torso's free joint), transpose to split lows and highs.
- **Mistakes fixed:**
  - The policy outputs **24** numbers for Go1 (mean + spread per joint), not 12; on the robot only tanh(mean) is used. The red panda will output 30. Weight count: 192,408.
  - The squash is **swish** (x × sigmoid(x), a smooth ReLU), not ReLU: Brax's PPO default, and Go1 doesn't override it.
  - Soft limits ×0.95 shrink toward **zero**, not the middle: no warning zone on ranges that end at 0 (all my knees at the straight-leg end) and the soft limit lands outside ranges that don't include 0 (Go1's knee).
- **Decisions and why:**
  - Plan A: finish all 16 terms before writing the red panda env, because the feet terms have size-dependent numbers (`max_foot_height` = 10 cm is 40% of my leg).
  - Rewards in two passes: (1) minimal port, only fixing what's broken for my size (torque weight, foot height, sigma, command speeds), as a baseline; (2) my own redesign, each change tested against pass 1.
- **Red panda notes (decide later):**
  - Tripod: `orientation` and `pose` both pull the body back to level/home, so it needs its own target tilt and tripod pose. Tracking terms drop out.
  - Home pose: the model has no keyframe and stands with straight knees (0°, at the limit); pick a crouch, check torques in sim.
  - Soft limits: shrink toward the middle of the range (hind knee 0..150° → 3.75..146.25°).
  - Fall rules: "z < 0" would end the free stand at 92° tilt; "anything but feet touches ground" would end the tripod. Use torso/head contact or height.
  - Holding a skill must be worth more than falling out of it.
  - Candidates: second-difference smoothness term, torque² heat term, action filter, ramping action_rate up during training.
  - Top speed, sigma, and command ranges come from the sim (servo ~5 rad/s vs Go1 ~30, legs 24 cm vs 43 cm). Time the Go1 run on the first GPU session and note GPU + cost.
- **Next:** ~~Group E (feet)~~ (done, see below), then prep the model for training (home keyframe, IMU site + sensors, foot contacts).

### Go1 rewards, part 3 (group E: feet, 16 of 16 done)
- **Did:** Read the feet terms (`joystick.py:512-548`) and the per-foot bookkeeping in `step` (lines 258-267, 294-296).
- **Learned:**
  - `feet_slip` (−0.1): Σ foot speed² × contact, only while walking. A foot on the ground shouldn't slide (skating fails on real floors); feet in the air are ignored.
  - `feet_clearance` (−2.0): Σ |foot height − 0.1 m| × √(foot speed). The speed works like a dimmer switch: planted feet cost nothing, moving feet should be at the target height. Dragging low (0.02 m at 1 m/s) costs −0.16; too high costs too.
  - Foot height is measured at the **center** of the foot ball, so a planted Go1 foot reads ~0.023 m (its radius), and the 0.1 m target is really ~7.7 cm of lift.
  - Bookkeeping per foot: `feet_air_time` = stopwatch (runs in the air, resets on the ground), `swing_peak` = high-water mark of this flight, `first_contact` = 1 only on the landing step (the "ding" that triggers scoring). `contact_filt` = touching now or last step, so a flickering landing isn't counted twice.
  - `feet_air_time` (+0.1): Σ (air time − 0.1 s) at landing, only while walking. 0.3 s step → +0.02; 0.06 s shuffle → −0.004. Discourages tiny tapping steps; no upper cap (other terms keep it in check).
  - `feet_height` (−0.2): Σ (peak / 0.1 − 1)² at landing. Peak 0.07 m → −0.018. Clearance checks the whole path every step; feet_height checks only the top of each step's arc.
- **Red panda notes (decide later):**
  - Target foot height: 0.1 m is 42% of my 24 cm leg (Go1: 23%). Scaled: ~4.5 cm of lift + 1 cm foot radius ≈ 5.5 cm; maybe lower because the servos are slow.
  - Air-time threshold (0.1 s): shorter legs say ~0.075 s, slow servos say ≥ 0.14 s just to swing the hip 40°. Probably longer; log real swing times in the first run and set it just below a typical good step.
  - The model needs foot sites (feet are currently just the 1 cm capsule ends) + foot velocity sensors + foot–floor contact sensors.
  - Energy weight also needs rescaling (watts much smaller than Go1's).
- **Next:** ~~Prep the model for training~~ (done, see below). Rudin 2021 moved to training-config time.

### Model prep for training
- **Did:** Added to `sim/red_panda.xml`: a `home` keyframe, an IMU site + 9 sensors, foot spheres + sites + 12 foot sensors, and feet-only collision. Wrote `tools/check_home_pose.py` (settles the home pose, prints height, tilt, knee directions, foot offsets, torques). Switched to a hybrid way of working (CLAUDE.md updated).
- **Learned:**
  - **Home pose:** knee bend sweep in sim (torque on each knee, standing height): 0° → 0 N·m, 24.8 cm; 40° → 0.18, 23.2; 60° → 0.27, 21.4; **80° → 0.35, 18.9**; 100° → 0.41, 15.9. Hand check at 60°: 4.4 N × 6 cm = 0.26 N·m ✓. Front knees carry a bit more (balance point is 0.22 cm forward).
  - **Home, leash, and foot lift are linked:** target = home + action × 0.5 rad. From 60° the knee can only bend enough to lift the foot 3.6 cm; from 80°, 4.4 cm; 6 cm needs a leash ≥ 0.67 rad from 80°. Max reach forward grows with the crouch (60° → 11.9 cm, 80° → 15.3 cm).
  - **qpos vs. ctrl:** qpos = actual pose (25 numbers, body-tree order: torso xyz, quaternion, legs, neck, tail); ctrl = motor targets (18, actuator order: legs, tail, neck). Tail and neck are **swapped** between them.
  - **IMU:** gyro, up arrow, and gravity read the same anywhere on the torso; speed and accelerometer don't (merry-go-round edge). Put it where the real chip will be (center for now). The accelerometer reads +9.8 **up** when still: it measures every push except gravity (the floor pushing up); 0 in free fall.
  - **Policy inputs must exist on the real robot; rewards don't.** Go1's policy reads body speed (`local_linvel`), which an IMU can't measure. Give it to the critic only (asymmetric actor-critic), or estimate it.
  - **Foot height reads 0.97 cm when planted** (1 cm foot radius, sunk 0.3 mm). A 4 cm gap under the foot = a 5 cm target.
  - **Collision:** two geoms touch if (A.contype & B.conaffinity) or (B.contype & A.conaffinity). Default 0/0, floor contype 1, feet conaffinity 1 = only feet touch the floor (like Go1's feet-only XML). Different policies can use different collision rules.
  - Tail lift in the home pose: lowest tail point 3.7 cm at −30°, **hits the floor at −40°**.
- **Mistakes fixed:**
  - My first home recommendation (60°) only allowed 3.6 cm of foot lift; caught by asking "are you sure?" and checking in sim.
  - The "15 outputs" in the earlier log disagreed with CLAUDE.md (tail is animation-driven while walking): resolved below.
- **Decisions and why:**
  - **Home: 80° knee bend** (hip 38.3°, feet under hips, 18.9 cm). Lower body like a real red panda, more lift (4.4 cm) and reach. Cost: ~1.7× the copper heat of 60°, still small vs. stall; test on the servo rig.
  - **Walking policy: 12 outputs (legs only), and it sees the tail angles and speeds** (A+). The tail is animated while walking and randomized in training, so the policy learns to balance against any tail motion.
  - **Feet-only collision for walking.** The walking env keeps the tail lift above ~−25° (animation and randomization). Episode ends if upside down **or torso center below 10 cm** (half the home height; even a 100° crouch stands at 15.9 cm). Switch to torso/head contact for terrain or fall recovery; the tripod env turns tail contact back on.
  - **Hybrid way of working:** run first, go deep only where mistakes are silent or costly; plumbing in build mode.
  - **Fall recovery is a separate policy** (starts fallen, rewards getting upright; walking needs "fall = episode over"). Matches Go1Getup and Lee 2019. **Terrain isn't a policy**, it's a training condition for the walking policy: flat first (so failures point at the rewards), then ~1 cm bumps for real floors, real terrain only if wanted. Terrain means switching the 10 cm height cutoff to torso contact.
- **Red panda notes (decide later):**
  - Policy inputs drop body speed: 54 → 51 (critic still gets it).
  - Knee leash and foot-lift target (4.4 cm max with a 0.5 leash at 80°; 6 cm needs a bigger leash).
  - Style for reward pass 2: long forward reach and torso sway like a real red panda (Walk These Ways, gait clock); could measure ratios from videos.
  - Watch for front/hind leg collisions at long strides (no leg-leg collision in sim).
  - Real robot: the home pose must match exactly in the robot code; servos need zero calibration.
  - Look up the STS3215 rated torque; heat test on the rig.
  - **Policy doesn't see body speed (51 inputs).** No sensor measures it; Go1 uses an estimator (planted foot + joint speeds + IMU). Joint speeds ≠ body speed (hanging on a stand: big joint speeds, body speed 0). Reward and critic still use it. **Test it:** first GPU session, train blind vs. sees-speed (one config flag) and compare speed-tracking error. Big gap → build the estimator.
- **Next:** ~~Port the Go1 env to the red panda~~ (done, see below), then a first training run.

### Walking env (Go1 port) + reward pass 1 size fixes
- **Did:** Wrote `sim/red_panda_walk.py` (Go1's `joystick.py` ported; every change marked `RED PANDA:`) and `tools/smoke_test_walk.py`. Smoke test passes on the Mac (~16 s): 57 inputs / 12 outputs, holds home at 18.9 cm with tail and head moving, height cutoff works, 60 inputs with the body-speed switch on. Looked up the STS3215 C018 (retailer listings, not Feetech's PDF): no-load 0.222 s/60° = **4.7 rad/s**, rated **0.98 N·m**, stall 2.94 N·m. Home knee torque 0.345 N·m = 35% of rated ✓.
- **Learned:**
  - An "environment" is the file that runs the training game: `reset` (start pose, random command) and `step` (actions → motors → physics → observation → reward → done?). PPO doesn't change between robots; the env does.
  - The joystick **command** (forward, sideways, turn) is what's asked of the whole robot; motor targets are the policy's job. `command_config.a` = the biggest command used in training (each new one is random in ±a, about every 5 s), so one policy learns straight, curves, turning, backward, and stopping. It's the max it's *trained* for, not what it can physically do; Go1's 1.5 m/s is a training choice (the real Go1 goes faster).
  - Servo max speed ≠ robot speed: the robot's speed = stride × steps per second, decided by the gait the policy learns.
  - Height and reach are separate: `max_foot_height` sets how high the foot lifts; stride length is the policy's choice, limited by the hip leash. At leash 0.5 the front foot can land 8 cm ahead / 12 cm behind (20 cm stride), so the leash isn't the limit.
  - Two "max lift" numbers: foot kept **under the hip** (4.5 cm at leash 0.5, the one that matters mid-step) vs. foot anywhere (7.9 cm, includes the whole leg swinging forward). Go1's target uses 84% of its under-hip max (7.7 of 9.1 cm).
  - Weight × amount = "bill". A lighter robot has smaller amounts, so the weight goes **up** to keep the same pressure (heavier → down).
  - Early-training check (small random jitter, 300 steps): reward above 0 on 23% of steps vs. Go1's 9%, so it should be at least as learnable. But energy was 2.4× Go1's share (light legs flick fast; placeholder kp 10 and no servo speed limit add to it).
- **Mistakes fixed:**
  - Turn command 1.0 rad/s was a gut guess; calculated by scaling Go1 by foot distance from center (0.14 vs 0.23 m) it's 0.5.
  - "Torques ~10× smaller" was a guess; measured standing torque cost: Go1 45.5 vs ours 2.6 = **17×**.
  - First lift sweep let the foot go anywhere (7.9 cm); redid it with the foot under the hip (4.5 cm).
  - Height-cutoff test teleported the torso to 8 cm; the feet ended up inside the floor and launched it back up. Test the rule directly instead.
  - The log's rough "~5.5 cm foot target" scaled by leg length alone; at leash 0.5 it would be 100% of max lift.
- **Decisions and why:**
  - **Neck animated like the tail:** random slow "looking around" (turn ≤30°, nod ≤20°, tilt ≤15°, one look every 2–10 s), switchable; the policy sees neck angles + speeds (57 inputs). Head + neck = 0.24 kg (13%), so it affects balance. CLAUDE.md updated.
  - **Leg-only rewards:** `pose`, `stand_still`, `torques`, `energy`, `dof_pos_limits` count only the 12 leg joints (the policy doesn't control the tail or neck; their random motion would be noise).
  - **Commands:** forward 0.4 m/s (servo-speed estimate 0.3–0.5), sideways 0.2 (Go1's 53% ratio; kept for small shuffles, set to 0 if run 1 struggles), turn 0.5 rad/s.
  - **Two sigmas:** walk 0.016 (speeds 4× smaller → 0.25/16), turn 0.043 (2.4× → 0.25/5.8). Same grading as Go1: standing still at 0.25 m/s scores 0.018, like Go1 at 1.0 m/s.
  - **Foot target 4.0 cm** (3 cm lift + 0.97 radius), leash stays 0.5: low and long like a red panda, quick for slow servos.
  - **torques −0.0035** (×17), **energy −0.007** (×7, not ×17: an energy penalty that's too high early on teaches "stand still" and wastes the run; too low only costs some efficiency).
  - Soft-limit quirk kept for walking: the leash keeps the knees at 51–109° of bend, far from either end. Revisit for the tripod.
- **Red panda notes (decide later):**
  - Long reach (pass 2): raise the `feet_air_time` threshold (0.1 s, still Go1's) for fewer, longer steps, or a gait clock.
  - Tail and neck sway ranges are guesses; replace with video ratios or the personality animation.
  - After run 1: measure real top speeds (command 0.1 → 0.5 m/s, see where tracking breaks) and compare each `reward/*` term's share with Go1's run.
- **Next:** ~~First GPU session~~ (done, see below).

### First GPU session: the red panda walks (in sim)
- **Did:** Wrote `tools/train_walk.py` (Brax PPO with Playground's Go1 settings, both robots; saves `progress.csv`, `params.pkl`, `config.json` to `runs/<name>/`), `tools/pod_setup.sh` (installs pinned versions on the pod), and `tools/watch_policy.py` (runs a trained policy on the Mac with a scripted joystick, prints commanded vs. measured speed, saves a GIF). Rented an RTX 4090 on RunPod (Secure Cloud, EU-RO-1, $0.89/h) with a 20 GB network volume for the software and results. Ran: Go1 check → red panda run 1 → sees-speed → part of a full Go1 run.
- **Results:**

  | Run | Steps | Train time | Final reward | Episode length | Walk / turn tracking |
  |---|---|---|---|---|---|
  | Go1 check | 23M (stopped) | 26 min | 18.7 | 1000 | (setup check passed) |
  | Go1 full | 92M (stopped) | 62 min | 26.4, climbing | 991 | |
  | **Red panda run 1** (blind, 57 inputs) | 206M | **7.9 min** | 31.8 (best 32.3) | 986 | 0.895 / 0.808 |
  | Red panda sees-speed (60 inputs) | 206M | 7.6 min | 31.7 | 985 | 0.901 / 0.796 |

  Scripted joystick test on the Mac (run 1, measured after speeding up): forward 0.30 → **0.30**, curve 0.20 + 0.40 → **0.21 + 0.40**, stop → 0.01, backward −0.20 → **−0.19**, sideways 0.15 → **0.15**, turn −0.50 → **−0.48**. Never fell. Video: `runs/rp_run1/walk.gif` (not committed).
- **Learned:**
  - The red panda learned almost everything in the first 23M steps (3 min): length 37 → 1000, walk tracking 0.002 → 0.82. Then slow gains, flat around reward 32.
  - It's not the "stand still" trap: standing still scores ~0.02 tracking with our sigma; it scored 0.82–0.90.
  - **Blind vs. sees-speed: no difference** (0.895 vs 0.901 walk, 0.808 vs 0.796 turn; one run each, within noise). Decision A (no body speed) holds in sim.
  - Red panda trains ~14× faster than Go1 here. Go1's XML uses `iterations="1"` (a deliberate speed trick: 1 round of contact-force solving); MuJoCo 4.0.1's Warp backend warns about it every step for every robot ("solver iterations limit reached"), flooding the log (114 MB) and likely slowing Go1. The red panda uses MuJoCo's default (100 rounds max), so it doesn't hit the limit. Not a broken setup: a version mismatch between Playground 0.2.0 and MuJoCo 4.0.1.
  - Installing to network storage is slow (~13 GB, thousands of small files, ~20 min the first time); next sessions reuse it.
  - Cost: 2.46 h of pod time ≈ **$2.19**, plus the network volume's monthly storage fee.
- **Mistakes fixed:**
  - Brax 0.14.2 (newest) calls `jax.device_put_replicated`, which JAX removed in 0.10: added JAX's suggested replacement at the top of `train_walk.py`.
  - The pod can't stop itself: its built-in RunPod key gets "Unauthorized". Someone has to stop it (phone or Claude).
  - `pkill -f run_all.sh` also matched the SSH command that contained that text and killed the session. Kill by PID instead.
  - Copying a whole run folder pulled 85 MB of checkpoints and the 114 MB log; copy only `params.pkl`, `progress.csv`, `config.json`.
  - "Red panda won't hit the warning" was a guess; the new script filtered those lines, so it's backed by the speed difference, not the log. Confirm next time.
- **Decisions and why:**
  - Ran the red panda before the full Go1 run (order "B"): Go1 had already passed the setup check at 23M steps (reward 0.001 → 18.7, never falls), and the full 200M would have taken ~3 h.
  - Keep the policy blind to body speed (no estimator for now).
- **Red panda notes (decide later):**
  - Occasional falls in test episodes (length 985–993, not 1000): look at which situations.
  - The sim is still too easy: no servo speed limit or model (BAM), no action delay, no domain randomization (mass ±10%, friction). Add before trusting it.
  - Style is generic, not red panda (reward pass 2: reach, sway).
  - Measure the real top speed: command 0.1 → 0.5 m/s, see where tracking breaks.
  - Silence Go1's solver warning at the source (`warn_overflow`) and test it on the GPU; confirm the red panda run without the filter.
  - Commit a cleaned run-order script (the pod's `run_all.sh` / `run_b.sh` aren't in the repo).
- **Next:** ~~Watch the GIF and plan run 2~~ (runs 2-5, see below).

### Runs 2-5: from a robot trot to a red panda walk
- **Did:** Four more ~7-minute GPU runs (one fresh RTX 4090 pod each: stopped pods couldn't restart because their GPUs got taken). Added a measured footfall chart to `tools/watch_policy.py`, made it rebuild each run's env from its own `config.json`, and kept old runs replayable (new settings fall back to the old values). Wrote my red panda walk spec and had it implemented as a stride clock + footfall-schedule reward (run 5).
- **What I saw:** Run 1 walks perfectly by the numbers but looks like a spider/robot: quick tiny steps (2.6 per second, 11 cm stride, 2 cm lift), using a third of the leg range, and the tail tip whipped +/-31 deg.
- **Results** (forward walking; gait measured in sim on the Mac):

  | Run | Change (cumulative) | Steps/s | Stride | Hip swing (front/hind) | Footfall |
  |---|---|---|---|---|---|
  | 1 | Go1 rewards + size fixes | 2.6 | 11.4 cm | 19 / 19 deg | trot (diagonal pairs) |
  | 2 | A: calmer tail (wag 0-15, tip 0-10 deg) + B: air-time threshold 0.1 -> 0.25 s | 2.6 | 11.4 cm | 17 / 17 | trot |
  | 3 | C: pose hip weight 1.0 -> 0.2 | 2.25 | 13.3 cm | 20 / 20 | trot |
  | 4 | B': feet_air_time weight 0.1 -> 2.0 | 1.75-1.9 | 16 cm @0.3 m/s | 22 / 26 | trot |
  | 5 | My walk spec (stride clock, footfall schedule, body height, joint accel, synced tail/head) | **1.25** | **16.8 cm @0.2 m/s** | **32 / 37** | **lateral-sequence walk** |

  Run 5 footfall (robot vs. my spec, 1 char = 5% of a stride):
  ```
        robot                  spec
  LH  █████████████·····██   ██████████████······
  LF  ·····████████████···   ····██████████████··
  RH  ███······███████████   ████······██████████
  RF  ████████······██████   ████████······██████
  ```
  All runs track the joystick and essentially never fall. Run 5 matched the footfall chart 89% of the time (random: 40%) and kept walk tracking at 0.91.
- **Learned:**
  - A reward term = what it measures x its weight (the "price"). Run 2 changed what `feet_air_time` measured but left its price at 0.1: the term was -1 per episode next to `pose` +477 and tracking +1,329, so nothing changed. Make sure a term is big enough to compete before judging it.
  - Removing a brake (C: pose no longer fines a +/-25 deg hip swing ~135 points) helped a little; adding a reason (B': pay for long air time) helped a lot. Took both.
  - Reward tweaks can't turn a trot into a red panda walk. Real red pandas walk with a lateral-sequence walk (Plocek & Dunham 2023, 132 filmed strides; footfall timing like other ground-dwelling carnivorans). RL finds the trot because it's easiest; you have to tell it the footfall order (stride clock + schedule reward, the Walk These Ways idea).
  - Speed = stride x strides per second. At 1.25 strides/s, a moderate 16 cm stride gives 0.2 m/s; 0.4 m/s would need 32 cm (the legs max out ~20 cm). So run 5's commands are 0.2 / 0.1 / 0.5.
  - The run 5 head "shaking" isn't randomness (the random nod was off): my scripted nod (6 deg, 2.5 times/s) + the springy placeholder neck servo overshooting (actual 11.6 deg) + the torso pitching 5.6 deg each step = 16 deg of head tilt, 2.7 times per second. Fix: drop the nod and stabilize the head against torso tilt (from the IMU), like real animals.
  - Spine idea (measured on a split-torso scratch model, front half 0.94 kg held in the air): an up/down arching spine needs 1.14 N·m (116% of rated, fails long holds); a side-to-side spine needs 0.00 N·m (vertical axis, gravity doesn't load it). Side-bend is the easy and the walking-relevant one; the hard part would be the mechanical hinge.
- **Mistakes fixed:**
  - Viewer replayed old runs with today's settings; now it rebuilds each run's env from its `config.json`.
  - Footfall chart thought run 4 had a stride clock (every run's info has `gait_phase` now); use the env's on/off flag.
  - My scripted speed test still asks 0.3 m/s, above run 5's 0.2 max (it reached 0.26). Update the script for run 5-style runs.
  - zsh doesn't split `$VAR` into words like bash; write ssh/scp options out in full.
- **Decisions and why:**
  - Walk style comes from a footfall schedule, not reward-weight tweaks. Spec: lateral-sequence walk LH -> LF -> RH -> RF, touchdowns 0 / 0.2 / 0.5 / 0.7, duty 0.70, 1.25 strides/s, low steps (~15% of leg = 3.6 cm lift), low body, narrow stance, head low with small nod, tail level swaying opposite the body's roll.
  - Kept from runs 2-4: calmer tail ranges, pose hip weight 0.2. Turned off: feet_air_time (the clock sets timing; its 0.25 s threshold would fine the spec's 0.24 s swings).
  - Spine: try a side-to-side spine in sim first (like the tail experiment); hardware only if it clearly looks better.
- **Red panda notes (decide later):**
  - Hind feet barely lift in run 5 (0.8-1.7 cm vs front 2.4-3.4; hind knee swings only 12 deg): risk of scuffing on real floors.
  - Head: drop the nod + stabilize against torso pitch; torso choreography (target roll/yaw/pitch that follows the stride clock); "head end carried low" body pitch; tail droop needs a pitch joint at the tip (doesn't exist).
  - Realism still missing: servo model (BAM), delay, domain randomization, pushes.
  - Cost: runs 2-5 about $0.57 (4 pods x ~9 min); today total ~$2.76. One stopped pod + the 20 GB volume remain.
- **Next:** ~~Run 6: head stabilization + torso choreography~~ (done, see below).

### Run 6: steady head, waddling body
- **Did:** Run 6 = run 5 + head stabilization (no scripted nod) + torso choreography (lean toward the supporting feet, head end low) + tail counterbalancing the lean. Restarted the run 5 pod (its GPU was free this time): 6.5 min, ~$0.13. The viewer now keeps its scripted commands inside each run's trained range.
- **Results** (walking forward at 0.2 m/s, measured in sim on the Mac):

  | | Run 5 | **Run 6** |
  |---|---|---|
  | Head tilt swing in the world | 16.0 deg | **2.4 deg** |
  | Head bob | 3.3 cm | **1.0 cm** |
  | Torso lean (target +/-3 deg) | no target (2.1 deg swing) | **+/-3.1 deg, correlation 0.99 with the target** |
  | Torso pitch (target 3 deg nose-down) | 0.1 deg | **3.0 deg** |
  | Front / hind duty (spec 0.70) | 0.68 / 0.75 | 0.57 / 0.75 |
  | Hind foot lift | 0.8-1.7 cm | **0.6-0.7 cm** (worse) |
  | Speed test | ok | ok (0.20 -> 0.19, turn -0.50 -> -0.48), never fell |

  Footfall order is still LH -> LF -> RH -> RF.
- **Learned:**
  - The head shake was mine, not randomness: a scripted nod plus a springy placeholder neck servo plus torso pitch. Head stabilization = neck angle = (where the head should point) - (torso tilt), with torso tilt from the gravity direction in the body frame, exactly what the real IMU gives: g_body = (sin p, -sin r cos p, -cos r cos p).
  - Torso choreography = Go1's orientation cost around a target instead of flat (identical to Go1's when the target is 0), plus a `body_sway` exp reward. Lean target: roll toward the side whose feet are down (left pair around phase 0.45, right pair around 0.95), 3 deg.
  - A rigid torso can't twist (yaw) each stride without actually turning: +/-2 deg would read as +/-0.27 rad/s of turning and fight turn tracking. Twist needs the side-bend spine.
  - Sign conventions checked in sim before training: + pitch = nose down, + roll = right side down; tilting the torso 10 deg nose-down makes the neck target 8 - 10 = -2 deg.
  - Rewards trade off: leaning cost some footfall timing (front feet down 57% instead of 70%).
- **Mistakes fixed:** flipped the roll sign in my scratch check script (it already matched the env's convention); fixed a broken f-string in the same script.
- **Decisions and why:** lean 3 deg toward the support side (weight shift = the waddle), head end 3 deg low, body_sway weight 0.5 with sigma (2 deg)^2, head held 8 deg down in the world, tail sign flipped to counter the new lean.
- **Red panda notes (decide later):**
  - Hind feet skim the floor (0.6-0.7 cm lift): give them their own clearance target or weight. Biggest remaining problem.
  - Front duty drifted to 0.57: maybe raise gait_schedule 1.0 -> ~1.5.
  - The neck servo overshoot comes from placeholder gains (kp 10); BAM will replace them.
  - Cost so far: ~$2.89 total (session 1 $2.19 + runs 2-6 ~$0.70). Stopped pods: `hf9abldp5ohuh8` (run 4, GPU taken, can delete) and `hxbmsgtzxtmq8u` (runs 5-6), plus the 20 GB volume.
- **Next:** Hind-foot lift fix, then realism: servo model (BAM), action delay, domain randomization (mass +/-10%, friction), pushes.

### Run 7: hind feet lift
- **Did:** Run 7 = run 6 + a `swing_height` cost (weight -300). While a foot is scheduled to be in the air, its height should follow an arc: resting height (0.97 cm), up to 3.6 cm higher at mid-swing, back down at touchdown. Cost = (height - arc)^2 summed over the swinging feet. Same pod restarted: 6.5 min, ~$0.13.
- **Results** (walking forward at 0.2 m/s, measured in sim on the Mac):

  | | Run 6 | **Run 7** |
  |---|---|---|
  | Hind foot lift (LH / RH) | 0.6 / 0.7 cm | **3.7 / 3.9 cm** |
  | Front foot lift (LF / RF) | 3.6 / 2.9 cm | 3.7 / 4.0 cm |
  | Hind feet off the ground when scheduled up | ~64% | **80% / 87%** |
  | Hind knee swing | ~12 deg | **21 deg** |
  | Head tilt swing / head bob | 2.4 deg / 1.0 cm | 1.9 deg / 1.0 cm |
  | Torso lean (target +/-3 deg) | +/-3.1 deg, corr 0.99 | +/-3.5 deg, corr 0.99 |
  | Torso pitch (target 3 deg) | 3.0 deg | 3.0 deg |
  | Front / hind duty (spec 0.70) | 0.57 / 0.75 | 0.57 (LF 0.62, RF 0.52) / 0.71 |
  | Speed test | ok | ok (0.20 -> 0.19, turn -0.50 -> -0.51), never fell |

  Footfall chart (one stride, folded by the gait clock; each column = 1/20 stride):
  ```
  LH █████████████······█
  LF ·····█████████████··
  RH ███······███████████
  RF ████████··········██   <- lands ~0.2 stride late (spec: down from column 14)
  ```
- **Learned:**
  - Why run 6's hind feet skimmed: two holes. feet_clearance only cost ~10 points per episode, and gait_schedule counted a foot hovering 2 mm up as "in the air". Pricing the height along the whole swing closes both: at -300, run 6's skimming gait would have cost ~126 per episode in progress.csv's per-term units. (Corrected in run 10's entry: those per-term numbers aren't multiplied by the 0.02 s step, so they can't be compared with the ~55 total; in the same units speed tracking earns ~840.)
  - A target along the whole path (an arc) beats a target at one moment (the peak height): the policy can't cheat on the parts nobody checks.
- **Mistakes fixed:** my first reachability check let the hind foot move anywhere (7.9 cm lift). Redone with the foot kept under the hip: 4.5 cm max, so 3.6 cm is reachable.
- **Red panda notes (decide later):**
  - The right front foot lands ~0.2 stride late (down 52% instead of 70%). Front duty is 0.57 in both runs 6 and 7, so the lean is the likely cause, not the new term. Option: raise gait_schedule 1.0 -> 1.5.
  - Cost so far: ~$3.02 total. Stopped pods: `hf9abldp5ohuh8` (run 4, GPU taken, can delete) and `hxbmsgtzxtmq8u` (runs 5-7), plus the 20 GB volume.
- **Next:** front timing (run 8), then realism: domain randomization (mass +/-10%, friction), pushes, action delay; BAM servo model once the servo rig exists.

### Run 8: front feet back on the beat
- **Did:** Run 8 = run 7 with gait_schedule 1.0 -> 1.5 (one number). The run 5-7 pod's GPU was taken, so a new pod on the same volume (`9hgcar5rn7v23f`): 7.3 min of training, ~$0.15. Deleted the two old stopped pods.
- **Results** (walking forward at 0.2 m/s, measured in sim on the Mac):

  | | Run 7 | **Run 8** | Spec |
  |---|---|---|---|
  | Right front foot down | 52% | **62%** | 70% |
  | Left front foot down | 62% | **69%** | 70% |
  | Hind feet down (LH / RH) | 72 / 69% | 75 / 68% | 70% |
  | Torso lean (target +/-3 deg) | +/-3.5 deg, corr 0.99 | +/-3.3 deg, corr 0.99 | |
  | Torso pitch | 3.0 deg | 2.8 deg | 3 deg |
  | Hind lift (LH / RH) | 3.7 / 3.9 cm | 3.4 / 3.6 cm | 3.6 cm |
  | Head tilt / head bob | 1.9 deg / 1.0 cm | 2.7 deg / 1.7 cm | |
  | Speed test | ok | ok (all within 0.02), never fell | |

  ```
  LH ██████████████·····█
  LF ····██████████████··   exactly on spec
  RH ███······███████████
  RF ████████·······█████   1 column late (was 4)
  ```
- **Learned:**
  - The late foot only cost ~1 point per episode: weight 1.0 x 0.05 (one foot of 4 wrong 20% of the time) x 0.02 s x 1000 steps. A 50% raise was enough to tip it; the lean held.
  - A weight is relative: raising one term makes every other term matter less. At 10, footfall (worth up to 200/episode) would swamp speed tracking (20) and the lean (10); marching in place in perfect rhythm would be the cheap answer.
  - Side effect: head bob 1.0 -> 1.7 cm, probably from firmer front footfalls. Still far below run 5's 3.3 cm.
- **Mistakes fixed:** stopped the pod before config.json finished copying back. Rebuilt it from run 7's config + the one change and checked it against the code (only `impl` differs: warp on the GPU, jax on the Mac).
- **Cost so far:** ~$3.17 total. One stopped pod (`9hgcar5rn7v23f`) plus the 20 GB volume.
- **Next:** realism: domain randomization (mass +/-10%, friction), pushes, action delay; BAM servo model once the servo rig exists.

### Realism stress test (run 8's policy)
- **Did:** Before adding randomization, threw one real-world difference at a time at run 8's policy in sim on the Mac: 16 robots per condition walking 0.2 m/s for 10 s.
- **Results:**

  | Condition | Fell | Speed (target 0.20) |
  |---|---|---|
  | Baseline | 0/16 | 0.19 |
  | Friction 0.5 / 0.3 / 0.2 / 0.1 (floor and feet) | 0/16 each | 0.18-0.19 |
  | Mass x0.9 / x1.1 / x1.2 | 0/16 each | 0.19 |
  | Center of mass shifted 2 cm (any direction) or 5 cm forward | 0/16 each | 0.18-0.20 |
  | Servo stiffness x1.3 / x0.7 / x0.5 | 0/16 each | 0.19 / 0.16 / 0.12 |
  | Motor delay 20 ms / 40 ms | 0/16, **15/16** | 0.11 / 0.00 |
  | Pushes 0.1 / 0.2 / 0.3 / 0.4 m/s | 0, 1, 5, **16** of 16 | 0.16-0.19 |

- **Learned:**
  - Delay is the biggest threat by far: each balance correction arrives after the body has moved on, so it overcorrects and the wobble grows (torso roll swing 6 deg at 0 ms, 10-15 deg at 20 ms, 25 -> 35 deg -> fall at 40 ms). Like adjusting a shower that takes seconds to respond.
  - MuJoCo uses the HIGHER of the two friction values in a contact. Our feet are 1.0, so Go1's randomizer (floor only) would silently do nothing. Randomize the feet too.
  - Go1's biggest push (velocity_kick 3.0) changes our robot's speed by 0.41 m/s (checked in sim, = M_torso x kick / (pi x M_total)): twice our top walking speed.
  - 1 fall in 16 can be noise: a condition with physics identical to the baseline had one.
  - **Torque check:** run 8's knee swung at 6.1 rad/s, faster than the STS3215 can turn even with no load (4.7 rad/s). The sim's servos gave full stall torque at any speed; a real one gives less the faster it turns. The walk spec gives each foot 0.24 s in the air, and folding the front knee ~36 deg and back in that time needs ~5 rad/s.

### Run 9: real servo speed limit
- **Did:** Added a servo torque-speed limit (`servo_config`): each physics substep, the servo's twist is capped at stall x (1 - speed / 4.7 rad/s) in the direction it's already turning (braking allowed up to stall). Done by computing the position actuator's torque ourselves and handing MuJoCo the target that gives exactly the capped torque. Also built a motor delay (0-40 ms, in 4 ms substeps) in the same loop, off for now. Configs from runs 1-8 have no servo_config, so they replay unchanged. New pod (`oc5icdqjmzdrwh`, the run 8 pod's GPU was taken): 7.3 min of training, ~$0.16.
- **Checks before training:** smoke test passes; 0 of 27,000 actuator samples above the curve (using the speed the servo saw); delay reproduces the stress test (20 ms -> 0.13 m/s, 40 ms -> 15/16 fall); run 8's policy still walks with the limit (knee 6.1 -> 3.9 rad/s).
- **Results** (walking forward at 0.2 m/s, measured in sim on the Mac):

  | | Run 8 | **Run 9** |
  |---|---|---|
  | Duty LF / RF / LH / RH (spec 0.70) | 0.69 / 0.62 / 0.75 / 0.68 | **0.67 / 0.68 / 0.71 / 0.71** |
  | Foot lift LF / RF / LH / RH | 3.9 / 3.9 / 3.4 / 3.6 cm | 3.8 / 3.6 / 3.4 / 3.8 cm |
  | Head tilt / bob | 2.7 deg / 1.7 cm | 2.2 deg / 1.3 cm |
  | Lean / pitch | +/-3.3 deg (corr 0.99) / 2.8 deg | +/-3.4 deg (corr 0.99) / 3.0 deg |
  | Front knee / hip swing | 35.8 / 28.2 deg | **26.8 / 33.3 deg** |
  | Speed test | ok | ok, never fell |

  ```
  LH █████████████······█
  LF ····██████████████··
  RH ███······███████████
  RF ████████······██████   on spec
  ```
- **Torques at 1.8 kg** (rule: average below rated 0.98, peaks below 60-70% of stall = 1.76-2.06 N*m):

  | Joint | Average | Peak (99th pct) | Top speed |
  |---|---|---|---|
  | hip_side | 0.42 (43%) | 1.77 (edge) | 3.2 rad/s |
  | hip | 0.61 (62%) | **2.33 (fail)** | 4.0 rad/s |
  | knee | 0.76 (78%, run 8: 88%) | **2.74 (fail)** | 3.9 rad/s (run 8: 6.1) |

  At 1.98 kg: almost identical.
- **Learned:**
  - A servo's push (torque) comes from current, and a spinning motor makes its own voltage that fights the battery, so the faster it turns, the less it can twist. Twisting hard and turning fast are different things (stuck jar lid vs. loose lid).
  - With the real limit, the policy steps more from the hip and folds the knee less, and the feet still lift the same. The walk the real servos can do came out tidier than the one they can't.
- **Mistakes fixed:** my first "over the curve" check compared torque with the speed after the physics step; checking with the speed the servo actually saw gave exactly 0. Stopped run 8's pod before config.json copied (logged in run 8); this time the pod was stopped only after all 3 files arrived.
- **Red panda notes (decide later):** hip and knee peaks still fail the torque rule. The servo stiffness is still a placeholder (kp 10), so the BAM model will move these numbers.
- **Cost so far:** ~$3.33 total. Stopped pods: `oc5icdqjmzdrwh` (run 9) and `9hgcar5rn7v23f` (run 8, GPU taken), plus the 20 GB volume.
- **Next:** run 10, randomization (motor delay, pushes, servo stiffness, mass, friction on floor and feet, center of mass).

### Run 10: domain randomization (tougher, but hind feet skim again)
- **Did:** Run 10 = run 9 + domain randomization. Each training robot gets its own friction (0.3-1.0 on floor AND feet), servo stiffness (x0.7-1.3), whole-robot mass (1.62-1.98 kg), torso center of mass (+/-2 cm), motor delay (0-40 ms), and pushes up to 0.6 m/s every 1-3 s. New pod `tat0hmgxcgxnwh`: 7.1 min, ~$0.17. The viewer now shows the walk itself by default (no pushes, 0 delay; `--pushes`, `--delay_ms` to stress it).
- **Push-size mistake caught:** the env's push force uses the whole robot's mass, so a kick k changes speed by k / pi (checked in sim). My first stress test used the torso's mass alone, so its push labels were 2.3x too small ("0.3 m/s" was really 0.70). Re-ran with correct sizes before choosing 0.6 m/s (where run 9 starts falling).
- **Results, stress test** (16 robots per condition, 0.2 m/s, 10 s):

  | | Run 8 | Run 9 | **Run 10** |
  |---|---|---|---|
  | Delay 40 ms | 15/16 fell | 0/16, 0.08 m/s | **0/16, 0.15 m/s** |
  | Delay 20 ms | 0.11 m/s | 0.15 m/s | **0.17 m/s** |
  | Push 0.7 m/s | 5/16 fell | 5/16 | **0/16** |
  | Push 0.9 m/s | 16/16 | 13/16 | **6/16** |
  | Servo stiffness x0.5 | 0.12 m/s | 0.11 m/s | **0.18 m/s** |
  | Everything moderate at once | 0.11 m/s | 0.14 m/s | **0.18 m/s** |

- **Results, the walk** (no pushes, 0 delay):

  | | Run 9 | Run 10 |
  |---|---|---|
  | Hind foot lift | 3.4 / 3.8 cm | **0.4 / 0.3 cm** |
  | Front foot lift | 3.8 / 3.6 cm | 2.5 / 3.0 cm |
  | Head bob | 1.3 cm | 2.2 cm |
  | Duty front / hind | 0.67 / 0.71 | 0.62 / 0.75 |
  | Speed test | ok | ok, never fell |

- **Torques at 1.8 kg:** knee peak 2.74 -> **1.91 N*m (65% of stall, passes)**; hip peak 2.12 (72%, just over); hip_side 0.93; all averages pass (knee 0.61 = 63% of rated).
- **Learned:**
  - The walk got defensive: hind feet kept near the floor for instant "emergency touchdowns". Pushes every 1-3 s made training a constantly shaking bus, so it bus-walks all the time.
  - Skimming costs 120 per episode in per-term units vs. 825 for speed tracking: cheap enough to accept. My run 7 entry compared per-term units with the episode total (fixed above).
  - The policy has no push clock (pushes are only in the critic's inputs), so rarer pushes don't make it fail on an early one; they just change how much of its life is spent being shoved. Back-to-back pushes are the case to keep practicing.
  - A camera-sized change (50 g at the chest: COM 0.4 cm forward, +3% mass) is well inside what run 10 already handles (2 cm, +10%).
- **Decisions and why:** delay 0-40 ms (careful build ~5-10 ms, slow one ~25-30 ms), pushes to 0.6 m/s (run 9's falling point; bigger shoves are the get-up policy's job), all at once. Depth camera noted in CLAUDE.md as a Tier 2/3 candidate (lower front corner of the torso, behind a dark window).
- **Next:** run 11 as two one-setting experiments: 11a = swing_height -300 -> -600; 11d = pushes every 1-10 s instead of 1-3 s. Pick by hind-foot lift and stress test.

### Run 11: two experiments on the skimming hind feet (11a vs 11d)
- **Did:** Two one-setting runs from run 10, back to back on one pod (`k2fayf9zwsrt4p`, a ~25% slower host): 11a = swing_height -300 -> -600; 11d = pushes every 1-10 s instead of 1-3 s (gap picked once per episode). Added `--set key=value` to `tools/train_walk.py` for this; each override is saved in config.json and a misspelled name stops the run. ~25 min for both, ~$0.37.
- **Results:**

  | | Run 10 | 11a (swing x2) | 11d (rarer pushes) |
  |---|---|---|---|
  | Hind foot lift | 0.4 / 0.3 cm | **3.1 / 3.4 cm** | 0.9 / 1.0 cm |
  | Front foot lift | 2.5 / 3.0 cm | 3.3 / 3.5 cm | 3.7 / 3.2 cm |
  | Body bounce (torso up/down per stride) | 1.9 cm | 2.8 cm | ~2.8 cm |
  | Duty front / hind | 0.62 / 0.75 | 0.58 / 0.70 | 0.63 / 0.76 |
  | Push 0.9 m/s | 6/16 fell | 3/16 | **0/16** |
  | Delay 40 ms | 0.15 m/s | 0.16 m/s | 0.16 m/s |
  | Knee torque peak (rule < 2.06) | 1.91 | 2.31 | 2.02 |
  | Hip torque peak | 2.12 | 2.14 | 2.13 |
  | Turn in place (target -0.50) | -0.48 | -0.51 | -0.43 |

- **Head motion test (11a):** turn +/-30 deg, tilt +/-20 deg up to 2 Hz, plus +/-10 deg nods at 2.5 Hz: 0/16 falls, same speed, same body bounce. The head + neck (~185 g) doesn't disturb the walk; personality animations are safe in this range.
- **Learned:**
  - Forcing the lift didn't cost stability (11a beat run 10 on pushes): skimming was the cheapest way to be stable, not the only way.
  - Constant shoving was part of the cause: rarer pushes alone tripled the hind lift and gave the most robust walk yet (0/16 at 0.9 m/s), probably because it got more practice walking well.
  - "Head bob" was the whole body bouncing (torso and head move the same up/down); head stabilization only corrects angles, not up/down motion.
  - A depth camera on the torso front corner loads the front legs like a seesaw: front share = (x + 12) / 24 for supports at +/-12 cm (50 g at x = 15 cm: front +6%, hind -1% in sim). The battery moved back can balance it.
- **Decisions and why:** run 12 = pushes every 1-10 s (11d) + swing_height -450 (halfway: rarer pushes need less push to lift) + wider head motion in training (turn +/-30, tilt +/-20, random nods, faster), which costs nothing and covers personality animations. Camera stays out of the model until it's chosen (Phase 2 masses).
- **Cost so far:** ~$3.87. Stopped pod `k2fayf9zwsrt4p` plus the 20 GB volume.

### Run 12: rarer pushes + swing height -450 + personality-sized head motion
- **Did:** Run 12 = run 10 + pushes every 1-10 s (11d) + swing_height -450 (between 11a's -600 and -300) + wider head motion while walking (turn +/-30, tilt +/-20, random nods +/-10 deg, up to 1.5 Hz; new `gait_nod_drift_deg`, 0 in older configs so they replay unchanged). New pod `745gnnmsvdx6zg` (4090s come and go in EU-RO-1; a 5090 vanished between the stock read and the create). ~13 min, ~$0.20.
- **Results:**

  | | 11a | 11d | **Run 12** |
  |---|---|---|---|
  | Hind foot lift | 3.1 / 3.4 cm | 0.9 / 1.0 cm | **3.1 / 2.7 cm** |
  | Front foot lift | 3.3 / 3.5 cm | 3.7 / 3.2 cm | 3.6 / 3.5 cm |
  | Body bounce | 2.8 cm | 2.4 cm | 2.9 cm |
  | Duty front / hind | 0.58 / 0.70 | 0.63 / 0.76 | 0.61 / 0.72 |
  | Stress test | 3/16 fell at 0.9 m/s | 0/16 | **0/16 in every condition** (icy, 40 ms, 0.9 m/s, +20% mass) |
  | Knee torque peak (forward) | 2.31 | 2.02 | 2.27 |
  | Turn in place (target -0.50) | -0.51 | -0.43 | -0.51 |

  ```
  LH █████████████······█
  LF ····█████████████···
  RH ███······██████████·
  RF ████████······██·███
  ```
- **Hardware check** (run 12, the env's own random commands + head motion, 16 robots x 20 s; STS3215 C018 reseller specs at 12 V: stall 2.94 N*m / 2.7 A, rated 0.98 N*m / 0.9 A, no-load 0.18 A, 4.7 rad/s; overheat cutoff ~70 C from a community test). Current = 0.18 A + torque / 1.17 N*m/A; heat = I_rms^2 x 4.4 ohm.

  | Servo | Avg | Peak (99th) | RMS current | vs rated 0.9 A | Heat |
  |---|---|---|---|---|---|
  | hip_side | 0.40 N*m | 1.52 | 0.65 A | 73% | 1.9 W |
  | hip | 0.42 | 1.75 | 0.67 A | 74% | 2.0 W |
  | **knee** | **0.82** | **2.26** | **1.01 A** | **112%** | **4.5 W** |
  | tail (3) | <= 0.12 | <= 0.28 | <= 0.29 A | <= 33% | <= 0.4 W |
  | neck (3) | <= 0.17 | <= 0.65 | <= 0.34 A | <= 38% | <= 0.5 W |

  - All 18 servos: ~9.4 A average, 14-17 A peaks (upper estimate). A 3S 2200 mAh pack: ~10 min of walking. One daisy chain of thin servo cables can't carry that: give each leg its own power wire; capacitor near the bus.
  - Battery voltage: stall torque and top speed scale with voltage (12.6 V full: 3.09 N*m, 4.93 rad/s; 10.5 V empty: 2.57, 4.11). The walk still works at 10.5 V (0/16 falls, same speed error), but peaks become a bigger share of stall (knee 2.26 = 83% at 11.1 V).
  - Not modeled yet: servo stiffness / friction / backlash (BAM), joint zero offsets, real masses, IMU mounting tilt, floor compliance, measured delay, printed-part strength (2.9 N*m on a 2 cm screw radius = ~145 N), tripod holding loads.
- **Learned:**
  - Heat follows RMS current (I^2 R), not average torque: the knee's average (0.82 N*m) is under rated, but its RMS current is 112% of rated.
  - Rarer pushes + a moderate lift price gave both: feet up and the most robust walk yet.
  - Six kinds of hardware failure to check every time: thermal, mechanical, electrical, timing, sensing, modeling.
- **Next:** fix knee heat (options with tradeoffs to decide), add joint-offset randomization.
- **Cost so far:** ~$4.07. Stopped pod `745gnnmsvdx6zg` plus the 20 GB volume.

### Run 13: torque price x3 + joint offsets (and the knee load is jitter)
- **Did:** Run 13 = run 12 + `torques` -0.0035 -> -0.0105 + joint calibration offsets (`servo_config.joint_offset_deg` 2.0): each leg servo's "0" is off by a fixed random angle per episode; the policy reads angle + offset, the servo drives the joint to target - offset, the critic sees true angles. Run 12 untrained handled +/-2 deg (0/16 falls), +/-6 deg slowed it to 0.13 m/s. Pod `oc24xznq0zaal3`: ~11 min, ~$0.17.
- **Results:** walk and robustness held (0/16 falls everywhere except 1/16 at a 0.9 m/s push), but knee RMS current only went 1.01 -> 0.97 A (112% -> 108% of rated): the policy paid the tripled torque price (-38 -> -107 per episode) instead of changing.
- **Learned (and a wrong explanation corrected):**
  - I said half the knee load was legs pushing against each other. Measuring ground forces while standing showed it isn't: sideways forces are 0.3-0.5 N vs 4-5 N of weight per foot.
  - It's jitter: only ~0.4 N*m of the knee's 0.75-0.82 N*m holds the body up; the policy's targets jump ~3 deg every 20 ms and the knee torque reverses direction in ~40% of steps. The policy reacts to sensor noise. On a real servo: buzzing, heat, gear wear. The smoothness price is tiny (action_rate -3.5 per episode vs +790 tracking).
  - A general torque penalty doesn't target jitter; a smoothness price (action_rate) would.
- **Next considered:** J1 = action_rate -0.01 -> ~-0.1.

### Posture numbers + run 14: stance and leg length (14a vs 14b)
- **Why:** run 13's walk looked wrong (deep crouch, zigzag legs, feet far ahead of the shoulders). Checked with numbers instead of GIFs:
  - The home pose itself is a crouch: knee 80 deg, hip +/-38.3 deg puts the feet under the hips at 19.2 cm (= base_height 0.19). Run 13 stood at 16.3-16.5 cm with knees ~93 deg.
  - pose_weights [hip_side, hip, knee] = [1.0, 0.2, 0.1]: run 13's crouch still earned 415 of 500 pose points; base_height cost only ~-13 per episode.
  - Holding torque (standing, 1.8 kg): knee 80 deg 0.34 N*m, 45 deg 0.20, 30 deg 0.14.
- **Tooling:** `tools/watch_policy.py` now also walks a fixed command (default 0.15 m/s, `--vx`, `--seconds`) and prints mean torso height plus each foot's touchdown / liftoff position relative to its own hip in the robot's frame (saved to `runs/<run>/posture.json`); the GIF is that fixed-command walk from a low side-view camera that turns with the robot. `sim/red_panda_walk.py` gained `model_config` (`leg_scale`, `stance_knee_deg`) applied by `build_spec()` when the model loads (runs 1-13 load the original model unchanged, checked).
- **Red panda proportions:** shoulder height ~20-34 cm on a 50-64 cm head-and-body (~40-45%), i.e. a torso ~13-17 cm up for our 32 cm body. Pupper legs (23.85 cm) are 75% of our body length.
- **Did:** two runs from run 13, back to back (~28 min incl. idle, ~$0.42):
  - 14a = home knee 45 deg / hip +/-21.7 deg, Pupper legs, base_height 0.229.
  - 14b = legs x0.7 (8.64 + 8.05 cm), home knee 45 deg, base_height 0.163, swing lift 2.5 cm, max_foot_height 0.035, commands [0.14, 0.07, 0.5]. foot_rest_height stays (it's the foot ball's radius, unchanged); fall cutoff 10 cm leaves 6 cm margin.
- **Results** (posture at 0.12 m/s; hardware check at 1.8 kg with mixed commands; stress test at 0.12 m/s):

  | | Run 13 | 14a (Pupper, 45 deg) | 14b (x0.7, 45 deg) |
  |---|---|---|---|
  | Torso height (target) | 16.4 cm (19.0) | 20.4 cm (22.9) | 14.3 cm (16.3) |
  | Front feet land / lift off (ahead of hip) | +8.3 / +5.9 cm | +10.9 / +8.0 | +7.8 / +4.4 |
  | Hind feet land / lift off | +4.2 / -2.2 | +1.2 / -4.5 | +0.0 / -3.2 |
  | Feet outward | 4.4-6.6 cm | 4.9-6.7 | 3.4-5.0 |
  | **Knee RMS current (rated 0.9 A)** | 0.97 A (108%) | 0.80 A (89%) | **0.62 A (69%)** |
  | **Knee peak (99th)** | 2.21 N*m (75% stall) | 1.74 (59%) | **1.24 (42%)** |
  | Hip peak (99th) | 1.88 | 1.90 | 1.55 (53%) |
  | All 18 servos | 9.2 A avg | 8.2 A | 7.3 A |
  | Stress test (icy, +20% mass, kp x0.5, 40 ms, 0.9 m/s push, all) | 0/16 everywhere | 1/16 at x1.2 mass and at 0.9 m/s | **0/16 everywhere** |
  | Speed test | ok | ok | ok within 0.14 range (backward -0.06 of -0.09, curve 0.34 of 0.40) |

- **Learned:**
  - Most of the knee savings come from bending the knee less (0.38 -> 0.20 N*m holding); shorter legs add a smaller cut (-> 0.14) and are the only way to straighten the legs without a 23 cm-tall body. (My first comparison changed leg length and bend at once and credited the lever; an outside review caught it.)
  - The crouch wasn't what made the front feet reach forward: they land 7.8-10.9 cm ahead of the hips in all three bodies. That needs a foot-placement cost (P4).
  - Both new bodies still sit ~2-2.5 cm below their height target (base_height is cheap).
  - Double touchdowns (front-left / right-hind ~2 per stride) persist: jitter / bounce, same as run 13.
  - Compare runs by measurements, not total reward, once the body or rewards change.
- **Cost so far:** ~$4.65. Stopped pod `oc24xznq0zaal3` plus the 20 GB volume.

### Run 15: three experiments in parallel on the short-leg body (15a / 15b / 15c)
- **Did:** Made run 14b's body the default (legs x0.7, home knee 45 deg, base_height 0.163, swing lift 2.5 cm, max_foot_height 0.035, commands [0.14, 0.07, 0.5]); CLAUDE.md's "Rough-model body" row updated. Added a `foot_under_hip` cost: per planted foot, (fore/aft^2 + sideways^2) relative to its thigh joint (where a straight leg hangs; the hip body is 3 cm further in), in the torso frame. Checked: the env's term gives -249/episode on run 14b's gait at weight -30, matching a separate measurement. Three pods in parallel on the shared volume (each writes only its own folder): 15a = foot_under_hip -30, 15b = action_rate -0.01 -> -0.5, 15c = base_height -20 -> -500. ~12.7 min each, ~$0.56 total.
- **Results** (posture at 0.12 m/s; hardware check at 1.8 kg; stress test at 0.12 m/s):

  | | 14b | 15a feet under hips | 15b smoothness x50 | 15c height x25 |
  |---|---|---|---|---|
  | Torso height (target 16.3) | 14.3 cm | 15.4 | 15.0 | **15.8** |
  | Front feet land / lift off | +7.8 / +4.4 cm | **+4.1 / +1.2** | +1.8 / -2.4 | +6.6 / +2.0 |
  | Feet outward | 3.4-5.0 cm | **3.0-4.4** | 3.8-6.8 | 2.9-4.3 |
  | Touchdowns per foot in 10 s (~12.5 = one per stride) | 14-26 | 13-21 | **12-19** | 12-21 (front fixed) |
  | Footfall pattern / speed test | on spec / ok | on spec / ok | **drifted / sideways 0.03 of 0.07** | on spec / ok |
  | Knee RMS current (rated 0.9 A) | 0.62 A (69%) | 0.56 A (62%) | 0.53 A (58%) | 0.63 A (70%) |
  | Knee peak (99th) | 1.24 N*m | 1.06 | 1.08 | 1.25 |
  | Stress test | 0/16 everywhere | 0/16 everywhere | 0/16 everywhere | 1/16 at 0.9 m/s push |

- **Hind feet skim again** (measured at 0.12 m/s, inside the new speed range): 0.1-0.4 cm lift in 14b, 15a, 15c (front 2.1-2.5 cm). The swing_height cost is (height - arc)^2, so shrinking the lift 3.6 -> 2.5 cm halved the price of skimming. Fix for run 16: -450 -> -930 = x(3.6/2.5)^2.
- **Can the short legs lift 3.6 cm?** The joints reach ~11 cm, but the gait gives the foot 0.12 s to rise: 2.5 cm needs ~34 deg of knee fold (avg 4.9 rad/s, already the servo's no-load speed), 3.6 cm needs 45 deg (6.5 rad/s, impossible). Getting 3.6 cm would need a slower stride, less time on the ground per foot (toward a trot), longer legs, or faster servos. Decision: keep 2.5 cm (= 15% of leg, the spec).
- **Jumping, ruled out:** full-effort crouch-and-extend in sim leaves the ground 40 ms (~2 mm; flight height = g t^2 / 8). Without the servo speed limit: 218 ms (~6 cm). The servos are strong enough but too slow. Recorded in CLAUDE.md; climbing over objects (slow) added to Tier 3, walking over small bumps to Tier 2.
- **Learned:**
  - Run several one-change experiments in parallel from the same base: same cost, a third of the wall time, clean comparisons. Combine the winners next and re-check (interactions).
  - A price that's squared in a length changes with the square of the body scale: rescale weights when the body changes.
  - These servos are strong but slow: slow or held skills (rearing, careful climbing) fit; fast ones (jumping, quick high steps) don't.
- **Tooling:** `tools/train_walk.py` now records where the time goes (`timing.json`: imports, env build, setup + first compile, each eval, each training chunk, throughput) and takes `--ppo key=value` and `--no_checkpoints` for speed experiments.
- **Next:** run 16 = 15a + 15c + action_rate -0.15 (gentler than 15b) + swing_height -930, with profiling. Run 17: bumps (rough terrain).
- **Cost so far:** ~$5.21.

### Runs 16-20: soft landing, profiling, and the hind feet that won't lift
- **Run 16** = 15a + 15c + action_rate -0.15 + swing_height -930 (rescaled for the 2.5 cm lift). Best walk then: front feet land +3.1 / lift off -1.6 cm (centred), hind lift 2.6/2.4 cm, torso 15.9 cm, knee 0.73 A (82% rated), stress test 0/16 except 2/16 at 0.9 m/s pushes. But hind feet "tapped" 27-33 times in 10 s: 1 mm hops, ~50 ms, mid-stance, exactly when the other hind foot landed hard (0.18-0.21 m/s) -> the rear bounced.
- **Run 17** = + soft_landing -15 (downward foot speed^2 in the last cm before touchdown). Landings 4x softer (hind 0.05 m/s), taps 18 -> 6, but hind feet skimmed again (0.3 cm). Cause: the swing arc (sine) itself asked the foot to land at 0.32 m/s, so the two rewards fought.
- **Reward audit** (tools/eval/reward_audit.py): scored ideal foot paths against all foot terms together. feet_clearance (Go1's) rewarded "hover then drop" (a stomp). Fix -> **run 18**: bell-shaped arc (lift x sin^2, zero speed at touchdown; `reward_config.swing_arc`), soft_landing -5, feet_clearance 0. Hind still 0.6-0.7 cm.
- **Runs 19a/19b**: ground time per foot (duty) 0.65 / 0.60 (more swing time). Front lift up to 2.5-2.8 cm; hind still 0.2-0.5 cm. Ruled out with measurements: servo speed (front fine), balance (centre of mass centred), load (52% front / 48% hind).
- **Hind legs were nearly straight** (stance knee ~30 vs front ~64 deg) because the 3 deg nose-down slant target fought the level 45/45 home pose. **Runs 20a/20b**: home pose = the walking posture (`model_config.stance_height` 0.158, `stance_pitch_deg` 2.0 -> front knees 61, hind 49 deg; base_height 0.158, pitch_deg 2.0), duty 0.70 / 0.60. Hind still 0.5-0.7 cm; foot placement cleaner.
- **In-air leg test** (tools/eval/leg_in_air.py: torso held up, leg traces the bell path through the servo model): at duty 0.70 (0.24 s swing) the servos physically top out at ~1.6 cm (hind) / ~1.9 cm (front); at 0.60 (0.32 s) hind reaches 2.4 cm, front 2.7 cm, both landing softly. So at 0.60 the hind skimming is "won't" (training), not "can't". A hind lift needs 47 deg of knee fold vs 40 front -> under-priced.
- **Speed work (profiling)**: `tools/train_walk.py` writes timing.json (+ `--ppo`, `--no_checkpoints`, `--compile_cache`); `tools/bench_env.py` measures physics throughput. Findings: imports were 88 s (thousands of small files on the network disk), compile ~50-90 s, training ~320 s at ~620k steps/s; physics alone 1.19M steps/s at 8192 robots, so about half of training time is the PPO update. My first benchmark called the step from Python once per step and measured ~24 ms of call overhead (fixed with lax.scan). Fixes: JAX + Warp compile caches on the volume (setup 47 -> 11-16 s when warm); import fix = small Python files packed once into /workspace/fastlib.tar (zstd, 186 MB) and unpacked locally, big files linked back (imports 88 -> 3 s). Total run time 513 s (run 16) -> 377 s (run 20a), -26%. Packing the whole 13 GB venv would not fit the 20 GB volume (caught by a size check; a pkill -f once killed my own ssh session).
- **Hardware notes**: STS3215 C018 reseller specs (stall 2.94 N*m / 2.7 A, rated 0.98 N*m / 0.9 A, no-load 4.7 rad/s, ~70 C cutoff); heat follows RMS current; jumping ruled out (40 ms in the air = ~2 mm).
- **Cost so far:** ~$7.00.
- **Open decision (next):** run 21a = duty 0.60 + hind-only swing price x2; run 21b = duty 0.70 + lift target 1.6 cm (the physical limit) + hind price x2. Then terrain (prep -> 17a/b shins on/off -> stage 2 with per-robot curriculum).

### Runs 21-24: hind price x2, bumps, knee rewards, and the clearance floor that worked
- **Run 21a** (duty 0.60, new `swing_hind_mult` = 2: hind feet pay twice the swing price). Hind lift 0.7 -> 2.2/2.6 cm, but front dropped 2.1/2.6 -> 1.0/1.0 cm (front feet touching 31-39% of their air time). Total swing penalty barely moved (-169 -> -149): the policy moved the lifting effort from one end to the other (a seesaw), it didn't lift more. Confirms the hind skim at 0.60 is "won't", not "can't". Dropped 21b: lowering the lift target to 1.6 cm shrinks the miss for a skimming foot (1.9 -> 1.0 cm, squared = x0.28), so even with x2 it would have made skimming cheaper than before.
- **Terrain (Tier 2 prep, built)**: `sim/terrain.py` makes a lumpy height field: 10 difficulty rows (lumps up to 0.15 -> 1.5 cm) x 8 columns (6 lumpy, 2 flat = 75/25), 3 m tiles, 3 m flat border, 1 cm grid, smooth cos^2 lumps 4-8 cm wide (my choices: lumpy only first, 4-8 cm, 75/25, max 1.5 cm). `terrain_config.enable` swaps the floor plane for it (off = runs 1-21 exactly). Every height in the rewards and the fall check is now measured from the ground under it: `_feet_height` (17 sample points under the foot ball, so a foot touching a lump's side reads its radius; checked against MuJoCo's own contact distance: mean error 0.22 mm vs 0.64 mm for "ground straight below"), `_torso_height` (from the ground under the 4 feet, not under the belly: a lump under the belly isn't something the legs stand on). The height lookup copies MuJoCo's triangle split of each grid cell: 0.0 um error vs raycasts. Robots spawn on a random tile, raised above the tallest lump under them. Contact buffers: up to 12 contacts per robot on lumps (vs 4 on flat).
- **Bumpy eval** (`tools/eval/terrain_eval.py`): falls, speed, "toe catches" (foot touching while it should be in the air) and swing peak, per terrain row. Flat-trained 20b and 21a on the hardest lumps: 0/16 falls each, both slowed to 0.086 m/s. The ball feet ride over smooth lumps like wheels, so skimming costs speed, not falls. My prediction check: I guessed 21a (front feet low) would trip more because more weight is on the front; it was a draw (weight is ~52/48 anyway).
- **What a skimming leg does** (joint angles at mid-swing vs lift-off): the hip ("top motor") swings forward 16-21 deg on every leg; the difference is the knee. Lifting legs bend the knee +10 to +22 deg; skimming hind legs straighten it (-2 to -15 deg) and slide the foot along the floor like a stiff pendulum.
- **Run 22a** (bumps, everything else as 20b): first try crashed (GPU out of memory at an eval: Warp's curved-contact scratch buffers are sized by the contact budget, and the eval env had the training env's 16 x 8192 budget for 128 robots). Fix: eval env gets 16 x 128, and JAX takes 70% of the GPU on terrain runs. Result: much better on lumps (row 9 speed 0.107 vs 0.086, front toe catches 6% vs 10-14%, knee 64% of rated vs ~82%), stress test 1/16 at 0.9 m/s pushes and 1/16 at +20% mass. But hind feet still skim (0.7 cm) with straightening knees: the lumps were too forgiving to force lifting.
- **Run 22b** (flat + `swing_knee` -0.9: (knee - target)^2 in swing, target = home knee + 28 front / 32 hind deg x bell; IK: that lifts the foot 2.5 cm straight up from home; same weight on all four knees so no seesaw). Front kept 2.0/2.5 cm, hind 0.7 -> 1.3/1.0 cm, but mostly by crouching: every knee bent 7-9 deg more all the time (torso 15.6 -> 15.1 cm), and hind knees bent only +2 / -1 deg more in swing than in stance. Lesson: a reward on a fixed joint position can be met by changing where the joint starts.
- **Run 23**: knee fold measured from the knee's own angle at lift-off (`swing_knee_ref` "liftoff", 40 / 47 deg from IK, weight -0.5), so a knee that is bent all the time earns nothing. New `_track_swings` remembers the lift-off knee angle and each swing's highest point by the gait clock (a dragging foot never leaves the ground, so contact-based bookkeeping would miss it). New `swing_floor` (my idea): once per swing, (1.5 cm - swing peak)^2, zero if it cleared; a minimum, not a target, so it never pushes higher than needed. Weights sized so 20b's skimming hind legs pay what `swing_height` charges them. Results: **23a** (knee fold, flat) hind 0.5/0.4 cm, no change; **23b** (+ bumps) one hind leg 1.9 cm, the other 0.3; **23c** (+ bumps + floor -22000) all four lift (hind 2.3/2.5 cm), but knee current 112% of rated, peaks 84% of stall, hind knees pinned at their torque-speed limit 33-36% of the time, torso bouncing 14.9-16.7 cm. Cause: the knee target asked for 47 deg of fold in 0.16 s, a peak knee speed of 0.82 rad x pi / 0.32 s = 8 rad/s, but the servo tops out at 4.7 rad/s. The reward asked for something physically impossible, so the policy pinned the knees trying.
- **New checks**: knee margin (% of time each knee uses >= 95% of the torque available at its current speed; 99th percentile knee speed) and a battery-sag test (servo torque and speed x0.9, about a 3S pack under 12-14 A).
- **Run 24** (bumps + floor, no knee term): **24a** hind lift 1.6/2.6 cm (1.5-1.7 on lumps), knee 72% of rated, peak 48% of stall, hind knees pinned < 1% of the time (< 2% with 10% weaker servos), fastest on the hardest lumps yet (0.116 m/s), torso steady 15.5-16.1 cm, stress test 0/16 everywhere except 1/16 at 0.9 m/s pushes. **24b** (+ torque price x2, -0.021): knee 65% but hind lift fell under the floor (1.3/1.1 cm). The floor alone does the lifting; the knee term only added strain. Leftovers: front toe catches up to 15-19%, hind feet still tap mid-stance (~30 touchdowns in 10 s vs ~12 planned).
- **Decision: 24a is the new default** (duty 0.60, terrain on, `swing_floor` -22000; knee term kept but off).
- **Lessons:** a reward on a joint *position* can be met by changing where the joint starts (22b's crouch); a reward on a joint *path* has to respect the servo's speed (23c). Asking for the result (foot clears 1.5 cm) and letting the policy find the joint motion was simpler and gentler on the servos. Change one thing per run: 23c changed three, and only 24a showed which one mattered.
- **Ops:** one watcher per pod, and stop each pod as soon as its results are copied (a stalled copy left run 24's pods idle ~25 min, ~$0.75). Next: run the evals on the pod right after training (GPU is much faster for the 400-robot stress test).
- **Cost so far:** ~$9.15.


### Evals on the pod (commit 34fc79f)
- **Did:** the standard checks now run on the pod right after training (`tools/eval/run_evals.sh`, called by `pod_run.sh`): lift, knee margin (`knee_margin.py`, moved out of the scratchpad), hardware, terrain, stress, GIF. `fetch_run.sh` copies a run's small results back; `pod_env.sh` holds the shared pod setup (venv + fast imports).
- **Timings on the 4090:** lift 31 s, knee 62 s, hardware 38 s, terrain 42 s, stress 49 s, GIF 47 s: the whole suite in ~4.5 min (the stress test was the slow one on the Mac). `terrain_eval.py` went 226 -> 42 s by running every tile in one batch (`reset(rng, tile)` picks the tile per robot, so it compiles once instead of once per tile).
- **EGL fix (GIF on a headless pod):** the pod image has NVIDIA's EGL driver but not the `libEGL.so.1` loader, so `pod_env.sh` installs `libegl1` (~7 s). `MUJOCO_GL=egl` is set only for the GIF step, because with it set, importing mujoco loads that GL library in every check.
- **Measurement fix:** 24a trained with terrain on, so the standard checks were quietly running on random lumpy tiles. `rpw.use_flat()` now puts them on the flat floor (terrain has its own check), so runs compare fairly.
- **24a on true flat** (`runs/rp_run24a/eval_flat/`): hind lift 2.2/2.2 cm (front 2.3/2.3), not the 1.5-1.7 measured on lumps; knee 74% of rated, peak 1.47 N*m (50% of stall); hind knees pinned < 1% of the time (0.4-1.7% with 10% weaker servos; front 4.8-8.8%); stress 0/16 everywhere except 1/16 at friction 0.2 floor-only and 1/16 at 0.9 m/s pushes.
- **Learned:** when the training setup changes (terrain on), check that every eval still measures what it says it measures. The "hind lift 1.6 cm" headline was the lumps, not the gait.
- **Ops:** two test pods had a broken public SSH port (sshd was up per the pod logs); a fresh pod worked. If SSH times out on a new pod, replace it rather than debug.

### Runs 25-26: foot force, the same-side prop, and a new footfall schedule
- **Did:** measured every 24a touchdown against the gait clock (on time / early / mid-stance tap). The hind mid-stance taps were run 16's bounce again: hind feet landed early and hard (0.30 m/s) and 28 of 30 taps came right after the other hind foot landed. The front feet weren't slamming: they lifted 50-60 ms late, and the policy asked for the knee fold on time, but the knee was pinned on torque (92%) because the foot still carried the body.
- **Foot force sensor** (`sim/red_panda.xml`, contact sensor data="force", up to 4 points per foot; rewards only, the real robot can't measure it). Matches MuJoCo's `mj_contactForce` exactly; Warp OK (standing: 17.78 N vs 17.66 N weight). 24a's front feet held ~50% of the body at their scheduled lift-off, *more* than mid-stance (44%); hind feet unloaded in time.
- **Run 25a** (soft_landing -5 -> -15): hind landings 0.30 -> 0.11-0.17 m/s, taps 30 -> 16, but hind lift 2.2 -> 1.0 cm. The floor was squared: 0.5 cm short cost only 0.55 per swing, cheaper than landing softly from 2.2 cm.
- **Run 25b** (new `unload` -4: (foot force / body weight)^2, ramping over the last 60 ms of stance + the first 100 ms of swing): front toe catches 15-18% -> 0%, but hind lift 0.8 cm and hind toe catches 55%. Force check: when a front foot lifted, the hind foot on the same side came down mid-swing as a prop carrying 41% of the body. The window didn't cover that part of the hind swing: a free zone. 24a already did a milder version (29%).
- **Why:** at duty 0.60 with the old offsets, each front lift-off left the robot on the two feet of the *other* side. The body is 7 cm inside that line; balancing torques (prop x 14 cm = weight x 7 cm) needs a 50% prop. No reward weight can remove that; only the schedule can. Charging the whole swing would just make the policy split the load between the front and hind foot (squared: 25/25 costs half of 50/0), back to 24a.
- **Schedule table** (`tools/eval/schedule_table.py`, uses `leg_in_air.py`): even offsets + duty 0.75 is the first schedule with 3 feet down at every front lift-off (the old offsets need 0.80). The cost: 0.20-0.25 s swings, so the servos lift a hind foot only ~1.4-1.8 cm. Slower strides give more time but longer steps; checked the knees stay >= 31 deg bent at +/-5.3 cm.
- **Run 26** (my decisions: even offsets, duty 0.75, 1.0 Hz for longer steps; floor 1.2 cm made **linear** -330 so going even a bit under costs; `foot_under_hip` measured from where the gait clock says the foot should be, so long centred steps are free but shifted ones aren't; swing arc 1.8 cm; unload off): front toe catches 1%, hind 4-5%, no early hind landings, hind lift 1.7-1.8 cm, knee 65% of rated (lowest yet), steps ~7-8 cm (24a ~6), 0/16 falls on the hardest lumps at 0.119 m/s, stress 1/16 at friction 0.2 floor-only, 0.7 and 0.9 m/s pushes. **Run 26 is the new default.**
- **Leftovers:** hind micro-taps (0.2 mm, 40 ms, ~27 in 10 s, after the other hind foot lands at 0.20-0.25 m/s); RF lands ~40 ms late; hind feet lift off ~1 cm early.
- **Bug caught:** `foot_under_hip` "clock" first cost *more* than "hip" on 24a. A foot landing early (stride progress 0.9) was compared to the lift-off spot 5.8 cm behind. Fix: a foot down outside its stance is judged from the nearer end of its stance (timing is gait_schedule's job).
- **Learned:**
  - Predict before training: list the policy's options, price them (look for free zones and squared prices: small misses are nearly free, and splitting a load across two feet halves a squared price), cross out what physics forbids, and pick the cheapest one close to what it already does. 25a and 25b were both predictable this way.
  - A torque balance (lever x force) settles "can a 12% prop hold the body?" in one line: no, it needs 50%.
  - A reward can't fix a problem the schedule builds in. Find the physical cause first.
- **Cost:** ~$0.75 (three runs incl. on-pod evals). Running total ~$9.90.

### Rearing runs rr1-rr6: from "stand and pay" to rearing up from a walking start
- **Did:** rearing env (`sim/red_panda_rear.py`): the policy owns 12 leg + 3 tail servos; tail and torso can touch the floor (sensors tell the tail tip from "sitting"); episodes start from states the run-26 walking policy actually produced (`tools/rear_starts.py`: standing and mid-walk); target = tripod option A (70 deg, hips 16 cm). Physics speed bound (`tools/eval/rear_speed_limit.py`): joint travel alone 0.23 s; the whole body pivoting 72 deg about the hind feet with perfect timing 0.39 s at stall, **0.44 s (1.8 kg) / 0.47 s (1.98 kg)** with torque capped at 65% of stall. Joint motion alone doesn't rear (front feet stay down, hind legs fold under), and the front legs can't throw (a straight front leg is only 1.3 cm longer).
- **Rewards (my decisions):** tilt, height, tripod_contact, tail_first, bad_contact; **"arrived"** = full tripod (contacts + tilt 70 +/- 5 + hips 16 +/- 1.5 cm) held 0.2 s, so speed can't be bought with height; **late -10/s, linear**, past a per-robot deadline (curriculum 2.0 -> 0.8 s, like Rudin's terrain levels); a fall before arriving pays the remaining time (no falling to stop the bill); **servo_pinned -1** (>= 95% of available torque), **over_peak -3** (per N*m above 65% of stall); walking's regularizers; **no clip at 0** (with a clip a -10 cost is no worse than -1). Later: **lean +3** = tilt/70 x height bell (a reward, not a "still down" penalty, so falling can't escape it).
- **Plumbing:** `train_walk.py --env rear` (300-step episodes; `num_resets_per_eval=0`, else Go1's resets wipe the per-robot curriculum every ~250 steps; training-robot metrics to `train_progress.csv`; `--restore` warm start); `tools/eval/rear_check.py` (arrival, servo load, cheat checks: ghost knees, tail under the body, tail landing speed; GIF; optional ladder rung).

  | Run | Change | From standing / mid-walk starts | What happened |
  |---|---|---|---|
  | rr1 | starter set | 0/64 | stood still, paid the full late bill. **Bug:** tail could only reach -29 deg, planting it from standing needs -35 |
  | rr2a / rr2b | tail range fixed / + tail_first off | 0 / 0 | 2a: tail as a kickstand, slams at 1.2 m/s; 2b: tail in the air |
  | rr3 | + lean 3 | 0 | never leaned (lean reward 0.00 all run) |
  | rr4b | cost curriculum x0.1 -> x1 (Hwangbo 2019) | 0 | thrashed 60x more early, still never leaned |
  | rr4a2 | reverse ladder, 25% of robots | 0 | placed in the tripod, it walked itself down in 0.4 s |
  | rr5a | rr4a2 + discount 0.99 | 0 | same as 4a2 |
  | rr5b | ladder, 100% of robots | 0 | **first rearing:** from 10-30 deg and 60-70 deg starts 15-16/16 in ~1 s; hole at 40-50 deg |
  | rr6a | warm start from 5b + bridge rungs (5 deg; standing + tail planted) | 4/64 | forgot the upper rungs |
  | **rr6b** | rr6a + 50% replay of passed rungs | **50/64, 0 falls** | median 4.1 s (fastest 1.24), tilt 70, hips 14.9 cm, every rung 9-16/16 |

- **Learned:**
  - A smooth reward doesn't help if the robot can't move smoothly toward it. Leaning past ~3 deg needs the front feet off the floor, and below ~55 deg (hips at 16 cm) the robot is unbalanced: a cliff, not a slope. PPO's fresh random noise every 20 ms averages out through slow servos; it never makes the coordinated push.
  - The rewards weren't the problem: a scripted hold in the tripod earns +5/s with every penalty ~0, and rr5b earns +23 per episode from 10 deg vs -40 standing. It was exploration and practice.
  - The discount sets how far ahead training credits an action: 0.97 (Go1's) looks ~0.67 s ahead, so a reward 2 s away counts 4.8%. Walking pays off within half a second; rearing pays 1-2 s after the effort. (rr5a showed the 25% mix was the bigger blocker, though.)
  - A shared network learns what most robots practise: 25% on the ladder -> the standing habit won; 100% -> it learned to rear.
  - A reverse curriculum (Florensa et al. 2017) forgets old rungs unless they are replayed (rr5b's 40-50 deg hole, rr6a's collapse).
  - Mac tests (CPU jax) don't catch GPU-only errors: `jax.lax.cond` around `mjx.forward` and `jax.tree.map` merges of physics state fail on Warp; use `Data.where` and smoke-test on the pod first.
  - My predictions: the tail slam (cheat #5) showed up in rr2a/rr3; ghost knees didn't.
- **Open:** speed (median 4.1 s vs 0.8 s target; the deadline curriculum barely runs for walking starts yet), hips 14.9 cm, right hind hip peak 2.18 N*m (over 65% of stall 2.3% of the time), success 78% (go/no-go needs 90%).
- **Cost:** ~$1.80 for rearing runs (EU-RO-1 GPUs scarce: often several minutes of retries). Running total ~$11.70.

### Rearing runs rr7a/rr7b: speed, and the stress test
- **Did:** the deadline curriculum in rr6b had stalled (average deadline ~1.4 s while arrivals got *slower*, 1.7 -> 2.6 s). Two reasons: one deadline per robot mixed very different starts (a 70 deg replay arrives in 0.3 s and tightens it, a walking start takes ~4 s and loosens it), and the 0.97 discount hides a late bill that starts seconds after the effort. Both runs warm-start from rr6b (ladder 100%, replay 50%, 300M steps):
  - **rr7a:** discount 0.97 -> **0.99** (look-ahead ~0.67 s -> ~2 s).
  - **rr7b (my idea, made fair per start):** `deadline_by_distance`: each episode's deadline = full budget x (70 deg - start tilt) / 70 (min 0.1 s), and the full budget ladders down from **4.0 s** (what rr6b already did) to 0.8 s.

  | From walking starts (64) | rr6b | **rr7a (0.99)** | rr7b (fair deadline) |
  |---|---|---|---|
  | Reached the tripod | 50 | **64** | 64 |
  | Median / fastest / slowest | 4.1 / 1.24 / 5.8 s | **0.87 / 0.74 / 1.16 s** | 1.83 / 1.18 / 3.40 s |
  | Within 0.8 s | 0 | **17** | 0 |
  | Hind hip peaks (99th pct; 65% of stall = 1.91 N*m) | 2.18 | 1.96 / 2.07 (over 1-2% of the time) | 1.69 / 1.83 |
  | Hips at the end (target 16) | 14.9 cm | 15.0 cm | 15.3 cm |

- **Stress test** (`tools/eval/rear_stress.py`, new): rr7a, 16 robots per condition from walking starts, one difference at a time. **All 20 conditions 16/16 reached, 0 falls**: friction 0.4 / 0.3 / 0.2 (past training), mass x0.9 / x1.1 / x1.2 (past training), centre of mass +/-2 cm, servo stiffness x0.7 / x1.3, delay 20 / 40 ms, pushes 0.2 and 0.4 m/s forward and backward while holding and 0.2 m/s backward while getting up (pushes never trained), all-at-once, and battery sag (servo torque and speed x0.9, never trained; slowest at 1.17 s). Only blemish: friction 0.2, one robot slipped out of the tripod by 6 s (no fall). Checked that the test can fail: a 2 m/s backward shove knocks 4/4 over; a 1 m/s forward shove doesn't.
- **Learned:**
  - The look-ahead decided speed: once the robot could rear (rr6b), discount 0.99 took it from 4.1 s to 0.87 s. In rr5a, before it could rear, 0.99 did nothing: change what limits it *now*.
  - A curriculum has to start where the learner is (4.0 s, not 2.0 s: a first rung you can't reach never gets climbed) and judge every start fairly. It helped (4.1 -> 1.8 s) but stalled at ~1.5 s with the short look-ahead.
  - Training randomization is practice (a random mix per robot); the stress test is the exam (one condition at a time, at and past the edges, plus things never trained).
  - Speed costs servo margin: rr7a's hind hips go over the 65%-of-stall rule 1-2% of the time; rr7b's slower rear-up stays under it.
- **Sim go/no-go so far:** rearing >= 90% with servo limits and randomization: yes (rough model). Torques: averages fine, peaks over the rule 1-2% of the time (hind hips). Caveats: 16 robots per condition is a small sample; still the box model, not CAD masses or the measured servo model.
- **Next:** bring the hind-hip peaks under 1.91 N*m (maybe 7a + the fair deadline), hips 15.0 -> 16 cm, then the transition back down and fall recovery.
- **Cost:** ~$0.49 (two runs). Running total ~$12.20.

### Rearing run rr8a: a straight tail and symmetric legs
- **Saw (from a straight-behind view I asked for):** rr7a held the tripod crooked. New `tools/eval/rear_symmetry.py` (64 robots holding the tripod): tail bent sideways (wag +27, wag2 +15 deg; tip 10-16 cm right of the centre line, pointing 20 deg right of straight back; it bends more toward the tip), left hind leg splayed 16 deg wider (hip_side 23 vs 7 deg), right hind hip working 21% harder (0.74 vs 0.61 N*m average; it also had the 2.07 N*m peak), centre of mass 2.2 cm right, roll +1.7 deg. Nothing in the rewards asked for symmetry, so the policy picked a side early and kept it. The joint angles (27 + 15) overstate the visible bend (20 deg): with the body up 70 deg and the tail lifted 50 deg, the "sideways" joints partly twist the tail instead. Hind knees rest perfectly straight on their 0 deg end stop (the real robot needs a hard stop there).
- **Did:** `rear_check.py` now renders a side view and a straight-behind view (camera square to the line between the hind feet, so a twisted torso shows instead of turning the camera). rr8a = warm start from rr7a + **tail_straight -2** (wag^2 + wag2^2) + **leg_symmetry -5** ((left - right hind hip_side)^2). Caught a sign slip in my own check (+2/+5 would pay it to be crooked): costs need negative weights.

  | | rr7a | **rr8a** |
  |---|---|---|
  | Tail wag / wag2; tip sideways | +27 / +15 deg; 15.7 cm | **+7 / +6 deg; 5.9 cm** |
  | Hind hip_side left / right | 23 / 7 deg | **12 / 14 deg** |
  | Hind foot load left / right | 7.4 / 8.1 N | **7.8 / 7.8 N** |
  | Hind hip torque left / right | 0.61 / 0.74 N*m | **0.61 / 0.65 N*m** |
  | Roll | +1.7 deg | **+4.1 deg (worse; up to 9 deg)** |
  | From walking starts: reached / median / within 0.8 s | 64 / 0.87 s / 17 | **64 / 0.78 s / 42** |
  | Stress test | all 16/16, 0 falls | all 16/16 except **0.4 m/s forward push: 3/16 fell** |

- **Learned:**
  - Look from the angle that shows the problem: the side view hid all of this.
  - Symmetry has to be asked for. Straightening the tail and legs moved the lean into the body (roll 1.7 -> 4.1 deg): fixing one part shifts the compensation somewhere else.
  - The bent tail was doing some balancing work after all: straight, it loses some forward-push tolerance.
  - Falls (my question): backward tips from the tripod are hard (the tail is a kickstand 35 cm behind; it took a 2 m/s shove) but would hit hard (~3-4 m/s, rough). Plan in CLAUDE.md: pushes in training, a bail-out reflex, go limp on falls, bumpers, breakaway tail. Mid-air righting like a cat isn't possible: the 160 g tail on slow servos turns the body only ~12 deg (roll) / ~4 deg (pitch) in a 20 cm fall.
- **Next:** rr9a = rr8a + roll guard; rr9b = rr8a + pushes in training.
- **Cost:** ~$0.24. Running total ~$12.45.

### Rearing runs rr9a/rr9b: roll guard vs. pushes in training
- **Did:** both warm-start from rr8a (straight tail, symmetric legs). **rr9a:** + `roll` -20 (roll^2, rad): rr8a had moved the lean into the body (4.1 deg). **rr9b:** + pushes in training (walking's random torso shoves, up to 0.5 m/s, every 1-6 s, any direction; the rearing step never called them before). The check tools now switch the env's random pushes off, so they judge the policy, not the luck of the shoves.

  | | rr8a | **rr9a (roll guard)** | rr9b (pushes) |
  |---|---|---|---|
  | Roll | +4.1 deg | **-0.1 deg (all robots within +/-2.2)** | **+21.8 deg** |
  | Hind foot load left / right | 7.8 / 7.8 N | 8.0 / 7.9 N | 6.2 / 9.3 N |
  | Hind hip torque left / right (rated 0.98) | 0.61 / 0.65 | **0.59 / 0.58** | 0.44 / 0.93 |
  | Tail tip sideways | 5.9 cm | **2.3 cm** | 14.8 cm |
  | Hips (target 16) | 15.3 cm | **15.8 cm** | 14.8 cm |
  | From walking starts: median / within 0.8 s | 0.78 s / 42 of 64 | 1.00 s / 0 | 0.84 s / 21 |
  | Hind hip peaks (65% of stall = 1.91) | 2.04 / 1.89 | 2.31 / 2.21 | 1.88 / 2.16 |
  | Stress: 0.4 m/s forward push while holding | 3/16 fell | **16/16** | **16/16** |
  | Stress: the rest | all pass | **all pass** | stiffness x0.7 14/16, all-at-once 15/16 |

- **Learned:**
  - rr9a is the cleanest pose yet (upright, symmetric, tail nearly straight, hips 15.8 cm), and standing straight alone won back the forward-push tolerance. It got slower (1.0 s) and pushes harder getting up (peaks 2.3 N*m).
  - rr9b: pushes from random directions with nothing against leaning -> it braced 22 deg sideways and put nearly all the hip work on one leg (0.93 N*m, ~rated). A cheap way to resist shoves, bad for the hardware. Same lesson again: the robot compensates wherever we didn't ask; pushes need the roll guard with them.
  - Speed and the hip-peak rule pull against each other now (rr8a fast but over; rr9a slower and further over).
- **Next:** rr9a + pushes (keep the roll guard); separately the hip peaks (e.g. rr9a + the fair deadline from rr7b).
- **Cost:** ~$0.55. Running total ~$13.00.

### Rearing runs rr10a/rr10b: pushes with the roll guard, and the fair deadline for the hip peaks
- **Did:** both warm-start from rr9a (symmetric; roll guard -20). **rr10a:** + pushes in training (up to 0.5 m/s, every 1-6 s). **rr10b:** + the distance-scaled deadline (full budget 1.5 s -> 0.8 s; 1.5 = just above rr9a's slowest arrival, so it starts where the robot is). The budget reached ~0.9 s.

  | | rr9a | rr10a (+ pushes) | **rr10b (+ fair deadline)** |
  |---|---|---|---|
  | From walking starts: reached / falls / median | 64 / 0 / 1.00 s | 64 / 0 / **0.94 s** | 64 / 0 / 0.98 s |
  | Roll (spread across robots) | -0.1 deg (+/-2) | -0.8 deg (+/-4) | **+0.2 deg (+/-2)** |
  | Hind foot load left / right | 8.0 / 7.9 N | 7.8 / 7.7 N | **8.0 / 8.0 N** |
  | Tail tip sideways | 2.3 cm | 1.4 cm | **1.0 cm** |
  | Hips (target 16) | 15.8 cm | 15.2 cm | **15.8 cm** |
  | Hind hip peaks, 99th pct (% of stall 2.94) | 2.31 / 2.21 (79 / 75%) | 2.02 / 2.15 (69 / 73%) | **1.96 / 1.95 (67 / 66%)** |
  | Stress test (20 conditions) | all pass | all pass | all pass |

- **Learned:**
  - With the roll guard on, pushes no longer make it brace sideways (rr9b leaned 22 deg; rr10a 0.8 deg).
  - The fair deadline, started where the robot is, lowered the hip peaks into the CLAUDE.md pass band ("peaks below ~60-70% of stall"; I had been checking a stricter 65% line) without losing speed.
  - rr10b is the best-behaved rearing policy so far; rr10a is the one that practised being shoved.
- **Next:** rr11 = rr10b + pushes in training (both together). Then the next Tier 1 piece: coming back down from the tripod.
- **Cost:** ~$0.53. Running total ~$13.55.

### Rearing run rr11: the rearing policy to keep
- **Did:** warm start from rr10b + pushes in training (both changes together). Full deadline budget 1.3 s -> 0.8 s (started just above rr10b's slowest arrival).

  | | rr10b | **rr11** |
  |---|---|---|
  | From walking starts: reached / falls | 64 / 0 | **64 / 0** |
  | Median / within 0.8 s | 0.98 s / 3 | **0.90 s / 8** (fastest 0.76) |
  | Roll | +0.2 deg | **-0.3 deg (all robots within +/-3)** |
  | Hind foot load left / right | 8.0 / 8.0 N | 7.8 / 7.6 N |
  | Tail tip sideways | 1.0 cm | **0.1 cm** |
  | Hind hip average / peaks (% of stall) | 0.58 N*m / 67, 66% | 0.67 N*m / **69, 67%** (inside the 60-70% band) |
  | Hips (target 16) | 15.8 cm | 15.0 cm |
  | Stress (20 conditions) | all pass | **all 16/16 reached, 0 falls**; friction 0.2: 12/16 still in the tripod at 6 s (slid out, no fall) |

- **Sim go/no-go for rearing (rough model):** >= 90% with servo limits + randomization: yes (64/64; 320/320 in the stress test). Torques: average 0.67 N*m (rated 0.98), peaks 67-69% of stall: pass. Speed: median 0.90 s vs the ~0.45 s physics limit.
- **Did also:** rr11's settings are now the env defaults (ladder 100% + 50% replay, fair deadline, symmetry + roll terms, pushes) and `train_walk.py --env rear` uses discount 0.99 unless `--ppo` sets it. Roadmap: rearing ticked.
- **Ops:** the copy-back watcher missed some logs once (scp hiccup); fetched them by hand before stopping the pod.
- **Next:** coming back down from the tripod onto four feet (Tier 1: rear up, hold, come back down).
- **Cost:** ~$0.44. Running total ~$14.00.

### Coming back down (dn_run1-2) and a solid robot body
- **Physics first (sim, joints held stiff in the tripod):** forward nudges up to 2 rad/s just rock it and it settles back; at 4 rad/s it topples and the front feet hit at **1.77 m/s** (walking feet land at 0.05-0.30). So coming down has two jobs: start it on purpose (shift the weight forward) and brake the fall.
- **Design (my choices):** a separate come-down policy (`sim/red_panda_down.py`, built on the rearing env); starts from 320 tripod states the rearing policy actually holds (`tools/down_starts.py`); "landed" = all four feet down + tail tip off the floor + walking stance (tilt -2 +/- 5 deg, hind hips 16 +/- 1.5 cm) held 0.2 s; rewards: stance bells, `lower` (progress down x height bell), `four_feet`, `front_landing` (each front touchdown costs its speed squared x 10), fixed 1.5 s deadline with the late bill, plus everything rearing has (servo costs, symmetry, roll guard, pushes). `tools/eval/down_check.py` also hands the final state to the walking policy (rp_run26) and has it walk 2 s.
- **Solid body:** my question "can the head feel the floor?" showed the head, neck, thighs, shins and hip blocks passed straight through the floor in the sim (only feet, torso and tail were solid). Now `solid_body` (on for rearing/coming down, off for walking so old runs replay unchanged) makes them collide with the floor only. The shin first touched the floor 99% of the time (its capsule ends inside the foot ball), so the collision shin now stops 3 cm above the foot. Checks: rearing rr11 and walking rp26 behave exactly the same with it on; false leg contacts 0%; a nose-first drop now stops the head at +5.3 cm instead of 5.8 cm inside the floor. New `head_contact` -5 (head/neck on the floor: the eye screens), legs added to `bad_contact`.

  | | dn_run1 (no ladder) | dn_run2 (reverse ladder 10 -> 70 deg, solid body) |
  |---|---|---|
  | What it learned | **froze in the tripod** (leaned back to 76 deg) | comes down, no falls |
  | Landed (all four feet) | 0/64 | 0/64: **keeps the right front paw up** (every robot) |
  | Hand-off to walking | 63/63 fell (never came down) | **64/64 walk off** |
  | Front landing speed | - | 0.18 m/s from 10 deg; **0.74 m/s from the real tripod** (2.7 bounces) |
  | Worst servo | - | right hind hip 2.90 N*m from the tripod (stall) |

- **Learned:**
  - dn_run1: early on any random move flipped it or face-planted it (episodes 0.12 s long), and a fall costs the same remaining late bill as staying, so "freeze" won. Same trap as rearing run 1; the scripted tests that landed fine were coordinated, random jitter isn't.
  - dn_run2: the all-or-nothing `four_feet` paid three feet the same as zero, so the fourth paw never had a reason to come down; "landed" was never met, the ladder never left its first rung, and it never practised from higher up. Same lesson as the late bill: graded beats all-or-nothing.
  - Check what the evaluation actually starts from: `down_check` first inherited the run's ladder and started on the 10 deg rung.
- **Next:** dn_run3 = continue from dn_run2 with a graded `feet_down` (feet down / 4) instead of `four_feet`.
- **Cost:** ~$0.40. Running total ~$14.40.

## 2026-11-17

### Coming down dn_run3-5: the paw that stayed up, a fresh start, and hind-hip strain
- **dn_run3a/b** (continue from dn_run2 with a graded `feet_down` = feet down / 4; 3b also ramps servo costs 0.3 -> 1): still **0/64 landed**, ladder stuck on its first rung. 3b had the softest landings (0.38 m/s).
- **Why (tests on the Mac):** every "landed" condition passed except the right front paw, 4.4 cm up on every robot (not a sensor bug). Holding all four feet at home scores **+8.5/s vs the policy's -2.7/s**, and the policy lifts the paw even when started on four feet. It balanced on a lopsided triangle (left front hip-side at its -20 deg limit under the body, right front swung out 24 deg and folded, one hind knee straight); moving one leg down alone made things worse (-10/s), so PPO's small steps couldn't get out. dn_run2 invented the habit when the fourth paw paid nothing, and `--restore` carried it over. The 10 deg start rung is symmetric (both front paws 5.1 cm up).
- **dn_run4a/b: from scratch** (graded `feet_down`, cost ramp 0.3 -> 1), 4b with a new `front_symmetry` -5 ((left - right front hip-side)^2 + (left - right front knee)^2):

  | From the real 70 deg tripod | dn_run3b | dn_run4a | dn_run4b (+ symmetry) |
  |---|---|---|---|
  | Landed / falls | 0/64 / 0 | **64/64** / 0 | **64/64** / 0 |
  | Landing median (slowest) | - | 1.02 s (1.68) | 0.90 s (1.10) |
  | First front touchdown median / worst | 0.38 / 0.68 m/s | 0.21 / 0.30 | 0.19 / 0.33 |
  | Training ladder at the end | rung 0 | rung ~4.4 of 6 | rung 6 (top), 90% landing |
  | Front hip-side L / R, hind knees L / R | -20/+24, 36/0 deg | +20/+6, 13/37 | +19/+23, 17/19 |

- **Strain, measured every 4 ms** (new `tools/eval/down_load.py`; the other checks sample every 20 ms): feet hit the floor with 1.1-1.9 body weights, **half of a walking step** (3.2-3.9). But 4b's hind hips sat at stall for 56-80 ms (max 148) and were **back-driven** (braking at full torque while the joint still turns backwards, up to 4.7 rad/s = the gear-tooth case) for up to 56 ms. Walking never back-drives.
- **dn_run5a/b** (continue from 4b, full costs): 5a `over_peak` -3 -> -15; 5b also a new `backdriven` -2 (|speed| of each servo braking at >= 90% of stall).

  | 1.8 kg | dn_run4b | dn_run5a | **dn_run5b (kept)** |
  |---|---|---|---|
  | Landed / falls | 64/64 / 0 | 64/64 / 0 | 63/64 / 0 (one took 3.6 s) |
  | Landing median | 0.90 s | 0.93 s | 1.20 s |
  | Hind hips over 65% stall, median / max | 56-80 / 148 ms | 26-42 / 92 | 28-34 / 88 |
  | Back-driven, max | 48-56 ms | 16-32 ms | **8-12 ms** |
  | Touchdown median / worst | 0.19 / 0.33 | 0.20 / 0.34 | 0.17 / 0.25 m/s |

- **When (`down_load.py --timeline`):** two moments. Push-off at 0-0.1 s (hips + tail shove the body off the stable tripod, ~13 ms over 65%, same in every run) and braking while tipping from 37 to 9 deg before the front paws arrive (4b 57 ms + back-driving; 5b 22 ms, ~none).
- **Learned:**
  - A restored policy can be stuck in a habit the reward no longer wants. When the better pose needs several joints to change at once, start fresh.
  - Check what fails part by part before changing rewards: "landed" failed on exactly one condition, and the reward already preferred four feet.
  - Sampling every 20 ms hid the stall spikes (99th percentile 2.1-2.2 N*m vs 2.9 every 4 ms).
  - 5x a cost only halved the strain; a cost aimed at the actual harm (back-driving) did more.
- **Decision (mine):** keep dn_run5b; its settings are the come-down defaults. Recheck the hip strain with the BAM servo model in January (how a real STS3215 brakes when forced backwards is what the sim guesses).
- **Ops:** EU-RO-1 had no 4090/5090 for hours; RTX PRO 4500 Blackwell ($0.72/h) works with the same venv (13.4 min per 200M steps vs ~11 on a 4090). Retrying every 3 minutes got GPUs. Once three requests sent as fallbacks all succeeded; the extra pod was deleted within a minute.
- **Next:** the full Tier 1 chain in sim: walk -> rear -> hold -> come down -> walk, switching the moment each skill reports done.
- **Cost:** ~$1.75. Running total ~$16.15.

### The whole Tier 1 chain in sim (chain_check)
- **Did:** `tools/eval/chain_check.py` runs walk (rp_run26) -> rear (rr_run11) -> hold -> come down (dn_run5b) -> walk, switching like the robot's switcher will: each hand-off copies the exact pose and speeds into the next policy's env with every servo holding its current angle. "Rear" is pressed while walking at 0.12 m/s; "down" after ~3 s of holding; each robot switches to walking the moment it lands. Nominal robot (flat, no shoves, delay fixed 20 ms, no calibration error) so only the switching is tested.

  | Phase (32 robots) | Result |
  |---|---|
  | Walk 3 s | 0 falls, 32 cm (ideal 36) |
  | Rear up from walking | 32/32 arrived, median 0.84 s (slowest 0.94) |
  | Hold | 32/32 in the tripod at "down", tilt 73 deg |
  | Come down | 32/32 landed, median 1.34 s (slowest 1.52), 0 falls |
  | Walk off at landing | 0 falls, 35 cm in 3 s (ideal 36) |
  | **Whole chain** | **32/32** |

- **Learned:** handing over to walking the instant "landed" is true works; the walking policy fixes the splayed stance in its first steps. That correction is a small jolt: hind-hip peak in the first 0.5 s median 1.83 N*m, worst 2.55 (87% of stall; steady walking ~2.0). Coming down from where rearing actually ended (73 deg) takes 1.34 s vs 1.20 from the saved start poses.
- **Did also:** README now shows the chain GIF (`media/tier1_chain.gif`, 2.2 MB, cut down from 11 MB) and the milestone.
- **Next:** the same chain on a randomized robot (mass, delay, calibration, shoves); turning on command with the current walking policy.

### The chain on randomized robots (chain_check --random)
- **Did:** `chain_check.py --random`: each robot gets one random world from training's own randomizer (friction 0.3-1.0, servo stiffness x0.7-1.3, mass 1.62-1.96 kg, torso balance point +/-2 cm), its own servo delay (0-40 ms) and calibration error (+/-2 deg), **kept the same through all three policies** (a real robot's servos don't change between skills), and each policy's training shoves are on.

  | Phase (32 robots) | Nominal | Randomized + shoves |
  |---|---|---|
  | Walk 3 s | 0 falls | 0 falls |
  | Rear up from walking | 32/32 | 32/32, median 0.88 s |
  | In the tripod at "down" | 32/32 | 30/32 (2 shoved to 63-65 deg; both still landed) |
  | "Landed" within 3 s | 32/32 | 29/32 |
  | Falls coming down / walking off | 0 / 0 | **0 / 0** |

- **Why 3 missed "landed":** all four feet down and tilt fine, but the hind hips at 14.6-14.7 cm, right on the edge of the 14.5-17.5 cm box (one also had its tail tip still down). dn_run5b's stance sits low (14.8 cm vs 16; dn_run4b 15.3), and a heavy robot, soft servos or a slippery floor sag it a few mm further. All 3 walked off fine when handed over at the 3 s timeout.
- **Learned:** the training "landed" box is stricter than what walking needs. Test failures need a "why": here the robot was fine and the definition was the problem.
- **Decision (mine):** the switcher hands over to walking on "ready to walk" = 4 feet down + tail up + body still for 0.2 s (no tilt/height limits), with a timeout fallback. Fix the low, splayed end stance later in a polish run.
