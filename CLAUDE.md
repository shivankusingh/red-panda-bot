# red-panda-bot

A small, custom-built red panda robot that walks on four legs, rears up into a tripod stance on its hind legs and tail, recovers from falls, and (stretch) balances on its hind legs using its tail as a counterweight. All movement skills are learned with reinforcement learning (RL) in simulation and transferred to real hardware. It has a cute, expressive personality: screen eyes, moving ears, head motion, idle animations, reactions.

**The twist:** the tail is a third support point in the tripod stance and a swinging counterweight in the free stand. Everything else (parallel PPO, domain randomization, servo modeling, teacher-student) is a known method that we rebuild and credit properly.

**Project goal:** a project that is strong on the software side, aimed at robotics internships (target: summer 2028 applications, Aug–Oct 2027).

## How to work with me (most important)

The goal is that I deeply understand the parts that matter, not that I type every line. **Hybrid approach:** learn deeply where mistakes are silent or costly; move fast everywhere else. Keep momentum: get something training early and learn from what breaks.

**Learning mode:**
- For software and RL: **run it first, then take it apart.** Get something working (e.g. port Go1's rewards as-is and train), then explain what matters by looking at the results. Don't front-load theory.
- For physics, sensors, and the real robot: **explain first, but briefly.** A wrong mental model there costs a print or a servo.
- Go deep only where it bites: reward math, physics, what the policy sees vs. what the real robot can measure, sim-to-real. Skim the rest with a short summary.
- Keep check-ins short: one quick question per step (predict, explain back, or a small change), not several.
- Point out important design decisions and let me make them. Prefer small, readable code over clever code.

**Build mode** (only after I say so, e.g. "switch rewards to build mode"): write code from my prompts at normal speed. Still give a short summary of what changed and why, and flag anything surprising or risky.

| Area | Mode |
|---|---|
| Robot model (bodies, joints, sensors, home pose) and randomization | Learning (brief) |
| Reward functions and training (reward math, curriculum, servo model) | Learning (run first) |
| Env and training plumbing (env boilerplate ported from Go1, PPO config files, viewers, run scripts) | Build |
| On-robot control code (real-time loop, servo control, policy runtime) | Learning |
| CAD and mechanical design | Learning |
| Personality and animation | Learning |
| Tooling (logging, plotting, evaluation, converters, setup scripts, test harnesses) | Build |

Update this table when I switch an area.

**Decisions are mine:** reward terms and weights, robot dimensions and masses, servo choices, anything that changes how the robot behaves. Propose options with tradeoffs and a recommendation, then let me choose.

**About me:** first-year undergrad, mostly software experience, new to physical robots, mechanical design, and RL. Comfortable with vectors, derivatives, PyTorch, and C++; take more care with rotations (quaternions, frames). I learn best visually: diagrams, tables, traced examples with actual numbers.

- Before committing, I can explain the design and the mechanisms: every reward term and why it exists, what the robot observes, the physics limits, and why each fix worked. Plumbing and tooling can stay a black box. If I accept a mechanism too quickly, slow me down and quiz me on it.
- When I'm confused, re-explain differently instead of adding new terms. Define any new term in plain words.
- Ask before reorganizing files, deleting anything, changing robot dimensions or masses, or changing reward math.
- Get torque and load numbers from simulation, not hand calculations alone.
- **Effort suggestions:** end each response with the `/effort` level suggested for the next step (low: tooling and chores; medium: most learning-mode work; high: reward design, physics, robot model, non-obvious debugging; xhigh: stuck problems). If a new request needs a higher level than the current one, say so at the start before diving in.

## Git rules

- This repo is public. Never commit secrets, API keys, passwords, personal information, or code/data from outside sources I haven't confirmed I'm allowed to share.
- No AI attribution in commits: no Co-Authored-By lines, no "Generated with" footers, no session links.
- Before committing: show `git status`, explain what's being committed, and wait for my OK before `git push`.
- Short, descriptive commit messages (e.g. "Add tripod stance reward terms").
- Credit every paper, open-source design, and dataset in the README.
- Don't commit large files (big checkpoints, videos, CAD exports) unless I say so. Small exported policies (ONNX, under a few MB) are fine.

## Skills and tiers

| Tier | Skills |
|---|---|
| 1: the project | Walk and turn on gamepad command; rear up into the tail tripod, hold it, come back down; eyes blink and react. On the real robot. |
| 2: polish | Fall recovery; idle behaviors; reactions; measured results; walking over small bumps, cables and rug edges (blind, rough-terrain training) |
| 3: stretch | Free stand with tail balancing; voice commands; climbing over objects (slow high steps over ~3-5 cm things; climbing onto a low ~5-8 cm ledge: front paws up, shift weight, hind legs up; needs the depth camera) |

Training order: walking → rearing into the tripod → fall recovery → free stand.

## Key decisions

| Topic | Decision |
|---|---|
| Rearing sequence | Plant the tail, move the center of mass over the hind feet, then lift the front feet |
| Tripod target pose | ~70° nose-up tilt, hind hips 16 cm up, hind feet ~7 cm ahead of the hips, tail lying back along the floor (option A). Static hold at 1.98 kg: hind hip 0.70, knee 0.27, tail lift 0.57 N·m (all below rated); CoM 3.2 cm inside the feet+tail triangle (~7° forward rock before it tips onto the front feet). Worst servo in the tripod is the hind hip (feet ahead of the hips), not the knee. Getting up passes 30–50° tilt where the hip is over rated (1.0–1.3 N·m) briefly. From `tools/eval/tripod_static.py` |
| Rearing training | Rewards: tilt, height, lean (tilt/70 x height bell), tripod contact, tail-first, sitting; late -10/s linear past a per-robot deadline (2.0 -> 0.8 s) after "arrived" (full tripod held 0.2 s); servo pinned / over 65% stall costs; no reward clip. Learned with a reverse ladder of start poses (70 deg -> standing, incl. bridge rungs), 100% of robots on it, 50% replay of passed rungs, warm-started; PPO discount 0.99 (rearing pays off 1-2 s after the effort). rr7a: 64/64 from walking starts, median 0.87 s; stress test (`tools/eval/rear_stress.py`) 16/16 in all 20 conditions, 0 falls. Symmetry is asked for explicitly: tail_straight -2, leg_symmetry -5 (rr8a: tail tip 16 -> 6 cm off centre, hind loads equal, median 0.78 s; check with `tools/eval/rear_symmetry.py` and the straight-behind GIF); roll -20 keeps the body upright (rr9a: roll -0.1 deg, loads and hip torques equal, all stress conditions pass, median 1.0 s). Pushes in training need the roll guard: without it rr9b braced 22 deg sideways. **Kept: rr_run11** (rr10b + pushes; its settings are the env defaults): 64/64 from walking starts, median 0.90 s, symmetric, hip peaks 67-69% of stall, all 20 stress conditions pass |
| Coming down | Separate policy from the tripod onto four feet (`sim/red_panda_down.py`), starts from tripod states rearing actually holds, reverse ladder 10 -> 70 deg. "Landed" = 4 feet down, tail up, walking tilt and height, held 0.2 s. Rewards: stance bells, lower, graded feet_down (all-or-nothing let it keep a paw up), front_landing (touchdown speed^2), front_symmetry -5, over_peak -15, backdriven -2, 1.5 s deadline. A restored policy can't unlearn a lopsided 3-leg stance: retrain from scratch. **Kept: dn_run5b** (4b from scratch with cost ramp 0.3 -> 1, then continued): 63/64 landed, 0 falls, median 1.2 s, front paws touch at 0.17 m/s (walking steps hit ~2x harder), hand-off to walking 64/64. Strain (`tools/eval/down_load.py --timeline`, every 4 ms): hind hips over 65% stall ~30 ms per landing (push-off from the tripod + braking the forward tip), back-driving max 8-12 ms. Recheck with the BAM servo model (real braking when forced backwards is the uncertain part) |
| Policy structure | Separate policies plus a switcher for Tier 1; rearing training starts from the walking policy's standing poses. Switcher hand-off to walking after coming down: "ready to walk" = 4 feet down + tail tip up + body still (< 5 cm/s, < 0.5 rad/s) for 0.2 s, no tilt/height limits (the training "landed" box is stricter than walking needs), 3 s timeout fallback. Full chain (`tools/eval/chain_check.py`): 32/32 nominal; randomized + shoves 30/32 ready, 0 falls in any phase |
| Tail control | Animation-driven sway (with follow-through) while walking and idle; RL policy controls it in tripod and free stand |
| Personality scope (Tier 1) | Head, ears, and eyes only; legs and tail belong to the policy |
| Fur | None. Printed colored shell with a ringed tail |
| Servos | STS3215 C018 (12 V) for all legs and the tail. Compare loads to rated torque, not stall |
| Power | One 3S LiPo for servos; separate 5 V / 5 A regulator for the Pi; shared ground. Open question: a seller lists the STS3215 12 V at 4.0–12 V, but a full 3S is 12.6 V. Check Feetech's C018 datasheet before designing the power board (options: don't charge to full, or a 12 V regulator on the servo bus that can take the current spikes) |
| Bus | Legs, tail, and neck on the serial bus; ears on PWM |
| Rough-model body | 32 × 10 × 5 cm, hips ±12 cm front/back of center; legs = Pupper ×0.7 (8.64 + 8.05 cm, run 14b), standing with knees bent 45° and feet under the hips (torso ~16 cm, red panda height). Final lengths depend on fitting the servos in CAD. Later sim test: Pupper's 27.6 cm body for rearing comparison |
| Walking gait | Lateral-sequence walk (LH → LF → RH → RF), evenly spaced, each foot down 75% of a 1.0 s stride: always 3 feet down (no same-side 2-leg moments), steps ±4.5 cm at 0.12 m/s, hind lift ~1.5–1.8 cm (servo-limited), hard floor 1.2 cm (run 26) |
| Rough-model mass | 1.8 kg nominal, a bit above the parts estimate (~1.64–1.74 kg, servos ≈ 55 g each × 18) so the torque checks have margin. Train with mass randomized ±10% (1.62–1.98 kg); report torques at 1.8 kg (go/no-go) and 1.98 kg (stress test) |
| Neck | 3 servos: turn, then nod, then tilt (head tilt for personality). Like the tail while walking: animation-driven, randomized "looking around" during training (switchable), and the walking policy sees the neck angles and speeds |
| Battery placement | Center of the torso for the rough model; front/back is a later sim experiment |
| Tail joints | 3 servos: 2 together at the tail base (side to side, then up/down) + 1 side-to-side at 11 cm along for an S-curve wag. Tail is 27.5 cm, light and rigid between joints. Later sim tests: tail length 22.5 / 27.5 / 32.5 cm; mid-tail servo locked (2-servo tail) vs. active |
| Feet | Identical legs with bolt-on feet (test point feet vs. longer hind feet) |
| Leg geometry | Borrowed from the original Stanford Pupper (hobby-servo version); tail, body, head original |
| Knee direction | Animal-like: front knees point backward, hind knees point forward (hind knees stay clear of the ground and tail when the body tilts up for the tripod). Later sim test: all knees backward (Go1-style) |
| Knee drive (A2) | Knee servo screwed to the top of the thigh, right at the hip axis (it swings with the thigh), driving the shin through a parallelogram rod (crank = lever on the shin). It sets the knee bend, exactly like the sim's knee joint, so the trained policies and torque checks carry over. Rejected (A): knee servo fixed to the body, coaxial with the hip: it sets the shin's direction and the hip servo then holds the foot's push on the thigh (hip servo = hip - knee torque; standing 0 -> 0.19 N·m; the trained motions would need hip peaks 4.5 N·m > 2.94 stall and knee servo speeds up to 7.6 > 4.7 rad/s; `tools/eval/chain_check.py --legs`). Knee bend used: front 29-77°, hind 0-60°; design for at least 0-110°. Linkage (2026-11-19, `cad/knee_linkage.py`): crank = lever = **20 mm**, **M4** threaded rod (a turnbuckle lets the rod length be tuned to the thigh's 86.45 mm) with ball links. Lever on the side the knee points to (like a kneecap: weight-bearing pulls the rod), ~50° off the thigh line at a straight knee (psi0 = 130°, >= 20° from the dead point over 0-110°). Rod force = knee torque / (lever x sin(rod-to-lever angle)). Sim chain (nominal + randomized): p99 74 N, peaks 161 N pull / 181 N push (the knee torque flips, so the rod is pushed as hard as pulled); servo-stall bound over the bends used 192 N; M4 buckles at ~1218 N for an ideal rod (6.3x; M3 would be 2.1x). Open: the knee servo's output shaft takes the same force sideways; check on the rig |
| Joint sign convention | Right-hand rule. Right legs' hip-side axes are mirrored (`-1 0 0`) so + = outward on every leg; flip those servo signs on the real robot |
| CAD | Onshape |
| Simulator | MuJoCo Playground (Open Duck Mini + BAM servo model already work there). Bake-off vs. Isaac Lab skipped; a short Go1 example run checks the setup instead |
| Compute | Rented cloud GPU (~$10–40/month); university GPU cluster as backup. Shut machines down when idle |
| Control code | Python first; C++ once the Python version walks |
| Jumping | Ruled out for STS3215 servos. Sim (run 15 era, legs x0.7, 1.8 kg): a full-effort crouch-and-extend leaves the ground for 40 ms = ~2 mm. Torque is enough; speed isn't (without the speed limit: 218 ms = ~6 cm). Jumping needs fast direct-drive motors (e.g. Mini Cheetah class) |
| Depth camera (candidate, Tier 2/3) | For stepping over obstacles (blind policy only recovers after bumping). Mount in the lower front corner of the torso, angled ~35° down, behind a dark tinted window (reads as the dark chest fur). Not on the head: it would nearly double the neck nod's holding torque, and a moving head makes the height map harder. Shift the battery back to balance it. Leave room in the torso CAD. Add after Tier 1 works on the real robot; train with height maps + teacher-student |
| Voice (Tier 3) | Commands + simple conversation. Answers with creature sounds and expressions (eyes, ears, head), not spoken words (option A). Pi 5 does wake word + a fixed command list offline; conversation goes over WiFi to laptop/cloud (speech-to-text -> LLM). The LLM only picks from a fixed menu (sound, emotion, animation, switcher skill); it never drives joints, and the switcher's safety checks still apply. No Jetson unless a measured need appears (2026-11: Orin Nano Super $399, adds mass and up to ~25 W). Leave room in the CAD now for a mic array (away from the leg servos) and a small speaker |
| Teacher-student | Only if it measurably helps; try asymmetric actor-critic first |

## Hardware plan

- 12 leg servos (3 per leg) + 3 tail joints (with a mechanical hard stop; breakaway tip) + neck servos on the bus + ear servos on PWM.
- Raspberry Pi 5 runs the policy onboard (ONNX, 50 Hz or faster). IMU for orientation and angular velocity (use the projected gravity vector, not Euler angles). Two small eye screens.
- Target mass ~1.5–1.65 kg. Weigh every printed part and update the model.
- Torque check in sim: servo torque–speed curve as the joint limit; log peak and average torque per joint. Pass: average below rated, peaks below ~60–70% of stall. Likely worst case: hind knees in a deep crouch.
- Servos cut torque above ~70°C: read temperature over the bus and leave the tripod at ~60°C.
- Falls and protection: the tail tripod makes backward tips hard (a 2 m/s shove in sim), but a backward topple from the tripod hits at ~3-4 m/s (rough estimate; check in sim). Plan: train rearing with pushes (incl. backward); a bail-out reflex in the switcher (past ~85° tilt or tipping back fast -> drop forward onto the front feet); servos go limp when a fall is detected; breakaway tail tip + hard stop; soft TPU bumpers at impact points (back of torso, top of head, tail base); head tucks forward, eye screens behind a recessed clear cover; hind-knee hard stop at 0° (the rearing policy rests straight knees on it).
- Safety: gamepad emergency stop that cuts servo torque; first real tests hanging or on a stand; LiPo low-voltage alarm; charge outside the robot.
- Budget ~$600–850 for hardware, plus cloud compute. Ask me before recommending anything expensive.

## Methods to replicate, then extend

1. Parallel PPO locomotion: Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning" (2021). Reproduce a standard quadruped example first.
2. Servo model: Hwangbo et al., "Learning agile and dynamic motor skills for legged robots" (Science Robotics, 2019), adapted via Rhoban's BAM (pendulum-rig identification). Model the servo's internal P controller and match Kp to firmware; randomize action delay.
3. Teacher-student: Lee et al., "Learning quadrupedal locomotion over challenging terrain" (Science Robotics, 2020); Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots" (2021).
4. Fall recovery: Lee et al., "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning" (2019).
5. Optional: Margolis & Agrawal, "Walk These Ways" (2022).
6. Personality: Grandia et al., "Design and Control of a Bipedal Robotic Character" (Disney Research, 2024).
7. Tail experiment background: Libby et al., "Tail-assisted pitch control in lizards, robots and dinosaurs" (Nature, 2012).

Closest existing project: Open Duck Mini (STS3215 servos, Raspberry Pi, MuJoCo Playground, BAM).

## Roadmap

**Phase 1: simulation (Nov 2026 → ~Feb 2027)**
- [x] Rough box-and-cylinder red panda model with 12 V servo limits (on the Mac)
- [x] First GPU session: run the Playground Go1 example ~10 min and compare its reward curve to the published one (setup check), then train the red panda
- [ ] I can explain every reward term (read Rudin 2021 alongside)
- [x] Walking + turning on the rough model (rp_run26; `tools/eval/turn_check.py`: turns in place and on curves at 81-97% of the asked rate, 0 falls in 8 commands x 32 robots, nominal and randomized. Later polish: slight left bias, slow backing up)
- [x] Rearing into the tail tripod; joint torques logged (rr_run11, rough model)
- [x] Coming back down (dn_run5b) and the full Tier 1 chain in sim: walk -> rear -> hold -> come down -> walk (`tools/eval/chain_check.py`: 32/32 nominal, randomized + shoves 30/32 ready to walk, 0 falls)
- [ ] January 2027 (winter break is vacation; the printer is on campus): servo test rig (2–3 servos), test-printed leg, bus timing measurements. Parts, shipped to campus: 6x STS3215 12 V / 30 kg·cm (C018; spec 0.222 s/60° = 4.72 rad/s matches the sim), a bus adapter (FE-URT-1 or Waveshare Bus Servo Adapter (A)), a 12 V ≥ 5 A bench supply

**Go/no-go for building (~Jan–Feb 2027):** rearing ≥ 90% success in sim with realistic servo limits and randomization; torques pass; servo model matches rig data; bus supports 50 Hz; budget available.

**Phase 2: hardware (if built)**
- [ ] Onshape CAD → sim with real masses; retrain (started 2026-11-19: sim numbers as Onshape variables, knee rod linkage chosen)
- [ ] Parts ordered; printed, assembled, wired; servo IDs set; every servo tested
- [ ] Sim-vs-real joint-angle overlay check
- [ ] Real walking with gamepad (target: end of spring 2027)
- [ ] Real tail tripod (target: end of summer 2027)
- [ ] Fall recovery; personality (eyes, ears, head, idle behaviors, reactions)

**Phase 3: polish (summer 2027)**
- [ ] Results table and tail experiment
- [ ] C++ control loop
- [ ] Free stand (stretch)
- [ ] README, technical write-up, 60-second demo video (by ~Aug 2027)

## Measure everything

- **Results table:** remove one technique at a time from the full recipe (domain randomization, servo model, delay randomization), plus servo model none vs. BAM vs. learned. Fall rate = falls over 20 trials of 30 s, with spread.
- **Tail experiment:** tail locked vs. active on push recovery, fall recovery, rearing success; focus on fast swings.
- **Sim-vs-real plot:** same command sequence on both, overlay joint angles.
- **Python vs. C++ loop timing chart.**
- Also: walking speed, turning rate, tripod success and hold time, recovery success, policy inference time on the Pi.
- Onboard logging of every observation and action at 50 Hz, replayable in sim.
- Real-world tracking: overhead webcam + AprilTag on the robot's back.

## Machines

- **Mac:** coding, CAD, tooling, plotting, docs; MuJoCo for viewing policies.
- **Lenovo Legion (Windows, GTX 1650 or 1660 Ti):** can't run Isaac Sim (no RT cores). Not used for main training.
- **Cloud GPU:** training. Setup notes go here once known.
- `git pull` when switching machines.

## Repo layout (proposed; ask before changing)

- `CLAUDE.md`: this file
- `README.md`: public project page with credits
- `cad/`: CAD files and scripts
- `sim/`: simulation environments, robot model, rewards, training configs
- `robot/`: on-robot code (control loop, servo driver, policy runtime, personality)
- `tools/`: logging, plotting, evaluation, converters
- `data/`: recorded servo data, small results (large files ignored by git)
- `logs/log.md`: day-by-day log (what I did, what I learned, mistakes fixed, decisions and why, next steps)
