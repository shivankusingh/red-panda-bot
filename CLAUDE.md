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

- I won't commit code I can't explain line by line. If I accept code too quickly, slow me down and quiz me.
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
| 2: polish | Fall recovery; idle behaviors; reactions; measured results |
| 3: stretch | Free stand with tail balancing; voice commands |

Training order: walking → rearing into the tripod → fall recovery → free stand.

## Key decisions

| Topic | Decision |
|---|---|
| Rearing sequence | Plant the tail, move the center of mass over the hind feet, then lift the front feet |
| Policy structure | Separate policies plus a switcher for Tier 1; rearing training starts from the walking policy's standing poses |
| Tail control | Animation-driven sway (with follow-through) while walking and idle; RL policy controls it in tripod and free stand |
| Personality scope (Tier 1) | Head, ears, and eyes only; legs and tail belong to the policy |
| Fur | None. Printed colored shell with a ringed tail |
| Servos | STS3215 C018 (12 V) for all legs and the tail. Compare loads to rated torque, not stall |
| Power | One 3S LiPo for servos; separate 5 V / 5 A regulator for the Pi; shared ground |
| Bus | Legs, tail, and neck on the serial bus; ears on PWM |
| Rough-model body | 32 × 10 × 5 cm, hips ±12 cm front/back of center; Pupper legs (12.35 + 11.5 cm). Later sim test: Pupper's 27.6 cm body for rearing comparison |
| Rough-model mass | 1.8 kg nominal, a bit above the parts estimate (~1.64–1.74 kg, servos ≈ 55 g each × 18) so the torque checks have margin. Train with mass randomized ±10% (1.62–1.98 kg); report torques at 1.8 kg (go/no-go) and 1.98 kg (stress test) |
| Neck | 3 servos: turn, then nod, then tilt (head tilt for personality). Like the tail while walking: animation-driven, randomized "looking around" during training (switchable), and the walking policy sees the neck angles and speeds |
| Battery placement | Center of the torso for the rough model; front/back is a later sim experiment |
| Tail joints | 3 servos: 2 together at the tail base (side to side, then up/down) + 1 side-to-side at 11 cm along for an S-curve wag. Tail is 27.5 cm, light and rigid between joints. Later sim tests: tail length 22.5 / 27.5 / 32.5 cm; mid-tail servo locked (2-servo tail) vs. active |
| Feet | Identical legs with bolt-on feet (test point feet vs. longer hind feet) |
| Leg geometry | Borrowed from the original Stanford Pupper (hobby-servo version); tail, body, head original |
| Knee direction | Animal-like: front knees point backward, hind knees point forward (hind knees stay clear of the ground and tail when the body tilts up for the tripod). Later sim test: all knees backward (Go1-style) |
| Joint sign convention | Right-hand rule. Right legs' hip-side axes are mirrored (`-1 0 0`) so + = outward on every leg; flip those servo signs on the real robot |
| CAD | Onshape |
| Simulator | MuJoCo Playground (Open Duck Mini + BAM servo model already work there). Bake-off vs. Isaac Lab skipped; a short Go1 example run checks the setup instead |
| Compute | Rented cloud GPU (~$10–40/month); university GPU cluster as backup. Shut machines down when idle |
| Control code | Python first; C++ once the Python version walks |
| Teacher-student | Only if it measurably helps; try asymmetric actor-critic first |

## Hardware plan

- 12 leg servos (3 per leg) + 3 tail joints (with a mechanical hard stop; breakaway tip) + neck servos on the bus + ear servos on PWM.
- Raspberry Pi 5 runs the policy onboard (ONNX, 50 Hz or faster). IMU for orientation and angular velocity (use the projected gravity vector, not Euler angles). Two small eye screens.
- Target mass ~1.5–1.65 kg. Weigh every printed part and update the model.
- Torque check in sim: servo torque–speed curve as the joint limit; log peak and average torque per joint. Pass: average below rated, peaks below ~60–70% of stall. Likely worst case: hind knees in a deep crouch.
- Servos cut torque above ~70°C: read temperature over the bus and leave the tripod at ~60°C.
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
- [ ] Rough box-and-cylinder red panda model with 12 V servo limits (on the Mac)
- [ ] First GPU session: run the Playground Go1 example ~10 min and compare its reward curve to the published one (setup check), then train the red panda
- [ ] I can explain every reward term (read Rudin 2021 alongside)
- [ ] Walking + turning on the rough model (budget 4–6 weeks)
- [ ] Rearing into the tail tripod; joint torques logged
- [ ] Winter break: servo test rig (2–3 servos), test-printed leg, bus timing measurements

**Go/no-go for building (~Jan–Feb 2027):** rearing ≥ 90% success in sim with realistic servo limits and randomization; torques pass; servo model matches rig data; bus supports 50 Hz; budget available.

**Phase 2: hardware (if built)**
- [ ] Onshape CAD → sim with real masses; retrain
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
