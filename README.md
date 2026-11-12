# red-panda-bot

A small red panda robot that walks on four legs, rears up into a tripod stance using its tail as a third support, and recovers from falls. All movement skills are learned with reinforcement learning in simulation and transferred to real hardware.

**Status:** Phase 1, simulation (started Nov 2026).

## Repo layout

| Folder | Contents |
|---|---|
| `sim/` | Simulation environments, robot model, rewards, training configs |
| `robot/` | On-robot code: control loop, servo driver, policy runtime, personality |
| `tools/` | Logging, plotting, evaluation, converters |
| `cad/` | CAD files and scripts |
| `data/` | Recorded servo data and small results |
| `logs/log.md` | Day-by-day build log |

## Credits

This project rebuilds known methods; the tail is the new part.

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning" (2021)
- Zakka et al., "MuJoCo Playground" (2025), with Brax's PPO; the Go1 joystick example is the starting point for the rewards
- Hwangbo et al., "Learning agile and dynamic motor skills for legged robots" (Science Robotics, 2019)
- Rhoban, BAM servo identification
- Lee et al., "Learning quadrupedal locomotion over challenging terrain" (Science Robotics, 2020)
- Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots" (2021)
- Lee et al., "Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning" (2019)
- Margolis & Agrawal, "Walk These Ways" (2022)
- Grandia et al., "Design and Control of a Bipedal Robotic Character" (Disney Research, 2024)
- Libby et al., "Tail-assisted pitch control in lizards, robots and dinosaurs" (Nature, 2012)
- Stanford Pupper (leg geometry)
- Open Duck Mini (closest existing project)
