# CRA373

Reinforcement Learning and Simulation Environment for the CRA373 Quadruped Robot.

This repository contains the simulation model, actuator configuration, reinforcement learning environments, perception modules, and training/evaluation scripts developed for the CRA373 quadruped robot using **NVIDIA Isaac Lab**.

## Overview

The main goal of this project is to develop and evaluate reinforcement learning-based locomotion and control policies for the CRA373 quadruped robot in simulation, with future deployment toward the physical robot.

The project currently focuses on:

* Quadruped locomotion using Reinforcement Learning (RL)
* Isaac Lab simulation and environment development
* Custom actuator modeling
* Terrain perception and terrain-aware locomotion
* Affordance-based terrain information
* PPO-based policy training
* Simulation-to-Real preparation
* Robot model and sensor configuration

## Repository Structure

```text
CRA373/
├── CRA373_model/
│   ├── configuration/
│   ├── meshes/
│   ├── cra373.urdf
│   ├── cra373.usd
│   └── config.yaml
│
├── cra373/
│   ├── actuator/
│   │   └── XB42M.py
│   │
│   ├── affordance/
│   │   └── terrain_affordance.py
│   │
│   ├── envs/
│   │   ├── agents/
│   │   ├── mdp/
│   │   ├── CRA373.py
│   │   ├── cra373_env.py
│   │   ├── cra373_env_cfg.py
│   │   ├── flat_env_cfg.py
│   │   ├── rough_env_cfg.py
│   │   ├── jump_env_cfg.py
│   │   └── parkour_env_cfg.py
│   │
│   ├── perception/
│   │   └── terrain_encoder.py
│   │
│   ├── policies/
│   │   └── policy_manager.py
│   │
│   └── skill_selection/
│       └── skill_selector.py
│
├── manager_based_check/  # Previous implementation / reference code
│   ├── CRA373/
│   ├── mdp/
│   └── parkour/
│
├── scripts/
│   └── rsl_rl/
│       ├── train.py
│       ├── play.py
│       ├── play_affordance.py
│       └── cli_args.py
│
└── .gitignore
```
> **Note:** `manager_based_check/` contains previous implementation files and is kept for reference and comparison. It is **not the main implementation** currently used for CRA373 development.

## Robot Model

The `CRA373_model` directory contains the simulation assets for the CRA373 quadruped robot, including:

* USD robot model
* URDF model
* Robot configuration files
* Physics configuration
* Sensor configuration
* STL meshes

These assets are used to construct and simulate the CRA373 robot in Isaac Lab.

## Actuator Model

The project includes a custom actuator model for the CRA373 robot:

```text
cra373/actuator/XB42M.py
```

The actuator model is intended to represent the characteristics of the XB42M-based joint actuators used by the robot.

The actuator modeling work includes:

* Position control behavior
* Motor torque limitations
* Torque-speed characteristics
* Actuator dynamics
* Simulation-to-real actuator calibration

The actuator model will be further refined as additional hardware and motor parameters become available.

## Reinforcement Learning

The project uses reinforcement learning to learn quadruped locomotion policies.

The current training pipeline is based on **PPO (Proximal Policy Optimization)** and the RSL-RL framework.

Training scripts are located at:

```text
scripts/rsl_rl/
```

Main scripts:

```text
train.py
play.py
play_affordance.py
```

### Training

A typical training command is:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py --task CRA373-Flat-v0
```

### Policy Evaluation

A trained policy can be evaluated using:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task CRA373-Flat-v0 \
    --checkpoint <PATH_TO_CHECKPOINT>
```

The exact task and checkpoint path depend on the current experiment configuration.

## Environment

The main environment implementation is located in:

```text
cra373/envs/
```

Different environments are provided for different locomotion scenarios:

* Flat terrain
* Rough terrain
* Jumping
* Parkour
* Terrain-aware locomotion

Reward functions, termination conditions, and curriculum mechanisms are implemented under:

```text
cra373/envs/mdp/
```

## Perception

The project also includes terrain perception and terrain representation modules:

```text
cra373/perception/
cra373/affordance/
```

The long-term goal is to incorporate terrain information into the locomotion policy and investigate perception-aware control.

## Simulation-to-Real

A major objective of the project is to reduce the gap between simulation and the physical CRA373 robot.

Current areas of investigation include:

### Actuator

* Motor torque-speed characteristics
* Position controller parameters
* Joint dynamics
* Actuator delay
* Motor/gearbox characteristics

### Observation

* IMU measurements
* Joint position
* Joint velocity
* Joint torque/current feedback
* Sensor noise
* Observation delay

### Simulation

* Contact dynamics
* Friction
* Mass and inertia
* Joint damping
* Ground interaction
* Sensor placement

The actuator and sensor models will be calibrated using measurements from the physical robot as hardware information becomes available.

## Development Status

### Completed / In Progress

* [x] CRA373 robot model integration
* [x] Isaac Lab simulation environment
* [x] Custom actuator implementation
* [x] PPO locomotion training pipeline
* [x] Flat terrain environment
* [x] Rough terrain environment
* [x] Jumping environment
* [x] Terrain perception modules
* [x] Affordance-related modules
* [ ] Actuator model calibration
* [ ] Sensor noise and delay calibration
* [ ] Sim-to-Real validation
* [ ] Physical robot deployment

## Requirements

The project is developed using:

* NVIDIA Isaac Lab
* NVIDIA Isaac Sim
* Python
* PyTorch
* RSL-RL

The exact versions depend on the current development environment.

## Notes

This repository is primarily intended for research and development of the CRA373 quadruped robot.

The implementation is actively under development, and interfaces, configurations, reward functions, actuator models, and training settings may change as the project progresses.

## Author

**Yujing Liang**

GitHub: [yujingliang1111](https://github.com/yujingliang1111)
