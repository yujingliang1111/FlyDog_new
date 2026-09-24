# CRA373 New

Linux development workspace for the CRA373 quadruped robot, its NVIDIA Isaac Lab reinforcement-learning environments, and the FlyDog neural controller integration.

## Overview

This repository combines three main components:

- `CRA373_12313`: the current CRA373 robot model, Isaac Lab environments, terrain definitions, rewards, and RSL-RL training scripts.
- `FlyDog`: the bridge that converts FlyDrones neural-controller output into CRA373 planar velocity commands.
- `FlyDrones`: the fly-inspired neural controller used by FlyDog.

The FlyDog demo runs a trained CRA373 PPO locomotion policy in Isaac Sim. FlyDrones produces forward, lateral, and yaw commands; FlyDog maps those commands into the CRA373 control frame; the trained policy converts them into the robot's 12 joint actions.

## Repository layout

```text
CRA373_new/
├── CRA373_12313/
│   ├── CRA373_model/       Robot USD, URDF, configuration, and meshes
│   ├── cra373/             Isaac Lab environments and actuator model
│   ├── scripts/rsl_rl/     Training and policy playback scripts
│   └── logs/               Local training outputs (ignored by Git)
├── FlyDog/
│   ├── isaac_demo.py       FlyDog and CRA373 Isaac Sim integration
│   └── src/flydog/         Neural-to-locomotion command bridge
├── FlyDrones/              Fly-inspired neural controller
└── README.md
```

`CRA373_12313` is the current CRA373 implementation used by the examples in this README.

## Requirements

- Ubuntu or another supported Linux distribution
- NVIDIA GPU and a compatible NVIDIA driver
- NVIDIA Isaac Sim
- NVIDIA Isaac Lab
- Python and PyTorch versions provided by Isaac Lab
- RSL-RL

Before using FlyDog, confirm that the CRA373 environment can start and that its trained checkpoint is compatible with the selected task. The current flat environment uses 48 policy observations, while the current rough environment uses 235.

## Install FlyDog and FlyDrones

Run the following commands from the root of your Isaac Lab installation:

```bash
./isaaclab.sh -p -m pip install -e /home/dog101/CRA373_new/FlyDrones
./isaaclab.sh -p -m pip install -e /home/dog101/CRA373_new/FlyDog
```

The editable installs allow source-code changes in this workspace to take effect without reinstalling the packages.

## Run the FlyDog Isaac Sim demo

From the Isaac Lab root directory, run:

```bash
./isaaclab.sh -p /home/dog101/CRA373_new/FlyDog/isaac_demo.py \
  --cra373-root /home/dog101/CRA373_new/CRA373_12313 \
  --checkpoint /home/dog101/CRA373_new/CRA373_12313/logs/rsl_rl/cra373_flat/2026-09-22_16-06-36/model_3550.pt \
  --task CRA373-Flat-v0 \
  --seconds 18 \
  --real-time
```

Options:

- Change `--checkpoint` to use another trained model.
- Change `--seconds` to adjust the demo duration.
- Omit `--real-time` to run without real-time pacing.
- Add `--headless` to run without the Isaac Sim viewer.

Training checkpoints and logs are local artifacts and are intentionally excluded from Git.

## Train a CRA373 policy

From the Isaac Lab root directory:

```bash
./isaaclab.sh -p /home/dog101/CRA373_new/CRA373_12313/scripts/rsl_rl/train.py \
  --task CRA373-Flat-v0
```

Training output is written below:

```text
CRA373_12313/logs/rsl_rl/
```

## Play a trained CRA373 policy

To evaluate the CRA373 policy without FlyDog:

```bash
./isaaclab.sh -p /home/dog101/CRA373_new/CRA373_12313/scripts/rsl_rl/play.py \
  --task CRA373-Flat-v0 \
  --checkpoint /path/to/model.pt
```

## Available environments

The main environment configurations are located in `CRA373_12313/cra373/envs/` and include:

- Flat terrain locomotion
- Rough terrain locomotion
- Jumping
- Parkour
- Terrain-aware locomotion experiments

Reward functions, curricula, and termination conditions are located in `CRA373_12313/cra373/envs/mdp/`.

## Robot assets and actuator model

Robot assets are stored in `CRA373_12313/CRA373_model/`:

- USD simulation assets
- URDF robot description
- Visual and collision meshes
- Physics and sensor configuration

The custom XB42M actuator model is located at `CRA373_12313/cra373/actuator/XB42M.py`.

## Notes

- This project currently targets simulation and research workflows.
- Do not commit private or large training checkpoints unless they are intentionally being released.
- Keep the task name, observation layout, and checkpoint architecture compatible.
- The absolute paths in the examples match the current Linux workstation. Update them if the repository is moved.

## Author

Yujing Liang — [GitHub](https://github.com/yujingliang1111)
