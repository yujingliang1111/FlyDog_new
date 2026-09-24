# FlyDog: FlyDrones brain controls CRA373 walking

This repository connects the [FlyDrones](https://github.com/SpikeCalls/FlyDrones) spiking fly brain to a **trained CRA373 quadruped locomotion policy**. The fly brain produces planar velocity goals (`vx`, `vy`, `yaw rate`); the CRA373 RSL-RL policy converts those goals and robot observations into 12 joint actions. The bridge does not replace or retrain the locomotion policy.

```text
Scripted hand gesture (FlyDrones demo timeline)
  -> optic-flow illusion -> Retina -> InputEncoder
  -> MiniFly or MaleCNS LIF brain -> MotorDecoder
  -> FlyDog ground adapter (forward / lateral / yaw, escape = stop)
  -> CRA373 _commands -> trained PPO policy -> 12 joint actions -> Isaac Sim
                                           ^ robot observation and yaw feedback
```

The default demo follows FlyDrones' hand sequence: open palm, fist, move right, rush toward camera, drop hand. These inputs are **scripted**. The current CRA373 environment does not enable its front camera, so this version does not feed a robot camera image into the fly eye. The Isaac demo feeds measured yaw rate back into the FlyDrones haltere input. Adding a real robot-camera image is a later integration step.

## Repository layout

```text
FlyDog/
  src/flydog/bridge.py   Shared fly-brain controller and grounded velocity mapping
  src/flydog/demo.py     FlyDrones-style standalone preview, CSV and GIF
  isaac_demo.py          CRA373 checkpoint playback in Isaac Sim
  pyproject.toml         Standalone Python package metadata
```

FlyDog can be pushed to its own GitHub repository. It does not copy source files or model assets from the two upstream projects. Clone/install those projects alongside it. Do not upload private checkpoints unless you intend to share them.

## 1. Install the command preview on Linux

Prerequisites: Python 3.10+ and a clone of FlyDrones. From a folder containing both `FlyDog/` and `FlyDrones/`:

```bash
cd /home/dog101/CRA373_new/FlyDog
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ../FlyDrones
python -m pip install -e .
```

Run the command preview without Isaac Lab or a checkpoint:

```bash
flydog-demo --seconds 18 --live
flydog-demo --seconds 18 --record outputs/flydog.gif --csv outputs/commands.csv
```

The preview shows neural activity, velocity commands and a **kinematic command path**. The plotted marker is not a physics simulation of the dog. `--live` needs a GUI; `--record` can write a GIF without one. The MiniFly brain is FlyDrones' synthetic 850-neuron demo brain, not the full MaleCNS connectome.

## 2. Run the trained dog in Isaac Sim

Prerequisites: a working Isaac Lab/Isaac Sim installation that already runs CRA373's `scripts/rsl_rl/play.py`, the CRA373 repository with its USD assets, and a compatible RSL-RL `model_*.pt` checkpoint. **No checkpoint is included in these repositories.** Use the same task family and observation dimensions as the checkpoint's training run: the newer CRA373_12313 environment uses Flat = 48 and Rough = 235; the older gait-clock environment uses Flat/Flat-Run = 51 and Rough/Slope = 238.

Install FlyDog and FlyDrones into Isaac Lab's Python environment. Run these commands from the Isaac Lab root directory:

```bash
./isaaclab.sh -p -m pip install -e /home/dog101/CRA373_new/FlyDrones
./isaaclab.sh -p -m pip install -e /home/dog101/CRA373_new/FlyDog
```

Then run the CRA373 viewer demo:

```bash
./isaaclab.sh -p /home/dog101/CRA373_new/FlyDog/isaac_demo.py \
  --cra373-root /home/dog101/CRA373_new/CRA373_12313 \
  --checkpoint /home/dog101/CRA373_new/CRA373_12313/logs/rsl_rl/cra373_flat/2026-09-22_16-06-36/model_3550.pt \
  --task CRA373-Flat-v0 \
  --seconds 18 \
  --real-time
```

The command must be run from the Isaac Lab root directory, where `isaaclab.sh` is located. Change the checkpoint path when using a different training run. Omit `--real-time` to run as fast as the simulation allows, or add `--headless` for a display-free run. The control loop updates the fly brain at 20 Hz and runs the trained policy at CRA373's environment step rate. The Isaac viewer shows the actual simulated robot walking.

For a MaleCNS brain built by FlyDrones, replace `--brain minifly` with `--brain /path/to/malecns_brain.npz`; calibrate its readout with FlyDrones first and pass a YAML override with `decoder.readout_file` using `--fly-config`. The bridge maps FlyDrones' **throttle** readout to dog forward speed, so a readout calibrated for flight should be checked in the command preview before Isaac playback.

## Mapping and limits

| FlyDrones output | CRA373 command | Default limit |
|---|---|---:|
| `throttle + forward` | forward velocity | ±0.8 m/s |
| `lateral` | sideways velocity | ±0.5 m/s |
| `yaw` | yaw rate | ±1.0 rad/s |
| giant-fiber escape | stop all velocity commands | 0 |

The Isaac script also clamps goals to the selected CRA373 task's training command ranges. Adjust `--forward-mps` and `--yaw-rps` conservatively to stay within the trained policy's behavior. The bridge writes CRA373's current `DirectRLEnv` command buffer and supports the current 48/235-observation layouts as well as the older 51/238-observation gait-clock layouts. Other observation layouts require updating the integration. The current demo is simulation-only and does not command physical hardware.
