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

## 1. Install the command preview

Prerequisites: Python 3.10+ and a clone of FlyDrones. From a folder containing both `FlyDog/` and `FlyDrones/`:

```powershell
cd FlyDog
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ..\FlyDrones
python -m pip install -e .
```

On Linux, activate with `source .venv/bin/activate` and use `../FlyDrones`.

Run the command preview without Isaac Lab or a checkpoint:

```powershell
flydog-demo --seconds 18 --live
flydog-demo --seconds 18 --record outputs/flydog.gif --csv outputs/commands.csv
```

The preview shows neural activity, velocity commands and a **kinematic command path**. The plotted marker is not a physics simulation of the dog. `--live` needs a GUI; `--record` can write a GIF without one. The MiniFly brain is FlyDrones' synthetic 850-neuron demo brain, not the full MaleCNS connectome.

## 2. Run the trained dog in Isaac Sim

Prerequisites: a working Isaac Lab/Isaac Sim installation that already runs CRA373's `scripts/rsl_rl/play.py`, the CRA373 repository with its USD assets, and a compatible RSL-RL `model_*.pt` checkpoint. **No checkpoint is included in these repositories.** Use the same task family and observation dimensions as the checkpoint's training run: Flat/Flat-Run = 51, Rough/Slope = 238.

Install FlyDog and FlyDrones into Isaac Lab's Python environment, for example from the Isaac Lab root:

```powershell
.\isaaclab.bat -p -m pip install -e E:\ITRI\RL\CRA373_new\FlyDrones
.\isaaclab.bat -p -m pip install -e E:\ITRI\RL\CRA373_new\FlyDog
```

Then run the viewer demo (replace the checkpoint path):

```powershell
.\isaaclab.bat -p E:\ITRI\RL\CRA373_new\FlyDog\isaac_demo.py `
  --cra373-root E:\ITRI\RL\CRA373_new\CRA373 `
  --checkpoint C:\path\to\model_300.pt `
  --task CRA373-Flat-v0 `
  --seconds 18 `
  --real-time `
  --csv E:\ITRI\RL\CRA373_new\FlyDog\outputs\isaac_trace.csv
```

On Linux use `./isaaclab.sh -p /path/to/FlyDog/isaac_demo.py` and Linux paths. Omit `--real-time` to run as fast as the simulation allows. Add `--headless` for a display-free run. The control loop updates the fly brain at 20 Hz and runs the trained policy at CRA373's environment step rate. The Isaac viewer shows the actual simulated robot walking.

For a MaleCNS brain built by FlyDrones, replace `--brain minifly` with `--brain /path/to/malecns_brain.npz`; calibrate its readout with FlyDrones first and pass a YAML override with `decoder.readout_file` using `--fly-config`. The bridge maps FlyDrones' **throttle** readout to dog forward speed, so a readout calibrated for flight should be checked in the command preview before Isaac playback.

## Mapping and limits

| FlyDrones output | CRA373 command | Default limit |
|---|---|---:|
| `throttle + forward` | forward velocity | ±0.8 m/s |
| `lateral` | sideways velocity | ±0.5 m/s |
| `yaw` | yaw rate | ±1.0 rad/s |
| giant-fiber escape | stop all velocity commands | 0 |

The Isaac script also clamps goals to the selected CRA373 task's training command ranges. Adjust `--forward-mps` and `--yaw-rps` conservatively to stay within the trained policy's behavior. The bridge reads and writes CRA373's current `DirectRLEnv` internals (`_commands`, observation columns 12:15); changes to CRA373's observation layout require updating the bridge. The current demo is simulation-only and does not command physical hardware.
