# CRA373 New

Linux development workspace for the CRA373 quadruped robot, its NVIDIA Isaac Lab reinforcement-learning environments, and the FlyDog neural controller integration.

## Overview

This repository combines three main components:

- `CRA373_12313`: the current CRA373 robot model, Isaac Lab environments, terrain definitions, rewards, and RSL-RL training scripts.
- `FlyDog`: the bridge that converts FlyDrones neural-controller output into CRA373 planar velocity commands.
- `FlyDrones`: the fly-inspired neural controller used by FlyDog.

The FlyDog demo runs a trained CRA373 PPO locomotion policy in Isaac Sim. FlyDrones produces forward, lateral, and yaw commands; FlyDog maps those commands into the CRA373 control frame; the trained policy converts them into the robot's 12 joint actions.

## Project goal

The goal is to study a layered, biologically inspired control system for a quadruped: a simulated fruit-fly nervous system chooses high-level motion from visual stimuli, while a learned CRA373 locomotion policy handles balance and joint-level walking. Keeping these layers separate makes it possible to inspect the fly brain, replace either controller independently, and work toward a later simulation-to-real deployment on the physical CRA373 robot.

The current milestone is a simulation-only proof of concept. Gesture-driven optic-flow illusions stimulate the fly brain, the bridge converts descending-neuron activity into planar velocity goals, and the trained PPO policy moves the dog in Isaac Sim. A real robot camera, hardware actuation, and end-to-end sim-to-real validation remain future work.

```text
scripted visual stimulus -> fly retina -> spiking fly brain
    -> motor decoder -> FlyDog command bridge
    -> CRA373 PPO locomotion policy -> 12 joints -> Isaac Sim
           ^                                      |
           +--------- robot yaw feedback ---------+
```

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

Activate the Python 3.12 environment in which Isaac Lab is installed, then run the installation commands from the root of your Isaac Lab checkout:

```bash
conda activate env_isaaclab_2
cd /home/dog101/IsaacLab

./isaaclab.sh -p -m pip install -e /home/dog101/CRA373_new/FlyDrones
./isaaclab.sh -p -m pip install -e /home/dog101/CRA373_new/FlyDog
```

Do not run the demo from Conda's `base` environment. On this workstation, `base` uses Python 3.14, but Isaac Sim 6.x requires Python 3.12. You can confirm the selected interpreter with `./isaaclab.sh -p -c "import sys; print(sys.executable, sys.version)"`.

The editable installs allow source-code changes in this workspace to take effect without reinstalling the packages. The dashboard uses a GUI-enabled OpenCV build when available and automatically falls back to Tk when Isaac Sim provides headless OpenCV.

## Run the FlyDog Isaac Sim demo

From the Isaac Lab root directory, with `env_isaaclab_2` still active, run:

```bash
./isaaclab.sh -p /home/dog101/CRA373_new/FlyDog/isaac_demo.py \
  --cra373-root /home/dog101/CRA373_new/CRA373_12313 \
  --checkpoint /home/dog101/CRA373_new/CRA373_12313/logs/rsl_rl/cra373_flat/2026-09-22_16-06-36/model_3550.pt \
  --task CRA373-Flat-v0 \
  --seconds 18 \
  --real-time \
  --live
```

Options:

- Change `--checkpoint` to use another trained model.
- Change `--seconds` to adjust the demo duration.
- Omit `--real-time` to run without real-time pacing.
- Omit `--live` to disable the separate fly-eye/brain dashboard.
- Use `--dashboard-hz 5` if dashboard rendering is too expensive (default: 10 Hz).
- Add `--viz none` and omit `--live` for a display-free run.

With `--live`, a second window complements the Isaac viewer. It displays the false-colour stimulus reaching the fly's two eyes, live neural spikes and descending-neuron rates, the CRA373 path from above, and the resulting velocity goals. The eye panel currently visualizes scripted ommatidia input; it is not a CRA373 camera feed.

## 即時畫面與控制流程說明

### 畫面顯示什麼

- `FLY EYES`
  - 顯示送進果蠅視覺系統的左右眼刺激。
  - 這是果蠅複眼感受到的簡化光流（optic flow），不是一般相機照片。
  - 顏色含義：
    - 紅色：物體快速接近，產生 looming／碰撞威脅。
    - 綠色：上下方向的視覺移動。
    - 藍色：左右方向的視覺移動。
- `FLY BRAIN`
  - 顯示果蠅神經網路內神經元的即時放電。
  - 每一個小點或短線代表一次 spike。
  - 不同顏色區分感覺神經元、中間神經元和輸出神經元。
- `DESCENDING NEURONS`
  - 顯示下降神經元的放電頻率，單位是 Hz。
  - 這些神經元是果蠅大腦送往運動系統的輸出。
  - FlyDog 會把這些輸出轉換成機器人的前進、側移與旋轉命令。
- `CRA373 SCENE`
  - 從上方顯示 CRA373 的實際模擬位置。
  - 綠線是機器人走過的路徑。
  - 箭頭代表目前朝向。
- `CRA373 VELOCITY GOALS`
  - 顯示送給 PPO 步態策略的速度目標：
    - `forward`：前進速度。
    - `lateral`：橫向移動速度。
    - `yaw`：旋轉速度。
- `COMMAND HISTORY`
  - 顯示最近一段時間內三個速度命令的變化。

### 速度命令是自己辨識的，還是程式寫好的？

兩者各負責不同部分。果蠅神經網路會根據視覺刺激和神經連接，自行產生當下的神經放電；但是「如何把神經輸出解讀成 CRA373 的速度命令」是原始碼中明確寫好的規則，不是系統在執行時自己發現或學會的。Git 紀錄顯示這套 FlyDog 轉換原本由 `yujingliang1111` 提交，之後又調整了 CRA373 左右方向的符號。

主要轉換寫在 `FlyDog/src/flydog/bridge.py`：

```text
forward = clip(flight.throttle + flight.forward) × forward_mps
lateral = -clip(flight.lateral) × lateral_mps
yaw     = -clip(flight.yaw) × yaw_rps

如果觸發 escape：forward = lateral = yaw = 0
```

負號用來修正 FlyDrones 與 CRA373 控制座標中左右方向的定義差異。`forward_mps`、`lateral_mps` 和 `yaw_rps` 則限制命令的實際尺度。接著 `isaac_demo.py` 會再依照訓練環境允許的 command range 進行裁切。

因此，各層的性質如下：

| 階段 | 目前如何產生 | 是否自行學習 |
|---|---|---|
| 示範手勢 | 預先寫好的時間表 | 否 |
| 手勢轉光流 | `GestureIllusion` 的固定規則 | 否 |
| 果蠅神經放電 | 由 connectome 與 LIF 神經動態對刺激產生 | 執行時自行反應，但不是在線學習 |
| 神經輸出轉飛行意圖 | `MotorDecoder` 的設定與規則 | 否 |
| 飛行意圖轉 CRA373 速度 | `FlyDogBridge` 的固定公式 | 否 |
| 速度目標轉 12 個關節動作 | 已訓練的 PPO locomotion policy | 是，來自先前的強化學習訓練；示範時只做推論 |

### 果蠅怎麼知道手在移動？

在目前的 `isaac_demo.py` 中，果蠅其實沒有透過攝影機看見操作者的手。程式使用 FlyDrones 的 `demo_timeline()` 預先安排手勢：

- 0–2.5 秒：沒有手
- 2.5 秒：張開手掌
- 4.5 秒：握拳
- 9.5 秒：拳頭移向右方
- 12 秒：拳頭回到中央
- 13.5 秒：手快速接近相機
- 15.5 秒：手離開畫面

`GestureIllusion` 會把這些標籤轉成果蠅可能看到的光流，而不是直接把「向左」、「向右」等命令交給神經網路。例如：

- 張開手掌會產生垂直光流，模擬果蠅正在下沉。
- 手向右移動會產生左右眼不對稱的旋轉光流，引發視動反射。
- 手快速接近會產生擴張光流，刺激 looming-sensitive 路徑並可能觸發 escape；FlyDog 會在 escape 時讓機器人停止。
- 手離開畫面會產生相反方向的垂直光流。

所以目前看到的是「腳本手勢 → 模擬視覺刺激 → 果蠅神經反應 → 固定速度轉換 → PPO 步態」的概念驗證。FlyDrones 雖然具有 MediaPipe/OpenCV 的真實 webcam 手勢辨識功能，但它尚未接入這個 CRA373 Isaac 示範；若要用真實手勢控制，仍需把 webcam 的 `GestureState` 接到 FlyDog 控制迴圈。

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
