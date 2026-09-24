"""Run a CRA373 RSL-RL checkpoint under FlyDrones brain control in Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import os
import sys
import time
from pathlib import Path


def main() -> int:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fly brain -> trained CRA373 dog in Isaac Sim")
    parser.add_argument("--cra373-root", type=Path, required=True, help="CRA373 repository root")
    parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL model_*.pt checkpoint")
    parser.add_argument("--task", default="CRA373-Flat-v0",
                        choices=["CRA373-Flat-v0", "CRA373-Flat-Run-v0", "CRA373-Rough-v0", "CRA373-Slope-v0"])
    parser.add_argument("--brain", default="minifly", help="minifly or MaleCNS .npz path")
    parser.add_argument("--fly-config", type=Path)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--brain-hz", type=float, default=20.0)
    parser.add_argument("--forward-mps", type=float, default=0.8)
    parser.add_argument("--yaw-rps", type=float, default=1.0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="open a FlyDrones-style fly-eye/brain/CRA373 dashboard")
    parser.add_argument("--dashboard-hz", type=float, default=10.0,
                        help="live dashboard refresh rate (default: 10)")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.visualizer is None and not args.headless and args.livestream in (-1, 0):
        args.visualizer = ["kit"]
    root = args.cra373_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (root / "cra373" / "envs" / "__init__.py").is_file():
        parser.error(f"not a CRA373 repository: {root}")
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    if args.seconds <= 0 or args.brain_hz <= 0 or args.dashboard_hz <= 0:
        parser.error("seconds, brain-hz, and dashboard-hz must be positive")
    os.environ["CRA373_ROOT"] = str(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(project / "src"))
    app = AppLauncher(args).app
    env = None
    dashboard = None
    dashboard_window = None
    try:
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401 - registers base Isaac Lab tasks
        import cra373.envs  # noqa: F401 - registers CRA373 tasks
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        from flydog.bridge import FlyDogBridge

        env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        agent_cfg.device = env_cfg.sim.device
        env_cfg.seed = agent_cfg.seed
        env_cfg.viewer.eye = (3.0, 3.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        if agent_cfg.class_name != "OnPolicyRunner":
            raise RuntimeError(f"unsupported runner: {agent_cfg.class_name}")
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg),
                                clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        bridge = FlyDogBridge(brain_source=args.brain, fly_config=args.fly_config,
                              forward_mps=args.forward_mps, yaw_rps=args.yaw_rps)
        if args.live:
            from flydog.dashboard import DashboardWindow, FlyDogDashboard

            dashboard = FlyDogDashboard(bridge)
            dashboard_window = DashboardWindow("FlyDog · eyes, brain, and CRA373")
        dt = float(env.unwrapped.step_dt)
        brain_dt = 1 / args.brain_hz
        bridge.warmup(brain_dt)
        obs = env.get_observations()
        policy_obs = obs["policy"]
        if policy_obs.shape != (1, int(env_cfg.observation_space)):
            raise RuntimeError(f"unexpected CRA373 policy observation shape: {tuple(policy_obs.shape)}")
        # The older gait-clock policy has three extra observations before the commands.
        command_start = 12 if policy_obs.shape[1] in (51, 238) else 9
        if policy_obs.shape[1] not in (48, 51, 235, 238):
            raise RuntimeError(f"unsupported CRA373 policy observation size: {policy_obs.shape[1]}")
        cmd = bridge.tick(0.0, brain_dt)
        rows = []
        last_label = None
        next_brain_tick = brain_dt
        next_dashboard_frame = 0.0
        print(f"FlyDog Isaac demo | {args.task} | {checkpoint}")
        for k in range(int(args.seconds / dt)):
            if not app.is_running():
                break
            started = time.monotonic()
            t = k * dt
            if t + 1e-9 >= next_brain_tick:
                # CRA373 +yaw is left; FlyDrones haltere input expects +yaw right.
                ang_vel_b = env.unwrapped._robot.data.root_ang_vel_b.torch
                if hasattr(env.unwrapped, "_to_control_frame"):
                    ang_vel_b = env.unwrapped._to_control_frame(ang_vel_b)
                yaw_dps = -float(ang_vel_b[0, 2].item()) * 180 / math.pi
                cmd = bridge.tick(t, brain_dt, yaw_rate_dps=yaw_dps)
                next_brain_tick += brain_dt
                if dashboard is not None:
                    root_pos = env.unwrapped._robot.data.root_pos_w.torch[0]
                    root_quat = env.unwrapped._robot.data.root_quat_w.torch[0]
                    # Isaac Lab stores quaternions as (w, x, y, z).
                    qw, qx, qy, qz = (float(value.item()) for value in root_quat)
                    yaw = math.atan2(2 * (qw * qz + qx * qy),
                                     1 - 2 * (qy * qy + qz * qz))
                    dashboard.push(t, cmd, float(root_pos[0].item()), float(root_pos[1].item()))
                    if t + 1e-9 >= next_dashboard_frame:
                        frame = dashboard.render(t, cmd, float(root_pos[0].item()),
                                                 float(root_pos[1].item()), yaw)
                        if not dashboard_window.show(frame):
                            break
                        next_dashboard_frame += 1 / args.dashboard_hz
            ranges = getattr(env_cfg, "command_ranges", ((-1.0, 1.0),) * 3)
            values = [cmd.forward, cmd.lateral, cmd.yaw]
            limited = [max(lo, min(hi, val)) for val, (lo, hi) in zip(values, ranges)]
            with torch.inference_mode():
                env.unwrapped._commands[0] = torch.tensor(limited, device=env.unwrapped.device)
                obs["policy"][:, command_start:command_start + 3] = env.unwrapped._commands
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            if cmd.gesture != last_label:
                print(f"t={t:5.1f}s  {cmd.gesture:<24} vx={limited[0]:+.2f} vy={limited[1]:+.2f} yaw={limited[2]:+.2f}")
                last_label = cmd.gesture
            if args.csv:
                pos = env.unwrapped._robot.data.root_pos_w.torch[0]
                rows.append(dict(t=t, gesture=cmd.gesture, vx=limited[0], vy=limited[1],
                                 yaw=limited[2], escape=cmd.escape, x=float(pos[0]), y=float(pos[1])))
            if args.real_time:
                remaining = dt - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        if args.csv and rows:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"CSV saved: {args.csv}")
        return 0
    finally:
        if dashboard_window is not None:
            dashboard_window.close()
        if dashboard is not None:
            dashboard.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
