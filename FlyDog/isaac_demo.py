"""Run a CRA373 RSL-RL checkpoint under FlyDrones brain control in Isaac Sim."""

from __future__ import annotations

import argparse
import csv
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
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    root = args.cra373_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (root / "cra373" / "envs" / "__init__.py").is_file():
        parser.error(f"not a CRA373 repository: {root}")
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    if args.seconds <= 0 or args.brain_hz <= 0:
        parser.error("seconds and brain-hz must be positive")
    os.environ["CRA373_ROOT"] = str(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(project / "src"))
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401 - registers base Isaac Lab tasks
        import cra373.envs  # noqa: F401 - registers CRA373 tasks
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        from flydog.bridge import FlyDogBridge

        env_cfg = parse_env_cfg(args.task, num_envs=1)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        if agent_cfg.class_name != "OnPolicyRunner":
            raise RuntimeError(f"unsupported runner: {agent_cfg.class_name}")
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg),
                                clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        bridge = FlyDogBridge(brain_source=args.brain, fly_config=args.fly_config,
                              forward_mps=args.forward_mps, yaw_rps=args.yaw_rps)
        dt = float(env.unwrapped.step_dt)
        brain_dt = 1 / args.brain_hz
        bridge.warmup(brain_dt)
        obs = env.get_observations()
        if not isinstance(obs, torch.Tensor) or obs.shape[1] != int(env_cfg.observation_space):
            raise RuntimeError(f"unexpected CRA373 policy observation: {type(obs)} {getattr(obs, 'shape', None)}")
        # CRA373Env._get_observations() places vx, vy, yaw command at columns 12:15.
        # Restrict the demo to a single robot and the matching flat/rough policy family.
        cmd = bridge.tick(0.0, brain_dt)
        rows = []
        last_label = None
        next_brain_tick = brain_dt
        print(f"FlyDog Isaac demo | {args.task} | {checkpoint}")
        for k in range(int(args.seconds / dt)):
            if not app.is_running():
                break
            started = time.monotonic()
            t = k * dt
            if t + 1e-9 >= next_brain_tick:
                yaw_dps = float(env.unwrapped._to_control_frame(
                    env.unwrapped._robot.data.root_ang_vel_b)[0, 2].item()) * 180 / math.pi
                cmd = bridge.tick(t, brain_dt, yaw_rate_dps=yaw_dps)
                next_brain_tick += brain_dt
            ranges = env_cfg.command_ranges
            values = [cmd.forward, cmd.lateral, cmd.yaw]
            limited = [max(lo, min(hi, val)) for val, (lo, hi) in zip(values, ranges)]
            with torch.inference_mode():
                env.unwrapped._commands[0] = torch.tensor(limited, device=env.unwrapped.device)
                obs[:, 12:15] = env.unwrapped._commands
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            if cmd.gesture != last_label:
                print(f"t={t:5.1f}s  {cmd.gesture:<24} vx={limited[0]:+.2f} vy={limited[1]:+.2f} yaw={limited[2]:+.2f}")
                last_label = cmd.gesture
            if args.csv:
                pos = env.unwrapped._robot.data.root_pos_w[0]
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
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
