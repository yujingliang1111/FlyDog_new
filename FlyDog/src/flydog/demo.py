"""Checkpoint-free FlyDrones-style preview of the brain-to-dog command bridge."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .bridge import FlyDogBridge


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniFly brain -> CRA373 velocity command preview")
    p.add_argument("--seconds", type=float, default=18.0)
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--brain", default="minifly", help="minifly or MaleCNS .npz path")
    p.add_argument("--fly-config", help="FlyDrones YAML override")
    p.add_argument("--forward-mps", type=float, default=0.8)
    p.add_argument("--yaw-rps", type=float, default=1.0)
    p.add_argument("--record", type=Path, help="write animated GIF")
    p.add_argument("--csv", type=Path, help="write command trace")
    p.add_argument("--live", action="store_true", help="show matplotlib window")
    p.add_argument("--every", type=int, default=2, help="render every N control ticks")
    return p


def render(rows, brain_name: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    latest = rows[-1]
    fig, (ax, traces) = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.suptitle(f"FlyDog | {brain_name} -> CRA373 velocity goals | {latest['t']:.1f} s")
    xs, ys = [r["x"] for r in rows], [r["y"] for r in rows]
    ax.plot(xs, ys, color="#39b6c8", lw=2)
    ax.scatter([xs[0]], [ys[0]], color="#555", s=25)
    ax.scatter([xs[-1]], [ys[-1]], color="#d95475", s=90)
    ax.add_patch(FancyArrowPatch((xs[-1], ys[-1]),
                                 (xs[-1] + 0.35 * math.cos(latest["heading"]),
                                  ys[-1] + 0.35 * math.sin(latest["heading"])),
                                 arrowstyle="-|>", mutation_scale=18, color="#d95475", lw=2))
    ax.set_xlim(min(-1, min(xs) - 0.5), max(1, max(xs) + 0.5))
    ax.set_ylim(min(-1, min(ys) - 0.5), max(1, max(ys) + 0.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_title("Command-only path preview")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ts = [r["t"] for r in rows]
    traces.plot(ts, [r["forward"] for r in rows], label="forward m/s")
    traces.plot(ts, [r["yaw"] for r in rows], label="yaw rad/s")
    traces.plot(ts, [r["dng_left"] / 100 for r in rows], alpha=0.6, label="DNg02 L / 100")
    traces.plot(ts, [r["dng_right"] / 100 for r in rows], alpha=0.6, label="DNg02 R / 100")
    traces.set_xlim(0, max(18, ts[-1] + 0.5))
    traces.set_ylim(-1.2, 1.2)
    traces.grid(alpha=0.25)
    traces.legend(fontsize=8, loc="upper right")
    traces.set_title(f"Gesture: {latest['gesture']}" + (" | ESCAPE STOP" if latest["escape"] else ""))
    traces.set_xlabel("time (s)")
    fig.tight_layout()
    fig.canvas.draw()
    import numpy as np
    image = np.asarray(fig.canvas.buffer_rgba()).copy()[..., :3]
    plt.close(fig)
    return image


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.seconds <= 0 or args.hz <= 0 or args.every < 1:
        raise SystemExit("seconds and hz must be positive; every must be >= 1")
    bridge = FlyDogBridge(brain_source=args.brain, fly_config=args.fly_config,
                          forward_mps=args.forward_mps, yaw_rps=args.yaw_rps)
    dt = 1 / args.hz
    bridge.warmup(dt)
    rows, frames = [], []
    x = y = heading = 0.0
    last_gesture = None
    if args.live:
        import matplotlib.pyplot as plt
        plt.ion()
        window = plt.figure("FlyDog Demo")
        viewer = window.add_subplot(111)
        viewer.axis("off")
    print("FlyDog demo: MiniFly optic-flow illusion -> fly brain -> velocity goals")
    print("Path is a command preview; run isaac_demo.py with a checkpoint for robot physics.")
    try:
        for k in range(int(args.seconds * args.hz)):
            t = k * dt
            cmd = bridge.tick(t, dt)
            heading += cmd.yaw * dt
            x += (cmd.forward * math.cos(heading) - cmd.lateral * math.sin(heading)) * dt
            y += (cmd.forward * math.sin(heading) + cmd.lateral * math.cos(heading)) * dt
            row = dict(t=round(t, 3), gesture=cmd.gesture, forward=cmd.forward,
                       lateral=cmd.lateral, yaw=cmd.yaw, escape=cmd.escape,
                       dng_left=cmd.rates.get("DNg02_L", 0), dng_right=cmd.rates.get("DNg02_R", 0),
                       x=x, y=y, heading=heading)
            rows.append(row)
            if cmd.gesture != last_gesture:
                print(f"t={t:5.1f}s  {cmd.gesture:<24} vx={cmd.forward:+.2f}  yaw={cmd.yaw:+.2f}")
                last_gesture = cmd.gesture
            if (args.record or args.live) and k % args.every == 0:
                frame = render(rows, args.brain)
                if args.record:
                    frames.append(frame)
                if args.live:
                    viewer.imshow(frame)
                    viewer.axis("off")
                    window.canvas.draw_idle()
                    plt.pause(0.001)
                    viewer.clear()
    except KeyboardInterrupt:
        print("stopped")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved: {args.csv}")
    if args.record and frames:
        from PIL import Image
        args.record.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(f) for f in frames]
        images[0].save(args.record, save_all=True, append_images=images[1:],
                       duration=max(20, int(1000 * dt * args.every)), loop=0)
        print(f"GIF saved: {args.record}")
    if rows:
        print(f"final command path: x={x:.2f} m, y={y:.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
