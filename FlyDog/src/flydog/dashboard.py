"""Live fly-eye, fly-brain, and CRA373 control dashboard."""

from __future__ import annotations

from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG = "#07090d"
PANEL = "#0d1117"
GRID = "#1f2630"
TEXT = "#d7e0ea"
MUTED = "#7d8a99"
SPIKE = "#39ff88"
INPUT = "#4cc9f0"
OUTPUT = "#ff4d8d"
WARN = "#ffb020"


class DashboardWindow:
    """Display dashboard frames with OpenCV or a Tk fallback.

    Isaac Sim commonly installs ``opencv-python-headless``.  That package can
    process images but its ``imshow`` and ``destroyAllWindows`` functions
    raise at runtime, so checking that ``cv2`` imports is not sufficient.
    """

    def __init__(self, name: str = "FlyDog"):
        self.name = name
        self.backend = None
        self.closed = False
        self.cv2 = None
        self.root = None
        self.label = None
        self.photo = None
        errors = []

        try:
            import cv2

            gui_line = next(
                (line for line in cv2.getBuildInformation().splitlines()
                 if line.strip().startswith("GUI:")),
                "GUI: NONE",
            )
            if not gui_line.upper().endswith("NONE"):
                self.cv2 = cv2
                self.backend = "opencv"
                return
            errors.append("OpenCV was built without GUI support")
        except Exception as exc:  # pragma: no cover - depends on local packages
            errors.append(f"OpenCV unavailable: {exc}")

        try:
            import tkinter as tk
            from PIL import Image, ImageTk

            self.tk = tk
            self.Image = Image
            self.ImageTk = ImageTk
            self.root = tk.Tk()
            self.root.title(name)
            self.root.protocol("WM_DELETE_WINDOW", self._request_close)
            self.root.bind("<KeyPress-q>", lambda _event: self._request_close())
            self.label = tk.Label(self.root, bg=BG, borderwidth=0)
            self.label.pack()
            self.backend = "tk"
        except Exception as exc:  # pragma: no cover - requires a display server
            errors.append(f"Tk unavailable: {exc}")
            raise RuntimeError(
                "--live could not open a dashboard window (" + "; ".join(errors) + ")"
            ) from exc

    def _request_close(self) -> None:
        self.closed = True

    def show(self, rgb: np.ndarray) -> bool:
        if self.closed:
            return False
        if self.backend == "opencv":
            try:
                self.cv2.imshow(self.name, rgb[..., ::-1])
                return (self.cv2.waitKey(1) & 0xFF) != ord("q")
            except Exception as exc:
                raise RuntimeError(
                    "OpenCV could not display the dashboard; install a GUI-enabled "
                    "OpenCV build or use the Tk fallback"
                ) from exc

        try:
            self.photo = self.ImageTk.PhotoImage(self.Image.fromarray(rgb))
            self.label.configure(image=self.photo)
            self.root.update_idletasks()
            self.root.update()
            return not self.closed
        except self.tk.TclError:
            self.closed = True
            return False

    def close(self) -> None:
        if self.backend == "opencv" and self.cv2 is not None:
            try:
                self.cv2.destroyAllWindows()
            except Exception:
                pass
        elif self.backend == "tk" and self.root is not None:
            try:
                self.root.destroy()
            except self.tk.TclError:
                pass
        self.closed = True


class FlyDogDashboard:
    """Render the fly controller and physical dog state into an RGB frame."""

    def __init__(self, bridge, history_s: float = 2.0):
        self.bridge = bridge
        self.brain = bridge.brain
        self.history_s = history_s
        self.raster: deque[tuple[float, np.ndarray]] = deque()
        self.trail: deque[tuple[float, float]] = deque(maxlen=600)
        self.commands: deque[tuple[float, float, float, float]] = deque(maxlen=300)

        connectome = self.brain.connectome
        recorded = self.brain.record
        roles = np.zeros(recorded.size, dtype=int)
        for name in self.brain.input_specs:
            roles[np.isin(recorded, connectome.group(name))] = 1
        for name in self.brain.output_specs:
            roles[np.isin(recorded, connectome.group(name))] = 2
        order = np.argsort(roles, kind="stable")
        self.row_of = np.empty_like(order)
        self.row_of[order] = np.arange(order.size)
        self.roles = roles
        self.fig = plt.figure(figsize=(12.8, 5.4), dpi=100, facecolor=BG)

    @staticmethod
    def _style(ax, title: str) -> None:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_title(title, color=TEXT, fontsize=9, loc="left", pad=4,
                     fontfamily="monospace")

    def push(self, t: float, command, x: float, y: float) -> None:
        """Collect one fly-brain update without drawing it."""
        brain_ms = self.brain.net.t_ms
        for spike_ms, positions in self.brain.last_raster:
            self.raster.append((t + (spike_ms - brain_ms) / 1000.0, positions.copy()))
        while self.raster and self.raster[0][0] < t - self.history_s:
            self.raster.popleft()
        self.trail.append((x, y))
        self.commands.append((t, command.forward, command.lateral, command.yaw))

    def _eye_image(self) -> np.ndarray:
        """Convert ommatidia features into a visible two-eye false-colour image."""
        eyes = []
        for side in ("L", "R"):
            grids = self.bridge.last_vision.eyes[side].grids
            horizontal = np.maximum(grids["ftb"], grids["btf"])
            vertical = np.maximum(grids["up"], grids["down"])
            looming = np.maximum(grids["loom"], grids["loom_speed"])
            brightness = grids["brightness"] * 0.25
            eyes.append(np.stack((np.maximum(looming, brightness),
                                  np.maximum(vertical, brightness),
                                  np.maximum(horizontal, brightness)), axis=-1))
        separator = np.full((eyes[0].shape[0], 1, 3), 0.08, dtype=np.float32)
        return np.concatenate((eyes[0], separator, eyes[1]), axis=1)

    def render(self, t: float, command, x: float, y: float, yaw_rad: float) -> np.ndarray:
        fig = self.fig
        fig.clf()
        grid = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.35, 1.0],
                                height_ratios=[1, 1, 0.8], left=0.035,
                                right=0.985, top=0.86, bottom=0.07,
                                wspace=0.18, hspace=0.45)
        connectome = self.brain.connectome
        fig.text(0.035, 0.945, "FlyDog · fly brain → CRA373",
                 color=TEXT, fontsize=15, fontweight="bold", fontfamily="monospace")
        fig.text(0.035, 0.905,
                 f"{self.brain.n_neurons:,} neurons · {connectome.n_connections:,} connections · {connectome.name}",
                 color=MUTED, fontsize=9, fontfamily="monospace")
        fig.text(0.985, 0.945, f"t = {t:5.1f} s", color=SPIKE,
                 fontsize=12, ha="right", fontfamily="monospace")
        speed = self.brain.realtime_factor
        speed_text = f"brain speed {speed:4.1f}x real time" if np.isfinite(speed) else ""
        fig.text(0.985, 0.905, speed_text, color=MUTED, fontsize=9,
                 ha="right", fontfamily="monospace")

        ax = fig.add_subplot(grid[0:2, 0])
        self._style(ax, "FLY EYES  (ommatidia stimulus)")
        ax.imshow(self._eye_image(), interpolation="nearest", vmin=0, vmax=1,
                  aspect="auto")
        ax.axvline(self.bridge.retina.cols - 0.5, color=INPUT, lw=1.0, alpha=0.8)
        ax.text(0.25, 1.01, "LEFT", transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=7, fontfamily="monospace")
        ax.text(0.75, 1.01, "RIGHT", transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=7, fontfamily="monospace")
        label = self.bridge.illusion.mode
        ax.text(0.02, 0.03, label, transform=ax.transAxes,
                color=WARN if "loom" in label else TEXT, fontsize=8,
                fontfamily="monospace",
                bbox={"facecolor": BG, "alpha": 0.75, "edgecolor": "none"})
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(grid[0:2, 1])
        self._style(ax, "FLY BRAIN  (live spikes: sensory / interneurons / descending)")
        if self.raster:
            times = np.concatenate([np.full(pos.size, ts) for ts, pos in self.raster])
            positions = np.concatenate([pos for _, pos in self.raster])
            rows = self.row_of[positions]
            colors = np.where(self.roles[positions] == 1, INPUT,
                              np.where(self.roles[positions] == 2, OUTPUT, SPIKE))
            ax.scatter(times, rows, s=1.2, c=colors, marker="|", linewidths=0.6)
        ax.set_xlim(t - self.history_s, t + 0.02)
        ax.set_ylim(-2, self.roles.size + 2)
        ax.set_yticks([])
        ax.set_xlabel("seconds", color=MUTED, fontsize=7)

        ax = fig.add_subplot(grid[2, 1])
        self._style(ax, "DESCENDING NEURONS  (Hz)")
        names = list(self.brain.output_specs)
        values = [command.rates.get(name, 0.0) for name in names]
        ax.bar(range(len(names)), values, color=OUTPUT, width=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7, color=TEXT, fontfamily="monospace")
        ax.set_ylim(0, max(100.0, max(values, default=0.0) * 1.1))
        ax.grid(axis="y", color=GRID, lw=0.5)

        ax = fig.add_subplot(grid[0:2, 2])
        self._style(ax, "CRA373 SCENE  (top view)")
        if len(self.trail) > 1:
            trail = np.asarray(self.trail)
            ax.plot(trail[:, 0], trail[:, 1], color=SPIKE, lw=1.2, alpha=0.7)
        ax.plot(x, y, "o", color=OUTPUT, ms=7)
        ax.arrow(x, y, 0.35 * np.cos(yaw_rad), 0.35 * np.sin(yaw_rad),
                 color=OUTPUT, width=0.018, head_width=0.12,
                 length_includes_head=True)
        ax.text(x + 0.12, y + 0.12, "CRA373", color=TEXT, fontsize=8,
                fontfamily="monospace")
        if self.trail:
            trail = np.asarray(self.trail)
            span = max(1.25, float(np.ptp(trail[:, 0])) / 2 + 0.5,
                       float(np.ptp(trail[:, 1])) / 2 + 0.5)
            cx = (float(trail[:, 0].min()) + float(trail[:, 0].max())) / 2
            cy = (float(trail[:, 1].min()) + float(trail[:, 1].max())) / 2
            ax.set_xlim(cx - span, cx + span)
            ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(labelbottom=False, labelleft=False)

        ax = fig.add_subplot(grid[2, 2])
        self._style(ax, "CRA373 VELOCITY GOALS")
        values = [command.forward, command.lateral, command.yaw]
        ax.barh(range(3), values,
                color=[SPIKE if value >= 0 else INPUT for value in values], height=0.6)
        ax.set_yticks(range(3))
        ax.set_yticklabels(("forward", "lateral", "yaw"), fontsize=7,
                           color=TEXT, fontfamily="monospace")
        limit = max(1.0, max(abs(value) for value in values) * 1.1)
        ax.set_xlim(-limit, limit)
        ax.axvline(0, color=GRID)
        if command.escape:
            ax.text(0.98, 0.1, "ESCAPE STOP", transform=ax.transAxes,
                    ha="right", color=WARN, fontsize=8, fontweight="bold",
                    fontfamily="monospace")

        ax = fig.add_subplot(grid[2, 0])
        self._style(ax, "COMMAND HISTORY")
        if self.commands:
            commands = np.asarray(self.commands)
            ax.plot(commands[:, 0], commands[:, 1], color=SPIKE, lw=1.1, label="forward")
            ax.plot(commands[:, 0], commands[:, 2], color=INPUT, lw=1.1, label="lateral")
            ax.plot(commands[:, 0], commands[:, 3], color=OUTPUT, lw=1.1, label="yaw")
        ax.set_xlim(max(0, t - 10), max(10, t))
        ax.set_ylim(-1.1, 1.1)
        ax.grid(color=GRID, lw=0.5)
        ax.legend(loc="upper left", fontsize=6, frameon=False, labelcolor=TEXT)

        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self) -> None:
        plt.close(self.fig)
