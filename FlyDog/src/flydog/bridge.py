"""Shared fly-brain controller for the preview and Isaac Sim demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flydrones.brain import Brain, load_connectome
from flydrones.config import load_config
from flydrones.motor import MotorDecoder
from flydrones.senses import GestureIllusion, InputEncoder, Retina
from flydrones.senses.gestures import ScriptedGestures, demo_timeline


@dataclass(frozen=True)
class DogCommand:
    forward: float
    lateral: float
    yaw: float
    escape: bool
    gesture: str
    rates: dict[str, float]


class FlyDogBridge:
    """Use a fly brain to command velocity; CRA373 policy handles 12 joints."""

    def __init__(self, *, brain_source: str = "minifly", fly_config: str | Path | None = None,
                 forward_mps: float = 0.8, lateral_mps: float = 0.5,
                 yaw_rps: float = 1.0, seed: int = 0):
        if min(forward_mps, lateral_mps, yaw_rps) < 0:
            raise ValueError("command scales must be nonnegative")
        self.cfg = load_config(fly_config, overrides={"brain": {"source": brain_source, "seed": seed},
                                                     "decoder": {"escape": {"mode": "brake"}}})
        self.brain = Brain(load_connectome(brain_source), self.cfg, seed=seed)
        self.retina = Retina.from_config(self.cfg)
        self.encoder = InputEncoder(self.brain.connectome, self.cfg)
        self.illusion = GestureIllusion()
        self.gestures = ScriptedGestures(demo_timeline())
        self.decoder = MotorDecoder(self.cfg)
        self.forward_mps = forward_mps
        self.lateral_mps = lateral_mps
        self.yaw_rps = yaw_rps
        # Expose the latest sensory state for live visualizers.  Keeping this
        # on the bridge avoids running the retina/illusion pipeline twice.
        self.last_vision = self.retina.encode(None)
        self.last_gesture = None
        self.last_flight = None

    def warmup(self, dt: float = 0.05) -> None:
        for _ in range(int((self.decoder.settle_s + 0.1) / dt) + 1):
            vision = self.retina.encode(None)
            rates = self.brain.tick(self.encoder.encode(vision, 0.0), ms=dt * 1000)
            self.decoder.update(rates, dt)

    def tick(self, t: float, dt: float, *, yaw_rate_dps: float = 0.0) -> DogCommand:
        gesture = self.gestures.read(t)
        vision = self.illusion.apply(self.retina.encode(None), gesture, t)
        rates = self.brain.tick(self.encoder.encode(vision, yaw_rate_dps), ms=dt * 1000)
        flight = self.decoder.update(rates, dt)
        self.last_vision = vision
        self.last_gesture = gesture
        self.last_flight = flight
        # FlyDrones lift becomes forward walking; looming means stop on the ground.
        # FlyDrones +yaw/+lateral mean right. CRA373's control frame uses +yaw/+y
        # for left, so both signed axes must be inverted.
        forward = max(-1.0, min(1.0, flight.throttle + flight.forward)) * self.forward_mps
        lateral = -max(-1.0, min(1.0, flight.lateral)) * self.lateral_mps
        yaw = -max(-1.0, min(1.0, flight.yaw)) * self.yaw_rps
        if flight.escape:
            forward = lateral = yaw = 0.0
        return DogCommand(forward, lateral, yaw, flight.escape, gesture.label, rates)
