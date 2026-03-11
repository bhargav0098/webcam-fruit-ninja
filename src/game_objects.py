"""Game entities and physics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from . import config


@dataclass
class Fruit:
    kind: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    sprite_name: str
    born_at: float
    value: int
    is_golden: bool = False
    sliced_at: float | None = None
    removed: bool = False
    halves: list = field(default_factory=list)

    def is_bomb(self)   -> bool: return self.kind == "bomb"
    def is_freeze(self) -> bool: return self.kind == "freeze"

    def mark_sliced(self, t: float) -> None:
        self.sliced_at = t
        angle = np.random.uniform(0, np.pi)
        speed = np.random.uniform(160, 300)
        for sign in (-1, 1):
            offset = np.array([np.cos(angle), np.sin(angle)]) * self.radius * 0.35
            vel = np.array([
                np.cos(angle) * speed * sign + self.velocity[0] * 0.25,
                -abs(np.sin(angle) * speed) + self.velocity[1] * 0.20,
            ])
            self.halves.append({
                "pos":   self.position.copy() + offset,
                "vel":   vel,
                "angle": np.degrees(angle),
                "spin":  np.random.uniform(160, 380) * sign,
            })


@dataclass
class Splash:
    position: Tuple[int, int]
    color: Tuple[int, int, int]
    born_at: float
    duration: float = 0.40

    def alpha(self, now: float) -> float:
        return max(0.0, 1.0 - (now - self.born_at) / self.duration)


def spawn_fruit(width: int, height: int, level_cfg: dict,
                rng: np.random.Generator, now: float) -> Fruit:
    roll = rng.random()
    bomb_chance = level_cfg["bomb_chance"]

    if roll < bomb_chance:
        kind, sprite, scale, value = "bomb",   "bomb.png",   1.1, 0
        is_golden = False
    elif roll < bomb_chance + 0.03:
        kind, sprite, scale, value = "freeze", "splash.png", 0.9, 0
        is_golden = False
    else:
        sprite, scale, value = config.FRUIT_CATALOG[rng.integers(len(config.FRUIT_CATALOG))]
        kind = "fruit"
        is_golden = rng.random() < config.GOLDEN_FRUIT_CHANCE

    radius = rng.uniform(config.FRUIT_MIN_RADIUS, config.FRUIT_MAX_RADIUS) * scale
    x = rng.uniform(radius, width - radius)
    y = height + radius + rng.uniform(5, 25)
    apex = rng.uniform(height * 0.10, height * 0.40)
    vy = -np.sqrt(2 * level_cfg["gravity"] * max(80.0, y - apex)) * rng.uniform(0.92, 1.05)
    vx = rng.uniform(-280, 280)

    return Fruit(
        kind=kind,
        position=np.array([x, y], dtype=np.float32),
        velocity=np.array([vx, vy], dtype=np.float32),
        radius=radius,
        sprite_name=sprite,
        born_at=now,
        value=int(value),
        is_golden=is_golden,
    )
