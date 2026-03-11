"""Runtime asset generation/loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .config import ASSET_DIR

ASSET_VERSION = "3"


class AssetManager:
    """Ensures PNG sprites exist and loads them with alpha."""

    def __init__(self, asset_dir: Path = ASSET_DIR) -> None:
        self.asset_dir = Path(asset_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.asset_dir / ".asset_version"
        self._cache: Dict[str, np.ndarray] = {}
        self._bootstrap_assets()

    def get(self, name: str) -> np.ndarray:
        if name not in self._cache:
            sprite = cv2.imread(str(self.asset_dir / name), cv2.IMREAD_UNCHANGED)
            if sprite is None:
                raise FileNotFoundError(f"Missing sprite: {name}")
            self._cache[name] = sprite
        return self._cache[name]


    def _bootstrap_assets(self) -> None:
        current = self._read_version()
        force = current != ASSET_VERSION
        self._ensure_core_assets(force=force)
        if force:
            self.version_file.write_text(ASSET_VERSION)

    def _read_version(self) -> str:
        try:
            return self.version_file.read_text().strip()
        except FileNotFoundError:
            return ""

    def _ensure_core_assets(self, *, force: bool = False) -> None:
        generators = {
            "watermelon.png": self._draw_watermelon,
            "banana.png": self._draw_banana,
            "orange.png": self._draw_orange,
            "apple.png": self._draw_apple,
            "pineapple.png": self._draw_pineapple,
            "grapes.png": self._draw_grapes,
            "strawberry.png": self._draw_strawberry,
            "peach.png": self._draw_peach,
            "cherries.png": self._draw_cherries,
            "mango.png": self._draw_mango,
            "kiwi.png": self._draw_kiwi,
            "pear.png": self._draw_pear,
            "lemon.png": self._draw_lemon,
            "melon.png": self._draw_melon,
            "blueberries.png": self._draw_blueberries,
            "bomb.png": self._draw_bomb,
            "splash.png": self._draw_splash,
            "life.png": self._draw_life,
        }
        for filename, drawer in generators.items():
            path = self.asset_dir / filename
            if not force and path.exists():
                continue
            sprite = drawer()
            cv2.imwrite(str(path), sprite, [cv2.IMWRITE_PNG_COMPRESSION, 2])

    def _add_highlight_arc(self, canvas: np.ndarray, center, radius: int) -> None:
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        axes = (int(radius * 0.5), int(radius * 0.8))
        cv2.ellipse(mask, (center[0] - radius // 4, center[1] - radius // 3), axes, -20, 200, 340, 255, -1)
        highlight = mask.astype(bool)
        if not np.any(highlight):
            return
        canvas[highlight, :3] = np.clip(canvas[highlight, :3] * 0.4 + 255 * 0.6, 0, 255)
        canvas[highlight, 3] = np.maximum(canvas[highlight, 3], 200)

    def _circle_canvas(self, size: int = 256) -> np.ndarray:
        canvas = np.zeros((size, size, 4), dtype=np.uint8)
        return canvas

    def _draw_watermelon(self) -> np.ndarray:
        size = 320
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2)
        outer = size // 2 - 6
        rind_colors = [
            (18, 70, 15, 255),
            (28, 110, 30, 255),
            (35, 150, 35, 255),
        ]
        flesh_colors = [
            (30, 30, 130, 255),
            (60, 40, 200, 255),
            (0, 170, 0, 255),
        ]
        for idx, color in enumerate(rind_colors):
            cv2.circle(canvas, center, outer - idx * 10, color, -1)
        for idx, color in enumerate(flesh_colors):
            cv2.circle(canvas, center, outer - 40 - idx * 22, color, -1)

        rng = np.random.default_rng(42)
        for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False):
            offset = outer - 85 + rng.uniform(-6, 6)
            x = int(center[0] + offset * np.cos(angle))
            y = int(center[1] + offset * np.sin(angle))
            cv2.ellipse(canvas, (x, y), (10, 22), np.degrees(angle), 0, 360, (25, 25, 25, 245), -1)

        self._add_highlight_arc(canvas, center, outer)
        return canvas

    def _draw_banana(self) -> np.ndarray:
        size = 360
        canvas = self._circle_canvas(size)
        mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2 - 10, size // 2 + 70)
        axes = (170, 90)
        cv2.ellipse(mask, center, axes, -48, 200, 340, 255, thickness=70)

        inner = np.zeros_like(mask)
        cv2.ellipse(inner, (center[0] - 16, center[1] - 12), (axes[0] - 30, axes[1] - 24), -52, 205, 335, 255, thickness=40)

        canvas[..., 0][mask > 0] = 40
        canvas[..., 1][mask > 0] = 210
        canvas[..., 2][mask > 0] = 255
        canvas[..., 3][mask > 0] = 255

        canvas[..., 0][inner > 0] = 80
        canvas[..., 1][inner > 0] = 240
        canvas[..., 2][inner > 0] = 255

        for offset in (-110, 120):
            cv2.circle(canvas, (center[0] + offset, center[1] - 40), 30, (40, 28, 10, 255), -1)

        shine = np.zeros_like(mask)
        cv2.ellipse(shine, (center[0] - 70, center[1] - 50), (80, 45), -48, 210, 330, 255, thickness=20)
        highlight = shine > 0
        canvas[highlight, :3] = np.clip(canvas[highlight, :3] * 0.5 + 255 * 0.5, 0, 255)
        canvas[highlight, 3] = 255
        return canvas

    def _draw_orange(self) -> np.ndarray:
        size = 260
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2)
        for radius in range(size // 2, 20, -8):
            color = (20 + radius // 3, 100 + radius // 4, 230, 255)
            cv2.circle(canvas, center, radius, color, -1)

        pores = np.zeros_like(canvas)
        rng = np.random.default_rng(7)
        for _ in range(140):
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.2, 0.9)
            r = int((size // 2 - 18) * dist)
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            cv2.circle(pores, (x, y), 2, (255, 190, 120, 80), -1)
        mask = pores[..., 3] > 0
        canvas[mask] = np.clip(canvas[mask].astype(int) + pores[mask].astype(int), 0, 255).astype(np.uint8)

        leaf = np.zeros_like(canvas)
        cv2.ellipse(leaf, (center[0] + 30, center[1] - 70), (60, 30), -20, 0, 360, (20, 150, 30, 255), -1)
        mask = leaf[..., 3] > 0
        canvas[mask] = leaf[mask]
        self._add_highlight_arc(canvas, center, size // 2 - 10)
        return canvas

    def _draw_apple(self) -> np.ndarray:
        size = 260
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2 + 20)
        gradients = [
            (size // 2 - 10, (40, 0, 160, 255)),
            (size // 2 - 30, (60, 0, 220, 255)),
            (size // 2 - 55, (90, 20, 255, 255)),
        ]
        for radius, color in gradients:
            cv2.circle(canvas, center, radius, color, -1)

        stem_base = (center[0], center[1] - 80)
        cv2.line(canvas, stem_base, (stem_base[0], stem_base[1] - 40), (40, 25, 10, 255), 10)
        cv2.ellipse(canvas, (stem_base[0] + 40, stem_base[1] - 30), (60, 25), -45, 0, 360, (30, 150, 70, 255), -1)
        self._add_highlight_arc(canvas, center, size // 2 - 20)
        return canvas

    def _draw_pineapple(self) -> np.ndarray:
        size = 320
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2 + 30)
        base_color = (40, 180, 255, 255)
        cv2.ellipse(canvas, center, (100, 130), 0, 0, 360, base_color, -1)

        for angle in range(-50, 60, 20):
            cv2.ellipse(canvas, center, (100, 130), angle, 0, 360, (60, 120, 200, 255), 10)

        for y in range(-3, 4):
            for x in range(-2, 3):
                px = int(center[0] + x * 35)
                py = int(center[1] + y * 32)
                cv2.circle(canvas, (px, py), 8, (30, 90, 140, 255), -1)

        for i in range(6):
            tip = (center[0] + i * 15 - 40, center[1] - 150 - i * 12)
            cv2.ellipse(canvas, tip, (30, 70), -30 + i * 12, 0, 360, (20, 140, 40, 255), -1)
        self._add_highlight_arc(canvas, center, 120)
        return canvas

    def _draw_grapes(self) -> np.ndarray:
        size = 260
        canvas = self._circle_canvas(size)
        rng = np.random.default_rng(13)
        base = np.array([size // 2, size // 2 + 10])
        for row in range(4):
            for col in range(4 - row // 2):
                offset = np.array([
                    (col - 1.5 + row * 0.2) * 40,
                    (row - 1.5) * 35,
                ])
                center = tuple((base + offset).astype(int))
                radius = int(28 + rng.uniform(-4, 4))
                color = (80, 0, 200 + row * 12, 255)
                cv2.circle(canvas, center, radius, color, -1)
        stem_start = (size // 2, size // 2 - 70)
        cv2.line(canvas, (stem_start[0], stem_start[1] - 40), stem_start, (40, 80, 30, 255), 10)
        cv2.ellipse(canvas, (stem_start[0] + 30, stem_start[1] - 50), (50, 20), -20, 0, 360, (20, 140, 40, 255), -1)
        self._add_highlight_arc(canvas, (size // 2, size // 2), 110)
        return canvas

    def _draw_strawberry(self) -> np.ndarray:
        size = 240
        canvas = self._circle_canvas(size)
        contour = np.array(
            [
                [120, 40],
                [200, 90],
                [190, 170],
                [120, 210],
                [50, 170],
                [40, 90],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [contour], (40, 20, 200, 255))
        for x in range(55, 185, 30):
            for y in range(80, 180, 30):
                cv2.circle(canvas, (x, y), 5, (30, 160, 255, 255), -1)
        leaf_center = (120, 40)
        for angle in range(0, 360, 60):
            cv2.ellipse( canvas, leaf_center, (30, 18), angle, 0, 360, (20, 160, 40, 255), -1)
        self._add_highlight_arc(canvas, (110, 120), 80)
        return canvas

    def _draw_peach(self) -> np.ndarray:
        size = 260
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2 + 15)
        colors = [
            (20, 80, 255, 255),
            (40, 120, 220, 255),
            (60, 150, 200, 255),
        ]
        for idx, color in enumerate(colors):
            cv2.circle(canvas, (center[0] - 20, center[1]), size // 2 - idx * 20, color, -1)
            cv2.circle(canvas, (center[0] + 20, center[1]), size // 2 - idx * 20, color, -1)
        cv2.line(canvas, (center[0], center[1] - 60), (center[0], center[1] + 60), (120, 80, 60, 255), 4)
        cv2.ellipse(canvas, (center[0] - 10, center[1] - 110), (45, 22), -30, 0, 360, (20, 150, 60, 255), -1)
        self._add_highlight_arc(canvas, center, 110)
        return canvas

    def _draw_cherries(self) -> np.ndarray:
        size = 220
        canvas = self._circle_canvas(size)
        centers = [(90, 150), (140, 150)]
        for c in centers:
            cv2.circle(canvas, c, 45, (40, 0, 200, 255), -1)
            cv2.circle(canvas, (c[0] - 10, c[1] - 10), 30, (60, 20, 220, 255), -1)
        cv2.line(canvas, (centers[0][0], centers[0][1] - 60), (110, 40), (30, 80, 40, 255), 8)
        cv2.line(canvas, (centers[1][0], centers[1][1] - 60), (110, 40), (30, 80, 40, 255), 8)
        cv2.ellipse(canvas, (120, 30), (25, 18), 0, 0, 360, (20, 150, 40, 255), -1)
        self._add_highlight_arc(canvas, (120, 150), 70)
        return canvas

    def _draw_mango(self) -> np.ndarray:
        size = 260
        canvas = self._circle_canvas(size)
        center = (size // 2 + 20, size // 2 + 20)
        cv2.ellipse(canvas, center, (90, 120), -25, 0, 360, (30, 140, 255, 255), -1)
        cv2.ellipse(canvas, (center[0] - 20, center[1] - 20), (70, 90), -25, 0, 360, (0, 200, 255, 255), -1)
        cv2.circle(canvas, (center[0] - 70, center[1] - 70), 25, (0, 150, 0, 255), -1)
        self._add_highlight_arc(canvas, center, 110)
        return canvas

    def _draw_kiwi(self) -> np.ndarray:
        size = 240
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2)
        cv2.circle(canvas, center, size // 2 - 6, (30, 130, 40, 255), -1)
        cv2.circle(canvas, center, size // 2 - 20, (40, 180, 60, 255), -1)
        cv2.circle(canvas, center, size // 2 - 50, (200, 230, 200, 255), -1)
        for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
            x = int(center[0] + np.cos(angle) * 50)
            y = int(center[1] + np.sin(angle) * 50)
            cv2.circle(canvas, (x, y), 4, (40, 80, 30, 255), -1)
        self._add_highlight_arc(canvas, center, 90)
        return canvas

    def _draw_pear(self) -> np.ndarray:
        size = 260
        canvas = self._circle_canvas(size)
        top = (size // 2, 60)
        bottom = (size // 2, 210)
        cv2.ellipse(canvas, bottom, (60, 70), 0, 0, 360, (20, 180, 120, 255), -1)
        cv2.ellipse(canvas, top, (40, 50), 0, 0, 360, (30, 200, 100, 255), -1)
        cv2.line(canvas, (top[0], top[1] - 40), (top[0], top[1] - 80), (50, 30, 0, 255), 10)
        cv2.ellipse(canvas, (top[0] + 30, top[1] - 50), (40, 18), -20, 0, 360, (20, 150, 40, 255), -1)
        self._add_highlight_arc(canvas, (size // 2, size // 2 + 10), 110)
        return canvas

    def _draw_lemon(self) -> np.ndarray:
        size = 240
        canvas = self._circle_canvas(size)
        cv2.ellipse(canvas, (size // 2, size // 2), (100, 70), 0, 0, 360, (20, 220, 255, 255), -1)
        cv2.ellipse(canvas, (size // 2, size // 2), (80, 50), 0, 0, 360, (40, 240, 255, 255), -1)
        cv2.ellipse(canvas, (size // 2, size // 2), (50, 30), 0, 0, 360, (200, 255, 255, 255), -1)
        self._add_highlight_arc(canvas, (size // 2, size // 2), 90)
        return canvas

    def _draw_melon(self) -> np.ndarray:
        size = 280
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2)
        cv2.circle(canvas, center, size // 2 - 6, (30, 120, 60, 255), -1)
        cv2.circle(canvas, center, size // 2 - 20, (50, 170, 90, 255), -1)
        for angle in range(0, 360, 30):
            cv2.ellipse(canvas, center, (size // 2 - 25, size // 2 - 30), angle, 0, 360, (80, 200, 140, 255), 4)
        self._add_highlight_arc(canvas, center, 110)
        return canvas

    def _draw_blueberries(self) -> np.ndarray:
        size = 220
        canvas = self._circle_canvas(size)
        centers = [
            (110, 120),
            (70, 140),
            (140, 160),
            (90, 90),
        ]
        for cx, cy in centers:
            cv2.circle(canvas, (cx, cy), 45, (120, 60, 230, 255), -1)
            cv2.circle(canvas, (cx - 10, cy - 10), 30, (80, 30, 190, 255), -1)
            cv2.circle(canvas, (cx - 3, cy - 3), 10, (40, 20, 140, 255), -1)
        leaf_center = (150, 80)
        cv2.ellipse(canvas, leaf_center, (40, 20), -30, 0, 360, (30, 140, 50, 255), -1)
        self._add_highlight_arc(canvas, (110, 130), 90)
        return canvas

    def _draw_bomb(self) -> np.ndarray:
        size = 300
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2 + 20)
        cv2.circle(canvas, center, 100, (20, 20, 20, 255), -1)
        cv2.circle(canvas, center, 80, (45, 45, 45, 255), -1)
        cv2.circle(canvas, (center[0] + 35, center[1] - 30), 30, (70, 70, 70, 255), -1)
        cv2.rectangle(canvas, (center[0] - 12, center[1] - 130), (center[0] + 12, center[1] - 70), (80, 80, 80, 255), -1)
        cv2.line(canvas, (center[0], center[1] - 125), (center[0] + 50, center[1] - 200), (120, 120, 120, 255), 10)

        fire = np.zeros_like(canvas)
        cv2.circle(fire, (center[0] + 60, center[1] - 210), 25, (0, 220, 255, 255), -1)
        cv2.circle(fire, (center[0] + 70, center[1] - 210), 15, (0, 255, 120, 255), -1)
        mask = fire[..., 3] > 0
        canvas[mask] = fire[mask]
        self._add_highlight_arc(canvas, center, 100)
        return canvas

    def _draw_splash(self) -> np.ndarray:
        size = 196
        canvas = self._circle_canvas(size)
        center = (size // 2, size // 2)
        palette = [
            (0, 190, 0, 100),
            (0, 220, 0, 150),
            (0, 255, 0, 220),
        ]
        for idx, color in enumerate(palette):
            cv2.circle(canvas, center, size // 2 - idx * 18, color, -1)
        canvas[:] = cv2.GaussianBlur(canvas, (0, 0), 9)
        return canvas

    def _draw_life(self) -> np.ndarray:
        canvas = self._circle_canvas(96)
        heart = np.zeros_like(canvas)
        for angle in np.linspace(0, np.pi, 200):
            x = 24 * np.sin(angle) ** 3
            y = 13 * np.cos(angle) - 5 * np.cos(2 * angle) - 2 * np.cos(3 * angle) - np.cos(4 * angle)
            px = int(heart.shape[1] / 2 + x)
            py = int(heart.shape[0] / 2 - y * 2)
            cv2.circle(heart, (px, py), 8, (0, 0, 255, 255), -1)
        mask = heart[..., 3] > 0
        canvas[mask] = heart[mask]
        return canvas
