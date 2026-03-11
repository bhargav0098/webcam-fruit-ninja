"""Fruit Ninja — single-hand, player-selected level, no waves."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .asset_utils import AssetManager
from .game_objects import Fruit, Splash, spawn_fruit
from .hand_tracker import HandLandmark, HandTracker
from .ui_overlay import (
    draw_centered_banner, draw_freeze_bar, draw_level_intro,
    draw_level_select, draw_lives, draw_prompt, draw_score_panel,
    draw_slice_flash, draw_timer, draw_trail, overlay_sprite,
)

ST_SELECT  = "select"    
ST_INTRO   = "intro"     
ST_PLAYING = "playing"   
ST_RESULT  = "result"    

INTRO_DUR  = 2.5         


@dataclass
class FloatingText:
    text: str
    pos: np.ndarray
    color: Tuple[int, int, int]
    born: float
    dur: float  = 1.0
    scale: float = 1.1

    def alpha(self, now: float) -> float:
        return max(0.0, 1.0 - (now - self.born) / self.dur)


@dataclass
class SliceFlash:
    center: Tuple[int, int]
    radius: int
    color:  Tuple[int, int, int]
    born:   float
    dur:    float = 0.22

    def alpha(self, now: float) -> float:
        return max(0.0, 1.0 - (now - self.born) / self.dur)


class FruitNinjaApp:

    
    SELECT_HOLD = 1.5   

    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

        self.tracker = HandTracker()
        self.assets  = AssetManager()
        self.rng     = np.random.default_rng()

        self.trail: Deque[Tuple[int, int, float]] = deque(maxlen=config.TRAIL_HISTORY)

        self.fruits:        List[Fruit]        = []
        self.splashes:      List[Splash]       = []
        self.slice_flashes: List[SliceFlash]   = []
        self.popups:        List[FloatingText]  = []

        self.best_score = 0
        self.state      = ST_SELECT
        self.chosen_level = 1

        
        self._hover_card      = -1
        self._hover_start     = 0.0
        self._select_cards    = {}   

        
        self._prompt      = ""
        self._prompt_until = 0.0

        
        self.score     = 0
        self.lives     = 5
        self.combo     = 1.0
        self.combo_chain = 0
        self.last_slice  = 0.0
        self.level_start = 0.0
        self.last_spawn  = 0.0
        self.freeze_active = False
        self.freeze_until  = 0.0
        self._last_life_milestone = 0
        self._warned_10 = False
        self._warned_5  = False
        self.intro_start = 0.0
        self.level_cfg: dict = {}

    

    def _set_prompt(self, text: str, now: float, dur: float = 2.5) -> None:
        self._prompt       = text
        self._prompt_until = now + dur

    def _begin_level(self, level: int, now: float) -> None:
        self.chosen_level = level
        self.level_cfg    = config.LEVELS[level]
        self.state        = ST_INTRO
        self.intro_start  = now

        self.score     = 0
        self.lives     = self.level_cfg["lives"]
        self.combo     = 1.0
        self.combo_chain = 0
        self.last_slice  = 0.0
        self.freeze_active = False
        self.freeze_until  = 0.0
        self._last_life_milestone = 0
        self._warned_10 = False
        self._warned_5  = False

        self.fruits.clear()
        self.splashes.clear()
        self.slice_flashes.clear()
        self.popups.clear()
        self.trail.clear()

    def _start_play(self, now: float) -> None:
        self.state      = ST_PLAYING
        self.level_start = now
        self.last_spawn  = now
        self._set_prompt("Swipe your index finger to slice fruit!", now, 3.5)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        last_t = time.perf_counter()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            dt  = min(now - last_t, 1/30)
            last_t = now

            lm = self.tracker.locate_index_finger(frame)
            self._update_trail(lm, now)
            self._tick(dt, now, frame, lm)
            self._render(frame, lm, now)

            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            self._handle_key(key, now)

        self.tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()

    def _handle_key(self, key: int, now: float) -> None:
        if key == ord(" ") and self.state == ST_INTRO:
            self._start_play(now)
        if key == ord("r") and self.state == ST_RESULT:
            self.state = ST_SELECT
        if key == ord("1") and self.state == ST_SELECT:
            self._begin_level(1, now)
        if key == ord("2") and self.state == ST_SELECT:
            self._begin_level(2, now)
        if key == ord("3") and self.state == ST_SELECT:
            self._begin_level(3, now)

    # ── Tick ─────────────────────────────────────────────────────────────

    def _tick(self, dt: float, now: float, frame, lm: Optional[HandLandmark]) -> None:
        if self.state == ST_SELECT:
            self._tick_select(lm, now)
        elif self.state == ST_INTRO:
            if now - self.intro_start >= INTRO_DUR:
                self._start_play(now)
        elif self.state == ST_PLAYING:
            self._tick_play(dt, now, frame.shape[1], frame.shape[0])

    def _tick_select(self, lm: Optional[HandLandmark], now: float) -> None:
        """Detect which card the finger hovers over and hold to select."""
        if not self._select_cards or lm is None:
            self._hover_card  = -1
            self._hover_start = now
            return

        fx, fy = lm.point
        hit = -1
        for lvl, (cx, cy, cw, ch) in self._select_cards.items():
            if cx <= fx <= cx + cw and cy <= fy <= cy + ch:
                hit = lvl
                break

        if hit != self._hover_card:
            self._hover_card  = hit
            self._hover_start = now
        elif hit != -1 and (now - self._hover_start) >= self.SELECT_HOLD:
            self._begin_level(hit, now)

    def _tick_play(self, dt: float, now: float, W: int, H: int) -> None:
        cfg = self.level_cfg
        elapsed   = now - self.level_start
        remaining = cfg["time_limit"] - elapsed

        if remaining <= 0:
            self._end_game(now)
            return

        
        if remaining <= 10 and not self._warned_10:
            self._warned_10 = True
            self._set_prompt("10 seconds left — slice faster!", now, 3.5)
        if remaining <= 5 and not self._warned_5:
            self._warned_5 = True
            self._set_prompt("HURRY! Last 5 seconds!", now, 5.0)

        if self.freeze_active and now >= self.freeze_until:
            self.freeze_active = False

        self._spawn(now, W, H)
        self._update_fruits(dt, now, W, H)
        self._detect_slices(now)
        self._check_life_milestone(now)
        self._prune(now)



    def _end_game(self, now: float) -> None:
        self.best_score = max(self.best_score, self.score)
        self.state = ST_RESULT

    

    def _spawn(self, now: float, W: int, H: int) -> None:
        cfg      = self.level_cfg
        interval = cfg["spawn_interval"]
        while now - self.last_spawn >= interval:
            self.fruits.append(spawn_fruit(W, H, cfg, self.rng, now))
            self.last_spawn += interval
            
            if self.rng.random() < cfg["multi_spawn_chance"]:
                self.fruits.append(spawn_fruit(W, H, cfg, self.rng, now))

    

    def _update_fruits(self, dt: float, now: float, W: int, H: int) -> None:
        g  = self.level_cfg["gravity"]
        sf = config.FREEZE_SPEED_FACTOR if self.freeze_active else 1.0
        for f in self.fruits:
            if f.sliced_at is None:
                f.velocity[1] += g * dt * sf
                f.position    += f.velocity * dt * sf
            else:
                for hf in f.halves:
                    hf["vel"][1] += g * dt * sf
                    hf["pos"]    += hf["vel"] * dt * sf
                    hf["angle"]  += hf["spin"] * dt
            if f.sliced_at and now - f.sliced_at > 0.50:
                f.removed = True
            elif f.sliced_at is None and f.position[1] - f.radius > H + 20:
                f.removed = True
                if not f.is_bomb() and not f.is_freeze():
                    self._miss(now)
        self.fruits = [f for f in self.fruits if not f.removed]

    def _miss(self, now: float) -> None:
        self.lives      -= 1
        self.combo       = 1.0
        self.combo_chain = 0
        if self.lives <= 0:
            self._end_game(now)
        else:
            self._set_prompt(f"Missed!  {self.lives} {'life' if self.lives==1 else 'lives'} left", now, 1.8)

    

    def _detect_slices(self, now: float) -> None:
        hist = list(self.trail)
        if len(hist) < 2:
            return
        for fruit in self.fruits:
            if fruit.sliced_at is not None:
                continue
            for i in range(len(hist) - 1):
                x1, y1, t1 = hist[i]
                x2, y2, t2 = hist[i+1]
                elapsed = t2 - t1
                if elapsed <= 0:
                    continue
                speed = np.linalg.norm([x2-x1, y2-y1]) / elapsed
                if speed < self.level_cfg["slice_threshold"]:
                    continue
                dist = _seg_dist(fruit.position,
                                 np.array([x1, y1], np.float32),
                                 np.array([x2, y2], np.float32))
                if dist <= fruit.radius * 0.92:
                    self._slice(fruit, now)
                    break

    def _slice(self, fruit: Fruit, now: float) -> None:
        fruit.mark_sliced(now)
        pos = tuple(fruit.position.astype(int))

        
        if fruit.is_freeze():
            self.freeze_active = True
            self.freeze_until  = now + config.FREEZE_DURATION
            self.splashes.append(Splash(pos, (255, 170, 0), now, 0.5))
            self.slice_flashes.append(SliceFlash(pos, 52, (255, 200, 0), now))
            self.popups.append(FloatingText("FREEZE!", fruit.position.copy(),
                                            (255, 210, 0), now, 1.4, 1.3))
            self._set_prompt("Freeze! All fruit slowed down", now, 2.5)
            return

        col = (30, 30, 200) if fruit.is_bomb() else (50, 210, 50)
        self.splashes.append(Splash(pos, col, now))
        self.slice_flashes.append(SliceFlash(pos, int(fruit.radius), col, now))

        
        if fruit.is_bomb():
            self.popups.append(FloatingText("BOMB! -1 LIFE", fruit.position.copy(),
                                            (30, 30, 220), now, 1.3, 1.2))
            self.combo       = 1.0
            self.combo_chain = 0
            self.lives       = max(0, self.lives - 1)
            if self.lives <= 0:
                self._end_game(now)
            else:
                self._set_prompt(f"Bomb hit!  {self.lives} {'life' if self.lives==1 else 'lives'} left", now, 2.0)
            return

        
        if now - self.last_slice < config.COMBO_WINDOW:
            self.combo       = min(config.MAX_COMBO, self.combo + 0.35)
            self.combo_chain += 1
        else:
            self.combo       = 1.0
            self.combo_chain = 1
        self.last_slice = now

        mult  = config.GOLDEN_FRUIT_MULTIPLIER if fruit.is_golden else 1.0
        pts   = int(fruit.value * self.combo * mult)
        self.score      += pts
        self.best_score  = max(self.best_score, self.score)

        p_col = (0, 215, 255) if fruit.is_golden else (255, 215, 0)
        label = f"+{pts} GOLDEN!" if fruit.is_golden else f"+{pts}"
        self.popups.append(FloatingText(label, fruit.position.copy(), p_col, now, 1.0,
                                        scale=1.5 if fruit.is_golden else 1.05))
        if fruit.is_golden:
            self._set_prompt("Golden fruit — 3x points!", now, 1.5)

        if self.combo_chain >= 3:
            self.popups.append(FloatingText(
                f"{self.combo_chain}x COMBO!",
                fruit.position.copy() - np.array([0, 52]),
                (255, 130, 20), now, 1.3, 1.3))
            if self.combo_chain == 3:
                self._set_prompt("Combo! Keep slicing fast!", now, 1.5)

    def _check_life_milestone(self, now: float) -> None:
        m = self.score // config.EXTRA_LIFE_SCORE
        if m > self._last_life_milestone:
            self._last_life_milestone = m
            self.lives += 1
            self.popups.append(FloatingText(
                "+1 LIFE!", np.array([640.0, 360.0]), (80, 255, 80), now, 1.8, 1.4))
            self._set_prompt("Bonus life earned!", now, 2.0)

    

    def _update_trail(self, lm: Optional[HandLandmark], now: float) -> None:
        if lm is None:
            self.trail.clear()
        else:
            self.trail.append((*lm.point, now))



    def _prune(self, now: float) -> None:
        self.splashes      = [s for s in self.splashes      if s.alpha(now) > 0.03]
        self.slice_flashes = [f for f in self.slice_flashes if f.alpha(now) > 0.03]
        self.popups        = [p for p in self.popups        if p.alpha(now) > 0.03]

    

    def _render(self, frame: np.ndarray, lm: Optional[HandLandmark], now: float) -> None:

        
        if self.state == ST_SELECT:
            self._select_cards = draw_level_select(frame, self._hover_card,
                                                   config.LEVELS, now)
            
            if self._hover_card != -1:
                held = min(1.0, (now - self._hover_start) / self.SELECT_HOLD)
                cx, cy, cw, ch = self._select_cards[self._hover_card]
                bary = cy + ch + 6
                cv2.rectangle(frame, (cx, bary), (cx+cw, bary+8), (60, 60, 60), -1)
                col = config.LEVELS[self._hover_card]["color"]
                cv2.rectangle(frame, (cx, bary), (cx+int(cw*held), bary+8), col, -1)
                hint = "Hold to select..."
                hs = cv2.getTextSize(hint, cv2.FONT_HERSHEY_PLAIN, 1.1, 1)[0]
                cv2.putText(frame, hint,
                            (cx+(cw-hs[0])//2, bary+24),
                            cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1, cv2.LINE_AA)

            
            if lm:
                cv2.circle(frame, lm.point, 14, (255, 255, 255), 2)
                cv2.circle(frame, lm.point,  5, (200, 255, 100), -1)

            
            cv2.putText(frame, "or press  1 / 2 / 3  to select",
                        (frame.shape[1]//2 - 160, frame.shape[0] - 24),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (140, 140, 140), 1, cv2.LINE_AA)
            return

        
        if self.state == ST_INTRO:
            prog = (now - self.intro_start) / INTRO_DUR
            draw_level_intro(frame, self.chosen_level, self.level_cfg, prog)
            return

        
        cfg = self.level_cfg

        
        for fruit in self.fruits:
            sprite = self.assets.get(fruit.sprite_name)
            scale  = (fruit.radius * 2) / sprite.shape[0]
            if fruit.sliced_at is None:
                tint = (80, 255, 255) if fruit.is_golden else None
                overlay_sprite(frame, sprite, tuple(fruit.position.astype(int)), scale, tint)
                if fruit.is_bomb():
                    cv2.circle(frame, tuple(fruit.position.astype(int)),
                               int(fruit.radius * 1.08), (0, 0, 200), 2)
                elif fruit.is_freeze():
                    r = int(fruit.radius * 1.08)
                    cv2.circle(frame, tuple(fruit.position.astype(int)), r, (255, 170, 0), 2)
                    cv2.putText(frame, "ICE",
                                (int(fruit.position[0])-14, int(fruit.position[1])-r-5),
                                cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 230, 255), 1, cv2.LINE_AA)
                elif fruit.is_golden:
                    pulse = int(3 + 2 * abs(np.sin(now * 7)))
                    cv2.circle(frame, tuple(fruit.position.astype(int)),
                               int(fruit.radius * 1.12), (0, 215, 255), pulse)
            else:
                hs = scale * 0.52
                for hf in fruit.halves:
                    ch2, cw2 = sprite.shape[0]//2, sprite.shape[1]//2
                    M = cv2.getRotationMatrix2D((cw2, ch2), hf["angle"], hs)
                    rot = cv2.warpAffine(sprite, M, (sprite.shape[1], sprite.shape[0]),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0,0,0,0))
                    overlay_sprite(frame, rot, tuple(hf["pos"].astype(int)), 1.0)

        
        draw_trail(frame, list(self.trail), now, config.TRAIL_MAX_AGE)

        
        for sf in self.slice_flashes:
            draw_slice_flash(frame, sf.center, sf.radius, sf.color, sf.alpha(now))

        
        for sp in self.splashes:
            a = sp.alpha(now)
            cv2.circle(frame, sp.position, int(65*a+10), sp.color, 5)

        
        self._draw_popups(frame, now)

        
        remaining = max(0.0, cfg["time_limit"] - (now - self.level_start))
        draw_timer(frame, remaining, cfg["time_limit"], cfg["color"])
        draw_score_panel(frame, self.score, cfg["score_target"],
                         f"LVL {self.chosen_level}  {cfg['name']}", cfg["color"],
                         self.combo, self.freeze_active)
        draw_lives(frame, self.lives, self.assets.get("life.png"))

        if self.freeze_active:
            draw_freeze_bar(frame, self.freeze_until - now, config.FREEZE_DURATION)

        
        if lm:
            cv2.circle(frame, lm.point, 13, (255, 255, 255), 2)
            cv2.circle(frame, lm.point,  4, (180, 255, 90),  -1)

        
        draw_prompt(frame, self._prompt, now, self._prompt_until)

        
        if self.state == ST_RESULT:
            passed = self.score >= cfg["score_target"] and self.lives > 0
            if passed:
                title = "YOU WIN!"
                line2 = f"Score: {self.score} / {cfg['score_target']}   Best: {self.best_score}   Press R"
                col   = (60, 220, 60)
            else:
                reason = "Out of lives!" if self.lives <= 0 else "Time's up!" 
                title  = f"GAME OVER  —  {reason}"
                short  = cfg['score_target'] - self.score
                line2  = (f"Score: {self.score}  ({short} pts short)   Press R"
                          if self.score < cfg["score_target"]
                          else f"Score: {self.score}   Press R to play again")
                col = (30, 30, 210)
            draw_centered_banner(frame, title, line2, col, 0.42)

    def _draw_popups(self, frame: np.ndarray, now: float) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        for p in self.popups:
            a = p.alpha(now)
            if a <= 0:
                continue
            col = tuple(int(c * a) for c in p.color)
            y   = int(p.pos[1]) - int(60 * (1 - a))
            x   = int(p.pos[0])
            cv2.putText(frame, p.text, (x+2, y+2), font, p.scale, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, p.text, (x,   y),   font, p.scale, col,     2, cv2.LINE_AA)




def _seg_dist(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    if np.allclose(ab, 0):
        return float(np.linalg.norm(point - a))
    t  = np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * ab)))


def main() -> None:
    FruitNinjaApp().run()


if __name__ == "__main__":
    main()
