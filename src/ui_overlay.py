"""HUD drawing helpers."""
from __future__ import annotations
from typing import List, Tuple, Optional
import cv2
import numpy as np

Color = Tuple[int, int, int]
Point = Tuple[int, int]


def overlay_sprite(frame: np.ndarray, sprite: np.ndarray, center: Point,
                   scale: float = 1.0, tint: Optional[Color] = None) -> None:
    if sprite is None or sprite.size == 0:
        return
    h, w = sprite.shape[:2]
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    spr = cv2.resize(sprite, (nw, nh))
    cx, cy = center
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(fw, x0 + nw), min(fh, y0 + nh)
    sx1, sy1 = x1 - x0, y1 - y0
    ow, oh = x2 - x1, y2 - y1
    if ow <= 0 or oh <= 0:
        return
    roi  = frame[y1:y1+oh, x1:x1+ow]
    sroi = spr[sy1:sy1+oh, sx1:sx1+ow]
    if sroi.shape[2] == 4:
        a   = sroi[..., 3:] / 255.0
        rgb = sroi[..., :3].astype(np.float32)
        if tint:
            rgb = np.clip(rgb * (np.array(tint, np.float32) / 128.0), 0, 255)
        roi[:] = np.clip((1 - a) * roi + a * rgb, 0, 255).astype(np.uint8)
    else:
        roi[:] = sroi


def draw_trail(frame: np.ndarray,
               points: List[Tuple[int, int, float]],
               now: float,
               max_age: float = 0.16) -> None:
    recent = [(x, y, t) for x, y, t in points if now - t <= max_age]
    if len(recent) < 2:
        return
    overlay = np.zeros_like(frame)
    for i in range(1, len(recent)):
        x1, y1, t1 = recent[i - 1]
        x2, y2, t2 = recent[i]
        a = max(0.0, 1.0 - (now - t2) / max_age)
        b = int(255 * a)
        cv2.line(overlay, (x1, y1), (x2, y2), (b, b, b), 2, cv2.LINE_AA)
        cv2.line(overlay, (x1, y1), (x2, y2), (0, int(170*a), int(210*a)), 4, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(overlay, (0, 0), 3)
    cv2.addWeighted(frame, 1.0, blurred,  0.45, 0, frame)
    cv2.addWeighted(frame, 1.0, overlay,  0.55, 0, frame)



def draw_slice_flash(frame: np.ndarray, center: Point, radius: int,
                     color: Color, alpha: float) -> None:
    if alpha <= 0:
        return
    ov = frame.copy()
    for ang in range(0, 360, 45):
        rad = np.radians(ang)
        ex = int(center[0] + np.cos(rad) * radius)
        ey = int(center[1] + np.sin(rad) * radius)
        cv2.line(ov, center, (ex, ey), color, 2, cv2.LINE_AA)
    cv2.circle(ov, center, max(4, radius // 3), color, -1)
    cv2.addWeighted(ov, alpha * 0.75, frame, 1 - alpha * 0.75, 0, frame)



def _panel(frame, x, y, w, h, alpha=0.70):
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x+w, y+h), (12, 12, 12), -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)



def draw_score_panel(frame: np.ndarray, score: int, target: int,
                     level_name: str, level_color: Color,
                     combo: float, frozen: bool = False) -> None:
    _panel(frame, 12, 30, 270, 120)
    cv2.rectangle(frame, (12, 30), (282, 150), level_color, 2)
    cv2.putText(frame, f"Score  {score:>4}", (24, 68),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Target {target:>4}", (24, 96),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, level_name, (24, 124),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, level_color, 2, cv2.LINE_AA)
    c_col = (0, 200, 255) if frozen else (255, 185, 50)
    c_lbl = "FROZEN!" if frozen else f"x{combo:.1f}"
    cv2.putText(frame, c_lbl, (148, 124),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, c_col, 2, cv2.LINE_AA)



def draw_timer(frame: np.ndarray, remaining: float, total: float,
               level_color: Color) -> None:
    fw = frame.shape[1]
    bw = fw - 40
    filled = int(bw * max(0.0, remaining / total))
    y0, y1 = 6, 22
    cv2.rectangle(frame, (20, y0), (20+bw, y1), (40, 40, 40), -1)
    col = (0, 0, 200) if remaining < 10 else level_color
    cv2.rectangle(frame, (20, y0), (20+filled, y1), col, -1)
    cv2.rectangle(frame, (20, y0), (20+bw, y1), (160, 160, 160), 1)
    label = f"{int(remaining)}s"
    cv2.putText(frame, label, (fw//2 - 18, y1 - 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)



def draw_lives(frame: np.ndarray, lives: int, sprite: np.ndarray) -> None:
    disp = min(lives, 5)
    if disp <= 0:
        return
    bw = disp * 54 + 28
    x0 = frame.shape[1] - bw - 14
    _panel(frame, x0, 30, bw, 68)
    cv2.rectangle(frame, (x0, 30), (x0+bw, 98), (80, 50, 10), 1)
    for i in range(disp):
        overlay_sprite(frame, sprite, (x0+20+i*54, 64), 0.48)
    if lives > 5:
        cv2.putText(frame, f"+{lives-5}", (frame.shape[1]-46, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 100), 1)



def draw_freeze_bar(frame: np.ndarray, remaining: float, total: float) -> None:
    fw = frame.shape[1]
    bw = 240
    x0 = (fw - bw) // 2
    _panel(frame, x0-8, 28, bw+16, 28)
    filled = int(bw * max(0.0, remaining / total))
    cv2.rectangle(frame, (x0, 31), (x0+filled, 51), (255, 160, 0), -1)
    cv2.rectangle(frame, (x0, 31), (x0+bw,    51), (200, 200, 200), 1)
    cv2.putText(frame, "FREEZE", (x0+bw//2-28, 47),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)



def draw_prompt(frame: np.ndarray, text: str, now: float,
                show_until: float, y: int = 676) -> None:
    if not text or now >= show_until:
        return
    alpha = min(1.0, (show_until - now) / 0.5)
    fw = frame.shape[1]
    font = cv2.FONT_HERSHEY_PLAIN
    sz = cv2.getTextSize(text, font, 1.35, 2)[0]
    x = (fw - sz[0]) // 2
    col = tuple(int(c * alpha) for c in (230, 230, 70))
    cv2.putText(frame, text, (x+1, y+1), font, 1.35, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y),   font, 1.35, col,     2, cv2.LINE_AA)



def draw_centered_banner(frame: np.ndarray, line1: str, line2: str,
                         color: Color, y_ratio: float = 0.42) -> None:
    h, w = frame.shape[:2]
    font1, sc1, th1 = cv2.FONT_HERSHEY_DUPLEX, 1.9, 3
    font2, sc2, th2 = cv2.FONT_HERSHEY_PLAIN,  1.3, 2
    s1 = cv2.getTextSize(line1, font1, sc1, th1)[0]
    s2 = cv2.getTextSize(line2, font2, sc2, th2)[0] if line2 else (0, 0)
    bw = max(s1[0], s2[0]) + 64
    bh = s1[1] + (s2[1] + 18 if line2 else 0) + 52
    x0 = (w - bw) // 2
    y0 = int(h * y_ratio) - bh // 2
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x0+bw, y0+bh), (8, 8, 8), -1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0+bw, y0+bh), color, 2)
    tx = (w - s1[0]) // 2
    ty = y0 + s1[1] + 18
    cv2.putText(frame, line1, (tx, ty), font1, sc1, color, th1, cv2.LINE_AA)
    if line2:
        tx2 = (w - s2[0]) // 2
        cv2.putText(frame, line2, (tx2, ty + s2[1] + 16),
                    font2, sc2, (200, 200, 200), th2, cv2.LINE_AA)



def draw_level_select(frame: np.ndarray, hovered: int,
                      levels: dict, now: float) -> None:
    """
    Draw a full-screen level selection menu.
    hovered: 1, 2, or 3 — whichever button the finger is over (-1 = none).
    """
    h, w = frame.shape[:2]
    
    ov = np.zeros_like(frame)
    cv2.addWeighted(frame, 0.30, ov, 0.70, 0, frame)

    
    title = "FRUIT NINJA"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.2, 4)[0]
    cv2.putText(frame, title, ((w-ts[0])//2, 110),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 215, 255), 4, cv2.LINE_AA)
    sub = "Point your index finger at a level to select"
    ss = cv2.getTextSize(sub, cv2.FONT_HERSHEY_PLAIN, 1.3, 2)[0]
    cv2.putText(frame, sub, ((w-ss[0])//2, 155),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (200, 200, 200), 2, cv2.LINE_AA)

    
    card_w, card_h = 300, 220
    gap = 40
    total_w = 3 * card_w + 2 * gap
    start_x = (w - total_w) // 2
    card_y  = h // 2 - card_h // 2 + 20

    cards = {}
    for i, lvl in enumerate([1, 2, 3]):
        cfg  = levels[lvl]
        cx   = start_x + i * (card_w + gap)
        cy   = card_y
        cards[lvl] = (cx, cy, card_w, card_h)

        col   = cfg["color"]
        is_hov = hovered == lvl

        
        ov2 = frame.copy()
        fill = (40, 40, 40) if not is_hov else (60, 60, 60)
        cv2.rectangle(ov2, (cx, cy), (cx+card_w, cy+card_h), fill, -1)
        cv2.addWeighted(ov2, 0.80, frame, 0.20, 0, frame)

        border_thick = 4 if is_hov else 2
        cv2.rectangle(frame, (cx, cy), (cx+card_w, cy+card_h), col, border_thick)

        
        lnum = f"LEVEL {lvl}"
        ls = cv2.getTextSize(lnum, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
        cv2.putText(frame, lnum, (cx+(card_w-ls[0])//2, cy+46),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, col, 2, cv2.LINE_AA)

        
        ns = cv2.getTextSize(cfg["name"], cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
        cv2.putText(frame, cfg["name"], (cx+(card_w-ns[0])//2, cy+96),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, col, 3, cv2.LINE_AA)

        
        lines = [
            f"Time:   {cfg['time_limit']}s",
            f"Target: {cfg['score_target']} pts",
            f"Lives:  {cfg['lives']}",
            f"Bombs:  {int(cfg['bomb_chance']*100)}%",
        ]
        for j, ln in enumerate(lines):
            cv2.putText(frame, ln, (cx+18, cy+128+j*22),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, (210, 210, 210), 1, cv2.LINE_AA)

        
        if is_hov:
            pulse = 0.5 + 0.5 * abs(np.sin(now * 4))
            ring_col = tuple(int(c * pulse) for c in col)
            cv2.rectangle(frame, (cx-3, cy-3), (cx+card_w+3, cy+card_h+3),
                          ring_col, 3)

    return cards



def draw_level_intro(frame: np.ndarray, level: int, cfg: dict,
                     progress: float) -> None:
    alpha = min(1.0, 2.5 * min(progress, 1 - progress))
    h, w  = frame.shape[:2]
    ov    = np.zeros_like(frame)
    cv2.addWeighted(frame, 1-0.6*alpha, ov, 0.6*alpha, 0, frame)
    col   = tuple(int(c * alpha) for c in cfg["color"])
    whi   = tuple(int(c * alpha) for c in (255, 255, 255))

    big = f"LEVEL {level}  —  {cfg['name']}"
    sz  = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, 1.7, 3)[0]
    cv2.putText(frame, big, ((w-sz[0])//2, h//2-28),
                cv2.FONT_HERSHEY_DUPLEX, 1.7, col, 3, cv2.LINE_AA)

    info = f"Time: {cfg['time_limit']}s   Target: {cfg['score_target']} pts   Lives: {cfg['lives']}"
    si   = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, 1.25, 2)[0]
    cv2.putText(frame, info, ((w-si[0])//2, h//2+22),
                cv2.FONT_HERSHEY_PLAIN, 1.25, whi, 2, cv2.LINE_AA)

    hint = "SPACE to skip"
    sh   = cv2.getTextSize(hint, cv2.FONT_HERSHEY_PLAIN, 1.1, 1)[0]
    cv2.putText(frame, hint, ((w-sh[0])//2, h//2+60),
                cv2.FONT_HERSHEY_PLAIN, 1.1, (150, 150, 150), 1, cv2.LINE_AA)
