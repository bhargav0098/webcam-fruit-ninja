"""Configuration — Fruit Ninja, player-selected single level."""
from pathlib import Path

CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
WINDOW_NAME   = "Fruit Ninja"
TARGET_FPS    = 60


LEVELS = {
    1: dict(
        name="EASY",
        color=(80, 210, 80),
        time_limit=45,               
        spawn_interval=1.30,          
        gravity=1250.0,
        slice_threshold=400,          
        bomb_chance=0.09,             
        multi_spawn_chance=0.12,      
        score_target=250,             
        lives=4,
    ),
    2: dict(
        name="MEDIUM",
        color=(220, 180, 30),         
        time_limit=45,
        spawn_interval=0.90,
        gravity=1600.0,
        slice_threshold=500,
        bomb_chance=0.13,
        multi_spawn_chance=0.22,
        score_target=350,
        lives=3,
    ),
    3: dict(
        name="HARD",
        color=(40, 40, 230),          
        time_limit=45,
        spawn_interval=0.60,
        gravity=2000.0,
        slice_threshold=600,
        bomb_chance=0.17,
        multi_spawn_chance=0.35,
        score_target=450,
        lives=2,
    ),
}


COMBO_WINDOW          = 1.25
MAX_COMBO             = 5.0
TRAIL_HISTORY         = 8
TRAIL_MAX_AGE         = 0.16     
FRUIT_MIN_RADIUS      = 42
FRUIT_MAX_RADIUS      = 86
FREEZE_DURATION       = 3.5
FREEZE_SPEED_FACTOR   = 0.22
GOLDEN_FRUIT_CHANCE   = 0.06
GOLDEN_FRUIT_MULTIPLIER = 3.0
EXTRA_LIFE_SCORE      = 100      
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR    = PROJECT_ROOT / "assets"

FRUIT_CATALOG = [
    ("apple.png",       0.85, 7),
    ("banana.png",      0.95, 6),
    ("grapes.png",      0.85, 7),
    ("watermelon.png",  1.25, 8),
    ("strawberry.png",  0.80, 7),
    ("pineapple.png",   1.35, 9),
    ("orange.png",      0.90,  4),
    ("peach.png",       0.95, 6),
    ("cherries.png",    0.75, 6),
    ("mango.png",       1.00, 7),
    ("kiwi.png",        0.80, 7),
    ("pear.png",        0.95, 6),
    ("lemon.png",       0.85,  4),
    ("melon.png",       1.15, 8),
    ("blueberries.png", 0.75, 7),
]
