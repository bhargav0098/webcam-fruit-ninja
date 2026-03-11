
# Fruit Ninja — Level Select Edition 🍉

Webcam Fruit Ninja game where you slice fruits using your **index finger through a webcam**.

## Python Requirement

⚠️ This project requires **Python 3.10 environment** to run properly.

## How to Play

1. Run the game — you land on the **level-select screen**
2. Point your finger at a card and **hold for 1.5 seconds** (or press **1 / 2 / 3**)
3. Swipe your finger fast across fruit to slice it
4. Reach the **score target** before time runs out to win

## Levels (45 seconds each)

| Level | Difficulty | Target | Lives | Bombs | Fruit Speed |
|------|------------|-------|------|------|-------------|
| 1 | EASY | 120 pts | 5 | 6% | Slow |
| 2 | MEDIUM | 180 pts | 4 | 11% | Medium |
| 3 | HARD | 250 pts | 3 | 17% | Fast |

## Specials

| Item | Effect |
|-----|-------|
| 🟡 Golden Fruit | Pulsing cyan ring — worth **3× points** |
| 🧊 ICE Fruit | Freezes all fruits for **3.5 seconds** |
| 💣 Bomb | **Lose 1 life** — avoid slicing |

## Controls

| Key / Action | Effect |
|--------------|-------|
| Hover finger on card for **1.5 s** | Select level |
| **1 / 2 / 3** | Select level (keyboard) |
| **SPACE** | Skip intro |
| **R** | Back to level select |
| **Q** | Quit game |

## Installation & Run

1. Install dependencies

pip install -r requirements.txt

2. Run the game

python -m src.main

## Project Structure


├── assets/              # Auto-generated PNG sprites (created on first run)
├── requirements.txt
├── README.md
└── src/
    ├── asset_utils.py   # Ensures sprites exist and loads them with alpha
    ├── config.py        # Tunable gameplay constants
    ├── game_objects.py  # Dataclasses + spawn helpers
    ├── hand_tracker.py  # MediaPipe wrapper for index-finger detection
    ├── ui_overlay.py    # HUD drawing helpers
    └── main.py          # Game loop, physics, UI, gesture detection

