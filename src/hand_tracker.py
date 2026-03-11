"""Single-hand tracker using MediaPipe Hands."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp


@dataclass
class HandLandmark:
    point: Tuple[int, int]
    confidence: float


class HandTracker:
    """Detects the index fingertip of ONE hand."""

    def __init__(
        self,
        detection_confidence: float = 0.65,
        tracking_confidence: float = 0.65,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def locate_index_finger(self, frame) -> Optional[HandLandmark]:
        """Return index-fingertip pixel coordinate, or None if no hand found."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        h, w = frame.shape[:2]
        lm = results.multi_hand_landmarks[0].landmark[8]   # index tip
        return HandLandmark(
            point=(int(lm.x * w), int(lm.y * h)),
            confidence=lm.visibility,
        )

    def close(self) -> None:
        self._hands.close()
