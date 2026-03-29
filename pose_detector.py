import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import numpy as np

# Hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20


def _is_finger_extended(landmarks, pip_idx, tip_idx, mcp_idx) -> bool:
    """A finger is extended if its tip is farther from the wrist than its PIP joint."""
    wrist = landmarks[WRIST]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    # Compare distance from wrist to tip vs wrist to pip
    tip_dist = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
    pip_dist = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
    return tip_dist > pip_dist


# --- Gesture checkers ---
# Each returns True if the gesture is detected.
# To add a new gesture, write a checker function and register it in POSE_CHECKERS.

def _check_peace_sign(landmarks, _tolerance: float) -> bool:
    """Peace sign: index and middle fingers extended, ring and pinky curled."""
    index_up = _is_finger_extended(landmarks, INDEX_PIP, INDEX_TIP, INDEX_MCP)
    middle_up = _is_finger_extended(landmarks, MIDDLE_PIP, MIDDLE_TIP, MIDDLE_MCP)
    ring_down = not _is_finger_extended(landmarks, RING_PIP, RING_TIP, RING_MCP)
    pinky_down = not _is_finger_extended(landmarks, PINKY_PIP, PINKY_TIP, PINKY_MCP)

    return index_up and middle_up and ring_down and pinky_down


POSE_CHECKERS = {
    "peace_sign": _check_peace_sign,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


class PoseDetector:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            num_hands=1,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts = 0

    def detect(self, frame: np.ndarray):
        """Run hand detection. Returns landmarks list or None."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self._frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            return result.hand_landmarks[0]
        return None

    def match_poses(self, landmarks, tolerance: float) -> list[str]:
        """Return names of all currently matched gestures."""
        return [
            name for name, checker in POSE_CHECKERS.items()
            if checker(landmarks, tolerance)
        ]

    def close(self):
        self.landmarker.close()
