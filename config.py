import os

_DIR = os.path.dirname(__file__)

# Pose-to-song mapping: pose name -> path to audio file
POSE_SONG_MAP = {
    "peace_sign": os.path.join(_DIR, "songs", "song.mp3"),
}

# How long (seconds) before the same pose can trigger again
COOLDOWN_SECONDS = 5.0

# MediaPipe confidence thresholds
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.5

# How many degrees off "perfect" a joint angle can be and still match
ANGLE_TOLERANCE_DEGREES = 25.0

# OpenCV camera index (0 = default webcam)
CAMERA_INDEX = 0
