import math
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
Landmark = mp_pose.PoseLandmark


def _angle(a, b, c) -> float:
    """Angle in degrees at point b formed by segments b->a and b->c."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def _visible(landmarks, *indices, threshold=0.5) -> bool:
    """Check that all specified landmarks have sufficient visibility."""
    return all(landmarks[i].visibility > threshold for i in indices)


# --- Pose checkers ---
# Each returns True if the pose is detected.
# To add a new pose, write a checker function and register it in POSE_CHECKERS.

def _check_t_pose(landmarks, tolerance: float) -> bool:
    """Arms extended horizontally, roughly perpendicular to torso."""
    lm = landmarks

    required = [
        Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST,
        Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST,
        Landmark.LEFT_HIP, Landmark.RIGHT_HIP,
    ]
    if not _visible(lm, *[r.value for r in required]):
        return False

    # Arms should be straight (elbow angle near 180)
    left_arm = _angle(lm[Landmark.LEFT_SHOULDER], lm[Landmark.LEFT_ELBOW], lm[Landmark.LEFT_WRIST])
    right_arm = _angle(lm[Landmark.RIGHT_SHOULDER], lm[Landmark.RIGHT_ELBOW], lm[Landmark.RIGHT_WRIST])

    # Arms should be horizontal (shoulder angle near 90 from torso)
    left_shoulder = _angle(lm[Landmark.LEFT_HIP], lm[Landmark.LEFT_SHOULDER], lm[Landmark.LEFT_ELBOW])
    right_shoulder = _angle(lm[Landmark.RIGHT_HIP], lm[Landmark.RIGHT_SHOULDER], lm[Landmark.RIGHT_ELBOW])

    return (
        abs(left_arm - 180) < tolerance
        and abs(right_arm - 180) < tolerance
        and abs(left_shoulder - 90) < tolerance
        and abs(right_shoulder - 90) < tolerance
    )


POSE_CHECKERS = {
    "t_pose": _check_t_pose,
}


class PoseDetector:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame: np.ndarray):
        """Run pose detection. Returns landmarks list or None."""
        results = self.pose.process(frame)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def match_poses(self, landmarks, tolerance: float) -> list[str]:
        """Return names of all currently matched poses."""
        return [
            name for name, checker in POSE_CHECKERS.items()
            if checker(landmarks, tolerance)
        ]

    def close(self):
        self.pose.close()
