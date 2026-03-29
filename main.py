import time

import cv2
import numpy as np

import config
from pose_detector import PoseDetector
from audio_player import AudioPlayer

# How long (seconds) to wait after losing the pose before stopping audio
STOP_GRACE_PERIOD = 1.0


def draw_landmarks(frame, landmarks):
    """Draw hand landmarks and connections on the frame."""
    h, w = frame.shape[:2]

    # MediaPipe Hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),     # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),     # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (5, 9), (9, 13), (13, 17),            # palm
    ]

    points = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))

    for i, j in connections:
        if i < len(points) and j < len(points):
            cv2.line(frame, points[i], points[j], (0, 255, 0), 2)

    for px, py in points:
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)


def main():
    detector = PoseDetector(
        detection_confidence=config.DETECTION_CONFIDENCE,
        tracking_confidence=config.TRACKING_CONFIDENCE,
    )
    player = AudioPlayer()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam. Check camera permissions in System Settings.")
        return

    print("Pose Player running. Press 'q' to quit.")

    last_pose_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detector.detect(rgb)

        status = "No body detected"
        color = (200, 200, 200)
        pose_active = False

        if landmarks:
            draw_landmarks(frame, landmarks)

            # Check for poses
            matched = detector.match_poses(landmarks, config.ANGLE_TOLERANCE_DEGREES)

            if matched:
                pose_active = True
                last_pose_time = time.time()
                pose_name = matched[0]
                song_path = config.POSE_SONG_MAP.get(pose_name)
                if song_path:
                    player.play(song_path)

                display_name = pose_name.replace("_", " ").upper()
                if player.is_playing():
                    status = f"{display_name} - Playing!"
                    color = (0, 255, 0)
                else:
                    status = f"{display_name} detected"
                    color = (0, 200, 255)
            else:
                status = "Listening..."
                color = (255, 255, 255)

        if not pose_active and player.is_playing():
            if time.time() - last_pose_time > STOP_GRACE_PERIOD:
                player.stop()

        # Draw status overlay
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Pose Player", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    player.stop()


if __name__ == "__main__":
    main()
