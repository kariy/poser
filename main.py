import cv2
import mediapipe as mp
import numpy as np

import config
from pose_detector import PoseDetector
from audio_player import AudioPlayer

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


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

        if landmarks:
            # Draw skeleton overlay
            landmark_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
            landmark_proto.landmark.extend([
                mp.framework.formats.landmark_pb2.NormalizedLandmark(
                    x=lm.x, y=lm.y, z=lm.z
                ) for lm in landmarks
            ])
            mp_drawing.draw_landmarks(frame, landmark_proto, mp_pose.POSE_CONNECTIONS)

            # Check for poses
            matched = detector.match_poses(landmarks, config.ANGLE_TOLERANCE_DEGREES)

            if matched:
                pose_name = matched[0]
                song_path = config.POSE_SONG_MAP.get(pose_name)
                triggered = False
                if song_path:
                    triggered = player.play_if_ready(pose_name, song_path, config.COOLDOWN_SECONDS)

                display_name = pose_name.replace("_", " ").upper()
                if triggered:
                    status = f"{display_name} - Playing!"
                    color = (0, 255, 0)
                else:
                    status = f"{display_name} detected"
                    color = (0, 200, 255)
            else:
                status = "Listening..."
                color = (255, 255, 255)

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
