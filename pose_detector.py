from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = Path(__file__).parent / "pose_landmarker_full.task"

# Indicii MediaPipe: (sold, genunchi, glezna)
KNEE = {"left": (23, 25, 27), "right": (24, 26, 28)}

# Perechile de puncte unite la desenarea scheletului (fara fata si degete)
CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # trunchi
    (11, 13), (13, 15), (12, 14), (14, 16),  # brate
    (23, 25), (25, 27), (24, 26), (26, 28),  # picioare
    (27, 29), (29, 31), (28, 30), (30, 32),  # talpi
]


class PoseDetector:
    def __init__(self, model_path=MODEL_PATH):
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._landmarker.close()

    def detect(self, frame, timestamp_ms):
        """Intoarce o matrice 33x3 (x, y in pixeli, vizibilitate) sau None.

        timestamp_ms trebuie sa creasca de la un cadru la altul.
        """
        # OpenCV citeste BGR, MediaPipe asteapta RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        h, w = frame.shape[:2]
        return np.array([(p.x * w, p.y * h, p.visibility) for p in result.pose_landmarks[0]])
