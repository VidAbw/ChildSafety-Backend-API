# detector/pose.py
from dataclasses import dataclass
import mediapipe as mp
import numpy as np

# MediaPipe landmark indices used by the rule engine
NOSE = 0
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass
class Keypoints:
    """
    Holds the normalised (0–1) x,y coordinates and visibility for each of the
    33 MediaPipe Pose landmarks.  If pose detection failed, all values are None.
    """
    landmarks: list | None  # list of (x, y, visibility) tuples, length 33

    def get(self, index: int) -> tuple[float, float, float] | None:
        if self.landmarks is None:
            return None
        lm = self.landmarks[index]
        return lm.x, lm.y, lm.visibility

    def is_valid(self) -> bool:
        return self.landmarks is not None


class PoseEstimator:
    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,          # 0 = lite, fastest
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_rgb: np.ndarray) -> Keypoints:
        """
        Run pose estimation on a full RGB frame.
        Returns Keypoints with normalised landmark positions.
        """
        result = self._pose.process(frame_rgb)
        if result.pose_landmarks:
            return Keypoints(landmarks=result.pose_landmarks.landmark)
        return Keypoints(landmarks=None)

    def extract_region(self, frame_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Keypoints:
        """
        Crop to a person's bounding box before running pose estimation.
        Coordinates are pixel values in the original frame.
        """
        h, w = frame_rgb.shape[:2]
        x1c = max(0, int(x1))
        y1c = max(0, int(y1))
        x2c = min(w, int(x2))
        y2c = min(h, int(y2))
        crop = frame_rgb[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return Keypoints(landmarks=None)
        return self.extract(crop)

    def close(self):
        self._pose.close()
