# detector/pose.py
import os
import urllib.request
from dataclasses import dataclass

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe landmark indices used by the rule engine
NOSE = 0
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Pose landmarker model (lite = fastest, auto-downloaded on first run)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")


def _ensure_model() -> str:
    path = os.path.abspath(_MODEL_PATH)
    if not os.path.exists(path):
        print("[pose] Downloading pose landmarker model (one-time) ...")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print(f"[pose] Model saved to {path}")
    return path


@dataclass
class Keypoints:
    """
    Holds normalised (0–1) x,y coordinates and visibility for each of the
    33 MediaPipe Pose landmarks relative to the FULL frame.
    If pose detection failed, landmarks is None.
    """
    landmarks: list | None

    def get(self, index: int) -> tuple[float, float, float] | None:
        if self.landmarks is None:
            return None
        lm = self.landmarks[index]
        return lm.x, lm.y, lm.visibility

    def is_valid(self) -> bool:
        return self.landmarks is not None


class PoseEstimator:
    def __init__(self):
        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=4,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def extract_all(self, frame_rgb: np.ndarray) -> list[Keypoints]:
        """
        Run pose on the FULL frame. Returns one Keypoints per detected person.
        Landmarks are normalised to the full frame (0–1).
        Up to 4 poses returned.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        return [Keypoints(landmarks=pose) for pose in result.pose_landmarks]

    def match_to_persons(
        self,
        all_keypoints: list[Keypoints],
        persons: list,
        frame_w: int,
        frame_h: int,
    ) -> dict[int, Keypoints]:
        """
        Match each detected pose to the nearest YOLO person bounding box.
        Uses hip midpoint (or nose as fallback) as the pose anchor.
        Picks the closest unmatched pose for each person — no strict containment
        requirement, so partial crops and corner-mounted cameras work correctly.
        Returns {person_index: Keypoints}.
        """
        matched: dict[int, Keypoints] = {}
        used_poses: set[int] = set()

        # Pre-compute anchors for all poses
        anchors: dict[int, tuple[float, float]] = {}
        for pose_idx, kp in enumerate(all_keypoints):
            if not kp.is_valid():
                continue
            lhip = kp.get(LEFT_HIP)
            rhip = kp.get(RIGHT_HIP)
            nose = kp.get(NOSE)
            if lhip and rhip:
                anchors[pose_idx] = (
                    (lhip[0] + rhip[0]) / 2 * frame_w,
                    (lhip[1] + rhip[1]) / 2 * frame_h,
                )
            elif nose:
                anchors[pose_idx] = (nose[0] * frame_w, nose[1] * frame_h)

        for person_idx, person in enumerate(persons):
            cx = (person.x1 + person.x2) / 2
            cy = (person.y1 + person.y2) / 2
            best_pose_idx = None
            best_dist = float("inf")

            for pose_idx, anchor in anchors.items():
                if pose_idx in used_poses:
                    continue
                dist = ((anchor[0] - cx) ** 2 + (anchor[1] - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_pose_idx = pose_idx

            if best_pose_idx is not None:
                matched[person_idx] = all_keypoints[best_pose_idx]
                used_poses.add(best_pose_idx)

        return matched

    def close(self):
        self._landmarker.close()
