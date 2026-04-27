# detector/face.py
"""
Face recognition module.

Loads every image from `nanny_cam_guardian/known_faces/` on startup, encodes
each face once, then on each call compares the cropped adult region against
those embeddings. Returns the matched name or "unknown".

To keep FPS reasonable we:
    - only process adult bounding boxes (children are not recognised)
    - re-check each tracked adult every RECOGNITION_INTERVAL_FRAMES
    - cache the last result per person index between checks
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from deepface import DeepFace

KNOWN_FACES_DIR = Path(__file__).resolve().parent.parent / "known_faces"
MODEL_NAME = "SFace"          # lightweight, no extra downloads after first run
DETECTOR_BACKEND = "opencv"   # fastest available backend
DISTANCE_METRIC = "cosine"
MATCH_THRESHOLD = 0.55        # cosine distance below this → match (SFace default ~0.593)
RECOGNITION_INTERVAL_FRAMES = 30   # re-check each tracked adult every ~1s @ 30fps
CROP_PADDING = 0.15           # expand adult bbox by 15% to include full head


class FaceRecognizer:
    def __init__(self) -> None:
        self._known: dict[str, np.ndarray] = {}      # name → embedding
        self._cache: dict[int, str] = {}             # person_idx → last label
        self._frame_count: dict[int, int] = {}       # person_idx → frames since last check
        self._load_known_faces()

    def _load_known_faces(self) -> None:
        if not KNOWN_FACES_DIR.exists():
            print(f"[face] known_faces folder missing: {KNOWN_FACES_DIR}")
            return
        files = [
            f for f in KNOWN_FACES_DIR.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not files:
            print(f"[face] No reference photos in {KNOWN_FACES_DIR} — every adult will be Unknown.")
            return
        for fp in files:
            try:
                rep = DeepFace.represent(
                    img_path=str(fp),
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True,
                )
                embedding = np.array(rep[0]["embedding"])
                self._known[fp.stem] = embedding
                print(f"[face] Loaded reference: {fp.stem}")
            except Exception as e:
                print(f"[face] Could not encode {fp.name}: {e}")

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _identify_crop(self, crop_bgr: np.ndarray) -> str:
        if not self._known:
            return "unknown"
        try:
            rep = DeepFace.represent(
                img_path=crop_bgr,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,   # crop may be tight; don't fail on no-detection
            )
            embedding = np.array(rep[0]["embedding"])
        except Exception:
            return "unknown"

        best_name, best_dist = "unknown", float("inf")
        for name, ref in self._known.items():
            d = self._cosine_distance(embedding, ref)
            if d < best_dist:
                best_dist, best_name = d, name
        return best_name if best_dist <= MATCH_THRESHOLD else "unknown"

    def identify(self, frame_bgr: np.ndarray, persons: list) -> dict[int, str]:
        """
        Returns {person_index: label} for every adult in the frame.
        Children are skipped. Each adult is re-checked every
        RECOGNITION_INTERVAL_FRAMES; otherwise the cached label is reused.
        """
        h, w = frame_bgr.shape[:2]
        results: dict[int, str] = {}

        for idx, person in enumerate(persons):
            if person.is_child:
                continue

            count = self._frame_count.get(idx, RECOGNITION_INTERVAL_FRAMES)
            if count < RECOGNITION_INTERVAL_FRAMES and idx in self._cache:
                results[idx] = self._cache[idx]
                self._frame_count[idx] = count + 1
                continue

            # Time to re-check: crop adult region with padding around the head
            pad_x = (person.x2 - person.x1) * CROP_PADDING
            pad_y = (person.y2 - person.y1) * CROP_PADDING
            x1 = max(0, int(person.x1 - pad_x))
            y1 = max(0, int(person.y1 - pad_y))
            x2 = min(w, int(person.x2 + pad_x))
            y2 = min(h, int(person.y2 + pad_y))
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                results[idx] = "unknown"
                continue

            label = self._identify_crop(crop)
            self._cache[idx] = label
            self._frame_count[idx] = 0
            results[idx] = label

        # Drop cache entries for persons no longer in frame
        active = set(range(len(persons)))
        for stale in list(self._cache):
            if stale not in active:
                del self._cache[stale]
                self._frame_count.pop(stale, None)

        return results

    @property
    def has_known_faces(self) -> bool:
        return bool(self._known)
