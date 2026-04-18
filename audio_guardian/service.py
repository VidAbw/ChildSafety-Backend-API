from __future__ import annotations

import numpy as np

from audio_guardian.schemas import AudioFrameSchema


class AcousticModel:
    """Lightweight placeholder until the trained 1D-CNN + LSTM is plugged in."""

    def predict(self, mfcc_matrix: np.ndarray) -> dict[str, float]:
        if mfcc_matrix.size == 0:
            return {"trauma_score": 0.0, "duration_score": 0.0}

        # Heuristic scores based on MFCC energy/variance to keep prototype non-blocking.
        energy = float(np.mean(np.abs(mfcc_matrix)))
        variance = float(np.var(mfcc_matrix))

        trauma_score = float(np.clip(energy / 30.0 + variance / 100.0, 0.0, 1.0))
        duration_score = float(np.clip(mfcc_matrix.shape[0] / 120.0, 0.0, 1.0))
        return {
            "trauma_score": round(trauma_score, 3),
            "duration_score": round(duration_score, 3),
        }


_model = AcousticModel()


def evaluate_audio_frame(frame: AudioFrameSchema) -> tuple[bool, dict[str, float], list[int]]:
    mfcc_matrix = np.array(frame.mfcc, dtype=np.float32)
    scores = _model.predict(mfcc_matrix)

    trauma_detected = (
        scores["trauma_score"] >= frame.trauma_threshold
        and scores["duration_score"] >= frame.duration_threshold
    )

    return trauma_detected, scores, list(mfcc_matrix.shape)
