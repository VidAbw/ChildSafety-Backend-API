# logic/tracker.py
from collections import deque
import numpy as np

FRAME_BUFFER_SIZE = 10   # shorter window catches bursts faster
EMA_ALPHA = 0.65         # high alpha = responsive to fast movements; noise floor handles jitter


class VelocityTracker:
    """
    Tracks a single 2-D point (e.g. a wrist) across a sliding window of the
    last FRAME_BUFFER_SIZE frames and computes its velocity in pixels/second.
    Applies EMA smoothing to positions before storing so that landmark jitter
    doesn't register as motion.
    """

    def __init__(self):
        self._history: deque[tuple[tuple[float, float], float]] = deque(
            maxlen=FRAME_BUFFER_SIZE
        )
        self._smoothed: tuple[float, float] | None = None

    def update(self, pos: tuple[float, float], timestamp: float) -> None:
        """EMA-smooth the position then append to the buffer."""
        if self._smoothed is None:
            self._smoothed = pos
        else:
            sx, sy = self._smoothed
            px, py = pos
            self._smoothed = (
                EMA_ALPHA * px + (1.0 - EMA_ALPHA) * sx,
                EMA_ALPHA * py + (1.0 - EMA_ALPHA) * sy,
            )
        self._history.append((self._smoothed, timestamp))

    def get_velocity(self) -> float:
        """
        Return peak frame-to-frame speed (px/s) in the buffer window.
        Using peak rather than average displacement captures fast arm
        swings that only last a few frames.
        """
        if len(self._history) < 2:
            return 0.0
        peak = 0.0
        for i in range(1, len(self._history)):
            (p1, t1) = self._history[i - 1]
            (p2, t2) = self._history[i]
            dt = t2 - t1
            if dt <= 0:
                continue
            speed = float(np.linalg.norm(np.array(p2) - np.array(p1))) / dt
            if speed > peak:
                peak = speed
        return peak

    def get_direction_vector(self) -> tuple[float, float]:
        """Return normalised direction vector from oldest to newest position."""
        if len(self._history) < 2:
            return (0.0, 0.0)
        p1 = np.array(self._history[0][0])
        p2 = np.array(self._history[-1][0])
        diff = p2 - p1
        norm = np.linalg.norm(diff)
        if norm == 0:
            return (0.0, 0.0)
        return tuple((diff / norm).tolist())

    def reset(self) -> None:
        self._history.clear()
        self._smoothed = None
