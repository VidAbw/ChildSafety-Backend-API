# logic/tracker.py
from collections import deque
import numpy as np

FRAME_BUFFER_SIZE = 10


class VelocityTracker:
    """
    Tracks a single 2-D point (e.g. a wrist) across a sliding window of the
    last FRAME_BUFFER_SIZE frames and computes its velocity in pixels/second.
    """

    def __init__(self):
        self._history: deque[tuple[tuple[float, float], float]] = deque(
            maxlen=FRAME_BUFFER_SIZE
        )

    def update(self, pos: tuple[float, float], timestamp: float) -> None:
        """Append the current position and timestamp to the buffer."""
        self._history.append((pos, timestamp))

    def get_velocity(self) -> float:
        """Return speed in pixels/second over the buffered window."""
        if len(self._history) < 2:
            return 0.0
        (p1, t1) = self._history[0]
        (p2, t2) = self._history[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        dist = float(np.linalg.norm(np.array(p2) - np.array(p1)))
        return dist / dt

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
