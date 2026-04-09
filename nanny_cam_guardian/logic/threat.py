# logic/threat.py
from dataclasses import dataclass, field
import numpy as np

from nanny_cam_guardian.detector.yolo import DetectionResult, PersonBox
from nanny_cam_guardian.detector.pose import Keypoints, NOSE, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP
from nanny_cam_guardian.logic.tracker import VelocityTracker

# ── Thresholds ────────────────────────────────────────────────────────────────
HAZARD_PROXIMITY_PX = 100       # hazard within this many pixels of child bbox → Level 1
FALL_FRAME_THRESHOLD = 30       # consecutive frames nose below hips → Level 2
ABUSE_PROXIMITY_RATIO = 0.5     # adult centroid within 50% of frame height from child → Level 3
ABUSE_VELOCITY_THRESHOLD = 300  # pixels/second wrist speed → Level 3 trigger


@dataclass
class ThreatEvent:
    level: int          # 0=safe, 1=hazard, 2=fall, 3=abuse_suspected
    type: str           # 'safe' | 'hazard' | 'fall' | 'abuse_suspected'
    probability: float
    details: dict = field(default_factory=dict)


def _box_proximity(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    """Return the minimum pixel distance between two axis-aligned bounding boxes."""
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    return float(np.sqrt(dx * dx + dy * dy))


def _centroid_distance(a: PersonBox, b: PersonBox) -> float:
    ax, ay = a.centroid
    bx, by = b.centroid
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def _direction_toward(
    direction_vector: tuple[float, float],
    from_centroid: tuple[float, float],
    toward_centroid: tuple[float, float],
) -> float:
    """
    Returns a score in [0, 1] representing how much the motion direction
    points from `from_centroid` toward `toward_centroid`.
    """
    target = np.array(toward_centroid) - np.array(from_centroid)
    norm = np.linalg.norm(target)
    if norm == 0:
        return 0.0
    target_unit = target / norm
    dot = float(np.dot(np.array(direction_vector), target_unit))
    return max(0.0, dot)   # clamp to [0, 1]


class ThreatRuleEngine:
    def __init__(self):
        self._fall_counter: int = 0
        # Trackers keyed by person index (order in DetectionResult.persons list)
        self._trackers: dict[int, VelocityTracker] = {}

    def _get_tracker(self, idx: int) -> VelocityTracker:
        if idx not in self._trackers:
            self._trackers[idx] = VelocityTracker()
        return self._trackers[idx]

    def evaluate(
        self,
        detection: DetectionResult,
        keypoints_map: dict[int, Keypoints],   # person index → Keypoints
        frame_height: int,
        timestamp: float,
    ) -> ThreatEvent:
        persons = detection.persons
        hazards = detection.hazards
        children = [p for p in persons if p.is_child]
        adults = [p for p in persons if not p.is_child]

        # ── Update velocity trackers for each person ───────────────────────
        for idx, person in enumerate(persons):
            kp = keypoints_map.get(idx)
            if kp and kp.is_valid():
                tracker = self._get_tracker(idx)
                # Use the wrist with higher visibility
                lw = kp.get(LEFT_WRIST)
                rw = kp.get(RIGHT_WRIST)
                wrist = None
                if lw and rw:
                    wrist = lw[:2] if lw[2] >= rw[2] else rw[:2]
                elif lw:
                    wrist = lw[:2]
                elif rw:
                    wrist = rw[:2]
                if wrist:
                    tracker.update(wrist, timestamp)

        # ── Level 3: Abuse suspected ───────────────────────────────────────
        if adults and children:
            for adult_idx, adult in enumerate(
                [p for i, p in enumerate(persons) if not p.is_child]
            ):
                adult_person_idx = [i for i, p in enumerate(persons) if not p.is_child][adult_idx]
                tracker = self._get_tracker(adult_person_idx)
                velocity = tracker.get_velocity()

                if velocity < ABUSE_VELOCITY_THRESHOLD:
                    continue

                for child in children:
                    dist = _centroid_distance(adult, child)
                    if dist > ABUSE_PROXIMITY_RATIO * frame_height:
                        continue

                    # All three conditions met — calculate probability
                    direction = tracker.get_direction_vector()
                    direction_score = _direction_toward(direction, adult.centroid, child.centroid)
                    max_dist = ABUSE_PROXIMITY_RATIO * frame_height
                    proximity_score = max(0.0, 1.0 - dist / max_dist)
                    max_v = ABUSE_VELOCITY_THRESHOLD * 3
                    velocity_score = min(1.0, velocity / max_v)

                    probability = min(
                        1.0,
                        velocity_score * 0.5 + proximity_score * 0.3 + direction_score * 0.2,
                    )

                    self._fall_counter = 0
                    return ThreatEvent(
                        level=3,
                        type="abuse_suspected",
                        probability=round(probability, 3),
                        details={
                            "adult_hand_velocity": round(velocity, 2),
                            "skeleton_distance": round(dist, 2),
                            "triggered_by": ["proximity", "velocity", "direction"],
                        },
                    )

        # ── Level 2: Fall detected ─────────────────────────────────────────
        for idx, child in enumerate(children):
            kp = keypoints_map.get(idx)
            if not kp or not kp.is_valid():
                continue
            nose = kp.get(NOSE)
            lhip = kp.get(LEFT_HIP)
            rhip = kp.get(RIGHT_HIP)
            if nose and lhip and rhip:
                hip_y = (lhip[1] + rhip[1]) / 2
                if nose[1] > hip_y:   # normalised coords: larger y = lower on screen
                    self._fall_counter += 1
                else:
                    self._fall_counter = 0

                if self._fall_counter >= FALL_FRAME_THRESHOLD:
                    return ThreatEvent(
                        level=2,
                        type="fall",
                        probability=1.0,
                        details={"triggered_by": ["fall_pose"], "fall_frames": self._fall_counter},
                    )

        # ── Level 1: Hazard near child ──────────────────────────────────────
        if hazards and children:
            for hazard in hazards:
                for child in children:
                    dist = _box_proximity(
                        hazard.x1, hazard.y1, hazard.x2, hazard.y2,
                        child.x1, child.y1, child.x2, child.y2,
                    )
                    if dist <= HAZARD_PROXIMITY_PX:
                        return ThreatEvent(
                            level=1,
                            type="hazard",
                            probability=1.0,
                            details={
                                "hazard_object": hazard.label,
                                "distance_px": round(dist, 1),
                                "triggered_by": ["hazard_proximity"],
                            },
                        )

        # ── Level 0: Safe ──────────────────────────────────────────────────
        self._fall_counter = 0
        return ThreatEvent(level=0, type="safe", probability=0.0)
