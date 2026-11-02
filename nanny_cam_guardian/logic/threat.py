# logic/threat.py
from dataclasses import dataclass, field
import numpy as np

from nanny_cam_guardian.detector.yolo import DetectionResult, PersonBox, HAZARD_CONF_THRESHOLD
from nanny_cam_guardian.detector.pose import Keypoints, NOSE, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP
from nanny_cam_guardian.logic.tracker import WristVelocityTracker, VelocityTracker

# ── Thresholds ────────────────────────────────────────────────────────────────
# Every spatial/kinematic threshold is a fraction of the live frame height
# (H_frame; 640 px reference, so 0.156 → 100 px at 640p) and, where a subject
# is involved, is additionally capped by a fraction of that subject's remembered
# maximum height (H_child_max / H_adult_max). The effective gate is the
# min() of the two, which keeps it perspective-invariant: a distant (small)
# subject gets a proportionally smaller threshold, a close one never exceeds
# the frame-normalised reference.
H_FRAME_REF = 640

# Level 1 — hazard proximity (AABB gap)
HAZARD_PROXIMITY_FRAME = 0.156      # × H_frame            (100 px @ 640p)
HAZARD_PROXIMITY_CHILD = 0.40       # × H_child_max
HAZARD_MIN_CONFIDENCE = HAZARD_CONF_THRESHOLD   # c_h >= 0.60 (production floor)
HAZARD_FRAME_THRESHOLD = 5          # F_hazard: 5 frames (166 ms)

# Level 2 — child fall
FALL_FRAME_THRESHOLD = 15           # F_fall: 15 frames (500 ms)
FALL_ASPECT_RATIO = 1.50            # w_bbox / h_bbox > 1.50
FALL_DROP_FRAME = 0.281             # × H_frame / s        (180 px/s @ 640p)
FALL_DROP_CHILD = 0.72              # × H_child_max / s

# Level 3 — unknown adult
UNKNOWN_PERSON_FRAME_THRESHOLD = 15 # F_unknown: 15 frames (500 ms)

# Level 4 — physical abuse (5-gate conjunction)
ABUSE_FRAME_THRESHOLD = 6           # F_abuse: 6 frames (200 ms)
ABUSE_VELOCITY_FRAME = 0.703        # × H_frame / s        (450 px/s @ 640p)
ABUSE_VELOCITY_ADULT = 1.25         # × H_adult_max / s
VELOCITY_NOISE_FRAME = 0.094        # × H_frame / s        (60 px/s @ 640p)
ABUSE_PROXIMITY_RATIO = 0.50        # d_centroid <= 0.50 × H_frame (320 px @ 640p)
ABUSE_MIN_DIRECTION_SCORE = 0.45    # S_d >= 0.45  (angle <= 63.26°)
ABUSE_MIN_PROBABILITY = 0.55        # P(Abuse) >= 0.55
ABUSE_VELOCITY_SCORE_SPAN = 3.0     # S_v saturates at 3 × the 450 px/s reference


@dataclass
class ThreatEvent:
    level: int          # 0=safe, 1=hazard, 2=fall, 3=unknown_person, 4=abuse_suspected
    type: str           # 'safe' | 'hazard' | 'fall' | 'unknown_person' | 'abuse_suspected'
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
    S_d = max(0, dot(v_hat_motion, u_hat_target)) — how much the motion
    direction points from `from_centroid` toward `toward_centroid`.
    """
    target = np.array(toward_centroid) - np.array(from_centroid)
    norm = np.linalg.norm(target)
    if norm == 0:
        return 0.0
    target_unit = target / norm
    dot = float(np.dot(np.array(direction_vector), target_unit))
    return max(0.0, dot)


def _height(p: PersonBox) -> float:
    """Remembered max height H_j_max, falling back to the current box height."""
    return p.h_max if p.h_max > 0 else p.height


class ThreatRuleEngine:
    def __init__(self):
        self._fall_counters: dict[int, int] = {}   # child track_id → consecutive fall frames
        self._abuse_counter: int = 0   # consecutive frames all abuse gates were met
        self._unknown_counter: int = 0 # consecutive frames an unknown adult was visible
        self._hazard_counter: int = 0  # consecutive frames a hazard was near a child
        # Wrist trackers persist per track_id; `_trackers` is rebuilt every frame as
        # person-index → tracker so the preview overlay can look velocities up by index.
        self._wrist_trackers: dict[int, WristVelocityTracker] = {}
        self._trackers: dict[int, WristVelocityTracker] = {}
        # Centroid-Y trackers (child fall), keyed by track_id
        self._centroid_trackers: dict[int, VelocityTracker] = {}

    def _prune(self, active_ids: set[int]) -> None:
        for store in (self._wrist_trackers, self._centroid_trackers, self._fall_counters):
            for tid in list(store):
                if tid not in active_ids:
                    del store[tid]

    def evaluate(
        self,
        detection: DetectionResult,
        keypoints_map: dict[int, Keypoints],   # person index → Keypoints
        frame_height: int,
        timestamp: float,
        frame_width: int = 640,
        face_labels: dict[int, str] | None = None,   # person index → name or "unknown"
    ) -> ThreatEvent:
        persons = detection.persons
        hazards = detection.hazards
        children = [p for p in persons if p.is_child]
        adults = [p for p in persons if not p.is_child]
        H = float(frame_height)

        self._prune({p.track_id for p in persons})

        # ── Update dominant-wrist trackers for each person ─────────────────
        # Landmarks are normalised to the full frame (0–1) → convert to pixels
        # so velocity is px/s relative to the full frame.
        self._trackers = {}
        for idx, person in enumerate(persons):
            pair = self._wrist_trackers.setdefault(person.track_id, WristVelocityTracker())
            kp = keypoints_map.get(idx)
            if kp and kp.is_valid():
                lw, rw = kp.get(LEFT_WRIST), kp.get(RIGHT_WRIST)
                pair.update(
                    (lw[0] * frame_width, lw[1] * frame_height) if lw else None,
                    (rw[0] * frame_width, rw[1] * frame_height) if rw else None,
                    timestamp,
                )
            self._trackers[idx] = pair

        # ── Level 4: Physical abuse (5-gate conjunction) ───────────────────
        # Gate 1 adult & child co-present · Gate 2 dominant-wrist V_peak above
        # threshold and noise floor · Gate 3 centroid proximity · Gate 4
        # directional alignment S_d · Gate 5 composite P(Abuse). All must hold
        # for F_abuse consecutive frames; any miss resets the counter.
        abuse_condition_met = False
        best_probability = 0.0
        best_details: dict = {}

        if adults and children:                                    # Gate 1
            noise_floor = VELOCITY_NOISE_FRAME * H
            max_dist = ABUSE_PROXIMITY_RATIO * H
            for idx, person in enumerate(persons):
                if person.is_child:
                    continue
                tracker = self._trackers[idx]
                velocity = tracker.get_velocity()                  # V_peak (EMA, 10-frame window)

                v_threshold = min(ABUSE_VELOCITY_FRAME * H,
                                  ABUSE_VELOCITY_ADULT * _height(person))
                if velocity <= noise_floor or velocity < v_threshold:   # Gate 2
                    continue

                for child in children:
                    dist = _centroid_distance(person, child)
                    if dist > max_dist:                            # Gate 3
                        continue

                    direction = tracker.get_direction_vector()
                    direction_score = _direction_toward(direction, person.centroid, child.centroid)
                    if direction_score < ABUSE_MIN_DIRECTION_SCORE:    # Gate 4
                        continue

                    velocity_score = min(
                        1.0, velocity / (ABUSE_VELOCITY_SCORE_SPAN * ABUSE_VELOCITY_FRAME * H))
                    proximity_score = max(0.0, 1.0 - dist / max_dist)
                    probability = min(
                        1.0,
                        0.5 * velocity_score + 0.3 * proximity_score + 0.2 * direction_score,
                    )
                    if probability < ABUSE_MIN_PROBABILITY:        # Gate 5
                        continue

                    abuse_condition_met = True
                    if probability > best_probability:
                        best_probability = probability
                        best_details = {
                            "adult_hand_velocity": round(velocity, 2),
                            "skeleton_distance": round(dist, 2),
                            "direction_score": round(direction_score, 2),
                            "triggered_by": ["proximity", "velocity", "direction"],
                        }

        if abuse_condition_met:
            self._abuse_counter += 1
        else:
            self._abuse_counter = 0

        if self._abuse_counter >= ABUSE_FRAME_THRESHOLD:
            self._fall_counters.clear()
            return ThreatEvent(
                level=4,
                type="abuse_suspected",
                probability=round(best_probability, 3),
                details=best_details,
            )

        # ── Level 2: Child fall ────────────────────────────────────────────
        # Evaluated per child track. Signals (any one increments that track's
        # counter; F_fall consecutive frames fire the alert):
        #   a. bbox aspect ratio w/h > 1.50
        #   b. postural inversion: y_nose > 0.5 * (y_left_hip + y_right_hip)
        #   c. downward centroid-Y velocity > min(0.281*H_frame, 0.72*H_child_max)
        fall_event = None
        for child in children:
            tid = child.track_id
            person_idx = persons.index(child)
            kp = keypoints_map.get(person_idx)

            aspect_ratio = child.width / child.height if child.height > 0 else 0
            is_horizontal = aspect_ratio > FALL_ASPECT_RATIO

            nose_below_hips = False
            if kp and kp.is_valid():
                nose = kp.get(NOSE)
                lhip = kp.get(LEFT_HIP)
                rhip = kp.get(RIGHT_HIP)
                if nose and lhip and rhip:
                    nose_below_hips = nose[1] > 0.5 * (lhip[1] + rhip[1])

            _, cy = child.centroid               # already in full-frame pixels
            ct = self._centroid_trackers.setdefault(tid, VelocityTracker())
            ct.update((0.0, cy), timestamp)
            drop_velocity = ct.get_peak_axis_velocity(axis=1, sign=1.0)   # downward only
            drop_threshold = min(FALL_DROP_FRAME * H, FALL_DROP_CHILD * _height(child))
            is_dropping = drop_velocity > drop_threshold

            triggered = []
            if is_horizontal:
                triggered.append("horizontal_bbox")
            if nose_below_hips:
                triggered.append("nose_below_hips")
            if is_dropping:
                triggered.append("vertical_drop")

            if triggered:
                self._fall_counters[tid] = self._fall_counters.get(tid, 0) + 1
            else:
                self._fall_counters[tid] = 0

            if fall_event is None and self._fall_counters[tid] >= FALL_FRAME_THRESHOLD:
                fall_event = ThreatEvent(
                    level=2,
                    type="fall",
                    probability=1.0,
                    details={
                        "triggered_by": triggered,
                        "fall_frames": self._fall_counters[tid],
                        "aspect_ratio": round(aspect_ratio, 2),
                        "drop_velocity_px_s": round(drop_velocity, 1),
                    },
                )
        if fall_event:
            return fall_event

        # ── Level 3: Unknown adult (SFace d_cos > tau_face) ───────────────
        unknown_adults = []
        if face_labels:
            for idx, person in enumerate(persons):
                if person.is_child:
                    continue
                label = face_labels.get(idx, "unknown")
                if label == "unknown":
                    unknown_adults.append(idx)

        if unknown_adults:
            self._unknown_counter += 1
        else:
            self._unknown_counter = 0

        if self._unknown_counter >= UNKNOWN_PERSON_FRAME_THRESHOLD:
            return ThreatEvent(
                level=3,
                type="unknown_person",
                probability=1.0,
                details={
                    "triggered_by": ["face_unrecognised"],
                    "unknown_count": len(unknown_adults),
                    "frames_seen": self._unknown_counter,
                },
            )

        # ── Level 1: Hazard near child ──────────────────────────────────────
        # d_box <= min(0.156*H_frame, 0.40*H_child_max), hazard confidence
        # >= 0.60, sustained F_hazard frames so one misclassified frame (a pen
        # read as a knife) cannot trigger.
        hazard_hit = None
        if hazards and children:
            for hazard in hazards:
                if hazard.confidence < HAZARD_MIN_CONFIDENCE:
                    continue
                for child in children:
                    dist = _box_proximity(
                        hazard.x1, hazard.y1, hazard.x2, hazard.y2,
                        child.x1, child.y1, child.x2, child.y2,
                    )
                    d_max = min(HAZARD_PROXIMITY_FRAME * H,
                                HAZARD_PROXIMITY_CHILD * _height(child))
                    if dist <= d_max:
                        hazard_hit = (hazard, dist)
                        break
                if hazard_hit:
                    break

        if hazard_hit:
            self._hazard_counter += 1
        else:
            self._hazard_counter = 0

        if self._hazard_counter >= HAZARD_FRAME_THRESHOLD:
            hazard, dist = hazard_hit
            return ThreatEvent(
                level=1,
                type="hazard",
                probability=1.0,
                details={
                    "hazard_object": hazard.label,
                    "distance_px": round(dist, 1),
                    "triggered_by": ["hazard_proximity"],
                    "hazard_frames": self._hazard_counter,
                },
            )

        # ── Level 0: Safe ──────────────────────────────────────────────────
        # Note: none of the sustain counters are reset here. Each is already
        # correctly maintained above (incremented when its condition is met,
        # reset to 0 otherwise) — resetting them again here would wipe any
        # counter still mid-sustain (below its threshold) every single frame,
        # making it impossible for level 2/3/4 to ever accumulate past 1.
        return ThreatEvent(level=0, type="safe", probability=0.0)
