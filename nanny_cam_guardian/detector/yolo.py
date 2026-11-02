# detector/yolo.py
import os
from collections import deque
from dataclasses import dataclass, field
from ultralytics import YOLO

HAZARD_CLASSES = {"knife", "scissors", "fork"}
PERSON_CONF_THRESHOLD = 0.5   # confidence floor for person detection — filters out weak/
                               # partial detections (e.g. a hand alone) that would otherwise
                               # flow through as an "unrecognized person"
CHILD_HEIGHT_RATIO = 0.60     # rho: H_j_max < rho * H_ref → child (unless identity verified)
IOU_MATCH_THRESHOLD = 0.25    # minimum IoU to link a detection to an existing track
HEIGHT_MEMORY_FRAMES = 90     # |W_j|: sliding window of box heights (3.0 s at 30 fps)
TRACK_MAX_AGE_FRAMES = 30     # keep an unmatched track (and its window) alive ~1 s for occlusions
H_FRAME_REF = 640             # reference frame height the pixel constants below are quoted at
ADULT_REF_HEIGHT_640 = 360    # room-calibrated standing adult height at 640x640 (px)
ADULT_REF_EMA = 0.05          # per-frame blend when a verified standing adult updates H_adult_ref
STANDING_MAX_WH_RATIO = 0.60  # standing posture: bbox width/height below this
EDGE_MARGIN = 0.01            # bbox touching the frame edge within this fraction = truncated
IDENTITY_UNKNOWN = "Unknown"
IDENTITY_VERIFIED_ADULT = "Verified_Adult"

# Optional fine-tuned hazard-specialist model. If present in the project root,
# it is loaded and used for hazards while stock yolov8s handles persons.
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
HAZARD_MODEL_PATH = os.path.join(MODELS_DIR, "hazard_yolo.pt")
HAZARD_CONF_THRESHOLD = 0.6    # confidence floor for the specialist model (raised after
                                # fine-tuning on pen/brush negatives + real knife positives;
                                # filters remaining pen/brush stragglers up to ~52% conf while
                                # keeping real knife detections, which score ~80%)
STOCK_HAZARD_CONF_THRESHOLD = 0.55  # higher floor for the generic stock model, which
                                     # misclassifies pens/sticks as knives at low confidence


@dataclass
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    is_child: bool = False
    confidence: float = 0.0
    track_id: int = -1          # persistent track id assigned by PersonTracker
    h_max: float = 0.0          # H_j_max: max box height over the sliding window (px)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class HazardBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float = 0.0

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    persons: list[PersonBox] = field(default_factory=list)
    hazards: list[HazardBox] = field(default_factory=list)


def _iou(a: PersonBox, b: PersonBox) -> float:
    """Intersection-over-Union between two bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = a.width * a.height
    area_b = b.width * b.height
    return inter / (area_a + area_b - inter)


@dataclass
class _Track:
    box: PersonBox
    heights: deque = field(default_factory=lambda: deque(maxlen=HEIGHT_MEMORY_FRAMES))
    identity: str = IDENTITY_UNKNOWN
    missed: int = 0

    @property
    def h_max(self) -> float:
        """H_j_max = max_{t in W_j} (y2_t - y1_t)."""
        return max(self.heights) if self.heights else 0.0


class PersonTracker:
    """
    IoU tracker with a per-person sliding-window height buffer W_j (90 frames).

        H_j_max = max over W_j of box height
        H_ref   = max(H_adult_ref, max_i H_i_max)
        child_j = H_j_max < rho * H_ref  AND  identity_j != Verified_Adult

    Anchoring on H_adult_ref keeps a lone child from being read as an adult;
    the remembered maximum and the identity override keep a seated/crouching
    (or SFace-verified) adult from being read as a child.

    H_adult_ref is stored as a fraction of frame height so it is resolution
    independent (360 px at 640x640 → 0.5625). It is updated whenever an
    authorised adult is verified in standing posture.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self._next_id: int = 0
        self._adult_ref_frac: float = ADULT_REF_HEIGHT_640 / H_FRAME_REF

    # ── calibration ────────────────────────────────────────────────────────
    def calibrate_adult_height(self, height_px: float, frame_h: float) -> None:
        """Room-installation calibration with a measured standing-adult height."""
        if height_px > 0 and frame_h > 0:
            self._adult_ref_frac = height_px / frame_h

    def adult_ref_height(self, frame_h: float) -> float:
        return self._adult_ref_frac * frame_h

    # ── tracking ───────────────────────────────────────────────────────────
    def classify(self, persons: list[PersonBox], frame_h: float = H_FRAME_REF) -> None:
        """Match persons to tracks, update windows, set track_id/h_max/is_child in-place."""
        # One-to-one greedy association by descending IoU
        pairs = []
        for idx, person in enumerate(persons):
            for tid, trk in self._tracks.items():
                score = _iou(person, trk.box)
                if score > IOU_MATCH_THRESHOLD:
                    pairs.append((score, idx, tid))
        pairs.sort(reverse=True)

        matched: dict[int, int] = {}      # person idx → track id
        used_tracks: set[int] = set()
        for _, idx, tid in pairs:
            if idx in matched or tid in used_tracks:
                continue
            matched[idx] = tid
            used_tracks.add(tid)

        for idx, person in enumerate(persons):
            if idx not in matched:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(box=person)
                matched[idx] = tid
            trk = self._tracks[matched[idx]]
            trk.heights.append(person.height)
            trk.box = person
            trk.missed = 0
            person.track_id = matched[idx]

        # Age out tracks not seen this frame (window survives short occlusions)
        for tid in list(self._tracks):
            if tid not in used_tracks and tid not in matched.values():
                self._tracks[tid].missed += 1
                if self._tracks[tid].missed > TRACK_MAX_AGE_FRAMES:
                    del self._tracks[tid]

        self._assign_classes(persons, frame_h)

    def _assign_classes(self, persons: list[PersonBox], frame_h: float) -> None:
        if not persons:
            return
        h_ref = max(
            self.adult_ref_height(frame_h),
            max(t.h_max for t in self._tracks.values()),
        )
        for person in persons:
            trk = self._tracks[person.track_id]
            person.h_max = trk.h_max
            person.is_child = (
                trk.h_max < CHILD_HEIGHT_RATIO * h_ref
                and trk.identity != IDENTITY_VERIFIED_ADULT
            )

    # ── identity feedback from SFace ───────────────────────────────────────
    def update_identities(self, persons: list[PersonBox],
                          labels: dict[int, str], frame_h: float) -> None:
        """
        Record face-verification results per track, refresh H_adult_ref from
        verified standing adults, then re-run classification so a verified
        adult is never labelled a child in the same frame.
        """
        for idx, person in enumerate(persons):
            label = labels.get(idx)
            trk = self._tracks.get(person.track_id)
            if label is None or trk is None:
                continue
            if label == "unknown":
                trk.identity = IDENTITY_UNKNOWN
                continue
            trk.identity = IDENTITY_VERIFIED_ADULT
            if self._is_standing(person, frame_h):
                frac = trk.h_max / frame_h
                self._adult_ref_frac += ADULT_REF_EMA * (frac - self._adult_ref_frac)
        self._assign_classes(persons, frame_h)

    @staticmethod
    def _is_standing(p: PersonBox, frame_h: float) -> bool:
        if p.height <= 0:
            return False
        upright = (p.width / p.height) < STANDING_MAX_WH_RATIO
        untruncated = (p.y1 > EDGE_MARGIN * frame_h
                       and p.y2 < (1.0 - EDGE_MARGIN) * frame_h)
        return upright and untruncated


class YOLODetector:
    def __init__(self, model_path: str = "yolov8s.pt"):
        self.model = YOLO(model_path)
        self._tracker = PersonTracker()
        self._frame_h: float = float(H_FRAME_REF)

        # Optional fine-tuned knife/hazard model — loaded only if present.
        self.hazard_model = None
        if os.path.isfile(HAZARD_MODEL_PATH):
            print(f"[yolo] Loading fine-tuned hazard model: {HAZARD_MODEL_PATH}")
            self.hazard_model = YOLO(HAZARD_MODEL_PATH)
        else:
            print(f"[yolo] No {HAZARD_MODEL_PATH} found — using stock model for hazards.")

    def detect(self, frame) -> DetectionResult:
        results = self.model(frame, verbose=False, imgsz=640)[0]
        persons: list[PersonBox] = []
        hazards: list[HazardBox] = []

        for box in results.boxes:
            label = results.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            if label == "person":
                if conf >= PERSON_CONF_THRESHOLD:
                    persons.append(PersonBox(x1, y1, x2, y2, confidence=conf))
            elif label in HAZARD_CLASSES and self.hazard_model is None:
                # Only use stock model for hazards when no specialist exists
                if conf >= STOCK_HAZARD_CONF_THRESHOLD:
                    hazards.append(HazardBox(x1, y1, x2, y2, label=label, confidence=conf))

        # Specialist hazard pass — fine-tuned model is far better at distance
        if self.hazard_model is not None:
            hres = self.hazard_model(frame, verbose=False, imgsz=640,
                                     conf=HAZARD_CONF_THRESHOLD)[0]
            for box in hres.boxes:
                label = hres.names[int(box.cls)]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                hazards.append(HazardBox(x1, y1, x2, y2, label=label, confidence=conf))

        self._frame_h = float(frame.shape[0])
        self._tracker.classify(persons, self._frame_h)
        return DetectionResult(persons=persons, hazards=hazards)

    def update_identities(self, persons: list[PersonBox], face_labels: dict[int, str]) -> None:
        """Feed SFace results back so verified adults override the height rule."""
        self._tracker.update_identities(persons, face_labels, self._frame_h)
