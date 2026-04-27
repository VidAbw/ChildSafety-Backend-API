# detector/yolo.py
from collections import deque
from dataclasses import dataclass, field
from ultralytics import YOLO

HAZARD_CLASSES = {"knife", "scissors", "fork"}
CHILD_HEIGHT_RATIO = 0.60     # remembered max height < 60% of tallest remembered → child
IOU_MATCH_THRESHOLD = 0.25    # minimum IoU to link a detection to an existing track
HEIGHT_MEMORY_FRAMES = 90     # frames to remember each person's max height (~3s at 30fps)


@dataclass
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    is_child: bool = False
    confidence: float = 0.0

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


class PersonTracker:
    """
    Tracks each person across frames using bbox IoU and remembers their
    maximum height over a sliding window. Classification uses the remembered
    max height so a sitting adult is never misclassified as a child.
    """

    def __init__(self) -> None:
        self._max_heights: dict[int, deque] = {}   # track_id → height history
        self._last_boxes: dict[int, PersonBox] = {} # track_id → last known box
        self._next_id: int = 0

    def classify(self, persons: list[PersonBox]) -> None:
        """Match persons to tracks, update height memory, set is_child in-place."""
        matched_ids: dict[int, int] = {}   # person_list_index → track_id

        for idx, person in enumerate(persons):
            best_id, best_iou = None, IOU_MATCH_THRESHOLD
            for tid, box in self._last_boxes.items():
                score = _iou(person, box)
                if score > best_iou:
                    best_iou, best_id = score, tid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._max_heights[best_id] = deque(maxlen=HEIGHT_MEMORY_FRAMES)

            self._max_heights[best_id].append(person.height)
            self._last_boxes[best_id] = person
            matched_ids[idx] = best_id

        # Drop stale tracks not seen this frame
        active = set(matched_ids.values())
        for tid in list(self._last_boxes):
            if tid not in active:
                del self._last_boxes[tid]
                del self._max_heights[tid]

        # Classify using remembered max height for each person
        if not persons:
            return
        remembered = {
            idx: max(self._max_heights[tid])
            for idx, tid in matched_ids.items()
        }
        tallest_remembered = max(remembered.values())
        for idx, person in enumerate(persons):
            person.is_child = remembered[idx] < tallest_remembered * CHILD_HEIGHT_RATIO


class YOLODetector:
    def __init__(self, model_path: str = "yolov8s.pt"):
        self.model = YOLO(model_path)
        self._tracker = PersonTracker()

    def detect(self, frame) -> DetectionResult:
        results = self.model(frame, verbose=False, imgsz=640)[0]
        persons: list[PersonBox] = []
        hazards: list[HazardBox] = []

        for box in results.boxes:
            label = results.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            if label == "person":
                persons.append(PersonBox(x1, y1, x2, y2, confidence=conf))
            elif label in HAZARD_CLASSES:
                hazards.append(HazardBox(x1, y1, x2, y2, label=label, confidence=conf))

        self._tracker.classify(persons)
        return DetectionResult(persons=persons, hazards=hazards)
