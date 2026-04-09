# detector/yolo.py
from dataclasses import dataclass, field
from ultralytics import YOLO

HAZARD_CLASSES = {"knife", "scissors", "fork"}
CHILD_HEIGHT_RATIO = 0.60   # bbox height < 60% of tallest person → classified as child


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


def _classify_children(persons: list[PersonBox]) -> None:
    if not persons:
        return
    tallest_height = max(p.height for p in persons)
    for person in persons:
        person.is_child = person.height < tallest_height * CHILD_HEIGHT_RATIO


class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame) -> DetectionResult:
        results = self.model(frame, verbose=False)[0]
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

        _classify_children(persons)
        return DetectionResult(persons=persons, hazards=hazards)
