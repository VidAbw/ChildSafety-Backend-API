# detector/capture.py
"""
Main camera loop — run this as a standalone process:
    python -m nanny_cam_guardian.detector.capture

Requires USER_ID to be set in .env.
Press 'q' in the preview window to quit.
"""
import os
import sys
import time

import cv2
import numpy as np
from dotenv import load_dotenv

# Allow imports from project root when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

load_dotenv()

from nanny_cam_guardian.detector.yolo import YOLODetector
from nanny_cam_guardian.detector.pose import PoseEstimator
from nanny_cam_guardian.detector.face import FaceRecognizer
from nanny_cam_guardian.logic.threat import ThreatRuleEngine, ThreatEvent
from nanny_cam_guardian.publisher.supabase_push import push_alert

USER_ID = os.getenv("USER_ID", "default_user")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
COLOUR_ADULT   = (255, 165, 0)    # orange
COLOUR_CHILD   = (0, 255, 255)    # yellow
COLOUR_HAZARD  = (0, 0, 255)      # red
COLOUR_SKELETON = (0, 255, 0)     # green

# Threat level → banner colour and label
LEVEL_STYLE = {
    0: ((50, 205, 50),  "SAFE"),
    1: ((0, 165, 255),  "HAZARD"),
    2: ((0, 100, 255),  "FALL DETECTED"),
    3: ((180, 0, 200),  "UNKNOWN PERSON"),
    4: ((0, 0, 220),    "ABUSE SUSPECTED"),
}

# MediaPipe pose connections (pairs of landmark indices to draw as skeleton)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
    (0, 11),  (0, 12),                                   # head to shoulders
]


WRIST_INDICES = {15: "LW", 16: "RW"}   # landmark index → label

# Velocity below this is treated as stationary (landmark noise floor)
VELOCITY_NOISE_FLOOR = 60.0   # px/s

# Speed colour gradient: green (still/slow) → yellow → red (fast)
def _speed_colour(velocity: float, threshold: float = 300.0) -> tuple:
    if velocity < VELOCITY_NOISE_FLOOR:
        return (0, 255, 0)   # green = stationary
    ratio = min(1.0, (velocity - VELOCITY_NOISE_FLOOR) / (threshold - VELOCITY_NOISE_FLOOR))
    r = int(255 * ratio)
    g = int(255 * (1.0 - ratio))
    return (0, g, r)   # BGR


def _draw_detections(frame: np.ndarray, detection, keypoints_map: dict,
                     trackers: dict, face_labels: dict | None = None) -> None:
    h, w = frame.shape[:2]
    face_labels = face_labels or {}

    for idx, person in enumerate(detection.persons):
        colour = COLOUR_CHILD if person.is_child else COLOUR_ADULT
        if person.is_child:
            label = "Child"
        else:
            face_name = face_labels.get(idx)
            if face_name and face_name != "unknown":
                label = face_name.capitalize()
                colour = (0, 200, 0)   # green = recognised
            elif face_name == "unknown":
                label = "Unknown"
                colour = (180, 0, 200)  # purple = unknown adult
            else:
                label = "Adult"
        cv2.rectangle(frame, (int(person.x1), int(person.y1)),
                      (int(person.x2), int(person.y2)), colour, 2)
        cv2.putText(frame, f"{label} {person.confidence:.0%}",
                    (int(person.x1), int(person.y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        kp = keypoints_map.get(idx)
        if kp and kp.is_valid():
            # Landmarks are normalised to the FULL frame now
            tracker  = trackers.get(idx)
            velocity = tracker.get_velocity() if tracker else 0.0

            points = {}
            for i in range(33):
                lm = kp.get(i)
                if lm and lm[2] > 0.1:   # low threshold — show even partially visible joints
                    px = int(lm[0] * w)
                    py = int(lm[1] * h)
                    px = max(0, min(w - 1, px))
                    py = max(0, min(h - 1, py))
                    points[i] = (px, py)

                    # Wrist dots: larger, colour-coded by speed
                    if i in WRIST_INDICES:
                        dot_colour = _speed_colour(velocity)
                        cv2.circle(frame, (px, py), 10, dot_colour, -1)
                        cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)  # white ring
                        cv2.putText(frame, WRIST_INDICES[i],
                                    (px + 12, py + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_colour, 1)
                    else:
                        cv2.circle(frame, (px, py), 4, COLOUR_SKELETON, -1)

            for a, b in POSE_CONNECTIONS:
                if a in points and b in points:
                    cv2.line(frame, points[a], points[b], COLOUR_SKELETON, 2)

            # Speed label below the bounding box
            speed_colour = _speed_colour(velocity)
            cv2.putText(frame, f"Speed: {velocity:.0f} px/s",
                        (int(person.x1), int(person.y2) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, speed_colour, 2)

    for hazard in detection.hazards:
        cv2.rectangle(frame, (int(hazard.x1), int(hazard.y1)),
                      (int(hazard.x2), int(hazard.y2)), COLOUR_HAZARD, 2)
        cv2.putText(frame, hazard.label.upper(),
                    (int(hazard.x1), int(hazard.y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOUR_HAZARD, 2)


def _draw_status_banner(frame: np.ndarray, event: ThreatEvent) -> None:
    colour, label = LEVEL_STYLE.get(event.level, ((128, 128, 128), "UNKNOWN"))
    h, w = frame.shape[:2]
    # Semi-transparent banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), colour, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    text = f"Level {event.level} — {label}"
    if event.level > 0:
        text += f"  ({event.probability:.0%})"
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def run():
    print(f"[capture] Starting camera {CAMERA_INDEX} for user '{USER_ID}' ...")
    print("[capture] Preview window open — press 'q' to quit.")

    yolo   = YOLODetector()
    pose   = PoseEstimator()
    face   = FaceRecognizer()
    engine = ThreatRuleEngine()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[capture] ERROR: Cannot open camera {CAMERA_INDEX}")
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[capture] WARNING: Failed to grab frame, retrying ...")
                time.sleep(0.1)
                continue

            timestamp = time.time()
            frame_h   = frame_bgr.shape[0]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── 1. Object + person detection ──────────────────────────────
            detection = yolo.detect(frame_bgr)

            # ── 2. Pose estimation on full frame, match to persons ────────
            all_kp = pose.extract_all(frame_rgb)
            keypoints_map = pose.match_to_persons(
                all_kp, detection.persons, frame_bgr.shape[1], frame_h
            )

            # ── 3. Face recognition (adults only, throttled) ──────────────
            face_labels = face.identify(frame_bgr, detection.persons)

            # ── 4. Threat classification ──────────────────────────────────
            event = engine.evaluate(detection, keypoints_map, frame_h, timestamp,
                                    frame_width=frame_bgr.shape[1],
                                    face_labels=face_labels)

            # ── 5. Push alert if actionable ───────────────────────────────
            push_alert(event, USER_ID)

            # ── 6. Draw preview ───────────────────────────────────────────
            _draw_detections(frame_bgr, detection, keypoints_map,
                             engine._trackers, face_labels)
            _draw_status_banner(frame_bgr, event)
            cv2.imshow("Nanny Cam Guardian — MM-ODG", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[capture] Quit by user.")
                break

    except KeyboardInterrupt:
        print("\n[capture] Stopped by user.")
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
