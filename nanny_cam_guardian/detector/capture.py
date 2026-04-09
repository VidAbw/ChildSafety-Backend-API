# detector/capture.py
"""
Main camera loop — run this as a standalone process:
    python detector/capture.py

Requires USER_ID to be set in .env.
"""
import os
import sys
import time

import cv2
from dotenv import load_dotenv

# Allow imports from project root when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

load_dotenv()

from nanny_cam_guardian.detector.yolo import YOLODetector
from nanny_cam_guardian.detector.pose import PoseEstimator
from nanny_cam_guardian.logic.threat import ThreatRuleEngine
from nanny_cam_guardian.publisher.supabase_push import push_alert

USER_ID = os.getenv("USER_ID", "default_user")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))


def run():
    print(f"[capture] Starting camera {CAMERA_INDEX} for user '{USER_ID}' ...")

    yolo = YOLODetector()
    pose = PoseEstimator()
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
            frame_h = frame_bgr.shape[0]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── 1. Object + person detection ──────────────────────────────
            detection = yolo.detect(frame_bgr)

            # ── 2. Pose estimation per person ─────────────────────────────
            keypoints_map = {}
            for idx, person in enumerate(detection.persons):
                kp = pose.extract_region(
                    frame_rgb,
                    person.x1, person.y1,
                    person.x2, person.y2,
                )
                keypoints_map[idx] = kp

            # ── 3. Threat classification ──────────────────────────────────
            event = engine.evaluate(detection, keypoints_map, frame_h, timestamp)

            # ── 4. Push alert if actionable ───────────────────────────────
            push_alert(event, USER_ID)

    except KeyboardInterrupt:
        print("\n[capture] Stopped by user.")
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
