# publisher/supabase_push.py
from datetime import datetime, timezone

from core.supabase import db
from nanny_cam_guardian.logic.threat import ThreatEvent


def push_alert(event: ThreatEvent, user_id: str) -> None:
    """
    Insert a ThreatEvent into the Supabase `alerts` table.
    Level 0 (safe) events are silently ignored.
    """
    if event.level == 0:
        return

    payload = {
        "user_id": user_id,
        "source": "nanny_cam",
        "type": event.type,
        "probability": event.probability,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": event.details,
    }

    response = db.table("alerts").insert(payload).execute()
    if not response.data:
        print(f"[publisher] WARNING: failed to insert alert — {payload}")
    else:
        print(f"[publisher] Alert pushed: level={event.level} type={event.type} prob={event.probability}")
