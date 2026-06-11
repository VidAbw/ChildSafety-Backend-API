# routers/iot.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time
from nanny_cam_guardian.detector.capture import NannyCamStreamer

streamer = NannyCamStreamer()
from core.supabase import db

router = APIRouter()


class AlertDetails(BaseModel):
    adult_hand_velocity: float | None = None
    skeleton_distance: float | None = None
    triggered_by: list[str] = []


class AlertSchema(BaseModel):
    user_id: str
    source: str = "nanny_cam"   # default — overridden if called by another component
    type: str                   # 'hazard' | 'fall' | 'abuse_suspected'
    probability: float
    timestamp: str
    details: AlertDetails | None = None


@router.post("/alert")
def create_alert(alert: AlertSchema):
    data = {
        "user_id": alert.user_id,
        "source": alert.source,
        "type": alert.type,
        "probability": alert.probability,
        "timestamp": alert.timestamp,
        "details": alert.details.model_dump() if alert.details else None,
    }
    response = db.table("alerts").insert(data).execute()

    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to save alert")

    return {"status": "success", "id": response.data[0]["id"]}

@router.post("/start")
def start_camera():
    streamer.start()
    return {"status": "success", "message": "Camera started"}

@router.post("/stop")
def stop_camera():
    streamer.stop()
    return {"status": "success", "message": "Camera stopped"}

@router.get("/stream")
def stream_camera():
    from fastapi import Response
    frame = streamer.get_frame()
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    if frame:
        return Response(content=frame, media_type="image/jpeg", headers=headers)
    
    # Return a 1x1 transparent pixel if not ready
    transparent_pixel = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
    return Response(content=transparent_pixel, media_type="image/gif", headers=headers)

