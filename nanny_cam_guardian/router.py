# routers/iot.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
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
