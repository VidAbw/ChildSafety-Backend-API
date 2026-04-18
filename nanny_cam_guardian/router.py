from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from audio_guardian import AudioFrameSchema, evaluate_audio_frame
from core.supabase import db

router = APIRouter()

AUDIO_ALERT_TYPE = "acoustic_trauma"
VISION_ALERT_TYPE = "rapid_kinetic_fall"
COMPOUND_ALERT_TYPE = "threat_event"
CORRELATION_WINDOW_SECONDS = 6.0


class VisionFrameSchema(BaseModel):
    user_id: str = "default_user"
    time: str | float | int | None = None
    is_shaking: bool = False
    is_fall: bool = False
    motion_score: float | None = None


class AlertSchema(BaseModel):
    user_id: str
    source: str = "nanny_cam"
    type: str
    probability: float
    timestamp: str
    details: dict | None = None


def _parse_timestamp(value: str | float | int | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class EventRecord:
    event_type: str
    timestamp: datetime


class CorrelationEngine:
    def __init__(self) -> None:
        self._events: dict[str, deque[EventRecord]] = defaultdict(lambda: deque(maxlen=200))

    def ingest(self, user_id: str, event_type: str, timestamp: datetime) -> bool:
        user_events = self._events[user_id]
        user_events.append(EventRecord(event_type=event_type, timestamp=timestamp))

        counterpart = VISION_ALERT_TYPE if event_type == AUDIO_ALERT_TYPE else AUDIO_ALERT_TYPE
        for record in user_events:
            if record.event_type != counterpart:
                continue
            delta = abs((timestamp - record.timestamp).total_seconds())
            if delta <= CORRELATION_WINDOW_SECONDS:
                return True
        return False


_correlation_engine = CorrelationEngine()
_state_lock = asyncio.Lock()


def _save_alert(payload: dict) -> dict:
    response = db.table("alerts").insert(payload).execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to save alert")
    return response.data[0]


def _base_payload(user_id: str, source: str, alert_type: str, probability: float, timestamp: datetime, details: dict) -> dict:
    return {
        "user_id": user_id,
        "source": source,
        "type": alert_type,
        "probability": round(probability, 3),
        "timestamp": timestamp.isoformat(),
        "details": details,
    }


async def _persist_and_correlate(payload: dict, event_type: str, event_time: datetime) -> tuple[dict, dict | None]:
    async with _state_lock:
        inserted = _save_alert(payload)

        is_correlated = _correlation_engine.ingest(
            user_id=payload["user_id"],
            event_type=event_type,
            timestamp=event_time,
        )
        if not is_correlated:
            return inserted, None

        correlation_payload = _base_payload(
            user_id=payload["user_id"],
            source="fusion_engine",
            alert_type=COMPOUND_ALERT_TYPE,
            probability=min(1.0, max(payload["probability"], 0.9)),
            timestamp=event_time,
            details={
                "risk_level": "critical",
                "matched_events": [AUDIO_ALERT_TYPE, VISION_ALERT_TYPE],
                "window_seconds": CORRELATION_WINDOW_SECONDS,
            },
        )
        correlation_inserted = _save_alert(correlation_payload)
        return inserted, correlation_inserted


@router.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_json()
            frame = AudioFrameSchema.model_validate(raw_data)
            event_time = _parse_timestamp(frame.time)
            trauma_detected, scores, mfcc_shape = evaluate_audio_frame(frame)

            if not trauma_detected:
                await websocket.send_json(
                    {
                        "status": "accepted",
                        "source": "audio",
                        "triggered": False,
                        "scores": scores,
                    }
                )
                continue

            payload = _base_payload(
                user_id=frame.user_id,
                source="acoustic_node",
                alert_type=AUDIO_ALERT_TYPE,
                probability=max(scores["trauma_score"], scores["duration_score"]),
                timestamp=event_time,
                details={
                    "scores": scores,
                    "mfcc_shape": mfcc_shape,
                },
            )

            inserted, correlated = await _persist_and_correlate(
                payload=payload,
                event_type=AUDIO_ALERT_TYPE,
                event_time=event_time,
            )

            await websocket.send_json(
                {
                    "status": "accepted",
                    "source": "audio",
                    "triggered": True,
                    "alert": inserted,
                    "correlated_alert": correlated,
                }
            )
    except WebSocketDisconnect:
        return


@router.websocket("/ws/vision")
async def vision_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_json()
            frame = VisionFrameSchema.model_validate(raw_data)
            event_time = _parse_timestamp(frame.time)

            kinematic_trigger = frame.is_shaking or frame.is_fall
            if not kinematic_trigger:
                await websocket.send_json(
                    {
                        "status": "accepted",
                        "source": "vision",
                        "triggered": False,
                    }
                )
                continue

            probability = 0.9 if frame.is_fall else 0.75
            payload = _base_payload(
                user_id=frame.user_id,
                source="vision_node",
                alert_type=VISION_ALERT_TYPE,
                probability=probability,
                timestamp=event_time,
                details={
                    "is_shaking": frame.is_shaking,
                    "is_fall": frame.is_fall,
                    "motion_score": frame.motion_score,
                },
            )

            inserted, correlated = await _persist_and_correlate(
                payload=payload,
                event_type=VISION_ALERT_TYPE,
                event_time=event_time,
            )

            await websocket.send_json(
                {
                    "status": "accepted",
                    "source": "vision",
                    "triggered": True,
                    "alert": inserted,
                    "correlated_alert": correlated,
                }
            )
    except WebSocketDisconnect:
        return


@router.post("/alert")
def create_alert(alert: AlertSchema):
    inserted = _save_alert(
        {
            "user_id": alert.user_id,
            "source": alert.source,
            "type": alert.type,
            "probability": alert.probability,
            "timestamp": alert.timestamp,
            "details": alert.details,
        }
    )
    return {"status": "success", "id": inserted["id"]}


@router.get("/status")
def router_status() -> dict[str, str]:
    return {"status": "online", "module": "central_logic_hub"}
