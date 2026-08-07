# app/api/v1/endpoints/alerts.py
import asyncio
import logging
from datetime import datetime, timezone
from math import ceil
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from core.supabase import db
from app.core.pipeline_manager import pipeline_manager
from app.schemas.alerts import (
    AlertLogResponse,
    AlertStatus,
    AudioDetectionRequest,
    IngestionAcceptedResponse,
    PaginatedAlertHistoryResponse,
    PipelineConfigResponse,
    ThresholdConfigUpdate,
    VisionDetectionRequest,
)

logger = logging.getLogger("AlertsAPI")

router = APIRouter()


# -----------------------------------------------------------------------------
# 1. Telemetry Ingestion Endpoints (Non-Blocking via BackgroundTasks)
# -----------------------------------------------------------------------------

@router.post(
    "/telemetry/audio-detection",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestionAcceptedResponse,
    summary="Ingest Audio Threat Telemetry",
    description="Asynchronously ingests audio threat detections (e.g. screaming, crying, vocal aggression) without blocking REST caller."
)
async def ingest_audio_detection(
    payload: AudioDetectionRequest,
    background_tasks: BackgroundTasks
):
    """Route incoming audio telemetry payload into SafetyAlertPipeline asynchronously."""
    background_tasks.add_task(
        pipeline_manager.ingest_audio,
        event_type=payload.event_type,
        confidence=payload.confidence,
        rms_db=payload.rms_db,
        device_info=payload.device_info,
        metadata=payload.metadata
    )
    return IngestionAcceptedResponse(
        status="accepted",
        message="Audio threat telemetry accepted and queued for pipeline evaluation",
        event_type=payload.event_type
    )


@router.post(
    "/telemetry/vision-detection",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestionAcceptedResponse,
    summary="Ingest Vision Hazard Telemetry",
    description="Asynchronously ingests computer vision hazard detections (e.g. fall, hit, hazard, unauthorized presence)."
)
async def ingest_vision_detection(
    payload: VisionDetectionRequest,
    background_tasks: BackgroundTasks
):
    """Route incoming vision telemetry payload into SafetyAlertPipeline asynchronously."""
    bbox_dict = payload.bounding_box.model_dump() if payload.bounding_box else None

    background_tasks.add_task(
        pipeline_manager.ingest_vision,
        event_type=payload.event_type,
        confidence=payload.confidence,
        bounding_box=bbox_dict,
        camera_id=payload.camera_id,
        metadata=payload.metadata
    )
    return IngestionAcceptedResponse(
        status="accepted",
        message="Vision hazard telemetry accepted and queued for pipeline evaluation",
        event_type=payload.event_type
    )


# -----------------------------------------------------------------------------
# 2. Alert Log Management REST APIs (Mobile Companion App)
# -----------------------------------------------------------------------------

@router.get(
    "/alerts/history",
    response_model=PaginatedAlertHistoryResponse,
    summary="Retrieve Alert Log History",
    description="Fetch recent alert logs with pagination and optional filtering by event_type or status."
)
async def get_alert_history(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    event_type: Optional[str] = Query(default=None, description="Filter by specific event type"),
    status: Optional[AlertStatus] = Query(default=None, description="Filter by alert status (triggered, suppressed, acknowledged)")
):
    """Query recent alert logs from Supabase with pagination and filters."""
    loop = asyncio.get_running_loop()

    def _query_supabase():
        # Base query for data
        query = db.table("alert_logs").select("*", count="exact")

        if event_type:
            query = query.eq("event_type", event_type)
        if status:
            query = query.eq("status", status.value if isinstance(status, AlertStatus) else status)

        # Ordering and Pagination
        offset_start = (page - 1) * limit
        offset_end = offset_start + limit - 1

        query = query.order("timestamp", desc=True).range(offset_start, offset_end)
        response = query.execute()
        return response

    try:
        response = await loop.run_in_executor(None, _query_supabase)
        data = response.data or []
        total_count = response.count if response.count is not None else len(data)

        total_pages = ceil(total_count / limit) if total_count > 0 else 1

        return PaginatedAlertHistoryResponse(
            total=total_count,
            page=page,
            limit=limit,
            pages=total_pages,
            items=[AlertLogResponse.model_validate(item) for item in data]
        )
    except Exception as exc:
        logger.error(f"Error fetching alert history from Supabase: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query alert history database: {str(exc)}"
        )


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertLogResponse,
    summary="Acknowledge Alert Log",
    description="Mark an active alert log as acknowledged by a parent or authorized user."
)
async def acknowledge_alert(alert_id: UUID):
    """Update alert log status in Supabase to 'acknowledged'."""
    str_id = str(alert_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    loop = asyncio.get_running_loop()

    def _update_supabase():
        # First check if record exists
        existing = db.table("alert_logs").select("*").eq("id", str_id).execute()
        if not existing.data:
            return None

        current_metadata = dict(existing.data[0].get("metadata") or {})
        current_metadata["acknowledged_at"] = now_iso

        result = db.table("alert_logs").update({
            "status": AlertStatus.ACKNOWLEDGED.value,
            "metadata": current_metadata
        }).eq("id", str_id).execute()

        return result.data[0] if result.data else None

    try:
        updated_record = await loop.run_in_executor(None, _update_supabase)
        if not updated_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert log with ID '{alert_id}' was not found."
            )
        return AlertLogResponse.model_validate(updated_record)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error acknowledging alert {alert_id} in Supabase: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database update failed: {str(exc)}"
        )


# -----------------------------------------------------------------------------
# 3. Dynamic Pipeline Threshold Configuration Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/config/thresholds",
    response_model=PipelineConfigResponse,
    summary="Get Pipeline Threshold Configuration",
    description="Retrieve current active pipeline cutoff thresholds (\theta) and cooldown parameters."
)
async def get_threshold_config():
    """Retrieve current PipelineConfig parameters."""
    return PipelineConfigResponse.model_validate(pipeline_manager.config.to_dict())


@router.put(
    "/config/thresholds",
    response_model=PipelineConfigResponse,
    summary="Update Pipeline Threshold Configuration",
    description="Dynamically update active pipeline cutoff thresholds (\theta) and cooldown parameters at runtime."
)
async def update_threshold_config(update_data: ThresholdConfigUpdate):
    """Dynamically update threshold values in PipelineConfig."""
    pipeline_manager.config.update(update_data)
    logger.info(f"Pipeline threshold configuration updated: {pipeline_manager.config.to_dict()}")
    return PipelineConfigResponse.model_validate(pipeline_manager.config.to_dict())
