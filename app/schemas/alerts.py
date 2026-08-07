# app/schemas/alerts.py
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AlertStatus(str, Enum):
    TRIGGERED = "triggered"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


# -----------------------------------------------------------------------------
# Telemetry Ingestion Request Models
# -----------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0, description="Top-left X coordinate (normalized 0.0 to 1.0)")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Top-left Y coordinate (normalized 0.0 to 1.0)")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Bottom-right X coordinate (normalized 0.0 to 1.0)")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Bottom-right Y coordinate (normalized 0.0 to 1.0)")


class AudioDetectionRequest(BaseModel):
    event_type: str = Field(
        default="vocal_aggression",
        description="Audio event classification type (e.g., crying, screaming, vocal_aggression)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model prediction confidence score (0.0 to 1.0)"
    )
    rms_db: float = Field(
        ...,
        description="Acoustic intensity level in decibels (dB)"
    )
    device_info: Optional[str] = Field(
        default="unknown_device",
        description="Ingestion device hardware ID or location name"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Arbitrary additional context metadata"
    )


class VisionDetectionRequest(BaseModel):
    event_type: str = Field(
        ...,
        description="Vision hazard category (e.g., fall, hit, hazard, unauthorized_presence)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Computer vision detection confidence score (0.0 to 1.0)"
    )
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Normalized bounding box coordinates of detected hazard"
    )
    camera_id: Optional[str] = Field(
        default="main_nanny_cam",
        description="Camera stream identifier or sensor zone"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Arbitrary additional context metadata"
    )


# -----------------------------------------------------------------------------
# Configuration Models
# -----------------------------------------------------------------------------

class ThresholdConfigUpdate(BaseModel):
    audio_confidence_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Audio threat confidence cutoff"
    )
    audio_rms_threshold_db: Optional[float] = Field(
        default=None, ge=0.0, le=140.0, description="Decibel intensity cutoff threshold"
    )
    vision_confidence_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Vision hazard confidence cutoff"
    )
    cooldown_seconds: Optional[float] = Field(
        default=None, ge=0.0, le=3600.0, description="Alert throttling cooldown period (seconds)"
    )
    enable_auto_suppression: Optional[bool] = Field(
        default=None, description="Toggle dynamic auto-suppression of duplicate alerts"
    )


class PipelineConfigResponse(BaseModel):
    audio_confidence_threshold: float
    audio_rms_threshold_db: float
    vision_confidence_threshold: float
    cooldown_seconds: float
    enable_auto_suppression: bool

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# Alert Log Response Models
# -----------------------------------------------------------------------------

class AlertLogResponse(BaseModel):
    id: UUID
    event_type: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    status: AlertStatus

    model_config = ConfigDict(from_attributes=True)


class PaginatedAlertHistoryResponse(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[AlertLogResponse]


class IngestionAcceptedResponse(BaseModel):
    status: str = "accepted"
    message: str
    event_type: str
