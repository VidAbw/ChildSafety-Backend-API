# app/core/pipeline_manager.py
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from core.supabase import db
from app.schemas.alerts import AlertStatus, ThresholdConfigUpdate

logger = logging.getLogger("SafetyAlertPipeline")


class PipelineConfig:
    """
    Dynamic configuration container for safety threshold parameters (\theta).
    Allows runtime threshold and cooldown updates without restarting the application.
    """
    def __init__(
        self,
        audio_confidence_threshold: float = 0.75,
        audio_rms_threshold_db: float = 70.0,
        vision_confidence_threshold: float = 0.70,
        cooldown_seconds: float = 10.0,
        enable_auto_suppression: bool = True
    ):
        self.audio_confidence_threshold = audio_confidence_threshold
        self.audio_rms_threshold_db = audio_rms_threshold_db
        self.vision_confidence_threshold = vision_confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.enable_auto_suppression = enable_auto_suppression

    def update(self, update_data: ThresholdConfigUpdate) -> None:
        """Dynamically update threshold values at runtime."""
        if update_data.audio_confidence_threshold is not None:
            self.audio_confidence_threshold = update_data.audio_confidence_threshold
        if update_data.audio_rms_threshold_db is not None:
            self.audio_rms_threshold_db = update_data.audio_rms_threshold_db
        if update_data.vision_confidence_threshold is not None:
            self.vision_confidence_threshold = update_data.vision_confidence_threshold
        if update_data.cooldown_seconds is not None:
            self.cooldown_seconds = update_data.cooldown_seconds
        if update_data.enable_auto_suppression is not None:
            self.enable_auto_suppression = update_data.enable_auto_suppression

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_confidence_threshold": self.audio_confidence_threshold,
            "audio_rms_threshold_db": self.audio_rms_threshold_db,
            "vision_confidence_threshold": self.vision_confidence_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "enable_auto_suppression": self.enable_auto_suppression
        }


class SafetyAlertPipeline:
    """
    Integration layer & core evaluation pipeline for Child Safety Detections.
    Processes audio and vision telemetry streams asynchronously, applies threshold cuts,
    evaluates cooldown suppression, and persists alert logs to Supabase database.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._last_alert_timestamps: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def ingest_audio(
        self,
        event_type: str,
        confidence: float,
        rms_db: float,
        device_info: str = "unknown_device",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest and evaluate audio threat telemetry."""
        meta = dict(metadata or {})
        meta.update({
            "rms_db": rms_db,
            "device_info": device_info,
            "source_pipeline": "audio_guardian"
        })

        is_threat = (
            confidence >= self.config.audio_confidence_threshold or
            rms_db >= self.config.audio_rms_threshold_db
        )

        return await self.ingest_detection(
            event_type=event_type,
            confidence=confidence,
            is_threat=is_threat,
            metadata=meta
        )

    async def ingest_vision(
        self,
        event_type: str,
        confidence: float,
        bounding_box: Optional[Dict[str, float]] = None,
        camera_id: str = "main_nanny_cam",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest and evaluate vision hazard telemetry."""
        meta = dict(metadata or {})
        meta.update({
            "bounding_box": bounding_box,
            "camera_id": camera_id,
            "source_pipeline": "nanny_cam_guardian"
        })

        is_threat = confidence >= self.config.vision_confidence_threshold

        return await self.ingest_detection(
            event_type=event_type,
            confidence=confidence,
            is_threat=is_threat,
            metadata=meta
        )

    async def ingest_detection(
        self,
        event_type: str,
        confidence: float,
        is_threat: bool,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Asynchronous detection ingestion core method.
        Evaluates threat status, checks cooldown rules, and persists event log to Supabase.
        """
        now = datetime.now(timezone.utc)
        status = AlertStatus.TRIGGERED.value

        async with self._lock:
            # Cooldown check for alert throttling
            last_time = self._last_alert_timestamps.get(event_type)
            if last_time and (now - last_time).total_seconds() < self.config.cooldown_seconds:
                if self.config.enable_auto_suppression:
                    status = AlertStatus.SUPPRESSED.value
                    metadata["suppression_reason"] = (
                        f"Alert throttled due to active cooldown window ({self.config.cooldown_seconds}s)"
                    )

            if not is_threat and status != AlertStatus.SUPPRESSED.value:
                status = AlertStatus.SUPPRESSED.value
                metadata["suppression_reason"] = "Detection confidence below active threshold parameter"

            # Update last alert timestamp if triggered
            if status == AlertStatus.TRIGGERED.value:
                self._last_alert_timestamps[event_type] = now

        record = {
            "id": str(uuid4()),
            "event_type": event_type,
            "confidence": float(confidence),
            "timestamp": now.isoformat(),
            "metadata": metadata,
            "status": status
        }

        # Asynchronously log event to Supabase
        await self._persist_to_supabase(record)
        return record

    async def _persist_to_supabase(self, record: Dict[str, Any]) -> None:
        """
        Persists alert log record to Supabase alert_logs table asynchronously.
        Executes database write in executor pool to prevent blocking the main thread.
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: db.table("alert_logs").insert(record).execute()
            )
            logger.info(
                f"Logged alert event to Supabase [Status: {record['status'].upper()}] - "
                f"Event: '{record['event_type']}', Confidence: {record['confidence']:.2f} (ID: {record['id']})"
            )
        except Exception as exc:
            logger.error(f"Error persisting alert event {record['id']} to Supabase: {exc}", exc_info=True)


# Global Singleton Pipeline Instance
pipeline_manager = SafetyAlertPipeline()
