import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import websockets

from core.supabase import db

logger = logging.getLogger(__name__)


class PhoneAudioListener:
    def __init__(self) -> None:
        phone_ip = os.getenv("PHONE_IP", "").strip()
        phone_port = os.getenv("PHONE_WS_PORT", "8080").strip()
        sensor_type = os.getenv("PHONE_SENSOR_TYPE", "android.sensor.sound").strip()

        ws_url = os.getenv("PHONE_WS_URL", "").strip()
        if ws_url:
            self.ws_url = ws_url
        elif phone_ip:
            self.ws_url = f"ws://{phone_ip}:{phone_port}/sensor/connect?type={sensor_type}"
        else:
            self.ws_url = ""

        self.threshold = float(os.getenv("AUDIO_STRESS_THRESHOLD", "85.0"))
        self.reconnect_seconds = float(os.getenv("AUDIO_RECONNECT_SECONDS", "3"))
        self.alerts_table = os.getenv("AUDIO_ALERTS_TABLE", "threat_alerts").strip()
        self.user_id = os.getenv("AUDIO_USER_ID", "").strip()

        self.connected = False
        self.last_value: float | None = None
        self.last_message_at: str | None = None
        self.last_error: str | None = None

        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if not self.ws_url:
            self.last_error = "PHONE_IP or PHONE_WS_URL is not configured."
            logger.warning(self.last_error)
            return

        if self._task and not self._task.done():
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def status(self) -> dict:
        return {
            "connected": self.connected,
            "ws_url": self.ws_url,
            "threshold": self.threshold,
            "last_value": self.last_value,
            "last_message_at": self.last_message_at,
            "last_error": self.last_error,
        }

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to SensorServer: %s", self.ws_url)
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as websocket:
                    self.connected = True
                    self.last_error = None
                    logger.info("Connected to SensorServer. Listening for audio packets.")

                    async for raw_message in websocket:
                        if self._stop_event.is_set():
                            break
                        self._handle_message(raw_message)

            except Exception as exc:
                self.last_error = str(exc)
                logger.warning("Audio listener disconnected: %s", exc)

            finally:
                self.connected = False

            if not self._stop_event.is_set():
                await asyncio.sleep(self.reconnect_seconds)

    def _handle_message(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return

        value = self._extract_audio_value(payload)
        if value is None:
            return

        self.last_value = value
        self.last_message_at = datetime.now(timezone.utc).isoformat()

        if value > self.threshold:
            logger.warning("High stress detected: %.2f dB", value)
            self._trigger_supabase_alert(value)
        else:
            logger.info("Background audio level: %.2f dB", value)

    def _extract_audio_value(self, payload: Any) -> float | None:
        if isinstance(payload, (int, float)):
            return float(payload)

        if not isinstance(payload, dict):
            return None

        values = payload.get("values")
        if isinstance(values, list) and values:
            first = values[0]
            if isinstance(first, (int, float, str)):
                try:
                    return float(first)
                except ValueError:
                    return None

        for key in ("value", "sound", "amplitude", "decibel", "db"):
            candidate = payload.get(key)
            if isinstance(candidate, (int, float, str)):
                try:
                    return float(candidate)
                except ValueError:
                    return None

        return None

    def _trigger_supabase_alert(self, intensity_score: float) -> None:
        data = {
            "sensor_type": "acoustic",
            "threat_category": "Vocal Aggression / Screaming",
            "intensity": intensity_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pending_verification",
        }

        if self.user_id:
            data["user_id"] = self.user_id

        try:
            db.table(self.alerts_table).insert(data).execute()
            logger.info("Alert pushed to Supabase table '%s'.", self.alerts_table)
        except Exception as exc:
            self.last_error = str(exc)
            logger.error("Failed to push alert to Supabase: %s", exc)


phone_audio_listener = PhoneAudioListener()
