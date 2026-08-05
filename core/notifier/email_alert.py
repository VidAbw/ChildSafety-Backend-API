# core/notifier/email_alert.py
"""
Shared email alerting for guardians, used by both nanny_cam_guardian
(hazard/fall/unknown_person/abuse_suspected) and audio_guardian (vocal
aggression). Every public function here is safe to call from any thread and
never raises — a notification failure must never break the caller (camera
capture loop, ESP32 upload handler, etc).
"""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

from core.supabase import db

if TYPE_CHECKING:
    from nanny_cam_guardian.logic.threat import ThreatEvent

logger = logging.getLogger(__name__)

APP_NAME = "Child Safety Monitor"

NANNY_CAM_SUBJECTS = {
    "hazard": "⚠️ Hazard Alert — Child Safety Monitor",
    "fall": "🚨 Fall Detected — Child Safety Monitor",
    "unknown_person": "👤 Unknown Person Detected — Child Safety Monitor",
    "abuse_suspected": "🚨 URGENT: Possible Abuse Suspected — Child Safety Monitor",
}
NANNY_CAM_HEADLINES = {
    "hazard": "A hazardous object was detected near your child.",
    "fall": "Your child may have fallen — please check immediately.",
    "unknown_person": "An unrecognized adult was detected near your child.",
    "abuse_suspected": "Possible abuse detected — please check immediately.",
}


def _send_email(to_email: str, subject: str, body: str) -> bool:
    """Low-level SMTP send. Never raises — logs and returns False on failure."""
    try:
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("ALERT_FROM_EMAIL")

        if not all([smtp_host, smtp_user, smtp_password, from_email]):
            logger.error(
                "[notifier] SMTP not configured (SMTP_HOST/SMTP_USER/SMTP_PASSWORD/"
                "ALERT_FROM_EMAIL) — cannot send alert email to %s",
                to_email,
            )
            return False

        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, [to_email], msg.as_string())

        logger.info("[notifier] Alert email sent to %s: %s", to_email, subject)
        return True
    except Exception as exc:
        logger.error("[notifier] Failed to send alert email to %s: %s", to_email, exc)
        return False


def get_guardian_email(user_id: str) -> tuple[str, str] | None:
    """
    Returns (email, name) for the guardian's real Supabase Auth account, or
    None if not found. `user_id` must be the account's actual Supabase Auth
    UUID (looked up via the Auth Admin API — requires a service_role key).
    """
    try:
        response = db.auth.admin.get_user_by_id(user_id)
        user = response.user
        if not user or not user.email:
            logger.warning("[notifier] No Supabase Auth account/email found for user_id=%s", user_id)
            return None

        metadata = user.user_metadata or {}
        name = metadata.get("full_name") or metadata.get("name") or ""
        return user.email, name
    except Exception as exc:
        logger.error("[notifier] Failed to look up guardian for user_id=%s: %s", user_id, exc)
        return None


def _greeting(guardian_name: str) -> str:
    return f"Hi {guardian_name}," if guardian_name else "Hi,"


def send_nanny_cam_alert_email(event: "ThreatEvent", user_id: str) -> None:
    """Emails the registered guardian about a nanny-cam threat event. Never raises."""
    try:
        guardian = get_guardian_email(user_id)
        if guardian is None:
            return
        guardian_email, guardian_name = guardian

        subject = NANNY_CAM_SUBJECTS.get(event.type, f"{APP_NAME} — Safety Alert")
        headline = NANNY_CAM_HEADLINES.get(event.type, "A safety event was detected.")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        body_lines = [
            _greeting(guardian_name),
            "",
            headline,
            "",
            f"Confidence: {event.probability:.0%}",
            f"Time: {timestamp}",
        ]
        if event.type == "hazard" and event.details.get("hazard_object"):
            body_lines.append(f"Object: {event.details['hazard_object']}")

        body_lines += ["", f"This is an automated alert from your {APP_NAME} system."]
        body = "\n".join(body_lines)

        _send_email(guardian_email, subject, body)
    except Exception as exc:
        logger.error("[notifier] send_nanny_cam_alert_email failed for user_id=%s: %s", user_id, exc)


def _intensity_descriptor(intensity_score: float) -> str:
    if intensity_score >= 90:
        return "very loud"
    if intensity_score >= 70:
        return "loud"
    return "elevated"


def send_audio_alert_email(
    user_id: str, intensity_score: float, threat_level: str, device_info: str
) -> None:
    """Emails the registered guardian about a vocal-aggression detection. Never raises."""
    try:
        guardian = get_guardian_email(user_id)
        if guardian is None:
            return
        guardian_email, guardian_name = guardian

        if threat_level == "high":
            subject = "🚨 URGENT: Loud Vocal Aggression Detected — Child Safety Monitor"
        else:
            subject = "⚠️ Possible Vocal Aggression Detected — Child Safety Monitor"

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        descriptor = _intensity_descriptor(intensity_score)

        body_lines = [
            _greeting(guardian_name),
            "",
            f"Elevated vocal stress/aggression was detected near {device_info}.",
            f"The sound level was {descriptor}.",
            "",
            f"Time: {timestamp}",
            "",
            f"This is an automated alert from your {APP_NAME} system.",
        ]
        body = "\n".join(body_lines)

        _send_email(guardian_email, subject, body)
    except Exception as exc:
        logger.error("[notifier] send_audio_alert_email failed for user_id=%s: %s", user_id, exc)
