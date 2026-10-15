"""
UAT Scenario 2: Safety Alert Pipeline & Telemetry Ingestion
Tests user expectations around alert ingestion, severity dispatch, and alert history pagination.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_user_receives_audio_threat_ingestion(client):
    """
    User Story:
    As a parent, when my child's room acoustic sensor detects high-decibel screaming or crying,
    the backend must immediately accept and queue this critical telemetry without blocking the IoT device.
    """
    payload = {
        "event_type": "screaming",
        "confidence": 0.94,
        "rms_db": -12.5,
        "device_info": "nursery_mic_01",
        "metadata": {"room": "nursery", "child_id": "child_001"}
    }
    response = client.post("/api/v1/telemetry/audio-detection", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert data["event_type"] == "screaming"
    assert "queued for pipeline evaluation" in data["message"]


def test_user_receives_vision_hazard_ingestion(client):
    """
    User Story:
    As a parent, when the nanny cam detects a dangerous object (e.g. sharp knife or fall risk),
    the camera sends vision telemetry with bounding boxes and must receive immediate acknowledgment.
    """
    payload = {
        "event_type": "hazard",
        "confidence": 0.89,
        "bounding_box": {
            "x_min": 0.12,
            "y_min": 0.08,
            "x_max": 0.24,
            "y_max": 0.31
        },
        "camera_id": "cam_living_room",
        "metadata": {"detected_object": "knife"}
    }
    response = client.post("/api/v1/telemetry/vision-detection", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert data["event_type"] == "hazard"


def test_guardian_views_alert_history_pagination(client):
    """
    User Story:
    As a guardian opening the Mobile Companion App,
    I want to browse past alert history with accurate pagination and filters,
    so that I can review past incidents involving my child.
    """
    mock_supabase_response = MagicMock()
    mock_supabase_response.data = [
        {
            "id": "e4a2a19b-6fc2-463d-a59d-6490e9d6d581",
            "event_type": "screaming",
            "status": "triggered",
            "severity": "CRITICAL",
            "confidence": 0.95,
            "timestamp": "2026-10-15T10:00:00Z",
            "metadata": {"device": "nursery_mic"}
        }
    ]
    mock_supabase_response.count = 1

    with patch("app.api.v1.endpoints.alerts.db") as mock_db:
        # Chain mocked query calls
        mock_table = MagicMock()
        mock_db.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.order.return_value = mock_table
        mock_table.range.return_value = mock_table
        mock_table.execute.return_value = mock_supabase_response

        response = client.get("/api/v1/alerts/history?page=1&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["total"] == 1
        assert data["page"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["event_type"] == "screaming"
