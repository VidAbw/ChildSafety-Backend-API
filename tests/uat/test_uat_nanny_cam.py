"""
UAT Scenario 4: Nanny Cam Guardian (MM-ODG)
Tests user expectations for camera controls, recognized faces, and alert dispatches.
"""
import io
import pytest
from unittest.mock import patch, MagicMock


def test_user_starts_and_stops_camera_stream(client):
    """
    User Story:
    As a parent, I want to start and stop the nanny camera stream from the companion app
    so that I can preserve privacy when monitoring is not required.
    """
    with patch("nanny_cam_guardian.router.streamer") as mock_streamer:
        # Start camera
        start_res = client.post("/api/iot/start")
        assert start_res.status_code == 200
        assert start_res.json().get("status") == "success"
        mock_streamer.start.assert_called_once()

        # Stop camera
        stop_res = client.post("/api/iot/stop")
        assert stop_res.status_code == 200
        assert stop_res.json().get("status") == "success"
        mock_streamer.stop.assert_called_once()


def test_user_dispatches_hazard_alert(client):
    """
    User Story:
    When the nanny cam detects physical abuse or dangerous objects near the child,
    it records an alert in the database and notifies the parent.
    """
    alert_payload = {
        "user_id": "usr_parent_42",
        "source": "nanny_cam",
        "type": "hazard",
        "probability": 0.96,
        "timestamp": "2026-10-15T10:15:00Z",
        "details": {
            "adult_hand_velocity": 1.45,
            "skeleton_distance": 0.85,
            "triggered_by": ["knife_detected", "rapid_approach"]
        }
    }

    mock_db_res = MagicMock()
    mock_db_res.data = [{"id": 101, "status": "saved"}]

    with patch("nanny_cam_guardian.router.db") as mock_db:
        mock_table = MagicMock()
        mock_db.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = mock_db_res

        res = client.post("/api/iot/alert", json=alert_payload)
        assert res.status_code == 200
        assert res.json().get("status") == "success"
        assert res.json().get("id") == 101
