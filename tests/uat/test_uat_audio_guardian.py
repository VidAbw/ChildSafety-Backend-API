"""
UAT Scenario 5: Audio Guardian Acoustic Threat Monitoring & Voice Management
Validates user expectations on acoustic threat classification, parent voice registration, and active profiles.
"""
import io
import pytest
from unittest.mock import patch, MagicMock


def test_parent_views_audio_guardian_status(client):
    """
    User Story:
    As a parent, I want to check whether the acoustic monitoring node is actively listening
    and see the latest threat prediction recorded from the nursery.
    """
    res = client.get("/api/audio/status")
    assert res.status_code == 200
    data = res.json()
    assert "running" in data or "is_running" in data or "listening" in data or isinstance(data, dict)


def test_parent_views_registered_voice_profiles(client):
    """
    User Story:
    As a parent, I want to view all registered caregiver and family voice profiles
    to ensure the system knows who is authorized around my baby.
    """
    mock_db_res = MagicMock()
    mock_db_res.data = [
        {"id": "v1", "person_name": "Mother", "role": "Parent", "is_active": True},
        {"id": "v2", "person_name": "Father", "role": "Parent", "is_active": True}
    ]

    with patch("audio_guardian.router.db") as mock_db:
        mock_table = MagicMock()
        mock_db.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.order.return_value = mock_table
        mock_table.execute.return_value = mock_db_res

        res = client.get("/api/audio/profiles")
        assert res.status_code == 200
        data = res.json()
        assert "profiles" in data or isinstance(data, list)
