"""
UAT Fixtures and Setup
Configures TestClient for FastAPI application with mocked external dependencies (like remote Supabase or physical devices) where appropriate.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from main import app

@pytest.fixture(scope="session")
def client():
    """Provides a synchronous FastAPI TestClient to test user journeys."""
    with TestClient(app) as test_client:
        yield test_client
