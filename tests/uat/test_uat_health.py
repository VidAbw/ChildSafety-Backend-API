"""
UAT Scenario 1: Health & System Readiness
Verifies the backend services are operational and responsive for the Guardian dashboard.
"""
def test_system_health_readiness(client):
    """
    User Story:
    As a system administrator or parent opening the app,
    I want to verify that the backend API is online and healthy,
    so that I know my child monitoring system is active.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online"
    assert "Child Safety Guardian API is running" in data.get("message", "")
