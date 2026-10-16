def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AI Chatbot Service"
    assert data["status"] == "running"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "dependencies" in data
    assert "redis" in data["dependencies"]
    assert "qdrant" in data["dependencies"]


def test_liveness_endpoint(client):
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
