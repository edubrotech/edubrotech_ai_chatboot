import os
import pytest
from fastapi.testclient import TestClient

# Ensure test mode environment variables before any app modules are imported
os.environ["ENVIRONMENT"] = "testing"
os.environ["DEFAULT_LLM_PROVIDER"] = "local"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["API_KEY"] = ""

from app.main import app


@pytest.fixture
def client():
    """Return FastAPI synchronous TestClient."""
    with TestClient(app) as test_client:
        yield test_client
