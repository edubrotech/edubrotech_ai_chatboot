import io


def test_chat_endpoint(client):
    payload = {
        "message": "Hello, how do I rent an item?",
        "session_id": "api-test-session-1",
        "use_rag": False,
        "provider": "local",
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "api-test-session-1"
    assert "message" in data
    assert data["provider"] == "local"


def test_chat_history_endpoints(client):
    session_id = "api-history-session-2"
    # Send a message first
    client.post(
        "/api/v1/chat",
        json={"message": "What is the cancellation policy?", "session_id": session_id, "use_rag": False},
    )

    # Get history
    hist_resp = client.get(f"/api/v1/chat/history/{session_id}")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["session_id"] == session_id
    assert len(hist_data["messages"]) >= 2

    # Clear history
    del_resp = client.delete(f"/api/v1/chat/history/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["cleared"] is True


def test_documents_lifecycle_endpoints(client):
    # Upload a document
    file_content = b"RentCrew equipment insurance covers up to $10,000 for accidental damage."
    files = {"file": ("insurance_policy.txt", io.BytesIO(file_content), "text/plain")}

    upload_resp = client.post("/api/v1/documents/upload", files=files)
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    assert upload_data["status"] == "success"
    assert upload_data["chunk_count"] > 0
    doc_id = upload_data["document_id"]

    # List documents
    list_resp = client.get("/api/v1/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert any(d["document_id"] == doc_id for d in docs)

    # Delete document
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"
