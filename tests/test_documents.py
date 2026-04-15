import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_txt_file(content: str = "Hello world. This is a test document."):
    return ("test.txt", io.BytesIO(content.encode()), "text/plain")


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload_txt_document():
    with patch("app.services.ingestion.ingest_document"):
        name, data, mime = _make_txt_file()
        response = client.post(
            "/api/v1/documents",
            files={"file": (name, data, mime)},
        )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert "document_id" in body


def test_upload_unsupported_format():
    response = client.post(
        "/api/v1/documents",
        files={"file": ("malware.exe", io.BytesIO(b"bad"), "application/octet-stream")},
    )
    assert response.status_code == 415


def test_get_nonexistent_document():
    response = client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404