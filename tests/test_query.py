from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import RetrievedChunk

client = TestClient(app)

MOCK_CHUNK = RetrievedChunk(
    document_id="doc-1",
    filename="test.txt",
    chunk_index=0,
    text="The capital of France is Paris.",
    similarity_score=0.92,
)


def test_query_no_documents():
    with patch(
        "app.routers.query.vector_store.total_vectors", new_callable=lambda: property(lambda self: 0)
    ):
        response = client.post(
            "/api/v1/query",
            json={"question": "What is the capital of France?"},
        )
    assert response.status_code == 422


def test_query_success():
    with (
        patch("app.routers.query.vector_store.total_vectors", 10),
        patch(
            "app.routers.query.embed_query",
            return_value=np.zeros((1, 384), dtype=np.float32),
        ),
        patch(
            "app.routers.query.vector_store.search",
            return_value=[MOCK_CHUNK],
        ),
        patch(
            "app.routers.query.generate_answer",
            return_value=("Paris", 123.4),
        ),
    ):
        response = client.post(
            "/api/v1/query",
            json={"question": "What is the capital of France?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris"
    assert body["latency_ms"] == 123.4
    assert len(body["retrieved_chunks"]) == 1


def test_query_too_short():
    response = client.post("/api/v1/query", json={"question": "Hi"})
    assert response.status_code == 422