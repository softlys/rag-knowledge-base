import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.chunking import chunk_text
from app.main import app

client = TestClient(app)


def test_chunk_text_basic():
    text = "a" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    # соседние куски должны пересекаться
    assert chunks[0][-50:] in text


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.add_chunks", return_value=3)
def test_upload_txt(mock_add):
    file_content = b"This is a test document with some content to chunk."
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks_added"] == 3
    mock_add.assert_called_once()


def test_upload_unsupported_type_returns_422():
    response = client.post(
        "/upload",
        files={"file": ("image.png", io.BytesIO(b"not a real doc"), "image/png")},
    )
    assert response.status_code == 422


@patch("app.main.answer_question", return_value="Ответ на основе документа.")
@patch("app.main.query", return_value=[{"text": "фрагмент текста", "source": "notes.txt", "distance": 0.1}])
def test_ask_success(mock_query, mock_answer):
    response = client.post("/ask", json={"question": "О чём этот документ?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Ответ на основе документа."
    assert body["sources"][0]["source"] == "notes.txt"


def test_ask_too_short_question_returns_422():
    response = client.post("/ask", json={"question": "ok"})
    assert response.status_code == 422


@patch("app.llm_client.settings.anthropic_api_key", None)
@patch("app.main.query", return_value=[{"text": "фрагмент текста", "source": "notes.txt", "distance": 0.1}])
def test_ask_without_api_key_returns_extractive_fallback(mock_query):
    # Без ключа сервис не должен падать — должен вернуть честный fallback
    # с найденным фрагментом вместо ошибки.
    response = client.post("/ask", json={"question": "О чём этот документ?"})
    assert response.status_code == 200
    body = response.json()
    assert "Демо-режим" in body["answer"]
    assert "фрагмент текста" in body["answer"]
