import hashlib
import sys
from pathlib import Path

import numpy as np
import httpx
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.database import Base
from app import api_routes
from app.services.document_pipeline import DocumentPipeline
import main
from main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    settings.UPLOAD_DIR = str(tmp_path / "uploads")
    settings.STORAGE_DIR = str(tmp_path / "storage")
    settings.VECTOR_STORE_PATH = str(tmp_path / "vector_store")
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=test_engine)
    monkeypatch.setattr(api_routes, "session_local", sessionmaker(autoflush=False, bind=test_engine))
    monkeypatch.setattr(main, "init_db", lambda: None)

    def deterministic_embeddings(self, texts):
        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = set(text.casefold().split())
            for token in tokens:
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big") % vectors.shape[1]
                vectors[row, index] += 1
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors

    monkeypatch.setattr(DocumentPipeline, "embed_texts", deterministic_embeddings)
    with TestClient(app) as test_client:
        yield test_client
    test_engine.dispose()


def create_pdf(path: Path, text: str) -> None:
    pdf = canvas.Canvas(str(path))
    pdf.setTitle("DocuMind Test")
    pdf.drawString(72, 720, text)
    pdf.save()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_pdf_upload_and_extraction(client, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path, "DocuMind AI is a multimodal document intelligence platform.")

    with pdf_path.open("rb") as handle:
        response = client.post("/api/documents/upload", files={"file": ("sample.pdf", handle.read(), "application/pdf")})

    assert response.status_code == 200, response.text
    payload = response.json()
    doc_id = payload["document_id"]
    assert payload["status"] == "complete"

    page_response = client.get(f"/api/documents/{doc_id}/pages/1")
    assert page_response.status_code == 200
    assert "DocuMind" in page_response.json()["text"]


def test_retrieval_and_chat(client, tmp_path):
    pdf_path = tmp_path / "retrieval.pdf"
    create_pdf(pdf_path, "DocuMind extracts text, tables, and visual evidence from policies.")

    with pdf_path.open("rb") as handle:
        upload = client.post("/api/documents/upload", files={"file": ("retrieval.pdf", handle.read(), "application/pdf")})

    doc_id = upload.json()["document_id"]
    chat = client.post(
        "/api/chat/query",
        json={"question": "What does DocuMind extract from policies?", "document_ids": [doc_id], "top_k": 5},
    )
    assert chat.status_code == 200, chat.text
    body = chat.json()
    assert "DocuMind" in body["answer"] or body["passages"]
    assert body["citations"][0]["document_id"] == doc_id
    assert body["citations"][0]["chunk_id"]
    assert body["generation_mode"] == "extractive"
    assert body["passages"][0]["dense_score"] >= 0.15
    assert body["passages"][0]["keyword_score"] >= 0


def test_insufficient_evidence_is_explicit(client):
    response = client.post(
        "/api/chat/query",
        json={"question": "What is the answer?", "document_ids": ["unknown-document"], "top_k": 5},
    )
    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
    assert response.json()["generation_mode"] == "insufficient_evidence"
    assert response.json()["citations"] == []


def test_llm_citations_are_resolved_from_retrieved_chunks(client, tmp_path, monkeypatch):
    pdf_path = tmp_path / "llm.pdf"
    create_pdf(pdf_path, "DocuMind extracts policy data using semantic retrieval.")
    with pdf_path.open("rb") as handle:
        uploaded = client.post("/api/documents/upload", files={"file": ("llm.pdf", handle.read(), "application/pdf")})
    document_id = uploaded.json()["document_id"]
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-only-key")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "It extracts policy data [E1]. [E999]"}}]}

    monkeypatch.setattr("app.services.document_service.httpx.post", lambda *args, **kwargs: FakeResponse())
    response = client.post(
        "/api/chat/query",
        json={"question": "What does DocuMind extract?", "document_ids": [document_id], "top_k": 5},
    )
    assert response.status_code == 200
    answer = response.json()
    assert answer["generation_mode"] == "llm"
    assert "[E999]" not in answer["answer"]
    assert answer["citations"][0]["document_id"] == document_id
    assert answer["citations"][0]["chunk_id"] == answer["passages"][0]["chunk_id"]


def test_provider_failure_uses_labeled_extractive_fallback(client, tmp_path, monkeypatch):
    pdf_path = tmp_path / "fallback.pdf"
    create_pdf(pdf_path, "Fallback answer remains grounded in retrieved source evidence.")
    with pdf_path.open("rb") as handle:
        uploaded = client.post("/api/documents/upload", files={"file": ("fallback.pdf", handle.read(), "application/pdf")})
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-only-key")

    def provider_error(*args, **kwargs):
        raise httpx.ConnectError("simulated provider outage")

    monkeypatch.setattr("app.services.document_service.httpx.post", provider_error)
    response = client.post(
        "/api/chat/query",
        json={"question": "What fallback answer remains grounded in evidence?", "document_ids": [uploaded.json()["document_id"]]},
    )
    assert response.status_code == 200
    assert response.json()["generation_mode"] == "extractive_fallback"
    assert response.json()["provider_error"]


def test_llm_insufficient_evidence_is_reported_separately(client, tmp_path, monkeypatch):
    pdf_path = tmp_path / "abstain.pdf"
    create_pdf(pdf_path, "The document mentions a process but gives no requested measurement.")
    with pdf_path.open("rb") as handle:
        uploaded = client.post("/api/documents/upload", files={"file": ("abstain.pdf", handle.read(), "application/pdf")})
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-only-key")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "There is insufficient evidence to answer this question."}}]}

    monkeypatch.setattr("app.services.document_service.httpx.post", lambda *args, **kwargs: FakeResponse())
    response = client.post(
        "/api/chat/query",
        json={"question": "What specific measurement is reported in the process?", "document_ids": [uploaded.json()["document_id"]]},
    )
    assert response.status_code == 200
    assert response.json()["generation_mode"] == "insufficient_evidence"
    assert response.json()["insufficient_evidence"] is True


def test_comparison_keeps_per_document_evidence(client, tmp_path):
    document_ids = []
    for filename, sentence in (
        ("one.pdf", "The plan cuts energy use by ten percent before 2026."),
        ("two.pdf", "The plan cuts energy use by ten percent before 2026."),
    ):
        pdf_path = tmp_path / filename
        create_pdf(pdf_path, sentence)
        with pdf_path.open("rb") as handle:
            upload = client.post("/api/documents/upload", files={"file": (filename, handle.read(), "application/pdf")})
        document_ids.append(upload.json()["document_id"])
    response = client.post(
        "/api/chat/compare",
        json={"question": "What is the energy target?", "document_ids": document_ids, "top_k": 3},
    )
    assert response.status_code == 200
    comparison = response.json()
    assert comparison["generation_mode"] == "extractive"
    assert {item["document_id"] for item in comparison["documents"]} == set(document_ids)
    assert all(item["has_evidence"] and item["passages"] for item in comparison["documents"])


def test_invalid_upload(client):
    response = client.post("/api/documents/upload", files={"file": ("bad.txt", b"not a pdf", "text/plain")})
    assert response.status_code == 400


def test_malformed_pdf_rejected(client):
    response = client.post("/api/documents/upload", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 400
    assert "valid, readable PDF" in response.json()["detail"]


def test_duplicate_upload_reuses_document(client, tmp_path):
    pdf_path = tmp_path / "duplicate.pdf"
    create_pdf(pdf_path, "This identical PDF must only be indexed once.")
    content = pdf_path.read_bytes()

    first = client.post("/api/documents/upload", files={"file": ("first.pdf", content, "application/pdf")})
    second = client.post("/api/documents/upload", files={"file": ("copy.pdf", content, "application/pdf")})

    assert first.status_code == second.status_code == 200
    assert first.json()["document_id"] == second.json()["document_id"]
    assert client.get("/api/documents").json()["count"] == 1


def test_page_image_and_visual_analysis_disabled(client, tmp_path):
    pdf_path = tmp_path / "visual.pdf"
    create_pdf(pdf_path, "Text available while optional visual analysis is disabled.")
    with pdf_path.open("rb") as handle:
        uploaded = client.post("/api/documents/upload", files={"file": ("visual.pdf", handle.read(), "application/pdf")})
    document_id = uploaded.json()["document_id"]
    image = client.get(f"/api/documents/{document_id}/pages/1/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    result = client.post(
        f"/api/documents/{document_id}/pages/1/analyze-visual",
        json={"question": "What is shown on this page?"},
    )
    assert result.status_code == 200
    assert result.json()["status"] == "unavailable"
    assert "disabled" in result.json()["message"]


def test_document_delete(client, tmp_path):
    pdf_path = tmp_path / "delete.pdf"
    create_pdf(pdf_path, "This document will be deleted.")
    with pdf_path.open("rb") as handle:
        upload = client.post("/api/documents/upload", files={"file": ("delete.pdf", handle.read(), "application/pdf")})
    doc_id = upload.json()["document_id"]
    delete_response = client.delete(f"/api/documents/{doc_id}")
    assert delete_response.status_code == 200
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
