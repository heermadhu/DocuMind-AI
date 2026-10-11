import json
from collections.abc import Generator

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from .config import provider_url_is_local, settings
from .database import session_local
from .models import ChunkRecord, PageRecord
from .schemas import ChatAnswer, ChatQuery, CompareQuery, HealthResponse, VisualQuery
from .services.document_service import DocumentService

router = APIRouter(prefix="/api")


def get_db() -> Generator[Session, None, None]:
    db = session_local()
    try:
        yield db
    finally:
        db.close()


def get_service() -> DocumentService:
    return DocumentService()


@router.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {
        "status": "ok",
        "backend": settings.APP_NAME,
        "model_status": {
            "embedding": settings.EMBEDDING_MODEL,
            "embedding_revision": settings.EMBEDDING_MODEL_REVISION,
            "ocr": "enabled" if settings.OCR_ENABLED else "disabled",
            "llm": settings.LLM_PROVIDER,
            "vision": settings.VISION_PROVIDER,
            "retrieval": settings.RETRIEVAL_MODE,
        },
    }


@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db), service: DocumentService = Depends(get_service)) -> dict:
    try:
        document = service.upload_document(db, file)
        return {"document_id": document.document_id, "filename": document.filename, "status": document.processing_status}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents")
def list_documents(db: Session = Depends(get_db), service: DocumentService = Depends(get_service), limit: int = Query(20, ge=1, le=50), offset: int = Query(0, ge=0)) -> dict:
    docs = service.list_documents(db, limit=limit, offset=offset)
    chunk_counts = dict(db.query(ChunkRecord.document_id, func.count(ChunkRecord.id)).group_by(ChunkRecord.document_id).all())
    ocr_counts = dict(
        db.query(PageRecord.document_id, func.count(PageRecord.id))
        .filter(PageRecord.extraction_method == "ocr")
        .group_by(PageRecord.document_id)
        .all()
    )
    return {"items": [{
        "id": doc.document_id,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "processing_status": doc.processing_status,
        "extraction_method": doc.extraction_method,
        "created_at": doc.created_at.isoformat(),
        "file_size": doc.file_size,
        "chunk_count": chunk_counts.get(doc.id, 0),
        "ocr_page_count": ocr_counts.get(doc.id, 0),
        "indexing_status": (
            "failed" if doc.processing_status == "failed"
            else "processing" if doc.processing_status == "processing"
            else "indexed" if chunk_counts.get(doc.id, 0)
            else "empty"
        ),
    } for doc in docs], "count": len(docs)}


@router.get("/documents/{document_id}")
def document_detail(document_id: str, db: Session = Depends(get_db), service: DocumentService = Depends(get_service)) -> dict:
    document = service.get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "id": document.document_id,
        "filename": document.filename,
        "page_count": document.page_count,
        "processing_status": document.processing_status,
        "extraction_method": document.extraction_method,
        "file_size": document.file_size,
        "storage_path": document.storage_path,
        "created_at": document.created_at.isoformat(),
    }


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db), service: DocumentService = Depends(get_service)) -> dict:
    try:
        service.delete_document(db, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": True, "document_id": document_id}


@router.get("/documents/{document_id}/pages/{page_number}")
def get_page(document_id: str, page_number: int, db: Session = Depends(get_db), service: DocumentService = Depends(get_service)) -> dict:
    try:
        page = service.get_page(db, document_id, page_number)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if page is None:
        raise HTTPException(status_code=404, detail="Page not found.")
    return {
        "document_id": document_id,
        "page_number": page.page_number,
        "text": page.text,
        "extraction_method": page.extraction_method,
        "ocr_confidence": page.ocr_confidence,
        "extraction_error": page.extraction_error,
        "processing_status": page.processing_status,
    }


@router.get("/documents/{document_id}/pages/{page_number}/image")
def get_page_image(
    document_id: str,
    page_number: int,
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_service),
) -> Response:
    try:
        image = service.rendered_page(db, document_id, page_number)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=image, media_type="image/png")


@router.post("/documents/{document_id}/pages/{page_number}/analyze-visual")
def analyze_visual(
    document_id: str,
    page_number: int,
    payload: VisualQuery,
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_service),
) -> dict:
    try:
        return service.analyze_visual(db, document_id, page_number, payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/documents/{document_id}/summarize")
def summarize_document(document_id: str, db: Session = Depends(get_db), service: DocumentService = Depends(get_service)) -> dict:
    try:
        summary = service.summarize_document(db, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"document_id": document_id, "summary": summary}


@router.post("/chat/query", response_model=ChatAnswer)
def chat_query(payload: ChatQuery, db: Session = Depends(get_db), service: DocumentService = Depends(get_service)) -> dict:
    answer = service.answer_question(db, payload.question, payload.document_ids, payload.top_k)
    service.add_chat_history(db, payload.question, answer["answer"], {"document_ids": payload.document_ids or [], "top_k": payload.top_k})
    return answer


@router.post("/chat/compare")
def compare_documents(payload: CompareQuery, db: Session = Depends(get_db), service: DocumentService = Depends(get_service)) -> dict:
    comparison = service.compare_documents(db, payload.question, payload.document_ids, payload.top_k)
    service.add_chat_history(db, payload.question, comparison["summary"], {"document_ids": payload.document_ids, "mode": "compare"})
    return comparison


@router.get("/chat/history")
def chat_history(db: Session = Depends(get_db), service: DocumentService = Depends(get_service), limit: int = Query(20, ge=1, le=50)) -> dict:
    rows = service.chat_history(db, limit=limit)
    return {"items": [{"question": row.question, "answer": row.answer, "created_at": row.created_at.isoformat(), "metadata": json.loads(row.payload or "{}") } for row in rows]}


@router.get("/search")
def search_documents(q: str = Query(..., min_length=1), document_ids: list[str] | None = Query(None), top_k: int = Query(5, ge=1, le=10), db: Session = Depends(get_db), service: DocumentService = Depends(get_service)) -> dict:
    hits = service.search_documents(db, q, document_ids=document_ids, top_k=top_k)
    return {"query": q, "results": hits}


@router.get("/models/status")
def model_status() -> dict:
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_revision": settings.EMBEDDING_MODEL_REVISION,
        "ocr_enabled": settings.OCR_ENABLED,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_status": (
            "disabled" if settings.LLM_PROVIDER == "disabled"
            else "configured" if settings.LLM_API_KEY or provider_url_is_local(settings.LLM_API_BASE_URL)
            else "missing_key"
        ),
        "vision_provider": settings.VISION_PROVIDER,
        "vision_status": (
            "disabled" if settings.VISION_PROVIDER == "disabled"
            else "configured" if settings.VISION_API_KEY or provider_url_is_local(settings.VISION_API_BASE_URL)
            else "missing_key"
        ),
        "retrieval_mode": settings.RETRIEVAL_MODE,
        "status": "available" if settings.LLM_PROVIDER == "disabled" else (
        "configured" if settings.LLM_API_KEY or provider_url_is_local(settings.LLM_API_BASE_URL) else "missing_key"
        ),
    }


@router.get("/settings")
def settings_info() -> dict:
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "ocr_enabled": settings.OCR_ENABLED,
        "llm_provider": settings.LLM_PROVIDER,
        "vision_provider": settings.VISION_PROVIDER,
        "retrieval_mode": settings.RETRIEVAL_MODE,
        "max_upload_size": settings.MAX_UPLOAD_SIZE,
        "max_pdf_pages": settings.MAX_PDF_PAGES,
        "upload_dir": settings.UPLOAD_DIR,
    }
