# DocuMind AI architecture and methodology

## Purpose

DocuMind AI is a local-first PDF document intelligence prototype. It combines
conventional document parsing and OCR with learned text embeddings and vector
retrieval. Optional hosted text and vision models can be enabled, but neither is
required for the core workflow.

## Components

| Component | Responsibility |
| --- | --- |
| `frontend/src/App.tsx` | React workspace for upload, library, page navigation, extracted text, search, Q&A, citations, visual analysis, and comparison. |
| `backend/main.py` | FastAPI application, CORS setup, and startup initialization. |
| `backend/app/api_routes.py` | REST endpoints, request validation, database dependency, and HTTP error translation. |
| `backend/app/services/document_service.py` | Upload lifecycle, persistence, hybrid retrieval, evidence-backed Q&A, comparison, rendering, and optional provider calls. |
| `backend/app/services/document_pipeline.py` | PDF text/table extraction, OCR fallback, chunking, and embedding generation. |
| `backend/app/models.py` | SQLAlchemy tables for documents, pages, chunks, and chat history. |
| SQLite | Document metadata, extracted page text, chunks, embeddings, checksums, and chat history. |
| Upload directory | Original PDFs, addressed by a stored database path and document ID. |

## Ingestion workflow

1. The API accepts a PDF upload, checks the extension and configured size cap,
   streams a SHA-256 digest, and reuses a successful record for an identical
   file. Failed records for the same content can be retried.
2. The PDF is parsed with PyMuPDF. Parse errors and page-limit violations are
   rejected and processing state is retained.
3. Each page first uses native PDF text extraction. A page with fewer than 25
   non-whitespace characters may use OCR if OCR is enabled.
4. OCR rasterizes only that page, applies grayscale and thresholding, and calls
   Tesseract. The page record stores native/OCR method, measured OCR confidence,
   and a visible error note if OCR failed or returned no text.
5. Where the installed PyMuPDF exposes table detection, extracted rows are
   appended as delimited text associated with the source page. This is best
   effort, not guaranteed table reconstruction.
6. Text is divided into configurable character windows with overlap. Each
   chunk remains linked to its document, filename via the parent, page, and
   chunk key.
7. The configured sentence-transformer encodes chunk texts once. Its vectors
   are normalized and stored with the model name. Exact duplicate file uploads
   reuse their existing indexed record.

## Retrieval methodology

The default embedding model is
`sentence-transformers/all-MiniLM-L6-v2`, using the configured
`EMBEDDING_MODEL_REVISION` (default `main`). It produces dense semantic vectors.
The model object is cached by configured model name in the backend process.
Persisted chunk vectors are reused when their model name and revision identifier match; missing,
malformed, or incompatible-model vectors are regenerated.

Two retrieval modes are available:

- `dense`: cosine similarity, calculated as inner product over L2-normalized
  vectors using FAISS `IndexFlatIP`.
- `hybrid`: dense rank fused with a local BM25 keyword rank using Reciprocal
  Rank Fusion: `1 / (60 + dense_rank) + 1 / (60 + bm25_rank)`.

RRF avoids adding raw cosine and BM25 values, which do not share a calibrated
scale. A dense similarity threshold removes low-similarity matches before they
are returned. The API exposes dense similarity, keyword score, ranks, and the
final ranking value separately. None is a probability or answer-confidence
estimate. For this small prototype, the FAISS structure is created in memory
per query from persisted vectors; embeddings are not recomputed on ordinary
queries.

## Answers and citation integrity

With `LLM_PROVIDER=disabled`, answers are explicitly extractive and return
retrieved passages. Empty/low-evidence retrieval results return an explicit
insufficient-evidence response.

With `LLM_PROVIDER=openai` or `openai-compatible` and an API key (or a
localhost-compatible endpoint), the backend sends a bounded evidence context
to a chat-completions-compatible endpoint.
Document text is labeled untrusted evidence and the instruction asks the model
not to follow instructions found inside it. Model-supplied `[E#]` references are
validated against the evidence entries actually sent. Backend code constructs
the returned document ID, filename, page, chunk ID, and passage from stored
records; invalid labels are removed. A provider failure or a response without
valid references is logged and falls back to a labeled extractive response.

This validates that citations identify real retrieved chunks. It does not
prove that every generated claim is entailed by its attached passage; semantic
claim-level grounding still needs qualitative evaluation.

## Comparison

Comparison searches each selected document independently. It reports evidence
and page-level sources per document. Exact shared sentences in retrieved
passages may be listed as common points. Without an LLM, per-document evidence
is labeled extractive and the system does not infer contradictions. With an
LLM, comparison synthesis uses the same bounded evidence and validated
evidence-label approach as Q&A.

## Visual analysis

The page viewer renders a selected PDF page through a backend image endpoint.
Visual analysis is a separate explicit request. With a compatible vision
provider and key configured (or a localhost-compatible endpoint), the backend
sends the selected rendered page and the question (plus bounded extracted-text
context) to that provider. Only the selected page is sent. With no provider configured, the API returns
`status=unavailable` and the UI says that no visual model was run.

OCR is text extraction and does not constitute chart, diagram, or layout
understanding. Hosted visual analysis sends the selected page outside the
local application; do not enable it for sensitive documents unless that data
transfer is acceptable.

## Configuration and API

Configuration is loaded from `.env`; see [`.env.example`](../.env.example).
Provider keys remain backend-only and are never returned by status routes.
FastAPI interactive API documentation is available at `/docs` while the server
is running.

Core endpoints:

- `POST /api/documents/upload`
- `GET /api/documents`, `GET /api/documents/{id}`, `DELETE /api/documents/{id}`
- `GET /api/documents/{id}/pages/{page}` and `/image`
- `POST /api/documents/{id}/pages/{page}/analyze-visual`
- `GET /api/search?q=...`
- `POST /api/chat/query`, `POST /api/chat/compare`
- `GET /api/chat/history`, `GET /api/health`, `GET /api/models/status`

## Security and reliability boundaries

The backend validates file extension, size, parseability, and page count, and
sanitizes storage filenames. SQLite parameterization is used through SQLAlchemy.
Provider requests have configured timeouts and secrets are kept on the backend.
CORS defaults to `*` for local development; set `CORS_ORIGINS` explicitly when
deploying.

This is not production-secure by default: it has no authentication, rate
limiting, tenant isolation, malware scanning, or encrypted storage. Avoid
internet exposure without those controls. Uploaded file content is untrusted.
LLM prompt-injection defenses are instructions and context separation, not a
formal security boundary.

## Known constraints

- Embedding weights may require a first-run download and local CPU memory.
- BM25 is a small in-process implementation and the retrieval corpus is loaded
  into memory for each query; this is not designed for large multi-tenant
  corpora.
- Chunk size uses characters, not tokenizer tokens.
- Table detection and OCR quality depend on source layout and local Tesseract.
- The synthetic evaluation set is too small to support general quality claims.
- There is no optional reranker in this version; adding one requires a measured
  quality/latency tradeoff.
