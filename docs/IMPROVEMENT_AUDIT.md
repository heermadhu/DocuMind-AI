# DocuMind AI improvement audit

## Scope and method

This audit follows the checked-in request path from the React workspace through
the FastAPI routes, service layer, PDF pipeline, SQLite models, and the existing
API tests. Findings below describe the implementation as inspected before the
upgrade; suggested changes are not described as already working.

## Confirmed implemented behavior

- The React app uploads PDF files, lists and deletes documents, displays
  extracted text for page 1, submits question and search requests, and requests
  a two-document comparison.
- FastAPI validates request bodies with Pydantic schemas and uses a SQLAlchemy
  session dependency. Startup creates the upload/storage directories and
  database tables.
- Uploads are restricted by filename extension and configured byte limit, then
  saved under a sanitized unique filename.
- PyMuPDF extracts page text. Pages with fewer than 25 non-whitespace
  characters use a Tesseract OCR fallback when enabled.
- Extracted pages and paragraph-oriented text chunks are persisted in SQLite.
- Search embeds the query and all eligible chunks and ranks them with FAISS
  inner-product similarity after L2 normalization.
- Question answers expose selected passages and document/page citations;
  unsupported queries with no hits return an insufficient-evidence message.
- The test suite covers health, native PDF upload/extraction, a retrieval/chat
  request, wrong-extension uploads, and deletion.

## Confirmed bugs and gaps

| Area | Verified finding | Impact |
| --- | --- | --- |
| Embedding lifecycle | A new `DocumentService` and `DocumentPipeline` are created for each dependency resolution. The transformer model is therefore not reliably reused between API requests. Search embeds every stored chunk again and constructs a new in-memory FAISS index on every query. | High latency, repeated model work, and no persistent vector index. |
| Indexing | Upload creates chunk records but does not embed or index those chunks. `embedding_key` is only a string identifier. | "Indexed" content is not actually indexed until a search request recomputes vectors. |
| OCR metadata | OCR confidence is computed by the pipeline but discarded; the service stores a hardcoded confidence of 95 for every OCR page. | Reported page confidence is misleading. |
| Error handling | OCR catches every exception and silently converts it into empty text. PDF processing exceptions are converted to `ValueError`, which the upload endpoint reports as client error 400, even when the cause may be a server/provider failure. | Operational problems are obscured and clients receive inaccurate error categories. |
| PDF validation | The upload route checks extension/size but not PDF signature or parseability before persistence. Page limits are checked only after page extraction. | Invalid files and over-limit PDFs consume processing work; malformed input has weak error reporting. |
| Chunking | Chunk size and overlap are hard-coded defaults, and chunk overlap only applies to a single oversized paragraph. | Chunk boundaries and retrieval context cannot be tuned; adjacent paragraphs may lose boundary context. |
| Grounded answers | The answer is concatenated from the first two retrieved chunks; there is no generative model call, relevance threshold, duplicate-evidence removal, or claim-level citation validation. | The feature is extractive, not LLM question answering; weak matches can be presented as answers. |
| Citations | Citation entries contain filename and page only, without document ID, chunk ID, or exact cited passage linkage. | Consumers cannot robustly verify a citation against a stored evidence record. |
| Comparison | Each document is searched separately, but the returned summary is a fixed generic sentence. It does not synthesize commonalities or differences. | The response is not a substantive document comparison. |
| Vision and tables | Vision settings are reported by status endpoints but no vision model receives rendered page images. No table extraction or layout-aware representation is implemented. | The current app does not perform multimodal visual understanding or structured table understanding. |
| Frontend metrics | The "indexed chunks" metric is estimated as `page_count * 2`; the OCR metric counts documents whose overall method is OCR rather than OCR pages. | Dashboard values do not describe actual backend measurements. |
| Search controls | The frontend global search does not pass the selected document ID, so it searches all documents. | Results can include documents outside the current selection. |
| CORS | The app allows every origin, method, and header and enables credentials. | Overly permissive defaults are unsuitable for deployment. |
| Deletion | The source file is deleted before related database rows are committed. | A database failure can leave metadata pointing to a missing file. |

## Possible improvements (not confirmed defects)

- Add a BM25 keyword retriever and evaluate it alongside dense retrieval rather
  than assuming a hybrid approach is always better.
- Add an optional cross-encoder reranker only if measured gains justify its
  extra model download and inference cost.
- Add optional OpenAI-compatible text and vision providers while keeping
  offline behavior available and clearly labeled.
- Build a small manually labeled retrieval benchmark and report computed
  Recall@k and MRR with dataset limitations.
- Add rendered page previews, page navigation, expandable diagnostics, and
  richer evidence attribution to the existing frontend.
- Add local API rate limits, authentication, stronger deployment controls, and
  a broader threat model before exposing the application beyond a trusted
  development machine.

## Testing and security notes

The current tests are useful smoke-level API tests, but they do not establish
retrieval quality, OCR correctness, citation integrity, persistence across
restart, comparison quality, or provider behavior. The application treats
document content as text evidence but does not currently pass it to an LLM.
Uploaded files are untrusted input; production deployment would still require
authentication, rate limits, secure storage, and a focused security review.

## Upgrade tracking

This document records the pre-change findings. Subsequent implementation and
validation status is described in the architecture and evaluation documents
and in the final task report; the audit itself is not evidence that a proposed
improvement has been implemented.
