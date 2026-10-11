import logging
import re
import threading
import time
import uuid
from pathlib import Path

import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer

from ..config import settings

logger = logging.getLogger(__name__)
_EMBEDDERS: dict[tuple[str, str], SentenceTransformer] = {}
_EMBEDDER_LOCK = threading.Lock()


class DocumentPipeline:
    def ensure_embedder(self) -> SentenceTransformer:
        model_name = settings.EMBEDDING_MODEL
        revision = settings.EMBEDDING_MODEL_REVISION
        cache_key = (model_name, revision)
        with _EMBEDDER_LOCK:
            if cache_key not in _EMBEDDERS:
                _EMBEDDERS[cache_key] = SentenceTransformer(model_name, revision=revision)
            return _EMBEDDERS[cache_key]

    def extract_pdf_text(self, file_path: str) -> tuple[list[str], int, list[str], list[float | None], list[str | None]]:
        started_at = time.perf_counter()
        page_texts: list[str] = []
        extraction_methods: list[str] = []
        confidences: list[float | None] = []
        errors: list[str | None] = []

        try:
            pdf = fitz.open(file_path)
        except (fitz.FileDataError, RuntimeError) as exc:
            raise ValueError("The uploaded file is not a valid, readable PDF.") from exc
        with pdf:
            total_pages = pdf.page_count
            if total_pages == 0:
                raise ValueError("The PDF does not contain any pages.")
            if total_pages > settings.MAX_PDF_PAGES:
                raise ValueError(f"Document exceeds {settings.MAX_PDF_PAGES} pages.")
            for page_number in range(total_pages):
                page = pdf[page_number]
                text = page.get_text("text", sort=True).strip()
                table_text = self._extract_tables(page)
                if table_text:
                    text = "\n\n".join(part for part in (text, table_text) if part)
                error: str | None = None
                confidence: float | None = None
                if len(re.sub(r"\s+", "", text)) < 25 and settings.OCR_ENABLED:
                    ocr_text, confidence, error = self._ocr_page(pdf, page_number)
                    if ocr_text:
                        text = ocr_text
                        extraction_methods.append("ocr")
                    else:
                        extraction_methods.append("native")
                else:
                    extraction_methods.append("native")
                page_texts.append(text)
                confidences.append(confidence)
                errors.append(error)
        logger.info(
            "PDF extraction completed: pages=%s ocr_pages=%s duration_ms=%.1f",
            total_pages,
            extraction_methods.count("ocr"),
            (time.perf_counter() - started_at) * 1000,
        )
        return page_texts, total_pages, extraction_methods, confidences, errors

    @staticmethod
    def _extract_tables(page: fitz.Page) -> str:
        find_tables = getattr(page, "find_tables", None)
        if not callable(find_tables):
            return ""
        try:
            tables = find_tables().tables
            rendered = []
            for table_number, table in enumerate(tables, start=1):
                rows = table.extract()
                if rows:
                    rendered.append(
                        f"[Extracted table {table_number}]\n"
                        + "\n".join(" | ".join(str(cell or "").strip() for cell in row) for row in rows)
                    )
            return "\n\n".join(rendered)
        except Exception:
            logger.warning("Table extraction failed on page %s", page.number + 1, exc_info=True)
            return ""

    def _ocr_page(self, pdf: fitz.Document, page_number: int) -> tuple[str, float | None, str | None]:
        if not settings.OCR_ENABLED:
            return "", None, None
        started_at = time.perf_counter()
        try:
            page = pdf[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = ImageOps.grayscale(img)
            img = img.point(lambda p: 255 if p > 180 else 0)
            if settings.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            text = pytesseract.image_to_string(img)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confs = [float(value) for value in data.get("conf", []) if value not in {"-1", ""}]
            confidence = sum(confs) / len(confs) if confs else None
            logger.info(
                "OCR completed: page=%s confidence=%s duration_ms=%.1f",
                page_number + 1,
                confidence,
                (time.perf_counter() - started_at) * 1000,
            )
            return text.strip(), confidence, None if text.strip() else "OCR completed but returned no text."
        except Exception as exc:
            logger.warning("OCR failed on page %s", page_number + 1, exc_info=True)
            logger.info("OCR duration_ms=%.1f", (time.perf_counter() - started_at) * 1000)
            return "", None, f"OCR failed: {exc}"

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
        if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
        normalized = "\n\n".join(p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip())
        if not normalized:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary
            chunk = normalized[start:end].strip()
            if chunk and (not chunks or chunk != chunks[-1]):
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        started_at = time.perf_counter()
        embedder = self.ensure_embedder()
        vectors = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        output = np.asarray(vectors, dtype="float32")
        logger.info("Embedding completed: texts=%s dimension=%s duration_ms=%.1f", len(texts), output.shape[-1], (time.perf_counter() - started_at) * 1000)
        return output

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename.strip())
        return cleaned or f"document_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def safe_upload_path(base_dir: str, filename: str) -> str:
        target = Path(base_dir) / DocumentPipeline.sanitize_filename(filename)
        counter = 1
        while target.exists():
            stem = target.stem
            suffix = target.suffix
            target = Path(base_dir) / f"{stem}_{counter}{suffix}"
            counter += 1
        return str(target)
