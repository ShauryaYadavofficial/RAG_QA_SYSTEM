import logging
import traceback
from datetime import datetime
from pathlib import Path

from app.models import DocumentRecord, DocumentStatus
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts
from app.services.vector_store import vector_store
from app.utils.file_parser import parse_file

logger = logging.getLogger(__name__)

# In-memory store; replace with Redis / SQL in production
_document_registry: dict[str, DocumentRecord] = {}


def register_document(record: DocumentRecord) -> None:
    _document_registry[record.document_id] = record


def get_document(document_id: str) -> DocumentRecord | None:
    return _document_registry.get(document_id)


def list_documents() -> list[DocumentRecord]:
    return list(_document_registry.values())


def ingest_document(document_id: str) -> None:
    """
    Background task: parse → chunk → embed → store.
    Updates the document record status throughout.
    """
    record = _document_registry.get(document_id)
    if not record:
        logger.error("ingest_document: unknown id %s", document_id)
        return

    _update_status(record, DocumentStatus.PROCESSING)
    logger.info("Ingesting document %s (%s)", document_id, record.filename)

    try:
        # 1. Parse raw text
        raw_text = parse_file(Path(record.file_path))
        if not raw_text.strip():
            raise ValueError("Document appears to be empty after parsing.")

        # 2. Chunk
        chunks = chunk_text(raw_text)
        logger.info("%d chunks created for %s", len(chunks), document_id)

        # 3. Embed
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        # 4. Build metadata
        metadata_list = [
            {
                "document_id": document_id,
                "filename": record.filename,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]

        # 5. Store in FAISS
        vector_store.add_chunks(embeddings, metadata_list)

        record.chunk_count = len(chunks)
        _update_status(record, DocumentStatus.READY)
        logger.info("Document %s ingested successfully.", document_id)

    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", document_id, exc)
        record.error = str(exc)
        _update_status(record, DocumentStatus.FAILED)


def _update_status(record: DocumentRecord, status: DocumentStatus) -> None:
    record.status = status
    record.updated_at = datetime.utcnow()