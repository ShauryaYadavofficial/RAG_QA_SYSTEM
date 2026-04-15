import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.models import (
    DocumentRecord,
    DocumentStatus,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.services.ingestion import (
    get_document,
    ingest_document,
    list_documents,
    register_document,
)
from app.utils.file_parser import SUPPORTED_EXTENSIONS

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


@router.post("/documents", response_model=DocumentUploadResponse, status_code=202)
@limiter.limit(settings.rate_limit_upload)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """
    Upload a document for ingestion.
    Accepts: PDF, TXT, DOCX, MD.
    Processing happens asynchronously; poll /documents/{id} for status.
    """
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"File type '{suffix}' not supported. "
                   f"Accepted: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Validate size
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_file_size_mb} MB limit.",
        )

    # Persist to disk
    document_id = str(uuid.uuid4())
    safe_name = f"{document_id}{suffix}"
    dest = settings.upload_dir / safe_name
    dest.write_bytes(content)

    # Register & kick off background ingestion
    record = DocumentRecord(
        document_id=document_id,
        filename=file.filename or safe_name,
        file_path=str(dest),
    )
    register_document(record)
    background_tasks.add_task(ingest_document, document_id)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=record.filename,
        status=DocumentStatus.PENDING,
        message="Document accepted. Ingestion started in the background.",
    )


@router.get("/documents", response_model=list[DocumentStatusResponse])
async def get_all_documents(request: Request) -> list[DocumentStatusResponse]:
    """List all uploaded documents and their statuses."""
    return [
        DocumentStatusResponse(
            document_id=d.document_id,
            filename=d.filename,
            status=d.status,
            chunk_count=d.chunk_count,
            error=d.error,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in list_documents()
    ]


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    request: Request, document_id: str
) -> DocumentStatusResponse:
    """Check the ingestion status of a specific document."""
    record = get_document(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentStatusResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status,
        chunk_count=record.chunk_count,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(request: Request, document_id: str) -> None:
    """Delete a document and its associated vectors."""
    record = get_document(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")

    from app.services.vector_store import vector_store
    vector_store.delete_document(document_id)

    file_path = Path(record.file_path)
    if file_path.exists():
        file_path.unlink()