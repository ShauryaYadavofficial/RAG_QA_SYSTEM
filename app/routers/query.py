import logging

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.models import QueryRequest, QueryResponse
from app.services.embedder import embed_query
from app.services.llm import generate_answer
from app.services.vector_store import vector_store

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit_query)
async def query_documents(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    """
    Ask a question over your uploaded documents.

    - Embeds the question and runs cosine similarity search over FAISS.
    - Passes top-k chunks as context to the LLM.
    - Returns the answer along with source chunks and latency metrics.
    """
    if vector_store.total_vectors == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents have been ingested yet. Please upload documents first.",
        )

    # 1. Embed query
    query_vec = embed_query(body.question)

    # 2. Retrieve
    chunks = vector_store.search(
        query_embedding=query_vec,
        top_k=body.top_k,
        document_ids=body.document_ids,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found. "
                   "Try rephrasing or uploading more documents.",
        )

    # Log similarity scores for observability
    scores = [c.similarity_score for c in chunks]
    logger.info(
        "query='%s' | top_score=%.4f | avg_score=%.4f | chunks=%d",
        body.question[:60],
        max(scores),
        sum(scores) / len(scores),
        len(chunks),
    )

    # 3. Generate answer
    answer, latency_ms = generate_answer(body.question, chunks)

    return QueryResponse(
        question=body.question,
        answer=answer,
        retrieved_chunks=chunks,
        latency_ms=round(latency_ms, 2),
        model_used=settings.openai_model,
    )