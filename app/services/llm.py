import logging
import time

from openai import OpenAI

from app.config import settings
from app.models import RetrievedChunk

logger = logging.getLogger(__name__)
_client = OpenAI(api_key=settings.openai_api_key)

_SYSTEM_PROMPT = """You are a precise document assistant.
Answer ONLY based on the provided context.
If the context does not contain enough information, say:
"I could not find a confident answer in the provided documents."
Do not hallucinate or use external knowledge."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} | {chunk.filename} | chunk {chunk.chunk_index}]\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, float]:
    """
    Generate an answer from retrieved chunks.
    Returns (answer_text, latency_ms).
    """
    context = build_context(chunks)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    start = time.perf_counter()
    response = _client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    answer = response.choices[0].message.content.strip()
    logger.info(
        "LLM answer generated | latency=%.1fms | tokens=%d",
        latency_ms,
        response.usage.total_tokens,
    )
    return answer, latency_ms