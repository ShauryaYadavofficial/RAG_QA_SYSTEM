from dataclasses import dataclass

from app.config import settings


@dataclass
class Chunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(text: str, chunk_size: int | None = None,
               overlap: int | None = None) -> list[Chunk]:
    """
    Sliding-window character-level chunker with sentence-boundary awareness.

    Strategy
    --------
    - Primary split on paragraph breaks (\\n\\n) to respect logical units.
    - If a paragraph is still larger than `chunk_size`, it is split at the
      nearest sentence boundary ('. ', '? ', '! ') before the limit.
    - A sliding overlap of `chunk_overlap` characters is preserved between
      consecutive chunks so that answers spanning a boundary are not lost.

    Chosen chunk size: 512 characters
    - Fits comfortably within the all-MiniLM-L6-v2 128-token window
      (512 chars ≈ 100-130 tokens for English text).
    - Small enough to be precise; large enough to hold a complete thought.
    - See DESIGN_DECISIONS.md for full reasoning.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            raw_chunks.append(para)
        else:
            raw_chunks.extend(_split_paragraph(para, chunk_size))

    # Merge tiny fragments with the previous chunk
    merged = _merge_small_chunks(raw_chunks, chunk_size)

    # Apply sliding overlap
    return _apply_overlap(merged, overlap)


def _split_paragraph(para: str, chunk_size: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries."""
    sentence_endings = (". ", "? ", "! ", ".\n", "?\n", "!\n")
    chunks: list[str] = []
    start = 0

    while start < len(para):
        end = start + chunk_size
        if end >= len(para):
            chunks.append(para[start:].strip())
            break

        # Walk backwards to find a sentence boundary
        split_at = end
        for i in range(end, max(start, end - 80), -1):
            if para[i : i + 2] in sentence_endings:
                split_at = i + 1
                break

        chunks.append(para[start:split_at].strip())
        start = split_at

    return [c for c in chunks if c]


def _merge_small_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    """Merge chunks shorter than 20% of chunk_size into the previous one."""
    min_size = int(chunk_size * 0.2)
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_size:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged


def _apply_overlap(chunks: list[str], overlap: int) -> list[Chunk]:
    result: list[Chunk] = []
    running_offset = 0

    for i, chunk in enumerate(chunks):
        # For display purposes, track approximate character offsets
        start = max(0, running_offset - overlap) if i > 0 else 0

        if i > 0 and overlap > 0:
            # Prepend a tail from the previous chunk
            prev_tail = chunks[i - 1][-overlap:]
            text = prev_tail + " " + chunk
        else:
            text = chunk

        result.append(
            Chunk(
                text=text.strip(),
                chunk_index=i,
                start_char=start,
                end_char=running_offset + len(chunk),
            )
        )
        running_offset += len(chunk)

    return result