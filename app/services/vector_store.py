import json
import pickle
from pathlib import Path
from threading import Lock
from typing import Optional

import faiss
import numpy as np

from app.config import settings
from app.models import RetrievedChunk


_METADATA_FILE = settings.faiss_index_dir / "metadata.pkl"
_INDEX_FILE = settings.faiss_index_dir / "index.faiss"


class VectorStore:
    """
    Thread-safe FAISS IndexFlatIP (inner product = cosine when normalised).

    Metadata is kept in a parallel Python list so we avoid separate DBs.
    In production, swap for PostgreSQL + pgvector or Pinecone.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._index: faiss.IndexFlatIP
        self._metadata: list[dict]   # parallel to FAISS vectors
        self._load_or_create()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        if _INDEX_FILE.exists() and _METADATA_FILE.exists():
            self._index = faiss.read_index(str(_INDEX_FILE))
            with open(_METADATA_FILE, "rb") as f:
                self._metadata = pickle.load(f)
        else:
            self._index = faiss.IndexFlatIP(settings.embedding_dim)
            self._metadata = []

    def _persist(self) -> None:
        faiss.write_index(self._index, str(_INDEX_FILE))
        with open(_METADATA_FILE, "wb") as f:
            pickle.dump(self._metadata, f)

    # ── Write ──────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        embeddings: np.ndarray,
        metadata_list: list[dict],
    ) -> None:
        """Add pre-computed embeddings with associated metadata."""
        with self._lock:
            self._index.add(embeddings)
            self._metadata.extend(metadata_list)
            self._persist()

    def delete_document(self, document_id: str) -> int:
        """
        Remove all vectors belonging to a document.
        FAISS FlatIP doesn't support in-place deletion, so we rebuild.
        Returns the number of vectors removed.
        """
        with self._lock:
            keep_idx = [
                i for i, m in enumerate(self._metadata)
                if m["document_id"] != document_id
            ]
            removed = len(self._metadata) - len(keep_idx)
            if removed == 0:
                return 0

            # Reconstruct index from kept vectors
            all_vecs = np.zeros(
                (self._index.ntotal, settings.embedding_dim), dtype=np.float32
            )
            self._index.reconstruct_n(0, self._index.ntotal, all_vecs)

            kept_vecs = all_vecs[keep_idx]
            new_index = faiss.IndexFlatIP(settings.embedding_dim)
            if len(kept_vecs) > 0:
                new_index.add(kept_vecs)

            self._index = new_index
            self._metadata = [self._metadata[i] for i in keep_idx]
            self._persist()
            return removed

    # ── Read ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most similar chunks.
        Optionally filters by a list of document_ids.
        """
        if self._index.ntotal == 0:
            return []

        # Over-fetch to allow post-filter by document_id
        fetch_k = min(top_k * 10, self._index.ntotal) if document_ids else top_k
        scores, indices = self._index.search(query_embedding, fetch_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            if document_ids and meta["document_id"] not in document_ids:
                continue
            results.append(
                RetrievedChunk(
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    chunk_index=meta["chunk_index"],
                    text=meta["text"],
                    similarity_score=float(score),
                )
            )
            if len(results) >= top_k:
                break

        return results

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal


# Singleton – one store per process
vector_store = VectorStore()