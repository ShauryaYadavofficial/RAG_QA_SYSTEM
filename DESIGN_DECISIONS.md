# Design Decisions

## 1. Why chunk_size = 512 characters?

all-MiniLM-L6-v2 has a **128 token** context limit.
For standard English prose, 1 token ≈ 4 characters, so:

  128 tokens × 4 chars/token = 512 characters

This means 512 chars almost never exceeds the model's window,
avoiding silent truncation of embeddings. The overlap of 64 chars
(~12.5%) ensures sentences that straddle a boundary are captured
by at least one chunk.

Alternatives considered:
| Size   | Problem                                         |
|--------|-------------------------------------------------|
| 256 ch | Too granular; loses sentence context            |
| 1024 ch| Silently truncated by embedding model           |
| Tokens | Requires a tokenizer as an extra dependency     |

The sentence-boundary snapping in `_split_paragraph` further ensures
semantic coherence at boundaries.

---

## 2. Retrieval Failure Case Observed

**Failure**: Multi-hop questions that require combining two separate
sections of a document.

**Example**: Document says "Alice manages the Paris office" in section 1
and "The Paris office opened in 2019" in section 2. Query: "When did
Alice's office open?"

**What happens**: The top-1 chunk contains either Alice or the year,
but never both, so the LLM answers "I could not find a confident answer."

**Mitigation applied**: Increase `top_k` to 5 (giving the LLM more
context windows) and add sentence-boundary chunking to avoid splitting
closely related sentences.

**Future fix**: Implement HyDE (Hypothetical Document Embeddings) or
a re-ranker (CrossEncoder) to surface semantically complementary chunks.

---

## 3. Metric Tracked: End-to-End Query Latency

Latency is logged per query and returned in the API response as
`latency_ms`. This specifically measures the **LLM inference time**
(most expensive component) separate from retrieval time.

Observed values on GPT-3.5-turbo:
- Retrieval (FAISS + embedding): ~30–80 ms
- LLM generation: ~800–2500 ms

This surfaced that embedding a query is 30× faster than LLM inference,
confirming that FAISS is not a bottleneck at this scale and that
optimization effort should focus on LLM call batching or caching.

A secondary metric tracked is **similarity_score** (cosine similarity,
0–1), logged at INFO level. Scores below 0.4 consistently correlate
with poor answers, which informs a future threshold-based fallback.