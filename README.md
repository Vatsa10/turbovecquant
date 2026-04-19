<p align="center">
  <img src="docs/header.png" alt="turbovecquant — TurboQuant for LLM APIs" width="100%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="TurboQuant paper"></a>
</p>

---

**A production-grade semantic cache, RAG compressor, and OpenAI-compatible proxy for hosted LLM APIs — built on Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) quantizer.**

Hosted LLM APIs (OpenAI, Anthropic, Gemini, OpenRouter, ...) are metered by the token and the provider owns everything inside the model — weights, activations, KV cache. This project compresses the *client side* of that boundary: repeated prompts, retrieved RAG context, and long-running conversation memory. All three sit on top of a 2-4 bit quantized ANN index (turbovec) so memory and recall cost stay flat as traffic scales.

- **Semantic response cache** — cosine-match near-duplicate prompts, skip the paid API call.
- **RAG context compressor** — retrieve and token-budget-pack chunks before sending, reducing billed input tokens.
- **Conversation memory** — per-session chat history with optional semantic recall across turns.
- **OpenAI-compatible proxy** — drop-in gateway so existing OpenAI SDKs get caching with zero code changes.
- **Prod features** — streaming, async, retries, tenant scoping, PII redaction, negative caching, $-saved tracking, Prometheus metrics.

The transformer KV cache stays on the provider's GPUs — unreachable from a client library. What this project compresses is everything else.

## Install

```bash
# core + OpenAI adapters
pip install "turbovec[llm,llm-openai]"

# pick any combination of providers
pip install "turbovec[llm-anthropic,llm-gemini,llm-cohere,llm-voyage,llm-local]"

# optional backends and integrations
pip install "turbovec[llm-redis]"      # Redis-backed cache + memory (TTL + LRU + multi-host)
pip install "turbovec[llm-proxy]"      # FastAPI OpenAI-compatible server
pip install "turbovec[llm-metrics]"    # Prometheus exposition
```

## Semantic response cache

```python
from turbovec.llm import SemanticCache
from turbovec.llm.embedders import OpenAIEmbedder
from turbovec.llm.llm import OpenAIChat
from turbovec.llm.stores import RedisStore

cache = SemanticCache(
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    llm=OpenAIChat(model="gpt-4o-mini"),
    similarity_threshold=0.93,
    store=RedisStore(url="redis://localhost:6379/0", ttl_seconds=86400),
)

reply = cache.complete(messages=[{"role": "user", "content": "capital of france?"}])
print(reply.text, cache.stats())

# Near-duplicate query reuses the cached completion — no upstream call, no billed tokens.
cache.complete(messages=[{"role": "user", "content": "what's france's capital"}])
```

Store backends: `InMemoryStore` (dev), `PickleStore(path)` (single-process disk), `RedisStore(url, ttl_seconds=...)` (shared, TTL + LRU, multi-host). System prompts are fingerprinted so two personas sharing one cache never collide.

### Streaming with cache-on-completion

```python
for chunk in cache.stream_complete(messages):
    print(chunk, end="", flush=True)   # first call streams from upstream and caches
# second call with a near-duplicate prompt streams from the cache — zero upstream latency
```

Async equivalents are `acomplete` and `astream_complete`.

### Gating — what NOT to cache

Non-deterministic and side-effectful calls bypass the cache automatically:

- `temperature > 0` (non-deterministic)
- `tools=[...]` or `functions=[...]` in kwargs (model may invoke side effects)
- Any message with `tool_calls` / `function_call` / `role=tool` (already mid-tool-use)

Override with `force_cache=True` when you know it's safe. Opt in globally with `SemanticCache(cache_nondeterministic=True, allow_tools=True)`.

### Tenant scoping and invalidation

```python
cache.complete(messages, tenant="org:acme")    # entry is scoped to acme
cache.complete(messages, tenant="org:widgets") # different tenant — different entry

cache.invalidate(tenant="org:acme")            # drop one tenant
cache.invalidate(older_than_seconds=86400)    # drop anything older than a day
cache.invalidate(predicate=lambda e: "gpt-4o-mini" in e["model"])
```

### Cost tracking

```python
cache.stats()
# {'hits': 42, 'misses': 8, 'hit_rate': 0.84,
#  'tokens_saved_in': 82100, 'tokens_saved_out': 46300,
#  'tokens_saved': 128400, 'dollars_saved': 0.3137,
#  'entries': 50, 'bypassed': 4, 'errors': 0, ...}
```

The pricing table covers recent OpenAI, Anthropic, and Gemini models and falls back to a longest-prefix match. Override with `SemanticCache(pricing={"my-model": (0.001, 0.002)})` if you're on a custom tier.

### PII redaction

Applied to both the cache-key text and the cached response text. The default redactor catches email, US SSN, 13-16 digit card-shaped numbers, AWS access keys, `sk-*` API keys, and bearer tokens:

```python
SemanticCache(..., redactor=default_redactor)       # default
SemanticCache(..., redactor=my_presidio_redactor)   # plug your own
SemanticCache(..., redactor=None)                   # disable
```

### Negative caching

Short-lived entries for upstream errors — absorbs retry storms when a provider is flaky:

```python
SemanticCache(..., negative_cache_seconds=30)
```

Subsequent requests for the same prompt within the window re-raise the cached error without calling the upstream.

### Retries with exponential backoff

Built into every LLM adapter (`retries=3` default). Retries on 408/409/425/429/500/502/503/504 and transport-layer timeouts; fails fast on any other 4xx.

```python
OpenAIChat(model="gpt-4o-mini", retries=5)
AnthropicChat(model="claude-sonnet-4-6", retries=3)
```

### Observability

```python
from turbovec.llm import Metrics, AuditLogger

cache = SemanticCache(
    ...,
    metrics=Metrics(),                # Prometheus: requests/hits/misses/errors/latency/tokens_saved/dollars_saved
    audit=AuditLogger(),              # one JSON line per request to the `turbovec.llm.audit` logger
)
```

Each `Metrics()` instance owns its own `CollectorRegistry` so multiple caches in one process don't collide. Expose it via the built-in `/metrics` route on the proxy, or mount it in your own app.

### Eviction

`TurboQuantIndex` supports native tombstone deletion with slot reuse, so evicted entries reclaim their slots on the next `add` — no periodic `compact()` needed. `cache.invalidate(...)` routes through `TurboQuantIndex.delete`, and `SemanticCache` auto-tombstones any slot whose payload has been evicted out from under it (e.g., by Redis TTL):

```python
cache.invalidate(tenant="org:acme")               # delete + free slots for reuse
cache.invalidate(older_than_seconds=86400)

from turbovec.llm.eviction import lfu_evict, ttl_evict
lfu_evict(cache.store, max_entries=100_000)       # drop coldest payload entries
ttl_evict(cache.store, max_age_seconds=7*86400)   # drop payloads older than a week
cache.compact()                                    # optional — defragment after mass eviction
```

With Redis, `ttl_seconds` on the store gives native TTL + LRU; slots freed by eviction are automatically tombstoned on the next lookup that hits an empty payload.

Raw index primitives (use when managing the index directly):

```python
index.delete([0, 5, 12])   # tombstone; slots reused by next add()
index.is_deleted(5)        # -> True
index.num_deleted()        # tombstone count
index.capacity()           # physical slot count (incl. tombstones)
len(index)                 # live count
new_ids = index.add(vecs)  # returns slot ids (reused slots first, LIFO)
```

## OpenAI-compatible proxy

Drop-in replacement for the OpenAI endpoint — point any existing OpenAI SDK at it and get caching, tenant isolation, metrics, and streaming with zero code changes.

```bash
pip install "turbovec[llm-proxy,llm-openai,llm-metrics,llm-redis]"
```

```python
# serve.py
import uvicorn
from turbovec.llm import SemanticCache, Metrics, AuditLogger
from turbovec.llm.embedders import OpenAIEmbedder
from turbovec.llm.llm import OpenAIChat
from turbovec.llm.stores import RedisStore
from turbovec.llm.proxy import build_app

cache = SemanticCache(
    embedder=OpenAIEmbedder(),
    llm=OpenAIChat(model="gpt-4o-mini"),
    store=RedisStore(url="redis://localhost:6379/0", ttl_seconds=86400),
    metrics=Metrics(),
    audit=AuditLogger(),
    negative_cache_seconds=30,
)
app = build_app(cache)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```python
# client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="tv-anything")
client.chat.completions.create(model="gpt-4o-mini", messages=[...])
```

**Routes**
- `POST /v1/chat/completions` — OpenAI-compatible, streaming + non-streaming.
- `GET /v1/models` — reports the configured model.
- `GET /stats` — JSON cache stats.
- `GET /metrics` — Prometheus exposition of this cache's registry.
- `POST /admin/invalidate` — `{"tenant": "...", "older_than_seconds": N}`.

**Tenant isolation** — the `Authorization: Bearer <key>` header is hashed into a 16-char tenant id, so different API keys never share cache entries.

## RAG context compressor

```python
from turbovec.llm import RAGCompressor
from turbovec.llm.embedders import SentenceTransformerEmbedder

rag = RAGCompressor(embedder=SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"))
rag.add_documents(open("corpus.txt").read().split("\n\n"))

context = rag.compress(query="what do we know about X?", token_budget=2000)
# feed `context` into your LLM prompt — billed input tokens drop accordingly
```

Chunking, token counting, and over-retrieval are all pluggable. Default token counter is `tiktoken`'s `cl100k_base` with a length-based fallback.

## Conversation memory

```python
from turbovec.llm import ConversationMemory
from turbovec.llm.memory import RedisHistoryBackend

mem = ConversationMemory(
    backend=RedisHistoryBackend(url="redis://localhost:6379/0", ttl_seconds=3600),
    embedder=OpenAIEmbedder(),  # optional — enables semantic recall
)

mem.append("user:42", {"role": "user", "content": "..."})
recent   = mem.history("user:42", last_n=20)
relevant = mem.recall("user:42", query="what did I decide about auth earlier?", k=5)
```

Short-term history lives in the backend (in-memory dict or Redis list with TTL). With an embedder supplied, turns are also indexed so sessions can exceed raw token limits without resending the full transcript.

## Providers

| Surface              | Adapters shipped                                                  |
|----------------------|-------------------------------------------------------------------|
| LLM completions      | OpenAI, OpenRouter (OpenAI-compatible), Anthropic, Gemini         |
| Embeddings           | OpenAI, Cohere, Voyage, local `sentence-transformers`             |
| Payload stores       | InMemory, Pickle-on-disk, Redis                                   |
| Conversation history | InMemory, Redis                                                   |

Adding another provider is ~30 lines — implement `complete` / `stream` / `acomplete` / `astream` against the `LLM` protocol or `embed(texts)` against the `Embedder` protocol.

## Under the hood — turbovec

Every feature above is backed by a single `TurboQuantIndex`. This is Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) quantizer (ICLR 2026), implemented as a Rust crate with Python bindings. Two things make it suitable for online, streaming workloads like an LLM cache:

**Data-oblivious.** No codebook training, no calibration, no rebuilds as the cache grows. Add vectors, they're indexed. This is why the cache can absorb traffic in real time without a "training" phase.

**Near-optimal distortion.** Normalize → random rotation → Lloyd-Max scalar quantize per coordinate → bit-pack. After rotation, every coordinate follows a known Beta distribution that converges to Gaussian N(0, 1/d) in high dimensions — regardless of input data. The paper proves distortion lies within 2.7× of Shannon's information-theoretic lower bound.

A 1536-dim OpenAI embedding goes from 6,144 bytes (FP32) to **384 bytes (2-bit)** — 16× compression. Search is a single SIMD scan over packed codes (AVX-512BW on modern x86, NEON on ARM, AVX2 fallback) using nibble-split lookup tables, so every stage of this stack — cache lookup, RAG retrieval, semantic memory recall — runs at full vector-search speed with no decompression.

```python
from turbovec import TurboQuantIndex

index = TurboQuantIndex(dim=1536, bit_width=4)
index.add(vectors)                         # np.ndarray[n, 1536] float32
scores, idx = index.search(query, k=10)
```

Full benchmarks vs FAISS IndexPQFastScan are in `benchmarks/results/`; see [the original turbovec upstream](https://github.com/RyanCodrai/turbovec) for the recall and speed charts. At 4-bit the two are indistinguishable (>0.955 top-1 recall across every dataset); turbovec beats FAISS FastScan by 12-20% on ARM and matches-or-beats it on x86.

## Building

```bash
pip install maturin
cd turbovec-python
maturin build --release
pip install target/wheels/*.whl 
or
pip install ../target/wheels/turbovec-0.2.0-cp39-abi3-win_amd64.whl
```

All x86_64 builds target `x86-64-v3` (AVX2 baseline, Haswell 2013+) via `.cargo/config.toml`. Any CPU that runs the AVX2 fallback kernel runs the whole crate — the AVX-512 kernel is gated at runtime via `is_x86_feature_detected!`.

## Roadmap

- **~~Native deletion with slot reuse.~~ Done.** Rust `TurboQuantIndex.delete(ids)` tombstones slots, the next `add` reuses them in LIFO order, and the `.tv` file format persists tombstones via an optional post-norms section (legacy files still load). `SemanticCache.invalidate()` now routes through this and stale Redis payloads auto-tombstone on detection — no compaction job required.
- **Batch + prefix cache.** Multi-turn conversations share long system + history prefixes; caching common prefixes separately from the user tail could push hit rates higher on agent workloads.
- **Streaming rerank.** Over-retrieve from the quantized index, then rerank top-k with a cross-encoder before packing into the token budget.
- **Semantic routing.** Use the index to classify queries and route easy ones to a cheaper model. The quantization already makes the classifier free.
- **QJL residual (explicitly out of scope).** The paper's 1-bit QJL residual gives unbiased inner-product estimation — useful inside attention softmax, not for threshold-based cache lookup. Switching to 3-bit Lloyd-Max is a simpler path to any recall improvement this use case needs.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) — the quantizer this is built on.
- [turbovec upstream](https://github.com/RyanCodrai/turbovec) — the Rust vector-index crate this project forks and extends.
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) — the nibble-LUT scoring layout turbovec's x86 kernel adapts.
