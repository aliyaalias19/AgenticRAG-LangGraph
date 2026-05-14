# Agentic RAG

Document-grounded chatbot built on a LangGraph state machine, hybrid retrieval (BM25 + dense vectors + RRF), cross-encoder reranking, self-grading, and query rewriting. Local-first (Ollama default), with optional OpenAI / Bedrock providers. FastAPI backend, Next.js frontend, Qdrant vector store.

---

## Thought Process & Implementation Flow

### Why Agentic RAG?

The motivation was to move beyond naive single-pass RAG — embed query → similarity search → prompt LLM — and build a pipeline that **inspects its own retrieval quality** and corrects itself before answering.

Design principles:

1. **Observe before answering.** The graph scores retrieval relevance, and only generates an answer when the chunks are good enough. If they aren't, it rewrites the query and retries.
2. **Don't pay tokens for routing.** Routing decisions in the loop (rewrite vs answer) are deterministic functions of the graph state — no LLM round-trip is spent on "what should I do next."
3. **Local-first by default.** The system runs end-to-end against a local Ollama instance with zero cloud calls. Cloud providers (OpenAI, Bedrock) are explicit opt-in via env var.

### Implementation Flow

**Stage 1 — Baseline RAG**
A single-pass pipeline: embed query → vector search → prompt LLM. Worked for direct factual lookups but failed silently on ambiguous queries (wrong chunks retrieved with no signal) and keyword-heavy queries (dense embeddings miss exact-term matches).

**Stage 2 — Hybrid Retrieval + Cross-Encoder Reranking**
Added BM25 sparse retrieval alongside dense vector search and fused both result sets via Reciprocal Rank Fusion (RRF, k=60). A local cross-encoder (`BAAI/bge-reranker-v2-m3`) re-scores the fused candidates. This raised precision on keyword-heavy and multilingual queries where dense embeddings alone underperformed.

**Stage 3 — Self-grading & Query Rewriting**
A two-tier grading node scores retrieved chunks 1–10 against the query. The heuristic tier (keyword overlap, zero tokens) handles clearly-good or clearly-bad cases; only borderline scores escalate to an LLM grade. When the score is below threshold, a rewriter rephrases the query (precise → broadened → decomposed strategies) and retrieval runs again. This eliminated the silent-bad-retrieval failure mode.

**Stage 4 — Migration to LangGraph**
The original imperative `while steps < max: decide(); act()` loop spent LLM tokens on the routing decision itself. It was replaced with a `LangGraph` `StateGraph`: nodes are pure functions of state, conditional edges encode the routing logic in code, and the graph is compiled once at startup. Routing now costs zero tokens, the topology is immutable, and each node can be unit-tested without any framework setup.

**Stage 5 — Post-generation grounding check & evaluation**
A `hallucination_check` node verifies the answer is grounded in the retrieved context. For high-relevance retrievals (score ≥ 8), the LLM call is skipped — the prior probability of hallucination is low enough that the token cost isn't justified. The system exposes two raw quality signals (`relevance_score`, `hallucination_score`) and refuses to fabricate a composite "confidence" with uncalibrated weights.

A RAGAS evaluation harness measures four metrics over a fixed test set; thresholds gate a CI deploy via `check_eval_thresholds()`.

---

## Traditional RAG vs Agentic RAG

### Traditional RAG

```
Query → Embed → Vector Search → Top-K → Prompt LLM → Answer
```

Fast and deterministic, but brittle:

| Characteristic | Detail |
|---|---|
| **Retrieval** | One pass, fixed top-K, no feedback |
| **Query handling** | Raw user query used as-is |
| **Quality control** | None — bad retrieval produces a wrong answer silently |
| **Multi-step reasoning** | Not supported |
| **Failure mode** | Hallucinates or returns low-confidence answers without surfacing why |

### Agentic RAG (this system)

```
Query → analyze_query → retrieve → grade
                                      │
                  ┌───────────────────┴───────────────────┐
                  │ score ≥ 3                   score < 3 │
                  ▼                                       ▼
              generate                                  rewrite ──┐
                  │                                              │
                  ▼                                  (loop back to retrieve)
          hallucination_check
                  │
                  ▼
                 END
```

| Characteristic | Detail |
|---|---|
| **Retrieval** | Multi-pass with self-grading; rewrites the query and retries if quality is low |
| **Query handling** | Rewriter chooses precise / broadened / decomposed strategy based on rewrite count |
| **Quality control** | 2-tier grader (heuristic + LLM) emits a 1–10 relevance score with structured feedback |
| **Routing** | Deterministic conditional edges in the LangGraph — zero LLM tokens spent on routing |
| **Failure mode** | Two raw quality signals returned; the UI surfaces a low-confidence answer rather than hiding it |

### Why LangGraph over a hand-rolled loop?

Honest answer: for six mostly-linear nodes, LangGraph is overhead — ~60 lines of plain Python with a typed dict would do. It earns its keep on (1) typed state with append semantics for `reasoning_trace` (`Annotated[List[str], operator.add]`), (2) nodes as pure `state → dict` functions that unit-test cleanly without importing the framework, and (3) a near-zero migration cost when the graph eventually wants checkpointing, interrupts (human-in-the-loop), or parallel fan-out. The trade-off is documented in [`backend/agents/graph.py`](backend/agents/graph.py).

---

## Architecture

```
                              POST /query  (LangGraph state machine)
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   analyze_query     │  regex tier classifier (fast | smart)
                            │   (0 LLM tokens)    │  caps step budget
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │      retrieve       │  Qdrant similarity_search (top 3)
                            │   (0 LLM tokens)    │  + BM25 (top 2)
                            │                     │  → RRF fusion (k=60)
                            │                     │  → cross-encoder rerank
                            └──────────┬──────────┘
                                       ▼
                            ┌─────────────────────┐
                            │       grade         │  heuristic keyword overlap
                            │   (0 tokens ~70%)   │  → LLM (Haiku/local) only on borderline
                            └──────────┬──────────┘
                                       │
                          ┌────────────┴────────────┐
                  score ≥ 3                    score < 3
                  & rewrites < 2               & rewrites < 2
                          │                            │
                          ▼                            ▼
                  ┌──────────────┐            ┌──────────────┐
                  │   generate   │            │   rewrite    │  precise → broadened
                  │   tier=fast  │            │  (back to    │  → decomposed
                  │   or smart   │            │   retrieve)  │
                  └───────┬──────┘            └──────────────┘
                          ▼
              ┌────────────────────────┐
              │  hallucination_check   │  fast-path if relevance ≥ 8
              │                        │  otherwise Haiku/local LLM verdict
              └───────────┬────────────┘
                          ▼
                         END
                          │
                          ▼
              Returns: answer · citations · relevance_score · hallucination_score
                       · rewritten_query · steps_taken · reasoning_trace · latency_ms
```

---

## Key Features

| Feature | Detail |
|---|---|
| **LangGraph state machine** | 6 nodes, deterministic conditional edges, compiled once at startup |
| **Local-first providers** | Default LLM + embeddings = Ollama. OpenAI / Bedrock are explicit opt-in |
| **Hybrid retrieval** | BM25 (sparse) + Qdrant (dense) → RRF fusion → cross-encoder rerank |
| **Cross-encoder reranker** | `BAAI/bge-reranker-v2-m3` (multilingual; ~570MB; pre-warmed at startup) |
| **Two-tier grading** | Heuristic keyword-overlap score; LLM grader only on borderline (~30% of queries) |
| **Query rewriting** | Three escalating strategies — precise / broadened / decomposed (max 2 rewrites) |
| **Hallucination check** | Skipped for high-relevance retrievals; LLM verdict otherwise. No composite confidence score |
| **Sentence-level citations** | `[N]` markers in the answer are mapped to the best-matching sentence in each chunk |
| **SSE streaming** | Per-node `step` events while the graph runs, then `token` / `citations` / `meta` |
| **Multi-tenancy** | `X-API-Key` → tenant_id. Vector store + BM25 + audit log all partition by tenant |
| **Audit log** | Append-only JSONL at `${DATA_DIR}/audit.jsonl` (ingest, delete, query, query_stream, evaluate) |
| **Circuit breakers** | Separate breakers for the LLM call and the vector store call |
| **Prometheus metrics** | `/metrics` in text-exposition format — requests, latency, retrieval latency, audit volume, auth rejections |
| **Structured JSON logs** | Per-request correlation ID propagated through thread-pool executors |
| **SHA-256 dedup** | Re-ingesting the same PDF content (per tenant) is a no-op |
| **Layout-aware chunking** | `pymupdf4llm` preserves headings, tables, page boundaries; chunked with markdown-aware separators |
| **RAGAS evaluation** | 4-metric evaluation with a declared shipping bar (`EVAL_THRESHOLDS`) |
| **Rate limiting** | SlowAPI: 20 req/min (`/query`), 10 req/min (`/query/stream`) |

---

## Quick Start

### Prerequisites

- Python 3.10+ and Node.js 18+
- **One of:** Ollama running locally (default), `OPENAI_API_KEY`, or AWS credentials for Bedrock

### Option A — Docker Compose (recommended)

Brings up `backend`, `frontend`, `qdrant`, and `ollama` together with persistent volumes.

```bash
cp backend/.env.example backend/.env
docker compose up --build
```

Then open [http://localhost:3000](http://localhost:3000). Qdrant is at `:6333`, the API at `:8000`.

### Option B — Local development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (in another shell)
cd frontend
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

Ollama must be reachable at `OLLAMA_BASE_URL` (default `http://localhost:11434`) with `llama3.2` and `nomic-embed-text` pulled, or set `LLM_PROVIDER`/`EMBEDDING_PROVIDER` to `openai` or `bedrock`.

---

## Configuration

All settings are env vars read at startup (see [`backend/.env.example`](backend/.env.example)).

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `bedrock` \| `auto` |
| `EMBEDDING_PROVIDER` | `ollama` | Same set; deliberately decoupled from `LLM_PROVIDER` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `LLM_MODEL` | `llama3.2` | Ollama model used across all tiers |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `QDRANT_URL` | — | When set, uses Qdrant server; otherwise embedded mode |
| `QDRANT_PATH` | `./qdrant_data` | Embedded-mode storage path |
| `DATA_DIR` | `./data` | PDFs, BM25 index, audit log, eval history |
| `COLLECTION_NAME` | `rag_docs` | Qdrant collection name |
| `VECTOR_TOP_K` / `BM25_TOP_K` / `FINAL_TOP_K` | `3` / `2` / `3` | Retrieval top-K per stage |
| `MIN_RELEVANCE_SCORE` | `3` | Below this triggers a rewrite |
| `MAX_STEPS` / `MAX_REWRITES` | `5` / `2` | Loop caps |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | RecursiveCharacterTextSplitter config |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder (multilingual). Override for English-only corpora |
| `RERANKING_ENABLED` | `true` | Disable to skip the rerank stage |
| `API_KEYS` | — | JSON map `{"key":"tenant_id"}`. Empty = single-tenant demo mode |
| `DEFAULT_TENANT` | `default` | Used when auth is off |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated |

### Provider resolution (auto mode)

```
LLM_PROVIDER=auto   →  Ollama (if reachable)  →  OpenAI (if API key set)  →  Bedrock
```

The order is **local-first**. Cloud APIs are not assumed reachable from an on-prem deployment.

### Per-tier model mapping

| Tier | Used for | Ollama | OpenAI | Bedrock |
|---|---|---|---|---|
| `fast` | Grading, hallucination check, classification | `LLM_MODEL` | `gpt-4o-mini` | Claude Haiku 4.5 |
| `smart` | Answer generation when the query is complex | `LLM_MODEL` | `gpt-4o` | Claude Sonnet 4.5 |
| `rewrite` | Query rewriting | `LLM_MODEL` | `gpt-4o-mini` | Claude Haiku 4.5 |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness, Ollama status, ingested files, resolved providers |
| `GET` | `/metrics` | Prometheus text-exposition |
| `POST` | `/documents` | Upload and ingest a single PDF (multipart) |
| `POST` | `/documents/bulk` | Ingest every PDF in the tenant's `DATA_DIR` |
| `GET` | `/documents` | List ingested documents for the caller's tenant |
| `DELETE` | `/documents/{filename}` | Remove a document and all its chunks |
| `POST` | `/supervisor/classify` | Classify message intent (keyword-only, no LLM) |
| `POST` | `/query` | Full agentic RAG query (non-streaming) |
| `POST` | `/query/stream` | Server-sent events: live per-node trace, then tokens, citations, meta |
| `POST` | `/evaluate` | RAGAS evaluation over a test set (uses sample suite by default) |

All non-public routes accept `X-API-Key` when `API_KEYS` is configured; the matched tenant_id is bound to the request and scopes every retrieval, ingestion, and audit row.

### Streaming response

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum investment?"}'
```

Events (newline-delimited JSON inside SSE `data:` frames):

```json
{"type":"step",     "node":"analyze_query", "message":"[QueryAnalysis] tier=FAST · max_steps=3"}
{"type":"step",     "node":"retrieve",      "message":"[Retrieve] 3 chunks (dense=3 bm25=2 reranked=True) · 412ms"}
{"type":"step",     "node":"grade",         "message":"[Grade] 8/10 — Strong keyword match (heuristic)"}
{"type":"step",     "node":"generate",      "message":"[Generate] tier=fast · 1 citations"}
{"type":"token",    "content":"The "}
{"type":"token",    "content":"minimum "}
{"type":"citations","data":[{"chunk_id":1,"page":3,"source_file":"doc.pdf","snippet":"...","full_chunk":"...","similarity_score":0.91}]}
{"type":"meta",     "original_query":"...", "rewritten_query":"...", "relevance_score":8, "hallucination_score":0.05, "steps_taken":1, "latency_ms":3214.0}
```

Followed by a final `data: [DONE]`.

### Why no `/agent/orchestrate`

An earlier version exposed `/agent/orchestrate` with `rag_then_summarise` and `rag_then_analyse` modes, framed as multi-agent orchestration. It wasn't — it ran the same RAG pipeline and appended a templated LLM pass. The endpoint was removed; summarisation and analysis are achieved by phrasing the question accordingly ("summarise the fee structure" → `/query` returns a summary).

---

## Retrieval Pipeline

```
Query (or rewritten query)
        │
        ├─ Dense:  Qdrant similarity_search    (VECTOR_TOP_K, payload filter on tenant_id)
        └─ Sparse: BM25Okapi over in-memory corpus (BM25_TOP_K, over-retrieve 5×, post-filter)
                  ▼
             RRF fusion  score = Σ 1 / (60 + rank + 1)
                  ▼
             Cross-encoder rerank (BAAI/bge-reranker-v2-m3, local, ~570MB)
                  ▼
             Top FINAL_TOP_K chunks
```

- **BM25 persistence:** the corpus is persisted as JSON at `${DATA_DIR}/bm25_index.json` (deliberately not pickle — `pickle.load` is a code-execution sink on tampered files). The index rebuilds automatically after every ingest or delete.
- **Tenant isolation:** Qdrant filters server-side via a payload index on `tenant_id`. BM25 shares one in-memory index across tenants and over-retrieves to satisfy a post-filter without recall loss.

---

## Multi-tenancy & Auth

Single-tenant by default. Setting `API_KEYS` turns on enforcement:

```bash
export API_KEYS='{"sk-alpha-123":"tenant-alpha","sk-beta-456":"tenant-beta"}'
```

- Every non-public route requires `X-API-Key: <key>`; missing / wrong returns `401`.
- The matched `tenant_id` is propagated through ingestion, retrieval, deletion, and audit — there is no path by which tenant A's chunks can surface in tenant B's results.
- Public paths (`/health`, `/metrics`, `/docs`, `/redoc`, `/openapi.json`) bypass auth so liveness probes work.
- API keys are never logged. The audit log stores only `key[:4]…key[-2:]` as actor fingerprint.

---

## Observability

| Surface | What it exposes |
|---|---|
| `/health` | Status, version, Ollama reachability, ingested files, resolved providers, feature flags |
| `/metrics` | Prometheus counters & histograms — HTTP requests/latency, retrieval latency, audit events, auth rejections |
| JSON logs | One JSON object per line on stdout, tagged with `request_id` correlation ID |
| Audit log | `${DATA_DIR}/audit.jsonl`, append-only, captures every state-changing or data-exposing event |

For multi-worker deployments, `prometheus_client` multiprocess mode requires `PROMETHEUS_MULTIPROC_DIR` and a small registry change — documented inline in `backend/infrastructure/prometheus.py`.

---

## Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-alpha-123" \
  -d '{"custom_test_cases": null}'
```

Uses the built-in 16-question suite by default (covering direct factual recall, multi-clause synthesis, out-of-scope refusal, ambiguous reference resolution, numeric precision, and negation). Append your own with `custom_test_cases`. RAGAS uses OpenAI (preferred via `OPENAI_API_KEY`) or Bedrock for grading; results append to `${DATA_DIR}/eval_history.jsonl`.

| Metric | What it measures | Shipping bar |
|---|---|---|
| Faithfulness | Answer claims grounded in retrieved context | ≥ 0.80 |
| Answer Relevancy | Response addresses the question | ≥ 0.80 |
| Context Precision | Retrieved chunks were relevant | ≥ 0.70 |
| Context Recall | Required info was retrieved | ≥ 0.70 |

`check_eval_thresholds()` gates a deploy on these — the bar travels with the code, not in CI YAML.

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

The suite is no-LLM and no-network — pure functions are exercised directly, anything stateful is patched at the dependency boundary.

- **`test_nodes.py`** — node-level unit tests
  - Query complexity classification (fast vs smart tier)
  - Heuristic grading (keyword overlap, score bounds)
  - `_route_after_grade` deterministic routing
  - Hallucination fast-path bypass and "no composite confidence" regression
  - `[N]` marker → sentence-level citation mapping
  - BM25 build / retrieve / persist / reload / tenant filter
  - Circuit breaker open/closed/reset
  - SHA-256 dedup
  - Markdown cleanup at ingest
  - API-key auth resolution (demo / public / valid / invalid)
  - Audit log writer (well-formed JSONL, non-serialisable detail coercion)
  - RAGAS threshold gate

- **`test_routes.py`** — end-to-end HTTP via `httpx.ASGITransport`
  - Auth gate (401 on missing / wrong key)
  - Public-path bypass (`/health`, `/metrics`)
  - Tenant ID propagation through the FastAPI dependency chain
  - Audit row emission on successful query

---

## Project Structure

```
agentic-rag/
├── backend/
│   ├── main.py                         # FastAPI app + lifespan + middleware
│   ├── core/
│   │   ├── config.py                   # Pydantic Settings, fail-fast validation
│   │   ├── auth.py                     # X-API-Key → TenantContext
│   │   ├── audit.py                    # Append-only JSONL audit log
│   │   ├── logging.py                  # JSON formatter + request_id contextvar
│   │   └── exceptions.py               # Typed exception hierarchy
│   ├── agents/
│   │   ├── graph.py                    # LangGraph StateGraph (6 nodes) + streaming
│   │   ├── state.py                    # RAGState TypedDict (with operator.add trace)
│   │   └── nodes/
│   │       ├── query_analysis.py       # Tier classification (0 tokens)
│   │       ├── retrieval.py            # Hybrid retrieval orchestration
│   │       ├── grading.py              # Heuristic + LLM grader (2-tier)
│   │       ├── rewriting.py            # 3 escalating strategies
│   │       ├── generation.py           # Answer + sentence-level citation mapping
│   │       └── hallucination.py        # Grounding check (fast-path at relevance ≥ 8)
│   ├── retrieval/
│   │   ├── hybrid.py                   # BM25 + vector + RRF + rerank
│   │   ├── bm25_retriever.py           # Thread-safe BM25, JSON-persisted index
│   │   ├── reranker.py                 # Cross-encoder reranker (local)
│   │   └── vector_store.py             # Qdrant facade (embedded or server)
│   ├── ingestion/
│   │   └── pipeline.py                 # PDF → pymupdf4llm → chunks → Qdrant + BM25
│   ├── services/
│   │   ├── llm.py                      # Tiered LLM (Ollama / OpenAI / Bedrock)
│   │   └── embedding.py                # Embedding provider (decoupled from LLM)
│   ├── infrastructure/
│   │   ├── circuit_breaker.py          # Thread-safe breaker
│   │   └── prometheus.py               # Counters + histograms + render
│   ├── models/
│   │   ├── domain.py                   # Citation, RetrievedChunk, RetrievalMetadata
│   │   ├── requests.py                 # Pydantic v2 request schemas
│   │   └── responses.py                # Pydantic v2 response schemas
│   ├── api/routes/
│   │   ├── documents.py                # POST/GET/DELETE /documents
│   │   ├── query.py                    # /query, /query/stream, /supervisor/classify
│   │   ├── health.py                   # /health
│   │   └── evaluation.py               # /evaluate
│   ├── evaluation/
│   │   └── ragas_eval.py               # 4-metric eval + threshold gate + sample suite
│   ├── tests/
│   │   ├── test_nodes.py               # Node-level unit tests
│   │   └── test_routes.py              # E2E HTTP tests (auth, tenant, audit)
│   ├── data/                           # PDFs · bm25_index.json · audit.jsonl · eval_history.jsonl
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/app/page.tsx                # Chat UI · live thinking trace · citations · eval tab
│   └── src/lib/api.ts                  # Typed client + SSE stream handler
│
└── docker-compose.yml                  # backend · frontend · qdrant · ollama + volumes
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (default) · OpenAI · AWS Bedrock (Claude Haiku/Sonnet 4.5) |
| Embeddings | Ollama (`nomic-embed-text`) · OpenAI · Bedrock (Cohere multilingual v3) |
| Vector DB | Qdrant (embedded or server) — payload-indexed on `tenant_id`, `source_file`, `doc_hash` |
| Sparse retrieval | `rank-bm25` (BM25Okapi), JSON-persisted index |
| Reranker | `sentence-transformers` cross-encoder, default `BAAI/bge-reranker-v2-m3` |
| Orchestration | LangGraph `StateGraph` (compiled once, pure-function nodes) |
| PDF ingest | `pymupdf4llm` (layout-aware markdown) + `RecursiveCharacterTextSplitter` |
| Backend | FastAPI · SlowAPI · `prometheus-client` |
| Frontend | Next.js 14 · Tailwind · `lucide-react` · `clsx` |
| Evaluation | RAGAS + HuggingFace `datasets` |
| Observability | Structured JSON logs · Prometheus `/metrics` · append-only audit JSONL · request correlation IDs |
