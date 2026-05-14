# Agentic RAG

 Chatbot built with a ReAct agent loop, hybrid retrieval (BM25 + vector + RRF), intelligent model routing, and a Next.js frontend. 

---

## Thought Process & Implementation Flow

### Why Agentic RAG?

The core motivation behind this system was to move beyond the limitations of naive RAG pipelines where a single retrieval pass either finds the right context or silently fails and instead build a system that **reasons about its own retrieval quality** and corrects itself before generating an answer.

The design philosophy can be summarised in three principles:

1. **Observe before answering.** The agent inspects what it retrieved, scores relevance, and decides whether to retry with a rewritten query rather than blindly feeding low-quality chunks to the LLM.
2. **Match cost to complexity.** Not every query needs a frontier model. Simple factual lookups are routed to Haiku; only genuinely complex or analytical queries escalate to Sonnet. This reduces cost by 4–8× with no user-visible quality loss.

### Implementation Flow

The system was built in layered stages, each solving a concrete failure mode of the previous version:

**Stage 1 — Baseline RAG**
A straightforward pipeline: embed query → similarity search → prompt LLM. This worked for simple factual questions but failed on ambiguous queries (wrong chunks retrieved) and multi-part questions (single retrieval pass insufficient).

**Stage 2 — Hybrid Retrieval + Reranking**
BM25 sparse retrieval was added alongside dense vector search, with Reciprocal Rank Fusion (RRF) to merge the result sets. A local CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`) then re-scored the fused candidates. This significantly improved precision on keyword-heavy and domain-specific queries where dense embeddings alone performed poorly.

**Stage 3 — ReAct Agent Loop**
The retrieval pipeline was wrapped in a Reason + Act loop (max 5 steps). At each step the agent can: rewrite the query (precise, broadened, or decomposed), retrieve again, and grade the new results. The loop exits early when the grader scores retrieval ≥ 7/10, or falls back to the best available context after 5 attempts. This eliminated the "silent bad retrieval" failure mode.

**Stage 4 — Supervisor + Model Router**
An upstream supervisor LLM classifies intent (`question / summarise / analyse / list_docs / chitchat`) before the agent loop runs, so trivially off-topic or conversational messages never touch the retrieval pipeline. The model router then classifies query complexity and selects the appropriate model tier, keeping the fast path genuinely fast (~1,500 tokens, Haiku only).

**Stage 5 — Multi-agent Pipeline & Evaluation**
Secondary specialist agents (SummariserAgent, AnalystAgent) were introduced to handle tasks that require structured output beyond a direct answer. A RAGAS evaluation harness was added to continuously measure retrieval and answer quality across four metrics, enabling regression testing as the system evolves.

---

## Traditional RAG vs Agentic RAG

Understanding the distinction between the two paradigms clarifies most of the design decisions in this system.

### Traditional RAG

Traditional RAG follows a fixed, single-pass pipeline:

```
Query → Embed → Vector Search → Top-K Chunks → Prompt LLM → Answer
```

It is deterministic and fast, but brittle:

| Characteristic | Detail |
|---|---|
| **Retrieval** | One pass, fixed top-K, no feedback loop |
| **Query handling** | Raw user query used as-is for embedding |
| **Quality control** | None — bad retrieval produces a hallucinated or unhelpful answer silently |
| **Model usage** | Typically a single model for all queries |
| **Cost** | Predictable but unoptimised — same cost for simple and complex queries |
| **Multi-step reasoning** | Not supported — cannot decompose a question into sub-questions |
| **Failure mode** | Silently returns low-confidence answers when relevant chunks are not in top-K |

Traditional RAG is appropriate when: documents are short and well-structured, queries are simple and predictable, latency is the primary concern, and the cost of an occasional wrong answer is low.

### Agentic RAG

Agentic RAG wraps retrieval inside a reasoning loop. The LLM is not just a consumer of retrieved context — it is an active participant in deciding **whether** the retrieval was good enough and **what to do next**:

```
Query → Supervisor → Model Router → [ReAct Loop: Rewrite → Retrieve → Grade] → Answer
```

| Characteristic | Detail |
|---|---|
| **Retrieval** | Multi-pass with self-grading; retries with rewritten queries if quality is low |
| **Query handling** | Query is rewritten before retrieval (precise / broadened / decomposed) |
| **Quality control** | LLM grades its own retrieval (1–10) and provides structured feedback |
| **Model usage** | Dynamic routing — cheap model for simple tasks, expensive model only when needed |
| **Cost** | Higher per-query overhead but ~4–8× savings vs always using a large model |
| **Multi-step reasoning** | Supported — complex questions can be decomposed across multiple retrieval steps |
| **Failure mode** | Explicit — agent reports low confidence rather than hallucinating |

### Key Trade-offs

| Dimension | Traditional RAG | Agentic RAG |
|---|---|---|
| Latency | Lower (single pass) | Higher (up to 5 loop iterations) |
| Accuracy on ambiguous queries | Lower | Higher |
| Cost predictability | High | Variable (depends on complexity routing) |
| Implementation complexity | Low | High |
| Observability | Limited | Full trace per query (steps, rewrites, grades) |
| Best for | Simple Q&A, high-volume, low-stakes | Complex documents, analytical queries, high-stakes answers |

### Design Decision: Why ReAct over Chain-of-Thought or Tool-Call patterns?

ReAct (Reason + Act) was chosen over a pure Chain-of-Thought or tool-use pattern because it naturally maps the retrieval problem to an interleaved reasoning loop: the agent produces a thought (is this retrieval good?), takes an action (rewrite and retrieve again), observes the result (new chunks + grade), and repeats. This makes the agent's behaviour auditable — every step is recorded in the trace events emitted by the streaming endpoint — and it constrains the search space without requiring a complex tool-calling schema.

---

## Architecture

```
User Query
    │
    ▼
[Supervisor]  LLM classifies intent
    │         question / summarise / analyse / list_docs / chitchat
    │
    ▼
[Model Router]  Classifies query complexity
    │           simple → Haiku (fast)   complex → Sonnet (smart)
    │
    ▼
[ReAct Agent Loop]  max 5 steps
    │
    ├─ rewrite_query   → Haiku rewrites query (precise / broadened / decomposed)
    ├─ retrieve        → Hybrid BM25 + vector search, fused via RRF, reranked
    ├─ grade           → Haiku scores retrieval 1–10 with structured feedback
    └─ answer          → Haiku (simple) or Sonnet (complex) generates answer
                         with inline [N] citations
    │
    ├─ Standard output
    │
    └─ [Optional] Multi-agent pipeline
           RAGAgent → SummariserAgent  (executive summary)
           RAGAgent → AnalystAgent     (risks + action items)
    │
    ▼
[RAGAS Evaluation]
    Faithfulness · Answer Relevancy · Context Precision · Context Recall
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Model routing** | Simple queries → Haiku; complex → Sonnet. ~4–8× cost reduction vs fixed-model |
| **Hybrid retrieval** | BM25 sparse + Chroma dense, fused via Reciprocal Rank Fusion |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` re-ranks fused chunks locally |
| **Parent-child chunking** | Child chunks (300 chars) retrieved; parent context (1200 chars) sent to LLM |
| **Self-grading** | LLM scores retrieval quality and provides structured feedback for retry decisions |
| **Streaming SSE** | Token-by-token streaming with trace, citations, and metadata events |
| **Multi-agent bus** | RAGAgent posts structured messages; secondary agents read and respond |
| **Supervisor routing** | LLM classifies intent with keyword fallback |
| **Circuit breakers** | Separate breakers for LLM and vector store calls |
| **Audit logging** | Document deletions written to `chroma/audit.jsonl` |
| **RAGAS evaluation** | 4-metric evaluation suite with per-question breakdown and history tracking |
| **Rate limiting** | SlowAPI: 20 req/min (query), 10 req/min (stream) |

---

## Token Budget (per query)

| Query type | Approximate input tokens | Models used |
|---|---|---|
| Simple | ~1,500 | Haiku throughout |
| Complex | ~2,500 | Haiku loop + Sonnet answer |
| v3.0 baseline | up to ~8,000 | Sonnet 8-step loop |

---

## Quick Start

### Prerequisites

- AWS credentials configured (for Bedrock) **or** `OPENAI_API_KEY` set **or** Ollama running locally
- Python 3.10+, Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt

# With Bedrock (default)
AWS_REGION=us-east-1 uvicorn main:app --reload

# With OpenAI
OPENAI_API_KEY=sk-... uvicorn main:app --reload

# With Ollama
OLLAMA_BASE_URL=http://localhost:11434 uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `OPENAI_API_KEY` | — | Enables OpenAI provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for RAGAS evaluation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `LLM_MODEL` | `llama3.2` | Ollama model name |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHROMA_PATH` | `./chroma` | ChromaDB persistence path |
| `DATA_DIR` | `./data` | Directory for bulk PDF ingest |

### Model provider priority

```
Smart tier:   Bedrock Sonnet 3.7 → OpenAI GPT-4o → Ollama
Fast tier:    Bedrock Haiku 3.5  → OpenAI GPT-4o-mini → Ollama
Rewrite tier: OpenAI GPT-4o-mini → Bedrock Haiku 3.5 → Ollama
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check, Ollama status, ingested files, feature list |
| `GET` | `/metrics` | Aggregated latency and throughput metrics |
| `POST` | `/documents` | Upload and ingest a single PDF |
| `POST` | `/documents/bulk` | Bulk ingest all PDFs in `DATA_DIR` |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{filename}` | Remove a document and all its chunks |
| `POST` | `/supervisor/classify` | Classify message intent via LLM |
| `POST` | `/query` | Full agentic RAG query (non-streaming) |
| `POST` | `/query/stream` | Streaming RAG via Server-Sent Events |
| `POST` | `/evaluate` | RAGAS evaluation over test cases |

### Streaming response format

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum investment?"}'
```

Events (newline-delimited JSON):
```json
{"type": "trace",    "steps": ["[Router] Complexity: FAST", ...]}
{"type": "token",    "content": "The minimum investment is RM10.00 [1]."}
{"type": "citations","data": [{"chunk_id": 1, "page": 3, "source_file": "doc.pdf", "snippet": "..."}]}
{"type": "meta",     "rewritten_query": "minimum initial investment amount", "relevance_score": 8, "steps_taken": 3}
data: [DONE]
```

### Why no `/agent/orchestrate`

An earlier version exposed `/agent/orchestrate` with `rag_then_summarise`
and `rag_then_analyse` modes, framed as multi-agent orchestration. It
wasn't — it ran the same RAG pipeline and appended a single templated
LLM pass over the answer. The endpoint was removed in favour of writing
the desired output style into the question itself: ask "summarise the
fund's fee structure" and `/query` returns a summary. Adding fake
multi-agent surface area for the appearance of sophistication is the
exact pattern this project rejects.

---

## Retrieval Pipeline

```
Query
  ├─ Dense:  ChromaDB similarity_search      (top 5)
  └─ Sparse: BM25Okapi on all indexed texts  (top 3)
              ▼
         RRF fusion  score = Σ 1 / (60 + rank + 1)
              ▼
         CrossEncoder reranker (ms-marco-MiniLM-L-6-v2, local)
              ▼
         Top 4 chunks returned
         (each carries parent_context for the LLM answer prompt)
```

BM25 index persists to `data/bm25_persistent.pkl` and rebuilds automatically when document count changes (thread-safe).

---

## Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"custom_test_cases": null}'
```

Pass custom cases or use the built-in 13-question suite. Results append to `eval_history.jsonl`.

| Metric | What it measures | Target |
|---|---|---|
| Faithfulness | Answer grounded in retrieved context | > 0.80 |
| Answer Relevancy | Response addresses the question | > 0.80 |
| Context Precision | Retrieved chunks are relevant | > 0.70 |
| Context Recall | All needed information retrieved | > 0.70 |

---

## Running Tests

```bash
# Unit tests — no LLM, vector store, or network required
cd backend
pytest tests/ -v

# RAGAS quality evaluation (requires OPENAI_API_KEY or AWS credentials + ingested PDF)
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"custom_test_cases": null}'
```

The unit test suite covers:
- **Query complexity classification** — fast vs smart tier routing
- **Heuristic grading** — keyword overlap scoring (zero LLM cost)
- **Graph routing logic** — `_route_after_grade` deterministic decisions
- **Hallucination check** — fast-path bypass, confidence formula
- **Citation extraction** — `[N]` marker mapping to sentence-level evidence
- **BM25 retriever** — build, retrieve, persist, reload
- **Circuit breaker** — failure threshold, open/half-open/closed states
- **Metrics collector** — windowed statistics

---

## Project Structure

```
agentic-rag/
├── backend/
│   ├── main.py                        # FastAPI entry point (~60 lines, thin)
│   ├── core/
│   │   ├── config.py                  # Pydantic settings via os.getenv
│   │   └── exceptions.py             # Typed exception hierarchy
│   ├── agents/
│   │   ├── graph.py                   # LangGraph StateGraph (6 nodes)
│   │   ├── state.py                   # RAGState TypedDict
│   │   └── nodes/
│   │       ├── query_analysis.py      # Tier classification (0 tokens)
│   │       ├── retrieval.py           # Hybrid retrieval orchestration
│   │       ├── grading.py             # 2-tier grading (heuristic + Haiku)
│   │       ├── rewriting.py           # Query rewriting (3 strategies)
│   │       ├── generation.py          # Answer + sentence-level citations
│   │       └── hallucination.py       # Grounding check + confidence score
│   ├── retrieval/
│   │   ├── hybrid.py                  # BM25 + vector + RRF + reranking
│   │   ├── bm25_retriever.py          # Thread-safe BM25 with disk cache
│   │   ├── reranker.py                # CrossEncoder reranker (local)
│   │   └── vector_store.py            # ChromaDB singleton
│   ├── ingestion/
│   │   └── pipeline.py                # PDF → chunks → ChromaDB + BM25
│   ├── services/
│   │   ├── llm.py                     # Tiered LLM routing (Bedrock/OpenAI/Ollama)
│   │   └── embedding.py              # Embedding provider abstraction
│   ├── infrastructure/
│   │   ├── circuit_breaker.py         # Thread-safe circuit breaker
│   │   └── metrics.py                 # Windowed metrics collector
│   ├── models/
│   │   ├── domain.py                  # Citation, RetrievedChunk, RetrievalMetadata
│   │   ├── requests.py                # Pydantic v2 request models
│   │   └── responses.py              # Pydantic v2 response models
│   ├── api/routes/
│   │   ├── documents.py               # POST/GET/DELETE /documents
│   │   ├── query.py                   # POST /query, /query/stream, /supervisor/classify
│   │   ├── health.py                  # GET /health, /metrics
│   │   └── evaluation.py             # POST /evaluate
│   ├── evaluation/
│   │   └── ragas_eval.py              # RAGAS 4-metric evaluation harness
│   └── tests/
│       └── test_nodes.py              # Unit tests (no LLM required)
│
├── frontend/
│   ├── src/app/page.tsx               # Chat UI — streaming, citations, eval tab
│   └── src/lib/api.ts                 # Typed API client + SSE stream handler
│
└── data/                              # Drop PDFs here for /documents/bulk
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | AWS Bedrock (Claude 3.5 Haiku / 3.7 Sonnet) · OpenAI · Ollama |
| Embeddings | AWS Bedrock (Cohere embed-multilingual-v3) · OpenAI · Ollama |
| Vector DB | ChromaDB (persistent, local) |
| Sparse retrieval | rank-bm25 (BM25Okapi), persisted index |
| Reranker | sentence-transformers CrossEncoder (local) |
| Backend | FastAPI + SlowAPI + asyncio thread pool |
| Frontend | Next.js 14 · Tailwind CSS · lucide-react |
| Evaluation | RAGAS + HuggingFace Datasets |
| Orchestration | LangGraph StateGraph (declarative, visualisable, interruptible) |
