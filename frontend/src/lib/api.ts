/**
 * Typed API client for the Agentic RAG backend.
 *
 * All requests go through this module — no raw fetch() calls scattered
 * across components. Type errors surface at compile time, not runtime.
 *
 * The interfaces below were hand-written. To regenerate them from the
 * backend's live OpenAPI schema (single source of truth, no drift):
 *
 *     # backend running on http://localhost:8000:
 *     npm run gen:api
 *     # custom URL:
 *     BACKEND_URL=http://api.dev.example.com npm run gen:api
 *
 * Generated types land in ``src/lib/api-schema.d.ts`` and can be imported
 * piecemeal as call sites are migrated. The duplicate interfaces below
 * stay until each one has a codegen replacement adopted.
 */

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Citation {
  chunk_id: number;
  page: number;
  source_file: string;
  snippet: string;
  full_chunk: string;
  similarity_score: number | null;
}

export interface RetrievalMetadata {
  query_used: string;
  vector_results_count: number;
  bm25_results_count: number;
  fused_results_count: number;
  reranking_applied: boolean;
  retrieval_latency_ms: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];

  // Query processing
  original_query: string;
  rewritten_query: string;

  // Two raw quality signals — no composite. Backend deliberately stopped
  // emitting a confidence_score because its 0.6/0.4 weights were
  // uncalibrated. Compose your own from these two if you must.
  relevance_score: number;
  hallucination_score: number;

  // Execution trace
  steps_taken: number;
  reasoning_trace: string[];

  // Performance
  latency_ms: number;

  // Retrieval diagnostics
  retrieval_metadata: RetrievalMetadata | null;
}

export interface IngestResult {
  status: "success" | "skipped" | "error";
  file: string;
  chunks?: number;
  message?: string;
  error?: string;
  embedding_provider?: string;
}

export interface DeleteResult {
  status: "deleted" | "not_found";
  file: string;
  chunks_deleted: number;
}

export interface DocumentsResponse {
  documents: string[];
  count: number;
}

export interface SupervisorResult {
  intent: "question" | "summarise" | "analyse" | "list_docs" | "chitchat";
  reasoning: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  ollama_connected: boolean;
  ollama_models: string[];
  ingested_files: string[];
  features: string[];
}

export interface EvalMetrics {
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
}

export interface EvalThresholdFailure {
  metric: string;
  value: number | null;
  threshold: number;
}

export interface EvalResponse {
  status: string;
  metrics: EvalMetrics;
  interpretation: Record<keyof EvalMetrics, string>;
  total_questions: number;
  per_question: Array<{
    question: string;
    answer: string;
    relevance_score: number;
    chunks_retrieved: number;
  }>;
  // Pass/fail gate against EVAL_THRESHOLDS in backend/evaluation/ragas_eval.py.
  // ``passed`` is false if any metric fell below its declared threshold; the
  // failures list names which ones.
  passed: boolean;
  threshold_failures: EvalThresholdFailure[];
}

// SSE stream event types — must match agents/graph.py run_rag_streaming output
export type StreamEvent =
  // Live per-node progress while the LangGraph pipeline executes. The UI
  // renders these as the visible "thinking" trace before answer tokens
  // arrive, matching Claude / ChatGPT progressive-disclosure UX.
  | { type: "step";      node: string; message: string }
  // ``trace`` is the legacy bulk event kept for back-compat with any
  // client that consumes it; the new path uses ``step`` instead.
  | { type: "trace";     steps: string[] }
  | { type: "token";     content: string }
  | { type: "citations"; data: Citation[] }
  | { type: "meta";
      original_query: string;
      rewritten_query: string;
      relevance_score: number;
      hallucination_score: number;
      steps_taken: number;
      latency_ms: number;
    }
  | { type: "error";     message: string };

// ── Helpers ───────────────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${path}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

export const api = {
  health: (): Promise<HealthResponse> =>
    request("/health"),

  listDocuments: (): Promise<DocumentsResponse> =>
    request("/documents"),

  ingestFile: (file: File): Promise<IngestResult> => {
    const form = new FormData();
    form.append("file", file);
    // Route: POST /documents (prefix set in documents.py router)
    return request("/documents", { method: "POST", headers: {}, body: form });
  },

  ingestDirectory: (): Promise<{ total: number; results: IngestResult[] }> =>
    // Route: POST /documents/bulk
    request("/documents/bulk", { method: "POST" }),

  deleteDocument: (filename: string): Promise<DeleteResult> =>
    request(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" }),

  classifyIntent: (message: string): Promise<SupervisorResult> =>
    request("/supervisor/classify", {
      method: "POST",
      body: JSON.stringify({ message }),
    }),

  query: (question: string, showReasoning = true): Promise<QueryResponse> =>
    request("/query", {
      method: "POST",
      body: JSON.stringify({ question, show_reasoning: showReasoning }),
    }),

  evaluate: (customTestCases?: Array<{ question: string; ground_truth: string }>): Promise<EvalResponse> =>
    request("/evaluate", {
      method: "POST",
      body: JSON.stringify({ custom_test_cases: customTestCases ?? null }),
    }),

  /**
   * Open an SSE stream for a query. Calls onEvent for each parsed event.
   * Returns a cleanup function — call it to abort the stream.
   */
  queryStream: (
    question: string,
    onEvent: (event: StreamEvent) => void,
    onDone: () => void,
  ): (() => void) => {
    const controller = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${BASE_URL}/query/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, show_reasoning: true }),
          signal: controller.signal,
        });

        if (!res.body) throw new Error("No response body for streaming");

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let   buffer  = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed === "data: [DONE]") {
              if (trimmed === "data: [DONE]") onDone();
              continue;
            }
            const data = trimmed.startsWith("data: ") ? trimmed.slice(6) : trimmed;
            try {
              onEvent(JSON.parse(data) as StreamEvent);
            } catch {
              // Malformed JSON in stream — skip
            }
          }
        }
        onDone();
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== "AbortError") {
          onEvent({ type: "error", message: err.message });
          onDone();
        }
      }
    })();

    return () => controller.abort();
  },
};
