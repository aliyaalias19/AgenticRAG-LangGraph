"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  FileText, Trash2, Upload, AlertCircle, CheckCircle2,
  ChevronDown, ChevronUp, Send, RefreshCw, BookOpen, Activity,
  BarChart2, MessageSquare, FolderOpen,
} from "lucide-react";
import clsx from "clsx";
import {
  api,
  Citation,
  StreamEvent,
  SupervisorResult,
  HealthResponse,
} from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

type Intent = SupervisorResult["intent"];

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: Intent;
  citations?: Citation[];
  trace?: string[];
  rewrittenQuery?: string;
  relevanceScore?: number;
  hallucinationScore?: number;
  stepsToken?: number;
  elapsed?: number;
  supervisorReasoning?: string;
  streaming?: boolean;
}

type Tab = "chat" | "eval";

// ── Helpers ───────────────────────────────────────────────────────────────────

const INTENT_LABEL: Record<Intent, string> = {
  question:  "Question",
  summarise: "Summary",
  analyse:   "Analysis",
  list_docs: "Document list",
  chitchat:  "Conversation",
};

function uid() { return Math.random().toString(36).slice(2); }

function relevanceTone(score: number) {
  if (score >= 7) return { label: "high",   text: "text-emerald-700", bar: "bg-emerald-500" };
  if (score >= 4) return { label: "medium", text: "text-amber-700",   bar: "bg-amber-500" };
  return            { label: "low",    text: "text-rose-700",    bar: "bg-rose-500" };
}

function hScoreTone(h: number) {
  if (h <= 0.15) return "text-emerald-700 bg-emerald-50 border-emerald-200";
  if (h <= 0.40) return "text-amber-700 bg-amber-50 border-amber-200";
  return            "text-rose-700 bg-rose-50 border-rose-200";
}

// ── Reusable bits ─────────────────────────────────────────────────────────────

function Collapsible({
  title, children, defaultOpen = false, icon, count,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  icon?: React.ReactNode;
  count?: number;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-t border-slate-100 pt-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 text-slate-500 hover:text-slate-800 text-xs font-medium w-full"
      >
        {icon}
        <span>{title}</span>
        {count != null && (
          <span className="text-slate-400 font-mono">({count})</span>
        )}
        <span className="ml-auto text-slate-400">
          {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </span>
      </button>
      {open && <div className="mt-3">{children}</div>}
    </div>
  );
}

// Display-time cleanup of legacy snippet text. The ingest pipeline now
// strips these artefacts before chunking, but chunks ingested by older
// versions of the pipeline still carry the raw markdown noise. This
// keeps the UI clean without forcing the user to re-ingest documents.
function cleanSnippet(s: string): string {
  return s
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/\*{2}_(.+?)_\*{2}/g, "$1")
    .replace(/\*{3}(.+?)\*{3}/g, "$1")
    .replace(/\*{2}(.+?)\*{2}/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/(?<!\w)\*([^*\n]+?)\*(?!\w)/g, "$1")
    .replace(/(?<!\w)_([^_\n]+?)_(?!\w)/g, "$1")
    .replace(/�/g, "•")
    .replace(/\s+/g, " ")
    .trim();
}

function CitationCard({ c }: { c: Citation }) {
  return (
    <div className="flex gap-3 py-2.5 px-3 rounded-lg border border-slate-200 bg-slate-50/50 hover:bg-white transition-colors">
      <span className="flex-shrink-0 w-6 h-6 rounded bg-white border border-slate-200 text-[11px] font-mono font-semibold text-slate-700 flex items-center justify-center">
        {c.chunk_id}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-slate-800 truncate">{c.source_file}</span>
          <span className="flex-shrink-0 text-[10px] text-slate-500 font-mono">p.{c.page}</span>
        </div>
        <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">{cleanSnippet(c.snippet)}</p>
      </div>
    </div>
  );
}

function TraceLog({ steps }: { steps: string[] }) {
  return (
    <div className="rounded-lg bg-slate-950 px-4 py-3 max-h-56 overflow-y-auto">
      {steps.map((s, i) => (
        <div key={i} className="flex gap-3 font-mono text-[11px] leading-relaxed">
          <span className="text-slate-600 flex-shrink-0">{String(i + 1).padStart(2, "0")}</span>
          <span className="text-emerald-300">{s}</span>
        </div>
      ))}
    </div>
  );
}

function RelevanceMeter({ score }: { score: number }) {
  const tone = relevanceTone(score);
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1 bg-slate-200 rounded-full overflow-hidden">
        <div
          className={clsx("h-full transition-all", tone.bar)}
          style={{ width: `${(score / 10) * 100}%` }}
        />
      </div>
      <span className={clsx("text-[11px] font-mono font-semibold", tone.text)}>
        {score}/10
      </span>
    </div>
  );
}

// ── Live thinking trace (visible while streaming, before tokens arrive) ──────

function LiveThinking({ steps }: { steps: string[] }) {
  // Render the trace as a stack of pill-style step messages, latest at the
  // bottom. Auto-scrolling not needed — the list is short (≤ 10 nodes).
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/70 px-4 py-3 max-h-64 overflow-y-auto">
      <div className="flex items-center gap-2 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
        <span className="flex gap-1">
          {[0, 1, 2].map(i => (
            <span
              key={i}
              className="w-1 h-1 rounded-full bg-slate-400 animate-bounce"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </span>
        Thinking
      </div>
      <ol className="space-y-1.5 font-mono text-[11px] text-slate-600 leading-relaxed">
        {steps.map((s, i) => (
          <li key={i} className="flex gap-2 animate-fade-in">
            <span className="text-slate-400 flex-shrink-0">{String(i + 1).padStart(2, "0")}</span>
            <span className="break-words">{s}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

// ── Assistant message ─────────────────────────────────────────────────────────

function AssistantMessage({ msg }: { msg: ChatMessage }) {
  // While streaming AND no answer tokens yet, show the live thinking trace
  // inline. Once tokens start arriving, the trace collapses into the
  // normal "Reasoning trace" expandable section below.
  const showLiveThinking = msg.streaming && !msg.content && (msg.trace?.length ?? 0) > 0;

  return (
    <div className="animate-fade-in space-y-3 max-w-[760px]">
      {/* Intent + supervisor reasoning */}
      {msg.intent && (
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          <span className="font-medium text-slate-700">{INTENT_LABEL[msg.intent]}</span>
          {msg.supervisorReasoning && (
            <>
              <span className="text-slate-300">·</span>
              <span className="italic">{msg.supervisorReasoning}</span>
            </>
          )}
        </div>
      )}

      {/* Live thinking — replaces the answer area until tokens start arriving */}
      {showLiveThinking && <LiveThinking steps={msg.trace ?? []} />}

      {/* Answer */}
      {!showLiveThinking && (
        <div className="text-[15px] leading-relaxed text-slate-800 whitespace-pre-wrap">
          {msg.content}
          {msg.streaming && (
            <span className="inline-block w-[2px] h-[1em] bg-blue-500 ml-0.5 animate-pulse align-middle" />
          )}
        </div>
      )}

      {/* Metadata strip */}
      {(msg.rewrittenQuery || msg.relevanceScore != null
        || msg.hallucinationScore != null
        || msg.elapsed != null
        || msg.stepsToken != null) && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-slate-500">
          {msg.relevanceScore != null && (
            <div className="flex items-center gap-2">
              <span className="text-slate-400">retrieval</span>
              <RelevanceMeter score={msg.relevanceScore} />
            </div>
          )}
          {msg.hallucinationScore != null && (
            <span
              className={clsx(
                "px-2 py-0.5 rounded-full border text-[10px] font-mono font-semibold",
                hScoreTone(msg.hallucinationScore)
              )}
              title="Estimated fraction of claims not supported by retrieved context."
            >
              h-score {msg.hallucinationScore.toFixed(2)}
            </span>
          )}
          {msg.stepsToken != null && (
            <span className="flex items-center gap-1 text-slate-500">
              <Activity size={11} />
              {msg.stepsToken} step{msg.stepsToken !== 1 ? "s" : ""}
            </span>
          )}
          {msg.elapsed != null && (
            <span className="font-mono">{msg.elapsed.toFixed(1)}s</span>
          )}
          {msg.rewrittenQuery && msg.rewrittenQuery !== msg.content && (
            <span className="text-slate-400 italic truncate max-w-[280px]" title={msg.rewrittenQuery}>
              rewrote → {msg.rewrittenQuery}
            </span>
          )}
        </div>
      )}

      {/* Citations */}
      {msg.citations && msg.citations.length > 0 && (
        <Collapsible
          title="Sources"
          count={msg.citations.length}
          icon={<BookOpen size={12} />}
          defaultOpen
        >
          <div className="space-y-1.5">
            {msg.citations.map(c => <CitationCard key={c.chunk_id} c={c} />)}
          </div>
        </Collapsible>
      )}

      {/* Reasoning trace */}
      {msg.trace && msg.trace.length > 0 && (
        <Collapsible title="Reasoning trace" count={msg.trace.length}>
          <TraceLog steps={msg.trace} />
        </Collapsible>
      )}
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function Sidebar({
  documents, health, onIngest, onDelete, onIngestDir, loading,
}: {
  documents: string[];
  health: HealthResponse | null;
  onIngest: (f: File) => void;
  onDelete: (name: string) => void;
  onIngestDir: () => void;
  loading: boolean;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const handleDelete = async (name: string) => {
    setDeleting(name);
    await onDelete(name);
    setDeleting(null);
  };

  const providerLine = health?.features
    ?.filter(f => f.startsWith("llm_provider=") || f.startsWith("embedding_provider="))
    .map(f => f.replace("=", " → "))
    .join("  ·  ");

  return (
    <aside className="w-72 flex-shrink-0 bg-white border-r border-slate-200 flex flex-col">
      {/* Header */}
      <div className="px-5 pt-5 pb-4 border-b border-slate-200">
        <div className="flex items-baseline gap-2">
          <h1 className="text-sm font-semibold text-slate-900 tracking-tight">Agentic RAG</h1>
          <span className="text-[10px] font-mono text-slate-400">
            {health ? `v${health.version}` : ""}
          </span>
        </div>
        <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
          LangGraph · hybrid retrieval · cross-encoder rerank
        </p>
      </div>

      {/* Status */}
      <div className="px-5 py-3 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <span className={clsx(
            "w-1.5 h-1.5 rounded-full",
            health?.ollama_connected
              ? "bg-emerald-500"
              : health
                ? "bg-rose-500"
                : "bg-slate-300 animate-pulse"
          )} />
          <span className="text-[12px] text-slate-700">
            {health
              ? (health.ollama_connected ? "Ollama connected" : "Ollama offline")
              : "Connecting…"}
          </span>
        </div>
        {providerLine && (
          <p className="mt-1.5 text-[10px] font-mono text-slate-500 leading-relaxed">
            {providerLine}
          </p>
        )}
        {health?.ollama_models && health.ollama_models.length > 0 && (
          <p className="mt-1 text-[10px] font-mono text-slate-400 leading-relaxed truncate">
            {health.ollama_models.join(", ")}
          </p>
        )}
      </div>

      {/* Ingest controls */}
      <div className="px-5 py-4 border-b border-slate-200 space-y-2">
        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={e => {
            const f = e.target.files?.[0];
            if (f) onIngest(f);
            e.currentTarget.value = "";  // allow re-uploading the same file
          }}
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 bg-slate-900 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium py-2 px-3 rounded-md transition-colors"
        >
          {loading ? <RefreshCw size={12} className="animate-spin" /> : <Upload size={12} />}
          Upload PDF
        </button>
        <button
          onClick={onIngestDir}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 bg-white hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed text-slate-700 text-xs font-medium py-2 px-3 rounded-md border border-slate-200 transition-colors"
        >
          <FolderOpen size={12} />
          Ingest /data
        </button>
      </div>

      {/* Documents */}
      <div className="px-5 py-4 flex-1 overflow-y-auto">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
            Documents
          </span>
          <span className="text-[10px] font-mono text-slate-400">{documents.length}</span>
        </div>
        {documents.length === 0 ? (
          <p className="text-[11px] text-slate-400 leading-relaxed">
            No documents ingested.<br />
            Upload a PDF to begin.
          </p>
        ) : (
          <ul className="space-y-0.5">
            {documents.map(doc => (
              <li
                key={doc}
                className="group flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-slate-50 transition-colors"
              >
                <FileText size={11} className="text-slate-400 flex-shrink-0" />
                <span className="text-[12px] text-slate-700 truncate flex-1" title={doc}>
                  {doc}
                </span>
                <button
                  onClick={() => handleDelete(doc)}
                  disabled={deleting === doc}
                  className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-rose-600 transition-all"
                  title={`Delete ${doc}`}
                >
                  {deleting === doc
                    ? <RefreshCw size={11} className="animate-spin" />
                    : <Trash2 size={11} />}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Footer — pipeline at a glance, accurate */}
      <div className="px-5 py-4 border-t border-slate-200">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
          Pipeline
        </span>
        <ol className="mt-2 space-y-1 text-[11px] text-slate-600">
          <li>1. Query analysis</li>
          <li>2. Hybrid retrieval (BM25 + vector + RRF)</li>
          <li>3. Cross-encoder rerank</li>
          <li>4. Relevance grade (heuristic + LLM fallback)</li>
          <li>5. Rewrite or generate</li>
          <li>6. Hallucination check</li>
        </ol>
      </div>
    </aside>
  );
}

// ── Welcome / empty state ─────────────────────────────────────────────────────

function WelcomeScreen({ onPick, ready }: { onPick: (q: string) => void; ready: boolean }) {
  const suggestions = ready
    ? [
        "What is the minimum investment amount?",
        "Summarise the fee structure",
        "Are there any redemption fees?",
        "What does the prospectus say about risks?",
      ]
    : [];

  return (
    <div className="max-w-2xl mx-auto pt-20 px-6">
      <h2 className="text-2xl font-semibold text-slate-900 tracking-tight">
        Ask a question.
      </h2>
      <p className="text-slate-500 text-[15px] mt-2 leading-relaxed">
        {ready
          ? "Your documents are indexed. Pick a starter below, or write your own."
          : "Upload a PDF in the sidebar to get started."}
      </p>
      {suggestions.length > 0 && (
        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-2">
          {suggestions.map(s => (
            <button
              key={s}
              onClick={() => onPick(s)}
              className="text-left px-4 py-3 rounded-lg border border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50 text-[13px] text-slate-700 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [tab, setTab]                       = useState<Tab>("chat");
  const [messages, setMessages]             = useState<ChatMessage[]>([]);
  const [input, setInput]                   = useState("");
  const [busy, setBusy]                     = useState(false);
  const [documents, setDocuments]           = useState<string[]>([]);
  const [health, setHealth]                 = useState<HealthResponse | null>(null);
  const [sidebarLoading, setSidebarLoading] = useState(false);
  const [toast, setToast]                   = useState<{ text: string; ok: boolean } | null>(null);
  const bottomRef                           = useRef<HTMLDivElement>(null);
  const streamCleanup                       = useRef<(() => void) | null>(null);
  const textareaRef                         = useRef<HTMLTextAreaElement>(null);

  const refreshDocs = useCallback(async () => {
    try {
      const d = await api.listDocuments();
      setDocuments(d.documents);
    } catch { /* swallow */ }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
    refreshDocs();
  }, [refreshDocs]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const showToast = (text: string, ok: boolean) => {
    setToast({ text, ok });
    setTimeout(() => setToast(null), 4000);
  };

  // ── Document ops ──────────────────────────────────────────────────────────

  const handleIngest = async (file: File) => {
    setSidebarLoading(true);
    try {
      const res = await api.ingestFile(file);
      if (res.status === "success")      showToast(`${res.file} — ${res.chunks} chunks indexed`, true);
      else if (res.status === "skipped") showToast(`Already indexed: ${res.file}`, true);
      else                                showToast(`Error: ${res.error}`, false);
      await refreshDocs();
    } catch (e: unknown) {
      showToast(`Upload failed: ${(e as Error).message}`, false);
    } finally {
      setSidebarLoading(false);
    }
  };

  const handleDelete = async (name: string) => {
    try {
      const res = await api.deleteDocument(name);
      if (res.status === "deleted") showToast(`Deleted ${name}`, true);
      else                           showToast(`Not found: ${name}`, false);
      await refreshDocs();
    } catch (e: unknown) {
      showToast(`Delete failed: ${(e as Error).message}`, false);
    }
  };

  const handleIngestDir = async () => {
    setSidebarLoading(true);
    try {
      const res = await api.ingestDirectory();
      showToast(`Processed ${res.total} file(s)`, true);
      await refreshDocs();
    } catch (e: unknown) {
      showToast(`Ingest failed: ${(e as Error).message}`, false);
    } finally {
      setSidebarLoading(false);
    }
  };

  // ── Chat submit ───────────────────────────────────────────────────────────

  const submit = async (question: string) => {
    if (!question.trim() || busy) return;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setBusy(true);

    setMessages(prev => [...prev, { id: uid(), role: "user", content: question }]);
    const start = Date.now();

    try {
      const { intent, reasoning } = await api.classifyIntent(question);

      if (intent === "list_docs") {
        const docs = documents.length > 0
          ? `Loaded documents:\n${documents.map(d => `• ${d}`).join("\n")}`
          : "No documents loaded. Upload a PDF in the sidebar.";
        setMessages(prev => [...prev, {
          id: uid(), role: "assistant", content: docs, intent,
          elapsed: (Date.now() - start) / 1000,
          supervisorReasoning: reasoning,
        }]);
        setBusy(false);
        return;
      }

      if (intent === "chitchat" || documents.length === 0) {
        const content = documents.length === 0
          ? "No documents loaded yet. Please upload a PDF first."
          : "Hello — ask me anything about the loaded documents.";
        setMessages(prev => [...prev, {
          id: uid(), role: "assistant", content, intent: "chitchat",
          elapsed: (Date.now() - start) / 1000,
          supervisorReasoning: reasoning,
        }]);
        setBusy(false);
        return;
      }

      // Everything else streams through /query/stream — the graph is general
      // and answers in the style the question implies.
      const assistantId = uid();
      setMessages(prev => [...prev, {
        id: assistantId, role: "assistant", content: "", intent,
        streaming: true, supervisorReasoning: reasoning,
      }]);

      let fullAnswer         = "";
      let trace: string[]    = [];
      let citations: Citation[] = [];
      let rewrittenQuery     = "";
      let relevanceScore     = 0;
      let hallucinationScore = 0;
      let stepsToken         = 0;

      streamCleanup.current = api.queryStream(
        question,
        (event: StreamEvent) => {
          if (event.type === "step") {
            // Append the live step to the trace as each LangGraph node
            // finishes. Render flushes immediately so the user sees a
            // moving "thinking" trace, not a 30-second blank screen.
            trace = [...trace, event.message];
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, trace } : m
            ));
          } else if (event.type === "trace") {
            // Legacy bulk-trace event — overwrite (back-compat).
            trace = event.steps;
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, trace } : m
            ));
          } else if (event.type === "token") {
            fullAnswer += event.content;
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, content: fullAnswer, streaming: true } : m
            ));
          } else if (event.type === "citations") {
            citations = event.data;
          } else if (event.type === "meta") {
            rewrittenQuery     = event.rewritten_query;
            relevanceScore     = event.relevance_score;
            hallucinationScore = event.hallucination_score;
            stepsToken         = event.steps_taken;
          } else if (event.type === "error") {
            setMessages(prev => prev.map(m =>
              m.id === assistantId
                ? { ...m, content: `Error: ${event.message}`, streaming: false }
                : m
            ));
          }
        },
        () => {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? {
              ...m, streaming: false, citations, rewrittenQuery,
              relevanceScore, hallucinationScore, stepsToken,
              elapsed: (Date.now() - start) / 1000,
            } : m
          ));
          setBusy(false);
          streamCleanup.current = null;
        },
      );
    } catch (e: unknown) {
      setMessages(prev => [...prev, {
        id: uid(), role: "assistant",
        content: `Error: ${(e as Error).message}`,
      }]);
      setBusy(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(input); }
  };

  const handleTextareaInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const t = e.currentTarget;
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 160)}px`;
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen overflow-hidden bg-[#fafafa]">
      <Sidebar
        documents={documents}
        health={health}
        onIngest={handleIngest}
        onDelete={handleDelete}
        onIngestDir={handleIngestDir}
        loading={sidebarLoading}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center justify-between px-6 py-3 border-b border-slate-200 bg-white">
          <nav className="flex items-center gap-1">
            {([
              ["chat", <MessageSquare key="c" size={13} />, "Chat"],
              ["eval", <BarChart2     key="e" size={13} />, "Evaluation"],
            ] as [Tab, React.ReactNode, string][]).map(([t, icon, label]) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={clsx(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-medium transition-colors",
                  tab === t
                    ? "bg-slate-100 text-slate-900"
                    : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                )}
              >
                {icon}
                {label}
              </button>
            ))}
          </nav>
          {messages.length > 0 && tab === "chat" && (
            <button
              onClick={() => { streamCleanup.current?.(); setMessages([]); }}
              className="text-[12px] text-slate-500 hover:text-slate-800"
            >
              New conversation
            </button>
          )}
        </header>

        {/* Body */}
        {tab === "chat" ? (
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto">
              {messages.length === 0 ? (
                <WelcomeScreen onPick={submit} ready={documents.length > 0} />
              ) : (
                <div className="max-w-3xl mx-auto px-6 py-8 space-y-8">
                  {messages.map(msg => (
                    <div key={msg.id}>
                      {msg.role === "user" ? (
                        <div className="flex justify-end">
                          <div className="max-w-[85%] bg-slate-900 text-white rounded-2xl rounded-br-sm px-4 py-2.5 text-[14px] leading-relaxed">
                            {msg.content}
                          </div>
                        </div>
                      ) : (
                        <AssistantMessage msg={msg} />
                      )}
                    </div>
                  ))}
                  {busy && messages[messages.length - 1]?.role === "user" && (
                    <div className="flex items-center gap-2 text-slate-400 text-[13px]">
                      <div className="flex gap-1">
                        {[0, 1, 2].map(i => (
                          <span
                            key={i}
                            className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce"
                            style={{ animationDelay: `${i * 0.15}s` }}
                          />
                        ))}
                      </div>
                      Thinking…
                    </div>
                  )}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>

            {/* Composer */}
            <div className="border-t border-slate-200 bg-white px-6 py-4">
              <div className="max-w-3xl mx-auto">
                <div className="flex items-end gap-2 bg-white border border-slate-300 rounded-xl px-3 py-2 focus-within:border-slate-500 transition-colors">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onInput={handleTextareaInput}
                    placeholder={documents.length === 0
                      ? "Upload a document first…"
                      : "Ask anything about your documents…"}
                    disabled={busy}
                    rows={1}
                    className="flex-1 resize-none bg-transparent text-[14px] text-slate-800 placeholder:text-slate-400 focus:outline-none disabled:opacity-50 leading-relaxed py-1.5"
                    style={{ minHeight: "24px", maxHeight: "160px" }}
                  />
                  <button
                    onClick={() => submit(input)}
                    disabled={busy || !input.trim()}
                    className="flex-shrink-0 w-8 h-8 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-200 disabled:cursor-not-allowed text-white rounded-lg flex items-center justify-center transition-colors"
                    aria-label="Send"
                  >
                    {busy
                      ? <RefreshCw size={13} className="animate-spin" />
                      : <Send size={13} />}
                  </button>
                </div>
                <p className="text-[11px] text-slate-400 mt-2 text-center">
                  Enter to send · Shift+Enter for newline
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto px-6 py-8">
              <EvalTab documents={documents} />
            </div>
          </div>
        )}
      </main>

      {/* Toast */}
      {toast && (
        <div className={clsx(
          "fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-2.5 rounded-lg shadow-lg text-[13px] font-medium border bg-white",
          toast.ok
            ? "border-emerald-200 text-emerald-800"
            : "border-rose-200 text-rose-800"
        )}>
          {toast.ok
            ? <CheckCircle2 size={15} className="text-emerald-500" />
            : <AlertCircle  size={15} className="text-rose-500" />}
          {toast.text}
        </div>
      )}
    </div>
  );
}

// ── Evaluation tab ────────────────────────────────────────────────────────────

function EvalTab({ documents }: { documents: string[] }) {
  const [running, setRunning] = useState(false);
  const [result,  setResult]  = useState<Awaited<ReturnType<typeof api.evaluate>> | null>(null);
  const [error,   setError]   = useState<string | null>(null);

  const run = async () => {
    setRunning(true); setError(null); setResult(null);
    try {
      setResult(await api.evaluate());
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const METRIC_DESC: Record<string, string> = {
    faithfulness:      "Answer grounded in retrieved context (target ≥ 0.80)",
    answer_relevancy:  "Answer addresses the question (target ≥ 0.80)",
    context_precision: "Top-ranked chunks are relevant (target ≥ 0.70)",
    context_recall:    "Required info was retrieved (target ≥ 0.70)",
  };

  const tone = (v: number, target: number) =>
    v >= target ? "text-emerald-700" : v >= target - 0.1 ? "text-amber-700" : "text-rose-700";

  const targets: Record<string, number> = {
    faithfulness: 0.80, answer_relevancy: 0.80,
    context_precision: 0.70, context_recall: 0.70,
  };

  return (
    <div>
      <header className="mb-6">
        <h2 className="text-xl font-semibold text-slate-900 tracking-tight">RAGAS evaluation</h2>
        <p className="text-slate-500 text-[14px] mt-1 leading-relaxed">
          Four orthogonal quality metrics over a fixed test set. Thresholds are enforced —
          a run is marked <span className="font-mono">passed</span> only when every metric is at or above its target.
        </p>
      </header>

      <button
        onClick={run}
        disabled={running || documents.length === 0}
        className="inline-flex items-center gap-2 bg-slate-900 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-2 rounded-md text-[13px] font-medium transition-colors"
      >
        {running ? <><RefreshCw size={13} className="animate-spin" /> Running…</> : "Run evaluation"}
      </button>

      {documents.length === 0 && (
        <p className="mt-4 text-[13px] text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
          Ingest at least one document to run evaluation.
        </p>
      )}

      {error && (
        <p className="mt-4 text-[13px] text-rose-700 bg-rose-50 border border-rose-200 rounded-md px-3 py-2">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-8 space-y-8">
          {/* Pass / fail banner */}
          <div className={clsx(
            "rounded-lg px-4 py-3 border text-[13px]",
            result.passed
              ? "border-emerald-200 bg-emerald-50 text-emerald-800"
              : "border-rose-200 bg-rose-50 text-rose-800"
          )}>
            <span className="font-mono font-semibold">
              {result.passed ? "PASSED" : "FAILED"}
            </span>
            {!result.passed && result.threshold_failures.length > 0 && (
              <span className="ml-2">
                · below threshold: {result.threshold_failures.map(f => f.metric).join(", ")}
              </span>
            )}
            <span className="ml-2 text-slate-500">
              · {result.total_questions} questions
            </span>
          </div>

          {/* Score grid */}
          <div className="grid grid-cols-2 gap-3">
            {(Object.entries(result.metrics) as [keyof typeof result.metrics, number][]).map(([k, v]) => (
              <div key={k} className="bg-white border border-slate-200 rounded-lg p-4">
                <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">{k.replace(/_/g, " ")}</p>
                <div className="flex items-baseline gap-2 mt-1">
                  <span className={clsx("text-2xl font-mono font-semibold", tone(v, targets[k] ?? 0.7))}>
                    {(v * 100).toFixed(1)}%
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">
                    target {(targets[k] ?? 0.7) * 100}%
                  </span>
                </div>
                <div className="h-1 bg-slate-100 rounded-full overflow-hidden mt-3">
                  <div
                    className={clsx("h-full", v >= (targets[k] ?? 0.7) ? "bg-emerald-500" : "bg-rose-500")}
                    style={{ width: `${v * 100}%` }}
                  />
                </div>
                <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">{METRIC_DESC[k]}</p>
              </div>
            ))}
          </div>

          {/* Per-question */}
          <div>
            <h3 className="text-[13px] font-semibold text-slate-800 mb-3">Per question</h3>
            <div className="space-y-2">
              {result.per_question.map((q, i) => (
                <div key={i} className="bg-white border border-slate-200 rounded-lg p-4">
                  <p className="text-[13px] font-medium text-slate-800">{q.question}</p>
                  <p className="text-[12px] text-slate-500 leading-relaxed mt-1.5">
                    {q.answer.slice(0, 240)}{q.answer.length > 240 ? "…" : ""}
                  </p>
                  <div className="flex gap-4 mt-3 text-[11px] font-mono">
                    <span className="text-slate-500">relevance <span className="text-slate-800">{q.relevance_score}/10</span></span>
                    <span className="text-slate-500">chunks <span className="text-slate-800">{q.chunks_retrieved}</span></span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
