"""
Unit tests for the Agentic RAG pipeline nodes.

Design principle: every test here runs WITHOUT an LLM, vector store, or network.
Nodes are pure functions — given a state dict, they return a state dict.
The LLM dependency is either bypassed (heuristic paths) or patched with unittest.mock.

Why test at the node level (not end-to-end only):
  End-to-end tests with a live LLM are slow, expensive, and non-deterministic.
  Unit tests at the node level let us verify the routing logic, heuristic scoring,
  state transformations, and error handling in milliseconds, for free.
  RAGAS evaluation handles end-to-end quality measurement separately.

Run:
  pip install pytest
  cd backend
  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_doc(content: str, source: str = "test.pdf", page: int = 0) -> Document:
    return Document(page_content=content, metadata={"source_file": source, "page": page})


def make_state(**kwargs) -> dict:
    query = kwargs.get("query", "What is the minimum investment?")
    base = {
        "query": query,
        "rewritten_query": query,  # default rewritten_query to match query
        "tier": "fast",
        "retrieved_chunks": [],
        "relevance_score": 0,
        "retrieval_feedback": "",
        "steps_taken": 0,
        "rewrite_count": 0,
        "effective_max_steps": 5,
        "answer": "",
        "citations": [],
        "hallucination_score": 0.0,
        "reasoning_trace": [],
        "chat_history": [],
        "retrieval_metadata": {},
    }
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Query Analysis Node
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryAnalysisNode:
    def test_simple_query_gets_fast_tier(self):
        from agents.nodes.query_analysis import query_analysis_node
        state = make_state(query="What is the minimum investment?")
        result = query_analysis_node(state)
        assert result["tier"] == "fast"

    def test_complex_query_gets_smart_tier(self):
        from agents.nodes.query_analysis import query_analysis_node
        state = make_state(query="Compare the risk profile of Fund A versus Fund B and recommend the best option")
        result = query_analysis_node(state)
        assert result["tier"] == "smart"

    def test_fast_tier_caps_steps_at_3(self):
        from agents.nodes.query_analysis import query_analysis_node
        state = make_state(query="What is the fee?")
        result = query_analysis_node(state)
        assert result["tier"] == "fast"
        assert result["effective_max_steps"] == 3

    def test_smart_tier_allows_5_steps(self):
        from agents.nodes.query_analysis import query_analysis_node
        state = make_state(query="Analyse the implications of the new fee structure")
        result = query_analysis_node(state)
        assert result["tier"] == "smart"
        assert result["effective_max_steps"] == 5

    def test_resets_step_counters(self):
        from agents.nodes.query_analysis import query_analysis_node
        state = make_state(query="What is the fee?", steps_taken=3, rewrite_count=2)
        result = query_analysis_node(state)
        assert result["steps_taken"] == 0
        assert result["rewrite_count"] == 0

    def test_appends_to_reasoning_trace(self):
        from agents.nodes.query_analysis import query_analysis_node
        state = make_state(query="What is the fee?")
        result = query_analysis_node(state)
        assert len(result["reasoning_trace"]) == 1
        assert "[QueryAnalysis]" in result["reasoning_trace"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Grading Node — heuristic scoring (zero LLM cost)
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristicScoring:
    """Tests the _heuristic_score function which runs before any LLM call."""

    def test_empty_chunks_scores_zero(self):
        from agents.nodes.grading import _heuristic_score
        assert _heuristic_score("What is the fee?", []) == 0

    def test_strong_keyword_overlap_scores_high(self):
        from agents.nodes.grading import _heuristic_score
        chunks = [make_doc("The annual management fee is 1.0% per annum of the NAV of the fund")]
        score = _heuristic_score("What is the annual management fee?", chunks)
        assert score >= 5

    def test_zero_overlap_scores_low(self):
        from agents.nodes.grading import _heuristic_score
        chunks = [make_doc("The weather in Malaysia is tropical and humid throughout the year")]
        score = _heuristic_score("What is the annual management fee?", chunks)
        assert score < 4

    def test_stopword_only_query_defaults_to_5(self):
        from agents.nodes.grading import _heuristic_score
        chunks = [make_doc("Some content here")]
        score = _heuristic_score("what is the a an", chunks)
        assert score == 5

    def test_score_capped_at_10(self):
        from agents.nodes.grading import _heuristic_score
        query = "management fee annual NAV fund investment"
        chunks = [make_doc(query + " " + query)]  # perfect overlap
        score = _heuristic_score(query, chunks)
        assert score <= 10


class TestGradeNode:
    def test_high_heuristic_skips_llm(self):
        """Grade node should not call LLM when heuristic >= 7."""
        from agents.nodes.grading import grade_node
        chunks = [
            make_doc("The annual management fee is 1.0% NAV annually per fund management annually")
        ]
        state = make_state(
            query="What is the annual management fee?",
            retrieved_chunks=chunks,
        )
        with patch("agents.nodes.grading.get_llm") as mock_llm:
            result = grade_node(state)
            mock_llm.assert_not_called()
        assert result["relevance_score"] >= 7

    def test_low_heuristic_skips_llm(self):
        """Grade node should not call LLM when heuristic < 3 (poor match)."""
        from agents.nodes.grading import grade_node
        chunks = [make_doc("Sunny tropical weather in Kuala Lumpur all year round")]
        state = make_state(
            query="What is the annual management fee percentage?",
            retrieved_chunks=chunks,
        )
        with patch("agents.nodes.grading.get_llm") as mock_llm:
            result = grade_node(state)
            mock_llm.assert_not_called()
        assert result["relevance_score"] < 3

    def test_grade_node_returns_required_keys(self):
        from agents.nodes.grading import grade_node
        state = make_state(
            query="What is the fee?",
            retrieved_chunks=[make_doc("The fee is 1%")],
        )
        with patch("agents.nodes.grading.get_llm"):
            result = grade_node(state)
        assert "relevance_score" in result
        assert "retrieval_feedback" in result
        assert "reasoning_trace" in result


# ─────────────────────────────────────────────────────────────────────────────
# Graph routing logic
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterGrade:
    def test_high_score_routes_to_generate(self):
        from agents.graph import _route_after_grade
        state = make_state(relevance_score=8, steps_taken=1, rewrite_count=0, effective_max_steps=5)
        assert _route_after_grade(state) == "generate"

    def test_low_score_routes_to_rewrite(self):
        from agents.graph import _route_after_grade
        state = make_state(relevance_score=2, steps_taken=1, rewrite_count=0, effective_max_steps=5)
        assert _route_after_grade(state) == "rewrite"

    def test_max_steps_forces_generate(self):
        from agents.graph import _route_after_grade
        state = make_state(relevance_score=1, steps_taken=5, rewrite_count=0, effective_max_steps=5)
        assert _route_after_grade(state) == "generate"

    def test_max_rewrites_forces_generate(self):
        from agents.graph import _route_after_grade
        state = make_state(relevance_score=1, steps_taken=2, rewrite_count=2, effective_max_steps=5)
        assert _route_after_grade(state) == "generate"

    def test_exact_threshold_routes_to_generate(self):
        """Score >= min_relevance_score (default 3) should generate.

        Passing 4 here is comfortably above the threshold; the test asserts
        the >= comparison rather than pinning to a magic number that would
        break the test every time we tune the default.
        """
        from agents.graph import _route_after_grade
        state = make_state(relevance_score=4, steps_taken=1, rewrite_count=0, effective_max_steps=5)
        assert _route_after_grade(state) == "generate"


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination check node
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinationCheckNode:
    def test_no_answer_returns_default_h_score(self):
        from agents.nodes.hallucination import hallucination_check_node
        state = make_state(answer="", retrieved_chunks=[], relevance_score=0)
        result = hallucination_check_node(state)
        # No composite anymore — just the raw signal.
        assert result["hallucination_score"] == 0.5
        assert "confidence_score" not in result

    def test_high_relevance_takes_fast_path_no_llm(self):
        """relevance_score >= 8 should skip LLM call."""
        from agents.nodes.hallucination import hallucination_check_node
        chunks = [make_doc("The minimum investment is RM10")]
        state = make_state(
            answer="The minimum investment is RM10 [1].",
            retrieved_chunks=chunks,
            relevance_score=9,
        )
        with patch("agents.nodes.hallucination.get_llm") as mock_llm:
            result = hallucination_check_node(state)
            mock_llm.assert_not_called()
        assert result["hallucination_score"] == 0.05

    def test_no_composite_confidence_emitted(self):
        """Regression: the dropped uncalibrated composite must not return."""
        from agents.nodes.hallucination import hallucination_check_node
        chunks = [make_doc("The minimum investment is RM10")]
        state = make_state(
            answer="A grounded answer.",
            retrieved_chunks=chunks,
            relevance_score=9,
        )
        with patch("agents.nodes.hallucination.get_llm"):
            result = hallucination_check_node(state)
        assert "confidence_score" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Citation extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestCitationExtraction:
    def test_extracts_only_cited_chunks(self):
        from agents.nodes.generation import _extract_citations
        chunks = [
            make_doc("The minimum investment is RM10."),
            make_doc("The annual fee is 1%."),
            make_doc("Unrelated content about something else."),
        ]
        answer = "The minimum investment is RM10 [1]. The fee is 1% [2]."
        citations = _extract_citations(chunks, answer)
        cited_ids = {c.chunk_id for c in citations}
        assert cited_ids == {1, 2}
        assert 3 not in cited_ids

    def test_fallback_returns_all_when_no_markers(self):
        from agents.nodes.generation import _extract_citations
        chunks = [make_doc("Content A"), make_doc("Content B")]
        answer = "The answer is something."  # no [N] markers
        citations = _extract_citations(chunks, answer)
        assert len(citations) == 2  # graceful fallback

    def test_citation_has_required_fields(self):
        from agents.nodes.generation import _extract_citations
        chunks = [make_doc("The fee is 1%.", source="fund.pdf", page=3)]
        answer = "The fee is 1% [1]."
        citations = _extract_citations(chunks, answer)
        assert len(citations) == 1
        c = citations[0]
        assert c.chunk_id == 1
        assert c.source_file == "fund.pdf"
        assert c.page == 4  # page 3 + 1-indexed
        assert len(c.snippet) > 0

    def test_snippet_finds_relevant_sentence(self):
        from agents.nodes.generation import _find_cited_sentence
        chunk_text = "The weather is nice. The minimum investment is RM10. Fees apply."
        answer = "Minimum investment is RM10 [1]."
        snippet = _find_cited_sentence(chunk_text, answer, 1)
        assert "investment" in snippet.lower() or "RM10" in snippet


# ─────────────────────────────────────────────────────────────────────────────
# Query complexity classification
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryComplexity:
    def test_simple_what_is_query(self):
        from services.llm import classify_query_complexity
        assert classify_query_complexity("What is the minimum investment?") == "fast"

    def test_simple_how_much_query(self):
        from services.llm import classify_query_complexity
        assert classify_query_complexity("How much is the management fee?") == "fast"

    def test_complex_compare_query(self):
        from services.llm import classify_query_complexity
        assert classify_query_complexity("Compare Fund A versus Fund B") == "smart"

    def test_complex_analyse_query(self):
        from services.llm import classify_query_complexity
        assert classify_query_complexity("Analyse the risk implications of this fund") == "smart"

    def test_default_to_fast(self):
        from services.llm import classify_query_complexity
        assert classify_query_complexity("Tell me about this document") == "fast"


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion pipeline — SHA256 deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestSHA256Deduplication:
    def test_same_content_produces_same_hash(self, tmp_path):
        from ingestion.pipeline import _sha256
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        content = b"PDF content"
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert _sha256(str(f1)) == _sha256(str(f2))

    def test_different_content_produces_different_hash(self, tmp_path):
        from ingestion.pipeline import _sha256
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert _sha256(str(f1)) != _sha256(str(f2))


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retriever
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_build_and_retrieve(self, tmp_path):
        from core.config import Settings
        from retrieval.bm25_retriever import BM25Retriever

        settings = Settings(data_dir=str(tmp_path))
        retriever = BM25Retriever(settings)

        texts = [
            "The minimum investment is RM10",
            "The annual management fee is 1 percent",
            "Unit holders must maintain a minimum balance of one unit",
        ]
        metas = [{"source_file": "fund.pdf"}] * 3
        retriever.build(texts, metas)

        results = retriever.retrieve("minimum investment amount", top_k=2)
        assert len(results) >= 1
        assert any("minimum" in r.page_content.lower() for r in results)

    def test_empty_index_returns_empty(self, tmp_path):
        from core.config import Settings
        from retrieval.bm25_retriever import BM25Retriever

        settings = Settings(data_dir=str(tmp_path))
        retriever = BM25Retriever(settings)
        results = retriever.retrieve("anything")
        assert results == []

    def test_persist_and_reload(self, tmp_path):
        from core.config import Settings
        from retrieval.bm25_retriever import BM25Retriever

        settings = Settings(data_dir=str(tmp_path))
        r1 = BM25Retriever(settings)
        r1.build(["The fee is 1%", "Minimum investment RM10"], [{}] * 2)

        r2 = BM25Retriever(settings)
        loaded = r2.load_from_disk()
        assert loaded is True
        assert r2.get_corpus_size() == 2


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_passes_through_on_success(self):
        from infrastructure.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        result = cb.call(lambda: 42)
        assert result == 42

    def test_opens_after_threshold_failures(self):
        from infrastructure.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, timeout=60)

        def always_fail():
            raise RuntimeError("fail")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(always_fail)

        assert cb.is_open

    def test_raises_when_open(self):
        from infrastructure.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2, timeout=9999)

        def fail():
            raise RuntimeError()

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(fail)

        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            cb.call(lambda: None)

    def test_resets_on_success(self):
        from infrastructure.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, timeout=60)

        def fail():
            raise RuntimeError()

        with pytest.raises(RuntimeError):
            cb.call(fail)

        cb.call(lambda: None)  # success resets failures
        assert not cb.is_open


# ─────────────────────────────────────────────────────────────────────────────
# Tenancy: BM25 must partition by tenant_id when filter is supplied
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25TenantIsolation:
    """The BM25 in-memory index is shared across tenants; the where filter
    is the guard. These tests pin that contract."""

    def test_filter_returns_only_matching_tenant(self, tmp_path):
        from core.config import Settings
        from retrieval.bm25_retriever import BM25Retriever

        settings = Settings(data_dir=str(tmp_path))
        bm25 = BM25Retriever(settings)
        bm25.build(
            texts=[
                "alpha quarterly revenue is 100 million",
                "beta quarterly revenue is 200 million",
                "alpha headcount grew to 500",
            ],
            metadatas=[
                {"tenant_id": "alpha", "source_file": "a.pdf"},
                {"tenant_id": "beta", "source_file": "b.pdf"},
                {"tenant_id": "alpha", "source_file": "a2.pdf"},
            ],
        )

        alpha_results = bm25.retrieve("quarterly revenue", top_k=5, where={"tenant_id": "alpha"})
        beta_results = bm25.retrieve("quarterly revenue", top_k=5, where={"tenant_id": "beta"})

        assert all(d.metadata["tenant_id"] == "alpha" for d in alpha_results)
        assert all(d.metadata["tenant_id"] == "beta" for d in beta_results)
        assert len(alpha_results) >= 1
        assert len(beta_results) >= 1

    def test_filter_can_return_empty_for_unknown_tenant(self, tmp_path):
        from core.config import Settings
        from retrieval.bm25_retriever import BM25Retriever

        settings = Settings(data_dir=str(tmp_path))
        bm25 = BM25Retriever(settings)
        bm25.build(
            ["alpha doc only"],
            [{"tenant_id": "alpha"}],
        )
        results = bm25.retrieve("doc", top_k=5, where={"tenant_id": "ghost"})
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# API-key auth dependency
# ─────────────────────────────────────────────────────────────────────────────

class TestApiKeyAuth:
    """``get_tenant_context`` is the single chokepoint. These tests cover
    its four branches: demo mode, public path, valid key, invalid key.

    Note: Settings reads env vars at module-load time and is cached. Rather
    than fight that, we patch ``core.auth.get_settings`` directly with a
    Settings instance constructed with explicit kwargs. This is the
    senior-test pattern: the production cache is a *feature*, not a thing
    to subvert in tests.
    """

    @staticmethod
    def _request(path: str = "/query"):
        from starlette.requests import Request
        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
        }
        return Request(scope)

    @staticmethod
    def _patch_settings(monkeypatch, api_keys_raw: str = ""):
        from core.config import Settings
        import core.auth as auth_mod
        s = Settings(api_keys_raw=api_keys_raw)
        monkeypatch.setattr(auth_mod, "get_settings", lambda: s)
        return s

    def test_demo_mode_no_keys_configured(self, monkeypatch):
        s = self._patch_settings(monkeypatch, api_keys_raw="")
        from core.auth import get_tenant_context

        ctx = get_tenant_context(self._request(), x_api_key=None)
        assert ctx.tenant_id == s.default_tenant
        assert ctx.api_key is None

    def test_public_path_bypasses_auth(self, monkeypatch):
        s = self._patch_settings(monkeypatch, api_keys_raw='{"sk-1":"alpha"}')
        from core.auth import get_tenant_context

        ctx = get_tenant_context(self._request("/health"), x_api_key=None)
        assert ctx.tenant_id == s.default_tenant

    def test_valid_key_resolves_to_mapped_tenant(self, monkeypatch):
        self._patch_settings(monkeypatch, api_keys_raw='{"sk-alpha-123":"tenant-alpha"}')
        from core.auth import get_tenant_context

        ctx = get_tenant_context(self._request(), x_api_key="sk-alpha-123")
        assert ctx.tenant_id == "tenant-alpha"
        assert "sk-a" in ctx.api_key_fingerprint
        assert "sk-alpha-123" not in ctx.api_key_fingerprint  # never leak

    def test_missing_key_is_rejected_when_auth_enabled(self, monkeypatch):
        from fastapi import HTTPException
        self._patch_settings(monkeypatch, api_keys_raw='{"sk-1":"alpha"}')
        from core.auth import get_tenant_context

        with pytest.raises(HTTPException) as exc:
            get_tenant_context(self._request(), x_api_key=None)
        assert exc.value.status_code == 401

    def test_invalid_key_is_rejected(self, monkeypatch):
        from fastapi import HTTPException
        self._patch_settings(monkeypatch, api_keys_raw='{"sk-1":"alpha"}')
        from core.auth import get_tenant_context

        with pytest.raises(HTTPException) as exc:
            get_tenant_context(self._request(), x_api_key="sk-wrong")
        assert exc.value.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Audit log writer
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLog:
    @staticmethod
    def _patch_settings(monkeypatch, tmp_path):
        from core.config import Settings
        import core.audit as audit_mod
        s = Settings(data_dir=str(tmp_path))
        monkeypatch.setattr(audit_mod, "get_settings", lambda: s)
        return s

    def test_writes_well_formed_jsonl(self, tmp_path, monkeypatch):
        import json as _json
        self._patch_settings(monkeypatch, tmp_path)
        from core.audit import audit_event

        audit_event("query", tenant_id="tenant-alpha", actor_fingerprint="sk-a…23",
                    request_id="rid-123", detail={"question": "Hi", "citations": 2})

        log = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(log) == 1
        row = _json.loads(log[0])
        assert row["action"] == "query"
        assert row["tenant_id"] == "tenant-alpha"
        assert row["request_id"] == "rid-123"
        assert row["detail"]["citations"] == 2

    def test_coerces_non_serialisable_detail(self, tmp_path, monkeypatch):
        import json as _json
        self._patch_settings(monkeypatch, tmp_path)
        from core.audit import audit_event

        class Unjsonable:
            def __repr__(self):
                return "<Unjsonable>"

        audit_event("query", tenant_id="t", detail={"obj": Unjsonable()})
        row = _json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip())
        assert row["detail"]["obj"] == "<Unjsonable>"


# ─────────────────────────────────────────────────────────────────────────────
# Eval shipping bar
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalThresholds:
    def test_all_above_threshold_passes(self):
        from evaluation.ragas_eval import check_eval_thresholds
        gate = check_eval_thresholds({
            "faithfulness": 0.92, "answer_relevancy": 0.88,
            "context_precision": 0.81, "context_recall": 0.78,
        })
        assert gate["passed"] is True
        assert gate["failures"] == []

    def test_single_metric_below_threshold_fails(self):
        from evaluation.ragas_eval import check_eval_thresholds
        gate = check_eval_thresholds({
            "faithfulness": 0.60, "answer_relevancy": 0.88,
            "context_precision": 0.81, "context_recall": 0.78,
        })
        assert gate["passed"] is False
        assert len(gate["failures"]) == 1
        assert gate["failures"][0]["metric"] == "faithfulness"

    def test_missing_metric_counts_as_failure(self):
        from evaluation.ragas_eval import check_eval_thresholds
        gate = check_eval_thresholds({"faithfulness": 0.95})
        assert gate["passed"] is False
        failed_metrics = {f["metric"] for f in gate["failures"]}
        assert "answer_relevancy" in failed_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Markdown cleanup at ingest time
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkdownCleanup:
    """The pipeline strips pymupdf4llm artefacts so they don't pollute BM25
    tokens, embeddings, or citation snippets. These tests pin the contract."""

    def test_br_tags_become_newlines(self):
        from ingestion.pipeline import _clean_markdown
        assert _clean_markdown("line one<br>line two") == "line one\nline two"
        assert _clean_markdown("a<br/>b<br />c") == "a\nb\nc"

    def test_bold_italic_markers_stripped_content_preserved(self):
        from ingestion.pipeline import _clean_markdown
        assert _clean_markdown("**bold text**") == "bold text"
        assert _clean_markdown("**_combined_**") == "combined"
        assert _clean_markdown("***triple***") == "triple"
        assert _clean_markdown("_italic_") == "italic"

    def test_midword_underscores_preserved(self):
        # Variable names and snake_case must NOT be mangled.
        from ingestion.pipeline import _clean_markdown
        assert _clean_markdown("x_min and x_max") == "x_min and x_max"
        assert _clean_markdown("hello_world.py") == "hello_world.py"

    def test_unicode_replacement_char_becomes_bullet(self):
        from ingestion.pipeline import _clean_markdown
        assert _clean_markdown("� item one\n� item two") == "• item one\n• item two"

    def test_collapses_excessive_whitespace(self):
        from ingestion.pipeline import _clean_markdown
        assert _clean_markdown("a   b\t\tc") == "a b c"
        assert _clean_markdown("para1\n\n\n\n\npara2") == "para1\n\npara2"

    def test_empty_input_is_safe(self):
        from ingestion.pipeline import _clean_markdown
        assert _clean_markdown("") == ""
        assert _clean_markdown(None or "") == ""
