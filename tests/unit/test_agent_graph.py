"""Tests for the agentic retrieval graph and its nodes."""

import json
from dataclasses import dataclass, field

import pytest

from agentic_rag.agent.graph import AgentRunner, is_answered, trace_summary
from agentic_rag.agent.nodes import GraphDeps, should_rewrite
from agentic_rag.agent.prompts import ABSTENTION_MESSAGE, UNSUPPORTED_MESSAGE
from agentic_rag.agent.state import QueryType, Termination, Verdict, initial_state
from agentic_rag.config.settings import Settings
from agentic_rag.llm.provider import EchoProvider
from agentic_rag.retrieval.store import SearchHit
from agentic_rag.security.tenancy import TenantContext

TENANT = TenantContext(tenant_id="public", key_fingerprint="test")


def hit(chunk_id: str, content: str = "Draining evicts pods.", **kw: object) -> SearchHit:
    defaults = {
        "doc_id": f"kubernetes:{chunk_id.split('#')[0]}",
        "doc_title": "Safely Drain a Node",
        "heading_path": ("Overview",),
        "section": "tasks",
        "language": "en",
    }
    defaults.update(kw)
    return SearchHit(chunk_id=chunk_id, score=1.0, content=content, **defaults)  # type: ignore[arg-type]


@dataclass
class FakeRetriever:
    """Returns a scripted result list per call, so loops can be exercised."""

    results: list[list[SearchHit]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    _position: int = 0

    def retrieve(self, query, config, context=None, language=None):
        del config, context, language
        self.queries.append(query)
        if self._position < len(self.results):
            out = self.results[self._position]
            self._position += 1
            return out
        return self.results[-1] if self.results else []


def analysis(query: str, query_type: str = "factual") -> str:
    return json.dumps({"query_type": query_type, "search_query": query, "notes": "n"})


def grades(*scores: float) -> str:
    return json.dumps(
        {
            "scores": [
                {"passage": i, "score": s, "reason": "r"} for i, s in enumerate(scores, start=1)
            ]
        }
    )


def rewrite(query: str) -> str:
    return json.dumps({"search_query": query, "rationale": "different angle"})


def verification(score: float) -> str:
    return json.dumps({"faithfulness": score, "unsupported_claims": [], "notes": "ok"})


def runner(responses: list[str], results: list[list[SearchHit]], **overrides: object):
    settings = Settings()
    for key, value in overrides.items():
        setattr(settings.agent, key, value)
    retriever = FakeRetriever(results=results)
    provider = EchoProvider(responses, settings=settings)
    deps = GraphDeps(provider=provider, retriever=retriever, settings=settings)
    return AgentRunner(deps), provider, retriever


class TestHappyPath:
    def test_answers_when_context_is_relevant(self) -> None:
        agent, _, _ = runner(
            [
                analysis("drain node"),
                grades(0.9, 0.8),
                "Use kubectl drain [1]. It evicts pods [2].",
                verification(0.95),
            ],
            [[hit("a#0"), hit("b#0")]],
        )
        state = agent.run("how do I drain a node", TENANT, "req-1")

        assert state["termination"] == Termination.ANSWERED
        assert state["verdict"] == Verdict.SUPPORTED
        assert state["rewrite_count"] == 0
        assert not state["suppressed"]
        assert "kubectl drain" in state["answer"]
        assert is_answered(state)

    def test_records_a_step_for_every_node(self) -> None:
        agent, _, _ = runner(
            [analysis("q"), grades(0.9), "answer [1]", verification(0.9)],
            [[hit("a#0")]],
        )
        state = agent.run("q", TENANT, "req-2")
        nodes = [s["node"] for s in trace_summary(state)]

        assert nodes == [
            "analyse_query",
            "retrieve",
            "grade_relevance",
            "rerank",
            "generate",
            "verify",
        ]

    def test_citations_follow_the_context(self) -> None:
        agent, _, _ = runner(
            [analysis("q"), grades(0.9, 0.85), "a [1][2]", verification(0.9)],
            [[hit("a#0"), hit("b#0", doc_title="Node Management")]],
        )
        state = agent.run("q", TENANT, "req-3")
        assert state["citations"] == [
            "Safely Drain a Node > Overview",
            "Node Management > Overview",
        ]

    def test_query_type_is_parsed(self) -> None:
        agent, _, _ = runner(
            [
                analysis("pending pods scheduling", "troubleshooting"),
                grades(0.9),
                "a [1]",
                verification(0.9),
            ],
            [[hit("a#0")]],
        )
        state = agent.run("my pods are stuck", TENANT, "req-4")
        assert state["query_type"] == QueryType.TROUBLESHOOTING


class TestSelfCorrectionLoop:
    def test_low_relevance_triggers_a_rewrite(self) -> None:
        agent, _, retriever = runner(
            [
                analysis("vague query"),
                grades(0.1, 0.2),
                rewrite("kubectl drain node eviction"),
                grades(0.9, 0.85),
                "answer [1]",
                verification(0.9),
            ],
            [[hit("x#0"), hit("y#0")], [hit("a#0"), hit("b#0")]],
        )
        state = agent.run("how do I take a node offline", TENANT, "req-5")

        assert state["rewrite_count"] == 1
        assert state["termination"] == Termination.ANSWERED
        assert retriever.queries == ["vague query", "kubectl drain node eviction"]

    def test_rewrite_budget_is_enforced(self) -> None:
        agent, _, retriever = runner(
            [
                analysis("q0"),
                grades(0.1),
                rewrite("q1"),
                grades(0.1),
                rewrite("q2"),
                grades(0.1),
            ],
            [[hit("x#0")]],
            max_rewrites=2,
        )
        state = agent.run("unanswerable question", TENANT, "req-6")

        assert state["rewrite_count"] == 2
        assert state["termination"] == Termination.REWRITE_BUDGET_EXHAUSTED
        assert state["answer"] == ABSTENTION_MESSAGE
        assert len(retriever.queries) == 3

    def test_zero_budget_never_rewrites(self) -> None:
        agent, _, retriever = runner([analysis("q0"), grades(0.1)], [[hit("x#0")]], max_rewrites=0)
        state = agent.run("q", TENANT, "req-7")

        assert state["rewrite_count"] == 0
        assert state["termination"] == Termination.REWRITE_BUDGET_EXHAUSTED
        assert len(retriever.queries) == 1

    def test_empty_retrieval_triggers_a_rewrite(self) -> None:
        """Grading short-circuits on an empty candidate list, saving a call."""
        agent, provider, _ = runner(
            [
                analysis("q0"),
                rewrite("q1"),
                grades(0.9),
                "answer [1]",
                verification(0.9),
            ],
            [[], [hit("a#0")]],
        )
        state = agent.run("q", TENANT, "req-8")

        assert state["rewrite_count"] == 1
        assert state["termination"] == Termination.ANSWERED
        # analyse, rewrite, grade, generate, verify -- no grading call on the
        # empty first round.
        assert provider.usage.calls == 5

    def test_repeated_rewrite_is_replaced(self) -> None:
        """A rewrite echoing an earlier query would loop without progress."""
        agent, _, retriever = runner(
            [analysis("q0"), grades(0.1), rewrite("q0"), grades(0.1)],
            [[hit("x#0")]],
            max_rewrites=1,
        )
        agent.run("original question", TENANT, "req-9")
        assert retriever.queries[1] != retriever.queries[0]

    def test_graph_terminates_under_pathological_grading(self) -> None:
        """Every grade is zero; the graph must still halt."""
        agent, _, _ = runner(
            [analysis("q"), grades(0.0), rewrite("q1"), grades(0.0), rewrite("q2"), grades(0.0)],
            [[hit("x#0")]],
            max_rewrites=2,
        )
        state = agent.run("q", TENANT, "req-10")
        assert state["termination"] == Termination.REWRITE_BUDGET_EXHAUSTED


class TestEvidenceGating:
    def test_unsupported_answer_is_suppressed(self) -> None:
        agent, _, _ = runner(
            [
                analysis("q"),
                grades(0.9),
                "Kubernetes was released in 1847 by Napoleon.",
                verification(0.1),
            ],
            [[hit("a#0")]],
        )
        state = agent.run("q", TENANT, "req-11")

        assert state["suppressed"]
        assert state["verdict"] == Verdict.UNSUPPORTED
        assert state["answer"] == UNSUPPORTED_MESSAGE
        assert state["citations"] == []
        assert state["termination"] == Termination.UNSUPPORTED_ANSWER_BLOCKED

    def test_partial_support_is_not_suppressed(self) -> None:
        agent, _, _ = runner(
            [analysis("q"), grades(0.9), "mostly right [1]", verification(0.45)],
            [[hit("a#0")]],
            faithfulness_threshold=0.8,
        )
        state = agent.run("q", TENANT, "req-12")

        assert state["verdict"] == Verdict.PARTIAL
        assert not state["suppressed"]

    def test_unparseable_verification_suppresses(self) -> None:
        """A verification response that cannot be read is not a pass."""
        agent, _, _ = runner(
            [analysis("q"), grades(0.9), "an answer [1]", "not json at all"],
            [[hit("a#0")]],
        )
        state = agent.run("q", TENANT, "req-13")
        assert state["suppressed"]

    def test_verification_can_be_disabled(self) -> None:
        agent, provider, _ = runner(
            [analysis("q"), grades(0.9), "an answer [1]"],
            [[hit("a#0")]],
            enable_verification=False,
        )
        state = agent.run("q", TENANT, "req-14")

        assert state["verdict"] == Verdict.NOT_RUN
        assert state["termination"] == Termination.ANSWERED
        assert provider.usage.calls == 3

    def test_no_context_abstains_without_calling_the_model(self) -> None:
        agent, provider, _ = runner([analysis("q"), grades(0.0)], [[hit("x#0")]], max_rewrites=0)
        state = agent.run("q", TENANT, "req-15")

        assert state["answer"] == ABSTENTION_MESSAGE
        assert provider.usage.calls == 2


class TestContextAssembly:
    def test_context_is_capped_at_the_budget(self) -> None:
        hits = [hit(f"doc{i}#0", doc_id=f"kubernetes:doc{i}") for i in range(12)]
        agent, _, _ = runner(
            [analysis("q"), grades(*[0.9] * 12), "answer [1]", verification(0.9)],
            [hits],
            max_context_chunks=4,
        )
        state = agent.run("q", TENANT, "req-16")
        assert len(state["context"]) == 4

    def test_one_document_cannot_monopolise_the_window(self) -> None:
        hits = [hit(f"same#{i}", doc_id="kubernetes:same") for i in range(8)]
        agent, _, _ = runner(
            [analysis("q"), grades(*[0.9] * 8), "answer [1]", verification(0.9)],
            [hits],
            max_context_chunks=6,
        )
        state = agent.run("q", TENANT, "req-17")
        assert len(state["context"]) == 3

    def test_context_is_ordered_by_relevance(self) -> None:
        agent, _, _ = runner(
            [
                analysis("q"),
                grades(0.6, 0.95, 0.8),
                "answer [1]",
                verification(0.9),
            ],
            [[hit("low#0"), hit("high#0"), hit("mid#0")]],
        )
        state = agent.run("q", TENANT, "req-18")
        assert [c.chunk_id for c in state["context"]] == ["high#0", "mid#0", "low#0"]


class TestDegradedInputs:
    def test_unparseable_analysis_falls_back_to_the_question(self) -> None:
        agent, _, retriever = runner(
            ["not json", grades(0.9), "answer [1]", verification(0.9)],
            [[hit("a#0")]],
        )
        state = agent.run("how do I drain a node", TENANT, "req-19")

        assert retriever.queries == ["how do I drain a node"]
        assert state["termination"] == Termination.ANSWERED

    def test_unparseable_grading_scores_zero(self) -> None:
        agent, _, _ = runner(["{}", "not json", "x"], [[hit("a#0")]], max_rewrites=0)
        state = agent.run("q", TENANT, "req-20")
        assert state["mean_relevance"] == 0.0
        assert state["termination"] == Termination.REWRITE_BUDGET_EXHAUSTED

    def test_out_of_range_passage_numbers_are_ignored(self) -> None:
        bad = json.dumps({"scores": [{"passage": 99, "score": 1.0}]})
        agent, _, _ = runner([analysis("q"), bad, "x"], [[hit("a#0")]], max_rewrites=0)
        state = agent.run("q", TENANT, "req-21")
        assert state["relevant"] == []


class TestEdgeFunction:
    def test_routes_to_rerank_when_relevant(self) -> None:
        settings = Settings()
        state = initial_state("q", TENANT, "r")
        state["relevant"] = [hit("a#0")]
        assert should_rewrite(state, settings) == "rerank"

    def test_routes_to_rewrite_when_budget_remains(self) -> None:
        settings = Settings()
        settings.agent.max_rewrites = 2
        state = initial_state("q", TENANT, "r")
        state["rewrite_count"] = 1
        assert should_rewrite(state, settings) == "rewrite_query"

    def test_routes_to_give_up_at_the_budget(self) -> None:
        settings = Settings()
        settings.agent.max_rewrites = 2
        state = initial_state("q", TENANT, "r")
        state["rewrite_count"] = 2
        assert should_rewrite(state, settings) == "give_up"

    @pytest.mark.parametrize("required", [1, 2, 3])
    def test_respects_minimum_relevant_chunks(self, required: int) -> None:
        settings = Settings()
        settings.agent.min_relevant_chunks = required
        state = initial_state("q", TENANT, "r")
        state["relevant"] = [hit(f"a{i}#0") for i in range(required)]
        assert should_rewrite(state, settings) == "rerank"

        state["relevant"] = state["relevant"][:-1]
        assert should_rewrite(state, settings) != "rerank"
