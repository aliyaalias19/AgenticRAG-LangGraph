"""Run the full agent end to end with no external services.

Everything is real except the model and the vector store: the graph, the
conditional edges, BM25 retrieval, RRF fusion, tenant scoping, evidence
gating and the trace all execute exactly as they do in production. Useful for
seeing the system work before Qdrant is running or a GPU is rented, and for
demonstrating the self-correction loop without paying for tokens.

    python scripts/demo_offline.py
"""

import json
import sys
from dataclasses import dataclass, field

from agentic_rag.agent.graph import AgentRunner, total_duration_ms, trace_summary
from agentic_rag.agent.nodes import GraphDeps
from agentic_rag.config.settings import Settings
from agentic_rag.llm.provider import ChatProvider, Completion
from agentic_rag.obs.logging import configure_logging
from agentic_rag.retrieval.bm25 import BM25Document, BM25Index
from agentic_rag.retrieval.hybrid import HybridRetriever, RetrievalConfig
from agentic_rag.retrieval.reranker import NullReranker
from agentic_rag.security.tenancy import TenantContext

CORPUS = [
    (
        "kubernetes:tasks/drain#0",
        "Safely Drain a Node",
        "tasks",
        "public",
        "Use kubectl drain to remove a node from service. The node is first "
        "cordoned so the scheduler stops placing new pods on it, then the "
        "running pods are evicted while respecting any PodDisruptionBudget.",
    ),
    (
        "kubernetes:tasks/drain#1",
        "Safely Drain a Node",
        "tasks",
        "public",
        "Pods managed by a DaemonSet are not evicted by default, because the "
        "DaemonSet controller would immediately recreate them on the same "
        "node. Pass --ignore-daemonsets to let the drain proceed.",
    ),
    (
        "kubernetes:concepts/scheduling#0",
        "Taints and Tolerations",
        "concepts",
        "public",
        "A taint on a node repels pods that do not tolerate it. Cordoning a "
        "node applies the unschedulable taint, which is why no new pods are "
        "placed on it.",
    ),
    (
        "kubernetes:setup/bootstrap#0",
        "Cluster Bootstrap",
        "setup",
        "restricted",
        "Certificate authority rotation requires regenerating the cluster CA "
        "and restarting the control plane components in order.",
    ),
]


class ScriptedProvider(ChatProvider):
    """Replays a fixed sequence of model responses."""

    name = "scripted"

    def __init__(self, responses: list[str], settings: Settings) -> None:
        super().__init__(settings)
        self.responses = responses
        self.position = 0

    def complete(self, messages, *, system="", max_tokens=None):
        del system, max_tokens
        text = self.responses[min(self.position, len(self.responses) - 1)]
        self.position += 1
        return self._record(
            Completion(
                text=text,
                input_tokens=sum(len(m.content) // 4 for m in messages),
                output_tokens=len(text) // 4,
                provider=self.name,
            )
        )


@dataclass
class LocalStore:
    """A vector store stand-in that returns nothing, so BM25 does the work."""

    calls: list[dict] = field(default_factory=list)

    def search_dense(self, vector, limit, tenant=None, language=None, sections=()):
        del vector, limit, language
        self.calls.append({"kind": "dense", "tenant": tenant, "sections": sections})
        return []

    def search_sparse(self, weights, limit, tenant=None, language=None, sections=()):
        del weights, limit, language
        self.calls.append({"kind": "sparse", "tenant": tenant, "sections": sections})
        return []


class StubEmbedder:
    def encode_query(self, text: str):
        del text
        return [0.0] * 8, {}


def build_index() -> BM25Index:
    index = BM25Index()
    for chunk_id, title, section, tenant, content in CORPUS:
        index.add(
            BM25Document(
                chunk_id=chunk_id,
                doc_id=chunk_id.split("#")[0],
                relative_id=chunk_id.split(":")[1].split("#")[0],
                doc_title=title,
                heading_path=("Overview",),
                content=content,
                language="en",
                section=section,
                tenant=tenant,
                source_path=f"{section}.md",
            )
        )
    index.finalise()
    return index


def build_runner(responses: list[str], settings: Settings) -> AgentRunner:
    retriever = HybridRetriever(
        embedder=StubEmbedder(),
        store=LocalStore(),
        reranker=NullReranker(),
        lexical=build_index(),
        settings=settings,
    )
    return AgentRunner(
        GraphDeps(
            provider=ScriptedProvider(responses, settings),
            retriever=retriever,
            settings=settings,
            retrieval_config=RetrievalConfig(
                "demo",
                use_dense=False,
                use_sparse=False,
                use_bm25=True,
                use_reranker=False,
                top_k=4,
            ),
        )
    )


def show(title: str, state) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(f"answer      : {state['answer'][:140]}")
    print(f"termination : {state['termination']}")
    print(f"verdict     : {state['verdict']}  (faithfulness {state['faithfulness']})")
    print(f"rewrites    : {state['rewrite_count']}")
    print(f"context     : {[h.chunk_id for h in state['context']]}")
    print(f"duration    : {total_duration_ms(state)} ms")
    print("trace       :")
    for step in trace_summary(state):
        detail = {k: v for k, v in step.items() if k not in {"node", "iteration", "duration_ms"}}
        print(f"  {step['iteration']}  {step['node']:<16} {json.dumps(detail)}")


def main() -> int:
    configure_logging()
    settings = Settings()
    settings.logging.level = "WARNING"
    public = TenantContext("public", (), "demo")

    # 1. The happy path.
    runner = build_runner(
        [
            json.dumps({"query_type": "procedural", "search_query": "drain node evict pods"}),
            json.dumps(
                {
                    "scores": [
                        {"passage": 1, "score": 0.95},
                        {"passage": 2, "score": 0.9},
                        {"passage": 3, "score": 0.3},
                    ]
                }
            ),
            "Run kubectl drain, which cordons the node and evicts its pods "
            "while respecting PodDisruptionBudgets [1]. DaemonSet pods are "
            "skipped unless you pass --ignore-daemonsets [2].",
            json.dumps({"faithfulness": 0.95, "notes": "supported"}),
        ],
        settings,
    )
    show(
        "1. Answered on the first attempt",
        runner.run("how do I take a node out of service", public, "demo-1"),
    )

    # 2. The self-correction loop: first retrieval is judged irrelevant.
    runner = build_runner(
        [
            json.dumps({"query_type": "troubleshooting", "search_query": "node unavailable"}),
            json.dumps({"scores": [{"passage": 1, "score": 0.1}, {"passage": 2, "score": 0.15}]}),
            json.dumps(
                {
                    "search_query": "drain evict pods daemonset",
                    "rationale": "name the mechanism, not the symptom",
                }
            ),
            json.dumps({"scores": [{"passage": 1, "score": 0.92}, {"passage": 2, "score": 0.88}]}),
            "Drain the node with kubectl drain [1]; DaemonSet pods need --ignore-daemonsets [2].",
            json.dumps({"faithfulness": 0.9, "notes": "supported"}),
        ],
        settings,
    )
    show(
        "2. Low relevance triggers a query rewrite",
        runner.run("my node needs maintenance", public, "demo-2"),
    )

    # 3. Evidence gating blocks an unsupported answer.
    runner = build_runner(
        [
            json.dumps({"query_type": "factual", "search_query": "drain node"}),
            json.dumps({"scores": [{"passage": 1, "score": 0.9}]}),
            "Draining a node requires a minimum of three control plane "
            "replicas and takes exactly 45 seconds.",
            json.dumps(
                {"faithfulness": 0.15, "unsupported_claims": ["three replicas", "45 seconds"]}
            ),
        ],
        settings,
    )
    show(
        "3. Unsupported answer is suppressed",
        runner.run("what does draining require", public, "demo-3"),
    )

    # 4. Tenant isolation: the restricted chunk is never retrievable.
    runner = build_runner(
        [
            json.dumps(
                {"query_type": "procedural", "search_query": "certificate authority rotation"}
            ),
            json.dumps({"scores": []}),
            json.dumps({"search_query": "cluster CA restart control plane"}),
            json.dumps({"scores": []}),
        ],
        settings,
    )
    state = runner.run("how do I rotate the cluster certificate authority", public, "demo-4")
    show("4. Restricted content stays invisible to the public tenant", state)
    leaked = [h for h in state["context"] if "setup" in h.chunk_id]
    print(f"\nrestricted chunks leaked: {len(leaked)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
