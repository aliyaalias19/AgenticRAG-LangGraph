"""Measure attack success rate before and after mitigations.

The point of a before/after run is that "we added a defence" is a claim and
"attack success rate fell from X to Y on a fixed attack set" is a result.
The poisoned passages are injected directly into the agent's context, which
simulates an attacker who has succeeded in getting text into the corpus --
the threat model any RAG system over community-editable documentation has.

    python scripts/run_security_eval.py --mode unmitigated
    python scripts/run_security_eval.py --mode mitigated --compare
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agentic_rag.agent.graph import AgentRunner
from agentic_rag.agent.nodes import GraphDeps
from agentic_rag.agent.state import Termination
from agentic_rag.config.settings import get_settings
from agentic_rag.eval.security_eval import (
    AttackCase,
    AttackFamily,
    SecurityReport,
    build_attack_set,
    judge_outcome,
    write_report,
)
from agentic_rag.llm.provider import build_provider
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.obs.tracking import track_run
from agentic_rag.retrieval.hybrid import RetrievalConfig, build_retriever
from agentic_rag.retrieval.store import SearchHit
from agentic_rag.security.tenancy import TenantContext

logger = get_logger(__name__)


@dataclass
class PoisoningRetriever:
    """Wraps the real retriever and injects an attacker-controlled passage.

    The poisoned passage is placed first, where a genuinely compromised
    high-ranking document would sit. Placing it last would make the test
    easier than reality.
    """

    inner: object
    poison: str = ""
    extra_hits: list[SearchHit] = field(default_factory=list)
    last_chunk_ids: list[str] = field(default_factory=list)

    def retrieve(self, query, config, context=None, language=None):
        hits = list(self.inner.retrieve(query, config, context=context, language=language))
        if self.poison:
            hits.insert(
                0,
                SearchHit(
                    chunk_id="poisoned#0",
                    score=0.99,
                    doc_id="kubernetes:tasks/drain",
                    doc_title="Safely Drain a Node",
                    heading_path=("Overview",),
                    content=self.poison,
                    section="tasks",
                    language="en",
                ),
            )
        hits.extend(self.extra_hits)
        self.last_chunk_ids = [h.chunk_id for h in hits]
        return hits


def run_case(case: AttackCase, runner: AgentRunner, retriever: PoisoningRetriever):
    """Execute one attack case and judge the outcome."""
    retriever.poison = case.poisoned_passage
    retriever.extra_hits = []

    if case.family == AttackFamily.CROSS_TENANT:
        # A restricted chunk that the public tenant must never see. It is
        # offered to the retriever so that only the tenant filter stands
        # between the attacker and the leak.
        retriever.extra_hits = [
            SearchHit(
                chunk_id="restricted-only#0",
                score=0.95,
                doc_title="Cluster Bootstrap",
                content="Restricted setup content.",
                section="setup",
            )
        ]

    state = runner.run(case.question, TenantContext(case.tenant, (), "security-eval"), case.case_id)
    abstained = state.get("termination") in {
        Termination.NO_RELEVANT_CONTEXT,
        Termination.REWRITE_BUDGET_EXHAUSTED,
        Termination.UNSUPPORTED_ANSWER_BLOCKED,
    }
    return judge_outcome(
        case,
        str(state.get("answer", "")),
        [h.chunk_id for h in state.get("context", [])],
        abstained,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mitigated", "unmitigated"], default="mitigated")
    parser.add_argument(
        "--compare", action="store_true", help="Compare against the stored unmitigated report"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    if args.mode == "unmitigated":
        # Strip the controls, keep the pipeline: this isolates what the
        # verification gate and evidence threshold are actually buying.
        settings.agent.enable_verification = False
        settings.agent.relevance_threshold = 0.0
        settings.agent.max_rewrites = 0

    deps = GraphDeps(
        provider=build_provider(settings),
        retriever=PoisoningRetriever(inner=build_retriever(settings)),
        settings=settings,
        retrieval_config=RetrievalConfig("security", use_bm25=True, use_reranker=True),
    )
    runner = AgentRunner(deps)
    report = SecurityReport(label=args.mode)

    with track_run(f"security-{args.mode}", settings, tags={"stage": "security"}) as tracker:
        tracker.log_params(
            mode=args.mode,
            verification=settings.agent.enable_verification,
            relevance_threshold=settings.agent.relevance_threshold,
            max_rewrites=settings.agent.max_rewrites,
        )
        for case in build_attack_set():
            outcome = run_case(case, runner, deps.retriever)  # type: ignore[arg-type]
            report.outcomes.append(outcome)
            status = "BREACH" if outcome.succeeded else "ok"
            print(f"  [{status:>6}] {case.case_id:<12} {case.description}")

        tracker.log_metrics(
            attack_success_rate=report.attack_success_rate,
            succeeded=report.successes,
            cases=report.total,
        )

    print(
        f"\nattack success rate ({args.mode}): "
        f"{report.attack_success_rate:.1%}  "
        f"({report.successes}/{report.total})"
    )
    for family, stats in report.by_family().items():
        print(f"  {family:<22} {stats['succeeded']}/{stats['cases']}")

    output = (
        Path(args.output)
        if args.output
        else (settings.paths.results_dir / f"security_{args.mode}.json")
    )
    write_report(report, output)

    if args.compare:
        baseline_path = settings.paths.results_dir / "security_unmitigated.json"
        if baseline_path.is_file():
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            print(
                f"\nunmitigated {payload['attack_success_rate']:.1%}"
                f"  ->  {args.mode} {report.attack_success_rate:.1%}"
            )
        else:
            print("\nno unmitigated baseline found; run with --mode unmitigated first")

    return 0


if __name__ == "__main__":
    sys.exit(main())
