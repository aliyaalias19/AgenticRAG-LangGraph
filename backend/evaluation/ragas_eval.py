"""
RAGAS evaluation harness.

RAGAS (Retrieval Augmented Generation Assessment) measures four orthogonal
quality dimensions:

  faithfulness:       Is the answer grounded in the retrieved context?
                      → Catches hallucinations relative to the corpus.
  answer_relevancy:   Does the answer address the question asked?
                      → Catches off-topic or evasive responses.
  context_precision:  What fraction of retrieved chunks were actually useful?
                      → Measures retrieval signal-to-noise ratio.
  context_recall:     Was all required information retrieved?
                      → Measures retrieval coverage.

Why these four metrics:
  Together they form a 2×2 matrix across generation quality (faithfulness,
  relevancy) and retrieval quality (precision, recall). A system that scores
  well on all four is genuinely reliable; gaming one metric at the expense
  of others is hard.

Target thresholds (production baseline) are defined in ``EVAL_THRESHOLDS``
below and enforced by ``check_eval_thresholds()``. Wire that helper into
CI so a regression drops the deploy rather than silently shipping.
"""

import datetime
import json
import logging
import os
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

# Declared shipping bar. A run is "passing" iff every metric meets its
# minimum. Numbers come from public RAGAS leaderboards for retrieval-
# augmented systems on enterprise document corpora; tune per customer.
# Lower bounds, not targets — the goal is to be substantially above.
EVAL_THRESHOLDS: Dict[str, float] = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.80,
    "context_precision": 0.70,
    "context_recall":    0.70,
}


def check_eval_thresholds(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a RAGAS summary against ``EVAL_THRESHOLDS``.

    Returns a dict ``{passed: bool, failures: [{metric, value, threshold}]}``.
    The shipping bar is enforced here, not in CI YAML, so the bar travels
    with the code and the test cases — one source of truth.
    """
    failures = []
    for metric, threshold in EVAL_THRESHOLDS.items():
        value = summary.get(metric)
        if value is None or value < threshold:
            failures.append({
                "metric": metric,
                "value": value,
                "threshold": threshold,
            })
    return {"passed": not failures, "failures": failures}

# The test set is chosen for diversity of failure mode, not size. Each
# case probes a distinct pipeline behaviour. Growing this set without
# growing coverage of new failure modes is busywork — every addition
# should target a specific category not yet exercised. Document the
# category in a trailing comment so the bar for additions is explicit.
SAMPLE_TEST_CASES: List[Dict[str, str]] = [
    # ── Direct factual recall — should be near-perfect ──────────────────────
    {
        "question": "What is the minimum initial investment for ASN Equity Malaysia?",
        "ground_truth": "The minimum initial investment is RM10.00 via cash or cash equivalent.",
    },
    {
        "question": "What is the sales charge for ASN Equity Malaysia?",
        "ground_truth": "There is no sales charge. The sales charge is Nil.",
    },
    {
        "question": "What is the annual management fee for ASN Equity Malaysia?",
        "ground_truth": "Up to a maximum of 1.0% per annum of the NAV of the Fund.",
    },
    {
        "question": "What is the benchmark for ASN Equity Malaysia?",
        "ground_truth": "90% FBM 100 and 10% Maybank 1-month Fixed Deposit Rate.",
    },
    {
        "question": "What is the minimum balance required to remain a unit holder?",
        "ground_truth": "Unit holders must maintain a minimum balance of one unit.",
    },
    {
        "question": "Can a minor invest in ASNB funds?",
        "ground_truth": "Yes, a guardian can apply on behalf of a Malaysian minor below 18.",
    },
    {
        "question": "What is the cut-off time for transactions?",
        "ground_truth": "The dealing cut-off time is 4.00 p.m. on any Business Day.",
    },
    {
        "question": "How many business days does repurchase payment take?",
        "ground_truth": "Payment will be made within seven business days from the repurchase date.",
    },
    {
        "question": "What is the fund's investment objective?",
        "ground_truth": "The fund aims to provide income and capital appreciation through equities.",
    },
    {
        "question": "Are there any penalties for early withdrawal?",
        "ground_truth": "No early withdrawal penalties are mentioned in the prospectus.",
    },
    # ── Long / multi-clause question — exercises chunk recall across pages ──
    # Probes whether the retriever finds two distinct chunks (fee + benchmark)
    # in one query, and whether the generator can synthesise both.
    {
        "question": "Compare the annual management fee and the benchmark — what are both, and how do they relate to investor returns?",
        "ground_truth": "The annual management fee is up to 1.0% per annum of NAV. The benchmark is 90% FBM 100 + 10% Maybank 1-month FD rate. The fee reduces the returns the fund delivers relative to the benchmark.",
    },
    # ── Out-of-scope query — must NOT hallucinate ──────────────────────────
    # Faithfulness on this one is the load-bearing metric. If the answer
    # invents a number, faithfulness drops; if it correctly says "not in
    # the document", faithfulness stays high and we see context_recall low.
    {
        "question": "What is the historical 10-year compound annual return of the fund?",
        "ground_truth": "The prospectus does not state a historical 10-year compound annual return.",
    },
    # ── Ambiguous reference — exercises query rewriting ─────────────────────
    # "It" has no antecedent. A well-rewritten query produces useful
    # retrieval; a naive pass-through retrieves noise. Tests the
    # rewrite-on-low-grade feedback loop.
    {
        "question": "What happens to it if I miss the cut-off?",
        "ground_truth": "Transactions received after the 4.00 p.m. cut-off are processed on the next Business Day.",
    },
    # ── Numeric edge — should pull the exact value, not paraphrase ──────────
    # Tests whether the model preserves precision (RM10.00, not "around ten
    # ringgit") when answering numeric questions.
    {
        "question": "Give the exact minimum investment amount as it appears in the prospectus.",
        "ground_truth": "RM10.00.",
    },
    # ── Negation — risk of inverted answer ──────────────────────────────────
    # "Are there any" + negative ground-truth is a classic hallucination
    # trigger. Tests grounding under negation.
    {
        "question": "Are there any redemption fees charged by the fund?",
        "ground_truth": "No, no redemption fees are charged.",
    },
]


def _get_ragas_providers():
    """Return (wrapped_llm, wrapped_embeddings) for RAGAS evaluation."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("RAGAS evaluation using OpenAI: %s", llm_model)
        return (
            LangchainLLMWrapper(ChatOpenAI(model=llm_model, api_key=openai_key, temperature=0)),
            LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=openai_key)),
        )

    # Bedrock fallback
    try:
        import boto3
        if boto3.Session().get_credentials():
            from langchain_aws import BedrockEmbeddings, ChatBedrock
            region = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
            return (
                LangchainLLMWrapper(ChatBedrock(
                    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                    region_name=region,
                )),
                LangchainEmbeddingsWrapper(BedrockEmbeddings(
                    model_id="cohere.embed-multilingual-v3",
                    region_name=region,
                )),
            )
    except Exception as exc:
        logger.warning("Bedrock unavailable for RAGAS: %s", exc)

    raise RuntimeError(
        "RAGAS evaluation requires either OPENAI_API_KEY or AWS credentials. "
        "Set OPENAI_API_KEY in .env to enable evaluation."
    )


def run_ragas_evaluation(
    test_cases: List[Dict[str, str]],
    agent_fn: Callable,
) -> Dict[str, Any]:
    """Run RAGAS evaluation over test cases.

    Args:
        test_cases: List of {"question": str, "ground_truth": str} dicts.
        agent_fn:   Callable(question: str) → RAGState-like object with
                    .answer and .retrieved_chunks attributes or dict keys.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    questions, answers, contexts, ground_truths = [], [], [], []
    per_question = []

    for tc in test_cases:
        q, gt = tc["question"], tc["ground_truth"]
        logger.info("Evaluating: %r", q)

        state = agent_fn(q)

        # Support both RAGState dict and legacy AgentState dataclass
        if isinstance(state, dict):
            ans = state.get("answer", "")
            chunks = state.get("retrieved_chunks", [])
            score = state.get("relevance_score", 0)
        else:
            ans = getattr(state, "answer", "")
            chunks = getattr(state, "retrieved_chunks", [])
            score = getattr(state, "relevance_score", 0)

        questions.append(q)
        answers.append(ans)
        contexts.append([c.page_content if hasattr(c, "page_content") else c for c in chunks])
        ground_truths.append(gt)
        per_question.append({
            "question": q,
            "answer": ans,
            "relevance_score": score,
            "chunks_retrieved": len(chunks),
        })

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    wrapped_llm, wrapped_embeddings = _get_ragas_providers()
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = wrapped_llm
        if hasattr(m, "embeddings"):
            m.embeddings = wrapped_embeddings

    results = evaluate(dataset, metrics=metrics)
    df = results.to_pandas()

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "faithfulness":      round(float(df["faithfulness"].mean()), 3),
        "answer_relevancy":  round(float(df["answer_relevancy"].mean()), 3),
        "context_precision": round(float(df["context_precision"].mean()), 3),
        "context_recall":    round(float(df["context_recall"].mean()), 3),
        "total_questions":   len(df),
    }

    # Append to history for trend tracking
    data_dir = os.getenv("DATA_DIR", "./data")
    os.makedirs(data_dir, exist_ok=True)
    history_path = os.path.join(data_dir, "eval_history.jsonl")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    summary["per_question"] = per_question
    return summary
