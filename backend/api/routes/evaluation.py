"""RAGAS evaluation endpoint. Runs against the caller's tenant corpus."""

import logging
from functools import partial

from fastapi import APIRouter, Depends, HTTPException

from core.audit import audit_event
from core.auth import TenantContext, get_tenant_context
from core.config import get_settings
from ingestion.pipeline import list_ingested_files
from models.requests import EvalRequest
from models.responses import EvalMetrics, EvalResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Evaluation"])
settings = get_settings()


@router.post("/evaluate", response_model=EvalResponse, summary="Run RAGAS evaluation")
async def evaluate_rag(
    request: EvalRequest,
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Evaluate pipeline quality using RAGAS over a set of ground-truth test cases.

    Runs against the caller's tenant corpus only.

    Metrics:
      faithfulness:       fraction of answer claims grounded in retrieved context
      answer_relevancy:   how well the answer addresses the question
      context_precision:  fraction of retrieved chunks that were actually relevant
      context_recall:     fraction of required information that was retrieved

    Target thresholds for production: faithfulness > 0.80, relevancy > 0.80.
    """
    if not list_ingested_files(settings, tenant_id=tenant.tenant_id):
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    from evaluation.ragas_eval import (
        SAMPLE_TEST_CASES,
        check_eval_thresholds,
        run_ragas_evaluation,
    )
    from agents.graph import run_rag

    test_cases = request.custom_test_cases or SAMPLE_TEST_CASES

    # Bind tenant_id into the run_rag call so the evaluator queries the
    # right corpus. partial() captures it before run_in_executor strips
    # access to the request context.
    def tenant_rag(query):
        return run_rag(query, tenant_id=tenant.tenant_id)

    import asyncio
    loop = asyncio.get_running_loop()

    try:
        results = await loop.run_in_executor(
            None, partial(run_ragas_evaluation, test_cases, tenant_rag)
        )
    except Exception as exc:
        logger.exception("RAGAS evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {exc}")

    gate = check_eval_thresholds(results)
    audit_event(
        "evaluate",
        tenant_id=tenant.tenant_id,
        actor_fingerprint=tenant.api_key_fingerprint,
        detail={"test_cases": len(test_cases), "passed": gate["passed"]},
    )

    return EvalResponse(
        status="complete",
        metrics=EvalMetrics(
            faithfulness=results["faithfulness"],
            answer_relevancy=results["answer_relevancy"],
            context_precision=results["context_precision"],
            context_recall=results["context_recall"],
        ),
        total_questions=results["total_questions"],
        per_question=results["per_question"],
        interpretation={
            "faithfulness":      "Fraction of answer claims grounded in retrieved context (1.0 = perfect)",
            "answer_relevancy":  "How well the answer addresses the question (1.0 = perfect)",
            "context_precision": "Fraction of retrieved chunks that were relevant",
            "context_recall":    "Fraction of required information that was retrieved",
        },
        passed=gate["passed"],
        threshold_failures=gate["failures"],
    )
