"""
End-to-end route tests via httpx ASGITransport.

Why these tests exist
─────────────────────
The unit tests in test_nodes.py exercise pure functions in isolation.
Real customers don't hit pure functions — they hit HTTP routes through
the middleware stack. The most important security guarantees in this
codebase (auth gate, tenant isolation, audit emission) live in the
wiring between the dependency, the route handler, and the retrieval
layer. That wiring has no unit-test analogue; it has to be exercised
end-to-end.

These tests do NOT call an LLM or a real vector store. The retrieval
and ingestion paths are patched at the FastAPI dependency boundary —
every test gives the route a controlled return value so we verify
exactly what crosses the wire and what doesn't.

Why not FastAPI's TestClient
────────────────────────────
The installed starlette TestClient is older than the installed httpx;
``TestClient(app)`` raises ``Client.__init__() got an unexpected keyword
argument 'app'`` because httpx ≥ 0.27 removed that constructor argument
in favour of ``transport=ASGITransport(app=...)``. We drive httpx
directly here — version-stable and zero-dependency.

Coverage focus
──────────────
  1. Auth gate: 401 with no key, 401 with wrong key, 200 with right key
  2. Tenant isolation: route propagates auth-derived tenant_id through
  3. Health and /metrics bypass auth (operator probes work)
  4. Audit row is emitted on a successful query, key fingerprint only
  5. /metrics serves valid Prometheus text-exposition format
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(coro):
    """Run an async coroutine to completion. Pytest is sync; httpx
    ASGITransport is async-only. This single helper lets every test
    stay sync while driving the ASGI app correctly."""
    return asyncio.run(coro)


@pytest.fixture
def app_with_auth(tmp_path, monkeypatch):
    """Build a FastAPI app configured with API-key auth and an isolated
    data dir so each test gets a clean audit log + Qdrant collection.

    Patches get_settings on the auth and audit modules so the API_KEYS
    map and DATA_DIR take effect without fighting the module-level
    settings cache.
    """
    from core.config import Settings

    keys = {"sk-alpha": "tenant-alpha", "sk-beta": "tenant-beta"}
    s = Settings(
        data_dir=str(tmp_path),
        qdrant_path=str(tmp_path / "qdrant"),
        api_keys_raw=json.dumps(keys),
    )

    import core.auth as auth_mod
    import core.audit as audit_mod
    monkeypatch.setattr(auth_mod, "get_settings", lambda: s)
    monkeypatch.setattr(audit_mod, "get_settings", lambda: s)

    from fastapi import FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware
    from api.routes import health, query, documents, evaluation
    from infrastructure.prometheus import render_metrics

    test_app = FastAPI()
    test_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    test_app.include_router(health.router)
    test_app.include_router(documents.router)
    test_app.include_router(query.router)
    test_app.include_router(evaluation.router)

    @test_app.get("/metrics")
    async def _metrics():
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    return test_app, s


def _request(app, method: str, path: str, **kwargs):
    """Issue a single request against the ASGI app and return the response."""
    async def _do():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as c:
            return await c.request(method, path, **kwargs)
    return _run(_do())


# ─────────────────────────────────────────────────────────────────────────────
# Auth gate
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthGate:
    def test_query_without_key_returns_401(self, app_with_auth):
        app, _ = app_with_auth
        r = _request(app, "POST", "/query", json={"question": "anything"})
        assert r.status_code == 401
        # Header is case-insensitive; httpx normalises lower.
        assert "ApiKey" in r.headers.get("www-authenticate", "")

    def test_query_with_wrong_key_returns_401(self, app_with_auth):
        app, _ = app_with_auth
        r = _request(
            app, "POST", "/query",
            json={"question": "anything"},
            headers={"X-API-Key": "sk-wrong"},
        )
        assert r.status_code == 401

    def test_documents_list_without_key_returns_401(self, app_with_auth):
        app, _ = app_with_auth
        r = _request(app, "GET", "/documents")
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Public-path bypass
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicPaths:
    def test_health_is_open(self, app_with_auth):
        app, _ = app_with_auth
        r = _request(app, "GET", "/health")
        assert r.status_code == 200

    def test_metrics_is_open_and_returns_prometheus_format(self, app_with_auth):
        app, _ = app_with_auth
        r = _request(app, "GET", "/metrics")
        assert r.status_code == 200
        # Prometheus text exposition format starts with "# HELP" or "# TYPE".
        body = r.text
        assert "# HELP" in body or "# TYPE" in body


# ─────────────────────────────────────────────────────────────────────────────
# Tenant isolation through the HTTP stack
# ─────────────────────────────────────────────────────────────────────────────

class TestTenantIsolationE2E:
    """The point of these tests is the WIRING — that the auth-derived
    tenant_id propagates all the way to the pipeline call. We patch
    list_ingested_files / delete_document so the storage layer is out
    of scope; only the wiring is under test."""

    def test_list_documents_passes_tenant_through(self, app_with_auth, monkeypatch):
        app, _ = app_with_auth
        seen_tenants = []

        def fake_list(settings, tenant_id=None):
            seen_tenants.append(tenant_id)
            return [f"{tenant_id}-doc.pdf"]

        from api.routes import documents as docs_route
        monkeypatch.setattr(docs_route, "list_ingested_files", fake_list)

        r_alpha = _request(app, "GET", "/documents", headers={"X-API-Key": "sk-alpha"})
        r_beta  = _request(app, "GET", "/documents", headers={"X-API-Key": "sk-beta"})

        assert r_alpha.status_code == 200
        assert r_beta.status_code == 200
        assert r_alpha.json()["documents"] == ["tenant-alpha-doc.pdf"]
        assert r_beta.json()["documents"]  == ["tenant-beta-doc.pdf"]
        assert seen_tenants == ["tenant-alpha", "tenant-beta"]

    def test_delete_passes_tenant_through(self, app_with_auth, monkeypatch):
        app, _ = app_with_auth
        seen = {}

        def fake_delete(filename, settings, tenant_id=None):
            seen["filename"] = filename
            seen["tenant_id"] = tenant_id
            return {"status": "deleted", "file": filename, "chunks_deleted": 3}

        from api.routes import documents as docs_route
        monkeypatch.setattr(docs_route, "delete_document", fake_delete)

        r = _request(app, "DELETE", "/documents/foo.pdf", headers={"X-API-Key": "sk-alpha"})
        assert r.status_code == 200
        assert seen["tenant_id"] == "tenant-alpha"
        assert seen["filename"] == "foo.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# Audit emission on a successful query
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditEmission:
    def test_successful_query_writes_audit_row(self, app_with_auth, monkeypatch):
        app, s = app_with_auth
        from api.routes import query as query_route

        monkeypatch.setattr(query_route, "_require_documents", lambda tid: None)
        fake_state = {
            "answer": "test", "rewritten_query": "test", "relevance_score": 8,
            "hallucination_score": 0.05, "steps_taken": 1, "reasoning_trace": [],
            "latency_ms": 12.3, "citations": [], "retrieval_metadata": {},
        }
        monkeypatch.setattr(query_route, "run_rag", lambda q, h, t: fake_state)

        r = _request(
            app, "POST", "/query",
            json={"question": "anything", "show_reasoning": False},
            headers={"X-API-Key": "sk-alpha"},
        )
        assert r.status_code == 200

        audit_path = Path(s.data_dir) / "audit.jsonl"
        assert audit_path.exists()
        lines = [json.loads(l) for l in audit_path.read_text(encoding="utf-8").strip().splitlines()]
        query_rows = [l for l in lines if l["action"] == "query"]
        assert len(query_rows) >= 1
        row = query_rows[-1]
        assert row["tenant_id"] == "tenant-alpha"
        assert row["actor"].startswith("sk-a")
        # Raw API key value never leaks into the audit row.
        assert "sk-alpha" not in row["actor"]
