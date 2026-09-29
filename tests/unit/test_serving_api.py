"""Tests for the HTTP surface: auth, contracts, streaming and telemetry."""

import json

import pytest
from fastapi.testclient import TestClient

from agentic_rag.agent.state import Termination, Verdict, initial_state
from agentic_rag.config.settings import Settings
from agentic_rag.retrieval.store import SearchHit
from agentic_rag.security.tenancy import TenantContext
from agentic_rag.serving.app import create_app

PUBLIC = {"X-API-Key": "demo-public-key"}
INTERNAL = {"X-API-Key": "demo-internal-key"}


class ScriptedRunner:
    """Stands in for the agent so the HTTP layer can be tested on its own."""

    def __init__(self, settings: Settings, **overrides: object) -> None:
        self.deps = type(
            "Deps",
            (),
            {
                "provider": type("P", (), {"name": "echo"})(),
                "retriever": type("R", (), {"lexical": None})(),
                "settings": settings,
            },
        )()
        self.overrides = overrides
        self.calls: list[tuple[str, TenantContext]] = []

    def run(self, question, tenant, request_id, language=None):
        del language
        self.calls.append((question, tenant))
        state = initial_state(question, tenant, request_id)
        state["answer"] = "Use kubectl drain to evict pods [1]."
        state["context"] = [
            SearchHit(
                chunk_id="kubernetes:tasks/drain#0",
                score=0.9,
                doc_title="Safely Drain a Node",
                heading_path=("Overview",),
                source_path="tasks/drain.md",
                content="body",
            )
        ]
        state["termination"] = Termination.ANSWERED
        state["verdict"] = Verdict.SUPPORTED
        state["faithfulness"] = 0.93
        state.update(self.overrides)  # type: ignore[arg-type]
        return state


@pytest.fixture
def client() -> TestClient:
    settings = Settings()
    settings.security.audit_log_enabled = False
    app = create_app(lambda s: ScriptedRunner(s), settings)  # type: ignore[arg-type]
    return TestClient(app)


def make_client(**overrides: object) -> TestClient:
    settings = Settings()
    settings.security.audit_log_enabled = False
    app = create_app(lambda s: ScriptedRunner(s, **overrides), settings)  # type: ignore[arg-type]
    return TestClient(app)


class TestAuthentication:
    def test_missing_key_is_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/query", json={"question": "q"})
        assert response.status_code == 401
        assert "X-API-Key" in response.json()["detail"]

    def test_invalid_key_is_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/query", json={"question": "q"}, headers={"X-API-Key": "nope"})
        assert response.status_code == 401

    def test_error_does_not_leak_valid_keys(self, client: TestClient) -> None:
        response = client.post("/v1/query", json={"question": "q"}, headers={"X-API-Key": "nope"})
        assert "demo-public-key" not in response.text

    def test_valid_key_is_accepted(self, client: TestClient) -> None:
        response = client.post("/v1/query", json={"question": "q"}, headers=PUBLIC)
        assert response.status_code == 200

    def test_tenant_reaches_the_agent(self) -> None:
        settings = Settings()
        settings.security.audit_log_enabled = False
        runner = ScriptedRunner(settings)
        app = create_app(lambda _s: runner, settings)  # type: ignore[arg-type,misc]
        with TestClient(app) as http:
            http.post("/v1/query", json={"question": "q"}, headers=INTERNAL)

        _, tenant = runner.calls[0]
        assert tenant.tenant_id == "internal"
        assert tenant.allowed_sections == ("concepts", "tasks", "tutorials")

    def test_tenant_cannot_be_set_from_the_body(self, client: TestClient) -> None:
        """An unknown field must not become a privilege escalation path."""
        response = client.post(
            "/v1/query",
            json={"question": "q", "tenant": "restricted"},
            headers=PUBLIC,
        )
        assert response.status_code == 200


class TestQueryContract:
    def test_response_carries_answer_and_provenance(self, client: TestClient) -> None:
        body = client.post(
            "/v1/query", json={"question": "how do I drain a node"}, headers=PUBLIC
        ).json()

        assert body["answer"].startswith("Use kubectl drain")
        assert body["termination"] == "answered"
        assert body["verdict"] == "supported"
        assert body["faithfulness"] == 0.93
        assert body["citations"][0]["title"] == "Safely Drain a Node"
        assert body["citations"][0]["heading_path"] == ["Overview"]

    def test_request_id_is_unique_per_call(self, client: TestClient) -> None:
        first = client.post("/v1/query", json={"question": "q"}, headers=PUBLIC).json()
        second = client.post("/v1/query", json={"question": "q"}, headers=PUBLIC).json()
        assert first["request_id"] != second["request_id"]

    def test_trace_is_omitted_by_default(self, client: TestClient) -> None:
        body = client.post("/v1/query", json={"question": "q"}, headers=PUBLIC).json()
        assert body["trace"] is None

    def test_trace_is_returned_on_request(self, client: TestClient) -> None:
        body = client.post(
            "/v1/query",
            json={"question": "q", "include_trace": True},
            headers=PUBLIC,
        ).json()
        assert isinstance(body["trace"], list)

    def test_empty_question_is_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/query", json={"question": ""}, headers=PUBLIC)
        assert response.status_code == 422

    def test_overlong_question_is_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/query", json={"question": "x" * 2001}, headers=PUBLIC)
        assert response.status_code == 422

    def test_invalid_language_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/query", json={"question": "q", "language": "fr"}, headers=PUBLIC
        )
        assert response.status_code == 422

    def test_suppressed_answer_is_reported_as_such(self) -> None:
        http = make_client(
            suppressed=True,
            termination=Termination.UNSUPPORTED_ANSWER_BLOCKED,
            verdict=Verdict.UNSUPPORTED,
        )
        body = http.post("/v1/query", json={"question": "q"}, headers=PUBLIC).json()
        assert body["suppressed"] is True
        assert body["termination"] == "unsupported_answer_blocked"


class TestStreaming:
    def _events(self, response) -> list[dict]:
        events = []
        for block in response.text.split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))
        return events

    def test_content_type_is_event_stream(self, client: TestClient) -> None:
        response = client.post("/v1/query/stream", json={"question": "q"}, headers=PUBLIC)
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_proxy_buffering_is_disabled(self, client: TestClient) -> None:
        """Without this header an nginx in front would batch the whole stream."""
        response = client.post("/v1/query/stream", json={"question": "q"}, headers=PUBLIC)
        assert response.headers["x-accel-buffering"] == "no"

    def test_event_order(self, client: TestClient) -> None:
        response = client.post("/v1/query/stream", json={"question": "q"}, headers=PUBLIC)
        types = [e["type"] for e in self._events(response)]

        assert types[0] == "status"
        assert types[1] == "status"
        assert "token" in types
        assert types[-2] == "citations"
        assert types[-1] == "done"

    def test_tokens_reassemble_into_the_answer(self, client: TestClient) -> None:
        response = client.post("/v1/query/stream", json={"question": "q"}, headers=PUBLIC)
        tokens = [e["data"]["text"] for e in self._events(response) if e["type"] == "token"]
        assert "".join(tokens) == "Use kubectl drain to evict pods [1]."

    def test_citations_are_streamed(self, client: TestClient) -> None:
        response = client.post("/v1/query/stream", json={"question": "q"}, headers=PUBLIC)
        citations = next(e for e in self._events(response) if e["type"] == "citations")
        assert citations["data"]["citations"][0]["title"] == "Safely Drain a Node"

    def test_done_event_carries_the_outcome(self, client: TestClient) -> None:
        response = client.post("/v1/query/stream", json={"question": "q"}, headers=PUBLIC)
        done = self._events(response)[-1]
        assert done["data"]["termination"] == "answered"
        assert done["data"]["duration_ms"] >= 0

    def test_stream_requires_authentication(self, client: TestClient) -> None:
        assert client.post("/v1/query/stream", json={"question": "q"}).status_code == 401


class TestOperations:
    def test_healthz_reports_configuration(self, client: TestClient) -> None:
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["provider"] == "echo"
        assert body["collection"] == "k8s_chunks"

    def test_metrics_endpoint_exposes_counters(self, client: TestClient) -> None:
        client.post("/v1/query", json={"question": "q"}, headers=PUBLIC)
        body = client.get("/metrics").text
        assert "agentic_rag_requests_total" in body
        assert "agentic_rag_request_duration_seconds" in body

    def test_metrics_are_labelled_by_tenant(self, client: TestClient) -> None:
        client.post("/v1/query", json={"question": "q"}, headers=INTERNAL)
        body = client.get("/metrics").text
        assert 'tenant="internal"' in body

    def test_auth_failures_are_counted(self, client: TestClient) -> None:
        client.post("/v1/query", json={"question": "q"}, headers={"X-API-Key": "x"})
        assert "agentic_rag_auth_failures_total" in client.get("/metrics").text
