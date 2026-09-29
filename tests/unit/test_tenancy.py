"""Tests for tenant resolution, scoping and audit logging."""

from pathlib import Path

import pytest

from agentic_rag.config.settings import Settings
from agentic_rag.security.audit import append_record, build_record, read_records
from agentic_rag.security.tenancy import (
    AuthenticationError,
    TenantContext,
    anonymous_context,
    fingerprint,
    resolve_tenant,
)


class TestFingerprint:
    def test_is_deterministic(self) -> None:
        assert fingerprint("secret-key") == fingerprint("secret-key")

    def test_differs_per_key(self) -> None:
        assert fingerprint("a") != fingerprint("b")

    def test_does_not_reveal_the_key(self) -> None:
        key = "demo-internal-key"
        assert key not in fingerprint(key)
        assert len(fingerprint(key)) == 12


class TestResolveTenant:
    def test_known_key_resolves(self) -> None:
        context = resolve_tenant("demo-internal-key", Settings())
        assert context.tenant_id == "internal"
        assert context.allowed_sections == ("concepts", "tasks", "tutorials")

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(AuthenticationError):
            resolve_tenant("not-a-real-key", Settings())

    def test_error_does_not_leak_valid_keys(self) -> None:
        try:
            resolve_tenant("bad", Settings())
        except AuthenticationError as error:
            assert "demo-internal-key" not in str(error)
            assert "bad" not in str(error)

    def test_public_tenant_is_unrestricted(self) -> None:
        assert resolve_tenant("demo-public-key", Settings()).is_unrestricted

    def test_anonymous_context_uses_the_default_tenant(self) -> None:
        assert anonymous_context(Settings()).tenant_id == "public"


class TestSectionPermissions:
    def test_unrestricted_permits_everything(self) -> None:
        assert TenantContext("public").permits_section("anything")

    def test_restricted_permits_only_its_sections(self) -> None:
        context = TenantContext("internal", ("concepts", "tasks"))
        assert context.permits_section("concepts")
        assert not context.permits_section("reference")

    def test_log_fields_exclude_the_raw_key(self) -> None:
        context = resolve_tenant("demo-restricted-key", Settings())
        fields = context.as_log_fields()
        assert "demo-restricted-key" not in str(fields)
        assert fields["tenant_id"] == "restricted"


class TestAudit:
    def test_records_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        context = TenantContext("internal", ("concepts",), "abc123")

        append_record(build_record(context, "query", "req-1", "answered", chunks=4), path)
        append_record(build_record(context, "query", "req-2", "suppressed", chunks=0), path)

        records = read_records(path)
        assert len(records) == 2
        assert records[0].action == "query"
        assert records[0].detail["chunks"] == 4
        assert records[1].outcome == "suppressed"

    def test_log_is_append_only(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        for i in range(3):
            append_record(build_record(TenantContext("public"), "query", f"r{i}", "ok"), path)
        assert len(read_records(path)) == 3

    def test_missing_log_reads_empty(self, tmp_path: Path) -> None:
        assert read_records(tmp_path / "absent.jsonl") == []

    def test_record_carries_the_fingerprint_not_the_key(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        context = resolve_tenant("demo-restricted-key", Settings())
        append_record(build_record(context, "query", "r", "ok"), path)
        assert "demo-restricted-key" not in path.read_text(encoding="utf-8")
