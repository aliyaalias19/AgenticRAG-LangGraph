"""Append-only audit log for tenant-scoped operations."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agentic_rag.obs.logging import get_logger
from agentic_rag.security.tenancy import TenantContext

logger = get_logger(__name__)

AUDIT_FILENAME = "audit.jsonl"


@dataclass(frozen=True)
class AuditRecord:
    """One auditable action taken on behalf of a tenant."""

    timestamp: str
    tenant_id: str
    key_fingerprint: str
    action: str
    request_id: str
    outcome: str
    detail: dict[str, str | int | float | bool] = field(default_factory=dict)


def build_record(
    context: TenantContext,
    action: str,
    request_id: str,
    outcome: str,
    **detail: str | int | float | bool,
) -> AuditRecord:
    """Construct an audit record for a tenant-scoped action."""
    return AuditRecord(
        timestamp=datetime.now(UTC).isoformat(),
        tenant_id=context.tenant_id,
        key_fingerprint=context.key_fingerprint,
        action=action,
        request_id=request_id,
        outcome=outcome,
        detail=dict(detail),
    )


def append_record(record: AuditRecord, destination: Path) -> None:
    """Append one audit record to the audit log."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records(source: Path) -> list[AuditRecord]:
    """Read every audit record from the log."""
    if not source.is_file():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(AuditRecord(**json.loads(line)))
    return records
