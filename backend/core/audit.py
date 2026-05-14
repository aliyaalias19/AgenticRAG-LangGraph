"""
Append-only audit log.

Why this exists
───────────────
Enterprise document retrieval often falls under regulations that require
a tamper-evident record of who queried what and when (SOX, HIPAA Article
164.312, GDPR Article 30). The audit log is the minimum viable answer to
"a customer says yesterday at 3pm someone retrieved confidential
document X — show me the trail."

The log captures every state-changing or data-exposing event:

  ingest         a document was added to a tenant's corpus
  delete         a document was removed
  query          a question was answered with citations attached
  unauthorized   a request was rejected at the auth layer

Storage
───────
Newline-delimited JSON in ``${DATA_DIR}/audit.jsonl``. Append-only on the
process side, but any operator can edit the file — for true tamper-
evidence you'd ship the lines to a write-once store (e.g. an S3 bucket
with Object Lock, or Splunk with immutable indexing). The log format
chosen here is compatible with both: each line is self-contained JSON
that an external tail-ship process can forward without reparsing.

Why JSONL not a database
────────────────────────
Operational simplicity. JSONL is grep-able, tail-able, restorable from a
plain file backup, and pipes cleanly into a log aggregator. A relational
schema for audit logs is the upgrade path, not the starting point — once
queries by tenant_id + date range exceed a few seconds, move to
postgres. The writer below is the only place that needs to change.

Synchronous writes
──────────────────
``audit_event`` writes synchronously. The cost is a single fsync-less
append (~50µs on local disk, ~5ms on networked storage). We choose
synchronous to guarantee that an audit row precedes the response: if the
server crashes before the response is sent, the audit row is already on
disk. An async fire-and-forget pattern would risk dropping rows.
"""

import datetime
import json
import logging
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from core.config import get_settings

logger = logging.getLogger(__name__)

# A single lock per process guards the append. The OS already guarantees
# that small writes to an O_APPEND file are atomic on most platforms, but
# this lock also serialises encoding so two threads can't interleave
# half-written JSON when CPython's GIL switches mid-encode.
_write_lock = threading.Lock()


def _audit_path() -> Path:
    settings = get_settings()
    p = Path(settings.data_dir) / "audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def audit_event(
    action: str,
    tenant_id: str,
    actor_fingerprint: str = "-",
    *,
    request_id: Optional[str] = None,
    detail: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append a structured audit event.

    Parameters
    ----------
    action:
        Short verb identifying what happened. Recommended values:
        ``ingest``, ``delete``, ``query``, ``unauthorized``,
        ``config_change``. Use lowercase; downstream filters key on it.
    tenant_id:
        The tenant whose data was touched. Required.
    actor_fingerprint:
        Short, non-secret identifier of the caller. Use
        ``TenantContext.api_key_fingerprint`` — never log raw keys.
    request_id:
        The X-Request-Id correlation ID. If None we read it from the
        current contextvar so callers don't have to thread it manually.
    detail:
        Free-form additional context (filename, query text, citation
        count). Values must be JSON-serialisable; non-serialisable
        values are str()-coerced rather than failing the audit write.
    """
    if request_id is None:
        # Avoid a hard dependency on core.logging at module import time.
        try:
            from core.logging import get_request_id
            request_id = get_request_id()
        except Exception:
            request_id = "-"

    row = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "action": action,
        "tenant_id": tenant_id,
        "actor": actor_fingerprint,
        "request_id": request_id,
    }
    if detail:
        # Coerce non-serialisable values; do not raise from the audit
        # path — losing an audit row is worse than losing fidelity.
        safe_detail = {}
        for k, v in detail.items():
            try:
                json.dumps(v)
                safe_detail[k] = v
            except (TypeError, ValueError):
                safe_detail[k] = str(v)
        row["detail"] = safe_detail

    line = json.dumps(row, ensure_ascii=False) + "\n"
    try:
        with _write_lock:
            with _audit_path().open("a", encoding="utf-8") as f:
                f.write(line)
    except OSError as exc:
        # An unwritable audit log is a serious operational issue but
        # should not return 500 to the user — we log loudly and continue.
        # A proper deployment would also page on this.
        logger.error("AUDIT WRITE FAILED", extra={"err": str(exc), "row": row})

    # Mirror to Prometheus so audit volume is observable per tenant.
    # Imported lazily to avoid a hard dependency on prometheus_client
    # from any module that imports core/audit during cold-start probing.
    try:
        from infrastructure.prometheus import AUDIT_EVENTS
        AUDIT_EVENTS.labels(action=action, tenant=tenant_id).inc()
    except Exception:
        pass
