"""
API-key authentication with tenant resolution.

Threat model
────────────
This module solves the *minimum* enterprise authentication problem:

  1. Refuse unauthenticated traffic when the operator has configured at
     least one API key.
  2. Map each accepted key to a tenant identifier that is propagated
     through every downstream component (ingestion, retrieval, audit
     log) so cross-tenant data leakage is structurally prevented, not
     merely conventionally avoided.

It is deliberately not a full IAM. For production you would replace
``verify_api_key`` with a call to an SSO provider (Okta, Auth0, Keycloak)
or mTLS termination at a reverse proxy. The interface — a FastAPI
``Depends`` that returns a ``TenantContext`` — does not change.

Modes
─────
  Demo / single-tenant:
    ``API_KEYS`` is unset. Every request is bound to
    ``DEFAULT_TENANT`` ("default" unless overridden). The demo runs
    unchanged with no headers required.

  Multi-tenant:
    ``API_KEYS`` is a JSON map of key → tenant_id. Every request must
    carry ``X-API-Key: <key>``. Missing or unrecognised keys return 401.
    The matched tenant_id is bound to the ``TenantContext`` and reaches
    every node of the graph.

Why a TenantContext object rather than a bare string
────────────────────────────────────────────────────
Future fields land here without rippling through signatures: user_id,
allowed_roles, ACL tags for per-document permissions, request-scoped
rate-limit overrides. Code that takes a ``TenantContext`` keeps working
when those fields are added.
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, Request

from core.config import get_settings


@dataclass(frozen=True)
class TenantContext:
    """Identity bound to a single request.

    ``tenant_id`` is mandatory and used as the partition key across the
    vector store, BM25 index, audit log, and metrics labels. ``api_key``
    is preserved for audit purposes only — never logged in plaintext.
    """
    tenant_id: str
    api_key: Optional[str] = None

    @property
    def api_key_fingerprint(self) -> str:
        """Short identifier for the API key, safe to log.

        Returns the first 4 and last 2 chars of the key, never the whole
        thing. Sufficient to correlate audit events without leaking the
        secret into logs.
        """
        if not self.api_key:
            return "-"
        if len(self.api_key) < 8:
            return "***"
        return f"{self.api_key[:4]}…{self.api_key[-2:]}"


def get_tenant_context(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> TenantContext:
    """FastAPI dependency returning the authenticated TenantContext.

    Resolution rules:
      * If ``API_KEYS`` is unset → return ``DEFAULT_TENANT`` (demo mode).
      * If the request path is in ``PUBLIC_PATHS`` → return
        ``DEFAULT_TENANT`` (health/docs/metrics bypass auth).
      * Otherwise the request must carry a matching ``X-API-Key``; we
        return the mapped tenant_id. Missing or wrong key → 401.

    Returns a frozen dataclass so downstream code cannot mutate the
    tenant_id mid-request.
    """
    settings = get_settings()

    if request.url.path in settings.public_path_set:
        return TenantContext(tenant_id=settings.default_tenant)

    key_map = settings.api_key_map
    if not key_map:
        return TenantContext(tenant_id=settings.default_tenant)

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    tenant = key_map.get(x_api_key)
    if tenant is None:
        # Constant detail string — do NOT reveal whether the key existed
        # at all. The 401 is identical for "no key" and "wrong key".
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return TenantContext(tenant_id=tenant, api_key=x_api_key)
