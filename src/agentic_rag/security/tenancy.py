"""Tenant identity and access scoping.

Tenant identity is derived from an authenticated API key and carried through
every layer as an explicit, typed object. It is never read back out of model
output, never taken from a request body, and never inferred from retrieved
text -- an LLM that is talked into claiming another tenant's identity still
cannot widen the filters applied here, because those filters were resolved
before generation began.
"""

import hashlib
from dataclasses import dataclass

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)


class AuthenticationError(Exception):
    """Raised when an API key does not map to a known tenant."""


@dataclass(frozen=True)
class TenantContext:
    """Resolved access scope for one authenticated caller."""

    tenant_id: str
    allowed_sections: tuple[str, ...] = ()
    key_fingerprint: str = ""

    @property
    def is_unrestricted(self) -> bool:
        """Return True when the tenant may read every section."""
        return not self.allowed_sections

    def permits_section(self, section: str) -> bool:
        """Return True when this tenant may read ``section``."""
        return self.is_unrestricted or section in self.allowed_sections

    def as_log_fields(self) -> dict[str, str]:
        """Return fields safe to attach to logs and traces."""
        return {
            "tenant_id": self.tenant_id,
            "key_fingerprint": self.key_fingerprint,
        }


def fingerprint(api_key: str) -> str:
    """Return a short, non-reversible fingerprint of an API key.

    Logs and audit records reference the fingerprint rather than the key, so a
    leaked log file cannot be replayed against the API.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def resolve_tenant(api_key: str, settings: Settings | None = None) -> TenantContext:
    """Resolve an API key into a tenant context.

    Raises ``AuthenticationError`` when the key is unknown. The error message
    deliberately omits the key and the list of valid keys.
    """
    settings = settings or get_settings()
    config = settings.security

    tenant_id = config.api_keys.get(api_key)
    if tenant_id is None:
        logger.warning("authentication_failed", key_fingerprint=fingerprint(api_key))
        message = "Unknown API key"
        raise AuthenticationError(message)

    context = TenantContext(
        tenant_id=tenant_id,
        allowed_sections=tuple(config.tenant_sections.get(tenant_id, ())),
        key_fingerprint=fingerprint(api_key),
    )
    logger.info("tenant_resolved", **context.as_log_fields())
    return context


def anonymous_context(settings: Settings | None = None) -> TenantContext:
    """Return the default public tenant context for unauthenticated reads."""
    settings = settings or get_settings()
    default = settings.security.default_tenant
    return TenantContext(
        tenant_id=default,
        allowed_sections=tuple(settings.security.tenant_sections.get(default, ())),
        key_fingerprint="anonymous",
    )
