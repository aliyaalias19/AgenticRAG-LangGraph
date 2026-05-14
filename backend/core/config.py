"""
Centralised configuration.

Design choices that matter at review time:

1. LOCAL-FIRST DEFAULTS.
   `LLM_PROVIDER` and `EMBEDDING_PROVIDER` both default to ``ollama``.
   Cloud providers (OpenAI, Bedrock) are explicit opt-in. This reflects the
   target environment: on-prem, air-gapped, storage-aware enterprise AI.
   Cloud APIs are not assumed reachable from a customer site.

2. LLM AND EMBEDDING PROVIDERS ARE DECOUPLED.
   Customers commonly want local embeddings (data sovereignty, no egress)
   with a cloud LLM, or vice versa. The previous single ``use_openai`` flag
   coupled them and made that impossible.

3. FAIL-FAST VALIDATION AT STARTUP.
   ``Settings.validate_at_startup()`` is called from the FastAPI lifespan
   handler. If a provider is selected but its credentials are missing, or
   if ``auto`` mode finds zero usable providers, the process raises
   :class:`core.exceptions.ConfigurationError` and refuses to serve traffic.
   Misconfiguration is caught at boot, not at the first user request.

4. PLAIN PYDANTIC v2 BaseModel.
   ``pydantic-settings`` is deliberately not used — it pins a different
   pydantic minor that conflicts with the rest of the stack.
"""

import os
from functools import lru_cache
from typing import List, Literal

from pydantic import BaseModel, field_validator

from core.exceptions import ConfigurationError

ProviderName = Literal["auto", "ollama", "openai", "bedrock"]


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be an integer, got {raw!r}") from exc


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _env_choice(key: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.getenv(key, default).lower().strip()
    if value not in choices:
        raise ConfigurationError(
            f"{key} must be one of {choices}, got {value!r}"
        )
    return value


def _aws_credentials_present() -> bool:
    """Best-effort credential check — does NOT call AWS."""
    try:
        import boto3
        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def _ollama_reachable(base_url: str, timeout: float = 1.0) -> bool:
    """Probe Ollama with a short HEAD to /api/tags. Non-blocking."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


class Settings(BaseModel):
    # ── Provider selection (the headline change) ─────────────────────────────
    # Explicit values: "ollama" | "openai" | "bedrock"
    # "auto" tries: ollama → openai → bedrock (LOCAL FIRST).
    llm_provider:       ProviderName = _env_choice("LLM_PROVIDER", "ollama", ("auto", "ollama", "openai", "bedrock"))  # type: ignore[assignment]
    embedding_provider: ProviderName = _env_choice("EMBEDDING_PROVIDER", "ollama", ("auto", "ollama", "openai", "bedrock"))  # type: ignore[assignment]

    # ── AWS Bedrock ──────────────────────────────────────────────────────────
    aws_region:         str = _env("AWS_REGION", "us-east-1")
    aws_default_region: str = _env("AWS_DEFAULT_REGION", "ap-southeast-1")
    bedrock_embed_model: str = _env("BEDROCK_EMBED_MODEL", "cohere.embed-multilingual-v3")

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key:     str = _env("OPENAI_API_KEY", "")
    openai_model:       str = _env("OPENAI_MODEL", "gpt-4o-mini")
    openai_embed_model: str = _env("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # ── Ollama (default local provider) ──────────────────────────────────────
    ollama_base_url:    str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model:          str = _env("LLM_MODEL", "llama3.2")
    embed_model:        str = _env("EMBED_MODEL", "nomic-embed-text")

    # ── Storage ──────────────────────────────────────────────────────────────
    # Qdrant deployment modes:
    #   QDRANT_URL set     → connect to a Qdrant server (multi-worker, HA, snapshots)
    #   QDRANT_URL empty   → embedded mode using QDRANT_PATH on local disk (dev only)
    qdrant_url:         str = _env("QDRANT_URL", "")
    qdrant_api_key:     str = _env("QDRANT_API_KEY", "")
    qdrant_path:        str = _env("QDRANT_PATH", "./qdrant_data")
    # ``chroma_path`` is retained for back-compat reading of the legacy audit
    # log location; it is no longer used by the vector store itself.
    chroma_path:        str = _env("CHROMA_PATH", "./chroma")
    data_dir:           str = _env("DATA_DIR", "./data")
    collection_name:    str = _env("COLLECTION_NAME", "rag_docs")

    # ── Retrieval tuning ─────────────────────────────────────────────────────
    vector_top_k:       int = _env_int("VECTOR_TOP_K", 3)
    bm25_top_k:         int = _env_int("BM25_TOP_K", 2)
    final_top_k:        int = _env_int("FINAL_TOP_K", 3)
    # Calibrated against the new grader prompt (see agents/nodes/grading.py).
    # A chunk that lets the reader infer the answer scores 6-8 on the new
    # rubric, so we accept down to 3. Lower than 3 means the chunks are
    # off-topic; that's a real signal to rewrite.
    min_relevance_score: int = _env_int("MIN_RELEVANCE_SCORE", 3)
    max_steps:          int = _env_int("MAX_STEPS", 5)
    max_rewrites:       int = _env_int("MAX_REWRITES", 2)

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size:         int = _env_int("CHUNK_SIZE", 800)
    chunk_overlap:      int = _env_int("CHUNK_OVERLAP", 150)

    # ── API ──────────────────────────────────────────────────────────────────
    api_title:          str = _env("API_TITLE", "Agentic RAG API")
    api_version:        str = _env("API_VERSION", "1.0.0")
    debug:              bool = _env_bool("DEBUG", False)
    log_level:          str = _env("LOG_LEVEL", "INFO").upper()
    # Comma-separated; "*" allowed for dev only. Defaults to localhost ports.
    cors_origins:       str = _env("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

    # ── Reranker ─────────────────────────────────────────────────────────────
    # Default: BAAI/bge-reranker-v2-m3 — multilingual, multi-domain, ~570MB.
    # Previously cross-encoder/ms-marco-MiniLM-L-6-v2, which is English-only
    # and web-passage tuned. The bge-v2 family is the current best
    # open-source choice for enterprise document retrieval — handles
    # Bahasa, Mandarin, and structured-document text far better than the
    # MiniLM cross-encoders. Override via RERANKER_MODEL for English-only
    # corpora where the smaller model is faster.
    reranker_model:     str = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranking_enabled:  bool = _env_bool("RERANKING_ENABLED", True)

    # ── Multi-tenancy + auth ─────────────────────────────────────────────────
    # API_KEYS is a JSON object mapping API key → tenant_id, e.g.:
    #   API_KEYS={"sk-alpha-123":"tenant-alpha","sk-beta-456":"tenant-beta"}
    # When empty (default), the API runs in single-tenant demo mode and every
    # request is treated as ``DEFAULT_TENANT``. Setting at least one entry
    # turns on enforcement: unrecognised or missing X-API-Key returns 401.
    # See core/auth.py for the resolution logic.
    api_keys_raw:       str = _env("API_KEYS", "")
    default_tenant:     str = _env("DEFAULT_TENANT", "default")
    # Endpoints excluded from auth even when API_KEYS is configured. The
    # health endpoint must be reachable by the orchestrator's liveness probe.
    public_paths:       str = _env("PUBLIC_PATHS", "/health,/metrics,/docs,/redoc,/openapi.json")

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("min_relevance_score")
    @classmethod
    def score_in_range(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError("min_relevance_score must be 1-10")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 800)
        if v >= size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({size})")
        return v

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, v: str) -> str:
        if v.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"log_level {v!r} not recognised")
        return v.upper()

    # ── Convenience accessors ────────────────────────────────────────────────

    @property
    def cors_origin_list(self) -> List[str]:
        items = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return items or ["*"]

    @property
    def api_key_map(self) -> dict:
        """Parsed API_KEYS env var. Empty dict ⇒ single-tenant demo mode."""
        if not self.api_keys_raw.strip():
            return {}
        try:
            import json
            parsed = json.loads(self.api_keys_raw)
            if not isinstance(parsed, dict):
                raise ConfigurationError("API_KEYS must be a JSON object")
            return {str(k): str(v) for k, v in parsed.items()}
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(f"API_KEYS is not valid JSON: {exc}") from exc

    @property
    def public_path_set(self) -> set:
        return {p.strip() for p in self.public_paths.split(",") if p.strip()}

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key_map)

    @property
    def use_openai(self) -> bool:
        """DEPRECATED. Retained only for backward compatibility with older code
        paths that still check this flag. New code should branch on
        ``llm_provider`` or ``embedding_provider`` directly.
        """
        return self.llm_provider == "openai" or self.embedding_provider == "openai"

    # ── Resolved providers (after auto-resolution) ───────────────────────────

    def resolve_llm_provider(self) -> str:
        """Resolve ``auto`` to a concrete provider name, local-first."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if _ollama_reachable(self.ollama_base_url):
            return "ollama"
        if self.openai_api_key:
            return "openai"
        if _aws_credentials_present():
            return "bedrock"
        raise ConfigurationError(
            "LLM_PROVIDER=auto but no provider is reachable. "
            "Start Ollama, set OPENAI_API_KEY, or configure AWS credentials."
        )

    def resolve_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        if _ollama_reachable(self.ollama_base_url):
            return "ollama"
        if self.openai_api_key:
            return "openai"
        if _aws_credentials_present():
            return "bedrock"
        raise ConfigurationError(
            "EMBEDDING_PROVIDER=auto but no provider is reachable."
        )

    # ── Startup validation ───────────────────────────────────────────────────

    def validate_at_startup(self) -> List[str]:
        """Fail-fast configuration check.

        Returns a list of warning strings (non-fatal). Raises
        :class:`ConfigurationError` on anything that would prevent the system
        from serving traffic correctly.

        Called from the FastAPI lifespan handler. Misconfiguration surfaces
        at boot, not at the first request.
        """
        warnings: List[str] = []

        # --- Explicit provider credentials must be present ---
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        if self.llm_provider == "bedrock" and not _aws_credentials_present():
            raise ConfigurationError(
                "LLM_PROVIDER=bedrock but no AWS credentials are available."
            )
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        if self.embedding_provider == "bedrock" and not _aws_credentials_present():
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=bedrock but no AWS credentials are available."
            )

        # --- Ollama reachability (warn only — operator may start it after boot) ---
        if self.llm_provider == "ollama" and not _ollama_reachable(self.ollama_base_url):
            warnings.append(
                f"LLM_PROVIDER=ollama but {self.ollama_base_url} is not reachable yet. "
                "First request will retry."
            )
        if self.embedding_provider == "ollama" and not _ollama_reachable(self.ollama_base_url):
            warnings.append(
                f"EMBEDDING_PROVIDER=ollama but {self.ollama_base_url} is not reachable yet."
            )

        # --- Auto-mode resolution preview ---
        if self.llm_provider == "auto":
            resolved = self.resolve_llm_provider()
            warnings.append(f"LLM_PROVIDER=auto resolved to {resolved}")
        if self.embedding_provider == "auto":
            resolved = self.resolve_embedding_provider()
            warnings.append(f"EMBEDDING_PROVIDER=auto resolved to {resolved}")

        # --- Cross-provider note ---
        # Different providers for LLM vs embeddings is supported but worth
        # surfacing — it implies an extra network hop and a divergent failure
        # surface. We do NOT raise, only inform.
        llm_concrete = self.resolve_llm_provider()
        emb_concrete = self.resolve_embedding_provider()
        if llm_concrete != emb_concrete:
            warnings.append(
                f"LLM uses {llm_concrete!r} but embeddings use {emb_concrete!r}. "
                "Verify this is intentional (network egress / cost / latency)."
            )

        # --- Storage paths writable ---
        for path_attr in ("data_dir", "chroma_path"):
            path = getattr(self, path_attr)
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as exc:
                raise ConfigurationError(
                    f"{path_attr.upper()}={path!r} is not creatable: {exc}"
                ) from exc

        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings. Reads env vars once per process."""
    return Settings()
