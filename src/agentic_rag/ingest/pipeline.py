"""End-to-end corpus ingestion pipeline."""

import json
from pathlib import Path

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.ingest.loader import load_documents
from agentic_rag.ingest.models import CorpusManifest, Document
from agentic_rag.ingest.repository import clone_or_update
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_FILENAME = "manifest.json"


def write_corpus(documents: list[Document], destination: Path) -> None:
    """Write documents to a JSON Lines file, one document per line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(document.model_dump_json() + "\n")
    logger.info("corpus_written", path=str(destination), count=len(documents))


def read_corpus(source: Path) -> list[Document]:
    """Read documents from a JSON Lines corpus file."""
    with source.open("r", encoding="utf-8") as handle:
        return [Document.model_validate_json(line) for line in handle if line.strip()]


def write_manifest(manifest: CorpusManifest, destination: Path) -> None:
    """Write the corpus manifest as formatted JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("manifest_written", path=str(destination))


def build_manifest(
    documents: list[Document],
    commit_sha: str,
    settings: Settings,
) -> CorpusManifest:
    """Construct a provenance manifest for the given document set."""
    return CorpusManifest(
        repo_url=settings.corpus.repo_url,
        repo_ref=settings.corpus.repo_ref,
        commit_sha=commit_sha,
        docs_subpath=settings.corpus.docs_subpath,
        document_count=len(documents),
        total_chars=sum(doc.char_count for doc in documents),
        min_document_chars=settings.corpus.min_document_chars,
        corpus_hash=CorpusManifest.compute_corpus_hash(documents),
    )


def ingest_corpus(settings: Settings | None = None) -> CorpusManifest:
    """Clone the source repository, load documents, and persist the corpus."""
    settings = settings or get_settings()
    settings.paths.ensure_exists()

    logger.info("ingestion_started", repo_url=settings.corpus.repo_url)

    repo_path = settings.paths.raw_dir / "source-repo"
    commit_sha = clone_or_update(
        repo_url=settings.corpus.repo_url,
        ref=settings.corpus.repo_ref,
        destination=repo_path,
    )

    docs_root = repo_path / settings.corpus.docs_subpath
    documents = load_documents(
        docs_root=docs_root,
        min_chars=settings.corpus.min_document_chars,
        excluded_prefixes=settings.corpus.excluded_path_prefixes,
    )

    write_corpus(documents, settings.paths.processed_dir / CORPUS_FILENAME)

    manifest = build_manifest(documents, commit_sha, settings)
    write_manifest(manifest, settings.paths.processed_dir / MANIFEST_FILENAME)

    logger.info(
        "ingestion_completed",
        document_count=manifest.document_count,
        total_chars=manifest.total_chars,
        corpus_hash=manifest.corpus_hash[:12],
    )
    return manifest
