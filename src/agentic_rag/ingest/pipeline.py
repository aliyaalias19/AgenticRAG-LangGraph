"""End-to-end corpus ingestion pipeline."""

import json
from pathlib import Path

from agentic_rag.config.settings import Settings, SourceSettings, get_settings
from agentic_rag.ingest.chunker import chunk_documents
from agentic_rag.ingest.loader import load_documents
from agentic_rag.ingest.models import Chunk, CorpusManifest, Document, SourceRecord
from agentic_rag.ingest.repository import clone_or_update
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_FILENAME = "manifest.json"
CHUNKS_FILENAME = "chunks.jsonl"


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
    chunks: list[Chunk],
    sources: list[SourceRecord],
    settings: Settings,
) -> CorpusManifest:
    """Construct a provenance manifest for the ingested corpus."""
    return CorpusManifest(
        sources=sources,
        document_count=len(documents),
        total_chars=sum(doc.char_count for doc in documents),
        min_document_chars=settings.corpus.min_document_chars,
        corpus_hash=CorpusManifest.compute_corpus_hash(documents),
        chunk_count=len(chunks),
        chunk_max_chars=settings.chunk.max_chars,
        chunk_overlap_chars=settings.chunk.overlap_chars,
    )


def ingest_source(
    source: SourceSettings,
    settings: Settings,
) -> tuple[list[Document], SourceRecord]:
    """Clone one source repository and load its documents."""
    logger.info("source_ingestion_started", source=source.name)

    repo_path = settings.paths.raw_dir / source.name
    commit_sha = clone_or_update(
        repo_url=source.repo_url,
        ref=source.repo_ref,
        destination=repo_path,
    )

    documents = load_documents(
        docs_root=repo_path / source.docs_subpath,
        min_chars=settings.corpus.min_document_chars,
        excluded_prefixes=source.excluded_path_prefixes,
        source_name=source.name,
        language=source.language,
    )

    record = SourceRecord(
        name=source.name,
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        commit_sha=commit_sha,
        docs_subpath=source.docs_subpath,
        language=source.language,
        document_count=len(documents),
    )

    logger.info(
        "source_ingestion_completed",
        source=source.name,
        language=source.language,
        document_count=len(documents),
    )
    return documents, record


def ingest_corpus(settings: Settings | None = None) -> CorpusManifest:
    """Ingest every configured source into a single corpus."""
    settings = settings or get_settings()
    settings.paths.ensure_exists()

    logger.info("ingestion_started", source_count=len(settings.corpus.sources))

    documents: list[Document] = []
    records: list[SourceRecord] = []

    for source in settings.corpus.sources:
        source_documents, record = ingest_source(source, settings)
        documents.extend(source_documents)
        records.append(record)

    write_corpus(documents, settings.paths.processed_dir / CORPUS_FILENAME)

    chunks = chunk_documents(
        documents,
        max_chars=settings.chunk.max_chars,
        overlap_chars=settings.chunk.overlap_chars,
        min_chars=settings.chunk.min_chars,
        max_heading_depth=settings.chunk.max_heading_depth,
    )
    write_chunks(chunks, settings.paths.processed_dir / CHUNKS_FILENAME)

    manifest = build_manifest(documents, chunks, records, settings)
    write_manifest(manifest, settings.paths.processed_dir / MANIFEST_FILENAME)

    logger.info(
        "ingestion_completed",
        document_count=manifest.document_count,
        chunk_count=len(chunks),
        corpus_hash=manifest.corpus_hash[:12],
    )
    return manifest


def write_chunks(chunks: list[Chunk], destination: Path) -> None:
    """Write chunks to a JSON Lines file, one chunk per line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    logger.info("chunks_written", path=str(destination), count=len(chunks))


def read_chunks(source: Path) -> list[Chunk]:
    """Read chunks from a JSON Lines file."""
    with source.open("r", encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]
