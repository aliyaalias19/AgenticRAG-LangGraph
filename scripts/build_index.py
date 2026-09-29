"""Embed the corpus and build both retrieval indexes.

Run order matters: embeddings are computed once and cached, then Qdrant and
the BM25 index are populated from the same chunk list, so the two backends can
never disagree about what is in the corpus.

    python scripts/build_index.py --embed      # GPU step, run once
    python scripts/build_index.py --load       # populate Qdrant and BM25
"""

import argparse
import sys
import time

from agentic_rag.config.settings import get_settings
from agentic_rag.ingest.pipeline import CHUNKS_FILENAME, read_chunks
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.retrieval.bm25 import INDEX_FILENAME, build_index
from agentic_rag.retrieval.embedder import (
    EMBEDDINGS_FILENAME,
    Embedder,
    load_embeddings,
    save_embeddings,
)
from agentic_rag.retrieval.store import VectorStore

logger = get_logger(__name__)


def embed_corpus(batch_size: int | None = None) -> None:
    """Embed every chunk and cache the vectors to disk."""
    settings = get_settings()
    if batch_size:
        settings.embedding.batch_size = batch_size

    chunks = read_chunks(settings.paths.processed_dir / CHUNKS_FILENAME)
    logger.info(
        "embedding_started",
        chunks=len(chunks),
        device=settings.embedding.device,
        batch_size=settings.embedding.batch_size,
    )

    embedder = Embedder(settings=settings)
    started = time.perf_counter()
    # contextual_text prefixes each chunk with its document title and heading
    # path, so a query matching the heading matches the chunk.
    result = embedder.encode([chunk.contextual_text for chunk in chunks])
    elapsed = time.perf_counter() - started

    save_embeddings(
        [c.chunk_id for c in chunks],
        result,
        settings.paths.processed_dir / EMBEDDINGS_FILENAME,
    )
    logger.info(
        "embedding_completed",
        chunks=len(chunks),
        seconds=round(elapsed, 1),
        chunks_per_second=round(len(chunks) / elapsed, 2) if elapsed else 0,
    )


def load_indexes(tenant: str = "public") -> None:
    """Populate Qdrant and build the BM25 index from cached embeddings."""
    settings = get_settings()
    chunks = read_chunks(settings.paths.processed_dir / CHUNKS_FILENAME)

    chunk_ids, embeddings = load_embeddings(settings.paths.processed_dir / EMBEDDINGS_FILENAME)
    if [c.chunk_id for c in chunks] != chunk_ids:
        message = (
            "Cached embeddings do not match the current chunks. "
            "Re-run with --embed after changing the corpus."
        )
        raise SystemExit(message)

    store = VectorStore(settings=settings)
    store.recreate_collection()
    store.upsert(chunks, embeddings, tenant=tenant)
    logger.info("qdrant_loaded", points=store.count())

    index = build_index(chunks, tenant=tenant)
    index.save(settings.paths.processed_dir / INDEX_FILENAME)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed", action="store_true", help="Compute embeddings")
    parser.add_argument("--load", action="store_true", help="Populate the indexes")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tenant", default="public")
    args = parser.parse_args(argv)

    configure_logging()
    if not (args.embed or args.load):
        parser.error("specify --embed, --load, or both")

    if args.embed:
        embed_corpus(args.batch_size)
    if args.load:
        load_indexes(args.tenant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
