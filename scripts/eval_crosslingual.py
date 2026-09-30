"""Measure cross-lingual retrieval over parallel EN/ZH documentation.

Dense and lexical retrieval are scored on identical queries. BM25 is expected
to do badly: Chinese and English share no content vocabulary. It does not do
as badly as that suggests, because technical documentation is partially
parallel at the token level -- identifiers, flags and command names survive
translation untouched. Quantifying that gap, and its direction, is the point.

    python scripts/eval_crosslingual.py --no-title --query-chars 150
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from agentic_rag.config.settings import get_settings
from agentic_rag.eval.crosslingual import (
    DEFAULT_QUERY_CHARS,
    load_pairs,
    make_query,
    rank_of,
    summarise,
)
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.retrieval.bm25 import INDEX_FILENAME

logger = get_logger(__name__)

CHUNKS_FILENAME = "chunks.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every pair")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-chars", type=int, default=DEFAULT_QUERY_CHARS)
    parser.add_argument("--no-title", action="store_true", help="omit doc_title from the query")
    parser.add_argument("--output", type=Path, default=Path("data/results/crosslingual_eval.json"))
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    pairs = load_pairs(settings.paths.processed_dir / CHUNKS_FILENAME)
    if args.limit:
        random.Random(settings.random_seed).shuffle(pairs)  # noqa: S311 - reproducibility
        pairs = sorted(pairs[: args.limit], key=lambda pair: pair.relative_id)
    logger.info("crosslingual_pairs_loaded", pairs=len(pairs))

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(settings.embedding.model_name, use_fp16=settings.embedding.use_fp16)
    client = QdrantClient(
        host=settings.vector_store.host,
        port=settings.vector_store.http_port,
        timeout=settings.vector_store.timeout_seconds,
    )

    bm25_path = settings.paths.processed_dir / INDEX_FILENAME
    bm25 = None
    if bm25_path.is_file():
        with bm25_path.open("rb") as handle:
            bm25 = pickle.load(handle)  # noqa: S301 - our own build artefact
        logger.info("bm25_index_loaded", documents=bm25.size)

    report = {
        "pairs": len(pairs),
        "top_k": args.top_k,
        "query_chars": args.query_chars,
        "title_in_query": not args.no_title,
        "directions": {},
    }

    for source, target in (("zh", "en"), ("en", "zh")):
        label = f"{source}->{target}"
        queries = [
            make_query(
                pair.second if source == "zh" else pair.first,
                args.query_chars,
                not args.no_title,
            )
            for pair in pairs
        ]
        truths = [pair.relative_id for pair in pairs]

        vectors = model.encode(
            queries,
            batch_size=settings.embedding.batch_size * 4,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]

        language_filter = Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=target))]
        )

        dense_ranks: list[int | None] = []
        bm25_ranks: list[int | None] = []
        for vector, relative_id, query in zip(vectors, truths, queries, strict=True):
            hits = client.query_points(
                collection_name=settings.vector_store.collection_name,
                query=vector.tolist(),
                using="dense",
                limit=args.top_k,
                query_filter=language_filter,
                with_payload=True,
            ).points
            dense_ranks.append(rank_of([hit.payload for hit in hits], relative_id))

            if bm25 is not None:
                lexical = bm25.search(query, limit=args.top_k, tenant="public", language=target)
                bm25_ranks.append(
                    rank_of([{"relative_id": hit.relative_id} for hit in lexical], relative_id)
                )

        report["directions"][label] = {"dense": summarise(dense_ranks, len(pairs))}
        if bm25 is not None:
            report["directions"][label]["bm25"] = summarise(bm25_ranks, len(pairs))

        print(f"\n{label}")
        for name, stats in report["directions"][label].items():
            print(
                f"  {name:6} R@1={stats['recall@1']:.3f}  R@5={stats['recall@5']:.3f}  "
                f"R@10={stats['recall@10']:.3f}  MRR={stats['mrr']:.3f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
