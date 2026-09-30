"""Cross-lingual retrieval evaluation over parallel EN/ZH documentation.

The corpus contains 598 documents that exist in both languages under the same
relative_id. That is hand-aligned parallel data: a passage and its translation
are, by construction, about the same thing. So a passage in one language makes
a legitimate query whose correct answer is the document with the same
relative_id in the other language -- ground truth with no labelling required.

Dense and lexical retrieval are scored on identical queries. BM25 is expected
to fail: Chinese and English share no content vocabulary, only identifiers and
command names. Quantifying that gap is the point.
"""

import argparse
import collections
import json
import random
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

CHUNKS = Path("data/processed/chunks.jsonl")
BM25_INDEX = Path("data/processed/bm25_index.pkl")
QUERY_CHARS = 400


def load_pairs():
    rows = [json.loads(line) for line in CHUNKS.open(encoding="utf-8") if line.strip()]
    by_key = collections.defaultdict(list)
    for row in rows:
        by_key[(row["relative_id"], row["language"])].append(row)

    relatives = collections.defaultdict(set)
    for relative_id, language in by_key:
        relatives[relative_id].add(language)

    pairs = []
    for relative_id, languages in relatives.items():
        if {"en", "zh"} <= languages:
            en = by_key[(relative_id, "en")][0]
            zh = by_key[(relative_id, "zh")][0]
            pairs.append((relative_id, en, zh))
    pairs.sort(key=lambda p: p[0])
    return pairs


def make_query(chunk, chars, use_title):
    text = f"{chunk['doc_title']}\n{chunk['content']}" if use_title else chunk["content"]
    return text[:chars]


def rank_of(results, relative_id):
    for position, payload in enumerate(results, start=1):
        if payload.get("relative_id") == relative_id:
            return position
    return None


def summarise(ranks, total):
    found = [r for r in ranks if r is not None]
    return {
        "queries": total,
        "recall@1": round(sum(1 for r in found if r <= 1) / total, 4),
        "recall@5": round(sum(1 for r in found if r <= 5) / total, 4),
        "recall@10": round(sum(1 for r in found if r <= 10) / total, 4),
        "mrr": round(sum(1.0 / r for r in found) / total, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-chars", type=int, default=QUERY_CHARS)
    parser.add_argument("--no-title", action="store_true", help="exclude doc_title from the query")
    parser.add_argument("--output", type=Path, default=Path("data/results/crosslingual_eval.json"))
    args = parser.parse_args()

    pairs = load_pairs()
    if args.limit:
        random.Random(42).shuffle(pairs)  # noqa: S311 - reproducibility, not cryptography
        pairs = pairs[: args.limit]
        pairs.sort(key=lambda p: p[0])
    print(f"parallel document pairs: {len(pairs)}")

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    client = QdrantClient(host="localhost", port=6333, timeout=120)

    try:
        import pickle

        with BM25_INDEX.open("rb") as handle:
            bm25 = pickle.load(handle)  # noqa: S301 - our own build artefact
        print(f"bm25 index loaded: {bm25.size} documents")
    except Exception as exc:
        print(f"bm25 unavailable ({exc}); dense only")
        bm25 = None

    report = {
        "pairs": len(pairs),
        "top_k": args.top_k,
        "query_chars": args.query_chars,
        "title_in_query": not args.no_title,
        "directions": {},
    }

    for source_lang, target_lang in (("zh", "en"), ("en", "zh")):
        label = f"{source_lang}->{target_lang}"
        queries, truths = [], []
        for relative_id, en, zh in pairs:
            queries.append(
                make_query(zh if source_lang == "zh" else en, args.query_chars, not args.no_title)
            )
            truths.append(relative_id)

        encoded = model.encode(
            queries,
            batch_size=32,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]

        language_filter = Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=target_lang))]
        )

        dense_ranks, bm25_ranks = [], []
        for vector, relative_id, query in zip(encoded, truths, queries, strict=True):
            hits = client.query_points(
                collection_name="k8s_chunks",
                query=vector.tolist(),
                using="dense",
                limit=args.top_k,
                query_filter=language_filter,
                with_payload=True,
            ).points
            dense_ranks.append(rank_of([h.payload for h in hits], relative_id))

            if bm25 is not None:
                lexical = bm25.search(
                    query, limit=args.top_k, tenant="public", language=target_lang
                )
                bm25_ranks.append(
                    rank_of([{"relative_id": h.relative_id} for h in lexical], relative_id)
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
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
