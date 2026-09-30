"""Tests for the cross-lingual evaluation helpers."""

import json
from pathlib import Path

import pytest

from agentic_rag.eval.crosslingual import (
    load_pairs,
    make_query,
    rank_of,
    summarise,
)


def write_chunks(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def chunk(relative_id: str, language: str, content: str, title: str = "T") -> dict:
    return {
        "relative_id": relative_id,
        "language": language,
        "content": content,
        "doc_title": title,
    }


class TestLoadPairs:
    def test_returns_only_documents_present_in_both_languages(self, tmp_path):
        path = write_chunks(
            tmp_path,
            [
                chunk("a", "en", "english a"),
                chunk("a", "zh", "中文 a"),
                chunk("b", "en", "english b only"),
                chunk("c", "zh", "只有中文"),
            ],
        )
        pairs = load_pairs(path)
        assert [pair.relative_id for pair in pairs] == ["a"]

    def test_keeps_the_first_chunk_of_each_document(self, tmp_path):
        path = write_chunks(
            tmp_path,
            [
                chunk("a", "en", "first"),
                chunk("a", "en", "second"),
                chunk("a", "zh", "第一"),
            ],
        )
        pairs = load_pairs(path)
        assert pairs[0].first["content"] == "first"

    def test_orders_pairs_deterministically(self, tmp_path):
        rows = []
        for relative_id in ("c", "a", "b"):
            rows.append(chunk(relative_id, "en", "x"))
            rows.append(chunk(relative_id, "zh", "y"))
        pairs = load_pairs(write_chunks(tmp_path, rows))
        assert [pair.relative_id for pair in pairs] == ["a", "b", "c"]


class TestMakeQuery:
    def test_includes_the_title_by_default(self):
        query = make_query(chunk("a", "en", "body", title="Pods"))
        assert query.startswith("Pods")

    def test_excludes_the_title_when_asked(self):
        query = make_query(chunk("a", "en", "body", title="Pods"), use_title=False)
        assert query == "body"

    def test_truncates_to_the_requested_length(self):
        query = make_query(chunk("a", "en", "x" * 500), chars=10)
        assert len(query) == 10


class TestRankOf:
    def test_returns_one_based_position(self):
        payloads = [{"relative_id": "a"}, {"relative_id": "b"}]
        assert rank_of(payloads, "b") == 2

    def test_returns_none_when_absent(self):
        assert rank_of([{"relative_id": "a"}], "z") is None

    def test_returns_the_first_match(self):
        payloads = [{"relative_id": "a"}, {"relative_id": "a"}]
        assert rank_of(payloads, "a") == 1


class TestSummarise:
    def test_counts_misses_against_the_total(self):
        stats = summarise([1, None], total=2)
        assert stats["recall@1"] == 0.5
        assert stats["mrr"] == 0.5

    def test_recall_cutoffs_are_cumulative(self):
        stats = summarise([1, 3, 7], total=3)
        assert stats["recall@1"] == pytest.approx(1 / 3, abs=1e-4)
        assert stats["recall@5"] == pytest.approx(2 / 3, abs=1e-4)
        assert stats["recall@10"] == 1.0

    def test_mrr_rewards_higher_ranks(self):
        assert summarise([1], 1)["mrr"] > summarise([5], 1)["mrr"]

    def test_empty_set_does_not_divide_by_zero(self):
        assert summarise([], total=0)["mrr"] == 0.0
