"""Tests for supervised fine-tuning dataset construction."""

from pathlib import Path

from agentic_rag.train.dataset import (
    InstructionExample,
    build_instruction,
    read_jsonl,
    split_by_document,
    write_jsonl,
)


def example(example_id: str, doc_id: str, **kw: object) -> InstructionExample:
    defaults = {
        "instruction": "Question: q\n\nPassages:\n[1] body",
        "response": "answer [1]",
        "question_type": "factual",
    }
    defaults.update(kw)
    return InstructionExample(
        example_id=example_id,
        source_doc_id=doc_id,
        **defaults,  # type: ignore[arg-type]
    )


class TestInstructionFormat:
    def test_passages_are_numbered_from_one(self) -> None:
        text = build_instruction("What is a pod?", ["first", "second"])
        assert "[1] first" in text
        assert "[2] second" in text

    def test_question_precedes_passages(self) -> None:
        text = build_instruction("q", ["p"])
        assert text.index("Question:") < text.index("Passages:")

    def test_chat_format_has_three_turns(self) -> None:
        messages = example("e1", "d1").to_messages()
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]

    def test_system_prompt_requires_citations(self) -> None:
        system = example("e1", "d1").to_messages()[0]["content"]
        assert "[1]" in system
        assert "say so" in system


class TestDocumentDisjointSplit:
    def _examples(self, documents: int, per_document: int) -> list[InstructionExample]:
        return [
            example(f"d{d}-e{e}", f"doc{d}") for d in range(documents) for e in range(per_document)
        ]

    def test_no_document_appears_on_both_sides(self) -> None:
        split = split_by_document(self._examples(20, 3), 0.2, seed=42)
        assert split.is_disjoint

    def test_all_examples_are_retained(self) -> None:
        examples = self._examples(20, 3)
        split = split_by_document(examples, 0.2, seed=42)
        assert len(split.train) + len(split.heldout) == len(examples)

    def test_heldout_fraction_is_approximately_respected(self) -> None:
        split = split_by_document(self._examples(100, 2), 0.2, seed=42)
        assert 15 <= len(split.heldout_documents) <= 25

    def test_split_is_reproducible(self) -> None:
        examples = self._examples(30, 2)
        first = split_by_document(examples, 0.2, seed=7)
        second = split_by_document(examples, 0.2, seed=7)
        assert [e.example_id for e in first.heldout] == [e.example_id for e in second.heldout]

    def test_different_seeds_differ(self) -> None:
        examples = self._examples(30, 2)
        a = split_by_document(examples, 0.2, seed=1).heldout_documents
        b = split_by_document(examples, 0.2, seed=2).heldout_documents
        assert a != b

    def test_examples_from_one_document_stay_together(self) -> None:
        """Splitting by example would leak near-duplicates across the split."""
        split = split_by_document(self._examples(20, 4), 0.25, seed=3)
        for document in split.heldout_documents:
            assert document not in split.train_documents

    def test_summary_reports_disjointness(self) -> None:
        summary = split_by_document(self._examples(10, 2), 0.2, seed=1).summary()
        assert summary["document_disjoint"] is True
        assert summary["train_examples"] + summary["heldout_examples"] == 20


class TestPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        examples = [example("e1", "d1"), example("e2", "d2", is_abstention=True)]
        path = tmp_path / "sft.jsonl"

        write_jsonl(examples, path)
        restored = read_jsonl(path)

        assert [e.example_id for e in restored] == ["e1", "e2"]
        assert restored[1].is_abstention

    def test_file_has_one_line_per_example(self, tmp_path: Path) -> None:
        path = tmp_path / "sft.jsonl"
        write_jsonl([example(f"e{i}", "d1") for i in range(5)], path)
        assert len(path.read_text("utf-8").strip().splitlines()) == 5

    def test_messages_are_written_for_the_trainer(self, tmp_path: Path) -> None:
        import json

        path = tmp_path / "sft.jsonl"
        write_jsonl([example("e1", "d1")], path)
        payload = json.loads(path.read_text("utf-8").splitlines()[0])
        assert len(payload["messages"]) == 3
