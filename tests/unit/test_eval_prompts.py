"""Tests for evaluation prompt construction."""

import pytest

from agentic_rag.eval.prompts import (
    TYPE_INSTRUCTIONS,
    build_generation_prompt,
)


def test_all_question_types_have_instructions() -> None:
    assert set(TYPE_INSTRUCTIONS) == {
        "direct",
        "paraphrased",
        "multi_hop",
        "scenario",
    }


@pytest.mark.parametrize("question_type", sorted(TYPE_INSTRUCTIONS))
def test_prompt_includes_type_instruction(question_type: str) -> None:
    prompt = build_generation_prompt(
        question_type=question_type,
        count=2,
        title="Pods",
        section="concepts",
        content="Some content.",
    )
    assert TYPE_INSTRUCTIONS[question_type] in prompt


def test_prompt_includes_document_and_metadata() -> None:
    prompt = build_generation_prompt(
        question_type="direct",
        count=3,
        title="Nodes",
        section="concepts",
        content="Node content here.",
    )

    assert 'title="Nodes"' in prompt
    assert 'section="concepts"' in prompt
    assert "Node content here." in prompt
    assert "exactly 3 questions" in prompt


def test_unknown_type_raises() -> None:
    with pytest.raises(KeyError):
        build_generation_prompt(
            question_type="nonsense",
            count=1,
            title="T",
            section="s",
            content="c",
        )
