"""Prompt templates for evaluation question generation."""

PROMPT_VERSION = "v1"

GENERATION_SYSTEM = """\
You write evaluation questions for a Kubernetes documentation retrieval system.

Your questions are used to measure whether a search system can find the right \
passage. This means the questions must be answerable from the document you are \
given, and must be phrased the way a real engineer would ask them.

Rules that apply to every question:
- The answer must be present in the provided document.
- Never refer to "the document", "this page", "the text above", or similar. The \
question is asked by someone who has not seen the document.
- Ask about substance, not document structure. Never ask what a section is called \
or how many items a list contains.
- Do not ask about version numbers, dates, or release notes.
- Keep each question under 30 words.

Return only valid JSON. No preamble, no markdown fences, no commentary.\
"""

TYPE_INSTRUCTIONS: dict[str, str] = {
    "direct": """\
Write questions that use the same technical terminology as the document. These \
are questions from someone who already knows the correct terms and wants a \
specific fact.

Example: "What does the kubectl drain command do to DaemonSet pods?"\
""",
    "paraphrased": """\
Write questions that avoid the document's distinctive vocabulary. Describe the \
concept in different words. Do not reuse the specific technical nouns and verbs \
the document uses for its central subject.

This is the most important constraint: if the document says "drain", your \
question must not say "drain". If it says "taint", your question must not say \
"taint". Describe the situation instead.

Example, for a document about draining nodes: "How do I move all the running \
workloads off a machine before I shut it down for maintenance?"\
""",
    "multi_hop": """\
Write questions whose answer requires combining two or more separate facts from \
the document. A single sentence should not be enough to answer them.

Example: "How does the default eviction behaviour differ between pods managed by \
a ReplicaSet and pods created directly?"\
""",
    "scenario": """\
Write questions from the perspective of an engineer with a problem. Describe a \
symptom or a goal, not a concept. The question should not name the mechanism that \
solves it.

Example: "My node needs a kernel upgrade but I cannot lose the workloads running \
on it. What should I do first?"\
""",
}

GENERATION_TEMPLATE = """\
{type_instruction}

Write exactly {count} questions of this type about the following documentation.

Return JSON in exactly this shape:
{{"questions": ["first question", "second question"]}}

<document title="{title}" section="{section}">
{content}
</document>\
"""


def build_generation_prompt(
    *,
    question_type: str,
    count: int,
    title: str,
    section: str,
    content: str,
) -> str:
    """Return the user prompt for generating questions of one type."""
    return GENERATION_TEMPLATE.format(
        type_instruction=TYPE_INSTRUCTIONS[question_type],
        count=count,
        title=title,
        section=section,
        content=content,
    )


LABELLING_SYSTEM = """\
You identify which documentation passages contain the answer to a question.

You are given a question and a numbered list of passages taken from a single \
document. Decide which passages a reader would need in order to answer the \
question correctly and completely.

Be strict:
- Include a passage only if it contains information required for the answer.
- A passage that merely mentions the topic is not sufficient.
- If no passage answers the question, return an empty list.
- Most questions need one or two passages. Returning more than three means you \
are probably including passages that merely relate to the topic.

Return only valid JSON. No preamble, no markdown fences, no commentary.\
"""

LABELLING_TEMPLATE = """\
Question: {question}

Passages:
{passages}

Return JSON in exactly this shape:
{{"passage_numbers": [1, 3], "reasoning": "one short sentence"}}\
"""


def build_labelling_prompt(question: str, passages: list[str]) -> str:
    """Return the user prompt for labelling gold passages."""
    numbered = "\n\n".join(f"[{index}]\n{text}" for index, text in enumerate(passages, start=1))
    return LABELLING_TEMPLATE.format(question=question, passages=numbered)
