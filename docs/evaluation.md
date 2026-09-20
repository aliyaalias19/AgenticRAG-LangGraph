# Evaluation set

## Purpose

Retrieval quality claims are only meaningful against a fixed question set with
known correct answers. This document describes how that set was built, what it
measures, and where it is weak.

All numbers in this repository are measured against corpus hash
`3df622e7d31539d10b17803b9da385411a298dfd7fa40c925f1027658c649b47`, built from
`kubernetes/website` at commit `b290c15b046f36d7b228f3ef5728ca08e86070bc`.

## Why not generate questions from chunks

The obvious approach is to show a model one chunk and ask for a question it
answers. This produces an evaluation set that is trivially easy: the question is
a paraphrase of the chunk, so any retriever matching on vocabulary finds it.
Measured recall would be high and would not distinguish a good retrieval system
from a bad one.

Questions here are generated from **whole documents**, and gold chunks are
identified by a **separate pass** that does not see the generation prompt.

## Three stages

| Stage | Input | Output | Can reject |
|---|---|---|---|
| 1. Generate | Whole document + type instruction | Questions | — |
| 2. Label | Question + all chunks of its source document | Candidate gold chunks | Stage 1 |
| 3. Verify | Question + one chunk, in isolation | Confirmed gold chunks | Stage 2 |

Each stage can reject the previous stage's output. Stage 3 sees one chunk at a
time with no alternatives, which removes the pressure to pick *something* that
stage 2 operates under.

## Sampling

150 documents, stratified by section and deliberately not proportional to the
corpus. Reference material is 51% of the corpus but a smaller share of what
people actually ask about.

| Section | Corpus share | Sample share | Documents |
|---|---|---|---|
| concepts | 18% | 30% | 45 |
| tasks | 22% | 30% | 45 |
| reference | 51% | 25% | 38 |
| tutorials | 3% | 10% | 15 |
| setup | 2% | 5% | 7 |

Allocation uses the largest-remainder method so the counts sum exactly.
Sampling is seeded (`random_seed=42`) and all collections are sorted before
sampling, so the set is reproducible.

Documents outside 1,000–12,000 characters are excluded: shorter ones cannot
support two distinct questions, longer ones cost more and dilute the model's
attention.

## Question types

Two questions per document, one type per document, cycled to give an exact
30/30/20/20 split.

| Type | Count | Intent |
|---|---|---|
| direct | 90 | Uses the document's own terminology |
| paraphrased | 90 | Describes the concept in different words |
| multi_hop | 60 | Answer requires combining separate facts |
| scenario | 60 | Describes a symptom or goal, not a mechanism |

## Lexical overlap

For each question, the fraction of its content tokens (stopwords and tokens of
three characters or fewer removed) that also appear in its gold chunks.

A value near 1.0 means the question reuses the passage vocabulary and lexical
retrieval will find it trivially. Near 0.0 means retrieval must match semantics.

| Type | Mean | Median | n |
|---|---|---|---|
| direct | 0.743 | 0.764 | 86 |
| multi_hop | 0.575 | 0.615 | 47 |
| scenario | 0.374 | 0.369 | 54 |
| paraphrased | 0.370 | 0.333 | 79 |

Overall: p10 0.222, median 0.519, p90 0.875.

The 2x gap between direct and paraphrased questions is the headroom that hybrid
retrieval and reranking are expected to close. A retrieval system that performs
equally on both is either very good or the measurement is broken.

Example of a zero-overlap question:

> *I need to list only the pods that are currently assigned to a specific
> worker node. How can I filter my query to show just those?*

The gold chunk describes `kubectl get pods --field-selector spec.nodeName=...`.
No content token is shared.

## Results

| Metric | Value |
|---|---|
| Questions generated | 300 |
| Answerable after labelling | 297 |
| Answerable after verification | 266 |
| Gold chunks after verification | 345 |
| Gold chunks rejected in stage 3 | 119 of 464 (26%) |

## Known limitations

**Multi-hop verification is too strict.** Stage 3 evaluates each chunk in
isolation without being told the question is multi-hop, so it applies a
single-passage standard to questions designed to need several. Multi-hop had the
highest rejection rate (38% of chunks, 13 of 60 questions dropped). Passing the
question type into the verification prompt would likely recover most of these.

| Type | Chunks rejected | Questions dropped |
|---|---|---|
| direct | 20/113 (18%) | 4 |
| paraphrased | 34/151 (23%) | 11 |
| multi_hop | 37/97 (38%) | 13 |
| scenario | 28/103 (27%) | 6 |

**Three questions are unanswerable from any single chunk.** Their answers
require aggregating a table that spans chunk boundaries. These are kept in the
set and documented rather than removed, since they represent a genuine
limitation of fixed-size chunking.

**Gold labels come from one model.** There is no second annotator and no human
adjudication. A hand-written subset would give a control against which to
measure the generated set.

**Questions are English-only.** Cross-lingual evaluation uses the 598 parallel
English–Chinese document pairs, described separately.

## Reproducing

```bash
make ingest
make generate-questions
make label-questions
make verify-questions
```

Requires `ANTHROPIC_API_KEY`. Responses are cached on disk by content hash, so
re-running costs nothing for unchanged inputs. Full cost is roughly USD 3.
