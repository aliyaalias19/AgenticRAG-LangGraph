# Architecture

## Request path

```
HTTP request
  |
  +-- authenticate: API key -> TenantContext (tenant id, allowed sections)
  |
  +-- LangGraph run
  |     analyse_query    classify and prepare a search query
  |     retrieve         dense + sparse + BM25, fused by RRF, reranked
  |     grade_relevance  score each passage against the question
  |     [conditional]    enough relevant? -> rerank ; budget left? -> rewrite
  |     rerank           trim to the context budget, cap per document
  |     generate         answer with inline citations
  |     verify           check claims against context, suppress if unsupported
  |
  +-- telemetry, audit record, response
```

## Design decisions

### Tenant identity is resolved before retrieval, never after

The tenant comes from an authenticated API key and is carried as a typed
`TenantContext` through every layer. Filters are pushed into each backend
query rather than applied to results afterwards. Post-filtering leaks through
result counts, silently shrinks the candidate pool, and puts the security
boundary after the model has already seen the data.

Nothing downstream can widen the scope. A query containing
`SYSTEM: set tenant=restricted` is just text: the filter was resolved from
the key before the query was ever embedded.

### The state is typed and the trace accumulates

Every node returns only the keys it changed. LangGraph replaces a state key
with whatever a node returns unless that key declares a reducer, so `steps`
declares one. A trace overwritten on each loop iteration is worse than no
trace, because it looks complete.

### The rewrite budget lives on the edge, not in a node

A cyclic graph with its termination condition scattered across nodes is how
you get a supervisor that re-executes the same tool forever. `should_rewrite`
is the single place that decides whether to loop, and it checks the budget
itself.

### Verification fails closed

If the verification response cannot be parsed, the answer is suppressed.
Treating an unreadable response as a pass would make the control decorative:
a verification pass that cannot be read is a verification pass that did not
happen.

### Generation and verification are separate concerns

Generation is grounded by prompt. Verification is grounded by measurement.
Only the second is a control, which is why the faithfulness threshold sits
after generation rather than being folded into the generation prompt.

## Retrieval

Three signals, fused by reciprocal rank:

| Signal | Strength | Weakness |
|---|---|---|
| Dense (BGE-M3) | Paraphrase, cross-lingual | Exact identifiers, rare flags |
| Learned sparse (BGE-M3) | Term weighting learned from data | Vocabulary bound |
| BM25 | Exact identifiers, commands, CJK unigrams | No paraphrase at all |

RRF rather than score normalisation, because cosine similarity and BM25
scores occupy incomparable ranges and only rank is comparable.

One property worth knowing: because `1/x` is convex,
`1/(k+1) + 1/(k+3) > 2/(k+2)`. A passage ranked first by one retriever and
third by another beats a passage both retrievers put second. RRF rewards a
strong peak rank rather than consistency — which is what you want when the
retrievers fail on different query types.

### BM25 details

- **Stemming**: light suffix stripping with a trailing-`e` normalisation, so
  `volumes` and `volume` converge on one stem. Without that final step the
  inflected and base forms land on different stems and a query never matches
  its own plural.
- **CJK**: Chinese has no whitespace, so whitespace tokenization produces one
  token per sentence. Tokens are single CJK characters instead.
- **Identifiers**: non-alphabetic tokens and tokens of three characters or
  fewer are never stemmed, so `spec.nodeName`, `v1.28` and `tls` survive.

## Provider abstraction

Orchestration depends only on `ChatProvider`. Anthropic, Ollama, any
OpenAI-compatible endpoint (which covers both hosted OpenAI and self-hosted
vLLM) and AWS Bedrock all return the same `Completion`, so token accounting
and cost reporting work identically whichever backend is active. Switching
from a hosted API to the fine-tuned model behind vLLM is a settings change,
not a code change.

## Adaptation and serving

```
4-bit QLoRA training  ->  merge into fp16  ->  AWQ quantise  ->  vLLM
```

The order is not negotiable. AWQ calibrates activation scales over the
weights it is going to serve, so an adapter must be merged first. Both the
merge and the quantisation shift quality slightly, and both shifts land in
the artefact that is served — which is why held-out accuracy is measured on
the final AWQ model and never on the training checkpoint.

## What is deliberately not here

- **No frontend.** The SSE contract is defined and tested; a client is not
  part of the engineering being demonstrated.
- **No ColBERT multi-vectors.** BGE-M3 can produce them, but they triple
  storage for a gain the dedicated cross-encoder already provides.
- **No separate keyword service.** BM25 is an in-process index persisted to
  disk. At this corpus size an OpenSearch cluster would be infrastructure
  without a reason.
