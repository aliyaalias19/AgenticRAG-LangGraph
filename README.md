# Agentic RAG

Reference platform for evaluating agentic retrieval, model adaptation and
self-hosted inference, built over bilingual Kubernetes documentation.

The point of this repository is not the chatbot. It is the measurement
apparatus around it: a frozen evaluation set with labelled gold passages, a
controlled experiment harness for retrieval policies, a mechanical security
evaluation, and a reproducible path from a base model to a quantised model
served behind vLLM.

## What is in it

| Layer | Implementation |
|---|---|
| Ingestion | Heading-aware chunking with code-fence handling, provenance manifest pinned to a commit SHA |
| Retrieval | Dense + learned-sparse (BGE-M3) + BM25, fused by Reciprocal Rank Fusion, cross-encoder reranked |
| Orchestration | LangGraph state machine, seven typed nodes, conditional edges, bounded self-correction loop |
| Security | Tenant scoping pushed into every backend query, evidence-gated generation, audit log |
| Evaluation | Recall/precision/MRR/nDCG, four RAGAS metrics with a ship gate, mechanical attack-success measurement |
| Adaptation | 4-bit NF4 QLoRA, adapter merge, AWQ quantisation |
| Serving | FastAPI with SSE streaming, Prometheus metrics labelled by tenant |

## Try it without any services

```bash
make install
python scripts/demo_offline.py
```

This runs the real graph, real BM25 retrieval, real RRF fusion, real tenant
filters and the real evidence gate against a scripted model. It prints four
scenarios: a direct answer, a query rewrite triggered by low relevance, a
suppressed unsupported answer, and a restricted document staying invisible to
a public tenant.

## Full pipeline

```bash
make ingest                              # build the corpus
python scripts/build_index.py --embed    # embed once (GPU)
make up                                  # start Qdrant
python scripts/build_index.py --load     # populate Qdrant and BM25

make eval-retrieval                      # Recall@10 across configurations
make eval-quality                        # the four RAGAS metrics + ship gate
make eval-security                       # attack success rate

python scripts/build_sft_dataset.py      # 500 instruction examples + MCQ set
python scripts/train_qlora.py --config configs/qlora_r16_a32.json
python scripts/merge_and_quantize.py --adapter data/models/qlora/r16_a32
python scripts/bench_vllm.py --concurrency 1 2 4 8 16
```

## The self-correction loop

```
analyse_query -> retrieve -> grade_relevance
                    ^              |
                    |              +-- enough relevant context --> rerank
                    |              |                                 |
                    +-- rewrite ---+ budget remaining             generate
                                   |                                 |
                                   +-- budget exhausted -> give_up  verify
```

A pipeline decides once. An agent decides again with what it learned. The
cycle between grading and rewriting is the difference, and the rewrite budget
lives on the conditional edge so the loop cannot run away regardless of how a
node behaves.

## Why three retrieval signals

Dense embeddings match meaning and miss identifiers. BM25 matches
identifiers, flags and command names exactly and misses paraphrases.
BGE-M3's learned sparse weights sit between the two. Fusing all three by
reciprocal rank rather than by score is deliberate: cosine similarities and
BM25 scores are on incomparable scales, and only their orderings carry
comparable information.

The evaluation set was built to make this measurable. Questions are split
into four types, and lexical overlap with the gold passage was measured for
each: direct questions share 74% of their content vocabulary with the answer,
paraphrased questions only 37%. That 2x gap is the headroom hybrid retrieval
and reranking have to close. See `docs/evaluation.md`.

## Results

Numbers are produced by the commands above and written to `data/results/`.
Each run records its corpus hash, retrieval configuration and model version
in MLflow, so every figure is traceable to the build that produced it.

| Metric | Command | Output |
|---|---|---|
| Recall@10 by configuration | `make eval-retrieval` | `data/results/retrieval_eval.json` |
| Four RAGAS metrics + ship gate | `make eval-quality` | `data/results/quality_*.json` |
| Attack success rate | `make eval-security` | `data/results/security_*.json` |
| Task accuracy, base vs tuned | `scripts/run_mcq_eval.py` | `data/results/mcq_comparison.json` |
| TTFT / ITL / throughput | `scripts/bench_vllm.py` | `data/results/vllm_benchmark.json` |

## Development

```bash
make check     # lint, format, typecheck, test
make test      # tests only
```

Documentation: [architecture](docs/architecture.md) ·
[evaluation](docs/evaluation.md) · [security](docs/security.md) ·
[lessons](docs/lessons.md)
