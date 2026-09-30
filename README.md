# Agentic RAG — an evaluation platform

A reference platform for measuring agentic retrieval, model adaptation and
self-hosted inference, built over bilingual (English/Chinese) Kubernetes
documentation.

The assistant is not the point. The measurement apparatus around it is: a
frozen evaluation set with labelled gold passages, a controlled experiment
harness for retrieval policies, a mechanical security evaluation, and a
reproducible path from a base model to a quantised model served behind vLLM.

Every number below was measured on this repository and written to
`data/results/`. Nothing is estimated.

## Measured results

| What | Result | Command |
|---|---|---|
| Hybrid retrieval vs dense | **Recall@10 0.573 → 0.722** (+0.150) | `make eval-retrieval` |
| Reranking on paraphrased queries | **0.287 → 0.513** (+79% relative) | same |
| Prompt-injection defences | **attack success 42.9% → 0%** (14 cases) | `make eval-security` |
| AWQ quantisation | **15GB → 5.4GB (2.8×), zero accuracy loss** | `merge_and_quantize.py` |
| Marlin vs GEMM AWQ kernel | **14.8 → 95.6 tok/s per stream (6.5×)** | `make bench` |
| Serving throughput | **701.7 tok/s aggregate at 16-way concurrency** | same |
| Cross-lingual retrieval (dense) | **R@10 0.997 zh→en, 0.962 en→zh** | `make eval-crosslingual` |
| QLoRA fine-tuning (closed-book) | **64.3% → 69.6%, not significant (p=0.29)** | `make eval-mcq-closedbook` |

Hardware: NVIDIA A40 48GB. Base model: Llama-3.1-8B-Instruct.
Embeddings and reranking: BGE-M3 and BGE-reranker-v2-m3.

## Four findings worth reading

### BM25 beats dense retrieval on technical documentation

| configuration | Recall@10 | MRR |
|---|---|---|
| dense only | 0.573 | 0.421 |
| learned sparse only | 0.583 | 0.436 |
| **BM25 only** | **0.627** | **0.502** |
| hybrid (RRF) | 0.670 | 0.510 |
| hybrid + cross-encoder rerank | **0.722** | **0.561** |
| hybrid + rerank, deeper pool | 0.716 | 0.561 |

Lexical matching outperforms embeddings alone on this corpus. Technical
documentation is dense with identifiers — `kubectl`, `PodDisruptionBudget`,
`--ignore-daemonsets` — which embeddings blur and exact matching nails. Fusing
by reciprocal rank rather than by score is deliberate: cosine similarities and
BM25 scores live on incomparable scales, and only their orderings carry
comparable information.

### Reranking pays off exactly where the evaluation set predicted

Recall@10 by question type:

| configuration | direct | multi-hop | paraphrased | scenario |
|---|---|---|---|---|
| dense only | 0.849 | 0.794 | 0.287 | 0.358 |
| hybrid + rerank | 0.959 | 0.858 | **0.513** | 0.534 |

Before any retrieval was built, lexical overlap between questions and their
gold passages was measured: 0.743 for direct questions, 0.370 for paraphrased
ones. That 2× gap was the predicted headroom. Direct questions were already
near ceiling and moved 11 points; paraphrased questions moved 23. The
prediction held.

### Cross-lingual retrieval, and an asymmetry in lexical matching

598 documents exist in both languages under the same `relative_id` — hand
aligned parallel data requiring no labelling. A passage in one language is a
query whose correct answer is its counterpart in the other.

| direction | method | R@1 | R@10 | MRR |
|---|---|---|---|---|
| zh → en | dense | 0.942 | **0.997** | 0.965 |
| zh → en | BM25 | 0.605 | 0.863 | 0.694 |
| en → zh | dense | 0.809 | **0.962** | 0.868 |
| en → zh | BM25 | 0.298 | 0.604 | 0.392 |

Dense retrieval crosses languages almost losslessly. BM25 degrades but does
not collapse — because technical documentation is partially parallel at the
token level: translators translate the prose and leave `kubectl` alone.

The asymmetry is the interesting part. BM25 scores MRR 0.694 zh→en but 0.392
en→zh. Chinese documents carry English identifiers; English documents carry no
Chinese. The shared-token signal flows in one direction only.

### The AWQ kernel mattered more than the model

| concurrency | GEMM tok/s/stream | Marlin tok/s/stream | GEMM aggregate | Marlin aggregate |
|---|---|---|---|---|
| 1 | 14.8 | **95.6** | 14.5 | 85.9 |
| 4 | 14.5 | **91.5** | 53.3 | 308.2 |
| 16 | 14.2 | **77.7** | 156.2 | **701.7** |

Same weights, same GPU, one flag. The default AWQ GEMM kernel starves the
tensor cores at low batch size; Marlin repacks the weights into a layout that
keeps them fed. Per-stream decode stays nearly flat as concurrency rises 16×
while aggregate throughput scales 8×, which is continuous batching working as
intended. TTFT p99 rises from 94ms to 445ms — that is the price.

At 701.7 tok/s on a $0.49/hour instance, self-hosted inference costs
**$0.194 per million output tokens**, assuming the GPU is saturated. An idle
GPU costs the same as a busy one, so that figure is a ceiling, not an average.

## What did not work

Reported because a result you only publish when it flatters you is not a
measurement.

- **A deeper reranker candidate pool did not help.** Recall@10 0.716 versus
  0.722 for the standard pool, at higher latency.
- **QLoRA fine-tuning produced no demonstrated gain.** Closed-book accuracy on
  the held-out MCQ set rose from 64.3% to 69.6%, but on 112 items that does
  not reach significance (McNemar's exact test, p = 0.29; 14 items improved,
  8 regressed). 208 training examples is too few, and the evaluation set is
  too small to resolve an effect this size.
- **The original MCQ evaluation could not have detected fine-tuning at all.**
  It supplied the source passages with the question, which measures reading
  comprehension — something the base instruct model already does at 96.4%.
  Fine-tuning changes what a model knows, not what it can read. Removing the
  passages dropped the base model to 64.3% and created the headroom the
  evaluation had been missing. Both variants are kept; the closed-book one is
  the one that can measure anything.

## Try it without any services

```bash
make install
python scripts/demo_offline.py
```

Runs the real graph, real BM25 retrieval, real RRF fusion, real tenant filters
and the real evidence gate against a scripted model. Four scenarios: a direct
answer, a query rewrite triggered by low relevance, a suppressed unsupported
answer, and a restricted document staying invisible to a public tenant.

## Architecture

| Layer | Implementation |
|---|---|
| Ingestion | Heading-aware chunking with code-fence handling, provenance manifest pinned to a commit SHA |
| Retrieval | Dense + learned-sparse (BGE-M3) + BM25, fused by Reciprocal Rank Fusion, cross-encoder reranked |
| Orchestration | LangGraph state machine, seven typed nodes, conditional edges, bounded self-correction loop |
| Security | Tenant scoping pushed into every backend query, evidence-gated generation, audit log |
| Evaluation | Recall/precision/MRR/nDCG, mechanical attack-success measurement, cross-lingual alignment, closed-book MCQ |
| Adaptation | 4-bit NF4 QLoRA, adapter merge, AWQ quantisation |
| Serving | FastAPI with SSE streaming, vLLM with AWQ-Marlin, Prometheus metrics labelled by tenant |

### The self-correction loop

analyse_query -> retrieve -> grade_relevance
^ |
| +-- enough relevant context --> rerank
| | |
+-- rewrite ---+ budget remaining generate
| |
+-- budget exhausted -> give_up verify


A pipeline decides once. An agent decides again with what it learned. The
cycle between grading and rewriting is the difference, and the rewrite budget
lives on the conditional edge so the loop cannot run away regardless of how a
node behaves.

## Corpus and evaluation set

- 1,303 documents, 15,081 chunks, English and Chinese
- 598 parallel document pairs, used as cross-lingual ground truth
- 266 verified questions across four types, 345 labelled gold chunks
- 112 held-out MCQ items, document-disjoint from the 208 SFT training examples

## Full pipeline

```bash
make ingest                              # build the corpus
python scripts/build_index.py --embed    # embed once (GPU, ~70s on an A40)
make up                                  # start Qdrant
python scripts/build_index.py --load     # populate Qdrant and BM25

make eval-retrieval                      # Recall@10 across configurations
make eval-crosslingual                   # cross-lingual alignment
make eval-security                       # attack success rate

python scripts/build_sft_dataset.py
python scripts/train_qlora.py --config configs/qlora_r16_a32.json
python scripts/merge_and_quantize.py --adapter data/models/qlora/r16_a32
make eval-mcq-closedbook                 # base vs tuned, closed-book
make bench                               # TTFT, ITL, throughput
```

## Limitations

- The attack set is 14 cases, written by the same person who wrote the
  defences. 0% means these defences hold against these attacks, not that the
  system is secure.
- The MCQ set is 112 items, which gives roughly ±9 percentage points at one
  standard error. Effects smaller than that cannot be resolved.
- One QLoRA configuration was trained, not a sweep. The seven configs in
  `configs/` are defined but unrun.
- All serving numbers are from a single A40 and do not generalise to other
  hardware.

## Development

```bash
make check     # ruff, mypy strict, 310 tests
make test
```

Documentation: [architecture](docs/architecture.md) ·
[evaluation](docs/evaluation.md) · [security](docs/security.md) ·
[lessons](docs/lessons.md)
