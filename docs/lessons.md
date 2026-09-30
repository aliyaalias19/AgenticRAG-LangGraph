# 踩坑记录 / Lessons

Concrete things that cost time, with the resolution.

## Anthropic SDK 1.0 removed sampling parameters

Version 1.0.0 of the Python SDK (20 August 2026) made `temperature`, `top_p`
and `top_k` a `TypeError` on every Messages method, and the signature has no
`**kwargs`. The Messages API reference still documents `temperature` with a
default of 1.0.

The SDK signature is the authority, not the docs. Diversity in question
generation now comes from the prompt — asking for N varied questions of a
named type — which is reproducible in a way temperature never was.

## `structlog.testing.capture_logs` bypasses the processor chain

It swaps in its own minimal chain, so `merge_contextvars` never runs and
bound context variables do not appear in captured events. Context propagation
was working; the test was asserting against the wrong mechanism. Test that
the processor is configured and that contextvars bind, or capture stdout.

## Nested pydantic-settings classes do not inherit `env_file`

Only the class that declares `env_file` reads it. A nested settings group
constructed by `default_factory` sees the process environment and nothing
else, so `.env` values silently fail to load. Every nested group needs its
own `env_file`, and using `PROJECT_ROOT / ".env"` rather than a bare `.env`
makes it independent of the working directory.

## Spaces around `=` in `.env`

`ANTHROPIC_API_KEY = sk-ant-...` parses the value with a leading space and
the key with a trailing one. No error, just an empty setting.

## `doc_id` collided across language variants

The English and Chinese corpora both derive `doc_id` from the path relative
to their docs root, so `concepts/pods` existed twice. Chunk IDs inherited the
collision, the labeller was silently choosing between English and Chinese
passages, and the "parallel documents" count was detecting the collision
rather than confirming a mapping.

Fixed by prefixing `doc_id` with the source name and adding `relative_id` for
cross-lingual pairing. Found by cross-checking one gold label against its
source chunk — the kind of bug that never surfaces from a passing test suite.

## Light stemming has to normalise both directions

Stripping `es` from `volumes` gives `volum`; `volume` stays whole. The two
never match. A trailing-`e` strip after suffix removal makes inflected and
base forms converge. Tokens of three characters or fewer and anything
non-alphabetic are exempt, so `tls`, `v1.28` and `spec.nodeName` survive.

## RRF does not reward consensus the way intuition says

`1/(k+1) + 1/(k+3) > 2/(k+2)` because the reciprocal is convex. A passage
ranked first and third beats one ranked second twice. Two tests asserting the
opposite were wrong; the implementation was right.

## Grading short-circuits on an empty candidate list

No candidates means no grading call, which saves an API call and shifts every
scripted response in a test by one. Worth knowing when writing fixtures.

## FastAPI startup events need a context-managed TestClient

`TestClient(app)` does not run startup events; `with TestClient(app) as c:`
does. Building the service eagerly in `create_app` removes the problem
entirely — every network-touching component is behind a lazy property, so
construction opens no sockets.

## `X-Accel-Buffering: no` on SSE responses

Without it, an nginx in front of the service buffers the entire stream and
delivers it at once. The symptom is indistinguishable from a hung backend.

## AWQ cannot quantise an adapter

The order is QLoRA train → merge into fp16 → AWQ quantise → serve. Merging a
4-bit-trained adapter into fp16 shifts quality slightly and AWQ shifts it
again, so held-out accuracy must be measured on the served artefact. Quoting
the adapter's score for a model you serve quantised is the most common way
this benchmark gets overstated.

## Qdrant point ids must be unique across the collection, not the call

The first implementation used the enumerate position as the point id. One
upsert of the whole corpus worked fine. A *second* upsert call started again
at id 0 and silently overwrote the first call's points -- no error, no
warning, just missing data.

It survived the mocked tests because a mock records the call rather than
enforcing the engine's uniqueness constraint. It only surfaced when the
store was tested against `QdrantClient(":memory:")`, which runs the real
engine in-process.

Point ids are now `uuid5(namespace, chunk_id)`: stable across processes,
unique per chunk, and idempotent on re-ingestion.

The general lesson is about where to draw the mock boundary. Mocking the
*client* tests your call construction; it cannot test whether the engine
accepts what you built. Anything with a uniqueness, filtering or indexing
constraint needs the real engine, and for Qdrant the in-memory mode makes
that nearly free.

## CPU embedding of 15k chunks is not viable

Measured 1.60 s/chunk at `max_length=256` on 4 WSL threads — 6.7 hours, and
that truncates longer chunks. `max_length` 512 and 1024 were within 3% of
each other, so padding was not the bottleneck; the model is simply large for
CPU inference. Rent a T4 for ten minutes instead and cache the vectors.

## A library upgrade can silently move training to the CPU

Installing the training stack pulled torch 2.14 built against CUDA 13.0. The
pod's driver was 12.8, so `torch.cuda.is_available()` returned `False` — and
nothing crashed. Training would have run on the CPU at roughly a thirtieth of
the speed, and the only symptom would have been the clock.

Worse, the parent process reported `cuda: True` while the worker process
failed, because a CUDA context is only really created where the work happens.

Any `pip install` that touches torch is now followed by an explicit assertion
that CUDA is still available, in the process that will use it.

## Quantising and serving could not share one environment

AutoAWQ was deprecated during this project, and its last tested combination
was transformers 4.51.3. vLLM 0.11 needs transformers ≥ 4.56 for its tokenizer
backend. Those two requirements cannot be satisfied at once.

The workaround was pinning down for quantisation and back up for serving. The
real answer is that these are different stages and belong in different
containers, which is why production ML pipelines look the way they do.

## The AWQ kernel was worth 6.5× more than anything else tried

Serving the quantised model with the default AWQ GEMM kernel gave 14.8 tok/s
per stream on an A40. Switching to `--quantization awq_marlin` gave 95.6 —
same weights, same hardware, one flag. GEMM dequantises in a layout that
starves the tensor cores at batch size 1; Marlin repacks for the access
pattern the hardware wants.

Throughput problems are worth profiling before they are worth re-architecting.

## An open-book evaluation cannot measure fine-tuning

The MCQ evaluation supplied the source passages alongside the question. The
base instruct model scored 96.4% on it, which looked like a strong system and
was in fact a broken measurement: with the answer in the context window, the
test measures reading comprehension, not knowledge.

Fine-tuning changes weights. Removing the passages dropped the base model to
64.3% and revealed a +5.4 point effect that the open-book version had been
structurally incapable of detecting.

An evaluation that cannot distinguish the thing being changed is not a
conservative evaluation. It is a broken one.

## A benchmark that reports zeros instead of failing

The first vLLM benchmark run returned 0.0 tok/s at every concurrency level
after 32 consecutive HTTP 404s. It printed a clean, plausible-looking table of
zeroes and exited successfully.

The cause was a default: `--model` defaulted to the weights directory rather
than the served model name, which vLLM rejects. The deeper problem is that a
harness which reports zeroes on total failure produces numbers that can be
quoted by accident. It now resolves defaults from settings, and a failed run
should be loud.

## Invisible characters in a header produce an empty 400

The Anthropic API returned `400` with `content-length: 0` and no error body.
The key was 168 characters instead of 108 and printed as empty when sliced —
carriage returns from a Windows clipboard paste. HTTP headers cannot contain
them, so the request was rejected at the edge before it ever reached the API,
which is why there was no message to read.

Secrets read interactively are now stripped of whitespace, and the length and
non-secret prefix are echoed as a check.
