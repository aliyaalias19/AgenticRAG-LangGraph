.PHONY: eval-crosslingual eval-mcq-closedbook install lint format typecheck test check ingest generate-questions \
        label-questions verify-questions build-sft embed load-index \
        eval-retrieval eval-quality eval-security eval-mcq bench serve \
        demo up down logs clean

install:
	uv sync --dev
	uv pip install -e .

lint:
	uv run ruff check --fix .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/

test:
	uv run pytest

check: lint format typecheck test

up:
	docker compose -f deploy/docker-compose.yml up -d

down:
	docker compose -f deploy/docker-compose.yml down

logs:
	docker compose -f deploy/docker-compose.yml logs -f

ingest:
	uv run agentic-rag ingest

generate-questions:
	uv run agentic-rag generate-questions

label-questions:
	uv run agentic-rag label-questions

verify-questions:
	uv run agentic-rag verify-questions

embed:
	uv run python scripts/build_index.py --embed

load-index:
	uv run python scripts/build_index.py --load

build-sft:
	uv run python scripts/build_sft_dataset.py

eval-retrieval:
	uv run python scripts/run_retrieval_eval.py

eval-quality:
	uv run python scripts/run_quality_eval.py

eval-security:
	uv run python scripts/run_security_eval.py --mode unmitigated
	uv run python scripts/run_security_eval.py --mode mitigated --compare

bench:
	uv run python scripts/bench_vllm.py --concurrency 1 2 4 8 16

serve:
	uv run uvicorn agentic_rag.serving.app:create_app --factory --host 0.0.0.0 --port 8080

demo:
	uv run python scripts/demo_offline.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

eval-crosslingual:
	uv run python scripts/eval_crosslingual.py --no-title --query-chars 150

eval-mcq-closedbook:
	uv run python scripts/run_mcq_closedbook.py --label base --model base-llama
	uv run python scripts/run_mcq_closedbook.py --label tuned-awq --model k8s-assistant-awq

eval-crosslingual:
	uv run python scripts/eval_crosslingual.py --no-title --query-chars 150

eval-mcq-closedbook:
	uv run python scripts/run_mcq_closedbook.py --label base --model base-llama
	uv run python scripts/run_mcq_closedbook.py --label tuned-awq --model k8s-assistant-awq
