.PHONY: install lint format typecheck test check ingest generate-questions label-questions verify-questions up down logs clean

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

ingest:
	uv run agentic-rag ingest

generate-questions:
	uv run agentic-rag generate-questions

label-questions:
	uv run agentic-rag label-questions

verify-questions:
	uv run agentic-rag verify-questions

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

up:
	docker compose -f deploy/docker-compose.yml up -d

down:
	docker compose -f deploy/docker-compose.yml down

logs:
	docker compose -f deploy/docker-compose.yml logs -f
