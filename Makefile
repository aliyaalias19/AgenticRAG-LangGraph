.PHONY: install lint format typecheck test check ingest generate-questions label-questions verify-questions clean

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
