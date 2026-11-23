.PHONY: install lint typecheck test integration eval gate check

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest --cov=rag_eval --cov-report=term-missing

integration:
	docker compose up -d --wait
	uv run pytest -m integration

eval:
	uv run rag-eval run

gate: eval
	uv run rag-eval compare evals/baseline.json results/current.json --fail-on-regression 0.02

check: lint typecheck test gate
