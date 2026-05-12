.PHONY: run-dev setup run-dev-backend fmt

run-dev: setup run-dev-backend

setup:
	uv sync --extra api

run-dev-backend:
	uvicorn greek_rnn.api:app --reload --host 127.0.0.1 --port 8000

fmt:
	uv run ruff format greek_rnn
	uv run ruff check --fix greek_rnn
