.PHONY: run-dev setup run-dev-backend run-dev-frontend fmt

run-dev: setup
	uvicorn greek_rnn.api:app --reload --host 127.0.0.1 --port 8000 & \
	npm run dev --prefix frontend & \
	wait

setup:
	uv sync --extra api

run-dev-backend:
	uvicorn greek_rnn.api:app --reload --host 127.0.0.1 --port 8000 

run-dev-frontend:
	npm run dev --prefix frontend

fmt:
	uv run ruff format greek_rnn
	uv run ruff check --fix greek_rnn
