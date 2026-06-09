.PHONY: run-dev setup run-dev-backend run-dev-frontend fmt

run-dev: setup
	uv run uvicorn greek_rnn.api:app --reload --host 127.0.0.1 --port 8000 & \
	sleep 2 && npm run dev --prefix frontend -- --open & \
	wait

setup:
	uv sync --extra api
	npm install --prefix frontend

run-dev-backend:
	uvicorn greek_rnn.api:app --reload --host 127.0.0.1 --port 8000 

run-dev-frontend:
	npm run dev --prefix frontend -- --open

fmt:
	uv run ruff format greek_rnn
	uv run ruff check --fix greek_rnn

gen-docker-reqs:
	@ uv export --no-dev --all-extras --no-hashes | grep -v torch | grep -v nvidia | grep -v triton > requirements-cpu.txt
	@ uv export --no-dev --all-extras --no-hashes | grep "^torch==" | sed 's/torch==\([^ \\]*\)/torch==\1+cpu --index-url https:\/\/download.pytorch.org\/whl\/cpu/' >> requirements-cpu.txt
