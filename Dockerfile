FROM debian:13-slim
WORKDIR /usr/src/app

COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /bin/

COPY ./pyproject.toml ./requirements-cpu.txt ./.python-version ./entrypoint.sh .

RUN uv venv
RUN uv pip install -r requirements-cpu.txt

COPY ./served_models/ ./served_models/
COPY ./greek_rnn/ ./greek_rnn/
COPY ./README.md .

RUN uv pip install .

RUN chmod +x entrypoint.sh
ENTRYPOINT ./entrypoint.sh
