FROM debian:13-slim
WORKDIR /usr/src/app

COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /bin/

COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./served_models/ ./served_models/
COPY ./greek_rnn/ ./greek_rnn/
COPY ./README.md .

RUN uv sync --no-dev --all-extras

RUN chmod +x entrypoint.sh
ENTRYPOINT ./entrypoint.sh
