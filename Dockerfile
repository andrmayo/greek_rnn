FROM debian:13-slim
WORKDIR /usr/src/app

COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /bin/

COPY ./.python-version .
RUN uv venv
COPY ./requirements-cpu-torch.txt .
RUN uv pip install -r requirements-cpu-torch.txt

COPY ./pyproject.toml ./requirements-cpu-no-torch.txt ./entrypoint.sh .

RUN uv pip install -r requirements-cpu-no-torch.txt
COPY ./served_models/ ./served_models/
COPY ./greek_rnn/ ./greek_rnn/
COPY ./README.md .

RUN uv pip install .

RUN chmod +x entrypoint.sh
ENTRYPOINT ./entrypoint.sh
