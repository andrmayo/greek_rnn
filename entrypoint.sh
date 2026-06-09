#!/usr/bin/env bash
exec uv uvicorn greek_rnn.api:app --host 0.0.0.0 --port ${PORT:-8080}
