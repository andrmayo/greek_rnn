import os
import secrets
import warnings
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

import greek_rnn.greek_utils as utils
from greek_rnn.api_config import DEFAULT_MODEL_NAME, DEFAULT_MODEL_PATH
from greek_rnn.main import setup_logging
from greek_rnn.routes import router

# get environment variables
secret_key = os.environ.get("SESSION_SECRET_KEY")
if not secret_key:
	secret_key = secrets.token_hex(32)
	warnings.warn(
		"""
        SESSION_SECRET_KEY not set, using a random key.prefix.
        Sessions will not persist across restarts.
        """
	)


# use FastAPI lifespan context manager to load language model
@asynccontextmanager
async def model_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
	setup_logging()
	default_model = torch.load(
		DEFAULT_MODEL_PATH, map_location=utils.device, weights_only=False
	)
	app.state.model_cache = {DEFAULT_MODEL_NAME: default_model}
	app.state.default_model_name = DEFAULT_MODEL_NAME
	yield
	# cleanup


app = FastAPI(lifespan=model_lifespan)
app.add_middleware(SessionMiddleware, secret_key=secret_key)
app.include_router(router, prefix="/api")
