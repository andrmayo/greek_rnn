import logging
import re
import torch
from fastapi import APIRouter, Request

import greek_rnn.greek_utils as utils
from greek_rnn.api_config import SERVED_MODELS_DIR
from greek_rnn.api_models import (
    PredictRequest,
    PredictResponse,
    PredictKRequest,
    PredictKResponse,
    RankRequest,
    RankResponse,
)
from greek_rnn.greek_char_generator import (
    predict_chars,
    predict_top_k,
    rank_reconstructions,
)
from greek_rnn.greek_utils import DataItem

router = APIRouter()
logger = logging.getLogger(__name__)


def get_model(req: Request):
    model_name = req.session.get("model_name", req.app.state.default_model_name)
    return req.app.state.model_cache[model_name]


@router.post("/predict/")
async def predict(body: PredictRequest, req: Request):
    """Generates top reconstruction of Greek sentence with lacunae in format [..] with one . per missing character."""
    model = get_model(req)
    sentence = body.sentence
    sentence = re.sub("<gap/>", "!", sentence)
    pattern = re.compile(r"\[.*?\]")
    sentence = pattern.sub(lambda x: x.group().replace(" ", ""), sentence)
    instance = DataItem(text=sentence)
    data_item = model.actual_lacuna_mask_and_label(instance)
    return PredictResponse(
        sentence=predict_chars(model, data_item), lacuna_mask=data_item.mask[1:-1]
    )


@router.post("/rank/")
async def rank(body: RankRequest, req: Request):
    """
    Ranks provided reconstructions for a lacuna. Expects sentence field with Greek sentence with lacunae marked as [..], and options field as a list of reconstructions to rank (without spaces or diacritics).
    """
    sentence, options = body.sentence, body.options

    model = get_model(req)
    return RankResponse(ranked=rank_reconstructions(model, sentence, options))


@router.post("/predict-k/")
async def predict_k(body: PredictKRequest, req: Request):
    """Returns top k reconstruction of Greek sentence with lacunae in format [..] with one . per missing character."""
    sentence, k = body.sentence, body.k
    sentence = re.sub("<gap/>", "!", sentence)
    pattern = re.compile(r"\[.*?\]")
    sentence = pattern.sub(lambda x: x.group().replace(" ", ""), sentence)

    model = get_model(req)

    instance = DataItem(text=sentence)
    data_item = model.actual_lacuna_mask_and_label(instance)
    sentences = predict_top_k(model, data_item, k)
    return PredictKResponse(sentences=sentences, lacuna_mask=data_item.mask[1:-1])


@router.patch("/change_model/{model_name}")
async def change_model(model_name: str, req: Request):
    model_dir = SERVED_MODELS_DIR / model_name
    if model_name not in req.app.state.model_cache:
        req.app.state.model_cache[model_name] = torch.load(
            max(model_dir.glob("*.pth"), key=lambda f: f.stat().st_mtime),
            location=utils.device,
            weights_only=False,
        )
    req.session["model_name"] = model_name
    return {"model": model_name}
