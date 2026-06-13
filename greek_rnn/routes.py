import json
import logging
import re
from typing import AsyncGenerator, cast

import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

import greek_rnn.greek_utils as utils
from greek_rnn.api_config import SERVED_MODELS_DIR
from greek_rnn.api_models import (
    PredictKRequest,
    PredictKResponse,
    PredictRequest,
    PredictResponse,
    RankRequest,
    RankResponse,
)
from greek_rnn.greek_char_generator import (
    predict_chars,
    predict_top_k,
    rank_reconstructions,
)
from greek_rnn.greek_model import RNN
from greek_rnn.greek_utils import DataItem
from greek_rnn.letter_tokenizer import LetterTokenizer

router = APIRouter()
logger = logging.getLogger(__name__)


def get_model(req: Request) -> RNN:
    model_name = req.session.get("model_name", req.app.state.default_model_name)
    return req.app.state.model_cache[model_name]


@router.post("/predict/")
async def predict(body: PredictRequest, req: Request) -> PredictResponse:
    """
    Generates top reconstruction of Greek text with lacunae in format [..]
    with one . per missing character.
    """
    # TODO: Clean this up by replacing assert checks with equivalent unit tests
    model = get_model(req)
    text = body.text
    text = re.sub("<gap/>", "!", text)
    pattern = re.compile(r"\[.*?\]")
    text = pattern.sub(lambda x: x.group().replace(" ", ""), text)
    tokenizer = LetterTokenizer(False, False)

    # use explit loop to keep track of whitespace so that
    # we can later display text as originally formatted
    text_buff = []
    space_idxs = []
    newline_idxs = []
    tab_idxs = []
    i = 0
    for char in text:
        if char in "[]":
            text_buff.append(char)
        elif char in " \n\t":
            if char == " ":
                space_idxs.append(i)
            elif char == "\n":
                newline_idxs.append(i)
            elif char == "\t":
                tab_idxs.append(i)
            i += 1
        else:
            tokens = list(tokenizer.process_token(char))
            text_buff += tokens
            i += len(tokens)

    text = "".join(text_buff)

    instance = DataItem(text=text)
    data_item = model.actual_lacuna_mask_and_label(instance)
    assert data_item.mask, (
        "RNN.actual_lacuna_mask_and_label not populating mask for data_item"
    )
    text = predict_chars(model, data_item)
    lacuna_mask = data_item.mask[1:-1]
    assert len(text) == len(lacuna_mask), (
        "text and lacuna_mask should both be of same length, "
        "with BOT and EOT tokens stripped"
    )

    # Add white space in, reversing index lists
    # so they can be used as stacks
    space_idxs.reverse()
    newline_idxs.reverse()
    tab_idxs.reverse()

    text_buff = []
    lacuna_buff = []
    i = 0

    def idx_peek_lists(idx: int) -> bool:
        if space_idxs and idx == space_idxs[-1]:
            return True
        if newline_idxs and idx == newline_idxs[-1]:
            return True
        if tab_idxs and idx == tab_idxs[-1]:
            return True
        return False

    for char, masked in zip(text, lacuna_mask):
        while idx_peek_lists(i):
            if space_idxs and i == space_idxs[-1]:
                text_buff.append(" ")
                space_idxs.pop()
                lacuna_buff.append(False)
            elif newline_idxs and i == newline_idxs[-1]:
                text_buff.append("\n")
                newline_idxs.pop()
                lacuna_buff.append(False)
            elif tab_idxs and i == tab_idxs[-1]:
                text_buff.append("\t")
                tab_idxs.pop()
                lacuna_buff.append(False)
            i += 1
        text_buff.append(char)
        lacuna_buff.append(masked)
        i += 1

    text = "".join(text_buff)
    lacuna_mask = lacuna_buff
    del text_buff, lacuna_buff
    logger.info(text)

    return PredictResponse(text=text, lacuna_mask=lacuna_mask)


@router.post("/rank/")
async def rank(body: RankRequest, req: Request) -> RankResponse:
    """
    Ranks provided reconstructions for a lacuna.
    Expects text field with Greek text with lacunae marked as [..],
    and options field as a list of reconstructions to rank
    (without spaces or diacritics).
    """
    text, options = body.text, body.options

    model = get_model(req)
    return RankResponse(ranked=rank_reconstructions(model, text, options))


@router.post("/predict-k/")
async def predict_k(body: PredictKRequest, req: Request) -> PredictKResponse:
    """
    Returns top k reconstruction of Greek text with lacunae in format [..]
    with one . per missing character.
    """
    text, k = body.text, body.k
    text = re.sub("<gap/>", "!", text)
    pattern = re.compile(r"\[.*?\]")
    text = pattern.sub(lambda x: x.group().replace(" ", ""), text)

    model = get_model(req)

    instance = DataItem(text=text)
    data_item = model.actual_lacuna_mask_and_label(instance)
    texts = predict_top_k(model, data_item, k)
    return PredictKResponse(texts=texts, lacuna_mask=data_item.mask[1:-1])


@router.post("/predict-file/")
async def predict_file(
    req: Request,
    file: UploadFile = File(
        ...,
        description=(
            "Either JSON file with format {'text': ['αβγ[...]', "
            "'αβγ[...]']}, or JSONL file with format "
            "{'text': 'αβγ[...]'} for each text"
        ),
    ),
) -> StreamingResponse:
    if not (file.filename and file.filename.endswith((".json", ".jsonl"))):
        raise HTTPException(status_code=422, detail="File must be .json or .jsonl")

    contents = await file.read()
    texts = (
        [
            json.loads(line.strip())["text"]
            for line in contents.decode("utf-8").splitlines()
            if line.strip()
        ]
        if cast(str, file.filename).endswith("jsonl")
        else json.loads(contents.strip())["text"]
    )
    model = get_model(req)

    async def generate() -> AsyncGenerator[str, None]:
        for text in texts:
            instance = DataItem(text=text)
            data_item = model.actual_lacuna_mask_and_label(instance)
            result = predict_chars(model, data_item)
            yield json.dumps({"text": text, "reconstruction": result}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/default-model/")
async def get_default_model(req: Request):
    return {"model": req.app.state.default_model_name}


@router.get("/get-current-model/")
async def get_current_model(req: Request):
    return {"model": req.session.get("model_name", req.app.state.default_model_name)}


@router.get("/models/")
async def get_models() -> list[str]:
    return [
        p.name
        for p in SERVED_MODELS_DIR.iterdir()
        if p.is_dir() and list(p.glob("*.pth"))
    ]


@router.patch("/change-model/{model_name}")
async def change_model(model_name: str, req: Request):
    model_dir = SERVED_MODELS_DIR / model_name
    if model_name not in req.app.state.model_cache:
        req.app.state.model_cache[model_name] = torch.load(
            max(model_dir.glob("*.pth"), key=lambda f: f.stat().st_mtime),
            map_location=utils.device,
            weights_only=False,
        )
    req.session["model_name"] = model_name
    return {"model": model_name}
