from typing import Annotated
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # sentence with lacuna to fill marked as [..] with one . per missing character
    sentence: str


class PredictResponse(BaseModel):
    sentence: str
    lacuna_mask: list[bool]


class PredictKRequest(BaseModel):
    sentence: str
    k: Annotated[int, Field(gt=0)]


class PredictKResponse(BaseModel):
    sentences: list[str]
    lacuna_masks: list[list[bool]]


class RankRequest(BaseModel):
    sentence: str
    options: list[str]


class RankResponse(BaseModel):
    ranked: list[tuple[str, float]]
