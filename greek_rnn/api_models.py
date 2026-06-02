from typing import Annotated

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
	# text with lacuna to fill marked as [..] with one . per missing character
	text: str


class PredictResponse(BaseModel):
	text: str
	lacuna_mask: list[bool]


class PredictKRequest(BaseModel):
	text: str
	k: Annotated[int, Field(gt=0)]


class PredictKResponse(BaseModel):
	texts: list[str]
	lacuna_mask: list[bool]


class RankRequest(BaseModel):
	text: str
	options: list[str]


class RankResponse(BaseModel):
	ranked: list[tuple[str, float]]
