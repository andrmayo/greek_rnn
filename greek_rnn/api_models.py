from pydantic import BaseModel


class PredictRequest(BaseModel):
    # sentence with lacuna to fill marked as [..] with one . per missing character
    sentence: str


class PredictResponse(BaseModel):
    sentence: str
    lacuna_mask: list[bool]


class RankRequest(BaseModel):
    sentence: str
    options: list[str]


class RankResponse(BaseModel):
    ranked: list[tuple[str, float]]
