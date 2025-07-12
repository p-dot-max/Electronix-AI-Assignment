from pydantic import BaseModel
from typing import List

class PredictReq(BaseModel):
    text: str

class PredictRes(BaseModel):
    label: str
    score: float

class BatchPredictReq(BaseModel):
    texts: List[str]

class BatchPredictRes(BaseModel):
    predictions: List[PredictRes]

