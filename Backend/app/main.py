'''
Fast API Implementation
'''

from fastapi import FastAPI
from app.schema import PredictReq, PredicRes
from app.model import BinarySentimentModel
from app.utils import set_seed
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = BinarySentimentModel()
set_seed()


@app.post("/predict", response_model=PredictRes)
async def predict(payload: PredictReq):
    start_time = time.time()
    result = model.predict(payload.text)
    result["inference_time"] = time.time() - start_time
    return result

@app.post("/predict_batch", response_model=BatchPredictRes)
async def predict_batch(payload: BatchPredictReq):
    start_time = time.time()
    predictions = [model.predict(text) for text in payload.texts]
    return {"predictions": predictions, "batch_inference_time": time.time() - start_time}





