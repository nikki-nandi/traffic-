from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.predictor import predict
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    NS: int
    EW: int
    NS_wait: int
    EW_wait: int
    signal: int

@app.post("/predict/{model}")
def run(model: str, data: Input):
    state = [data.NS, data.EW, data.NS_wait, data.EW_wait, data.signal]
    return predict(model, state)