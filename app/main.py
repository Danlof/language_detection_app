from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_pipeline
from app.model import __version__ as model_version

app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"language": language}
