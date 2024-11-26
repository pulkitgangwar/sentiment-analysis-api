from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline




app = FastAPI(debug=True, title="Sentiment Analysis API")

class Payload(BaseModel):
    text: str


async def classify_text(text):
    pipe = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")     
    return pipe(text)


@app.post("/")
async def get_sentiment(payload: Payload):
    result = await classify_text(payload.text)
    return JSONResponse({"data": result})
