from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from typing import List, Union
import uvicorn

from src.inference import predict_sentiment

app = FastAPI(title="Sentiment Analysis API")

class SentimentRequest(BaseModel):
    text: Union[str, List[str]]

@app.get("/")
def home():
    return {"health_check": "OK", "message": "Sentiment Analysis API Running"}

@app.post("/predict")
def predict(request: SentimentRequest):
    """
    Endpoint to predict sentiment.
    Accepts a single string or a list of strings.
    """
    try:
        results = predict_sentiment(request.text)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)