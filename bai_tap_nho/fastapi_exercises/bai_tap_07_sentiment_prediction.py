from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Sentiment Prediction", description="Bài tập 07: Dự đoán sentiment (mock)")

class FeedbackInput(BaseModel):
    text: str = Field(..., min_length=1, description="Câu feedback")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Nhãn sentiment")
    confidence: Optional[float] = Field(None, description="Độ tin cậy")

def predict_sentiment(text: str):
    # Mock prediction
    if "hay" in text.lower() or "tốt" in text.lower():
        return {"sentiment": "positive", "confidence": 0.8}
    elif "tệ" in text.lower() or "xấu" in text.lower():
        return {"sentiment": "negative", "confidence": 0.7}
    else:
        return {"sentiment": "neutral", "confidence": 0.6}

@app.post("/predict/sentiment", response_model=SentimentResponse)
async def predict_sentiment_only(feedback: FeedbackInput):
    text = feedback.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    result = predict_sentiment(text)
    return result