from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="Batch Prediction", description="Bài tập 09: Dự đoán batch (mock)")

class BatchFeedbackInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Danh sách feedback")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: Optional[float] = None

class TopicResponse(BaseModel):
    topic: str
    confidence: Optional[float] = None

class PredictionResponse(BaseModel):
    original_text: str
    sentiment: SentimentResponse
    topic: TopicResponse

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int

def predict_sentiment(text: str):
    if "hay" in text.lower() or "tốt" in text.lower():
        return {"sentiment": "positive", "confidence": 0.8}
    elif "tệ" in text.lower() or "xấu" in text.lower():
        return {"sentiment": "negative", "confidence": 0.7}
    else:
        return {"sentiment": "neutral", "confidence": 0.6}

def predict_topic(text: str):
    text_lower = text.lower()
    if "giảng viên" in text_lower or "dạy" in text_lower:
        return {"topic": "lecturer", "confidence": 0.9}
    elif "chương trình" in text_lower or "đào tạo" in text_lower:
        return {"topic": "training_program", "confidence": 0.8}
    elif "cơ sở" in text_lower or "vật chất" in text_lower:
        return {"topic": "facility", "confidence": 0.7}
    else:
        return {"topic": "others", "confidence": 0.5}

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchFeedbackInput):
    results = []
    for text in batch.texts:
        text = text.strip()
        if not text:
            continue
        sentiment = predict_sentiment(text)
        topic = predict_topic(text)
        results.append({
            "original_text": text,
            "sentiment": sentiment,
            "topic": topic
        })
    return {"results": results, "total": len(results)}