from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Topic Prediction", description="Bài tập 08: Dự đoán topic (mock)")

class FeedbackInput(BaseModel):
    text: str = Field(..., min_length=1, description="Câu feedback")

class TopicResponse(BaseModel):
    topic: str = Field(..., description="Nhãn topic")
    confidence: Optional[float] = Field(None, description="Độ tin cậy")

def predict_topic(text: str):
    # Mock prediction
    text_lower = text.lower()
    if "giảng viên" in text_lower or "dạy" in text_lower:
        return {"topic": "lecturer", "confidence": 0.9}
    elif "chương trình" in text_lower or "đào tạo" in text_lower:
        return {"topic": "training_program", "confidence": 0.8}
    elif "cơ sở" in text_lower or "vật chất" in text_lower:
        return {"topic": "facility", "confidence": 0.7}
    else:
        return {"topic": "others", "confidence": 0.5}

@app.post("/predict/topic", response_model=TopicResponse)
async def predict_topic_only(feedback: FeedbackInput):
    text = feedback.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    result = predict_topic(text)
    return result