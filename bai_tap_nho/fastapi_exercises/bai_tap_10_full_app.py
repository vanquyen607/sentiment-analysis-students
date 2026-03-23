from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import re
from pathlib import Path

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s\u0080-\u024F\u1E00-\u1EFF.,!?]', ' ', text)
    text = ' '.join(text.split())
    return text

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class FeedbackInput(BaseModel):
    text: str = Field(..., min_length=1, description="Câu feedback")

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
    processed_text: str
    sentiment: SentimentResponse
    topic: TopicResponse

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int

class HealthResponse(BaseModel):
    status: str
    message: str
    models_loaded: bool

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Vietnamese Students Feedback Classifier API",
    description="Bài tập 10: API đầy đủ (mock models)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# PREDICTION FUNCTIONS (MOCK)
# ============================================================================
def predict_sentiment(text: str):
    processed = preprocess_text(text)
    if "hay" in processed or "tốt" in processed:
        return {"sentiment": "positive", "confidence": 0.8}
    elif "tệ" in processed or "xấu" in processed:
        return {"sentiment": "negative", "confidence": 0.7}
    else:
        return {"sentiment": "neutral", "confidence": 0.6}

def predict_topic(text: str):
    processed = preprocess_text(text)
    if "giảng viên" in processed or "dạy" in processed:
        return {"topic": "lecturer", "confidence": 0.9}
    elif "chương trình" in processed or "đào tạo" in processed:
        return {"topic": "training_program", "confidence": 0.8}
    elif "cơ sở" in processed or "vật chất" in processed:
        return {"topic": "facility", "confidence": 0.7}
    else:
        return {"topic": "others", "confidence": 0.5}

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/api", response_model=HealthResponse)
async def root():
    return {"status": "healthy", "message": "API is running!", "models_loaded": True}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "message": "API operational", "models_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_feedback(feedback: FeedbackInput):
    text = feedback.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    processed = preprocess_text(text)
    sentiment = predict_sentiment(text)
    topic = predict_topic(text)
    return {
        "original_text": text,
        "processed_text": processed,
        "sentiment": sentiment,
        "topic": topic
    }

@app.post("/predict/sentiment", response_model=SentimentResponse)
async def predict_sentiment_only(feedback: FeedbackInput):
    text = feedback.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    return predict_sentiment(text)

@app.post("/predict/topic", response_model=TopicResponse)
async def predict_topic_only(feedback: FeedbackInput):
    text = feedback.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    return predict_topic(text)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchFeedbackInput):
    results = []
    for text in batch.texts:
        text = text.strip()
        if not text:
            continue
        processed = preprocess_text(text)
        sentiment = predict_sentiment(text)
        topic = predict_topic(text)
        results.append({
            "original_text": text,
            "processed_text": processed,
            "sentiment": sentiment,
            "topic": topic
        })
    return {"results": results, "total": len(results)}

@app.get("/models/info")
async def get_models_info():
    return {
        "sentiment_model": "MockModel",
        "topic_model": "MockModel",
        "sentiment_labels": ["negative", "neutral", "positive"],
        "topic_labels": ["lecturer", "training_program", "facility", "others"],
        "vectorizer": "TF-IDF (mock)",
        "version": "1.0.0"
    }