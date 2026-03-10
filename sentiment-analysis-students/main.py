# main.py - FastAPI Application
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import pickle
import re
from pathlib import Path
import uvicorn

# ============================================================================
# LOAD MODELS VÀ VECTORIZERS
# ============================================================================
print("🚀 Loading models...")

models_dir = Path("tuned_models")

# Load sentiment model và vectorizer
with open(models_dir / 'best_sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)
    
with open(models_dir / 'optimized_tfidf_sentiment.pkl', 'rb') as f:
    sentiment_vectorizer = pickle.load(f)

# Load topic model và vectorizer
with open(models_dir / 'best_topic_model.pkl', 'rb') as f:
    topic_model = pickle.load(f)
    
with open(models_dir / 'optimized_tfidf_topic.pkl', 'rb') as f:
    topic_vectorizer = pickle.load(f)

print("✓ Models loaded successfully!")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
def preprocess_text(text: str) -> str:
    """Tiền xử lý text giống như khi training"""
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Loại bỏ URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Loại bỏ email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Loại bỏ ký tự đặc biệt nhưng giữ dấu câu
    text = re.sub(r'[^\w\s\u0080-\u024F\u1E00-\u1EFF.,!?]', ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class FeedbackInput(BaseModel):
    text: str = Field(..., min_length=1, description="Câu feedback cần phân loại")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Giảng viên dạy rất hay và dễ hiểu"
            }
        }

class BatchFeedbackInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Danh sách các câu feedback")
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Giảng viên dạy rất hay và dễ hiểu",
                    "Cơ sở vật chất trường còn thiếu thốn"
                ]
            }
        }

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Nhãn sentiment: negative, neutral, positive")
    confidence: Optional[float] = Field(None, description="Độ tin cậy (0-1)")

class TopicResponse(BaseModel):
    topic: str = Field(..., description="Nhãn topic: lecturer, training_program, facility, others")
    confidence: Optional[float] = Field(None, description="Độ tin cậy (0-1)")

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
    description="API phân loại sentiment và topic của feedback sinh viên tiếng Việt",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục static nếu chưa có
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def predict_sentiment(text: str):
    """Dự đoán sentiment"""
    processed_text = preprocess_text(text)
    text_vec = sentiment_vectorizer.transform([processed_text])
    
    prediction = sentiment_model.predict(text_vec)[0]
    
    # Lấy confidence nếu model hỗ trợ
    confidence = None
    if hasattr(sentiment_model, 'predict_proba'):
        proba = sentiment_model.predict_proba(text_vec)[0]
        confidence = float(proba[prediction])
    
    sentiment_labels = ['negative', 'neutral', 'positive']
    
    return {
        'sentiment': sentiment_labels[prediction],
        'confidence': confidence
    }

def predict_topic(text: str):
    """Dự đoán topic"""
    processed_text = preprocess_text(text)
    text_vec = topic_vectorizer.transform([processed_text])
    
    prediction = topic_model.predict(text_vec)[0]
    
    # Lấy confidence nếu model hỗ trợ
    confidence = None
    if hasattr(topic_model, 'predict_proba'):
        proba = topic_model.predict_proba(text_vec)[0]
        confidence = float(proba[prediction])
    elif hasattr(topic_model, 'decision_function'):
        # Cho SVM
        decision = topic_model.decision_function(text_vec)[0]
        # Normalize thành pseudo-probability
        confidence = None  # SVM không có true probability
    
    topic_labels = ['lecturer', 'training_program', 'facility', 'others']
    
    return {
        'topic': topic_labels[prediction],
        'confidence': confidence
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def read_root():
    """Serve frontend HTML"""
    return FileResponse('static/index.html')

@app.get("/api", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Vietnamese Students Feedback Classifier API is running!",
        "models_loaded": True
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is operational",
        "models_loaded": sentiment_model is not None and topic_model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_feedback(feedback: FeedbackInput):
    """
    Dự đoán sentiment và topic cho một câu feedback
    
    - **text**: Câu feedback cần phân loại
    """
    try:
        text = feedback.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text không được rỗng")
        
        # Tiền xử lý
        processed_text = preprocess_text(text)
        
        # Dự đoán
        sentiment_result = predict_sentiment(text)
        topic_result = predict_topic(text)
        
        return {
            "original_text": text,
            "processed_text": processed_text,
            "sentiment": sentiment_result,
            "topic": topic_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")

@app.post("/predict/sentiment", response_model=SentimentResponse)
async def predict_sentiment_only(feedback: FeedbackInput):
    """
    Chỉ dự đoán sentiment
    
    - **text**: Câu feedback cần phân loại
    """
    try:
        text = feedback.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text không được rỗng")
        
        result = predict_sentiment(text)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")

@app.post("/predict/topic", response_model=TopicResponse)
async def predict_topic_only(feedback: FeedbackInput):
    """
    Chỉ dự đoán topic
    
    - **text**: Câu feedback cần phân loại
    """
    try:
        text = feedback.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text không được rỗng")
        
        result = predict_topic(text)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchFeedbackInput):
    """
    Dự đoán cho nhiều câu feedback cùng lúc
    
    - **texts**: Danh sách các câu feedback
    """
    try:
        results = []
        
        for text in batch.texts:
            text = text.strip()
            
            if not text:
                continue
            
            processed_text = preprocess_text(text)
            sentiment_result = predict_sentiment(text)
            topic_result = predict_topic(text)
            
            results.append({
                "original_text": text,
                "processed_text": processed_text,
                "sentiment": sentiment_result,
                "topic": topic_result
            })
        
        return {
            "results": results,
            "total": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán batch: {str(e)}")

@app.get("/models/info")
async def get_models_info():
    """Thông tin về các models đang được sử dụng"""
    return {
        "sentiment_model": str(type(sentiment_model).__name__),
        "topic_model": str(type(topic_model).__name__),
        "sentiment_labels": ["negative", "neutral", "positive"],
        "topic_labels": ["lecturer", "training_program", "facility", "others"],
        "vectorizer": "TF-IDF",
        "version": "1.0.0"
    }

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import webbrowser
    import threading
    import time
    
    print("\n" + "=" * 80)
    print("🚀 Starting FastAPI Server")
    print("=" * 80)
    print("\n📍 Server URLs:")
    print("   - Frontend: http://localhost:8000")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("\n🔗 Endpoints:")
    print("   - GET  / - Web UI Frontend")
    print("   - GET  /health - Health check")
    print("   - POST /predict - Dự đoán sentiment và topic")
    print("   - POST /predict/sentiment - Chỉ dự đoán sentiment")
    print("   - POST /predict/topic - Chỉ dự đoán topic")
    print("   - POST /predict/batch - Dự đoán batch")
    print("   - GET  /models/info - Thông tin models")
    print("\n⌨️  Nhấn Ctrl+C để dừng server")
    print("=" * 80 + "\n")
    
    # Auto-open browser sau 3 giây
    def open_browser():
        time.sleep(3)
        print("🌐 Đang mở trình duyệt tại http://localhost:8000 ...\n")
        webbrowser.open('http://localhost:8000')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start server
    uvicorn.run(
        "main:app",  # Dùng import string thay vì app object
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload khi code thay đổi
    )