"""
Sentiment Analysis API với FastAPI
Chạy: uvicorn main:app --reload
Truy cập: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import joblib
import json
import re
import os
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== TỰ ĐỘNG TÌM ĐÚNG THỨƯ MỤC =====
# Lấy thư mục chứa file main.py (không phải thư mục hiện tại)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # Chuyển working directory về thư mục chứa script

logger.info(f"📁 Working directory: {os.getcwd()}")

# ===== GLOBAL VARIABLES =====
model = None
vectorizer = None
metadata = None
label_names = {0: 'Tiêu cực', 1: 'Trung lập', 2: 'Tích cực'}

# ===== LIFESPAN EVENTS (Thay thế on_event) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý lifecycle của app"""
    global model, vectorizer, metadata
    
    # Startup
    try:
        logger.info("Đang load model...")
        
        # Kiểm tra file tồn tại
        model_file = 'best_sentiment_model.pkl'
        vectorizer_file = 'tfidf_vectorizer.pkl'
        metadata_file = 'model_metadata.json'
        
        if not os.path.exists(model_file):
            logger.error(f"❌ Không tìm thấy {model_file}")
            logger.error(f"📁 Thư mục hiện tại: {os.getcwd()}")
            logger.error(f"📁 Các file trong thư mục: {os.listdir('.')}")
            raise FileNotFoundError(f"Không tìm thấy {model_file}. Hãy chạy BƯỚC 3 để train model trước!")
        
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"✓ Load model thành công: {metadata['model_name']}")
        logger.info(f"✓ Test Accuracy: {metadata['test_accuracy']*100:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi load model: {e}")
        logger.error("💡 Hãy đảm bảo bạn đã:")
        logger.error("   1. Chạy BƯỚC 3 để train model")
        logger.error("   2. File main.py nằm cùng thư mục với best_sentiment_model.pkl")
        raise e
    
    yield  # App đang chạy
    
    # Shutdown
    logger.info("Đang shutdown...")

# ===== KHỞI TẠO FASTAPI =====
app = FastAPI(
    title="Vietnamese Sentiment Analysis API",
    description="API phân tích cảm xúc văn bản tiếng Việt (Feedback sinh viên)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Sử dụng lifespan thay vì on_event
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PYDANTIC MODELS (Updated for V2) =====
class TextInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Giảng viên dạy rất hay và nhiệt tình"
            }
        }
    )
    
    text: str = Field(..., min_length=1, max_length=1000, 
                     description="Văn bản cần phân tích (1-1000 ký tự)")

class BatchTextInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Giảng viên dạy rất hay",
                    "Môn học quá khó",
                    "Bình thường thôi"
                ]
            }
        }
    )
    
    texts: List[str] = Field(..., min_length=1, max_length=100,
                            description="Danh sách văn bản (tối đa 100)")

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    label: int
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    timestamp: str

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_name: str
    test_accuracy: float
    f1_score: float
    precision: float
    recall: float
    num_features: int
    label_mapping: Dict[int, str]

# ===== HÀM TIỀN XỬ LÝ =====
def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s.,!?àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ===== HÀM DỰ ĐOÁN =====
def predict_single(text: str) -> dict:
    """Dự đoán cảm xúc cho 1 văn bản"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")
    
    # Tiền xử lý
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Văn bản rỗng sau khi tiền xử lý")
    
    # Vectorize
    X = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(X)[0]
    
    result = {
        'text': text,
        'sentiment': label_names[int(prediction)],
        'label': int(prediction),
        'timestamp': datetime.now().isoformat()
    }
    
    # Thêm probability
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        result['confidence'] = float(max(probabilities))
        result['probabilities'] = {
            label_names[i]: round(float(prob), 4)
            for i, prob in enumerate(probabilities)
        }
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X)[0]
        result['confidence'] = float(max(np.abs(scores)))
    
    return result

# ===== API ENDPOINTS =====

@app.get("/", tags=["Root"])
async def root():
    """Trang chủ API"""
    return {
        "message": "Vietnamese Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running" if model is not None else "model not loaded",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "health": "/health"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Kiểm tra trạng thái server"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Lấy thông tin về model"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata không khả dụng")
    
    return {
        "model_name": metadata['model_name'],
        "test_accuracy": metadata['test_accuracy'],
        "f1_score": metadata['f1_score'],
        "precision": metadata['precision'],
        "recall": metadata['recall'],
        "num_features": metadata['num_features'],
        "label_mapping": label_names
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(input_data: TextInput):
    """
    Dự đoán cảm xúc cho 1 văn bản
    
    **Parameters:**
    - **text**: Văn bản tiếng Việt cần phân tích
    
    **Returns:**
    - sentiment: Tên cảm xúc (Tiêu cực/Trung lập/Tích cực)
    - label: Nhãn số (0/1/2)
    - confidence: Độ tin cậy (0-1)
    - probabilities: Xác suất cho từng class
    """
    try:
        result = predict_single(input_data.text)
        return result
    except Exception as e:
        logger.error(f"Lỗi prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(input_data: BatchTextInput):
    """
    Dự đoán cảm xúc cho nhiều văn bản cùng lúc
    
    **Parameters:**
    - **texts**: Danh sách văn bản (tối đa 100 văn bản)
    
    **Returns:**
    - results: Danh sách kết quả dự đoán
    - total: Tổng số văn bản đã xử lý
    """
    try:
        results = []
        for text in input_data.texts:
            result = predict_single(text)
            results.append(result)
        
        return {
            "results": results,
            "total": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Lỗi batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples", tags=["Examples"])
async def get_examples():
    """Lấy các ví dụ mẫu"""
    examples = [
        {"text": "Giảng viên dạy rất hay và nhiệt tình", "expected": "Tích cực"},
        {"text": "Môn học quá khó và nhàm chán", "expected": "Tiêu cực"},
        {"text": "Lớp học bình thường, không có gì đặc biệt", "expected": "Trung lập"},
        {"text": "Thầy giáo rất tâm huyết", "expected": "Tích cực"},
        {"text": "Nội dung quá lý thuyết, thiếu thực hành", "expected": "Tiêu cực"}
    ]
    
    results = []
    for example in examples:
        pred = predict_single(example["text"])
        results.append({
            **example,
            "predicted": pred["sentiment"],
            "confidence": pred.get("confidence")
        })
    
    return {"examples": results}

# ===== MAIN (ĐỂ CHẠY TRỰC TIẾP) =====
if __name__ == "__main__":
    import uvicorn
    
    # Kiểm tra file model trước khi start
    if not os.path.exists('best_sentiment_model.pkl'):
        print("\n" + "="*70)
        print("❌ LỖI: Không tìm thấy file model!")
        print("="*70)
        print("\n📁 Thư mục hiện tại:", os.getcwd())
        print("📁 Các file có sẵn:")
        for f in os.listdir('.'):
            print(f"   - {f}")
        print("\n💡 HƯỚNG DẪN FIX:")
        print("   1. Đảm bảo bạn đã chạy BƯỚC 3 để train model")
        print("   2. Copy các file sau vào thư mục hiện tại:")
        print("      • best_sentiment_model.pkl")
        print("      • tfidf_vectorizer.pkl")
        print("      • model_metadata.json")
        print("   3. Hoặc cd vào thư mục chứa model rồi chạy lại")
        print("="*70 + "\n")
    else:
        print("\n✓ Tìm thấy file model!")
        print("🚀 Starting server...\n")
        uvicorn.run(app, host="0.0.0.0", port=8000)