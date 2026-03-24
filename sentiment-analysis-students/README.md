# Vietnamese Students Feedback Classifier API 🎓

API phân loại sentiment và topic của feedback sinh viên tiếng Việt sử dụng Machine Learning.

## 📊 Hiệu suất Model

- **Sentiment Classification**: 89.12% F1-Score (Logistic Regression)
- **Topic Classification**: 86.32% F1-Score (Linear SVM)

### Labels

**Sentiment:**
- `negative`: Tiêu cực
- `neutral`: Trung tính
- `positive`: Tích cực

**Topic:**
- `lecturer`: Giảng viên
- `training_program`: Chương trình đào tạo
- `facility`: Cơ sở vật chất
- `others`: Khác

---

## 🚀 Quick Start

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy API Server

```bash
python main.py
```

API sẽ chạy tại: `http://localhost:8000`

### 3. Truy cập Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "API is operational",
  "models_loaded": true
}
```

### 2. Predict (Sentiment + Topic)
```bash
POST /predict
Content-Type: application/json

{
  "text": "Giảng viên dạy rất hay và dễ hiểu"
}
```

**Response:**
```json
{
  "original_text": "Giảng viên dạy rất hay và dễ hiểu",
  "processed_text": "giảng viên dạy rất hay và dễ hiểu",
  "sentiment": {
    "sentiment": "positive",
    "confidence": 0.95
  },
  "topic": {
    "topic": "lecturer",
    "confidence": 0.88
  }
}
```

### 3. Predict Sentiment Only
```bash
POST /predict/sentiment
Content-Type: application/json

{
  "text": "Giảng viên dạy hay"
}
```

### 4. Predict Topic Only
```bash
POST /predict/topic
Content-Type: application/json

{
  "text": "Phòng học thiếu thiết bị"
}
```

### 5. Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "texts": [
    "Giảng viên dạy hay",
    "Trường cần đầu tư thêm",
    "Chương trình học tốt"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "original_text": "Giảng viên dạy hay",
      "processed_text": "giảng viên dạy hay",
      "sentiment": {...},
      "topic": {...}
    },
    ...
  ],
  "total": 3
}
```

### 6. Models Info
```bash
GET /models/info
```

---

## 🧪 Testing

Chạy test script:

```bash
python test_api.py
```

Test script sẽ kiểm tra:
- ✓ Health check
- ✓ Single prediction
- ✓ Sentiment only
- ✓ Topic only
- ✓ Batch prediction
- ✓ Models info
- ✓ Error handling

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t feedback-classifier .
```

### Run Container

```bash
docker run -p 8000:8000 feedback-classifier
```

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WORKERS=4
    restart: unless-stopped
```

```bash
docker-compose up -d
```

---

## 📦 Project Structure

```
vietnamese-feedback-classifier/
├── main.py                          # FastAPI application
├── requirements.txt                 # Dependencies
├── test_api.py                     # Test client
├── Dockerfile                      # Docker configuration
├── README.md                       # Documentation
├── tuned_models/                   # Trained models
│   ├── best_sentiment_model.pkl
│   ├── best_topic_model.pkl
│   ├── optimized_tfidf_sentiment.pkl
│   ├── optimized_tfidf_topic.pkl
│   └── tuning_info.pkl
└── vietnamese_students_feedback/   # Dataset (optional)
```

---

## 🔧 Configuration

### Environment Variables

Tạo file `.env`:

```env
# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Logging
LOG_LEVEL=info

# CORS (optional)
CORS_ORIGINS=["*"]
```

### Production Settings

Để chạy trong production:

```bash
# Với Gunicorn
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120

# Hoặc với Uvicorn
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --no-reload
```

---

## 📈 Performance Tips

1. **Caching**: Implement Redis cache cho frequent requests
2. **Load Balancing**: Sử dụng Nginx làm reverse proxy
3. **Async Processing**: Sử dụng Celery cho batch jobs lớn
4. **Monitoring**: Add Prometheus + Grafana

---

## 🛠️ Development

### Local Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
uvicorn main:app --reload

# Run tests
pytest tests/

# Code formatting
black main.py
flake8 main.py
```

---

## 📝 API Examples

### Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Giảng viên dạy hay"}
)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Giảng viên dạy hay"}'
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Giảng viên dạy hay'})
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 🔒 Security Notes

- **Production**: Disable `reload=True` trong uvicorn
- **CORS**: Chỉ định cụ thể allowed origins
- **Rate Limiting**: Implement rate limiting với slowapi
- **Authentication**: Thêm API key hoặc JWT nếu cần
- **HTTPS**: Sử dụng SSL certificate trong production

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [Uvicorn](https://www.uvicorn.org/)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License

---

## 👥 Authors

- Your Name

---

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Made with ❤️ using FastAPI and Scikit-learn**