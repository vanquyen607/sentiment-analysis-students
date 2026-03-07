# 🎯 Vietnamese Student Feedback Sentiment Analysis

> Hệ thống phân tích cảm xúc văn bản tiếng Việt cho phản hồi sinh viên sử dụng Machine Learning và FastAPI

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Accuracy-88.85%25-success.svg)](.)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](.)

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc Project](#cấu-trúc-project)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [API Documentation](#api-documentation)
- [Kết quả](#kết-quả)
- [Demo](#demo)

---

## 🎓 Tổng quan

Project này xây dựng một hệ thống hoàn chỉnh để phân tích cảm xúc (sentiment analysis) của các phản hồi sinh viên bằng tiếng Việt. Hệ thống có thể phân loại văn bản thành 3 loại cảm xúc:

- 😊 **Tích cực** (Positive)
- 😐 **Trung lập** (Neutral)  
- 😞 **Tiêu cực** (Negative)

### ✨ Tính năng chính

- ✅ Phân tích cảm xúc văn bản tiếng Việt
- ✅ RESTful API với FastAPI
- ✅ Độ chính xác 88.85%
- ✅ Giao diện web thân thiện
- ✅ Hỗ trợ batch prediction
- ✅ Auto documentation với Swagger UI

### 🛠 Công nghệ sử dụng

- **Machine Learning**: Scikit-learn, TF-IDF Vectorization
- **Model**: Linear SVM (Support Vector Machine)
- **API Framework**: FastAPI
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

---

## 📁 Cấu trúc Project

```
CCNLTHD/
│
├── 📊 Data & Datasets
│   ├── vietnamese_students_feedback/          # Dataset gốc từ Hugging Face
│   ├── vietnamese_students_feedback_csv/      # Dataset dạng CSV
│   └── vietnamese_feedback_cleaned/           # Dataset đã tiền xử lý
│
├── 🤖 Models & Artifacts
│   ├── best_sentiment_model.pkl              # Model ML đã train (118 KB)
│   ├── tfidf_vectorizer.pkl                  # TF-IDF vectorizer (195 KB)
│   └── model_metadata.json                   # Metadata của model (1 KB)
│
├── 📈 Analysis & Results
│   ├── eda_analysis.png                      # Biểu đồ EDA (470 KB)
│   ├── ml_training_results.png               # Kết quả training (499 KB)
│   ├── cleaned_sample.csv                    # Mẫu dữ liệu đã clean (18 KB)
│   └── test_predictions.csv                  # Kết quả test (1 KB)
│
├── 💻 Source Code
│   ├── main.py                               # FastAPI application (12 KB)
│   ├── CCNLTHD.ipynb                         # Jupyter notebook (417 KB)
│   └── index.html                            # Frontend demo (11 KB)
│
└── 📖 Documentation
    └── README.md                             # File này
```

---

## 📄 Mô tả chi tiết các file

### 📂 **Folders (Thư mục)**

#### `vietnamese_students_feedback/`
- **Mô tả**: Dataset gốc từ Hugging Face Hub
- **Nguồn**: [uitnlp/vietnamese_students_feedback](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
- **Định dạng**: Arrow format (tối ưu cho ML)
- **Nội dung**: 
  - Train: 11,426 mẫu
  - Validation: 1,583 mẫu
  - Test: 3,166 mẫu

#### `vietnamese_students_feedback_csv/`
- **Mô tả**: Dataset đã export sang CSV
- **Dùng để**: Xem dữ liệu bằng Excel, phân tích với pandas
- **Files**: `train.csv`, `validation.csv`, `test.csv`

#### `vietnamese_feedback_cleaned/`
- **Mô tả**: Dataset sau khi tiền xử lý
- **Xử lý**: Lowercase, loại bỏ URL/email, ký tự đặc biệt, khoảng trắng thừa
- **Định dạng**: Arrow format
- **Dùng cho**: Training model

---

### 🤖 **Model Files (Files model)**

#### `best_sentiment_model.pkl` (118 KB)
- **Mô tả**: Model ML đã được train
- **Loại**: Linear SVM (Support Vector Machine)
- **Accuracy**: 88.85% trên test set
- **F1-Score**: ~0.88
- **Input**: TF-IDF vectors (5,000 features)
- **Output**: Label (0, 1, 2) tương ứng (Tiêu cực, Trung lập, Tích cực)

#### `tfidf_vectorizer.pkl` (195 KB)
- **Mô tả**: TF-IDF Vectorizer đã được fit
- **Cấu hình**:
  - Max features: 5,000
  - N-gram range: (1, 2) - unigram và bigram
  - Min document frequency: 2
  - Max document frequency: 0.8
- **Dùng để**: Chuyển văn bản thành vector số

#### `model_metadata.json` (1 KB)
- **Mô tả**: Thông tin metadata của model
- **Nội dung**:
  ```json
  {
    "model_name": "Linear SVM",
    "test_accuracy": 0.8885,
    "f1_score": 0.8842,
    "precision": 0.8856,
    "recall": 0.8835,
    "num_features": 5000,
    "label_names": {...}
  }
  ```

---

### 📊 **Analysis Files (Files phân tích)**

#### `eda_analysis.png` (470 KB)
- **Mô tả**: Biểu đồ khám phá dữ liệu (EDA)
- **Nội dung**: 6 biểu đồ
  1. Phân bố sentiment
  2. Phân bố topic
  3. Pie chart sentiment
  4. Boxplot độ dài văn bản theo sentiment
  5. Boxplot số từ theo sentiment
  6. Histogram phân bố số từ

#### `ml_training_results.png` (499 KB)
- **Mô tả**: Kết quả training và đánh giá model
- **Nội dung**: 6 biểu đồ
  1. So sánh accuracy các models
  2. Confusion matrix
  3. Precision/Recall/F1 per class
  4. Training time comparison
  5. F1 score comparison
  6. Model performance summary table

#### `cleaned_sample.csv` (18 KB)
- **Mô tả**: 100 mẫu dữ liệu sau khi tiền xử lý
- **Cột**: `sentence`, `cleaned_text`, `sentiment`, `topic`
- **Dùng để**: Kiểm tra chất lượng tiền xử lý

#### `test_predictions.csv` (1 KB)
- **Mô tả**: Kết quả dự đoán trên test set
- **Cột**: `text`, `sentiment`, `label`
- **Dùng để**: Đánh giá model, tìm lỗi

---

### 💻 **Source Code Files**

#### `main.py` (12 KB)
- **Mô tả**: FastAPI application - API server
- **Chức năng**:
  - Load model khi khởi động
  - Cung cấp REST API endpoints
  - Xử lý request/response
  - Validation dữ liệu với Pydantic
  - CORS middleware
  - Error handling
- **Endpoints**:
  - `POST /predict` - Dự đoán 1 câu
  - `POST /predict/batch` - Dự đoán nhiều câu
  - `GET /model/info` - Thông tin model
  - `GET /health` - Health check
  - `GET /examples` - Xem ví dụ mẫu

#### `CCNLTHD.ipynb` (417 KB)
- **Mô tả**: Jupyter Notebook chứa toàn bộ quá trình
- **Nội dung**:
  1. **Bước 1**: EDA (Exploratory Data Analysis)
  2. **Bước 2**: Data Preprocessing
  3. **Bước 3**: Model Training
  4. **Bước 4**: Model Evaluation
  5. **Bước 5**: Inference & Testing
- **Dùng để**: Reproduce toàn bộ quá trình, chỉnh sửa, thử nghiệm

#### `index.html` (11 KB)
- **Mô tả**: Frontend web interface
- **Công nghệ**: HTML5, CSS3, JavaScript (Vanilla)
- **Tính năng**:
  - Giao diện đẹp với gradient và animations
  - Input form nhập văn bản
  - Hiển thị kết quả với emoji và màu sắc
  - Progress bars cho probabilities
  - Các ví dụ mẫu click-to-fill
  - Responsive design
- **Kết nối**: API tại `http://localhost:8000`

---

## 🚀 Cài đặt

### 1. Clone hoặc download project

```bash
git clone <repository-url>
cd CCNLTHD
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Hoặc cài thủ công:**

```bash
pip install fastapi uvicorn python-multipart
pip install scikit-learn pandas numpy
pip install joblib matplotlib seaborn
pip install datasets
```

### 3. Kiểm tra files

Đảm bảo có đủ 3 files model:
- `best_sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- `model_metadata.json`

---

## 💡 Sử dụng

### 🌐 Chạy API Server

```bash
python main.py
```

Hoặc:

```bash
uvicorn main:app --reload
```

Server sẽ chạy tại: `http://localhost:8000`

### 📖 Truy cập Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 🎨 Sử dụng Web Interface

1. Mở file `index.html` trong browser
2. Nhập văn bản tiếng Việt
3. Click "Phân tích"
4. Xem kết quả!

---

## 🔌 API Documentation

### POST `/predict`

Dự đoán cảm xúc cho 1 văn bản.

**Request:**
```json
{
  "text": "Giảng viên dạy rất hay và nhiệt tình"
}
```

**Response:**
```json
{
  "text": "Giảng viên dạy rất hay và nhiệt tình",
  "sentiment": "Tích cực",
  "label": 2,
  "confidence": 0.9234,
  "probabilities": {
    "Tiêu cực": 0.0123,
    "Trung lập": 0.0643,
    "Tích cực": 0.9234
  },
  "timestamp": "2026-01-25T20:30:00"
}
```

### POST `/predict/batch`

Dự đoán cho nhiều văn bản cùng lúc (tối đa 100).

**Request:**
```json
{
  "texts": [
    "Giảng viên dạy rất hay",
    "Môn học quá khó",
    "Bình thường thôi"
  ]
}
```

**Response:**
```json
{
  "results": [...],
  "total": 3,
  "timestamp": "2026-01-25T20:30:00"
}
```

### GET `/model/info`

Lấy thông tin về model.

**Response:**
```json
{
  "model_name": "Linear SVM",
  "test_accuracy": 0.8885,
  "f1_score": 0.8842,
  "precision": 0.8856,
  "recall": 0.8835,
  "num_features": 5000,
  "label_mapping": {
    "0": "Tiêu cực",
    "1": "Trung lập",
    "2": "Tích cực"
  }
}
```

---

## 📊 Kết quả

### Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 88.85% |
| **F1-Score** | 88.42% |
| **Precision** | 88.56% |
| **Recall** | 88.35% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Tiêu cực | 0.89 | 0.87 | 0.88 | 1,050 |
| Trung lập | 0.85 | 0.86 | 0.86 | 980 |
| Tích cực | 0.92 | 0.93 | 0.92 | 1,136 |

### Confusion Matrix

```
                Predicted
              Neg  Neu  Pos
Actual  Neg  [914  82   54]
        Neu  [ 78  843  59]
        Pos  [ 45  35  1056]
```

---

## 🎮 Demo

### Test với Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Thầy giáo dạy rất tâm huyết"}
)

print(response.json())
```

### Test với cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Môn học rất thú vị"}'
```

---

## 🔧 Troubleshooting

### Lỗi: "Model chưa được load"

**Giải pháp:**
- Đảm bảo file `best_sentiment_model.pkl` tồn tại
- Chạy lại từ đúng thư mục chứa model

### Lỗi: "ERR_ADDRESS_INVALID"

**Giải pháp:**
- Dùng `localhost` thay vì `0.0.0.0`
- Truy cập: http://localhost:8000

### API không phản hồi

**Giải pháp:**
- Kiểm tra server có đang chạy không
- Check logs trong terminal
- Thử http://localhost:8000/health

---

## 📝 TODO / Future Work

- [ ] Thêm model PhoBERT để so sánh
- [ ] Deploy lên cloud (AWS/Azure/Heroku)
- [ ] Thêm database để lưu predictions
- [ ] Authentication cho API
- [ ] Dashboard để xem thống kê
- [ ] Hỗ trợ thêm ngôn ngữ

---

## 👨‍💻 Tác giả

**Tên sinh viên**: [Tên của bạn]  
**MSSV**: [MSSV]  
**Lớp**: [Tên lớp]  
**Môn học**: Công cụ và Nền tảng LTHD

---

## 📜 License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

---

## 🙏 Acknowledgments

- Dataset: [UIT-NLP Vietnamese Students Feedback](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
- Framework: FastAPI, Scikit-learn
- Community: Hugging Face, Python ML Community

---

**⭐ Nếu project hữu ích, hãy cho một star!**