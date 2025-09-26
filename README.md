# 📩 Phân loại tin nhắn rác - Machine Learning System

Hệ thống phân loại tin nhắn rác hoàn chỉnh với giao diện Streamlit và REST API, sử dụng các thuật toán học máy phổ biến.

## 🚀 Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

#### Option A: Streamlit GUI (Giao diện đồ họa)
```bash
streamlit run BaiTapLon.py
```
Truy cập: `http://localhost:8501`

#### Option B: FastAPI REST API 
```bash
python api.py
```
Truy cập: 
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

#### Option C: Chạy cả hai cùng lúc
```bash
# Terminal 1
streamlit run BaiTapLon.py

# Terminal 2  
python api.py
```

## 📋 Quy trình học máy đã triển khai

### 1. Data Source (Nguồn dữ liệu)

- **Upload CSV**: Tải file CSV với cột 'text' và 'label'
- **Dữ liệu mẫu**: Sử dụng 10 tin nhắn mẫu có sẵn
- **Kaggle Dataset**: Tải trực tiếp từ Kaggle

### 2. Preprocessing (Tiền xử lý)

- Làm sạch ký tự, chuyển lowercase
- Tách từ tiếng Việt đơn giản
- Xử lý stopwords, emoji, teencode
- Thay thế URL/EMAIL/PHONE/MONEY bằng placeholder

### 3. Feature Engineering (Trích chọn đặc trưng)

- **BoW (Bag of Words)** hoặc **TF-IDF**
- **N-gram** (1-3) để bắt cụm từ
- **Feature selection** với Chi-square hoặc Mutual Information

### 4. Classification Models

- **Naive Bayes (Multinomial)**
- **Logistic Regression**
- **Linear SVM** & **SVM (RBF Kernel)**
- **Random Forest**
- **K-NN**
- **Decision Tree**

### 5. Evaluation & Results

- **Cross-validation** (5-fold)
- **Accuracy, Precision, Recall, F1-score**
- **ROC-AUC curve** (cho binary classification)
- **Confusion Matrix**
- **Feature Importance** (cho tree-based models)
- **Thời gian training** và **số tham số**

## 🎯 Tính năng chính

### 🖥️ Streamlit GUI
- ✅ Giao diện thân thiện với người dùng
- ✅ Hỗ trợ nhiều nguồn dữ liệu (CSV, Kaggle, mẫu)
- ✅ Tiền xử lý văn bản tiếng Việt toàn diện
- ✅ Nhiều thuật toán học máy với hyperparameter tuning
- ✅ Đánh giá toàn diện với visualization (EDA, metrics, confusion matrix)
- ✅ So sánh hiệu quả các phương pháp
- ✅ Xuất mô hình đã huấn luyện (PKL format)
- ✅ Export dữ liệu phân tích sang CSV (metadata, predictions, vocabulary, features)
- ✅ Dự đoán tin nhắn mới real-time

### 🔗 REST API 
- ✅ FastAPI với Swagger UI documentation
- ✅ Endpoint phân loại tin nhắn (`/predict`)
- ✅ Health check endpoint (`/health`)
- ✅ Model info endpoint (`/model-info`) 
- ✅ Model reload endpoint (`/reload-model`)
- ✅ CORS support cho frontend integration
- ✅ Confidence score trong prediction
- ✅ Comprehensive error handling & logging

## 📊 Cách sử dụng

### 🖥️ Streamlit GUI
1. **Chọn nguồn dữ liệu** từ sidebar (CSV, Kaggle, hoặc mẫu)
2. **Cấu hình tiền xử lý** (lowercase, tách từ, stopwords, teencode, etc.)
3. **Chọn vectorizer** (BoW/TF-IDF/Sentence Embeddings) và feature selection
4. **Chọn mô hình** và điều chỉnh hyperparameters
5. **Bấm "Train & Evaluate"** để huấn luyện và đánh giá
6. **Xem kết quả** chi tiết với visualization và metrics
7. **Dự đoán tin nhắn mới** ở phần bên phải
8. **Download files:**
   - 📥 **Model (PKL)**: Mô hình hoàn chỉnh để deploy
   - 📊 **Metadata (CSV)**: Thông tin metrics và cấu hình
   - 🎯 **Predictions (CSV)**: Kết quả dự đoán chi tiết với confidence scores
   - 📚 **Vocabulary (CSV)**: Từ vựng đã học (TF-IDF/BoW)
   - ⚡ **Features (CSV)**: Feature importance (Random Forest/Decision Tree)

### 🔗 REST API

#### Test với curl:
```bash
# Kiểm tra sức khỏe API
curl -X GET "http://localhost:8000/health"

# Phân loại tin nhắn
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Khuyến mãi 50% tất cả sản phẩm!"}'

# Lấy thông tin mô hình
curl -X GET "http://localhost:8000/model-info"
```

#### Test với Python:
```python
import requests

# Phân loại tin nhắn
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Bạn có khỏe không?"}
)
print(response.json())
# Output: {"text": "Bạn có khỏe không?", "prediction": "ham", "confidence": 0.95, "status": "success"}
```

#### Tích hợp vào ứng dụng:
- React/Vue.js frontend
- Mobile applications
- Other web services
- Chatbots & messaging platforms

## 💡 Gợi ý

- Với dữ liệu lớn: Logistic Regression/Linear SVM + TF-IDF
- Thử n-gram 1-2 để bắt cụm từ như "khuyến mãi", "tải app"
- Cân bằng lớp dữ liệu nếu spam/ham không đều
- Sử dụng thư viện _underthesea_ cho tách từ tiếng Việt chuyên nghiệp

## 📁 Cấu trúc file

```
BaiTapLon/
├── BaiTapLon.py                    # 🖥️ Streamlit GUI chính
├── api.py                          # 🔗 FastAPI REST API  
├── text_preprocessor.py            # 🔧 Text preprocessing utilities
├── requirements.txt                # 📦 Dependencies
├── README.md                       # 📖 Hướng dẫn này
├── API_USAGE.md                    # 📚 Hướng dẫn sử dụng API chi tiết
├── spam_classifier_pipeline.pkl    # 🤖 Mô hình đã huấn luyện (sau khi train)
├── train.csv                       # 📊 Dữ liệu training (nếu có)
├── vietnamese-stopwords.txt        # 🛑 Stopwords tiếng Việt
└── teencode_dict.txt              # 📝 Từ điển teencode
```

## 🔧 Yêu cầu hệ thống

- Python 3.8+
- RAM: Tối thiểu 4GB (khuyến nghị 8GB)
- Dung lượng: ~500MB cho dependencies
- Port 8000 (API) và 8501 (Streamlit) available

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Trang chủ API với thông tin cơ bản |
| POST | `/predict` | Phân loại tin nhắn spam/ham |
| GET | `/health` | Kiểm tra trạng thái API & model |
| GET | `/model-info` | Thông tin chi tiết về mô hình |
| POST | `/reload-model` | Tải lại mô hình từ file |
| GET | `/docs` | Swagger UI documentation |

## 🎯 Use Cases

### 🏢 **Business Applications**
- Email spam filtering
- SMS marketing compliance
- Customer service message routing
- Social media content moderation

### 🔬 **Research & Education**  
- NLP preprocessing pipeline demo
- ML algorithm comparison
- Vietnamese text processing research
- Student ML projects

### 🛠️ **Development**
- Microservice integration
- API-first ML deployment
- Frontend/mobile app backend
- MLOps pipeline component

## 🚀 Production Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "api.py"]
```

### Manual Deployment
```bash
# Production server with gunicorn
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
```
