import joblib
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI với metadata
app = FastAPI(
    title="Spam Classifier API",
    description="API phân loại tin nhắn rác sử dụng Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Thêm CORS middleware để cho phép truy cập từ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tải mô hình đã được huấn luyện
MODEL_PATH = "spam_classifier_pipeline.pkl"
model = None

def load_model():
    """Tải mô hình từ file"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Đã tải mô hình thành công từ {MODEL_PATH}")
            return True
        else:
            logger.error(f"❌ Không tìm thấy file mô hình: {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
        return False

# Tải mô hình khi khởi động
load_model()

# Định nghĩa Pydantic models
class PredictRequest(BaseModel):
    text: str = Field(..., description="Tin nhắn cần phân loại", min_length=1, max_length=1000)

class PredictResponse(BaseModel):
    text: str
    prediction: str
    confidence: Optional[float] = None
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str

# Endpoint kiểm tra sức khỏe API
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Kiểm tra trạng thái hoạt động của API và mô hình
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        message="API đang hoạt động bình thường" if model is not None else "Mô hình chưa được tải"
    )

# Endpoint reload mô hình
@app.post("/reload-model", tags=["Admin"])
def reload_model():
    """
    Tải lại mô hình (hữu ích khi có mô hình mới)
    """
    success = load_model()
    if success:
        return {"status": "success", "message": "Đã tải lại mô hình thành công"}
    else:
        raise HTTPException(status_code=500, detail="Không thể tải lại mô hình")

# Endpoint chính để dự đoán
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Phân loại tin nhắn thành spam hoặc ham
    
    - **text**: Nội dung tin nhắn cần phân loại
    
    Trả về:
    - **prediction**: "spam" hoặc "ham"
    - **confidence**: Độ tin cậy (nếu có)
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Mô hình chưa được tải. Vui lòng kiểm tra lại."
        )

    try:
        # Dự đoán
        prediction = model.predict([request.text])[0]
        
        # Tính confidence nếu có thể
        confidence = None
        try:
            if hasattr(model.named_steps["clf"], "predict_proba"):
                # Lấy probability từ classifier
                probabilities = model.predict_proba([request.text])[0]
                classes = model.named_steps["clf"].classes_
                pred_idx = list(classes).index(prediction)
                confidence = float(probabilities[pred_idx])
        except Exception as e:
            logger.warning(f"Không thể tính confidence: {str(e)}")
        
        return PredictResponse(
            text=request.text,
            prediction=prediction,
            confidence=confidence,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình dự đoán: {str(e)}"
        )

# Endpoint thông tin mô hình
@app.get("/model-info", tags=["Info"])
def get_model_info():
    """
    Lấy thông tin về mô hình đang sử dụng
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được tải")
    
    try:
        info = {
            "model_type": type(model).__name__,
            "steps": [],
            "classes": None
        }
        
        # Lấy thông tin các bước trong pipeline
        if hasattr(model, 'named_steps'):
            info["steps"] = list(model.named_steps.keys())
            
            # Lấy classes nếu có
            if "clf" in model.named_steps:
                classifier = model.named_steps["clf"]
                if hasattr(classifier, "classes_"):
                    info["classes"] = classifier.classes_.tolist()
        
        return info
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin mô hình: {str(e)}")
        raise HTTPException(status_code=500, detail="Không thể lấy thông tin mô hình")

# Endpoint demo
@app.get("/", tags=["Demo"])
def root():
    """
    Trang chủ API với thông tin cơ bản
    """
    return {
        "message": "🚀 Spam Classifier API",
        "description": "API phân loại tin nhắn rác sử dụng Machine Learning",
        "endpoints": {
            "predict": "/predict - Phân loại tin nhắn",
            "health": "/health - Kiểm tra trạng thái",
            "docs": "/docs - Tài liệu API",
            "model_info": "/model-info - Thông tin mô hình"
        },
        "status": "ready" if model is not None else "model_not_loaded"
    }

# Khởi chạy server (cho development)
if __name__ == "__main__":
    import uvicorn
    print("🚀 Khởi động Spam Classifier API...")
    print("📚 Tài liệu API: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)