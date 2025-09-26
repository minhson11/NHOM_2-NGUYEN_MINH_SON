# -*- coding: utf-8 -*-
"""
Streamlit GUI: Phân loại tin nhắn rác (SMS/Chat/Email ngắn)

Quy trình tích hợp trong app:
1) Data Source: Upload CSV hoặc dùng mẫu tích hợp.
2) Preprocessing: làm sạch ký tự, lowercase, tách từ đơn giản, stopword, xử lý URL/emoji/teencode.
3) Feature / Representation: BoW hoặc TF-IDF, n-gram, min_df, max_df.
4) Classification Model: Naive Bayes / Logistic Regression / Linear SVM / KNN / Decision Tree.
5) Evaluation & Results: Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report + xuất model.

Chạy:
    pip install streamlit scikit-learn pandas numpy matplotlib joblib
    streamlit run app.py

Gợi ý dữ liệu: CSV có 2 cột [text, label]. label có thể là "spam"/"ham" hoặc tuỳ bạn.
"""

import io
import re
import time
import string
import unicodedata
from typing import List, Tuple, Optional, Union

import joblib
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import streamlit as st
from text_preprocessor import TextPreprocessor

st.set_page_config(page_title="Phân loại tin nhắn rác", page_icon="📩", layout="wide")

# ======= CSS đẹp mắt và hiện đại =======
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        margin: 10px;
        padding: 15px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .kpi {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section styling */
    .section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Text areas */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e1e5e9;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Checkbox */
    .stCheckbox > label > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader */
    .stFileUploader > div {
        border-radius: 15px;
        border: 2px dashed #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 15px;
        color: white;
    }
    
    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-radius: 15px;
        padding: 15px;
        color: white;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        border-radius: 15px;
        padding: 15px;
        color: white;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #48cae4 0%, #023e8a 100%);
        border-radius: 15px;
        padding: 15px;
        color: white;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 0 0 10px 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== Stopwords đơn giản (VN + dấu/không dấu) ==================
# ================ đơn giản (VN + dấu/không dấu) ==================
VN_STOPWORDS = {
    # thường gặp
    "là","và","của","có","cho","nhé","nhỉ","thì","lại","này","kia","ấy","đã","đang","sẽ",
    "rồi","với","được","khi","đi","đến","từ","trong","ra","vào","cũng","nhưng","nếu","vì","do",
    "trên","dưới","nữa","nên","thôi","ơi","ơi!","à","ạ","ạ!","hả","hà","ha","ư","ừ","ờ",
    # không dấu
    "la","va","cua","co","cho","nhe","nhi","thi","lai","nay","kia","ay","da","dang","se",
    "roi","voi","duoc","khi","di","den","tu","trong","ra","vao","cung","nhung","neu","vi","do",
    "tren","duoi","nua","nen","thoi","oi","oi!","a","ha","u","u~","o","ok","okie","okay",
}

# ================== Teencode map mở rộng ==================
TEENCODE_MAP = {
    # Phủ định
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "khong": "không",
    "kg": "không", "k0": "không", "0": "không",
    
    # Được/không được
    "dc": "được", "đc": "được", "dcf": "được", "duoc": "được",
    "kdc": "không được", "kdcf": "không được",
    
    # Đại từ
    "mik": "mình", "mk": "mình", "m": "mình", "minh": "mình",
    "bn": "bạn", "ban": "bạn", "b": "bạn",
    "t": "tôi", "toi": "tôi", "tao": "tôi",
    "n": "nó", "no": "nó", "nó": "nó",
    
    # Liên hệ
    "ae": "anh em", "ad": "admin", "ib": "inbox",
    "sdt": "số điện thoại", "lh": "liên hệ", "lienhe": "liên hệ",
    "vs": "với", "voi": "với", "w": "với",
    
    # Câu hỏi
    "j": "gì", "ji": "gì", "gi": "gì",
    "sao": "sao", "s": "sao",
    "nao": "nào", "nao": "nào",
    
    # Cảm thán
    "vl": "rất", "vcl": "rất", "rat": "rất",
    "oi": "ôi", "ui": "ui",
    
    # Thời gian
    "hum": "hôm", "hom": "hôm", "hnay": "hôm nay",
    "ngay": "ngày", "ng": "ngày",
    
    # Khác
    "ok": "ok", "oke": "ok", "okie": "ok",
    "thik": "thích", "thich": "thích",
    "iu": "yêu", "yeu": "yêu",
    "xau": "xấu", "xau": "xấu",
    "dep": "đẹp", "dep": "đẹp",
}

# ================== Tiền xử lý ==================
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\- ]{7,}\d)\b")
MONEY_PATTERN = re.compile(r"\b\d+[.,]?\d*\s*(?:k|đ|vnd|vnđ|usd|$)\b", flags=re.IGNORECASE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_teencode(text: str) -> str:
    """Chuẩn hóa teencode tiếng Việt"""
    tokens = text.split()
    out = []
    for t in tokens:
        key = t.lower().strip()
        # Loại bỏ dấu câu ở cuối từ
        punct_end = ""
        while key and key[-1] in string.punctuation:
            punct_end = key[-1] + punct_end
            key = key[:-1]
        
        # Tìm trong teencode map
        normalized = TEENCODE_MAP.get(key, t)
        out.append(normalized + punct_end)
    return " ".join(out)


def simple_vietnamese_tokenize(text: str) -> str:
    """Tách từ tiếng Việt đơn giản - thêm khoảng trắng trước/sau dấu câu"""
    # Thêm khoảng trắng trước/sau dấu câu
    text = re.sub(r'([.!?;:,])', r' \1 ', text)
    # Thêm khoảng trắng trước/sau dấu ngoặc
    text = re.sub(r'([()\[\]{}])', r' \1 ', text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def basic_preprocess(text: str,
        lowercase: bool = True,
        rm_punct: bool = True,
        rm_numbers: bool = False,
        rm_emoji: bool = True,
        use_teencode: bool = True,
        rm_stopwords: bool = True,
        keep_placeholders: bool = True,
        use_tokenize: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)

    # Tách từ tiếng Việt (nếu bật)
    if use_tokenize:
        text = simple_vietnamese_tokenize(text)

    # placeholders
    if keep_placeholders:
        text = URL_PATTERN.sub(" <URL> ", text)
        text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
        text = PHONE_PATTERN.sub(" <PHONE> ", text)
        text = MONEY_PATTERN.sub(" <MONEY> ", text)
    else:
        text = URL_PATTERN.sub(" ", text)
        text = EMAIL_PATTERN.sub(" ", text)
        text = PHONE_PATTERN.sub(" ", text)
        text = MONEY_PATTERN.sub(" ", text)

    if rm_emoji:
        text = EMOJI_PATTERN.sub(" ", text)

    if lowercase:
        text = text.lower()

    if use_teencode:
        text = normalize_teencode(text)

    # remove punctuation
    if rm_punct:
        text = text.translate(PUNCT_TABLE)

    # remove numbers (optional)
    if rm_numbers:
        text = re.sub(r"\d+", " ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    # simple stopword filter
    if rm_stopwords:
        tokens = [t for t in text.split() if t not in VN_STOPWORDS]
        text = " ".join(tokens)

    return text


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="distiluse-base-multilingual-cased"):
        self.model_name = model_name
        self.model = None
        
    def fit(self, X, y=None):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                st.error(f"Lỗi tải mô hình {self.model_name}: {e}")
                # Fallback to a simpler model
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self
    
    def transform(self, X):
        if self.model is None:
            raise ValueError("Model chưa được khởi tạo. Hãy gọi fit() trước.")
        
        # Convert to embeddings
        embeddings = self.model.encode(X, convert_to_tensor=False)
        return embeddings


# ============ Dữ liệu mẫu ============
SAMPLE_DATA = pd.DataFrame({
    "text": [
        "Chúc mừng! Bạn đã trúng thưởng 1 triệu đồng. Nhấn link để nhận: bit.ly/abc123",
        "Hôm nay trời đẹp quá, đi chơi không bạn?",
        "Khuyến mãi 50% tất cả sản phẩm. Gọi ngay 0123456789 để đặt hàng!",
        "Cảm ơn bạn đã mua hàng. Đơn hàng sẽ được giao trong 2-3 ngày.",
        "Tải app ngay để nhận 100k tiền thưởng. Link: app.com/install",
        "Bạn có khỏe không? Lâu rồi không gặp.",
        "Quà tặng đặc biệt! Mua 1 tặng 1. Chỉ hôm nay thôi!",
        "Hẹn gặp bạn lúc 7h tối nay nhé.",
        "Trúng thưởng iPhone 15 Pro Max! Nhấn đây để nhận: lucky.com/win",
        "Cảm ơn bạn đã tham gia chương trình khuyến mãi của chúng tôi."
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
})

# ============ Sidebar ============
st.sidebar.title("⚙️ Cấu hình thí nghiệm")

st.sidebar.subheader("1) Nguồn dữ liệu")
data_source = st.sidebar.radio("Chọn nguồn dữ liệu", 
    ["📁 Upload CSV", "📊 Dữ liệu mẫu", "🌐 Kaggle Dataset"], 
    index=1)

if data_source == "📁 Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=['csv'], help="File CSV cần có 2 cột: 'text' và 'label'")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.sidebar.success("✅ File đã được tải lên")
    else:
        st.session_state.uploaded_file = None

elif data_source == "📊 Dữ liệu mẫu":
    st.sidebar.info("Sử dụng dữ liệu mẫu có sẵn (10 tin nhắn)")
    st.session_state.uploaded_file = None

elif data_source == "🌐 Kaggle Dataset":
    kaggle_slug = st.sidebar.text_input("Kaggle dataset slug", "thedevastator/sms-spam-collection-a-more-diverse-dataset")
    dl_btn = st.sidebar.button("⬇️ Tải từ Kaggle", use_container_width=True)
    if dl_btn:
        try:
            import kagglehub
            kaggle_path = kagglehub.dataset_download(kaggle_slug)
            st.session_state.kaggle_path = kaggle_path
            st.sidebar.success(f"Đã tải về: {kaggle_path}")
        except Exception as e:
            st.sidebar.error(f"Lỗi tải Kaggle: {e}")
    if "kaggle_path" in st.session_state:
        files = sorted(glob.glob(os.path.join(st.session_state.kaggle_path, "**", "*.csv"), recursive=True) +
            glob.glob(os.path.join(st.session_state.kaggle_path, "**", "*.tsv"), recursive=True) +
            glob.glob(os.path.join(st.session_state.kaggle_path, "**", "*.txt"), recursive=True))
        if files:
            kaggle_file = st.sidebar.selectbox("Chọn file dữ liệu", files)
            st.session_state.kaggle_selected_file = kaggle_file
            st.sidebar.caption("App sẽ tự nhận diện cột văn bản và nhãn (ví dụ v1/v2 → label/text). Nếu không đúng, bạn có thể đổi tên cột trong file.")
        else:
            st.sidebar.warning("Không tìm thấy file .csv/.tsv/.txt trong thư mục Kaggle vừa tải.")

st.sidebar.subheader("2) Tiền xử lý")
lowercase = st.sidebar.checkbox("Lowercase", value=True)
use_tokenize = st.sidebar.checkbox("Tách từ tiếng Việt", value=True, help="Thêm khoảng trắng trước/sau dấu câu")
rm_punct = st.sidebar.checkbox("Bỏ dấu câu", value=True)
rm_numbers = st.sidebar.checkbox("Bỏ số", value=False)
rm_emoji = st.sidebar.checkbox("Bỏ emoji", value=True)
use_teencode = st.sidebar.checkbox("Chuẩn hoá teencode", value=True, help="Chuyển đổi teencode sang tiếng Việt chuẩn")
rm_stopwords = st.sidebar.checkbox("Bỏ stopwords", value=True)
keep_placeholders = st.sidebar.checkbox("Giữ <URL>/<EMAIL>/<PHONE>/<MONEY>", value=True)

st.sidebar.subheader("3) Biểu diễn đặc trưng")
feat_type = st.sidebar.radio("Vectorizer", 
                            ["BoW", "TF-IDF", "Sentence Embeddings"] if SENTENCE_TRANSFORMERS_AVAILABLE else ["BoW", "TF-IDF"], 
                            index=1)

if feat_type in ["BoW", "TF-IDF"]:
    ngram_max = st.sidebar.selectbox("N-gram tối đa", options=[1,2,3], index=1)
    min_df = st.sidebar.slider("min_df", 1, 10, 2)
    max_df = st.sidebar.slider("max_df (%)", 50, 100, 100)
    
    # Feature selection
    use_feature_selection = st.sidebar.checkbox("Feature Selection", value=False, help="Chọn đặc trưng quan trọng nhất")
    if use_feature_selection:
        feature_selection_method = st.sidebar.selectbox("Phương pháp", ["Chi-square", "Mutual Information"], index=0)
        n_features = st.sidebar.slider("Số đặc trưng", 100, 5000, 1000, 100)

elif feat_type == "Sentence Embeddings":
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        embedding_model = st.sidebar.selectbox("Mô hình embedding", [
            "all-MiniLM-L6-v2",  # Fast and good for most tasks
            "all-mpnet-base-v2",  # Better quality, slower
            "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
            "distiluse-base-multilingual-cased"  # Multilingual, good for Vietnamese
        ], index=3, help="Chọn mô hình pre-trained cho sentence embeddings")
        
        st.sidebar.info("💡 Mô hình multilingual sẽ hoạt động tốt hơn với tiếng Việt")
    else:
        st.sidebar.error("❌ Cần cài đặt sentence-transformers: pip install sentence-transformers")

st.sidebar.subheader("4) Mô hình")
model_name = st.sidebar.selectbox("Thuật toán", [
    "Naive Bayes (Multinomial)",
    "Logistic Regression",
    "Linear SVM",
    "SVM (RBF Kernel)",
    "Random Forest",
    "KNN",
    "Decision Tree",
], index=0)

# Hyperparams
if model_name == "Naive Bayes (Multinomial)":
    alpha = st.sidebar.slider("alpha", 0.0, 2.0, 1.0, 0.1)
elif model_name == "Logistic Regression":
    C = st.sidebar.slider("C (reg strength)", 0.01, 5.0, 1.0, 0.01)
    max_iter = st.sidebar.slider("max_iter", 100, 2000, 800, 100)
elif model_name == "Linear SVM":
    C = st.sidebar.slider("C (margin)", 0.01, 5.0, 1.0, 0.01)
elif model_name == "SVM (RBF Kernel)":
    C = st.sidebar.slider("C (margin)", 0.01, 5.0, 1.0, 0.01)
    gamma = st.sidebar.selectbox("gamma", ["scale", "auto", 0.001, 0.01, 0.1, 1.0], index=0)
elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100, 10)
    max_depth = st.sidebar.slider("max_depth", 1, 50, 10)
    min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, 2)
elif model_name == "KNN":
    n_neighbors = st.sidebar.slider("n_neighbors", 1, 20, 5)
    metric = st.sidebar.selectbox("metric", ["minkowski", "euclidean", "manhattan"], index=0)
elif model_name == "Decision Tree":
    max_depth = st.sidebar.slider("max_depth", 1, 50, 10)

st.sidebar.subheader("5) Đánh giá")
test_size = st.sidebar.slider("Tỷ lệ test", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)
pos_label = st.sidebar.text_input("Nhãn dương (ví dụ: spam)", value="spam")

# ============ DataFrame ============
@st.cache_data(show_spinner=False)
def load_data(file: Optional[Union[str, io.BytesIO]], encoding: str = "utf-8") -> pd.DataFrame:
    if file is None:
        df = SAMPLE_DATA.copy()
        return df
    else:
        try:
            return pd.read_csv(file, encoding=encoding)
        except Exception as e:
            st.warning(f"Không đọc được CSV với encoding {encoding}: {e}")
            file.seek(0)
            return pd.read_csv(file, encoding="utf-8", errors="ignore")

# Đọc dữ liệu dựa trên nguồn đã chọn
df = None

# Xử lý upload file
if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
    try:
        df_raw = pd.read_csv(st.session_state.uploaded_file, encoding="utf-8")
    except Exception:
        st.session_state.uploaded_file.seek(0)
        df_raw = pd.read_csv(st.session_state.uploaded_file, encoding="latin-1")
    
    # Nhận diện cột text/label
    lower_map = {c.lower(): c for c in df_raw.columns}
    label_candidates = ["label", "category", "class", "target", "v1", "is_spam", "spam"]
    text_candidates  = ["text", "message", "sms", "content", "body", "v2"]
    label_col = next((lower_map[k] for k in label_candidates if k in lower_map), None)
    text_col  = next((lower_map[k] for k in text_candidates if k in lower_map), None)
    
    if label_col and text_col:
        df = df_raw.rename(columns={label_col: "label", text_col: "text"})[["text", "label"]]
        df["label"] = df["label"].astype(str).str.strip().str.lower().replace({"1":"spam","0":"ham"})
    else:
        st.warning("Không tự nhận diện được cột 'text' và 'label'. Vui lòng đổi tên cột trong file CSV.")
        df = df_raw

# Xử lý dữ liệu mẫu
elif data_source == "📊 Dữ liệu mẫu":
    df = SAMPLE_DATA.copy()

# Xử lý Kaggle dataset
elif "kaggle_selected_file" in st.session_state:
    kaggle_file = st.session_state.kaggle_selected_file
    try:
        df_raw = pd.read_csv(kaggle_file, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df_raw = pd.read_csv(kaggle_file, sep=None, engine="python", encoding="latin-1", on_bad_lines="skip")
    
    # Nhận diện cột text/label phổ biến
    lower_map = {c.lower(): c for c in df_raw.columns}
    label_candidates = ["label", "category", "class", "target", "v1", "is_spam", "spam"]
    text_candidates  = ["text", "message", "sms", "content", "body", "v2"]
    label_col = next((lower_map[k] for k in label_candidates if k in lower_map), None)
    text_col  = next((lower_map[k] for k in text_candidates if k in lower_map), None)
    
    if label_col and text_col:
        df = df_raw.rename(columns={label_col: "label", text_col: "text"})[["text", "label"]]
        df["label"] = df["label"].astype(str).str.strip().str.lower().replace({"1":"spam","0":"ham"})
    else:
        st.warning("Không tự nhận diện được cột. App sẽ giữ nguyên để bạn xem trước.")
        df = df_raw

# Không có dữ liệu
if df is None:
    st.info("📊 Vui lòng chọn nguồn dữ liệu từ sidebar bên trái.")
    df = pd.DataFrame(columns=["text", "label"])

# ============ Header đẹp mắt ============
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 3rem; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">📩 Phân loại tin nhắn rác</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 1rem 0 0 0;">Machine Learning GUI với Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============ Quy trình workflow ============
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
        <h3 style="text-align: center; color: #2c3e50; margin-bottom: 1.5rem;">🔄 Quy trình học máy</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="text-align: center; flex: 1; min-width: 150px; margin: 0.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px; font-weight: 600;">1️⃣ Nguồn dữ liệu</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 150px; margin: 0.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px; font-weight: 600;">2️⃣ Tiền xử lý</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 150px; margin: 0.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px; font-weight: 600;">3️⃣ Đặc trưng</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 150px; margin: 0.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px; font-weight: 600;">4️⃣ Mô hình</div>
            </div>
            <div style="text-align: center; flex: 1; min-width: 150px; margin: 0.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px; font-weight: 600;">5️⃣ Đánh giá</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============ Data Preview Section ============
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50; margin-bottom: 1rem; text-align: center;">🗂️ Xem trước dữ liệu</h3>
    </div>
    """,
    unsafe_allow_html=True
)

if not {"text", "label"}.issubset(set(df.columns)):
    st.error("CSV phải có cột 'text' và 'label'. Hãy đổi tên cột cho đúng.")
else:
    # Data stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="subtle">Tổng số dòng</div><div class="kpi">{len(df)}</div></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="subtle">Số nhãn</div><div class="kpi">{df["label"].nunique()}</div></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        if len(df) > 0:
            spam_count = df['label'].value_counts().get('spam', 0)
            st.markdown(
                f'<div class="metric-card"><div class="subtle">Spam</div><div class="kpi">{spam_count}</div></div>',
                unsafe_allow_html=True
            )
    
    with col4:
        if len(df) > 0:
            ham_count = df['label'].value_counts().get('ham', 0)
            st.markdown(
                f'<div class="metric-card"><div class="subtle">Ham</div><div class="kpi">{ham_count}</div></div>',
                unsafe_allow_html=True
            )

# Data table with better styling
if len(df) > 0:
    st.markdown("### 📊 Bảng dữ liệu mẫu")
    st.dataframe(df.head(10), use_container_width=True)
    
    # ============ Khám phá dữ liệu chi tiết ============
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 20px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h3 style="color: #2c3e50; margin-bottom: 1rem; text-align: center;">🔍 Khám phá dữ liệu (EDA)</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Data structure analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Cấu trúc dữ liệu")
        st.write(f"**Số dòng:** {len(df)}")
        st.write(f"**Số cột:** {len(df.columns)}")
        st.write(f"**Kiểu dữ liệu:**")
        for col in df.columns:
            st.write(f"- {col}: {df[col].dtype}")
    
    with col2:
        st.markdown("#### 📊 Thống kê cơ bản")
        if 'text' in df.columns:
            text_lengths = df['text'].astype(str).str.len()
            st.write(f"**Độ dài trung bình văn bản:** {text_lengths.mean():.1f} ký tự")
            st.write(f"**Độ dài tối thiểu:** {text_lengths.min()} ký tự")
            st.write(f"**Độ dài tối đa:** {text_lengths.max()} ký tự")
            st.write(f"**Độ lệch chuẩn:** {text_lengths.std():.1f} ký tự")
            st.write(f"**Số lượng tin nhắn rỗng:** {(df['text'].astype(str).str.strip() == '').sum()}")

    # Label distribution with visualization
    if 'label' in df.columns:
        st.markdown("#### 📈 Phân phối nhãn")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plotly pie chart for better visualization
            label_counts = df['label'].value_counts()
            fig = px.pie(values=label_counts.values, names=label_counts.index, 
                        title="Phân phối nhãn", color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Chi tiết phân phối:**")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"**{label}:** {count} ({percentage:.1f}%)")
            
            # Check for class imbalance
            max_count = label_counts.max()
            min_count = label_counts.min()
            imbalance_ratio = max_count / min_count
            st.write(f"**Tỷ lệ mất cân bằng:** {imbalance_ratio:.2f}")
            if imbalance_ratio > 2:
                st.warning("⚠️ Dữ liệu có thể bị mất cân bằng lớp!")
            else:
                st.success("✅ Dữ liệu cân bằng tốt")
    
    # Text length analysis
    if 'text' in df.columns:
        st.markdown("#### 📏 Phân tích độ dài văn bản")
        
        text_lengths = df['text'].astype(str).str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of text lengths
            fig = px.histogram(x=text_lengths, nbins=30, title="Phân phối độ dài văn bản",
            labels={'x': 'Độ dài (ký tự)', 'y': 'Số lượng'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by label
            if 'label' in df.columns:
                df_temp = df.copy()
                df_temp['text_length'] = text_lengths
                fig = px.box(df_temp, x='label', y='text_length', 
                title="Độ dài văn bản theo nhãn",
                labels={'text_length': 'Độ dài (ký tự)', 'label': 'Nhãn'})
                st.plotly_chart(fig, use_container_width=True)
    
    # Word frequency analysis
    if 'text' in df.columns:
        st.markdown("#### 🔤 Phân tích tần suất từ")
        
        # Get most common words
        all_text = ' '.join(df['text'].astype(str).str.lower())
        words = all_text.split()
        word_freq = pd.Series(words).value_counts().head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top words overall
            fig = px.bar(x=word_freq.values, y=word_freq.index, orientation='h',
                        title="20 từ xuất hiện nhiều nhất", 
                        labels={'x': 'Tần suất', 'y': 'Từ'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Word frequency by label
            if 'label' in df.columns:
                spam_words = ' '.join(df[df['label'] == 'spam']['text'].astype(str).str.lower()).split()
                ham_words = ' '.join(df[df['label'] == 'ham']['text'].astype(str).str.lower()).split()
                
                spam_freq = pd.Series(spam_words).value_counts().head(10)
                ham_freq = pd.Series(ham_words).value_counts().head(10)
                
                st.write("**Top từ trong Spam:**")
                for word, freq in spam_freq.items():
                    st.write(f"- {word}: {freq}")
                
                st.write("**Top từ trong Ham:**")
                for word, freq in ham_freq.items():
                    st.write(f"- {word}: {freq}")

# ============ Build pipeline ============
if feat_type == "BoW":
    vectorizer = CountVectorizer(ngram_range=(1, ngram_max), min_df=min_df, max_df=max_df/100.0)
elif feat_type == "TF-IDF":
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, max_df=max_df/100.0)
elif feat_type == "Sentence Embeddings":
    vectorizer = SentenceEmbeddingTransformer(model_name=embedding_model)

# model
if model_name == "Naive Bayes (Multinomial)":
    clf = MultinomialNB(alpha=alpha)
elif model_name == "Logistic Regression":
    clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=None)
elif model_name == "Linear SVM":
    clf = LinearSVC(C=C)
elif model_name == "KNN":
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
elif model_name == "Decision Tree":
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

# Build pipeline based on feature type
if feat_type == "Sentence Embeddings":
    # For sentence embeddings, we don't need preprocessing or feature selection
    pipe = Pipeline([
        ("vec", vectorizer),
        ("clf", clf)
    ])
else:
    # For BoW/TF-IDF, use full preprocessing pipeline
    pipeline_steps = [
        ("prep", TextPreprocessor(lowercase=lowercase, rm_punct=rm_punct, rm_numbers=rm_numbers,
                                   rm_emoji=rm_emoji, use_teencode=use_teencode,
                                   rm_stopwords=rm_stopwords, keep_placeholders=keep_placeholders,
                                   use_tokenize=use_tokenize)),
        ("vec", vectorizer),
    ]

    if use_feature_selection:
        if feature_selection_method == "Chi-square":
            feature_selector = SelectKBest(chi2, k=n_features)
        else:  # Mutual Information
            feature_selector = SelectKBest(mutual_info_classif, k=n_features)
        pipeline_steps.append(("feature_selection", feature_selector))

    pipeline_steps.append(("clf", clf))

    pipe = Pipeline(pipeline_steps)

# ============ Main Action Section ============
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 20px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: center;">⚡ Thực hiện phân loại</h3>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(
        """
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(102, 126, 234, 0.2);">
            <h4 style="color: #2c3e50; margin-bottom: 1rem; text-align: center;">🧪 Huấn luyện & Đánh giá</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    run_train = st.button("🚀 Train & Evaluate", use_container_width=True)

with col2:
    st.markdown(
        """
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(102, 126, 234, 0.2);">
            <h4 style="color: #2c3e50; margin-bottom: 1rem; text-align: center;">🧠 Dự đoán nhanh</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    predict_text = st.text_area("Nhập tin nhắn muốn kiểm tra", height=100, placeholder="Ví dụ: Khuyến mãi 50%...")
    do_predict = st.button("🔍 Phân loại", use_container_width=True)

# session store
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "report" not in st.session_state:
    st.session_state.report = None

# ============ Train ============
if run_train:
    if not {"text", "label"}.issubset(set(df.columns)):
        st.stop()
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y))>1 else None
    )

    start = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary" if len(np.unique(y))==2 else "macro", zero_division=0, pos_label=pos_label)

    st.session_state.trained_model = pipe

    # Results header
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 20px; margin: 2rem 0; text-align: center;">
            <h3 style="color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">📊 Kết quả đánh giá mô hình</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI cards with improved styling
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown('<div class="metric-card"><div class="subtle">Accuracy</div><div class="kpi">{:.3f}</div></div>'.format(acc), unsafe_allow_html=True)
    with kpi2:
        st.markdown('<div class="metric-card"><div class="subtle">Precision</div><div class="kpi">{:.3f}</div></div>'.format(pr), unsafe_allow_html=True)
    with kpi3:
        st.markdown('<div class="metric-card"><div class="subtle">Recall</div><div class="kpi">{:.3f}</div></div>'.format(rc), unsafe_allow_html=True)
    with kpi4:
        st.markdown('<div class="metric-card"><div class="subtle">F1-score</div><div class="kpi">{:.3f}</div></div>'.format(f1), unsafe_allow_html=True)

    st.caption(f"⏱️ Thời gian train: {train_time:.2f}s | Số mẫu train/test: {len(X_train)}/{len(X_test)}")

    # Classification report with better styling
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">📊 Classification Report</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    report_str = classification_report(y_test, y_pred, zero_division=0)
    st.code(report_str, language='text')
    st.session_state.report = report_str
    
    # Export predictions as CSV
    predictions_df = pd.DataFrame({
        'text': X_test,
        'true_label': y_test,
        'predicted_label': y_pred,
        'correct': y_test == y_pred
    })
    
    # Add confidence scores if available
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        try:
            probabilities = pipe.predict_proba(X_test)
            max_proba = np.max(probabilities, axis=1)
            predictions_df['confidence'] = max_proba
        except Exception:
            pass
    
    st.session_state.predictions_df = predictions_df

    # Confusion Matrix with improved styling
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">🔢 Confusion Matrix</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    labels_sorted = sorted(np.unique(np.concatenate([y_test, y_pred])))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=labels_sorted, yticklabels=labels_sorted,
    title='Confusion Matrix',
    ylabel='True Label',
    xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Store results for comparison
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = []
    
    st.session_state.comparison_results.append({
        "method": feat_type,
        "model": model_name,
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "train_time": train_time
    })

    # Export model with better styling
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">💾 Xuất mô hình đã huấn luyện</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export PKL model
        buffer = io.BytesIO()
        joblib.dump(pipe, buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 Model (PKL)",
            data=buffer,
            file_name="spam_classifier_pipeline.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )
    
    with col2:
        # Export model metadata as CSV
        model_metadata = {
            "metric": ["accuracy", "precision", "recall", "f1_score", "train_time"],
            "value": [acc, pr, rc, f1, train_time],
            "model_type": [model_name] * 5,
            "vectorizer": [feat_type] * 5,
            "test_size": [test_size] * 5,
            "train_samples": [len(X_train)] * 5,
            "test_samples": [len(X_test)] * 5
        }
        
        metadata_df = pd.DataFrame(model_metadata)
        csv_buffer = io.StringIO()
        metadata_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        
        st.download_button(
            label="📊 Metadata (CSV)",
            data=csv_buffer.getvalue(),
            file_name="model_metadata.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with col3:
        # Export predictions as CSV
        if 'predictions_df' in st.session_state:
            predictions_csv = io.StringIO()
            st.session_state.predictions_df.to_csv(predictions_csv, index=False, encoding='utf-8')
            
            st.download_button(
                label="🎯 Predictions (CSV)",
                data=predictions_csv.getvalue(),
                file_name="predictions_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    # Additional export options
    st.markdown("#### 📁 Export bổ sung")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export vocabulary if available
        try:
            if hasattr(pipe.named_steps["vec"], "vocabulary_"):
                vocab = pipe.named_steps["vec"].vocabulary_
                vocab_df = pd.DataFrame([
                    {"word": word, "index": idx} 
                    for word, idx in vocab.items()
                ]).sort_values("index")
                
                vocab_csv = io.StringIO()
                vocab_df.to_csv(vocab_csv, index=False, encoding='utf-8')
                
                st.download_button(
                    label="📚 Vocabulary (CSV)",
                    data=vocab_csv.getvalue(),
                    file_name="vocabulary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception:
            st.info("Vocabulary không khả dụng cho mô hình này")
    
    with col2:
        # Export feature importance if available
        try:
            if hasattr(pipe.named_steps["clf"], "feature_importances_"):
                importances = pipe.named_steps["clf"].feature_importances_
                
                # Get feature names
                if hasattr(pipe.named_steps["vec"], "get_feature_names_out"):
                    feature_names = pipe.named_steps["vec"].get_feature_names_out()
                else:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                # Handle feature selection
                if "feature_selection" in pipe.named_steps:
                    selected_features = pipe.named_steps["feature_selection"].get_support()
                    feature_names = feature_names[selected_features]
                
                features_df = pd.DataFrame({
                    "feature": feature_names[:len(importances)],
                    "importance": importances
                }).sort_values("importance", ascending=False)
                
                features_csv = io.StringIO()
                features_df.to_csv(features_csv, index=False, encoding='utf-8')
                
                st.download_button(
                    label="⚡ Features (CSV)",
                    data=features_csv.getvalue(),
                    file_name="feature_importance.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Feature importance không khả dụng cho mô hình này")
        except Exception:
            st.info("Feature importance không khả dụng")

# ============ Predict ============
if do_predict:
    if st.session_state.trained_model is None:
        st.warning("Bạn chưa train mô hình. Hãy bấm 'Train & Evaluate' trước khi dự đoán.")
    else:
        model = st.session_state.trained_model
        if predict_text.strip() == "":
            st.info("Nhập nội dung cần phân loại ở khung trên.")
        else:
            pred = model.predict([predict_text])[0]
            proba_text = ""
            confidence = 0
            
            if hasattr(model.named_steps["clf"], "predict_proba"):
                try:
                    proba = model.named_steps["clf"].predict_proba(model.named_steps["vec"].transform(
                        model.named_steps["prep"].transform([predict_text])
                    ))
                    classes = model.named_steps["clf"].classes_
                    pred_idx = list(classes).index(pred)
                    confidence = proba[0, pred_idx]
                    proba_text = f" (Độ tin cậy: {confidence:.2f})"
                except Exception:
                    proba_text = ""
            
            # Beautiful prediction result
            if pred.lower() == 'spam':
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 2rem; border-radius: 20px; text-align: center; color: white; margin: 1rem 0;">
                        <h3 style="margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🚨 SPAM</h3>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{proba_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 2rem; border-radius: 20px; text-align: center; color: white; margin: 1rem 0;">
                        <h3 style="margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">✅ HAM (Tin nhắn bình thường)</h3>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{proba_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ============ So sánh hiệu quả các phương pháp ============
if "comparison_results" in st.session_state and len(st.session_state.comparison_results) > 0:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 20px; margin: 2rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: center;">📊 So sánh hiệu quả các phương pháp</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(st.session_state.comparison_results)
    
    # Display comparison table
    st.markdown("### 📋 Bảng so sánh kết quả")
    st.dataframe(comparison_df.round(3), use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = px.bar(comparison_df, x='method', y='accuracy', 
                    title='So sánh Accuracy theo phương pháp',
                    labels={'accuracy': 'Accuracy', 'method': 'Phương pháp'},
                    color='accuracy', color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # F1-score comparison
        fig = px.bar(comparison_df, x='method', y='f1', 
                    title='So sánh F1-score theo phương pháp',
                    labels={'f1': 'F1-score', 'method': 'Phương pháp'},
                    color='f1', color_continuous_scale='Plasma')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Training time comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(comparison_df, x='method', y='train_time', 
                    title='So sánh thời gian training',
                    labels={'train_time': 'Thời gian (s)', 'method': 'Phương pháp'},
                    color='train_time', color_continuous_scale='Blues')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Multi-metric comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=comparison_df['method'],
                y=comparison_df[metric],
                mode='lines+markers',
                name=metric.capitalize(),
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='So sánh tất cả metrics',
            xaxis_title='Phương pháp',
            yaxis_title='Score',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Best method recommendation
    best_accuracy_idx = comparison_df['accuracy'].idxmax()
    best_f1_idx = comparison_df['f1'].idxmax()
    fastest_idx = comparison_df['train_time'].idxmin()
    
    st.markdown("### 🏆 Kết quả tốt nhất")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 15px; text-align: center; color: white;">
                <h4 style="margin: 0;">🎯 Accuracy cao nhất</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">
                    {comparison_df.iloc[best_accuracy_idx]['method']}<br>
                    {comparison_df.iloc[best_accuracy_idx]['accuracy']:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; text-align: center; color: white;">
                <h4 style="margin: 0;">⚡ F1-score cao nhất</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">
                    {comparison_df.iloc[best_f1_idx]['method']}<br>
                    {comparison_df.iloc[best_f1_idx]['f1']:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 15px; text-align: center; color: white;">
                <h4 style="margin: 0;">🚀 Nhanh nhất</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">
                    {comparison_df.iloc[fastest_idx]['method']}<br>
                    {comparison_df.iloc[fastest_idx]['train_time']:.2f}s
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Clear comparison button
    if st.button("🗑️ Xóa kết quả so sánh", use_container_width=True):
        st.session_state.comparison_results = []
        st.rerun()

# ============ Footer ============
st.markdown("---")

# Footer with app info
st.markdown(
    """
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; margin-top: 2rem;">
        <h4 style="margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">📩 Phân loại tin nhắn rác - Machine Learning GUI</h4>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Được xây dựng với Streamlit & Scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============ Notes ============
with st.expander("ℹ️ Gợi ý & mở rộng", expanded=False):
    st.markdown("""
**📋 Quy trình NLP đã triển khai:**
1. **Khám phá dữ liệu:** Cấu trúc, thống kê, phân phối nhãn, độ dài văn bản, tần suất từ
2. **Tiền xử lý:** Lowercase, loại bỏ ký tự đặc biệt, tách từ tiếng Việt, xử lý teencode
3. **Vector hóa:** BoW, TF-IDF, Sentence Embeddings (pre-trained models)
4. **Huấn luyện:** Nhiều thuật toán ML với hyperparameter tuning
5. **Đánh giá:** Metrics đa dạng, visualization, so sánh phương pháp

**💡 Gợi ý cải thiện:**
- **Sentence Embeddings:** Sử dụng mô hình multilingual cho tiếng Việt tốt hơn
- **Tách từ chuyên nghiệp:** Thư viện *underthesea* hoặc *VnCoreNLP*
- **Feature Engineering:** Thử n-gram 1-2, feature selection với Chi-square
- **Mô hình:** So sánh BoW vs TF-IDF vs Sentence Embeddings
- **Cân bằng dữ liệu:** SMOTE hoặc undersampling nếu mất cân bằng lớp
""")


