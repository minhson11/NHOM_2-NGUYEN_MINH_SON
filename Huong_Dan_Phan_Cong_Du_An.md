# 📋 HƯỚNG DẪN PHÂN CÔNG DỰ ÁN - PHÂN LOẠI TIN NHẮN RÁC

## 📊 TỔNG QUAN DỰ ÁN

**Tên dự án:** Hệ thống phân loại tin nhắn rác sử dụng Machine Learning  
**Công nghệ:** Python, Streamlit, FastAPI, Scikit-learn  
**Mục tiêu:** Xây dựng hệ thống hoàn chỉnh từ tiền xử lý đến triển khai API  

---

# 👤 PHẦN 1: TIỀN XỬ LÝ VĂN BẢN (Đoàn Duy Mạnh)

## 🎯 NHIỆM VỤ CHÍNH
- Phát triển module `text_preprocessor.py`
- Xây dựng pipeline tiền xử lý văn bản đa ngôn ngữ (Anh + Việt)
- Tạo bộ từ điển stopwords và teencode cho cả tiếng Anh và tiếng Việt
- Viết unit tests

## 📁 FILE CHÍNH: `text_preprocessor.py`

### 1. IMPORT VÀ THIẾT LẬP CƠ BẢN

```python
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin
```

**Giải thích:**
- `re`: Thư viện regex để xử lý pattern matching
- `string`: Cung cấp các hằng số như punctuation
- `BaseEstimator, TransformerMixin`: Để tạo custom transformer tương thích với sklearn

### 2. BỘ TỪ ĐIỂN STOPWORDS ĐA NGÔN NGỮ

```python
# Stopwords tiếng Anh (sử dụng NLTK hoặc tự định nghĩa)
EN_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "through", "during", "before", "after", "above", 
    "below", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
    "further", "then", "once"
}

# Stopwords tiếng Việt (nếu cần xử lý dữ liệu tiếng Việt)
VN_STOPWORDS = {
    "là","và","của","có","cho","nhé","nhỉ","thì","lại","này","kia","ấy",
    "đã","đang","sẽ","rồi","với","được","khi","đi","đến","từ","trong",
    "ra","vào","cũng","nhưng","nếu","vì","do","trên","dưới","nữa","nên"
}

# Kết hợp stopwords theo ngôn ngữ
def get_stopwords(language="auto"):
    if language == "en":
        return EN_STOPWORDS
    elif language == "vi":
        return VN_STOPWORDS
    elif language == "auto":
        return EN_STOPWORDS.union(VN_STOPWORDS)  # Kết hợp cả 2
    else:
        return set()
```

**Giải thích:**
- **EN_STOPWORDS**: Từ dừng tiếng Anh phổ biến (articles, pronouns, prepositions)
- **VN_STOPWORDS**: Từ dừng tiếng Việt (chỉ cần nếu dữ liệu có tiếng Việt)
- **Auto-detection**: Tự động nhận diện ngôn ngữ hoặc xử lý cả 2

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Tích hợp NLTK**: `from nltk.corpus import stopwords` cho tiếng Anh chuẩn
2. **Language detection**: Thêm `langdetect` để tự động nhận diện ngôn ngữ
3. **Flexible stopwords**: Cho phép người dùng chọn ngôn ngữ cụ thể

### 3. BỘ TỪ ĐIỂN CHUẨN HÓA VĂN BẢN

```python
# Internet slang và abbreviations tiếng Anh (cho SMS/chat data)
EN_SLANG_MAP = {
    "u": "you", "ur": "your", "r": "are", "n": "and", "2": "to", "4": "for",
    "b4": "before", "c": "see", "omg": "oh my god", "lol": "laugh out loud", 
    "rofl": "rolling on floor laughing", "brb": "be right back", "ttyl": "talk to you later",
    "w8": "wait", "gr8": "great", "m8": "mate", "l8r": "later", "2morrow": "tomorrow",
    "2day": "today", "2nite": "tonight", "pls": "please", "plz": "please", 
    "thx": "thanks", "tx": "thanks", "tnx": "thanks", "msg": "message",
    "txt": "text", "pic": "picture", "vid": "video", "luv": "love", "wanna": "want to",
    "gonna": "going to", "dunno": "don't know", "kinda": "kind of", "sorta": "sort of"
}

# Teencode tiếng Việt (chỉ dùng khi có dữ liệu tiếng Việt)
VN_TEENCODE_MAP = {
    "ko": "không", "k": "không", "kh": "không",
    "dc": "được", "đc": "được", 
    "mik": "mình", "mk": "mình",
    "bn": "bạn", "ban": "bạn",
    "j": "gì", "ji": "gì", "gi": "gì",
    "vs": "với", "voi": "với", "w": "với"
}

def normalize_slang(text: str, language="en") -> str:
    """Chuẩn hóa slang/teencode theo ngôn ngữ"""
    if language == "en":
        slang_map = EN_SLANG_MAP
    elif language == "vi":
        slang_map = VN_TEENCODE_MAP
    else:
        # Auto mode: thử cả 2
        slang_map = {**EN_SLANG_MAP, **VN_TEENCODE_MAP}
    
    # Normalization logic here...
    return text
```

**Giải thích:**
- **EN_SLANG_MAP**: Internet slang tiếng Anh phổ biến trong SMS/chat
- **VN_TEENCODE_MAP**: Teencode tiếng Việt (chỉ dùng khi cần)
- **Language-aware**: Chọn dictionary phù hợp với ngôn ngữ dữ liệu

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Mở rộng EN_SLANG**: Thêm 200+ abbreviations tiếng Anh
2. **Conditional loading**: Chỉ load VN dictionary khi cần
3. **Auto-detection**: Tự động chọn dictionary dựa trên ngôn ngữ text

### 4. REGEX PATTERNS

```python
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\- ]{7,}\d)\b")
MONEY_PATTERN = re.compile(r"\b\d+[.,]?\d*\s*(?:k|đ|vnd|vnđ|usd|$)\b", flags=re.IGNORECASE)
```

**Giải thích từng pattern:**

- **EMOJI_PATTERN**: Nhận diện emoji Unicode
  - `[\U00010000-\U0010ffff]`: Khoảng Unicode chứa emoji
  - `re.UNICODE`: Hỗ trợ Unicode đầy đủ

- **URL_PATTERN**: Nhận diện URL
  - `https?://`: Bắt đầu với http hoặc https
  - `\S+`: Một hoặc nhiều ký tự không phải space
  - `www\.\S+`: Hoặc bắt đầu bằng www.

- **EMAIL_PATTERN**: Nhận diện email
  - `[\w\.-]+`: Một hoặc nhiều chữ cái, số, dấu chấm, gạch ngang
  - `@`: Ký tự @
  - `[\w\.-]+`: Domain name

- **PHONE_PATTERN**: Nhận diện số điện thoại
  - `\b`: Word boundary
  - `\+?`: Dấu + tùy chọn (cho số quốc tế)
  - `\d[\d\- ]{7,}\d`: Số đầu, 7+ ký tự số/dấu/space, số cuối

- **MONEY_PATTERN**: Nhận diện số tiền
  - `\d+[.,]?\d*`: Số với dấu phẩy/chấm tùy chọn
  - `(?:k|đ|vnd|vnđ|usd|$)`: Đơn vị tiền tệ

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Tối ưu regex:** Cải thiện độ chính xác
2. **Thêm patterns:** Hashtag, mention (@user), số thẻ tín dụng
3. **Test regex:** Viết unit tests cho từng pattern

### 5. HÀM CHUẨN HÓA TEENCODE

```python
def normalize_teencode(text: str) -> str:
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
```

**Giải thích từng bước:**
1. **`text.split()`**: Tách thành các từ
2. **`t.lower().strip()`**: Chuyển thường và loại space
3. **Xử lý dấu câu**: Tách dấu câu ra khỏi từ để mapping chính xác
4. **`TEENCODE_MAP.get(key, t)`**: Tìm trong dictionary, nếu không có thì giữ nguyên
5. **Ghép lại**: Từ đã chuẩn hóa + dấu câu ban đầu

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Xử lý case phức tạp:** Từ có nhiều dấu câu
2. **Optimize performance:** Sử dụng comprehension hoặc vectorization
3. **Handle edge cases:** Empty string, special characters

### 6. HÀM TOKENIZATION ĐA NGÔN NGỮ

```python
def simple_tokenize(text: str, language="en") -> str:
    """Tokenization cơ bản cho cả tiếng Anh và tiếng Việt"""
    
    # Xử lý contractions tiếng Anh
    if language in ["en", "auto"]:
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "'s": " is"  # hoặc "'s": " has" tùy context
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
    
    # Thêm khoảng trắng trước/sau dấu câu
    text = re.sub(r'([.!?;:,])', r' \1 ', text)
    # Thêm khoảng trắng trước/sau dấu ngoặc
    text = re.sub(r'([()\[\]{}])', r' \1 ', text)
    # Xử lý dấu ngoặc kép
    text = re.sub(r'(["\'])', r' \1 ', text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def advanced_tokenize(text: str, language="en") -> str:
    """Tokenization nâng cao với NLTK hoặc spaCy"""
    try:
        if language == "en":
            import nltk
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text.lower())
            return " ".join(tokens)
        elif language == "vi":
            # Có thể dùng underthesea cho tiếng Việt
            # from underthesea import word_tokenize
            # tokens = word_tokenize(text)
            return simple_tokenize(text, "vi")
        else:
            return simple_tokenize(text, "auto")
    except ImportError:
        # Fallback về simple tokenization
        return simple_tokenize(text, language)
```

**Giải thích cải tiến:**
- **Contractions handling**: Xử lý "don't" → "do not" cho tiếng Anh
- **Language-specific**: Tokenization khác nhau cho từng ngôn ngữ
- **Advanced options**: Tích hợp NLTK/spaCy khi có thể
- **Fallback mechanism**: Dùng simple tokenization khi thư viện nâng cao không có

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **NLTK integration**: Tích hợp `nltk.tokenize` cho tiếng Anh
2. **Language detection**: Auto-detect ngôn ngữ để chọn tokenizer
3. **Performance**: Benchmark các phương pháp tokenization khác nhau

### 7. HÀM TIỀN XỬ LÝ CHÍNH

```python
def basic_preprocess(text: str,
        lowercase: bool = True,
        rm_punct: bool = True,
        rm_numbers: bool = False,
        rm_emoji: bool = True,
        use_slang_normalization: bool = True,
        rm_stopwords: bool = True,
        keep_placeholders: bool = True,
        use_tokenize: bool = True,
        language: str = "en") -> str:
```

**Giải thích các tham số:**
- **lowercase**: Chuyển về chữ thường
- **rm_punct**: Loại bỏ dấu câu
- **rm_numbers**: Loại bỏ số
- **rm_emoji**: Loại bỏ emoji
- **use_slang_normalization**: Áp dụng chuẩn hóa slang/teencode theo ngôn ngữ
- **rm_stopwords**: Loại bỏ stopwords theo ngôn ngữ
- **keep_placeholders**: Giữ <URL>, <EMAIL> thay vì xóa hoàn toàn
- **use_tokenize**: Áp dụng tokenization phù hợp với ngôn ngữ
- **language**: Ngôn ngữ dữ liệu ("en", "vi", "auto")

**Pipeline xử lý (language-aware):**

```python
# 1. Tokenization theo ngôn ngữ
if use_tokenize:
    text = simple_tokenize(text, language)

# 2. Xử lý placeholders
if keep_placeholders:
    text = URL_PATTERN.sub(" <URL> ", text)
    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    text = PHONE_PATTERN.sub(" <PHONE> ", text)
    text = MONEY_PATTERN.sub(" <MONEY> ", text)

# 3. Loại bỏ emoji
if rm_emoji:
    text = EMOJI_PATTERN.sub(" ", text)

# 4. Chuyển thường
if lowercase:
    text = text.lower()

# 5. Chuẩn hóa slang/teencode theo ngôn ngữ
if use_slang_normalization:
    text = normalize_slang(text, language)

# 6. Loại bỏ dấu câu
if rm_punct:
    text = text.translate(PUNCT_TABLE)

# 7. Loại bỏ số
if rm_numbers:
    text = re.sub(r"\d+", " ", text)

# 8. Loại bỏ stopwords theo ngôn ngữ
if rm_stopwords:
    stopwords = get_stopwords(language)
    tokens = [t for t in text.split() if t not in stopwords]
    text = " ".join(tokens)

# 9. Làm sạch khoảng trắng cuối cùng
text = re.sub(r"\s+", " ", text).strip()
return text
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Language detection**: Implement auto-detection với `langdetect` library
2. **Stemming/Lemmatization**: Thêm Porter Stemmer cho tiếng Anh
3. **Performance**: Optimize cho datasets lớn với multiprocessing
4. **Flexibility**: Cho phép user define custom stopwords và slang dictionaries

### 8. CLASS TEXTPREPROCESSOR

```python
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase=True, rm_punct=True, ...):
        self.lowercase = lowercase
        self.rm_punct = rm_punct
        # ... các tham số khác

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [basic_preprocess(x, 
                 lowercase=self.lowercase,
                 rm_punct=self.rm_punct,
                 # ... các tham số khác
                 ) for x in X]
```

**Giải thích thiết kế:**
- **BaseEstimator**: Cung cấp get_params(), set_params()
- **TransformerMixin**: Cung cấp fit_transform()
- **fit()**: Không cần học gì từ dữ liệu, chỉ return self
- **transform()**: Áp dụng preprocessing cho mỗi text trong list

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Thêm validation:** Kiểm tra tham số hợp lệ
2. **Optimize**: Sử dụng multiprocessing cho dataset lớn
3. **Add caching**: Cache kết quả cho text giống nhau

## 📝 UNIT TESTS CẦN VIẾT

### 1. Test Regex Patterns
```python
def test_url_pattern():
    text = "Visit https://google.com or www.facebook.com"
    result = URL_PATTERN.findall(text)
    assert len(result) == 2

def test_phone_pattern():
    text = "Call 0123456789 or +84123456789"
    result = PHONE_PATTERN.findall(text)
    assert len(result) == 2
```

### 2. Test Teencode Normalization
```python
def test_teencode_normalization():
    text = "ko có j, mik đi r"
    result = normalize_teencode(text)
    expected = "không có gì, mình đi rồi"
    assert result == expected
```

### 3. Test Full Pipeline
```python
def test_basic_preprocess():
    text = "Ko có j!!! Visit www.google.com 😀"
    result = basic_preprocess(text)
    # Test kết quả mong đợi
```

## 📋 CHECKLIST HOÀN THÀNH

- [ ] **Language Support**: Hỗ trợ cả tiếng Anh và tiếng Việt
- [ ] **English Stopwords**: Tích hợp NLTK stopwords cho tiếng Anh (400+ từ)
- [ ] **English Slang**: Mở rộng EN_SLANG_MAP (200+ abbreviations)
- [ ] **Regex Patterns**: Tối ưu cho URLs, emails, phones (international formats)
- [ ] **Language Detection**: Auto-detect với `langdetect` library
- [ ] **Advanced Tokenization**: NLTK integration cho tiếng Anh
- [ ] **Stemming/Lemmatization**: Porter Stemmer cho English texts
- [ ] **Performance**: Multiprocessing cho large datasets
- [ ] **Unit Tests**: Comprehensive tests cho cả EN/VI
- [ ] **Flexibility**: Custom dictionaries support
- [ ] **Documentation**: Detailed code comments và usage examples

### 🌟 **BONUS TASKS (Nếu có thời gian)**
- [ ] **spaCy Integration**: Advanced NLP preprocessing
- [ ] **Custom Models**: Train custom tokenizer cho domain-specific text
- [ ] **Caching**: Cache processed results
- [ ] **Config Files**: JSON/YAML config cho easy customization

---

# 🖥️ PHẦN 2: STREAMLIT GUI & MACHINE LEARNING (Nguyễn Huy Hoàng)

## 🎯 NHIỆM VỤ CHÍNH
- Phát triển giao diện Streamlit hoàn chỉnh
- Tích hợp nhiều thuật toán Machine Learning
- Xây dựng pipeline huấn luyện và đánh giá
- Tạo visualization và EDA
- Triển khai tính năng xuất mô hình

## 📁 FILE CHÍNH: `BaiTapLon.py`

### 1. THIẾT LẬP STREAMLIT VÀ CSS

```python
import streamlit as st
st.set_page_config(page_title="Phân loại tin nhắn rác", page_icon="📩", layout="wide")
```

**Giải thích:**
- **page_title**: Tiêu đề tab trình duyệt
- **page_icon**: Icon hiển thị trên tab
- **layout="wide"**: Sử dụng toàn bộ chiều rộng màn hình

### 2. CSS STYLING

```python
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
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
</style>
""", unsafe_allow_html=True)
```

**Giải thích CSS:**
- **Google Fonts**: Import font Inter cho UI hiện đại
- **Gradient backgrounds**: Tạo hiệu ứng màu đẹp
- **Box shadows**: Tạo độ sâu 3D
- **Transitions**: Hiệu ứng hover mượt mà

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Tùy chỉnh theme**: Tạo color scheme nhất quán
2. **Responsive design**: Đảm bảo hiển thị tốt trên mobile
3. **Dark mode**: Thêm toggle chuyển đổi theme

### 3. SIDEBAR CẤU HÌNH

```python
st.sidebar.title("⚙️ Cấu hình thí nghiệm")

# 1. Nguồn dữ liệu
data_source = st.sidebar.radio("Chọn nguồn dữ liệu", 
    ["📁 Upload CSV", "📊 Dữ liệu mẫu", "🌐 Kaggle Dataset"], 
    index=1)

# 2. Tiền xử lý
lowercase = st.sidebar.checkbox("Lowercase", value=True)
use_tokenize = st.sidebar.checkbox("Tách từ tiếng Việt", value=True)
rm_punct = st.sidebar.checkbox("Bỏ dấu câu", value=True)
```

**Giải thích components:**
- **radio**: Chọn một trong nhiều options
- **checkbox**: Toggle boolean options
- **slider**: Chọn giá trị trong khoảng
- **selectbox**: Dropdown menu

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Thêm tooltips**: Giải thích cho từng option
2. **Grouping**: Nhóm các cấu hình liên quan
3. **Validation**: Kiểm tra tính hợp lệ của input

### 4. XỬ LÝ UPLOAD FILE

```python
if data_source == "📁 Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, encoding="latin-1")
```

**Giải thích xử lý:**
1. **file_uploader**: Widget upload file
2. **Multiple encodings**: Thử UTF-8 trước, fallback về latin-1
3. **seek(0)**: Reset file pointer về đầu

**Auto-detection columns:**
```python
# Nhận diện cột text/label
lower_map = {c.lower(): c for c in df_raw.columns}
label_candidates = ["label", "category", "class", "target", "v1"]
text_candidates = ["text", "message", "sms", "content", "body", "v2"]
label_col = next((lower_map[k] for k in label_candidates if k in lower_map), None)
text_col = next((lower_map[k] for k in text_candidates if k in lower_map), None)
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Mở rộng detection**: Thêm nhiều tên cột phổ biến
2. **Preview data**: Hiển thị sample data trước khi xử lý
3. **Data validation**: Kiểm tra format, missing values

### 5. EXPLORATORY DATA ANALYSIS (EDA)

```python
# Data stats cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f'<div class="metric-card"><div class="subtle">Tổng số dòng</div><div class="kpi">{len(df)}</div></div>',
        unsafe_allow_html=True
    )
```

**Visualization với Plotly:**
```python
# Pie chart phân phối nhãn
label_counts = df['label'].value_counts()
fig = px.pie(values=label_counts.values, names=label_counts.index, 
            title="Phân phối nhãn")
st.plotly_chart(fig, use_container_width=True)

# Histogram độ dài văn bản
text_lengths = df['text'].str.len()
fig = px.histogram(x=text_lengths, nbins=30, title="Phân phối độ dài văn bản")
st.plotly_chart(fig, use_container_width=True)
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Word clouds**: Tạo word cloud cho spam vs ham
2. **Statistical tests**: Chi-square test cho independence
3. **Advanced plots**: Violin plots, heatmaps
4. **Interactive filters**: Filter data theo criteria

### 6. MACHINE LEARNING PIPELINE

```python
# Vectorizers
if feat_type == "BoW":
    vectorizer = CountVectorizer(ngram_range=(1, ngram_max), min_df=min_df, max_df=max_df/100.0)
elif feat_type == "TF-IDF":
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, max_df=max_df/100.0)

# Models
if model_name == "Naive Bayes (Multinomial)":
    clf = MultinomialNB(alpha=alpha)
elif model_name == "Logistic Regression":
    clf = LogisticRegression(C=C, max_iter=max_iter)
elif model_name == "Linear SVM":
    clf = LinearSVC(C=C)
```

**Pipeline construction:**
```python
pipe = Pipeline([
    ("prep", TextPreprocessor(lowercase=lowercase, rm_punct=rm_punct, ...)),
    ("vec", vectorizer),
    ("clf", clf)
])
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Thêm models**: XGBoost, LightGBM, Neural Networks
2. **Hyperparameter tuning**: GridSearchCV, RandomizedSearchCV
3. **Cross-validation**: K-fold với stratification
4. **Feature selection**: Chi2, mutual info, ANOVA

### 7. TRAINING VÀ EVALUATION

```python
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Training
start = time.time()
pipe.fit(X_train, y_train)
train_time = time.time() - start

# Prediction và metrics
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
```

**Confusion Matrix visualization:**
```python
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
st.pyplot(fig)
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **ROC Curves**: Plot ROC cho multiple models
2. **Learning curves**: Validation curves
3. **Error analysis**: Analyze false positives/negatives
4. **Model comparison**: Side-by-side comparison table

### 8. MODEL EXPORT VÀ DEPLOYMENT

```python
# Export PKL model
buffer = io.BytesIO()
joblib.dump(pipe, buffer)
buffer.seek(0)
st.download_button(
    label="📥 Model (PKL)",
    data=buffer,
    file_name="spam_classifier_pipeline.pkl",
    mime="application/octet-stream"
)

# Export metadata CSV
model_metadata = {
    "metric": ["accuracy", "precision", "recall", "f1_score"],
    "value": [acc, pr, rc, f1],
    "model_type": [model_name] * 4
}
metadata_df = pd.DataFrame(model_metadata)
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **ONNX export**: Convert model to ONNX format
2. **Model versioning**: Version control cho models
3. **A/B testing**: Compare model versions
4. **Performance monitoring**: Track model drift

### 9. REAL-TIME PREDICTION

```python
predict_text = st.text_area("Nhập tin nhắn muốn kiểm tra", height=100)
if st.button("🔍 Phân loại"):
    if st.session_state.trained_model is not None:
        pred = st.session_state.trained_model.predict([predict_text])[0]
        
        # Display prediction với style đẹp
        if pred.lower() == 'spam':
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                        padding: 2rem; border-radius: 20px; text-align: center; color: white;">
                <h3>🚨 SPAM</h3>
            </div>
            """, unsafe_allow_html=True)
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Batch prediction**: Upload file để predict nhiều tin nhắn
2. **Confidence scores**: Hiển thị độ tin cậy
3. **Explanation**: LIME/SHAP để giải thích prediction
4. **History**: Lưu lại prediction history

## 📋 CHECKLIST HOÀN THÀNH

- [ ] Giao diện Streamlit đẹp với CSS custom
- [ ] EDA section hoàn chỉnh với charts
- [ ] 7+ thuật toán ML với hyperparameter tuning
- [ ] Train/validation/test pipeline
- [ ] Confusion matrix và classification report
- [ ] Model comparison dashboard
- [ ] Export functionality (PKL, CSV, metadata)
- [ ] Real-time prediction interface
- [ ] Error handling và user experience
- [ ] Documentation và help tooltips

---

# 🔗 PHẦN 3: REST API & DEPLOYMENT (Nguyễn Minh Sơn)

## 🎯 NHIỆM VỤ CHÍNH
- Phát triển FastAPI REST API
- Tích hợp model loading và prediction
- Triển khai logging và error handling
- Tạo Docker container
- Viết documentation và deployment guides

## 📁 FILE CHÍNH: `api.py`

### 1. THIẾT LẬP FASTAPI

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Spam Classifier API",
    description="API phân loại tin nhắn rác sử dụng Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
```

**Giải thích metadata:**
- **title**: Tên API hiển thị trong docs
- **description**: Mô tả chức năng
- **version**: Version API cho tracking
- **docs_url**: Swagger UI endpoint
- **redoc_url**: ReDoc documentation endpoint

### 2. CORS MIDDLEWARE

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Giải thích CORS:**
- **allow_origins**: Domain được phép truy cập API
- **allow_credentials**: Cho phép gửi cookies/auth headers
- **allow_methods**: HTTP methods được phép
- **allow_headers**: Headers được phép

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Security**: Restrict origins trong production
2. **Rate limiting**: Implement rate limiting middleware
3. **Authentication**: JWT authentication nếu cần

### 3. MODEL LOADING

```python
MODEL_PATH = "spam_classifier_pipeline.pkl"
model = None

def load_model():
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
```

**Giải thích pattern:**
- **Global variable**: Lưu model trong memory
- **Startup loading**: Tải model khi API khởi động
- **Error handling**: Graceful handling khi model không tải được
- **Logging**: Track loading status

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Model caching**: Redis cache cho multiple models
2. **Hot reload**: Reload model không restart server
3. **Model versioning**: Support multiple model versions
4. **Health monitoring**: Monitor model performance

### 4. PYDANTIC MODELS

```python
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
```

**Giải thích Pydantic:**
- **BaseModel**: Base class cho data validation
- **Field**: Define constraints và metadata
- **Optional**: Field có thể None
- **Type hints**: Python type annotations

**Validation benefits:**
- **Automatic validation**: FastAPI tự validate input
- **Error responses**: Trả về error messages chi tiết
- **Documentation**: Auto-generate API docs
- **IDE support**: Type hints cho autocomplete

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Advanced validation**: Custom validators
2. **Response models**: Standardize error responses
3. **Nested models**: Complex data structures
4. **Examples**: Add example values trong docs

### 5. HEALTH CHECK ENDPOINT

```python
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
```

**Giải thích decorator:**
- **@app.get**: HTTP GET endpoint
- **response_model**: Pydantic model cho response
- **tags**: Nhóm endpoints trong docs

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Extended health**: Check database, external services
2. **Metrics**: Response time, memory usage
3. **Dependencies**: Check all service dependencies
4. **Alerting**: Integration với monitoring systems

### 6. PREDICTION ENDPOINT

```python
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
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
        if hasattr(model.named_steps["clf"], "predict_proba"):
            probabilities = model.predict_proba([request.text])[0]
            classes = model.named_steps["clf"].classes_
            pred_idx = list(classes).index(prediction)
            confidence = float(probabilities[pred_idx])
        
        return PredictResponse(
            text=request.text,
            prediction=prediction,
            confidence=confidence,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình dự đoán: {str(e)}")
```

**Giải thích xử lý:**
1. **Model check**: Kiểm tra model đã load chưa
2. **Prediction**: Gọi model.predict()
3. **Confidence calculation**: Tính probability nếu có
4. **Error handling**: Catch và return HTTP errors

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Batch prediction**: Endpoint cho multiple texts
2. **Async processing**: Background tasks cho prediction lâu
3. **Caching**: Cache prediction results
4. **Input sanitization**: Clean và validate input text

### 7. LOGGING SYSTEM

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trong functions
logger.info(f"✅ Đã tải mô hình thành công từ {MODEL_PATH}")
logger.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
logger.warning(f"Không thể tính confidence: {str(e)}")
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Structured logging**: JSON format logs
2. **Log rotation**: Rotate log files
3. **Centralized logging**: ELK stack integration
4. **Performance logging**: Request/response times

### 8. ERROR HANDLING

```python
from fastapi import HTTPException

# Service unavailable
raise HTTPException(status_code=503, detail="Mô hình chưa được tải")

# Internal server error
raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

# Bad request
raise HTTPException(status_code=400, detail="Input không hợp lệ")
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Custom exception handlers**: Global error handling
2. **Error codes**: Standardized error codes
3. **User-friendly messages**: Vietnamese error messages
4. **Monitoring**: Error tracking và alerting

### 9. MODEL INFO ENDPOINT

```python
@app.get("/model-info", tags=["Info"])
def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được tải")
    
    try:
        info = {
            "model_type": type(model).__name__,
            "steps": list(model.named_steps.keys()) if hasattr(model, 'named_steps') else [],
            "classes": model.named_steps["clf"].classes_.tolist() if "clf" in model.named_steps else None
        }
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail="Không thể lấy thông tin mô hình")
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Extended info**: Feature names, model params
2. **Performance metrics**: Accuracy, training time
3. **Version info**: Model version, training date
4. **Data info**: Training data statistics

## 🐳 DOCKER DEPLOYMENT

### 1. DOCKERFILE

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Multi-stage build**: Optimize image size
2. **Security**: Non-root user, minimal dependencies
3. **Performance**: Gunicorn với multiple workers
4. **Monitoring**: Add monitoring tools

### 2. DOCKER-COMPOSE

```yaml
version: '3.8'

services:
  spam-classifier-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - spam-classifier-api
    restart: unless-stopped
```

**🔧 NHIỆM VỤ CỦA BẠN:**
1. **Load balancer**: Multiple API instances
2. **SSL termination**: HTTPS setup
3. **Monitoring**: Prometheus, Grafana
4. **Database**: If needed for logging/analytics

### 3. KUBERNETES DEPLOYMENT

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spam-classifier-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spam-classifier-api
  template:
    metadata:
      labels:
        app: spam-classifier-api
    spec:
      containers:
      - name: api
        image: spam-classifier-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## 📚 API DOCUMENTATION

### 1. README.md cho API

```markdown
# Spam Classifier API

## Endpoints

### POST /predict
Phân loại tin nhắn

**Request:**
```json
{
  "text": "Khuyến mãi 50% tất cả sản phẩm!"
}
```

**Response:**
```json
{
  "text": "Khuyến mãi 50% tất cả sản phẩm!",
  "prediction": "spam",
  "confidence": 0.95,
  "status": "success"
}
```

### GET /health
Kiểm tra trạng thái API

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "API đang hoạt động bình thường"
}
```
```

### 2. POSTMAN COLLECTION

```json
{
  "info": {
    "name": "Spam Classifier API",
    "description": "Collection for testing spam classifier API"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "{{base_url}}/health",
          "host": ["{{base_url}}"],
          "path": ["health"]
        }
      }
    },
    {
      "name": "Predict Spam",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"text\": \"Khuyến mãi 50% tất cả sản phẩm!\"\n}"
        },
        "url": {
          "raw": "{{base_url}}/predict",
          "host": ["{{base_url}}"],
          "path": ["predict"]
        }
      }
    }
  ]
}
```

## 📋 CHECKLIST HOÀN THÀNH

- [ ] FastAPI với 5+ endpoints hoàn chỉnh
- [ ] Pydantic models với validation
- [ ] Error handling và logging system
- [ ] Model loading và caching
- [ ] Health checks và monitoring
- [ ] CORS và security middleware
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] API documentation (Swagger + README)
- [ ] Postman collection
- [ ] Performance testing scripts
- [ ] Production deployment guide

---

# 🤝 TÍCH HỢP VÀ PHỐI HỢP

## 📅 TIMELINE TÍCH HỢP

### Tuần 1: Development song song
- **Người 1**: Hoàn thành `text_preprocessor.py`
- **Người 2**: Xây dựng Streamlit UI cơ bản
- **Người 3**: Setup FastAPI endpoints

### Tuần 2: Integration Points
- **Người 1 → 2**: Merge text preprocessor vào Streamlit
- **Người 2**: Hoàn thiện ML pipeline với preprocessing
- **Người 3**: Test API với sample models

### Tuần 3: Final Integration
- **Người 2 → 3**: Export trained models cho API
- **Người 3**: Tích hợp và test toàn bộ system
- **All**: Testing, documentation, deployment

## 🔗 INTEGRATION CHECKLIST

- [ ] `text_preprocessor.py` tương thích với cả Streamlit và API
- [ ] Model export từ Streamlit hoạt động với API
- [ ] Consistent input/output formats
- [ ] Error handling nhất quán
- [ ] Logging formats tương thích
- [ ] Documentation đầy đủ cho cả 3 phần

## 📞 COMMUNICATION PROTOCOL

1. **Daily standup**: 15 phút mỗi ngày
2. **Git workflow**: Feature branches + Pull Requests
3. **Code review**: Peer review trước khi merge
4. **Testing**: Unit tests + Integration tests
5. **Documentation**: Code comments + README updates

---

**🎯 CHÚC CÁC BẠN THỰC HIỆN THÀNH CÔNG!**
