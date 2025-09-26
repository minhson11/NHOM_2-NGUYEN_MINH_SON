import re
import string
from sklearn.base import BaseEstimator, TransformerMixin

# Stopwords, teencode, patterns (copy từ BaiTapLon.py nếu cần)
VN_STOPWORDS = {
    "là","và","của","có","cho","nhé","nhỉ","thì","lại","này","kia","ấy","đã","đang","sẽ",
    "rồi","với","được","khi","đi","đến","từ","trong","ra","vào","cũng","nhưng","nếu","vì","do",
    "trên","dưới","nữa","nên","thôi","ơi","ơi!","à","ạ","ạ!","hả","hà","ha","ư","ừ","ờ",
    "la","va","cua","co","cho","nhe","nhi","thi","lai","nay","kia","ay","da","dang","se",
    "roi","voi","duoc","khi","di","den","tu","trong","ra","vao","cung","nhung","neu","vi","do",
    "tren","duoi","nua","nen","thoi","oi","oi!","a","ha","u","u~","o","ok","okie","okay",
}
TEENCODE_MAP = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "khong": "không",
    "kg": "không", "k0": "không", "0": "không",
    "dc": "được", "đc": "được", "dcf": "được", "duoc": "được",
    "kdc": "không được", "kdcf": "không được",
    "mik": "mình", "mk": "mình", "m": "mình", "minh": "mình",
    "bn": "bạn", "ban": "bạn", "b": "bạn",
    "t": "tôi", "toi": "tôi", "tao": "tôi",
    "n": "nó", "no": "nó", "nó": "nó",
    "ae": "anh em", "ad": "admin", "ib": "inbox",
    "sdt": "số điện thoại", "lh": "liên hệ", "lienhe": "liên hệ",
    "vs": "với", "voi": "với", "w": "với",
    "j": "gì", "ji": "gì", "gi": "gì",
    "sao": "sao", "s": "sao",
    "nao": "nào", "nao": "nào",
    "vl": "rất", "vcl": "rất", "rat": "rất",
    "oi": "ôi", "ui": "ui",
    "hum": "hôm", "hom": "hôm", "hnay": "hôm nay",
    "ngay": "ngày", "ng": "ngày",
    "ok": "ok", "oke": "ok", "okie": "ok",
    "thik": "thích", "thich": "thích",
    "iu": "yêu", "yeu": "yêu",
    "xau": "xấu", "xau": "xấu",
    "dep": "đẹp", "dep": "đẹp",
}
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\- ]{7,}\d)\b")
MONEY_PATTERN = re.compile(r"\b\d+[.,]?\d*\s*(?:k|đ|vnd|vnđ|usd|$)\b", flags=re.IGNORECASE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize_teencode(text: str) -> str:
    tokens = text.split()
    out = []
    for t in tokens:
        key = t.lower().strip()
        punct_end = ""
        while key and key[-1] in string.punctuation:
            punct_end = key[-1] + punct_end
            key = key[:-1]
        normalized = TEENCODE_MAP.get(key, t)
        out.append(normalized + punct_end)
    return " ".join(out)

def simple_vietnamese_tokenize(text: str) -> str:
    text = re.sub(r'([.!?;:,])', r' \1 ', text)
    text = re.sub(r'([()\[\]{}])', r' \1 ', text)
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
    if use_tokenize:
        text = simple_vietnamese_tokenize(text)
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
    if rm_punct:
        text = text.translate(PUNCT_TABLE)
    if rm_numbers:
        text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if rm_stopwords:
        tokens = [t for t in text.split() if t not in VN_STOPWORDS]
        text = " ".join(tokens)
    return text

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase=True, rm_punct=True, rm_numbers=False,
        rm_emoji=True, use_teencode=True, rm_stopwords=True,
        keep_placeholders=True, use_tokenize=True):
        self.lowercase = lowercase
        self.rm_punct = rm_punct
        self.rm_numbers = rm_numbers
        self.rm_emoji = rm_emoji
        self.use_teencode = use_teencode
        self.rm_stopwords = rm_stopwords
        self.keep_placeholders = keep_placeholders
        self.use_tokenize = use_tokenize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [basic_preprocess(x,
             lowercase=self.lowercase,
             rm_punct=self.rm_punct,
             rm_numbers=self.rm_numbers,
             rm_emoji=self.rm_emoji,
             use_teencode=self.use_teencode,
             rm_stopwords=self.rm_stopwords,
             keep_placeholders=self.keep_placeholders,
             use_tokenize=self.use_tokenize) for x in X]