"""
Digikala Comment Sentiment Analyzer — Streamlit App
-----------------------------------------------------
Run with:  streamlit run app.py

Requires:
    pip install streamlit scikit-learn joblib numpy requests
    (Ollama must be running locally with the target model pulled,
     e.g.  ollama pull gemma3:1b)

Model: sklearn Stacking model (TF-IDF + Logistic Regression -> HistGradientBoosting
       meta-model on [text probs, Score]), trained in sentiment_digikala_stacking.ipynb.
       Place sentiment_stacking.joblib next to this file.
"""

import re

import joblib
import numpy as np
import streamlit as st
import requests

# ----------------------------------------------------------------------------
# Configuration (defaults — thresholds/Ollama settings are still editable from
# the sidebar at runtime; the model itself is bundled with the app, not uploaded)
# ----------------------------------------------------------------------------
MODEL_PATH = "sentiment_stacking.joblib"  # place this file next to app.py

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"
THRESHOLD = 0.1              # if P(bad) exceeds this, ask Ollama for a second opinion
CONFIDENCE_THRESHOLD = 0.6   # minimum confidence to trust the model's own verdict
DEFAULT_SCORE = 60           # پیش‌فرض اسلایدر امتیاز، وقتی کاربر نمره‌ای نمی‌داند

# ترتیب این لیست باید دقیقاً با ترتیب کلاس‌های مدل (۰=منفی، ۱=خنثی، ۲=مثبت) یکی باشد
CLASS_NAMES = ["bad", "neutral", "good"]
CLASS_FA = {"bad": "منفی 😠", "neutral": "خنثی 😐", "good": "مثبت 😊"}
CLASS_COLOR = {"bad": "#e74c3c", "neutral": "#95a5a6", "good": "#2ecc71"}

# ----------------------------------------------------------------------------
# Page setup + styling (RTL, Persian-friendly)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="تحلیل احساسات دیجی‌کالا", page_icon="🛍️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Vazirmatn', sans-serif; direction: rtl; }
    .main .block-container { direction: rtl; text-align: right; }
    .stTextArea textarea { direction: rtl; text-align: right; font-family: 'Vazirmatn', sans-serif; }
    .verdict-card {
        padding: 1.2rem 1.5rem; border-radius: 16px; margin-top: 1rem;
        color: white; font-size: 1.3rem; font-weight: 800; text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .ollama-card {
        padding: 1rem 1.3rem; border-radius: 14px; margin-top: 1rem;
        background: #1e1e2f; color: #f5f5f5; border-right: 6px solid #6c5ce7;
    }
    .prob-row { display: flex; align-items: center; margin-bottom: 10px; }

    /* اسلایدرها: اعداد و دستگیره باید LTR بمانند وگرنه با استایل RTL کلی صفحه نامرتب می‌شوند */
    div[data-testid="stSlider"] { direction: ltr; text-align: left; }
    div[data-testid="stSlider"] label { direction: rtl; text-align: right; display: block; }
    div[data-testid="stThumbValue"] { font-weight: 700 !important; font-size: 0.95rem !important; }

    .slider-value {
        display: inline-block; margin-top: 4px; padding: 2px 12px;
        background: #ee0836; border-radius: 8px; font-weight: 700; font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;'>🛍️ تحلیل احساسات کامنت‌های دیجی‌کالا</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>مدل Stacking سه‌کلاسه (متن + امتیاز کاربر) + نظرخواهی از Ollama برای موارد مشکوک به نظر منفی</p>",
    unsafe_allow_html=True,
)
# خط جدا کننده
st.divider()

# ----------------------------------------------------------------------------
# Sidebar: settings only — the model ships with the app itself
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ تنظیمات")

    threshold = st.slider(
        "آستانه احتمال منفی برای ارجاع به Ollama", 0.0, 1.0, THRESHOLD, 0.01, format="%.2f"
    )
    st.markdown(f"<span class='slider-value'>مقدار فعلی: {threshold:.0%}</span>", unsafe_allow_html=True)

    confidence_threshold = st.slider(
        "آستانه اطمینان مدل اصلی", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.01, format="%.2f"
    )
    st.markdown(f"<span class='slider-value'>مقدار فعلی: {confidence_threshold:.0%}</span>", unsafe_allow_html=True)

    st.subheader("Ollama")
    ollama_url = st.text_input("Ollama URL", value=OLLAMA_URL)
    ollama_model = st.text_input("نام مدل Ollama", value=MODEL_NAME)
    use_ollama = st.checkbox("فعال بودن نظر دوم Ollama", value=True)

# نتیجه این تابع را ذخیره کن تا مجبور نباشم هر بار دوباره مدل را از دیسک بارگذاری کنم
@st.cache_resource(show_spinner="در حال بارگذاری مدل...")
def _load_model():
    return joblib.load(MODEL_PATH)


try:
    model = _load_model()
except (OSError, IOError, FileNotFoundError) as e:
    st.error(
        f"❌ فایل مدل پیدا نشد.\n\n"
        f"مطمئن شو فایل `{MODEL_PATH}` کنار همین `app.py` قرار داره.\n\n"
        f"جزئیات خطا: {e}"
    )
    st.stop()

st.success("مدل با موفقیت بارگذاری شد ✅")


# ----------------------------------------------------------------------------
# Text/Score preprocessing — باید دقیقاً با نوت‌بوک آموزش یکی باشد
# ----------------------------------------------------------------------------
_AR2FA = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "ؤ": "و", "أ": "ا", "إ": "ا"})
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")
_REPEAT = re.compile(r"(.)\1{2,}")  # عاااالی -> عالی


def normalize(text: str) -> str:
    text = str(text).translate(_AR2FA).translate(_DIGITS)
    text = _DIACRITICS.sub("", text)
    text = text.replace("\u200c", " ")  # نیم‌فاصله
    text = _REPEAT.sub(r"\1", text)
    text = re.sub(r"[^\w\s!؟?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def snap_score(scores):
    """Score در داده‌ی آموزش تقریباً فقط ضرایب ۴ است؛ همان گِرد کردن را اینجا تکرار می‌کنیم."""
    return np.clip(np.round(np.asarray(scores, dtype=float) / 4) * 4, 0, 100)


def meta_X(text_probs, scores):
    return np.column_stack([text_probs, np.asarray(scores, dtype=float)])


def stack_proba(model, texts, scores):
    """برمی‌گرداند: (احتمال خام، احتمال با وزن کلاس‌ها نرمال‌شده)"""
    tp = model["text_pipe"].predict_proba(texts)
    raw = model["meta"].predict_proba(meta_X(tp, scores))
    adj = raw * model["weights"]
    adj = adj / adj.sum(axis=1, keepdims=True)
    return raw, adj


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def predict_sentiment(text: str, score: float) -> dict:
    clean = normalize(text)
    sc = snap_score([score])
    _, adj = stack_proba(model, [clean], sc)
    probs = adj[0]
    return {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}


def ask_ollama(text: str, model_probs: dict) -> str | None:
    prompt = (
        "این یک نظر کاربر درباره یک کالای دیجیتال است. "
        "فقط با یکی از سه کلمه «مثبت»، «خنثی» یا «منفی» به احساس این متن پاسخ بده و "
        "در یک جمله کوتاه دلیل را توضیح بده.\n\n"
        f"متن: \"{text}\""
    )
    try:
        resp = requests.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=1200,
        )
        # اگر خطا باشه وارد اکسپکت میکنه
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ اتصال به Ollama برقرار نشد ({e}). فقط نتیجه مدل نمایش داده می‌شود.")
        return None


def render_probabilities(probs: dict):
    for cls in CLASS_NAMES:
        pct = probs[cls] * 100
        st.markdown(
            f"""
            <div class="prob-row">
                <div style="width:90px; font-weight:600;">{CLASS_FA[cls]}</div>
                <div style="flex:1; background:#eee; border-radius:10px; overflow:hidden; height:22px; margin:0 10px;">
                    <div style="width:{pct:.1f}%; background:{CLASS_COLOR[cls]}; height:100%;"></div>
                </div>
                <div style="width:55px; text-align:left;">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Main interaction
# ----------------------------------------------------------------------------
comment = st.text_area("✍️ متن کامنت را وارد کن:", height=140, placeholder="مثلاً: کیفیت این محصول اصلا خوب نبود...")
score = st.slider(
    "⭐ امتیازی که کاربر به محصول داده (۰ تا ۱۰۰):",
    min_value=0, max_value=100, value=DEFAULT_SCORE, step=1, format="%d",
    help="همان نمره‌ای که دیجی‌کالا کنار نظر کاربر نشان می‌دهد. این عدد دقت مدل را به‌طور محسوسی بالا می‌برد.",
)
st.markdown(f"<span class='slider-value'>امتیاز انتخاب‌شده: {score} از ۱۰۰</span>", unsafe_allow_html=True)

if st.button("🔍 تحلیل کن", use_container_width=True) and comment.strip():
    with st.spinner("در حال تحلیل..."):
        probs = predict_sentiment(comment, score)
    # کلیدها را براساس مقدار متناظرشان در Dictionary مقایسه کن.
    pred_class = max(probs, key=probs.get)
    max_conf = probs[pred_class]

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📊 خروجی مدل")
        render_probabilities(probs)

    with col2:
        st.markdown(
            f"""<div class="verdict-card" style="background:{CLASS_COLOR[pred_class]};">
                    نتیجه مدل: {CLASS_FA[pred_class]}<br>
                    <span style="font-size:1rem; font-weight:400;">اطمینان: {max_conf*100:.1f}%</span>
                </div>""",
            unsafe_allow_html=True,
        )
        if max_conf < confidence_threshold:
            st.caption("⚠️ اطمینان مدل پایین است — نتیجه را با احتیاط در نظر بگیر.")

    final_verdict = CLASS_FA[pred_class]
    final_source = "مدل Stacking"

    # If the model suspects a negative comment above the threshold, get a second opinion from Ollama
    if use_ollama and probs["bad"] > threshold:
        st.divider()
        st.subheader("🤖 نظر دوم از Ollama (به دلیل احتمال منفی بالا)")
        ollama_answer = ask_ollama(comment, probs)
        if ollama_answer:
            st.markdown(f"<div class='ollama-card'>{ollama_answer}</div>", unsafe_allow_html=True)
            final_source = "Ollama (نظر دوم به دلیل شک به منفی بودن)"
            for key, fa in CLASS_FA.items():
                if fa.split(" ")[0] in ollama_answer or key == "bad" and "منفی" in ollama_answer:
                    final_verdict = fa
                    break

    st.divider()
    st.markdown(
        f"<h4 style='text-align:center;'>🏁 نتیجه نهایی: {final_verdict} &nbsp; "
        f"<span style='font-size:0.9rem; color:gray;'>(بر اساس {final_source})</span></h4>",
        unsafe_allow_html=True,
    )
elif comment.strip() == "":
    st.caption("برای شروع، یک متن وارد کن، امتیاز را تنظیم کن و روی «تحلیل کن» کلیک کن.")
