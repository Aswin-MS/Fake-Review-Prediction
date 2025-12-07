import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

st.set_page_config(page_title="Fake Review Detector", layout="wide")

#Preprocessing

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

#Load pickle files
@st.cache_resource
def load_artifacts(vec_path="models/vectorizer.pkl", model_path="models/model.pkl"):
    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensure {vec_path} and {model_path} exist in the app folder.")
    vec = pickle.load(open(vec_path, "rb"))
    model = pickle.load(open(model_path, "rb"))
    return vec, model

try:
    vec, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

#webpage
st.markdown("""
<style>
.title {
  font-size: 48px;
  font-weight: 800;
  text-align: center;
  background: linear-gradient(90deg, #ff4b4b, #ffae00, #00c6ff, #7d2cff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  white-space: nowrap;       
  overflow: hidden;          
  display: inline-block;
}


.typing {
  border-right: 4px solid #00c3ff;  
  animation: typing 3s steps(22) forwards, blink .6s step-end infinite;
  box-sizing: content-box; 
}

@keyframes typing {
  from { width: 0ch; border-right-color: #00c3ff; }
  99%  { border-right-color: #00c3ff; } 
  100% { width: 17ch; border-right-color: transparent; } 
}

@keyframes blink {
  70% { border-right-color: transparent; }
}

.title-wrap {
  display: flex;
  justify-content: center;
  margin-top: 6px;
  margin-bottom: 12px;
}
</style>

<div class="title-wrap">
  <h1 class="title typing">ReviewClassifier AI</h1>
</div>
""", unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 20px;'>Enter a review or upload a CSV to classify reviews as Fake or Genuine.</p>",
    unsafe_allow_html=True
)
#Threshold slider
with st.sidebar:
    st.sidebar.header("Threshold Slider")
    threshold = st.sidebar.slider("Probability threshold to mark as FAKE", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
    st.info(f"Reviews with **probability of fake ≥ {threshold:.2f}** will be marked as **FAKE**.\n "
            f"Reviews with probability of fake less than **{threshold:.2f}** will be marked as **Genuine**.")
    show_probs = st.sidebar.checkbox("Show probabilities", value=True)

# Review(user)
st.header("Predict a Single Review")
text_input = st.text_area("Enter a review here:", height=100)


if st.button("Predict single review"):
    if not text_input.strip():
        st.warning("Please enter a review text.")
    else:
        cleaned = clean_text(text_input)
        X = vec.transform([cleaned])
        try:
            probs = model.predict_proba(X)[0]

            prob_fake = float(probs[0]) if probs.shape[0] > 1 else None
        except Exception:
            prob_fake = None

        if prob_fake is None:
            pred = model.predict(X)[0]
            if isinstance(pred, (int, np.integer)):
                pred_label = "FAKE" if int(pred) == 0 else "Genuine"
            else:
                pred_label = str(pred)
            st.warning("This model does not support probability scores. Only class predictions are displayed.")
        else:
            pred_label = "FAKE" if prob_fake >= threshold else "Genuine"

        st.markdown(f"**Prediction:** `{pred_label}`")
        if show_probs and prob_fake is not None:
            st.markdown(f"**Probability (fake):** `{prob_fake:.3f}`")

st.markdown("---")

# CSV File
st.header("Predict CSV file")
st.write("Upload a CSV file containing reviews:")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()

    text_candidates = [c for c in df.columns if any(k in c.lower() for k in ("review", "text", "comment", "message"))]
    st.info(
        "The app will try to auto-detect the column containing review text\n\n"
        "If auto-detection is wrong or multiple text columns exist, please select your review column from the dropdown below."
    )
    if text_candidates:
        default_col = text_candidates[0] #if multiple text column select first one
        st.write(f"Auto-detected text column: **{default_col}**")
        text_col = st.selectbox("Select text column", df.columns, index=list(df.columns).index(default_col))
    else:
        text_col = st.selectbox("Select text column", df.columns)
    st.info("Please ensure the selected column contains **REVIEW TEXT**. Wrong selection will cause incorrect results.")

    if st.button("Run batch prediction"):
        if not hasattr(model, "predict_proba"):
            st.warning("This model does not support probability scores. Only class predictions are displayed.")
        df["cleaned_text"] = df[text_col].astype(str).map(clean_text)
        X = vec.transform(df["cleaned_text"])

        try:
            probs = model.predict_proba(X)[:,0]  
            df["probability(fake)"] = probs
            df["pred"] = np.where(df["prob_fake"] >= threshold, "FAKE", "Genuine")
        except Exception:
            preds = model.predict(X)
            df["pred"] = [("FAKE" if (isinstance(p, (int, np.integer)) and int(p) == 0) or (str(p).strip().lower() in ("0", "cg", "fake")) else "Genuine") for p in preds]
            df["probability(fake)"] = np.nan

        st.success("Completed — showing top rows:")
        st.dataframe(df[[text_col, "pred", "probability(fake)"]].head(300))

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv_out, file_name="ReviewClassifier_AI_predictions.csv", mime="text/csv")

st.markdown("---")
st.markdown("""
<style>
.block-container {
    padding-bottom: 0px !important;
}

.footer {
    width: 100%;
    background: #0e1117;
    color: #cccccc;
    text-align: center;
    padding: 15px 0 20px 0;
    font-size: 14px;
    border-top: 1px solid #333;
    margin-top: 90px;
}
.icon {
    width: 20px;
    height: 20px;
    vertical-align: middle;
    margin-right: 6px;
    filter: brightness(0) invert(1);
}           

.footer a {
    color: #00c3ff;
    text-decoration: none;
    margin: 0 8px;
}

.footer a:hover {
    text-decoration: underline;
}
</style>

<div class="footer">
    © 2025 ReviewClassifier AI • All Rights Reserved<br>
    Built by <strong>Aswin M S</strong><br>
     <span style="font-size:12px;">Predictions may not be 100% accurate.</span><br>
    Contact me: <a href="mailto:msaswin175@gmail.com"><img class="icon" src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/gmail.svg">Email</a> <br>
    <a href="https://github.com/Aswin-MS/Fake-Review-Prediction" target="_blank">📁 Project Repository</a> <br>
    <div class="footer-right">
        <strong>Let's Connect:</strong><br>
        <a href="https://github.com/Aswin-MS" target="_blank"><img class="icon" src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg">GitHub</a>
        <a href="https://linkedin.com/in/aswinms175" target="_blank"><img class="icon" src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg">LinkedIn</a>
    </div>

</div>
""", unsafe_allow_html=True)
