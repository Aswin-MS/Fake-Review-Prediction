# streamlit_app.py
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

# ------------------- UI -------------------
st.title("ReviewClassifier AI")
st.write("Paste a review or upload a CSV to classify reviews as **FAKE** or **Genuine**.")

# sidebar controls
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Probability threshold to mark as FAKE", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
show_probs = st.sidebar.checkbox("Show probabilities", value=True)

# Single review
st.header("Single review prediction")
text_input = st.text_area("Enter a review here", height=150)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Predict single review"):
        if not text_input.strip():
            st.warning("Please enter a review text.")
        else:
            cleaned = clean_text(text_input)
            X = vec.transform([cleaned])
            # try predict_proba, else fallback to predict
            try:
                probs = model.predict_proba(X)[0]
                # assume class 1 corresponds to Fake (if your labels were 0=genuine,1=fake)
                prob_fake = float(probs[1]) if probs.shape[0] > 1 else None
            except Exception:
                prob_fake = None

            if prob_fake is None:
                pred = model.predict(X)[0]
                # handle different label types
                if isinstance(pred, (int, np.integer)):
                    pred_label = "FAKE" if int(pred) == 1 else "Genuine"
                else:
                    pred_label = str(pred)
            else:
                pred_label = "FAKE" if prob_fake >= threshold else "Genuine"

            st.markdown(f"**Prediction:** `{pred_label}`")
            if show_probs and prob_fake is not None:
                st.markdown(f"**Probability (fake):** `{prob_fake:.3f}`")

with col2:
    st.info("Tips:\n- Ensure preprocessing in this app is identical to what you used in training.\n- If your model was saved as a pipeline (vectorizer+clf), adapt loading accordingly.")

st.markdown("---")

# Batch CSV
st.header("Batch prediction (CSV)")
st.write("Upload a CSV file containing reviews. The app will attempt to find a text column (names like 'review','text','comment').")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()

    # auto-detect text column
    text_candidates = [c for c in df.columns if any(k in c.lower() for k in ("review", "text", "comment", "message"))]
    if text_candidates:
        default_col = text_candidates[0]
        st.write(f"Auto-detected text column: **{default_col}**")
        text_col = st.selectbox("Select text column", df.columns, index=list(df.columns).index(default_col))
    else:
        text_col = st.selectbox("Select text column", df.columns)

    if st.button("Run batch prediction"):
        df["cleaned_text"] = df[text_col].astype(str).map(clean_text)
        X = vec.transform(df["cleaned_text"])

        try:
            probs = model.predict_proba(X)[:,1]  # probability of class '1' (fake)
            df["prob_fake"] = probs
            df["pred"] = np.where(df["prob_fake"] >= threshold, "FAKE", "Genuine")
        except Exception:
            preds = model.predict(X)
            # convert preds to readable labels
            df["pred"] = [("FAKE" if (str(p).lower() in ("1","true","fake")) or (isinstance(p,(int,np.integer)) and int(p)==1) else "Genuine") for p in preds]
            df["prob_fake"] = np.nan

        st.success("Done — showing top rows:")
        st.dataframe(df[[text_col, "pred", "prob_fake"]].head(300))

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv_out, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Model loaded from model.pkl and vectorizer.pkl. If you saved a pipeline containing both, load that pipeline and skip separate vectorizer transform.")
