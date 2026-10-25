import re
import html
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "models/sentiment_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
MAX_LEN = 300

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="centered")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def clean_review(text):
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and the trained BiLSTM model will estimate whether its sentiment is positive or negative.")
st.caption("Educational project only. Predictions may be incorrect, especially for sarcasm, mixed opinions, or unfamiliar wording.")

try:
    model, tokenizer = load_artifacts()
except Exception:
    st.error("Model files were not found. First run `python train_model.py` and make sure the models folder contains the saved model and tokenizer.")
    st.stop()

review = st.text_area(
    "Movie review",
    placeholder="Example: The acting was excellent and the story kept me interested...",
    height=160
)

if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if len(review.strip()) < 3:
        st.warning("Please enter a longer review.")
    else:
        cleaned = clean_review(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        score = float(model.predict(padded, verbose=0)[0][0])
        label = "Positive" if score >= 0.5 else "Negative"
        confidence = score if score >= 0.5 else 1 - score

        if label == "Positive":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")
        st.metric("Model confidence estimate", f"{confidence * 100:.1f}%")
        st.progress(float(confidence))
        st.caption(f"Positive score: {score:.3f} | Negative score: {1-score:.3f}")

with st.expander("Try a sample review"):
    st.write("Positive: “A wonderful film with excellent acting and a story that stayed with me.”")
    st.write("Negative: “The plot was dull, the acting was poor, and I regretted watching it.”")
