import re
import time
from pathlib import Path

import joblib
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
VEC_PATH = Path(__file__).resolve().parent / "vectorizer.pkl"

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "will", "would", "could", "should", "may", "might", "shall", "can",
    "a", "an", "the", "and", "but", "if", "or", "nor", "for", "so", "yet", "at", "by",
    "to", "up", "in", "on", "of", "with", "about", "as", "from", "into", "during",
    "before", "after", "through", "between", "out", "off", "over", "under", "again",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "own", "same", "than", "too", "very", "just", "because", "while", "also",
}


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VEC_PATH.exists():
        missing = []
        if not MODEL_PATH.exists():
            missing.append("model.pkl")
        if not VEC_PATH.exists():
            missing.append("vectorizer.pkl")
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")
    vectorizer = joblib.load(VEC_PATH)
    model = joblib.load(MODEL_PATH)
    return model, vectorizer


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def predict_sentiment(text: str, model, vectorizer):
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")
    if len(text.strip()) < 5:
        raise ValueError("Input text is too short (minimum 5 characters).")
    if len(text) > 10_000:
        raise ValueError("Input text is too long (maximum 10,000 characters).")

    started = time.perf_counter()
    cleaned = preprocess_text(text.strip())
    if not cleaned:
        raise ValueError(
            "After preprocessing, the text contained no meaningful words. Please provide more descriptive input."
        )

    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))
    processing_ms = (time.perf_counter() - started) * 1000

    return {
        "sentiment": "Positive" if prediction == 1 else "Negative",
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.1f}%",
        "label": prediction,
        "text_received": text.strip(),
        "processing_ms": round(processing_ms, 2),
    }


def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="centered")
    st.title("Sentiment Analysis App")
    st.caption("Logistic Regression + TF-IDF (same prediction logic as previous API)")

    try:
        model, vectorizer = load_artifacts()
    except Exception as exc:
        st.error(f"Model load failed: {exc}")
        st.stop()

    review = st.text_area(
        "Enter a movie review",
        height=180,
        placeholder="Type your review here...",
    )

    if st.button("Predict Sentiment", type="primary"):
        try:
            result = predict_sentiment(review, model, vectorizer)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        if result["sentiment"] == "Positive":
            st.success(f"Sentiment: {result['sentiment']}")
        else:
            st.error(f"Sentiment: {result['sentiment']}")

        st.metric("Confidence", result["confidence_pct"])
        st.progress(min(max(float(result["confidence"]), 0.0), 1.0))
        st.caption(f"Processing time: {result['processing_ms']} ms")

        with st.expander("Prediction details"):
            st.json(result)


if __name__ == "__main__":
    main()
