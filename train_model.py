"""
Train and export sentiment model artifacts for the FastAPI backend.

Outputs:
- model/model.pkl
- model/vectorizer.pkl
- model/training_metrics.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


STOP_WORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "nor",
    "for",
    "so",
    "yet",
    "at",
    "by",
    "to",
    "up",
    "in",
    "on",
    "of",
    "with",
    "about",
    "as",
    "from",
    "into",
    "during",
    "before",
    "after",
    "through",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "because",
    "while",
    "also",
}


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def main() -> None:
    project_model_dir = Path(__file__).resolve().parent

    print("Loading IMDB dataset from Hugging Face...")
    dataset = load_dataset("imdb", split="train").shuffle(seed=42)

    # Keep training fast/reproducible while still using enough data.
    max_rows = 10000
    sample = dataset.select(range(max_rows))
    texts = sample["text"]
    labels = sample["label"]
    print(f"Using {len(texts)} training rows.")

    t0 = time.perf_counter()
    cleaned = [preprocess_text(t) for t in texts]

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(x_train_vec, y_train)

    pred = model.predict(x_test_vec)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    train_s = time.perf_counter() - t0

    model_path = project_model_dir / "model.pkl"
    vec_path = project_model_dir / "vectorizer.pkl"
    metrics_path = project_model_dir / "training_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    metrics = {
        "rows_used": len(texts),
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "training_seconds": round(train_s, 2),
        "model": "LogisticRegression",
        "vectorizer": "TfidfVectorizer(max_features=15000, ngram_range=(1,2))",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved vectorizer: {vec_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
