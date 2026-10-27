
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "training_urls.csv"
MODEL_FILE = BASE_DIR / "phishing_model.joblib"


def train_model():
    if not DATA_FILE.exists():
        print("ERROR: training_urls.csv not found.")
        return

    data = pd.read_csv(DATA_FILE)

    if not {"url", "label"}.issubset(data.columns):
        print("ERROR: CSV needs url and label columns.")
        return

    data = data.dropna(subset=["url", "label"])
    data["url"] = data["url"].astype(str)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    data = data[data["label"].isin([0, 1])]

    if len(data) < 4 or data["label"].nunique() != 2:
        print("ERROR: Add at least 4 valid rows with both labels, 0 and 1.")
        return

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(data["url"], data["label"])
    joblib.dump(model, MODEL_FILE)

    print("Training completed!")
    print("Model saved to:", MODEL_FILE)


if __name__ == "__main__":
    train_model()
