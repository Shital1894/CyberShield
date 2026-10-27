
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "training_urls.csv"
MODEL_FILE = BASE_DIR / "phishing_model.joblib"


def train_model():
    if not DATA_FILE.exists():
        print("Error: training_urls.csv was not found.")
        print("Create this CSV file in the same folder first.")
        return

    data = pd.read_csv(DATA_FILE)

    if not {"url", "label"}.issubset(data.columns):
        print("CSV must have columns: url,label")
        return

    data = data.dropna(subset=["url", "label"])
    data["url"] = data["url"].astype(str)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    data = data[data["label"].isin([0, 1])]

    if len(data) < 10 or data["label"].nunique() != 2:
        print("Add at least 10 labeled URLs, including both classes.")
        return

    X = data["url"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=20000
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\nModel evaluation:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    joblib.dump(model, MODEL_FILE)
    print(f"Model saved successfully: {MODEL_FILE.name}")



if __name__ == "__main__":
    train_model()

