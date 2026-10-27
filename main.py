

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib

from detector import analyze_url

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "phishing_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except (FileNotFoundError, OSError, ValueError):
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")

    if not isinstance(url, str):
        return jsonify({
            "valid": False,
            "error": "Please enter a valid URL."
        }), 400

    result = analyze_url(url, model=model)

    if not result.get("valid"):
        return jsonify(result), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
