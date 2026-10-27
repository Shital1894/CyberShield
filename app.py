from flask import Flask, render_template, request, jsonify
from detector import analyze_url

app = Flask(__name__)

model = None
model_features = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"valid": False, "error": "Please enter a URL"}), 400

    try:
        result = analyze_url(url, model, model_features)

        if not result.get("valid", False):
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        app.logger.exception("Error analyzing URL")
        return jsonify({
            "valid": False,
            "error": "Failed to analyze URL"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
