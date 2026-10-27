
from urllib.parse import urlparse
import ipaddress

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "password", "account", "update",
    "secure", "banking", "confirm", "signin", "reset",
    "security"
]


def extract_features(url):
    """Convert a URL into numerical features."""

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    lower_url = url.lower()

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "dot_count": hostname.count("."),
        "hyphen_count": hostname.count("-"),
        "digit_count": sum(char.isdigit() for char in url),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip": has_ip,
        "has_at": int("@" in url),
        "has_punycode": int("xn--" in hostname.lower()),
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "keyword_count": sum(
            1 for keyword in SUSPICIOUS_KEYWORDS
            if keyword in lower_url
        )
    }


def analyze_url(url, model=None, model_features=None):
    """Analyze a URL and optionally use a trained ML model."""

    if not isinstance(url, str) or not url.strip():
        return {"valid": False, "error": "Please enter a URL."}

    url = url.strip()

    if len(url) > 2048:
        return {"valid": False, "error": "URL is too long."}

    # Add a scheme when the user enters a domain without one.
    candidate = url if "://" in url else "https://" + url
    parsed = urlparse(candidate)

    try:
        hostname = parsed.hostname or ""
        parsed.port  # Check that the port is valid.
    except ValueError:
        return {"valid": False, "error": "Invalid URL."}

    if (
        parsed.scheme.lower() not in ("http", "https")
        or not hostname
        or any(char.isspace() for char in url)
        or parsed.username is not None
        or parsed.password is not None
    ):
        return {"valid": False, "error": "Invalid URL."}

    hostname = hostname.lower()
    features = extract_features(candidate)

    reasons = []
    score = 0

    if parsed.scheme.lower() == "http":
        score += 1
        reasons.append("The URL uses HTTP instead of HTTPS.")

    if "@" in candidate:
        score += 2
        reasons.append("The URL contains an @ symbol that may obscure its structure.")

    if "xn--" in hostname:
        score += 2
        reasons.append("The hostname contains punycode.")

    if features["subdomain_count"] >= 3:
        score += 1
        reasons.append("The hostname has multiple subdomains.")

    found_keywords = [
        keyword for keyword in SUSPICIOUS_KEYWORDS
        if keyword in candidate.lower()
    ]

    if found_keywords:
        score += min(3, len(found_keywords))
        reasons.append(
            "Suspicious keywords found: " + ", ".join(found_keywords)
        )

    if features["has_ip"]:
        score += 2
        reasons.append("The hostname is an IP address.")

    score = min(score, 10)

    if score <= 1:
        risk_level = "Low"
    elif score <= 3:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    if not reasons:
        reasons.append(
            "No configured warning patterns were detected. "
            "This does not prove the URL is safe."
        )

    ml_result = "Model not used"
    ml_prediction = None

    if model is not None:
        try:
            # A text-based model trained by train_model.py
            # expects the URL itself, not numerical features.
            prediction = int(model.predict([candidate])[0])
            ml_prediction = prediction

            if prediction == 1:
                ml_result = "Potentially suspicious"
            else:
                ml_result = "Classified as legitimate by the model"

        except (ValueError, TypeError, AttributeError) as error:
            ml_result = "Model prediction unavailable: " + str(error)

    return {
        "valid": True,
        "url": candidate,
        "hostname": hostname,
        "scheme": parsed.scheme.lower(),
        "features": features,
        "score": score,
        "risk_level": risk_level,
        "reasons": reasons,
        "ml_result": ml_result,
        "ml_prediction": ml_prediction,
        "note": (
            "This is an educational assessment, not a verified verdict. "
            "A low score does not guarantee safety."
        )
    }
