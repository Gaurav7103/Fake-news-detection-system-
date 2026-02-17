from flask import Flask, render_template, request, jsonify
import joblib
import os
import csv
from datetime import datetime
from urllib.parse import urlparse, quote
import re
import requests

app = Flask(__name__)

# Load ML model with error handling
MODEL_PATH = "model.pkl"

def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please run train_model.py first.")

        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Trusted news sources
TRUSTED_SOURCES = {
    "bbc.com": 95, "reuters.com": 96, "apnews.com": 94, "thehindu.com": 88,
    "indianexpress.com": 85, "ndtv.com": 84, "timesofindia.indiatimes.com": 82,
    "cnn.com": 83, "aljazeera.com": 86, "bloomberg.com": 89,
    "timesnownews.com": 80, "lokmat.com": 78, "nytimes.com": 90
}

# Suspicious sources
SUSPICIOUS_SOURCES = {
    "wordpress.com": 25, "blogspot.com": 20, "medium.com": 45,
    "quora.com": 40, "reddit.com": 50, "facebook.com": 15,
    "whatsapp.com": 10, "telegram.org": 15
}

# Suspicious patterns for fake news detection
SUSPICIOUS_PATTERNS = [
    r'\b(breaking|shocking|urgent|warning|alert|viral|miracle|secret)\b',
    r'\b(cure|instantly|immediately|now|today|emergency|must see)\b',
    r'!{2,}', r'\b(aliens|conspiracy|cover.?up|truth|exposed|revealed)\b',
    r'\b(they.?don.?t.?want.?you.?to.?know|hidden|suppressed)\b'
]

def analyze_text_features(text):
    """Analyze text for fake news indicators"""
    if not text:
        return {
            'suspicion_score': 0,
            'exclamation_count': 0,
            'caps_ratio': 0,
            'word_count': 0,
            'sensational_words': 0
        }

    text_lower = text.lower()
    word_count = len(text.split())
    exclamation_count = text.count('!')
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = (caps_count / len(text)) * 100 if text else 0

    # Check for suspicious patterns
    sensational_words = 0
    for pattern in SUSPICIOUS_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        sensational_words += len(matches)

    # Calculate suspicion score
    suspicion_score = 0
    if exclamation_count > 2: suspicion_score += 2
    if caps_ratio > 30: suspicion_score += 2
    if sensational_words >= 2: suspicion_score += 3
    if word_count < 15: suspicion_score += 1

    return {
        'suspicion_score': min(suspicion_score, 10),
        'exclamation_count': exclamation_count,
        'caps_ratio': round(caps_ratio, 1),
        'word_count': word_count,
        'sensational_words': sensational_words
    }

def get_domain_analysis(url):
    """Analyze domain credibility"""
    if not url:
        return {
            'domain': 'N/A',
            'credibility_score': 50,
            'domain_type': 'unknown',
            'is_trusted': False,
            'is_suspicious': False
        }

    try:
        domain = urlparse(url).netloc.lower()

        # Check trusted sources
        for source, score in TRUSTED_SOURCES.items():
            if source in domain:
                return {
                    'domain': domain,
                    'credibility_score': score,
                    'domain_type': 'trusted',
                    'is_trusted': True,
                    'is_suspicious': False
                }

        # Check suspicious sources
        for source, score in SUSPICIOUS_SOURCES.items():
            if source in domain:
                return {
                    'domain': domain,
                    'credibility_score': score,
                    'domain_type': 'suspicious',
                    'is_trusted': False,
                    'is_suspicious': True
                }

        return {
            'domain': domain,
            'credibility_score': 50,
            'domain_type': 'unknown',
            'is_trusted': False,
            'is_suspicious': False
        }

    except Exception:
        return {
            'domain': 'Invalid URL',
            'credibility_score': 30,
            'domain_type': 'invalid',
            'is_trusted': False,
            'is_suspicious': True
        }

def generate_news_search_urls(text, url=""):
    """Generate search URLs for various news sources"""
    urls = {}

    if url:
        # Direct URL search
        encoded_url = quote(url)
        urls['google_search'] = f"https://www.google.com/search?q={encoded_url}"
        urls['google_news'] = f"https://news.google.com/search?q={encoded_url}"
    elif text:
        # Extract keywords for search
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        common_words = {'this', 'that', 'with', 'from', 'have', 'were', 'their', 'about', 'which'}
        keywords = [w for w in words if w not in common_words][:5]
        query = "+".join(keywords)

        # Generate search URLs
        urls['google_search'] = f"https://www.google.com/search?q={query}+news"
        urls['google_news'] = f"https://news.google.com/search?q={query}"
        urls['times_of_india'] = f"https://timesofindia.indiatimes.com/search?q={query}"
        urls['times_now'] = f"https://www.timesnownews.com/search?q={query}"
        urls['lokmat'] = f"https://www.lokmat.com/search/?q={query}"
        urls['nytimes'] = f"https://www.nytimes.com/search?query={query}"

    return urls

def get_verification_details(text, url=""):
    """Get comprehensive verification details"""
    domain_analysis = get_domain_analysis(url)
    text_analysis = analyze_text_features(text)
    news_urls = generate_news_search_urls(text, url)

    return {
        'domain_analysis': domain_analysis,
        'text_analysis': text_analysis,
        'fact_check_urls': news_urls
    }

def get_final_prediction(ml_prediction, ml_confidence, text, url=""):
    """Combine ML prediction with rule-based analysis"""
    verification = get_verification_details(text, url)

    # Start with ML prediction
    final_prediction = ml_prediction
    final_confidence = ml_confidence

    # Rule-based adjustments
    text_analysis = verification['text_analysis']
    domain_analysis = verification['domain_analysis']

    # Strong fake indicators override
    if text_analysis['suspicion_score'] >= 6:
        if ml_prediction == "real":
            final_prediction = "likely_fake"
            final_confidence = max(60, ml_confidence - 25)

    # Trusted domain but ML says fake
    elif domain_analysis['is_trusted'] and ml_prediction == "fake":
        final_prediction = "likely_real"
        final_confidence = 75

    # Suspicious domain but ML says real
    elif domain_analysis['is_suspicious'] and ml_prediction == "real":
        final_prediction = "likely_fake"
        final_confidence = 65

    # Adjust confidence based on text analysis
    if text_analysis['suspicion_score'] >= 4:
        final_confidence = max(40, final_confidence - 15)
    elif text_analysis['suspicion_score'] <= 1:
        final_confidence = min(90, final_confidence + 10)

    return final_prediction, final_confidence, verification

@app.route("/")
def home():
    return render_template("index.html", model_loaded=model is not None)

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news", "").strip()
    news_url = request.form.get("url", "").strip()

    if not news_text and not news_url:
        return render_template("index.html", error="Please enter news text or URL!")

    try:
        # Default values
        prediction = "unknown"
        confidence = 50
        google_url = "https://www.google.com"

        # Use ML model if available
        if model and news_text:
            ml_prediction = model.predict([news_text])[0]
            ml_proba = model.predict_proba([news_text])[0]
            ml_confidence = max(ml_proba) * 100

            # Get verification details
            verification = get_verification_details(news_text, news_url)
            google_url = verification['fact_check_urls'].get('google_search', 'https://www.google.com')

            # Get final prediction with adjustments
            prediction, confidence, verification = get_final_prediction(
                ml_prediction, ml_confidence, news_text, news_url
            )

        elif news_url:
            # URL-only analysis
            verification = get_verification_details("", news_url)
            google_url = verification['fact_check_urls'].get('google_search', 'https://www.google.com')
            domain_analysis = verification['domain_analysis']

            if domain_analysis['is_trusted']:
                prediction, confidence = "real", 80
            elif domain_analysis['is_suspicious']:
                prediction, confidence = "fake", 70
            else:
                prediction, confidence = "unknown", 50

        else:
            return render_template("index.html", error="Please enter news text for analysis.")

        # Save to file
        if news_url:
            os.makedirs("links", exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open("links/verified_links.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["date", "url", "result", "confidence"])
                writer.writerow([timestamp, news_url, prediction, f"{confidence:.1f}"])

        # Determine confidence level
        if confidence >= 80:
            confidence_level = "high"
        elif confidence >= 60:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return render_template("predict.html",
                            news=news_text if news_text else news_url,
                            result=prediction,
                            confidence=f"{confidence:.1f}%",
                            confidence_level=confidence_level,
                            verification_details=verification,
                            google_search_url=google_url,
                            result_type="text" if news_text else "url")

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template("index.html", error=f"Analysis error: {str(e)}")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint for news analysis"""
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        news_url = data.get('url', '')

        if model and news_text:
            ml_prediction = model.predict([news_text])[0]
            ml_confidence = max(model.predict_proba([news_text])[0]) * 100
            prediction, confidence, verification = get_final_prediction(
                ml_prediction, ml_confidence, news_text, news_url
            )
        else:
            verification = get_verification_details(news_text, news_url)
            prediction, confidence = "unknown", 50

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'verification_details': verification
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == "__main__":
    print("🌐 Starting Fake News Detection System...")
    print("🔍 Features: ML Analysis + Domain Verification + News Source Integration")

    if model is None:
        print("⚠️  ML model not loaded. Using rule-based analysis.")
        print("💡 Run 'python train_model.py' to train the model.")
    else:
        print("✅ ML model loaded successfully!")

    print("🚀 System ready at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
