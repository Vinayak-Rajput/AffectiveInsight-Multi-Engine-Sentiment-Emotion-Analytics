from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import io
import base64
from matplotlib.figure import Figure

app = Flask(__name__)

# Load models and vectorizers
# Note: Ensure these files exist in the /models directory
sentiment_model = joblib.load('models/sentiment_model.pkl')
sentiment_vec = joblib.load('models/sentiment_vectorizer.pkl')
emotion_model = joblib.load('models/emotion_model.pkl')
emotion_vec = joblib.load('models/emotion_vectorizer.pkl')

EMOTION_MAP = {0: "anger", 1: "fear", 2: "joy", 3: "love", 4: "sadness", 5: "surprise"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "")
    
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    # 1. Sentiment Analysis
    feat_sent = sentiment_vec.transform([input_text])
    pos_score = float(sentiment_model.predict_proba(feat_sent)[0][1])
    sentiment = "Neutral"
    if pos_score < 0.4: sentiment = "Negative"
    elif pos_score > 0.6: sentiment = "Positive"

    # 2. Emotion Detection
    feat_emot = emotion_vec.transform([input_text])
    emot_probs = emotion_model.predict_proba(feat_emot).flatten()
    dom_emot = EMOTION_MAP[np.argmax(emot_probs)]

    # 3. Thread-Safe Visualization
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    emotions = [EMOTION_MAP[i] for i in range(len(emot_probs))]
    ax.bar(emotions, emot_probs, color='#4a90e2')
    ax.set_title("Emotion Probability Distribution")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "sentiment": {"score": round(pos_score, 2), "label": sentiment},
        "emotion": {"dominant": dom_emot, "probabilities": emot_probs.tolist()},
        "plot": "data:image/png;base64," + plot_url
    })

if __name__ == "__main__":
    app.run(debug=True)
