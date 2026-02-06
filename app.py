from flask import Flask, request, jsonify, render_template
import joblib
import io
import base64
import numpy as np
from matplotlib.figure import Figure

app = Flask(__name__)

# Load models from the models directory
try:
    classifier = joblib.load('models/model.pkl')
    tfidf_vectorizer = joblib.load('models/vectorizer.pkl')
    emotion_model = joblib.load('models/emotion_model.pkl')
    emotion_vectorizer = joblib.load('models/emotion_vectorizer.pkl')
except FileNotFoundError:
    print("Error: Model files not found in /models directory. Please run training scripts first.")

EMOTION_MAP = {0: "anger", 1: "fear", 2: "joy", 3: "love", 4: "sadness", 5: "surprise"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_text = data.get("text", "")
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    # 1. Sentiment Logic
    input_features = tfidf_vectorizer.transform([input_text])
    y_proba = classifier.predict_proba(input_features)
    pos_score = float(y_proba[0][1])
    
    sentiment = "Neutral"
    if pos_score < 0.4: sentiment = "Negative"
    elif pos_score > 0.6: sentiment = "Positive"

    # 2. Emotion Logic
    emo_features = emotion_vectorizer.transform([input_text])
    emo_probs = emotion_model.predict_proba(emo_features).flatten()
    dominant_idx = int(np.argmax(emo_probs))
    dominant_emotion = EMOTION_MAP.get(dominant_idx, "Unknown")

    # 3. Thread-Safe Visualization
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    emotions = [EMOTION_MAP[i] for i in range(len(emo_probs))]
    ax.bar(emotions, emo_probs, color='skyblue')
    ax.set_title("Emotion Probabilities")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "sentiment": {"score": round(pos_score, 2), "label": sentiment},
        "emotion": {"label": dominant_emotion},
        "plot": "data:image/png;base64," + base64_image
    })

if __name__ == "__main__":
    app.run(debug=True)
