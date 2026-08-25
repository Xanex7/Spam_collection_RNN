"""
Flask app for the SimpleRNN sentiment analysis model (model_rnn.pkl).

Architecture detected from the pickle file:
    Embedding(input_dim=5000, output_dim=32, input_length=50)
    SimpleRNN(32, activation='relu')
    Dense(1, activation='sigmoid')

This matches the classic Keras IMDB-review sentiment tutorial:
  - Vocabulary size: 5000 (most frequent words)
  - Sequence length: 50 (padded/truncated)
  - Output: probability of positive sentiment (0 = negative, 1 = positive)

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import pickle
import re

from flask import Flask, render_template_string, request
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "model_rnn.pkl"
VOCAB_SIZE = 5000
MAX_LEN = 50
INDEX_FROM = 3  # Keras IMDB dataset reserves the first 3 indices

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------
# Load the same word index Keras used to build the IMDB dataset, so raw
# text can be converted into the integer sequences the model expects.
# ---------------------------------------------------------------------
_word_index = imdb.get_word_index()
# Keras shifts indices by INDEX_FROM and reserves 0/1/2 for
# padding/start/unknown tokens.
word_index = {w: (i + INDEX_FROM) for w, i in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2


def text_to_sequence(text: str):
    """Convert raw text into a padded integer sequence for the model."""
    tokens = re.findall(r"[a-z']+", text.lower())
    seq = [1]  # <START>
    for tok in tokens:
        idx = word_index.get(tok, 2)  # 2 = <UNK>
        if idx >= VOCAB_SIZE:
            idx = 2
        seq.append(idx)
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding="pre", truncating="pre")
    return padded


def predict_sentiment(text: str):
    seq = text_to_sequence(text)
    prob = float(model.predict(seq, verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    return label, prob


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RNN Sentiment Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 640px; margin: 60px auto; padding: 0 20px; }
    textarea { width: 100%; height: 120px; font-size: 15px; padding: 10px; box-sizing: border-box; }
    button { margin-top: 10px; padding: 10px 24px; font-size: 15px; cursor: pointer; }
    .result { margin-top: 24px; padding: 16px; border-radius: 8px; font-size: 18px; }
    .positive { background: #e6f4ea; color: #1e7e34; }
    .negative { background: #fdecea; color: #c62828; }
    .prob { color: #666; font-size: 14px; }
  </style>
</head>
<body>
  <h2>Movie Review Sentiment Analyzer (SimpleRNN)</h2>
  <form method="post">
    <textarea name="review" placeholder="Type a movie review here...">{{ review or '' }}</textarea><br>
    <button type="submit">Predict Sentiment</button>
  </form>
  {% if label %}
  <div class="result {{ 'positive' if label == 'Positive' else 'negative' }}">
    <strong>{{ label }}</strong>
    <div class="prob">Confidence score: {{ '%.4f' % prob }}</div>
  </div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    label, prob, review = None, None, None
    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            label, prob = predict_sentiment(review)
    return render_template_string(PAGE, label=label, prob=prob, review=review)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return {"error": "Field 'text' is required."}, 400
    label, prob = predict_sentiment(text)
    return {"label": label, "score": prob}


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
