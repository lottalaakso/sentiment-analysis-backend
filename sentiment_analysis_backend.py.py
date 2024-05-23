from textblob import TextBlob
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data['text']
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return jsonify(sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
