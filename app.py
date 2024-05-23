from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)  # Initialize CORS with the Flask app

# Initialize NLTK's Vader sentiment analyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Perform sentiment analysis
    scores = sid.polarity_scores(text)
    sentiment = 'positive' if scores['compound'] >= 0 else 'negative'
    
    return jsonify({'sentiment': sentiment, 'scores': scores})

if __name__ == '__main__':
    app.run(debug=True)
