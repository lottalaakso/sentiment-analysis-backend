from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']

# Example usage:
text_to_analyze = "I hate this product!"
sentiment_score = analyze_sentiment(text_to_analyze)
print(f"Sentiment score: {sentiment_score}")
