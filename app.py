from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
api = Api(app)
sid = SentimentIntensityAnalyzer()

class SentimentAnalysis(Resource):
    def post(self):
        data = request.get_json()
        text = data.get('text')
        if text:
            scores = sid.polarity_scores(text)
            return jsonify(scores)
        return {'message': 'No text provided'}, 400

api.add_resource(SentimentAnalysis, '/analyze')

if __name__ == '__main__':
    app.run(debug=True)
