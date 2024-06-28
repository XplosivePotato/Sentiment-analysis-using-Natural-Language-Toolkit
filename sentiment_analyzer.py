import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def assess_sentiment(input_text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(input_text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def run_sentiment_analysis():
    
    # Prompt
    user_input = input("Enter the text you'd like to analyze for sentiment: ")

    # Analyze
    sentiment = assess_sentiment(user_input)

    # Output
    print(f"Sentiment Analysis Result: {sentiment}")

if __name__ == "__main__":

    nltk.download('vader_lexicon')
    
    run_sentiment_analysis()
