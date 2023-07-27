import streamlit as st
from pickle import load
from textblob import TextBlob
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.title("Welcome to Finsights")

with open('model_pickle', 'rb') as f:
    model = load(f)

#not recognizing the model_tech_pickle file
# with open('finsights_ml\model_tech_pickle', 'rb') as f:
#      model_tech = load(f)

def predict_sentiment(text):
    now = datetime.now()

    # Calculate the date and time for 10 AM yesterday
    yesterday = now - timedelta(days=1)
    user_date = datetime(yesterday.year, yesterday.month, yesterday.day, hour=10, minute=0, second=0)    
    start_date = user_date - timedelta(days=365)
    # Fetch real-time stock data using yfinance
    #stock_data is empty?
    stock_symbol = 'TSLA'
    stock_data = yf.download(stock_symbol, start=start_date, end=user_date)
    if stock_data.empty:
        st.write("No data available for the specified date.")
    
    # Extract the stock price (Close price) for the specified date
    stock_price = stock_data['Close'].values[0]

    # Analyze sentiment of the user text using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment

    # Extract sentiment analysis values from the user input, similar to how its done in training model
    subjectivity = sentiment.subjectivity
    polarity = sentiment.polarity
    compound = sentiment.polarity
    negative = sentiment.polarity < 0
    neutral = sentiment.polarity == 0
    positive = sentiment.polarity > 0
    label = 1 if sentiment.polarity > 0 else 0 if sentiment.polarity == 0 else 0

    # use the model.predict function here
    input_features = [stock_data['Open'].values[0], stock_data['High'].values[0],
                        stock_data['Low'].values[0], stock_data['Volume'].values[0], subjectivity, polarity, compound, negative, neutral, positive]

    #add in technical indicators in the model, to make better predictions
    input_features_array = np.array(input_features).reshape(1, -1)
    sentiment_prediction = model.predict(input_features_array)
    return sentiment_prediction[0]

# def predict_stock_price(sma, rsi):
#     # Create the input features for the technical indicator model
#     input_features = [sma, rsi]
#     input_features_array = np.array(input_features).reshape(1, -1)
    
#     # Predict the stock price using the technical indicator model
#     price_prediction = model_tech.predict(input_features_array)
#     return price_prediction[0]

def extract_key_sentiment_indicators(text):
    # Analyze sentiment of the user text using TextBlob
    blob = TextBlob(text)

    # Extract individual words
    words = blob.words
    
    # Calculate sentiment polarity for each word
    word_sentiments = {word: TextBlob(word).sentiment.polarity for word in words}
    
    # Sort words by polarity to find most positive and negative words
    positive_words = [word for word, score in sorted(word_sentiments.items(), key=lambda x: x[1], reverse=True) if score > 0]
    negative_words = [word for word, score in sorted(word_sentiments.items(), key=lambda x: x[1]) if score < 0]

    # Extract positive and negative sentences
    positive_sentences = [str(sentence) for sentence in blob.sentences if sentence.sentiment.polarity > 0]
    negative_sentences = [str(sentence) for sentence in blob.sentences if sentence.sentiment.polarity < 0]
    return positive_words, negative_words, positive_sentences, negative_sentences

def plot_sentiment_distribution(text):
    # Analyze sentiment of the user text using TextBlob
    blob = TextBlob(text)

    # Calculate sentiment polarity for each word
    sentiments = [TextBlob(word).sentiment.polarity for word in blob.words]

    # Create histogram bins and counts
    bins = np.linspace(-1, 1, 21)  # Equally spaced bins from -1 to 1
    counts, _ = np.histogram(sentiments, bins=bins)

    # Calculate bin centers for bar plot
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Plot the sentiment distribution bar chart
    plt.bar(bin_centers, counts, width=0.1, align='center')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Word Count')
    plt.title('Sentiment Distribution of Words')
    plt.xticks(np.arange(-1, 1.1, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the chart in Streamlit
    st.pyplot()

def main():
    st.title('Sentiment and Stock Price Prediction App')
    user_input = st.text_input('Enter a sentence:')
    if st.button('Predict Sentiment'):
        sentiment_prediction = predict_sentiment(user_input)

        # Replace the following placeholder values with actual values from your stock data
        sma = 50.0  # Replace with the actual SMA value
        rsi = 60.0  # Replace with the actual RSI value

        # Predict the stock price using technical indicators
        #price_prediction = predict_stock_price(sma, rsi)

        st.write(f'The sentiment prediction is: {sentiment_prediction}')
        #st.write(f'The predicted stock price is: {price_prediction:.2f}')
        # Visualization: Pie Chart for Sentiment Distribution
        st.subheader('Sentiment Distribution')
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment_counts = np.bincount(sentiment_prediction)
        st.pyplot(plt.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%'))

        # ... (your existing code for stock data and predictions)

        # Visualization: Line Chart for Stock Price History
        st.subheader('Stock Price History')
        st.line_chart(stock_data['Close'])
    if st.button('Analyze Sentiment'):
        positive_words, negative_words, positive_sentences, negative_sentences = extract_key_sentiment_indicators(user_input)

        # Display most positive and negative words
        st.subheader('Most Positive Words:')
        st.write(positive_words)

        st.subheader('Most Negative Words:')
        st.write(negative_words)

        # Display positive and negative sentences
        st.subheader('Positive Sentences:')
        st.write(positive_sentences)

        st.subheader('Negative Sentences:')
        st.write(negative_sentences)
    if st.button('Analyze Sentiment with Sentiment Distribution'):
        # Call the function to plot sentiment distribution
        plot_sentiment_distribution(user_input)
if __name__ == '__main__':
    main()

