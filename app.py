import streamlit as st
from pickle import load
from textblob import TextBlob
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Import the SentimentIntensityAnalyzer object
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#Roberta Pretrained Model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import math

st.title("Welcome to Finsights")

with open('model_pickle', 'rb') as f:
    model = load(f)

#import the price prediction model
with open('model_price_pickle', 'rb') as f:
    model_price = load(f)

with(open('model_tech_pickle', 'rb')) as f:
    model_tech = load(f)

#i should change this to? 2 day on the market?
def download_stock_data(ticker):
    now = datetime.now()

    # Calculate the date and time for 10 AM yesterday
    #gets the stock data for the previous day
    today = datetime.today() - timedelta(days=3)
    
    # Calculate the date for the previous day
    previous_day = today - timedelta(days=2)
    
    try:
        # Fetch real-time stock data using yfinance
        stock_data = yf.download(ticker, start=previous_day, end=today)

        # Check if the data is empty
        if stock_data.empty:
            return "Not a valid ticker"

        return stock_data

    except Exception as e:
        # Handle any potential errors during data download
        print(f"Error occurred while downloading stock data: {e}")
        return None

#FUNCTIONS FOR ADVANCED SENTIMENT ANALYSIS
def polarity_scores_roberta(text):
    #Data from https://www.kaggle.com/kazanova/sentiment140
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

def get_sa_with_vader(text):
    #df not defined here
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            tweet = row['Tweet']
            my_stock = row['Stock Name']
            vader_result = sia.polarity_scores(tweet)
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value
        
            roberta_result = polarity_scores_roberta(tweet)
            both = {**vader_result_rename, **roberta_result}
            res[my_stock] = both
            
        except RuntimeError:
            print(f'Broke for stock {my_stock}')


#Get the sentiment score
#Create a function to get the sentiment scores
def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

#FUNCTIONS FOR GETTTING THE PREDICTIONS
def extract_key_features_price(text, stock_symbol):
    stock_data = download_stock_data(stock_symbol)
    
    # Analyze sentiment of the user text using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment

    # Extract sentiment analysis values from the user input
    subjectivity = sentiment.subjectivity
    polarity = sentiment.polarity
    compound = []
    negative = []
    neutral = []
    positive = []
    #instead of df_merge, i have text, so instead of a for loop, i just need to peform the code in the loop once
    SIA = getSIA(text)
    compound = (SIA['compound'])
    negative = SIA['neg']
    neutral = (SIA['neu'])
    positive = (SIA['pos'])

    #Extract daily percent change from the stock data -> Daily_Pct_Change

    # Prepare the input features for the regression model
    #features i need to have in order to have the PERCENT CHANGE model work:
        #Label and Daily_Pct_Change
    #Calculate daily percent change
    #this is calculating the daily percent change incorrectly

    #need to implement error checking here
    #values[0] is the value for the previous day -> Monday
    #values[1] is the value for the current day -> Tuesday
    label = predict_sentiment(text, stock_symbol)

    #it seems like the daily percent change is not being calculated correctly
    #i need to get the percent change from the previous day to the current day
    daily_pct_change = stock_data['Close'].pct_change().values[1]

    st.write("percent change: ", daily_pct_change)
    if math.isnan(daily_pct_change):
        daily_pct_change = 0
    input_features = [stock_data['Open'].values[0], stock_data['High'].values[0],
                      stock_data['Low'].values[0], stock_data['Volume'].values[0],
                      subjectivity, polarity, compound, negative, neutral, positive, label, daily_pct_change] 
    #it appears the last input feature does not affect the predicted_price_change for some reason, the only thing that affects the percent change is the start_date


    # Convert the input features to a NumPy array and reshape it to match the model input
    #looks like this turns this into a 2d array
    input_features_array = np.array(input_features).reshape(1, -1)
    #this line is just getting the value 365 days ago, I need to just get the value for the previous day
    #or rather use the extended time in the sma and rsi function

    return input_features_array


def extract_key_features_sentiment(text, stock_symbol):
    stock_data = download_stock_data(stock_symbol)
    
    # Analyze sentiment of the user text using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment

    # Extract sentiment analysis values from the user input
    subjectivity = sentiment.subjectivity
    polarity = sentiment.polarity
    compound = sentiment.polarity
    negative = sentiment.polarity < 0
    neutral = sentiment.polarity == 0
    positive = sentiment.polarity > 0

    #Extract label from the user input -> Label
    if negative == True:
        label = 0
    else:
        label = 1
    #Extract daily percent change from the stock data -> Daily_Pct_Change

    # Prepare the input features for the regression model
    #features i need to have in order to have the PERCENT CHANGE model work:
        #Label and Daily_Pct_Change
    input_features = [stock_data['Open'].values[0], stock_data['High'].values[0],
                      stock_data['Low'].values[0], stock_data['Volume'].values[0],
                      subjectivity, polarity, compound, negative, neutral, positive]

    # Convert the input features to a NumPy array and reshape it to match the model input
    input_features_array = np.array(input_features).reshape(1, -1)
    #this line is just getting the value 365 days ago, I need to just get the value for the previous day
    #or rather use the extended time in the sma and rsi function

    return input_features_array

def predict_sentiment(text, stock_symbol):
    # Extract the input features for the sentiment prediction model
    input_features_array = extract_key_features_sentiment(text, stock_symbol)
    sentiment_prediction = model.predict(input_features_array)
    return sentiment_prediction[0]

def predict_price(text, stock_symbol):
    # extract the input features for the price prediction model
    input_features_array = extract_key_features_price(text, stock_symbol)
    price_prediction = model_price.predict(input_features_array)
    #it looks like this value is numpy array 
    return price_prediction[0]         #was price_prediction[0]


def predict_stock_price(sma, rsi):
    # Create the input features for the technical indicator model
    input_features = [sma, rsi]
    input_features_array = np.array(input_features).reshape(1, -1)
    
    # Predict the stock price using the technical indicator model
    price_prediction = model_tech.predict(input_features_array)
    return price_prediction[0]

def extract_technical_indicators(ticker):
    now = datetime.now()

    # Calculate the date and time for 10 AM yesterday
    yesterday = now - timedelta(days=1)
    user_date = datetime(yesterday.year, yesterday.month, yesterday.day, hour=10, minute=0, second=0)    
    start_date = user_date - timedelta(days=4)
    stock_data = yf.download(ticker, start=start_date, end=user_date)

    #get the sma
    #get the rsi


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

    # Calculate ROBERTA scores
    # Extract positive and negative fsentences
    positive_sentences = [str(sentence) for sentence in blob.sentences if sentence.sentiment.polarity > 0]
    negative_sentences = [str(sentence) for sentence in blob.sentences if sentence.sentiment.polarity < 0]
    return positive_words, negative_words, positive_sentences, negative_sentences 


#FUNCTIONS TO VISUALIZE THE RESULTS OF THE PREDICTIONS
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
def get_stock_price(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    stock_price = data['Close'].values[0]
    return stock_price

#MAIN FUNCTION
def main():
    st.title('Sentiment and Stock Price Prediction App')
    data_type = st.selectbox('Type of Financial Data', ['Tweet', 'News'])
    user_input = st.text_input('Enter a sentence:')
    user_date = st.date_input('Predict stock price for:')
    ticker = st.text_input('Enter a ticker symbol:')
    submit = st.button('Submit')
    #Error handling for invalid ticker
    download_stock_data(ticker)
    stock_data = download_stock_data(ticker)

    if stock_data is None:
        st.write("Not a valid ticker. Please enter a valid stock ticker.")
        return  
    #get the proper dates
    now = datetime.now()
    yesterday = now - timedelta(days=3)
    end = datetime(yesterday.year, yesterday.month, yesterday.day, hour=10, minute=0, second=0)    
    start = end - timedelta(days=365)
    

    if st.button("Get percent change without sentiment"):
        st.write("percent: ", stock_data['Close'].pct_change().values[1])
    if st.button('Predict Price'):
        stock_data = download_stock_data(ticker)
        #get the predictions
        sentiment_label_prediction = predict_sentiment(user_input, ticker)
        stock_price_change_prediction = predict_price(user_input, ticker)
        predicted_price = stock_data['Close'].values[0] * (1 + stock_price_change_prediction)
        #output the predictions
        #st.write(f'Predicted Stock Price: {stock_price_prediction:.2f}')
        if sentiment_label_prediction == 0:
            st.write('The stock price will decrease')
        else:
            st.write('The stock price will increase')
        st.write("Closing price", stock_data['Close'].values[0])
        st.write("Opening price", stock_data['Open'].values[0])
        st.write("Volume", stock_data['Volume'].values[0])
        st.write("Price Prediction: ", predicted_price)
        st.write("The predicted percent change is: ", stock_price_change_prediction)
        st.write(f'Predicted Sentiment Label: {sentiment_label_prediction}')
    

    #here i dont use my model, but i should
    #i need to show my ability to improve the sentiment analysis model
    if st.button('Predict Sentiment'):
        sentiment_prediction = predict_sentiment(user_input, ticker)

        # Replace the following placeholder values with actual values from your stock data

        # Predict the stock price using technical indicators
        #price_prediction = predict_stock_price(sma, rsi)

        st.write(f'The sentiment prediction is: {sentiment_prediction}')
        #st.write(f'The predicted stock price is: {price_prediction:.2f}')
        # Visualization: Pie Chart for Sentiment Distribution
        st.subheader('Sentiment Distribution')
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        #sentiment_counts = np.bincount(sentiment_prediction)
        #st.pyplot(plt.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%'))
    if st.button('Get Roberta Values'):
       # Display the Roberta sentiment values for the user's input text
        roberta_scores = polarity_scores_roberta(user_input)
        st.subheader('Roberta Sentiment Scores')
        st.write(roberta_scores)

        # Visualization: Line Chart for Stock Price History
        st.subheader('Stock Price History')
        #cant convert stock price to a dataframe
        st.line_chart(get_stock_price(ticker, start, end))
    if st.button('Get stock price'):
        st.write(get_stock_price(ticker, start, end))
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

