from django.shortcuts import render
from django.http import HttpResponse
from pickle import load
import yfinance as yf
from datetime import datetime, timedelta
from textblob import TextBlob
import pandas as pd
import numpy as np

with open('finsights_app/model_pickle', 'rb') as f:
    model = load(f)

def dow_view(request):
    return render(request, 'main.html')

def User(request):
    username = request.POST.get('username')
    print(username)
    return render(request, 'user.html', {'username': username})

def predictor(request):
    return render(request, 'predictor.html')

def form_view(request):
    #x = date and key words, y = stock data from yfinance
    date = request.POST.get('date')
    key_words = request.GET.get('key_words')
    y_pred = model.predict(date)
    if y_pred == 1:
        y_pred = 'The Dow Jones will go up'
    elif y_pred == 0:
        y_pred = 'The Dow Jones will go down'
    else:
        y_pred = 'The Dow Jones will stay the same'
    return render(request, 'result.html', {'result': y_pred, 'date': date, 'key_words': key_words})

# The new view to handle the form submission
# handle user input and real-time stock data from yfinanc here
# use model.predict to get the prediction of price for the next day
def process_user_input(request):
    if request.method == 'POST':
        # Get the user input from the form
        user_text = request.POST.get('user_input', '')
        #user_date = request.POST.get('user_date', '')
        #user_date = today_date = datetime.today()
        user_date = datetime.now() - timedelta(days=5)

        # Fetch real-time stock data using yfinance
        #stock_data is empty?
        stock_symbol = 'TSLA'
        stock_data = yf.download(stock_symbol, start=user_date, end=user_date)
        if stock_data.empty:
            return render(request, 'result.html', {'result': 'No data available for the specified date.'})
        
        # Extract the stock price (Close price) for the specified date
        stock_price = stock_data['Close'].values[0]

        # Analyze sentiment of the user text using TextBlob
        blob = TextBlob(user_text)
        sentiment = blob.sentiment

        # Extract sentiment analysis values from the user input, similar to how its done in training model
        subjectivity = sentiment.subjectivity
        polarity = sentiment.polarity
        compound = sentiment.polarity
        negative = sentiment.polarity < 0
        neutral = sentiment.polarity == 0
        positive = sentiment.polarity > 0
        label = 1 if sentiment.polarity > 0 else 0 if sentiment.polarity == 0 else 0

        # Assuming your model will return the predicted value as a number (e.g., $10.00)
        # use the model.predict function here
        #need to make this a dataframe
        input_features = [stock_data['Open'].values[0], stock_data['High'].values[0],
                          stock_data['Low'].values[0], stock_data['Volume'].values[0], subjectivity, polarity, compound, negative, neutral, positive]

        # Create a DataFrame from the list of lists
        #need to edit the parameter, so that it can fit my model

        input_features_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_features_array)
        

        # Render the result.html template with the prediction result
        return render(request, 'result.html', {'result': f'The {stock_symbol} on {user_date} will increase by ${prediction[0]:.2f}',
                                       'subjectivity': subjectivity,
                                       'polarity': polarity,
                                       'compound': compound,
                                       'negative': negative,
                                       'neutral': neutral,
                                       'positive': positive,
                                       'label': label})


    # If the request method is not POST, render the form template
    return render(request, 'result.html', {'result': 'The Dow Jones price will increase by $10.00'})
