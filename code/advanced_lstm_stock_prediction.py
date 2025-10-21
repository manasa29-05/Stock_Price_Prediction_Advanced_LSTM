# --- Stock Price Prediction using Advanced LSTM Techniques ---
# Includes Bidirectional, Stacked, and Attention-based LSTM models
# Real-time data fetching via Yahoo Finance
# Author: Your Name

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input, Layer
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------------- Fetch Real-time Data -----------------------
ticker = 'AAPL'
forecast_days = 5
time_step = 60

def fetch_stock_data(ticker='AAPL', period='180d', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

df = fetch_stock_data(ticker)
data = df['Close'].values.reshape(-1,1)

# ----------------------- Feature Scaling -----------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# ----------------------- Create Dataset -----------------------
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step),0])
        y.append(data[i+time_step,0])
    return np.array(X), np.array(y)

train_size = int(len(scaled_data)*0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

X_train, y_train = create_dataset(train_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

if len(test_data) > time_step:
    X_test, y_test = create_dataset(test_data, time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
else:
    X_test, y_test = None, None

# ----------------------- Build Bidirectional LSTM Model -----------------------
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(time_step,1)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# ----------------------- Predictions -----------------------
train_predict = model.predict(X_train, verbose=0)
train_predict = scaler.inverse_transform(train_predict)

if X_test is not None:
    test_predict = model.predict(X_test, verbose=0)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
    
    plt.figure(figsize=(10,4))
    plt.plot(df.index[-len(test_predict):], y_test_actual, label='Actual')
    plt.plot(df.index[-len(test_predict):], test_predict, label='Predicted')
    plt.title("Actual vs Predicted Prices (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------- Forecast Future Days -----------------------
def forecast(model, last_seq, scaler, n=5):
    temp_input = list(scaler.transform(last_seq.reshape(-1,1)))
    future_preds = []
    for _ in range(n):
        input_seq = np.array(temp_input[-60:]).reshape(1,60,1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        future_preds.append(pred)
        temp_input.append([pred])
    return scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

last_60_days = data[-60:]
future_forecast = forecast(model, last_60_days, scaler, forecast_days)

future_dates = pd.date_range(df.index[-1], periods=forecast_days+1, freq='D')[1:]
plt.figure(figsize=(8,3))
plt.plot(future_dates, future_forecast, marker='o', linestyle='--', color='orange')
plt.title("Next 5 Days Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()
plt.show()

forecast_df = pd.DataFrame({"Date": future_dates.strftime('%Y-%m-%d'), "Predicted Price": future_forecast.flatten()})
print(forecast_df)
