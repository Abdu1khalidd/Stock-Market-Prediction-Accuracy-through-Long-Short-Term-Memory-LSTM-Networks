import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt



# Get stock symbol from user
stock_symbol = input("Enter the stock symbol: ")

# Download historical stock data
start_date = '2024-01-01'
end_date = '2024-08-06'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Extract closing prices
close_prices = data['Close'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# Create input data for LSTM
def create_lstm_data(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

time_steps = 10
x, y = create_lstm_data(close_prices_scaled, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(x, y, epochs=50, batch_size=32)

# Predict next 20 days
last_prices = close_prices[-time_steps:]
last_prices_scaled = scaler.transform(last_prices.reshape(-1, 1))
predicted_prices = []

for i in range(20):
    x_pred = np.array([last_prices_scaled[-time_steps:, 0]])
    x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
    predicted_price_scaled = model.predict(x_pred)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    predicted_prices.append(predicted_price[0][0])
    last_prices_scaled = np.vstack((last_prices_scaled, predicted_price_scaled))

# Generate future dates
future_dates = pd.date_range(start=end_date, periods=20)

# Create a DataFrame with predicted prices
future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})

# Print predicted prices
print(future_data)

# Plot predicted prices
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Actual Price')
plt.plot(future_data['Date'], future_data['Predicted Price'], label='Predicted Price')
plt.title('Actual and Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()