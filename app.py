import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima.arima import auto_arima
import joblib
from tensorflow.keras.models import load_model

# Function to calculate moving average
def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Function to forecast next 7 days' stock prices using Keras model
def forecast_next_7_days_knn(data):
    try:
        knn_model = load_model('knn_model.keras')  # Ensure this is the correct path to your LSTM model
        last_100_days = data['Close'][-100:].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_last_100_days = scaler.fit_transform(last_100_days)
        x_pred = scaled_last_100_days.reshape(1, 100, 1)
        forecasts = []
        for _ in range(7):
            next_day_pred = knn_model.predict(x_pred)[0, 0]
            forecasts.append(next_day_pred)
            x_pred = np.roll(x_pred, -1)
            x_pred[0, -1, 0] = next_day_pred
        forecasts = np.array(forecasts).reshape(-1, 1)
        return scaler.inverse_transform(forecasts).flatten()
    except Exception as e:
        st.error(f"Error in forecasting using Keras model: {str(e)}")
        return []

# Function to forecast next 7 days' stock prices using Linear Regression model
def forecast_next_7_days_linear_regression(data):
    try:
        linear_regression_model = joblib.load('linear_regression_model.pkl')  # Ensure this is the correct path to your LR model
        # Use the last 100 days for prediction as the model expects 100 features
        last_100_days = data['Close'][-100:].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_last_100_days = scaler.fit_transform(last_100_days)
        x_pred = scaled_last_100_days.reshape(1, 100)
        
        forecasts = []
        for _ in range(7):
            next_day_pred = linear_regression_model.predict(x_pred)[0]
            forecasts.append(next_day_pred)
            x_pred = np.roll(x_pred, -1)
            x_pred[0, -1] = next_day_pred
        
        forecasts = np.array(forecasts).reshape(-1, 1)
        return scaler.inverse_transform(forecasts).flatten()
    except Exception as e:
        st.error(f"Error in forecasting using Linear Regression model: {str(e)}")
        return []

# Function to forecast next 7 days' stock prices using Random Forest model
def forecast_next_7_days_random_forest(data):
    try:
        random_forest_model = joblib.load('random_forest_model.pkl')  # Ensure this is the correct path to your RF model
        # Use the last 100 days for prediction as the model expects 100 features
        last_100_days = data['Close'][-100:].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_last_100_days = scaler.fit_transform(last_100_days)
        x_pred = scaled_last_100_days.reshape(1, 100)
        
        forecasts = []
        for _ in range(7):
            next_day_pred = random_forest_model.predict(x_pred)[0]
            forecasts.append(next_day_pred)
            x_pred = np.roll(x_pred, -1)
            x_pred[0, -1] = next_day_pred
        
        forecasts = np.array(forecasts).reshape(-1, 1)
        return scaler.inverse_transform(forecasts).flatten()
    except Exception as e:
        st.error(f"Error in forecasting using Random Forest model: {str(e)}")
        return []

# Function to forecast next 7 days' stock prices using ARIMA model
def forecast_next_7_days_arima(data):
    try:
        # Train the ARIMA model with the given data
        arima_model = auto_arima(data['Close'], seasonal=False, trace=True)
        forecast = arima_model.predict(n_periods=7)
        return forecast
    except Exception as e:
        st.error(f"Error in forecasting using ARIMA model: {str(e)}")
        return []

# Streamlit UI
st.title('Stock Market Predictor')

# Sidebar: Input parameters
st.sidebar.subheader('Input Parameters')
stock = st.sidebar.text_input('Enter Stock Symbol', 'MSFT')
start_date = st.sidebar.date_input('Select Start Date', pd.to_datetime('2000-01-01'))
end_date = st.sidebar.date_input('Select End Date', pd.to_datetime('today'))

# Fetch stock data
data = yf.download(stock, start=start_date, end=end_date)
data.reset_index(inplace=True)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Calculate moving averages
ma_100_days = calculate_moving_average(data['Close'], 100)
ma_200_days = calculate_moving_average(data['Close'], 200)

# Plot moving averages
st.subheader('Moving Average Plots')
fig_ma100 = go.Figure()
fig_ma100.add_trace(go.Scatter(x=data['Date'], y=ma_100_days, mode='lines', name='MA100'))
fig_ma100.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
fig_ma100.update_layout(title='Price vs MA100', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma100)

fig_ma200 = go.Figure()
fig_ma200.add_trace(go.Scatter(x=data['Date'], y=ma_100_days, mode='lines', name='MA100', line=dict(color='red')))
fig_ma200.add_trace(go.Scatter(x=data['Date'], y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue')))
fig_ma200.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='green')))
fig_ma200.update_layout(title='Price vs MA100 vs MA200', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma200)

# Machine Learning Model Selection
ml_models = {
    'Linear Regression Model': forecast_next_7_days_linear_regression,
    'ARIMA Model': forecast_next_7_days_arima,
    'KNN Model': forecast_next_7_days_knn,
    'Random Forest Model': forecast_next_7_days_random_forest,
}

selected_model = st.selectbox('Select Model', list(ml_models.keys()))

# Forecast next 7 days' stock prices
forecasted_prices = ml_models[selected_model](data)

# Display forecasted prices
st.subheader('Next 7 Days Forecasted Close Prices')

# Debugging: Check the lengths of forecast_dates and forecasted_prices
forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)

# Ensure the lengths of forecast_dates and forecasted_prices match
if len(forecast_dates) == len(forecasted_prices):
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Close Price': forecasted_prices})
    st.write(forecast_df)
else:
    st.error("Error: The lengths of forecast_dates and forecasted_prices do not match.")
