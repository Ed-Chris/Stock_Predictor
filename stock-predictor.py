import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import time
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Set the Streamlit page configuration
st.set_page_config(page_title="Stock Market Price Predictor", page_icon=":chart_with_upwards_trend:")

# Function to create a candlestick chart with Moving Averages and RSI
def plot_candlestick_chart(ticker_symbol, period):
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period=period)
    if not historical_data.empty:
        historical_data['MA20'] = historical_data['Close'].rolling(window=20).mean()
        historical_data['MA50'] = historical_data['Close'].rolling(window=50).mean()
        historical_data['RSI'] = ta.momentum.RSIIndicator(historical_data['Close'], window=14).rsi()
        historical_data.reset_index(inplace=True)
        fig = go.Figure(data=[
            go.Candlestick(x=historical_data['Date'],
                           open=historical_data['Open'], high=historical_data['High'],
                           low=historical_data['Low'], close=historical_data['Close'], name='Candlestick'),
            go.Scatter(x=historical_data['Date'], y=historical_data['MA20'], line=dict(color='blue', width=1), name='MA20'),
            go.Scatter(x=historical_data['Date'], y=historical_data['MA50'], line=dict(color='orange', width=1), name='MA50'),
            go.Scatter(x=historical_data['Date'], y=historical_data['RSI'], line=dict(color='green', width=1), name='RSI', yaxis='y2')
        ])
        fig.update_layout(title=f'{ticker_symbol} Candlestick Chart with MA20, MA50, and RSI',
                          xaxis_title='Date', yaxis_title='Price', yaxis2=dict(title='RSI', overlaying='y', side='right'), xaxis_rangeslider_visible=False)
        return fig
    else:
        st.error("No data found for ticker symbol: " + ticker_symbol)

def format_large_number(number):
    if abs(number) >= 1e9:
        return f"{number / 1e9:.2f} Billion"
    elif abs(number) >= 1e6:
        return f"{number / 1e6:.2f} Million"
    else:
        return f"{number:.2f}"

def predict_closing_price(user_ticker_input, data, prediction_window):
    start_time = time.time()
    if not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
        st.error('Data is not in the expected format.')
        return

    data = data.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    if len(scaled_data) < prediction_window:
        st.error(f"Not enough data to predict. Need at least {prediction_window} days of data.")
        return

    x_train = []
    y_train = []
    for i in range(prediction_window, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_window:i, 0])
        y_train.append(scaled_data[i, 0])

    if not x_train:
        st.error(f"Not enough data to predict. Need at least {prediction_window} days of data.")
        return

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=20)

    # Predict today's closing price
    test_data = scaled_data[-prediction_window:]
    test_data = test_data.reshape((1, prediction_window, 1))
    predicted_price = scaler.inverse_transform(model.predict(test_data))
    st.subheader(f"The predicted closing price for {user_ticker_input} today is: {predicted_price[0][0]:.2f}")

    # Prediction logic for the last 10 days
    if len(scaled_data) >= prediction_window + 10:
        predictions = []
        for i in range(10):
            input_seq = scaled_data[-prediction_window - 10 + i: -10 + i]
            input_seq = input_seq.reshape((1, prediction_window, 1))
            pred_scaled = model.predict(input_seq)
            pred = scaler.inverse_transform(pred_scaled)
            predictions.append(pred[0,0])
    else:
        predictions = [np.nan] * 10  # Not enough data, fill the predictions with NAs

    actuals_last_10_days = data['Close'].values[-10:]
    comparison_df = pd.DataFrame({
        'Date': data.index[-10:],
        'Actual Close': actuals_last_10_days,
        'Predicted Close': predictions
    })

    st.subheader("Actual vs Predicted Closing Prices for the Last 10 Days")
    st.table(comparison_df)
    # Plotting the results
    plt.figure(figsize=(14,7))
    plt.plot(data['Close'][-prediction_window-10:], label='Actual Close', color='blue')
    # Since the prediction includes the entire period, we align it with the last day of the actual close
    predicted_dates = data.index[-len(predictions):]
    plt.plot(predicted_dates, predictions, label='Predicted Close', color='orange', linestyle='--')
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(plt)
    end_time = time.time()  # End timing
    duration = end_time - start_time
    st.write(f"Prediction and plotting took {duration:.2f} seconds.")  # Display the timing



def main():
    st.title('Stock Market Price Predictor')
    user_ticker_input = st.sidebar.text_input("Enter the ticker symbol:", value='AAPL', max_chars=5)
    period_options = ['1d', '5d', '1mo', '2mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    period = st.sidebar.selectbox('Select Time Period:', options=period_options, index=2)
    if user_ticker_input:
        data = yf.Ticker(user_ticker_input).history(period=period)
        if not data.empty:
            fig = plot_candlestick_chart(user_ticker_input, period)
            st.plotly_chart(fig)
            info = yf.Ticker(user_ticker_input).info
            financial_info = {
                "High of the day": format_large_number(info.get('dayHigh', 0)),
                "Low of the day": format_large_number(info.get('dayLow', 0)),
                "Open Price": format_large_number(info.get('open', 0)),
                "Close Price": format_large_number(info.get('previousClose', 0)),
                "Volume": format_large_number(info.get('volume', 0)),
                "Market Cap": format_large_number(info.get('marketCap', 0)),
                "Shares Outstanding": format_large_number(info.get('sharesOutstanding', 0)),
                "Earnings Per Share (EPS)": format_large_number(info.get('trailingEps', 0)),
                "52 Week High": format_large_number(info.get('fiftyTwoWeekHigh', 0)),
                "52 Week Low": format_large_number(info.get('fiftyTwoWeekLow', 0))
            }
            st.table(pd.DataFrame(list(financial_info.items()), columns=["Metric", "Value"]))
            if st.sidebar.button('Predict Today\'s Closing Price'):
                prediction_window = 30
                if period in ['1d', '5d']:
                    prediction_window = 5  # Adjusted for shorter periods
                predict_closing_price(user_ticker_input, data, prediction_window)
        else:
            st.error("No data found for ticker symbol: " + user_ticker_input)

if __name__ == '__main__':
    main()

