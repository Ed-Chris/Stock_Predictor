# Stock Predicting Application Using Deep Learning LSTM Modelling

## Project Overview
This project aims to develop a stock prediction application using deep learning Long-Short Term Memory (LSTM) modeling. The primary objective is to assist traders and investors in predicting stock prices and enhancing their returns by providing a user-friendly, dynamic front-end website with real-time stock price predictions.

## Table of Contents
- Introduction.
- Objective.
- Features.
- Technologies Used.
- Data Pipeline.
- Machine Learning Model.
- Limitations and Future Work.
- How to Run the Project.
- Contributors.
- References.

## Introduction
Financial literacy and investments in the stock market are crucial for building long-term wealth. This project focuses on creating a tool that simplifies stock trading decisions using machine learning to predict stock prices, thus helping investors and traders optimize their returns.

## Objective
The main goal is to develop a user-friendly web application that:
- Provides real-time stock price predictions.
- Displays comprehensive stock data, including moving averages and RSI.
- Utilizes a deep learning LSTM model for accurate stock price forecasting.
 
## Features
- Real-Time Stock Data: Access real-time stock prices and predictions for various companies.
- Interactive Charts: Display candlestick charts for different periods (1 day to 10 years) with moving averages and RSI indicators.
- Financial Information: View additional financial data such as opening price, closing price, volume, market cap, and more.
- Stock Price Prediction: Predict end-of-day stock prices using an LSTM model.

## Technologies Used
- Python: Programming language for developing the backend.
- Streamlit: Framework for building the front-end application.
- Docker: Containerization tool for deploying the application.
- AWS: Cloud services (EC2, ECS, Fargate, DynamoDB) for hosting and data storage.
- Yahoo Finance API: For fetching live stock data.
- Pandas: Data manipulation library.
- TensorFlow: Machine learning library for building the LSTM model.

## Data Pipeline
- **Data Collection:** Use Yahoo Finance API to fetch live stock data.
- **Data Transformation:** Convert data into a structured format using Pandas.
- **Model Training:** Train the LSTM model using historical stock data.
- **Prediction:** Predict stock prices using the trained model.
- **Deployment:** Deploy the application using Docker and AWS services.

## Machine Learning Model
The LSTM model is used for predicting stock prices. It:
- Collects historical closing prices for the selected stock.
- Normalizes the data and creates sequences for time series analysis.
- Identifies patterns in stock price movements.
- Predicts the next closing price based on the most recent data.

## Limitations and Future Work

**Current Limitations:**
- Limited variables in the model.
- Predictions are day-to-day rather than minute-by-minute.
- Reduced accuracy due to a shorter data window for faster computation.

***Future Improvements:**
- Incorporate more factors like stock sentiment and global news.
- Enhance prediction accuracy and reduce computation time.
- Explore methods for real-time second-by-second stock price updates.

## How to Run the Project

### Clone the Repository:
git clone <repository_url>

### Build Docker Image:
docker build -t stock-predictor.

### Run Docker Container:
docker run -p 8501:8501 stock-predictor

### Access the Application:
Open your browser and go to http://localhost:8501.

## Contributors

Yedu Krishnan


## References
J. R. Ph.D, "Stocks Vs. ETFs: Which Should You Invest In? | Bankrate," Bankrate Press. Available: Bankrate
