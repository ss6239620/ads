import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv("7.csv")

# Assuming the time columns start from index 11
date_columns = df.columns[11:]

# Sum up all the columns to get the total number of cases
time_series = df[date_columns].sum(axis=0)

# Convert the index to datetime format
time_series.index = pd.to_datetime(time_series.index)

# Check for missing values
if time_series.isnull().sum() > 0:
    print("Missing values detected! Filling missing values...")
    time_series = time_series.fillna(method='ffill')  # Forward fill missing values

# Plot the time series
def plot_series(series, title="Time Series"):
    plt.figure(figsize=(12, 5))
    plt.plot(series, label="Total Cases")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Total Cases")
    plt.legend()
    plt.show()

plot_series(time_series, title="COVID-19 Cases Over Time")

# Differencing the series to make it stationary
diff_series = time_series.diff().dropna()

# Fit ARIMA model with simpler parameters
try:
    model = ARIMA(time_series, order=(1, 1, 1))  # Try simpler ARIMA parameters (p=1, d=1, q=1)
    model_fit = model.fit()

    # Print model summary
    print(model_fit.summary())

    # Forecast the next 30 days
    forecast = model_fit.forecast(steps=30)

    # Plot the forecasted values
    def plot_forecast(series, forecast):
        plt.figure(figsize=(12, 5))
        plt.plot(series, label="Actual Cases")
        plt.plot(pd.date_range(start=series.index[-1], periods=30, freq="D"), forecast, label="Forecast", color="red")
        plt.title("COVID-19 Cases Forecast")
        plt.xlabel("Date")
        plt.ylabel("Total Cases")
        plt.legend()
        plt.show()

    plot_forecast(time_series, forecast)

except Exception as e:
    print(f"Error during ARIMA fitting: {e}")
