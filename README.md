# Stock_Price_Forecasting_BIST30_NASDAQ30
Stock_Price_Forecasting_BIST30_NASDAQ30
# Stock Price Forecasting: BIST30 & NASDAQ30

## 📊 Project Overview

This project demonstrates how to collect, analyze, and forecast stock prices for top companies in BIST30 and NASDAQ30 using Python and machine learning techniques. The goal is to showcase time series forecasting skills relevant for data analyst roles.

## 🔍 Features

- Pulls 2 years of daily historical data using `yfinance`
- Performs exploratory data analysis and visualization
- Applies ARIMA model for time series forecasting
- Handles missing data, resampling, and stationarity checks

## 🧰 Tech Stack

- Python 3
- pandas, matplotlib, seaborn
- yfinance
- statsmodels (for ARIMA)

## 🏁 Getting Started

```bash
pip install yfinance pandas matplotlib seaborn statsmodels
python get_data.py



---

#### 2. `get_data.py`

```python
import yfinance as yf

# BIST30 ve NASDAQ30'dan örnek semboller
symbols = {
    "BIST30": ["AKBNK.IS", "THYAO.IS", "ASELS.IS"],
    "NASDAQ30": ["AAPL", "MSFT", "NVDA"]
}

def download_data(symbols, start="2022-01-01", end="2024-12-31"):
    for group, tickers in symbols.items():
        for ticker in tickers:
            df = yf.download(ticker, start=start, end=end)
            df.to_csv(f"{ticker}_data.csv")
            print(f"Downloaded {ticker}")

if __name__ == "__main__":
    download_data(symbols)


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Örnek veri: Apple
df = pd.read_csv('AAPL_data.csv', index_col='Date', parse_dates=True)
df = df['Close']

# Görselleştir
df.plot(title='AAPL Closing Prices', figsize=(12,5))
plt.grid()
plt.show()

# ARIMA modeli
model = ARIMA(df, order=(5,1,0))
model_fit = model.fit()

# Tahmin
forecast = model_fit.forecast(steps=30)
forecast.plot(title='30-Day Forecast', figsize=(12,5))
plt.grid()
plt.show()
