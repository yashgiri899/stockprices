
# 📈 Stock Price Prediction using XGBoost & Technical Indicators

This project demonstrates how to predict stock prices (using Apple Inc. - AAPL as an example) with **machine learning** and **technical analysis features**. The model uses **XGBoost** on engineered time-series data to predict future closing prices.

## 🚀 Project Highlights

- Historical stock data fetched using `yfinance`
- Feature engineering with:
  - Moving Averages (MA5, MA20)
  - RSI, MACD, ATR, Bollinger Bands
  - Momentum, ROC, Volume Trends
- Time-series framing using 30-day sliding windows
- Data normalization with `MinMaxScaler`
- Trained XGBoost Regressor on sequence-transformed input
- Evaluation with MAE, RMSE, and R² Score
- Visualizations:
  - Predicted vs Actual Price
  - Top 20 Important Features

---

## 🔧 Tech Stack

- Python
- pandas, numpy
- matplotlib
- scikit-learn
- xgboost
- yfinance

---

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yashgiri899/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python stock_prediction_xgboost.py
   ```

---


## 🤝 Let’s Connect

If you’re interested in machine learning for finance, time-series modeling, or feature engineering — feel free to connect!

---

## 📌 License

This project is open-source and free to use under the MIT License.
