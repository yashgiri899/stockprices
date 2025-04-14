import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf
import xgboost as xgb

def add_features(data):
    # Technical indicators
    # Moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    data['TR'] = np.maximum(
        data['High'] - data['Low'],
        np.maximum(
            abs(data['High'] - data['Close'].shift(1)),
            abs(data['Low'] - data['Close'].shift(1))
        )
    )
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    # Stochastic Oscillator
    data['14-high'] = data['High'].rolling(14).max()
    data['14-low'] = data['Low'].rolling(14).min()
    
    # Price Rate of Change
    data['ROC'] = data['Close'].pct_change(periods=12) * 100
    data['20D_STD'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA20'] + (data['20D_STD'] * 2)
    data['Lower_Band'] = data['MA20'] - (data['20D_STD'] * 2)
    
    # Momentum
    data['Momentum'] = data['Close'].pct_change(periods=5)
    
    # Volume features
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
    
    # Fill NaN values
    data.fillna(method='bfill', inplace=True)
    return data

# Fetching stock data
ticker = "AAPL"
start_date = "2018-01-12"
end_date = "2021-08-09"
data = yf.download(ticker, start=start_date, end=end_date)

# Prepare data with additional features
data = add_features(data)

# Select features for model input
features = ['Close', 'MA5', 'MA20', 'RSI', 'MACD', 'Signal', 
           'Upper_Band', 'Lower_Band', 'Momentum', 'Volume_Change',
           'ATR',  'ROC']
target = 'Close'

# Scale the data
feature_data = data[features].values
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaled_feature_data = scaler_X.fit_transform(feature_data)
scaled_target_data = scaler_y.fit_transform(data[[target]].values)

# Create sequences
X, y = [], []
sequence_length = 30

for i in range(sequence_length, len(scaled_feature_data)):
    X.append(scaled_feature_data[i-sequence_length:i])
    y.append(scaled_target_data[i])

X = np.array(X)
y = np.array(y)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

# Reshape data for XGBoost
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

# Initialize XGBoost model
# Initialize XGBoost model with eval_metric in parameters
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='rmse',  # Move eval_metric here
    random_state=42
)

# Train the model
xgb_model.fit(
    X_train_2d, 
    y_train,
    eval_set=[(X_train_2d, y_train), (X_test_2d, y_test)],
    verbose=True
)
# Make predictions
predicted_train = xgb_model.predict(X_train_2d)
predicted_test = xgb_model.predict(X_test_2d)

# Inverse transform the predicted values
predicted_train_price = scaler_y.inverse_transform(predicted_train.reshape(-1, 1))
actual_train_price = scaler_y.inverse_transform(y_train.reshape(-1, 1))
predicted_test_price = scaler_y.inverse_transform(predicted_test.reshape(-1, 1))
actual_test_price = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mae = mean_absolute_error(actual_test_price, predicted_test_price)
rmse = np.sqrt(mean_squared_error(actual_test_price, predicted_test_price))
r2 = r2_score(actual_test_price, predicted_test_price)

# Output results
print("📊 XGBoost Model Performance")
print(f"**MAE**: {mae:.2f}")
print(f"**RMSE**: {rmse:.2f}")
print(f"**R² Score**: {r2:.4f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(actual_test_price, label='Actual')
plt.plot(predicted_test_price, label='Predicted')
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.title(f"{ticker} Stock Prediction (Test Data)")
plt.legend()
plt.grid(True)
plt.show()

# Feature importance plot
feature_importance = xgb_model.feature_importances_
feature_names = [f"{feat}_{i}" for feat in features for i in range(sequence_length)]
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 6))
plt.bar(range(20), importance_df['importance'])
plt.xticks(range(20), importance_df['feature'], rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.show()