# Stock Price Prediction with Multiple Models

This project implements a comprehensive stock price prediction system using three different machine learning models: XGBoost, Prophet, and ARIMA. The system downloads Apple (AAPL) stock data, processes it, trains multiple models, and compares their performance.

## Table of Contents
1. [Overview](#overview)
2. [Installation & Dependencies](#installation--dependencies)
3. [Data Preparation](#data-preparation)
4. [Models Implemented](#models-implemented)
5. [Results & Comparison](#results--comparison)
6. [How to Use](#how-to-use)
7. [Key Concepts](#key-concepts)
8. [Future Improvements](#future-improvements)

## Overview

This project demonstrates three different approaches to time series forecasting for stock prices:
1. **XGBoost**: A gradient boosting model with feature engineering and hyperparameter optimization
2. **Prophet**: Facebook's time series forecasting library designed for business metrics
3. **ARIMA**: A classical statistical method for time series analysis

The system downloads 3 years of Apple stock data, processes it, trains all three models, and compares their performance using Mean Absolute Percentage Error (MAPE).

## Installation & Dependencies

To run this project, you'll need to install the following Python libraries:

```bash
pip install yfinance prophet scikit-learn optuna statsmodels xgboost matplotlib numpy pandas
```

The main dependencies are:
- `yfinance`: For downloading stock data from Yahoo Finance
- `prophet`: Facebook's time series forecasting library
- `scikit-learn`: For machine learning utilities and metrics
- `optuna`: For hyperparameter optimization
- `statsmodels`: For statistical models including ARIMA
- `xgboost`: For gradient boosting regression
- `matplotlib`, `numpy`, `pandas`: For data manipulation and visualization

## Data Preparation

The code downloads 3 years of Apple (AAPL) stock data from Yahoo Finance and processes it:

1. **Data Download**: Fetches Open, High, Low, Close, Volume data
2. **Column Processing**: Flattens MultiIndex columns and keeps only the Close price
3. **Feature Engineering**: Creates time-based features and lag features
4. **Train-Test Split**: Uses an 80-20 split for model evaluation

## Models Implemented

### 1. XGBoost with Hyperparameter Optimization

The XGBoost implementation includes:

- **Feature Engineering**:
  - Time-based features: dayofweek, quarter, month, year, dayofyear, dayofmonth
  - Lag features: 12 previous days of closing prices

- **Hyperparameter Tuning** with Optuna:
  - n_estimators: Number of trees (50-200)
  - max_depth: Tree depth (3-6)
  - learning_rate: Step size (0.01-0.1)
  - subsample: Fraction of samples used (0.6-1.0)
  - colsample_bytree: Fraction of features used (0.6-1.0)

- **Training**: Model is trained with optimal parameters found by Optuna

### 2. Prophet

Facebook's Prophet model is used for its strength in handling:

- **Seasonality**: Automatic detection of daily, weekly, and yearly patterns
- **Holiday Effects**: Incorporation of holiday impacts
- **Trend Changes**: Handling of trend shifts in time series

The model is trained on 80% of the data and evaluated on the remaining 20%.

### 3. ARIMA

The Autoregressive Integrated Moving Average model:

- **Stationarity Check**: Uses Augmented Dickey-Fuller test to verify stationarity
- **Model Order**: Uses (1, 1, 1) configuration:
  - p=1: One autoregressive term
  - d=1: First difference for stationarity
  - q=1: One moving average term
- **Forecasting**: Makes 30-day future predictions

## Results & Comparison

The models are evaluated using Mean Absolute Percentage Error (MAPE):

| Model | MAPE | Accuracy |
|-------|------|----------|
| XGBoost | 3.54% | 96.46% |
| Prophet | 27.44% | 72.56% |
| ARIMA | 0.67% | 99.33% |

Visualization includes:
- Individual model performance plots
- Combined forecast visualization comparing all three models
- 60 days of historical data for context

## How to Use

1. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook stock_prediction.ipynb
   ```

2. **Modify Parameters**:
   - Change the stock symbol: `yf.download('AAPL', period='3y')`
   - Adjust prediction horizon: `num_days_pred = 30`
   - Modify hyperparameter search spaces in the Optuna objective function

3. **Interpret Results**:
   - Check MAPE values for each model
   - Examine the visualization plots
   - Use the best-performing model for your specific needs

## Key Concepts

### Time Series Characteristics
- **Stationarity**: A time series is stationary if its statistical properties don't change over time
- **Seasonality**: Regular pattern of fluctuations that repeat over a fixed period
- **Autocorrelation**: Correlation of a signal with a delayed copy of itself

### Feature Engineering
- **Lag Features**: Previous values used as predictors for current values
- **DateTime Features**: Components of dates (day of week, month, etc.) that capture patterns
- **Rolling Statistics**: Moving averages and standard deviations

### Model Selection Considerations
- **XGBoost**: Powerful for capturing complex patterns but requires careful feature engineering
- **Prophet**: Handles seasonality well and provides uncertainty intervals
- **ARIMA**: Classical approach, good for stationary series with clear patterns

## Future Improvements

1. **Additional Models**:
   - LSTM/GRU neural networks
   - Ensemble methods combining predictions from multiple models

2. **Enhanced Feature Engineering**:
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Sentiment analysis from news sources
   - Market volatility measures

3. **Hyperparameter Optimization**:
   - More extensive search spaces
   - Bayesian optimization techniques
   - Multi-objective optimization considering both accuracy and complexity

4. **Risk Management**:
   - Confidence intervals for predictions
   - Risk-adjusted performance metrics
   - Portfolio optimization integration

5. **Deployment**:
   - Web application for interactive forecasting
   - API endpoints for model inference
   - Automated retraining pipelines

# Stock Price Prediction with Multiple Models

This Jupyter notebook implements a comprehensive stock price prediction system using three different machine learning models: XGBoost, Prophet, and ARIMA. Let's break down each section in detail:

## 1. Import Libraries and Setup

```python
import datetime
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yfinance as yf

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
```

This section imports all necessary libraries:
- **yfinance**: For downloading stock data from Yahoo Finance
- **prophet**: Facebook's time series forecasting library
- **sklearn**: For machine learning utilities and metrics
- **statsmodels**: For statistical models including ARIMA
- **xgboost**: For gradient boosting regression
- **optuna**: For hyperparameter optimization

Warnings are suppressed to keep the output clean, and ggplot style is used for plots.

## 2. Data Download and Preparation

```python
stock_data = yf.download('AAPL', period='3y')  # Get 3 years of Apple stock data
num_days_pred = 30  # Number of days to predict in the future
```

The code downloads 3 years of Apple (AAPL) stock data and sets the prediction horizon to 30 days.

```python
# Fix MultiIndex columns and keep only Close price
stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
stock_data = stock_data[['Close_AAPL']].copy()
stock_data.rename(columns={'Close_AAPL': 'Close'}, inplace=True)
```

Yahoo Finance returns MultiIndex columns, so we flatten them and keep only the closing price, which is typically the most important for prediction.

## 3. Utility Functions

### MAPE Calculation
```python
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones_like(y_true), np.abs(y_true)))) * 100
```

Mean Absolute Percentage Error (MAPE) is a useful metric for regression problems as it expresses accuracy as a percentage.

### Feature Engineering Functions
```python
def add_lags(df, num_lags=12):
    """Add lag features with a reasonable number of lags"""
    # ... creates lag features for time series

def create_features(df):
    """Create time series features based on time series index"""
    # ... creates datetime features like dayofweek, month, etc.
```

These functions create additional features for the models:
- **Lag features**: Previous values of the time series
- **Datetime features**: Day of week, month, year, etc. to capture seasonality

### Stationarity Check
```python
def check_stationarity(timeseries):
    """Check if a time series is stationary using Augmented Dickey-Fuller test"""
    # ... performs ADF test
```

Stationarity is an important assumption for many time series models. The Augmented Dickey-Fuller test checks if a time series is stationary.

## 4. XGBoost Model Implementation

### Data Preparation
```python
def prepare_xgboost_data(df_xgb, add_lags_func, create_features_func):
    # Applies feature engineering and returns features (X) and target (y)
```

Prepares the data for XGBoost by creating features and splitting into X (features) and y (target).

### Hyperparameter Optimization
```python
def objective(trial):
    # Defines hyperparameter search space for Optuna
```

Uses Optuna for automated hyperparameter tuning. The search space includes:
- n_estimators: Number of trees
- max_depth: Maximum tree depth
- learning_rate: Step size shrinkage
- subsample: Subsample ratio of training instances
- colsample_bytree: Subsample ratio of columns

### Model Training and Evaluation
The XGBoost model is trained with the best parameters found by Optuna and evaluated on a test set.

## 5. Prophet Model Implementation

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

```python
# Format data for Prophet (requires 'ds' and 'y' columns)
train_prophet = train.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

# Create and fit model
prophet = Prophet()
prophet.fit(train_prophet)

# Make predictions
future = prophet.make_future_dataframe(periods=num_days_pred, freq='D', include_history=False)
forecast = prophet.predict(future)
```

## 6. ARIMA Model Implementation

ARIMA (AutoRegressive Integrated Moving Average) is a classical time series forecasting method.

```python
# Check stationarity first
is_stationary = check_stationarity(df_arima['Close'])

# Fit ARIMA model
arima = ARIMA(df_arima['Close'], order=(1, 1, 1))  # (p, d, q) parameters
arima_fit = arima.fit()

# Make forecasts
arima_forecast = arima_fit.forecast(steps=num_days_pred)
```

The (1, 1, 1) order means:
- p=1: One autoregressive term
- d=1: First difference for stationarity
- q=1: One moving average term

## 7. Model Comparison and Visualization

The code compares all three models using MAPE and creates a visualization showing:
- Historical data (last 60 days)
- Future predictions from all three models

## Key Concepts in Time Series Forecasting

1. **Stationarity**: A time series is stationary if its statistical properties don't change over time
2. **Seasonality**: Regular pattern of fluctuations that repeat over a fixed period
3. **Autocorrelation**: Correlation of a signal with a delayed copy of itself
4. **Lag Features**: Previous values used as predictors for current values
5. **Hyperparameter Tuning**: Optimizing model parameters for better performance

## Model Strengths and Weaknesses

- **XGBoost**: Powerful for capturing complex patterns but requires careful feature engineering
- **Prophet**: Handles seasonality well and provides uncertainty intervals
- **ARIMA**: Classical approach, good for stationary series with clear patterns

This implementation provides a solid foundation for stock price prediction that can be extended with more sophisticated features, additional models, or ensemble approaches.
