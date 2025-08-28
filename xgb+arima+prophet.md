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

This implementation provides a solid foundation for stock price prediction that can be extended with more sophisticated features, additional models, or ensemble approaches.
