import streamlit as st
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

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        background-color: #f8f9fa;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Function to calculate mean absolute error percentage
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones_like(y_true), np.abs(y_true)))) * 100

def add_lags(df, num_lags=12):
    """Add lag features with a reasonable number of lags"""
    target = 'Close'
    max_possible_lags = min(12, len(df) // 10)
    for i in range(1, max_possible_lags + 1):
        df[f'lag{i}'] = df[target].shift(i)
    return df

def create_features(df):
    """Create time series features based on time series index"""
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    return df

def check_stationarity(timeseries):
    """Check if a time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    return result[1] <= 0.05  # Return True if stationary

def prepare_xgboost_data(df_xgb, add_lags_func, create_features_func):
    df_xgb = create_features_func(df_xgb)
    df_xgb = add_lags_func(df_xgb)
    df_xgb.dropna(inplace=True)
    X = df_xgb.drop(columns='Close')
    y = df_xgb['Close']
    return X, y

# Main app
def main():
    st.title("📈 Stock Price Prediction Dashboard")
    st.markdown("Predict stock prices using XGBoost, Prophet, and ARIMA models")
    
    # Sidebar
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    period = st.sidebar.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=2)
    num_days_pred = st.sidebar.slider("Days to Predict", 7, 90, 30)
    
    # Download data
    with st.spinner("Downloading stock data..."):
        stock_data = yf.download(ticker, period=period)
    
    if stock_data.empty:
        st.error(f"Could not download data for {ticker}. Please check the ticker symbol.")
        return
    
    # Process data
    stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
    stock_data = stock_data[['Close_' + ticker]].copy()
    stock_data.rename(columns={'Close_' + ticker: 'Close'}, inplace=True)
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Points", len(stock_data))
    with col2:
        st.metric("Start Date", stock_data.index[0].strftime('%Y-%m-%d'))
    with col3:
        st.metric("End Date", stock_data.index[-1].strftime('%Y-%m-%d'))
    
    # Plot historical data
    st.subheader("Historical Price Data")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price')
    ax.set_title(f'{ticker} Historical Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)
    st.pyplot(fig)
    
    # Model selection
    st.sidebar.header("Model Selection")
    use_xgboost = st.sidebar.checkbox("XGBoost", value=True)
    use_prophet = st.sidebar.checkbox("Prophet", value=True)
    use_arima = st.sidebar.checkbox("ARIMA", value=True)
    
    if not (use_xgboost or use_prophet or use_arima):
        st.warning("Please select at least one model to run.")
        return
    
    # Initialize results
    results = {}
    forecasts = {}
    
    # XGBoost Model
    if use_xgboost:
        with st.spinner("Training XGBoost model..."):
            try:
                df_xgb = stock_data.copy()
                X, y = prepare_xgboost_data(df_xgb, add_lags, create_features)
                
                if len(X) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, shuffle=False
                    )
                    
                    # Hyperparameter optimization
                    def objective(trial):
                        param = {
                            'objective': 'reg:squarederror',
                            'eval_metric': 'rmse',
                            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                            'max_depth': trial.suggest_int('max_depth', 3, 6),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                            'verbosity': 0,
                        }
                        xgb = XGBRegressor(**param)
                        xgb.fit(X_train, y_train)
                        y_pred = xgb.predict(X_test)
                        return np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=5)
                    
                    best_params = study.best_trial.params
                    xgb_best = XGBRegressor(**best_params)
                    xgb_best.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_test = xgb_best.predict(X_test)
                    xgb_loss = mean_absolute_percentage_error(y_test, y_pred_test)
                    
                    # Create future predictions
                    last_date = df_xgb.index[-1]
                    future_dates = [last_date + pd.Timedelta(days=x) for x in range(1, num_days_pred+1)]
                    future_df = pd.DataFrame(index=future_dates)
                    future_df = create_features(future_df)
                    
                    for i in range(1, 13):
                        if i <= len(df_xgb):
                            future_df[f'lag{i}'] = df_xgb['Close'].iloc[-i]
                        else:
                            future_df[f'lag{i}'] = np.nan
                    
                    prediction_xgb = pd.DataFrame({
                        'pred': xgb_best.predict(future_df)
                    }, index=future_dates)
                    
                    results['XGBoost'] = xgb_loss
                    forecasts['XGBoost'] = prediction_xgb
                    
                    # Display results
                    with st.expander("XGBoost Results", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("XGBoost MAPE", f"{xgb_loss:.2f}%")
                        with col2:
                            st.metric("XGBoost Accuracy", f"{100 - xgb_loss:.2f}%")
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.scatter(X_test.index, y_test, color='blue', label='Actual', alpha=0.6)
                        ax.scatter(X_test.index, y_pred_test, color='red', label='Predicted', alpha=0.6)
                        ax.set_title('XGBoost: Actual vs Predicted Values')
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Price ($)')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                
                else:
                    st.warning("Not enough data for XGBoost training.")
            
            except Exception as e:
                st.error(f"XGBoost model failed: {str(e)}")
    
    # Prophet Model
    if use_prophet:
        with st.spinner("Training Prophet model..."):
            try:
                df_prophet = stock_data.copy()
                split_idx = int(len(df_prophet) * 0.8)
                train = df_prophet.iloc[:split_idx].copy()
                test = df_prophet.iloc[split_idx:].copy()
                
                train_prophet = train.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
                prophet_model = Prophet()
                prophet_model.fit(train_prophet)
                
                test_prophet = test.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
                test_predict = prophet_model.predict(test_prophet)
                
                prophet_loss = mean_absolute_percentage_error(test['Close'], test_predict['yhat'])
                
                # Future predictions
                prophet_data = df_prophet.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
                prophet_model_full = Prophet()
                prophet_model_full.fit(prophet_data)
                
                future = prophet_model_full.make_future_dataframe(periods=num_days_pred, freq='D', include_history=False)
                forecast = prophet_model_full.predict(future)
                forecast_prophet = forecast[['ds', 'yhat']].set_index('ds')
                
                results['Prophet'] = prophet_loss
                forecasts['Prophet'] = forecast_prophet
                
                # Display results
                with st.expander("Prophet Results", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prophet MAPE", f"{prophet_loss:.2f}%")
                    with col2:
                        st.metric("Prophet Accuracy", f"{100 - prophet_loss:.2f}%")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.scatter(test.index, test['Close'], color='blue', label='Actual', alpha=0.6)
                    ax.scatter(test.index, test_predict['yhat'], color='red', label='Predicted', alpha=0.6)
                    ax.set_title('Prophet: Actual vs Predicted Values')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Price ($)')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Prophet model failed: {str(e)}")
    
    # ARIMA Model
    if use_arima:
        with st.spinner("Training ARIMA model..."):
            try:
                df_arima = stock_data.copy()
                is_stationary = check_stationarity(df_arima['Close'])
                
                if not is_stationary:
                    st.info("Data is not stationary. Using differencing in ARIMA model.")
                
                arima = ARIMA(df_arima['Close'], order=(1, 1, 1))
                arima_fit = arima.fit()
                
                # Forecast
                arima_forecast = arima_fit.forecast(steps=num_days_pred)
                
                # Create index for forecast
                start_date = df_arima.index[-1] + pd.Timedelta(days=1)
                forecast_dates = pd.date_range(start=start_date, periods=num_days_pred, freq='D')
                arima_forecast_df = pd.DataFrame(arima_forecast, index=forecast_dates, columns=['Close'])
                
                # Calculate error if we have test data
                if len(df_arima) > 10:
                    test_arima = df_arima['Close'].iloc[-10:]
                    pred_arima = arima_fit.predict(start=len(df_arima)-10, end=len(df_arima)-1)
                    arima_loss = mean_absolute_percentage_error(test_arima, pred_arima)
                else:
                    arima_loss = float('inf')
                
                results['ARIMA'] = arima_loss
                forecasts['ARIMA'] = arima_forecast_df
                
                # Display results
                with st.expander("ARIMA Results", expanded=True):
                    if arima_loss != float('inf'):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("ARIMA MAPE", f"{arima_loss:.2f}%")
                        with col2:
                            st.metric("ARIMA Accuracy", f"{100 - arima_loss:.2f}%")
                    else:
                        st.info("ARIMA model trained but could not calculate accuracy metrics.")
            
            except Exception as e:
                st.error(f"ARIMA model failed: {str(e)}")
                # Create a simple baseline forecast
                last_value = df_arima['Close'].iloc[-1]
                start_date = df_arima.index[-1] + pd.Timedelta(days=1)
                forecast_dates = pd.date_range(start=start_date, periods=num_days_pred, freq='D')
                arima_forecast_df = pd.DataFrame([last_value] * num_days_pred, 
                                                index=forecast_dates, columns=['Close'])
                forecasts['ARIMA'] = arima_forecast_df
                results['ARIMA'] = float('inf')
    
    # Display comparison
    if results:
        st.subheader("Model Comparison")
        
        # Create comparison table
        comparison_data = []
        for model, mape in results.items():
            if mape != float('inf'):
                comparison_data.append({
                    'Model': model,
                    'MAPE': f"{mape:.2f}%",
                    'Accuracy': f"{100 - mape:.2f}%"
                })
        
        if comparison_data:
            st.table(pd.DataFrame(comparison_data))
        
        # Plot all predictions
        st.subheader(f"Future Price Predictions ({num_days_pred} days)")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(stock_data.index[-60:], stock_data['Close'][-60:], 'k-', label='Historical Data', linewidth=2)
        
        # Plot predictions
        colors = ['blue', 'red', 'green']
        for i, (model, forecast) in enumerate(forecasts.items()):
            if model in results and results[model] != float('inf'):
                color = colors[i % len(colors)]
                if model == 'XGBoost':
                    ax.plot(forecast.index, forecast['pred'], color=color, label=f'{model} Prediction', linestyle='--')
                elif model == 'Prophet':
                    ax.plot(forecast.index, forecast['yhat'], color=color, label=f'{model} Prediction', linestyle='--')
                elif model == 'ARIMA':
                    ax.plot(forecast.index, forecast['Close'], color=color, label=f'{model} Prediction', linestyle='--')
        
        ax.set_title(f'{ticker} Price Predictions for Next {num_days_pred} Days')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Show prediction values
        st.subheader("Prediction Values")
        prediction_df = pd.DataFrame()
        for model, forecast in forecasts.items():
            if model == 'XGBoost':
                prediction_df[f'{model}_Prediction'] = forecast['pred']
            elif model == 'Prophet':
                prediction_df[f'{model}_Prediction'] = forecast['yhat']
            elif model == 'ARIMA':
                prediction_df[f'{model}_Prediction'] = forecast['Close']
        
        st.dataframe(prediction_df.style.format("{:.2f}"))
        
        # Download predictions
        csv = prediction_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv",
        )
    else:
        st.warning("No models were successfully trained. Please check the configuration.")

if __name__ == "__main__":
    main()