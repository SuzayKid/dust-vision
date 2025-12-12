"""
PM (Particulate Matter) Forecasting Pipeline
============================================
A complete ML pipeline for predicting PM levels and triggering spray mitigation.
Designed for dashboard integration with adjustable forecast horizons.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import LSTM components (optional)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("TensorFlow not available. LSTM model will be disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the entire pipeline"""
    
    # Data settings
    RESAMPLE_FREQ = '5min'  # 5-minute intervals
    PM_COLUMN = 'PM2.5'  # Target column name (adjust based on your data)
    
    # Feature engineering
    LAG_PERIODS = [1, 2, 3, 6, 12]  # Lag periods in 5-min intervals
    ROLLING_WINDOWS = [6, 12, 24]  # Rolling windows (30min, 1hr, 2hr)
    DUST_THRESHOLD = 50  # PM value above which we consider it a dust event
    
    # Model settings
    DEFAULT_HORIZON = 30  # Default forecast horizon in minutes
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    
    # Spray logic
    SPRAY_THRESHOLD = 75  # PM value that triggers spray recommendation
    MITIGATION_FACTOR = 0.3  # 30% reduction in PM after spray


# ============================================================================
# 1. DATA INTAKE + PRE-PROCESSING MODULE
# ============================================================================

def prepare_data(csv_path, horizon=30):
    """
    Load and preprocess data into ML-ready format.
    
    Args:
        csv_path (str): Path to the clean CSV file
        horizon (int): Forecast horizon in minutes (for validation split)
    
    Returns:
        pd.DataFrame: Processed dataframe with 5-min frequency
    """
    print(f"📥 Loading data from {csv_path}...")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Convert timestamp column to datetime (adjust column name as needed)
    timestamp_col = df.columns[0]  # Assuming first column is timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    
    # Resample to 5-minute frequency with forward fill then interpolation
    print(f"🔄 Resampling to {Config.RESAMPLE_FREQ} frequency...")
    df_resampled = df.resample(Config.RESAMPLE_FREQ).mean()
    
    # Interpolate missing values (linear interpolation)
    df_resampled = df_resampled.interpolate(method='linear', limit_direction='both')
    
    # Fill any remaining NaNs with forward fill
    df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✅ Data prepared: {len(df_resampled)} rows at {Config.RESAMPLE_FREQ} intervals")
    
    return df_resampled


# ============================================================================
# 2. FEATURE ENGINEERING PIPELINE
# ============================================================================

def make_features(df, horizon=30):
    """
    Generate ML features from the processed dataframe.
    Parameterized so dashboard can adjust horizon dynamically.
    
    Args:
        df (pd.DataFrame): Processed dataframe from prepare_data()
        horizon (int): Forecast horizon in minutes
    
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    print(f"🔧 Engineering features for {horizon}-minute horizon...")
    
    df_feat = df.copy()
    
    # --- LAG FEATURES (based on 5-min intervals) ---
    for lag in Config.LAG_PERIODS:
        df_feat[f'PM_lag_{lag}'] = df_feat[Config.PM_COLUMN].shift(lag)
    
    # --- ROLLING WINDOW STATISTICS ---
    for window in Config.ROLLING_WINDOWS:
        df_feat[f'PM_rolling_mean_{window}'] = df_feat[Config.PM_COLUMN].rolling(window).mean()
        df_feat[f'PM_rolling_std_{window}'] = df_feat[Config.PM_COLUMN].rolling(window).std()
        df_feat[f'PM_rolling_max_{window}'] = df_feat[Config.PM_COLUMN].rolling(window).max()
    
    # --- WIND COMPONENTS (if wind speed and direction available) ---
    if 'wind_speed' in df_feat.columns and 'wind_direction' in df_feat.columns:
        # Convert wind to u (east-west) and v (north-south) components
        wind_rad = np.deg2rad(df_feat['wind_direction'])
        df_feat['wind_u'] = df_feat['wind_speed'] * np.sin(wind_rad)
        df_feat['wind_v'] = df_feat['wind_speed'] * np.cos(wind_rad)
    
    # --- DUST EVENT FLAGS ---
    df_feat['is_dust_event'] = (df_feat[Config.PM_COLUMN] > Config.DUST_THRESHOLD).astype(int)
    
    # Count dust events in last 30 minutes (6 periods of 5-min)
    df_feat['dust_event_count_30min'] = df_feat['is_dust_event'].rolling(6).sum()
    
    # --- TIME-BASED FEATURES ---
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    
    # Drop rows with NaN values created by lag/rolling operations
    df_feat = df_feat.dropna()
    
    print(f"✅ Features created: {df_feat.shape[1]} total columns")
    
    return df_feat


# ============================================================================
# 3. FORECASTING MODELS
# ============================================================================

class PMForecaster:
    """
    Unified forecasting interface supporting multiple model types.
    """
    
    def __init__(self, model_type='rf'):
        """
        Initialize forecaster with specified model type.
        
        Args:
            model_type (str): 'rf' for Random Forest, 'lstm' for LSTM
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def _prepare_rf_data(self, df_features, target_col=Config.PM_COLUMN):
        """Prepare data for Random Forest model"""
        # Use all columns except target as features
        self.feature_cols = [col for col in df_features.columns if col != target_col]
        
        X = df_features[self.feature_cols].values
        y = df_features[target_col].values
        
        return X, y
    
    def _prepare_lstm_data(self, df_features, target_col=Config.PM_COLUMN, seq_length=12):
        """Prepare sequential data for LSTM model"""
        self.feature_cols = [col for col in df_features.columns if col != target_col]
        
        data = df_features[self.feature_cols + [target_col]].values
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :-1])  # All features
            y.append(data[i+seq_length, -1])  # Target
        
        return np.array(X), np.array(y)
    
    def train(self, df_features):
        """
        Train the selected model on the provided features.
        
        Args:
            df_features (pd.DataFrame): Feature dataframe from make_features()
        """
        print(f"🎯 Training {self.model_type.upper()} model...")
        
        if self.model_type == 'rf':
            X, y = self._prepare_rf_data(df_features)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest
            self.model = RandomForestRegressor(
                n_estimators=Config.RF_N_ESTIMATORS,
                max_depth=Config.RF_MAX_DEPTH,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_scaled, y)
            
        elif self.model_type == 'lstm':
            if not LSTM_AVAILABLE:
                raise ImportError("TensorFlow is required for LSTM model")
            
            X, y = self._prepare_lstm_data(df_features)
            
            # Scale features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # Build LSTM model
            self.model = Sequential([
                LSTM(64, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            self.model.fit(X_scaled, y, epochs=Config.LSTM_EPOCHS, 
                          batch_size=Config.LSTM_BATCH_SIZE, verbose=0)
        
        print(f"✅ {self.model_type.upper()} model trained successfully")
    
    def predict(self, df_features, horizon_steps=6):
        """
        Generate predictions for the specified horizon.
        
        Args:
            df_features (pd.DataFrame): Feature dataframe
            horizon_steps (int): Number of 5-min steps to predict
        
        Returns:
            np.ndarray: Predictions
        """
        if self.model_type == 'rf':
            X, _ = self._prepare_rf_data(df_features)
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled[-horizon_steps:])
            
        elif self.model_type == 'lstm':
            X, _ = self._prepare_lstm_data(df_features)
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            predictions = self.model.predict(X_scaled[-horizon_steps:], verbose=0).flatten()
        
        return predictions


def predict_pm(model_name, horizon, df_features):
    """
    Unified prediction interface for dashboard integration.
    
    Args:
        model_name (str): 'rf' or 'lstm'
        horizon (int): Forecast horizon in minutes
        df_features (pd.DataFrame): Feature dataframe
    
    Returns:
        dict: Predictions with timestamps
    """
    # Calculate number of 5-min steps
    horizon_steps = horizon // 5
    
    # Initialize and train model
    forecaster = PMForecaster(model_type=model_name)
    forecaster.train(df_features)
    
    # Generate predictions
    predictions = forecaster.predict(df_features, horizon_steps)
    
    # Create timestamps for predictions
    last_timestamp = df_features.index[-1]
    pred_timestamps = [last_timestamp + timedelta(minutes=5*(i+1)) for i in range(horizon_steps)]
    
    return {
        'timestamps': [str(ts) for ts in pred_timestamps],
        'predictions': predictions.tolist(),
        'horizon': horizon,
        'model': model_name
    }


# ============================================================================
# 4. EVALUATION METRICS
# ============================================================================

def evaluate(df_features, model_name='rf', horizon=30, test_size=0.2):
    """
    Compute evaluation metrics for the specified model and horizon.
    
    Args:
        df_features (pd.DataFrame): Feature dataframe
        model_name (str): 'rf' or 'lstm'
        horizon (int): Forecast horizon in minutes
        test_size (float): Proportion of data to use for testing
    
    Returns:
        dict: Dictionary containing MAE and RMSE
    """
    print(f"📊 Evaluating {model_name.upper()} model with {horizon}-min horizon...")
    
    # Split data
    split_idx = int(len(df_features) * (1 - test_size))
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    # Train model
    forecaster = PMForecaster(model_type=model_name)
    forecaster.train(train_df)
    
    # Predict on test set
    horizon_steps = min(horizon // 5, len(test_df))
    predictions = forecaster.predict(test_df, horizon_steps)
    
    # Get actual values
    actual = test_df[Config.PM_COLUMN].values[-horizon_steps:]
    
    # Calculate metrics
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    
    print(f"✅ MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return {
        'model': model_name,
        'horizon': horizon,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2)
    }


def generate_metrics_table(df_features, horizons=[15, 30, 60], models=['rf']):
    """
    Generate a comprehensive metrics table for different models and horizons.
    
    Args:
        df_features (pd.DataFrame): Feature dataframe
        horizons (list): List of forecast horizons to evaluate
        models (list): List of models to evaluate
    
    Returns:
        pd.DataFrame: Metrics table
    """
    results = []
    
    for model in models:
        for horizon in horizons:
            metrics = evaluate(df_features, model_name=model, horizon=horizon)
            results.append(metrics)
    
    metrics_df = pd.DataFrame(results)
    
    # Save as markdown
    metrics_df.to_markdown('metrics_table.md', index=False)
    print(f"✅ Metrics table saved to metrics_table.md")
    
    return metrics_df


# ============================================================================
# 5. SPRAY SIGNAL LOGIC
# ============================================================================

def check_spray_signal(pred_dict, threshold=Config.SPRAY_THRESHOLD):
    """
    Determine if spray mitigation should be triggered based on predictions.
    
    Args:
        pred_dict (dict): Prediction dictionary from predict_pm()
        threshold (float): PM threshold for triggering spray
    
    Returns:
        dict: Spray event information or None
    """
    predictions = np.array(pred_dict['predictions'])
    timestamps = pred_dict['timestamps']
    
    # Find if any prediction exceeds threshold
    exceed_mask = predictions > threshold
    
    if exceed_mask.any():
        # Find first exceedance
        first_exceed_idx = np.where(exceed_mask)[0][0]
        exceed_value = predictions[first_exceed_idx]
        exceed_time = timestamps[first_exceed_idx]
        
        event = {
            'status': 'SPRAY_REQUIRED',
            'timestamp': exceed_time,
            'predicted_pm': round(float(exceed_value), 2),
            'threshold': threshold,
            'excess': round(float(exceed_value - threshold), 2),
            'message': f"PM predicted to reach {exceed_value:.1f} at {exceed_time}, exceeding threshold by {exceed_value - threshold:.1f}"
        }
        
        print(f"⚠️ SPRAY ALERT: {event['message']}")
        return event
    
    return {
        'status': 'NO_SPRAY_NEEDED',
        'max_predicted_pm': round(float(predictions.max()), 2),
        'threshold': threshold
    }


# ============================================================================
# 6. IMPACT SIMULATION
# ============================================================================

def simulate_impact(pred_dict, mitigation_factor=Config.MITIGATION_FACTOR):
    """
    Simulate the impact of spray mitigation on PM levels.
    
    Args:
        pred_dict (dict): Prediction dictionary from predict_pm()
        mitigation_factor (float): Proportion of PM reduction (0.3 = 30% reduction)
    
    Returns:
        dict: Simulation results with before/after comparisons
    """
    print(f"🌫️ Simulating spray impact ({mitigation_factor*100:.0f}% reduction)...")
    
    original_pm = np.array(pred_dict['predictions'])
    
    # Apply mitigation: reduce PM by mitigation_factor
    mitigated_pm = original_pm * (1 - mitigation_factor)
    
    # Calculate impact metrics
    mean_reduction = original_pm.mean() - mitigated_pm.mean()
    peak_reduction = original_pm.max() - mitigated_pm.max()
    
    results = {
        'original_pm': original_pm.tolist(),
        'mitigated_pm': mitigated_pm.tolist(),
        'timestamps': pred_dict['timestamps'],
        'mean_reduction': round(float(mean_reduction), 2),
        'peak_reduction': round(float(peak_reduction), 2),
        'mitigation_factor': mitigation_factor
    }
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    timestamps = range(len(original_pm))
    
    plt.plot(timestamps, original_pm, 'o-', label='Without Spray', color='#e74c3c', linewidth=2)
    plt.plot(timestamps, mitigated_pm, 's-', label='With Spray Mitigation', color='#2ecc71', linewidth=2)
    plt.axhline(y=Config.SPRAY_THRESHOLD, color='orange', linestyle='--', label=f'Threshold ({Config.SPRAY_THRESHOLD})')
    
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('PM2.5 Level', fontsize=12)
    plt.title('Impact of Spray Mitigation on PM Levels', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('impact_simulation.png', dpi=150)
    plt.close()
    
    print(f"✅ Mean PM reduction: {mean_reduction:.2f}")
    print(f"✅ Peak PM reduction: {peak_reduction:.2f}")
    print(f"✅ Impact simulation saved to impact_simulation.png")
    
    return results


# ============================================================================
# 7. DASHBOARD-FRIENDLY OUTPUTS
# ============================================================================

def format_for_dashboard(pred_dict, metrics_dict, spray_event, impact_results):
    """
    Format all outputs as JSON for React.js dashboard integration.
    
    Args:
        pred_dict (dict): Predictions
        metrics_dict (dict): Evaluation metrics
        spray_event (dict): Spray alert information
        impact_results (dict): Impact simulation results
    
    Returns:
        dict: Complete dashboard payload
    """
    dashboard_payload = {
        'predictions': pred_dict,
        'metrics': metrics_dict,
        'spray_alert': spray_event,
        'impact_simulation': impact_results,
        'timestamp_generated': str(pd.Timestamp.now())
    }
    
    # Save as JSON file
    with open('dashboard_data.json', 'w') as f:
        json.dump(dashboard_payload, f, indent=2)
    
    print(f"✅ Dashboard data saved to dashboard_data.json")
    
    return dashboard_payload


# ============================================================================
# 8. MAIN PIPELINE EXECUTION
# ============================================================================

def run_full_pipeline(csv_path, model_name='rf', horizon=30):
    """
    Execute the complete ML pipeline from data to dashboard outputs.
    
    Args:
        csv_path (str): Path to input CSV
        model_name (str): Model to use ('rf' or 'lstm')
        horizon (int): Forecast horizon in minutes
    
    Returns:
        dict: Complete pipeline results
    """
    print("\n" + "="*60)
    print("🚀 STARTING PM FORECASTING PIPELINE")
    print("="*60 + "\n")
    
    # 1. Load and prepare data
    df = prepare_data(csv_path, horizon)
    
    # 2. Engineer features
    df_features = make_features(df, horizon)
    
    # 3. Generate predictions
    pred_dict = predict_pm(model_name, horizon, df_features)
    
    # 4. Evaluate model
    metrics_dict = evaluate(df_features, model_name, horizon)
    
    # 5. Check spray signal
    spray_event = check_spray_signal(pred_dict)
    
    # 6. Simulate impact
    impact_results = simulate_impact(pred_dict)
    
    # 7. Format for dashboard
    dashboard_data = format_for_dashboard(pred_dict, metrics_dict, spray_event, impact_results)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return dashboard_data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Run the full pipeline
    # Replace 'clean_data.csv' with your actual CSV file path
    
    try:
        results = run_full_pipeline(
            csv_path='clean_data.csv',
            model_name='rf',  # or 'lstm' if TensorFlow is available
            horizon=30  # 30-minute forecast
        )
        
        print("\n📋 QUICK SUMMARY:")
        print(f"   Model: {results['predictions']['model']}")
        print(f"   Horizon: {results['predictions']['horizon']} minutes")
        print(f"   MAE: {results['metrics']['MAE']}")
        print(f"   RMSE: {results['metrics']['RMSE']}")
        print(f"   Spray Status: {results['spray_alert']['status']}")
        
    except FileNotFoundError:
        print("\n⚠️ Error: 'clean_data.csv' not found.")
        print("Please update the csv_path parameter with your actual data file.")
        print("\nYou can also use individual functions:")
        print("  df = prepare_data('your_file.csv')")
        print("  df_features = make_features(df)")
        print("  predictions = predict_pm('rf', 30, df_features)")
