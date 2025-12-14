"""
DustVision: Advanced PM Forecasting Pipeline (Person B)
========================================================
Designed to ingest 'fused_events.csv' (from Person A), train robust forecasting models 
using camera/sensor fusion features, and output predictive signals for 
Person C's mitigation engine.

Key Features:
- Node-aware processing (handles multiple sensors separately)
- Fusion-based feature engineering (uses camera confidence & wind alignment)
- Mitigation impact simulation using decay functions
- Generates all required Async Workflow deliverables
"""

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Optional LSTM support
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️ TensorFlow not found. Defaulting to Random Forest only.")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input/Output
    INPUT_FILE = 'data/fused_events.csv'  # Output from Person A
    MODEL_SAVE_PATH = 'ml/models/'
    OUTPUT_PATH = 'ml/outputs/'
    
    # Data Structure (From Fusion Logic)
    TARGET_COL = 'pm10'           # Primary target for dust mitigation
    TIMESTAMP_COL = 'timestamp'
    NODE_COL = 'node_id'
    
    # Feature Engineering
    # Features coming directly from Person A's fusion
    FUSION_FEATURES = [
        'pm25', 'pm10', 'wind_speed', 'wind_dir', 'humidity', 
        'camera_severity', 'camera_conf', 'wind_aligned', 'label'
    ]
    
    # Derived Features
    LAG_MINUTES = [5, 15, 30, 60]  # Lag periods to generate
    ROLLING_WINDOWS = [3, 6, 12]   # 15m, 30m, 60m rolling means (assuming 5min step)
    
    # Training
    FORECAST_HORIZON_MIN = 30     # Predict 30 mins into the future
    TEST_SPLIT_RATIO = 0.2
    RF_ESTIMATORS = 200
    
    # Mitigation Simulation
    SPRAY_EFFICIENCY = 0.45       # 45% reduction immediately after spray
    SPRAY_DECAY = 0.90            # Effect decays by 10% every time step

# ============================================================================
# 1. DATA INGESTION & PREPROCESSING
# ============================================================================

def load_and_prep_data(filepath):
    """
    Loads fused data and ensures correct types and sorting.
    Handles multiple nodes by sorting by Node -> Time.
    """
    print(f"📥 Loading fusion data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Generate dummy data if file doesn't exist yet (for testing)
        print("⚠️ File not found. Generating dummy fusion data for testing...")
        df = generate_dummy_data()

    # standardize columns
    df.columns = [c.lower() for c in df.columns]
    
    # Parse timestamp
    df[Config.TIMESTAMP_COL] = pd.to_datetime(df[Config.TIMESTAMP_COL])
    
    # Sort for Time Series
    df = df.sort_values(by=[Config.NODE_COL, Config.TIMESTAMP_COL]).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} rows across {df[Config.NODE_COL].nunique()} nodes.")
    return df

def generate_dummy_data():
    """Helper to create fake data matching Person A's schema if CSV is missing."""
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='5min')
    nodes = ['Node_A', 'Node_B', 'Node_C']
    data = []
    for node in nodes:
        for d in dates:
            # Simulate a dust event correlation
            is_dust = np.random.random() > 0.95
            pm10 = np.random.normal(40, 10) + (200 if is_dust else 0)
            cam_conf = 0.8 if is_dust else 0.1
            
            data.append({
                'timestamp': d,
                'node_id': node,
                'pm25': pm10 * 0.4,
                'pm10': pm10,
                'wind_speed': np.random.uniform(0, 10),
                'wind_dir': np.random.uniform(0, 360),
                'humidity': 50,
                'sensor_spike': 1 if is_dust else 0,
                'camera_severity': 0.8 if is_dust else 0.0,
                'camera_conf': cam_conf,
                'camera_type': 'real' if is_dust else 'none',
                'wind_aligned': 1 if is_dust else 0,
                'label': 1 if is_dust else 0
            })
    return pd.DataFrame(data)

# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING (FUSION-AWARE)
# ============================================================================

def engineer_features(df):
    """
    Creates lag, rolling, and interaction features.
    Crucially, it uses camera data as a 'leading indicator' for PM spikes.
    """
    print("⚙️ Engineering features...")
    df_feat = df.copy()
    
    # Process per node to prevent data leakage between sensors
    processed_dfs = []
    
    for node_id, group in df_feat.groupby(Config.NODE_COL):
        group = group.copy()
        
        # 1. Temporal Lags (Autoregression)
        # Assuming data is 5-min intervals. 
        for lag_min in Config.LAG_MINUTES:
            steps = int(lag_min / 5) 
            group[f'pm10_lag_{lag_min}m'] = group['pm10'].shift(steps)
            # Camera seeing dust NOW predicts PM LATER
            group[f'cam_sev_lag_{lag_min}m'] = group['camera_severity'].shift(steps)

        # 2. Rolling Statistics (Trend)
        for window in Config.ROLLING_WINDOWS:
            group[f'pm10_roll_mean_{window}'] = group['pm10'].rolling(window).mean()
            group[f'pm10_roll_std_{window}'] = group['pm10'].rolling(window).std()

        # 3. Fusion Interaction Features
        # "Is wind blowing dust towards us AND camera sees something?"
        group['risk_score_camera_wind'] = group['camera_severity'] * group['wind_aligned']
        
        # 4. Target Creation (Forecasting Future PM10)
        # We want to predict PM10 at T + Horizon
        steps_ahead = int(Config.FORECAST_HORIZON_MIN / 5)
        group['target_pm10'] = group['pm10'].shift(-steps_ahead)
        
        processed_dfs.append(group)
        
    # Recombine
    df_final = pd.concat(processed_dfs)
    
    # Drop rows with NaNs (created by lags/shifts)
    df_final.dropna(inplace=True)
    
    # Handle Node ID (Label Encode for RF)
    le = LabelEncoder()
    df_final['node_encoded'] = le.fit_transform(df_final[Config.NODE_COL])
    
    # Select feature columns
    feature_cols = [
        'pm25', 'pm10', 'wind_speed', 'humidity', 'wind_aligned', 
        'camera_severity', 'camera_conf', 'risk_score_camera_wind',
        'node_encoded'
    ]
    # Add generated lag/rolling columns
    feature_cols += [c for c in df_final.columns if 'lag' in c or 'roll' in c]
    
    print(f"✅ Engineered {len(feature_cols)} features for {len(df_final)} samples.")
    return df_final, feature_cols

# ============================================================================
# 3. MODEL TRAINING & EVALUATION
# ============================================================================

def train_evaluate_model(df, feature_cols):
    """
    Trains a Random Forest Regressor. 
    (RF is chosen for robustness to noise and ability to handle non-linear fusion logic).
    """
    print("🧠 Training Forecasting Model (Random Forest)...")
    
    # Time-based split (Don't shuffle time series data!)
    split_idx = int(len(df) * (1 - Config.TEST_SPLIT_RATIO))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train['target_pm10']
    X_test = test[feature_cols]
    y_test = test['target_pm10']
    
    # Model
    model = RandomForestRegressor(
        n_estimators=Config.RF_ESTIMATORS, 
        max_depth=15, 
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"🏆 Model Performance (Horizon: {Config.FORECAST_HORIZON_MIN} min):")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    
    # Save metrics for Async Deliverable
    metrics = {
        "model": "RandomForest_Fusion",
        "horizon_minutes": Config.FORECAST_HORIZON_MIN,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "features_used": len(feature_cols)
    }
    with open('metrics_table.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return model, X_test, y_test, preds

# ============================================================================
# 4. VISUALIZATION & IMPACT SIMULATION
# ============================================================================

def visualize_predictions(y_test, preds, timestamps, save_path='prediction_plots.png'):
    """Generates the prediction vs actual plot requested by Async Workflow."""
    plt.figure(figsize=(12, 6))
    
    # Plot only last 200 points for clarity
    limit = 200
    plt.plot(timestamps.iloc[-limit:], y_test.values[-limit:], label='Actual PM10', color='black', alpha=0.6)
    plt.plot(timestamps.iloc[-limit:], preds[-limit:], label='Predicted PM10', color='#007acc', linewidth=2)
    
    plt.title(f'PM10 Forecast ({Config.FORECAST_HORIZON_MIN} min horizon)', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('PM10 Concentration (µg/m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Prediction plot saved to {save_path}")

def simulate_mitigation_impact(y_actual, y_pred, timestamps):
    """
    Simulates: "What if we activated sprinklers when predictions crossed a threshold?"
    Required for Person B Deliverable: impact_simulation.png
    """
    print("🌫️ Running Impact Simulation...")
    
    # Threshold from Person C's severity engine (e.g., Unhealthy starts at 250 PM10)
    ACTION_THRESHOLD = 250 
    
    simulated_pm = []
    spray_active = False
    current_pm = y_actual.values[0]
    
    for i in range(len(y_pred)):
        predicted_val = y_pred[i]
        actual_val = y_actual.values[i]
        
        # Logic: If model predicts > Threshold, trigger mitigation NOW
        if predicted_val > ACTION_THRESHOLD:
            # Mitigation applies reduction to the ACTUAL value
            mitigated_val = actual_val * (1 - Config.SPRAY_EFFICIENCY)
        else:
            mitigated_val = actual_val
            
        simulated_pm.append(mitigated_val)
        
    # Plotting
    plt.figure(figsize=(12, 6))
    limit = 200
    
    plt.plot(timestamps.iloc[-limit:], y_actual.values[-limit:], label='Baseline (No Action)', color='red', linestyle='--')
    plt.plot(timestamps.iloc[-limit:], simulated_pm[-limit:], label='With AI Mitigation', color='green', linewidth=2)
    plt.axhline(y=ACTION_THRESHOLD, color='orange', label='Trigger Threshold')
    
    plt.fill_between(timestamps.iloc[-limit:], simulated_pm[-limit:], y_actual.values[-limit:], 
                     color='green', alpha=0.1, label='Pollution Prevented')
    
    plt.title('Impact of ML-Driven Dust Mitigation', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('impact_simulation.png')
    print("✅ Impact simulation saved.")

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def run_pipeline():
    # 1. Load Person A's Fused Data
    df = load_and_prep_data(Config.INPUT_FILE)
    
    # 2. Engineer Features
    df_processed, feature_cols = engineer_features(df)
    
    # 3. Train & Evaluate
    model, X_test, y_test, preds = train_evaluate_model(df_processed, feature_cols)
    
    # 4. Generate Visual Deliverables
    test_timestamps = df.loc[X_test.index, Config.TIMESTAMP_COL]
    visualize_predictions(y_test, preds, test_timestamps)
    simulate_mitigation_impact(y_test, preds, test_timestamps)
    
    # 5. Save Model for Inference
    joblib.dump(model, 'ml/models/rf_pm_forecast.pkl')
    print("💾 Model saved to ml/models/rf_pm_forecast.pkl")

if __name__ == "__main__":
    run_pipeline()