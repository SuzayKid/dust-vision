from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta

# Assuming pm_forecast.py is in the same directory or accessible in PYTHONPATH
from pm_forecast import load_and_prep_data, engineer_features, Config, RandomForestRegressor, LabelEncoder # Import necessary components

app = FastAPI(
    title="DustVision Prediction API",
    description="API for forecasting PM10 levels and simulating mitigation impact.",
    version="1.0.0"
)

# Load the trained model and LabelEncoder at startup
try:
    MODEL_PATH = 'ml/models/rf_pm_forecast.pkl'
    # Ensure the directory exists before attempting to load
    import os
    if not os.path.exists('ml/models/'):
        os.makedirs('ml/models/')
    
    # Check if model file exists, if not, train it
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model not found at {MODEL_PATH}. Running full pipeline to train it...")
        from pm_forecast import run_pipeline
        run_pipeline() # This will train and save the model
        print("✅ Model trained and saved.")
        
    model = joblib.load(MODEL_PATH)
    
    # Load or re-create LabelEncoder
    # The LabelEncoder is used for 'node_encoded' feature. It needs to be fitted on the same data as during training.
    # For simplicity, we'll re-fit it on a dummy dataset or ensure it's saved/loaded.
    # A more robust solution would save/load the encoder with the model.
    # For now, let's assume `engineer_features` handles it consistently.
    dummy_df = load_and_prep_data(Config.INPUT_FILE)
    _, feature_cols = engineer_features(dummy_df) # This will create and fit a LabelEncoder internally
    
    print("🧠 Prediction model and components loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model or related components: {e}")
    model = None # Indicate that model loading failed

class PredictionRequest(BaseModel):
    forecast_horizon_minutes: int = Config.FORECAST_HORIZON_MIN

class PredictionResponse(BaseModel):
    timestamps: list[str]
    predicted_pm10: list[float]
    forecast_horizon_minutes: int

class MetricsResponse(BaseModel):
    model: str
    horizon_minutes: int
    mae: float
    rmse: float
    features_used: int

class MitigationSimulationResponse(BaseModel):
    original_pm: list[float]
    mitigated_pm: list[float]
    timestamps: list[str]
    mean_reduction: float
    peak_reduction: float
    mitigation_factor: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_pm10(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model not loaded.")

    try:
        # Load and preprocess the latest data
        df = load_and_prep_data(Config.INPUT_FILE)
        
        # Engineer features
        df_processed, feature_cols = engineer_features(df)
        
        # Ensure the feature columns match the training data
        # This is a critical step for robust inference
        X_infer = df_processed[feature_cols]
        
        # Make prediction for the last entry in the dataset
        # We need to predict 'forecast_horizon_minutes' into the future
        # The 'engineer_features' already creates 'target_pm10' by shifting.
        # For a live prediction, we need to predict the *next* value based on current features.
        
        # The existing model in pm_forecast.py predicts 'target_pm10' which is shifted.
        # To get a 'live' prediction, we essentially need to simulate the input features
        # for the time step we want to predict.
        
        # For simplicity, let's re-use the prediction logic from pm_forecast.py's
        # run_pipeline and adapt it for single-point prediction.
        
        # To get a prediction for T + forecast_horizon_minutes, we need features at T.
        # The `engineer_features` creates 'target_pm10' based on 'pm10'.shift(-steps_ahead).
        # We need to predict the future, so we take the latest features and predict.
        
        # Get the latest features
        latest_features = X_infer.iloc[[-1]]
        
        # Predict
        prediction = model.predict(latest_features)[0] # Assuming single prediction
        
        # Generate future timestamp
        last_timestamp = df_processed[Config.TIMESTAMP_COL].iloc[-1]
        predicted_timestamp = (last_timestamp + timedelta(minutes=request.forecast_horizon_minutes)).isoformat()

        return PredictionResponse(
            timestamps=[predicted_timestamp],
            predicted_pm10=[round(float(prediction), 2)],
            forecast_horizon_minutes=request.forecast_horizon_minutes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    try:
        with open('metrics_table.json', 'r') as f:
            metrics = json.load(f)
        return MetricsResponse(**metrics)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics file not found. Run the prediction pipeline first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {e}")

# Note: The simulate_mitigation_impact function in pm_forecast.py currently generates a plot
# and doesn't directly return structured data for an API.
# To expose this via API, we would need to modify simulate_mitigation_impact
# to return the 'simulated_pm' and 'y_actual' data.
# For now, let's create a placeholder that would call it and return dummy data or
# indicate that the function needs modification.

@app.get("/simulate_mitigation", response_model=MitigationSimulationResponse)
async def get_mitigation_simulation():
    # This endpoint currently requires re-running the simulation logic from pm_forecast.py
    # and extracting the data. For a real-time API, this would likely be pre-calculated
    # or the function in pm_forecast.py would need to be refactored to return data.
    
    # To avoid re-running the full pipeline which creates plots,
    # we'll need to adapt the simulation function or rely on pre-generated data.
    # For now, let's assume we can get the data from a function call directly.
    
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model not loaded for simulation.")

    try:
        # Re-run a portion of the pipeline to get necessary data for simulation
        df = load_and_prep_data(Config.INPUT_FILE)
        df_processed, feature_cols = engineer_features(df)
        
        # To get y_test and preds for simulation, we'd ideally re-run the training/evaluation
        # on a small, recent subset or have these stored.
        # For this API, we will make a simplifying assumption:
        # We will use the latest data point to predict and then simulate mitigation.
        
        # This is a simplified approach. In a real scenario, you'd likely:
        # 1. Store the results of a full simulation run.
        # 2. Modify `simulate_mitigation_impact` to return data instead of just plotting.
        
        # For demonstration, let's get the latest actual PM10 and a predicted value
        # and apply a simple mitigation logic.
        
        latest_actual_pm10 = df_processed[Config.TARGET_COL].iloc[-1]
        latest_features = df_processed[feature_cols].iloc[[-1]]
        latest_predicted_pm10 = model.predict(latest_features)[0]
        
        # Apply mitigation logic (simplified, as per pm_forecast.py's `simulate_mitigation_impact`)
        ACTION_THRESHOLD = 250 # From pm_forecast.py
        
        original_pm_values = [latest_actual_pm10, latest_predicted_pm10]
        mitigated_pm_values = []
        
        if latest_predicted_pm10 > ACTION_THRESHOLD:
            mitigated_val = latest_actual_pm10 * (1 - Config.SPRAY_EFFICIENCY)
            mitigated_pm_values = [mitigated_val, latest_predicted_pm10 * (1 - Config.SPRAY_EFFICIENCY)]
        else:
            mitigated_pm_values = [latest_actual_pm10, latest_predicted_pm10]

        mean_reduction = (sum(original_pm_values) / len(original_pm_values)) - (sum(mitigated_pm_values) / len(mitigated_pm_values))
        peak_reduction = max(original_pm_values) - max(mitigated_pm_values)

        last_timestamp = df_processed[Config.TIMESTAMP_COL].iloc[-1]
        timestamps = [last_timestamp.isoformat(), (last_timestamp + timedelta(minutes=Config.FORECAST_HORIZON_MIN)).isoformat()]

        return MitigationSimulationResponse(
            original_pm=[round(float(p), 2) for p in original_pm_values],
            mitigated_pm=[round(float(p), 2) for p in mitigated_pm_values],
            timestamps=timestamps,
            mean_reduction=round(float(mean_reduction), 2),
            peak_reduction=round(float(peak_reduction), 2),
            mitigation_factor=Config.SPRAY_EFFICIENCY
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mitigation simulation failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
