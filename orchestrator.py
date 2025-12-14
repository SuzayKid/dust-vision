"""
DustVision System Orchestrator
==============================
Integrates Data (A), ML Forecasting (B), and Control Logic (C) 
into a single real-time decision loop.
"""

import pandas as pd
import joblib
import time
import json
from datetime import datetime

# --- IMPORT MODULES ---
# 1. The ML Pipeline (Person B)
from pm_forecast_pipeline import engineer_features, Config as MLConfig

# 2. The Control Modules (Person C)
from aq_severity_engine import SeverityRuleEngine
from action_mapping_system import ActionDispatcher

class DustVisionSystem:
    def __init__(self):
        print("🔌 Initializing DustVision System...")
        
        # Load ML Model (Person B)
        try:
            self.model = joblib.load('ml/models/rf_pm_forecast.pkl')
            print("✅ ML Model Loaded (Random Forest)")
        except FileNotFoundError:
            raise Exception("❌ Model not found! Run pm_forecast_pipeline.py first.")

        # Initialize Control Engines (Person C)
        self.severity_engine = SeverityRuleEngine()
        self.action_dispatcher = ActionDispatcher()
        print("✅ Governance & Action Engines Online")

    def process_realtime_stream(self, csv_path):
        """
        Simulates processing live data row-by-row from the CSV.
        """
        print(f"\n📡 Connecting to sensor stream: {csv_path}")
        
        # Load raw data
        raw_df = pd.read_csv(csv_path)
        
        # Pre-process features exactly how the ML model expects them
        # (Using the function from your pipeline to ensure consistency)
        processed_df, feature_cols = engineer_features(raw_df)
        
        print(f"▶️ Starting simulation loop ({len(processed_df)} events)...\n")
        
        # SIMULATION LOOP
        # We iterate through the dataframe acting like it's live data
        for index, row in processed_df.iterrows():
            
            # 1. PREDICT (The ML Step)
            # Extract features for model
            features = row[feature_cols].values.reshape(1, -1)
            predicted_pm10 = self.model.predict(features)[0]
            
            # 2. CLASSIFY (The Severity Step)
            # We classify based on the PREDICTED future value, not current value
            # This makes the system PROACTIVE rather than REACTIVE
            classification = self.severity_engine.classify(
                pm10=predicted_pm10,
                pm25=row['pm25'],          # Current PM2.5
                wind_speed=row['wind_speed'],
                humidity=row['humidity']
            )
            
            severity_level = classification['level']
            
            # 3. ACT (The Dispatch Step)
            # Only trigger actions if severity is NOT "good" or "moderate"
            # (In a real system, you might log everything, but act on bad air)
            action_result = None
            if severity_level not in ['good', 'moderate']:
                context = {
                    "timestamp": datetime.now(),
                    "location": f"Node_{int(row['node_encoded'])}",
                    "predicted_pm10": predicted_pm10
                }
                action_result = self.action_dispatcher.dispatch(severity_level, context)
            
            # 4. LOG/DISPLAY
            self._print_status(row, predicted_pm10, classification, action_result)
            
            # Sleep slightly to simulate real-time (optional, remove for speed)
            # time.sleep(0.5) 

    def _print_status(self, row, pred_pm10, classification, action_result):
        """Pretty prints the system state."""
        
        # Current vs Predicted
        curr_pm10 = row['pm10']
        trend_arrow = "↗️" if pred_pm10 > curr_pm10 else "↘️"
        
        print("-" * 60)
        print(f"⏱  TIME: {row.name} | 📍 NODE: {int(row['node_encoded'])}")
        print(f"📊 SENSOR: {curr_pm10:.1f} µg/m³  >>  🤖 FORECAST (+30m): {pred_pm10:.1f} µg/m³ {trend_arrow}")
        print(f"🚦 SEVERITY: {classification['display_name']} ({classification['color']})")
        
        if action_result and action_result['success']:
            print("🛡️  MITIGATION TRIGGERED:")
            for action in action_result['actions_executed']:
                print(f"   [⚡ {action['action_type'].upper()}]: {action.get('message', 'Executed')}")
                if action['action_type'] == 'sprinkler':
                    print(f"       -> Intensity: {action['details']['intensity']}% | Duration: {action['details']['duration']}s")
        else:
            print("✅ No mitigation required.")

# --- RUNNER ---
if __name__ == "__main__":
    system = DustVisionSystem()
    
    # Run the simulation using Person A's data
    # Ensure this path matches where you saved the csv
    system.process_realtime_stream('data/fused_events.csv')