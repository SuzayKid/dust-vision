


import numpy as np
import pandas as pd
from datetime import datetime
import os

# Output file path (goes to ../data/sensor_data.csv)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sensor_data.csv")

def main():
    print("Starting sensor simulation...")

    # SETTINGS
    num_nodes = 4
    total_hours = 24
    minutes = total_hours * 60

    # Time series (1-minute interval)
    timestamps = pd.date_range(
        start=datetime(2025, 1, 1, 0, 0),
        periods=minutes,
        freq="T"
    )

    rows = []

    for node in range(1, num_nodes + 1):

        # Baseline pollution
        base_pm25 = 30
        base_pm10 = 60

        # Noise
        noise_pm25 = np.random.normal(0, 2, minutes)
        noise_pm10 = np.random.normal(0, 3, minutes)

        # Random dust spikes
        spikes = np.random.choice([0, 1], size=minutes, p=[0.97, 0.03])
        spike_pm25 = spikes * np.random.uniform(40, 120, minutes)
        spike_pm10 = spikes * np.random.uniform(80, 260, minutes)

        # Weather data
        wind_speed = np.random.uniform(0.5, 5.0, minutes)
        wind_dir = np.random.uniform(0, 360, minutes)
        humidity = np.random.uniform(40, 80, minutes)

        for i in range(minutes):
            rows.append({
                "timestamp": timestamps[i],
                "node_id": node,
                "pm25": base_pm25 + noise_pm25[i] + spike_pm25[i],
                "pm10": base_pm10 + noise_pm10[i] + spike_pm10[i],
                "wind_speed": wind_speed[i],
                "wind_dir": wind_dir[i],
                "humidity": humidity[i],
                "is_spike": spikes[i]
            })

    df = pd.DataFrame(rows)

    # Save output
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✔ Sensor simulation completed!")
    print(f"✔ File saved at: {OUT_PATH}")
    print(df.head())

if __name__ == "__main__":
    main()
