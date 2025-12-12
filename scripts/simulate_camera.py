import pandas as pd
import numpy as np
import os
from datetime import timedelta

SENSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sensor_data.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "camera_events.csv")

def main():
    print("Starting camera event simulation...")

    df = pd.read_csv(SENSOR_PATH, parse_dates=["timestamp"])

    camera_events = []

    # 80% chance camera catches a real dust spike
    REAL_EVENT_PROB = 0.8

    # generate camera events from real spikes
    for _, row in df[df["is_spike"] == 1].iterrows():

        if np.random.rand() < REAL_EVENT_PROB:
            delay = np.random.randint(0, 3)  # camera may detect 0–2 mins later
            cam_time = row["timestamp"] + timedelta(minutes=delay)

            camera_events.append({
                "timestamp": cam_time,
                "node_id": row["node_id"],
                "severity": round(np.random.uniform(0.4, 1.0), 2),
                "confidence": round(np.random.uniform(0.6, 1.0), 2),
                "type": "real"
            })

    # now add false positives (camera mistakes)
    false_events_count = 30
    timestamps = df["timestamp"].unique()
    nodes = df["node_id"].unique()

    for _ in range(false_events_count):
        camera_events.append({
            "timestamp": np.random.choice(timestamps),
            "node_id": int(np.random.choice(nodes)),
            "severity": round(np.random.uniform(0.2, 0.6), 2),
            "confidence": round(np.random.uniform(0.3, 0.7), 2),
            "type": "false_positive"
        })

    # Convert to DataFrame and save
    cam_df = pd.DataFrame(camera_events)
    cam_df.sort_values("timestamp", inplace=True)

    cam_df.to_csv(OUT_PATH, index=False)

    print("✔ Camera simulation completed!")
    print(f"✔ File saved at: {OUT_PATH}")
    print(cam_df.head())

if __name__ == "__main__":
    main()
