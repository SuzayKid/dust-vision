DustVision – Sensor & Camera Fusion Logic

This document explains how sensor readings and camera detections are combined to identify true construction dust events. The fusion logic produces the final labeled dataset fused_events.csv, which is used by the Machine Learning module (Person B).

Input Data Sources

Fusion uses two independently simulated data streams:

Sensor Data — sensor_data.csv
Contains minute-level readings for each node:

PM2.5 concentration

PM10 concentration

Weather data (wind speed, direction, humidity)

Spike indicator (is_spike)

Camera Events — camera_events.csv
Contains event-level detections:

Timestamp

Node ID

Severity (0.1–1.0)

Confidence (0.0–1.0)

Type (real or false_positive)

Fusion Goals

The purpose of fusion is to determine:
“Is this timestamp at this node a real dust event?”

Fusion combines:

Sensor spikes

Camera detections

Wind direction

Timing proximity

Matching Logic (Time Window Fusion)

For every sensor reading at time T and node N:

Look for camera events at the same node.

A valid match occurs when a camera detection happens between:
T − 2 minutes and T + 2 minutes.

If a detection exists in that window, its severity, confidence, and type are added to the fused row.

If no detection is found, camera_severity = 0, camera_conf = 0, and camera_type = "none".

Wind Alignment Logic

Wind direction helps verify real dust movement.
Assumption: dust blowing toward city sensors occurs when the wind direction is between 50° and 140°.

If wind_dir is inside this range: wind_aligned = 1
Otherwise: wind_aligned = 0

Final Labeling Logic (Event Classification)

A timestamp is labeled as a true dust event (label = 1) only if:

sensor_spike equals 1

AND camera_confidence > 0.6

AND wind_aligned equals 1

Otherwise: label = 0

This prevents false alarms and ensures multi-sensor confirmation.

Output Columns in fused_events.csv

timestamp – minute-level timestamp
node_id – sensor node
pm25, pm10 – pollution values
wind_speed – weather data
wind_dir – wind direction
humidity – relative humidity
sensor_spike – simulated spike indicator
camera_severity – matched camera severity
camera_conf – confidence score
camera_type – real / false_positive / none
wind_aligned – 1 if wind direction supports dust flow
label – final classification of event (1 = true dust event)

Purpose for Machine Learning

The fused dataset creates a realistic ground-truth label for training ML models.
Inputs include pollution, weather, and camera info.
The label indicates validated dust events.

Conclusion

Fusion creates a robust multi-sensor dataset essential for DustVision’s prediction and governance systems. It strengthens reliability by combining sensor spikes, camera confirmation, and wind behavior into a single unified output.