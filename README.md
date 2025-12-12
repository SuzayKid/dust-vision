# DustVision: Adaptive Dust Mitigation System

A Smart‑City Framework for Detecting, Predicting, and Actively Mitigating Construction Dust Pollution

---

## 🚀 Overview

DustVision is a smart‑city–ready system engineered to **detect, forecast, and reduce construction‑driven dust pollution (PM2.5 & PM10)**. It integrates:

* Multi‑sensor data streams
* Simulated camera‑based dust event triggers
* Weather & wind metadata
* ML‑based AQI forecasting
* A rule‑driven mitigation + compliance engine

The repo contains **data pipelines, fusion logic, ML models, and mitigation algorithms**.
**UI, dashboards, and the final hackathon report are intentionally kept separate.**

---

## 🌟 Key Features

### **1. Sensor Simulation**

Generates realistic PM2.5/PM10 time‑series with controlled dust spikes.

### **2. Camera‑Event Simulation**

Synthetic CV triggers representing visible dust events (no actual image processing required).

### **3. Fusion Engine**

Combines:

* Sensor anomalies
* Camera event timestamps
* Wind direction & speed
  To localize, confirm, and score dust events.

### **4. ML‑Based Prediction**

Short‑term (30–60 min) PM forecasting using RF or LSTM models.

### **5. Impact Simulator**

Before/after mitigation AQI curves for decision evaluation.

### **6. Mitigation Engine**

Generates recommendations for:

* Smart sprinkling intensity + duration
* Material‑cover enforcement
* Work‑hour scheduling
* Automated contractor alerts/escalation

### **7. Governance & Compliance Logic**

Scores contractors based on events, mitigation adherence, and recurrence.

### **8. Scalable City Architecture**

From a **single pilot site → multiple wards → full‑city orchestration** using spatial interpolation.

---

## 🧩 Team Roles (Async Workflow)

### 👤 **Person A — Data & Sensor Pipeline Engineer**

**Focus:** dataset creation, simulation, fusion
**Deliverables:**

* `sensor_data.csv`
* `camera_events.csv`
* `fused_events.csv`
* `fusion_logic.md`

### 👤 **Person B — Machine Learning Engineer**

**Focus:** forecasting, evaluation, impact modeling
**Deliverables:**

* `prediction_notebook.ipynb`
* `metrics_table.md`
* `prediction_plots.png`
* `impact_simulation.png`

### 👤 **Person C — Control & Governance Systems Engineer**

**Focus:** mitigation logic, governance, city-scale deployment
**Deliverables:**

* `control_flow.md`
* `mitigation_strategies.md`
* `compliance_system.md`
* `city_scalability_plan.md`

---

## 📂 Repository Structure

```
/data
   sensor_data.csv
   camera_events.csv
   fused_events.csv

/ml
   prediction_notebook.ipynb
   metrics_table.md
   prediction_plots.png
   impact_simulation.png

/control
   control_flow.md
   mitigation_strategies.md
   compliance_system.md
   city_scalability_plan.md

/docs
   roadmap.md
   README.md
```

---

## 🛠️ Tech Stack

* **Simulation:** Python, Pandas, NumPy
* **Fusion Logic:** Python
* **Machine Learning:** scikit‑learn (RF), optional LSTM (TensorFlow/Keras)
* **Visualization:** Matplotlib, Seaborn
* **Documentation:** Markdown

Additional optional tools:

* **Geo‑modules (Shapely/Geopandas)** for ward-level interpolation
* **FastAPI** for turning DustVision into an API service (future phase)

---

## 📈 Workflow Summary

1. **Person A** → Simulates raw data + fusion outputs.
2. **Person B** → Trains forecasting models and produces predictions.
3. **Person C** → Builds mitigation + governance framework using model insights.
4. **THEN** UI/dashboard/report development begins.

---

## 🧪 Metrics to Report

* PM Forecasting Accuracy: **MAE / RMSE**
* Dust Event Fusion Accuracy
* Mitigation Effectiveness Curve
* Estimated PM Reduction (%)

Extended optional metrics:

* Event recurrence rate
* Contractor compliance score distribution
* Peak PM suppression time

---

## 🌍 Scalability

DustVision is designed for cost‑efficient deployment using:

* Sparse sensor grids
* Low‑cost camera nodes
* Mobile probes (municipal vehicles)
* Spatial interpolation models
* Prioritization algorithms for hotspot management

This architecture supports **real‑time ward‑level insights** and future integration into city emergency pollution response.

---

## 📝 Notes

* This README documents the **technical backbone only**.
* UI, dashboard, and hackathon documentation are **later‑stage deliverables**.
* All modules are designed to be modular, testable, and easily deployable.

---

## ✔️ Optional Enhancements (If Time Allows)

* Real‑time FastAPI backend
* SMS/WhatsApp alert integration
* Contractor leaderboard
* Cost‑benefit analysis generator
* Wind‑sensitive dynamic mitigation patterns

---

## 📜 License

MIT License (or project-specific license to be added).
