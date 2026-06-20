# Predictive Maintenance of Industrial Bearings: A Machine Learning & Power BI Approach

This repository hosts the source code and deployment architecture for an end-to-end intelligent diagnostic system tailored for industrial rotating machinery (asynchronous motors). The solution leverages multi-sensor data fusion to track bearing degradation in real time, minimize unexpected line shutdowns, and prevent catastrophic material failures.

## 🚀 Key Features
- **Multi-Sensor Fusion & Innovation**: Integrates high-frequency vibration parameters (RMS and Kurtosis) with physics-modeled synthetic indicators (Temperature and Magnetic Flux) to increase diagnostic robustness and mitigate false alarms.
- **Robust Machine Learning Model**: Employs a supervised Random Forest Classifier to categorize bearing health status across four operational modes: Normal, Inner Race Fault, Outer Race Fault, and Ball Fault.
- **Dual-Layer Visualization**:
  - A responsive **Web Interface (IHM)** for on-demand localized structural diagnosis.
  - An interactive **Power BI Dashboard** connected via a streaming data pipeline for factory-wide real-time fleet supervision and predictive maintenance tracking.

## 📁 Repository Structure
Refer to the directory architecture to locate data pipelines (`src/data_processing.py`), model workflows (`src/train.py`), the diagnostic web application (`web_app/`), and the final Power BI template (`dashboard/`).

## 📊 Dataset Reference
The vibration baseline relies on the internationally benchmarked **Case Western Reserve University (CWRU) Bearing Dataset** operating at a 12 kHz sampling rate, which was processed via a 50% overlapping sliding window mechanism.

---
**Author:** Yosr GABSI  
**Academic Framework:** National Engineering School of Carthage (ENICarthage), 2025-2026.
