# Predictive Maintenance of Industrial Bearings
### Machine Learning + Power BI · Python · Scikit-learn

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-orange)
![Power BI](https://img.shields.io/badge/Dashboard-Power%20BI-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Problem statement
Industrial bearing failures cause costly unplanned downtime in manufacturing environments.
Real-world fault data is scarce and expensive to collect — making standard supervised ML 
pipelines difficult to apply directly.

## Approach
To overcome the data scarcity problem, I engineered a **physical and mathematical simulation** 
that injects realistic multi-dimensional fault signatures directly into the dataset:

- **Temperature profiles** — normal operating range vs. fault-induced thermal deviation
- **Magnetic flux profiles** — modelled electromagnetic signatures for inner race, outer race, 
  and ball faults
- **Fault distribution injection** — statistical distributions fitted to real failure modes

This produced a robust, multimodal training dataset without requiring expensive physical test rigs.

## Model
| Component | Detail |
|-----------|--------|
| Algorithm | Random Forest Classifier (Scikit-learn) |
| Fault classes | Normal · Inner race · Outer race · Ball fault |
| Features | Vibration amplitude, temperature delta, magnetic flux variance |
| Validation | Cross-validation on simulated + benchmark data |

## Dashboard
A **live Power BI dashboard** was built for real-time monitoring:
- Streaming data ingestion
- Fault class probability indicators
- Alert threshold visualisation

> Screenshot of the Power BI dashboard:

![Dashboard](images/dashboard_screenshot.png)

## Repository structure
