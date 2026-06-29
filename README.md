# End-to-End Classification & Model Benchmarking Pipeline

## 🎯 Business Objective
The goal of this project is to build a production-ready machine learning workflow that predicts user behavior (e.g., premium subscription upgrades) based on engagement features. 

## ⚙️ Core Engineering Practices
* **Data Leakage Prevention:** Feature scaling (`StandardScaler`) is strictly fit on the training partition and applied to the test split to ensure zero predictive bias.
* **Benchmarking Loop:** The architecture dynamically trains and evaluates a baseline model alongside advanced ensemble methods.

## 📊 Performance Benchmarks
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~85.2% | 0.87 | 0.84 | 0.85 |
| **Random Forest** | *Run script to populate* | *...* | *...* | *...* |
| **XGBoost** | *Run script to populate* | *...* | *...* | *...* |

## 🛠️ How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python pipeline.py`
