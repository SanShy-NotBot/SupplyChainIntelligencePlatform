# 🚚 Supply Chain Analytics & Predictive Operations Dashboard

An end-to-end Statistical Machine Learning platform built on the **DataCo Supply Chain Dataset** (180,519 orders) that combines **Supervised Predictive Modeling** with **Unsupervised Operational Segmentation** in an interactive Streamlit dashboard.

## 🔍 Key Features
- **Data Preprocessing & Feature Engineering (`main.ipynb`):** Cleans missing/redundant fields, removes post-shipment leakage columns, extracts cyclical temporal features (`order_hour`, `order_dayofweek`, `is_weekend`), caps skewed financial outliers via 1st–99th percentile winsorization, and applies one-hot encoding.
- **Supervised Risk & Transit Modeling:** Uses `GroupShuffleSplit` on `Order Id` to prevent multi-item order leakage across train/test splits:
  - **Late Delivery Classifier (Tuned Random Forest):** Predicts binary SLA breach risk (`71.61%` accuracy, `75.37%` weighted precision, `71.24%` weighted F1-score).
  - **Shipping Duration Regressor (Random Forest):** Estimates continuous transit duration in days (`0.99` days MAE, `1.29` days RMSE).
- **Unsupervised Segmentation (`code4_clustering.ipynb`):** Standardizes 10 key operational features, projects them via 2D PCA, and segments shipments into distinct behavioral clusters using K-Means.
- **Interactive Decision-Support UI (`app.py`):** Features global sidebar filtering, live EDA visualizations, a real-time 3-model order simulator, and cluster trait profiling.

## 🛠️ Tech Stack
`Python` • `scikit-learn` • `pandas` • `NumPy` • `Streamlit` • `Plotly` • `Joblib`

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
