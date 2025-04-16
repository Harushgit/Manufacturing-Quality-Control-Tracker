# 🏭 Manufacturing Quality Control Tracker

An end-to-end manufacturing quality control system integrating Exploratory Data Analysis (EDA), trend analysis, root cause analysis, machine learning models (classification & predictive), a Tableau dashboard, and YOLO-based multi-defect detection for PCB boards — all deployed through a user-friendly Dash interface.

## 🚀 Project Overview

The Manufacturing Quality Control Tracker is designed to monitor, analyze, and improve the quality of products in a manufacturing process. By leveraging real-time data and statistical analysis, it identifies defects, tracks performance metrics, and provides actionable insights to enhance overall production quality. This tool ensures compliance with industry standards and helps in reducing waste, improving efficiency, and maintaining consistent product quality.

---

## 🧩 Key Features

- 📊 **EDA & Trend Analysis**: Interactive visual summaries of production and quality metrics.
- 🔍 **Root Cause Analysis**: Identify and visualize potential defect causes.
- 🧠 **Predictive Model**: Predict product quality metrics based on process data.
- 🏷️ **Classification Model**: Categorize products into defect classes.
- 📦 **YOLOv8 Multi-Defect Detection**: Detects multiple defects in PCB boards using a trained YOLOv8 model.
- 📈 **Tableau Dashboard Integration**: For rich, pre-built visualizations and analytics.
- 🧮 **Dash-based UI**: Integrated front-end dashboard to access all modules in one place.

---

## 📁 Project Structure

manufacturing-quality-control-tracker/ │ ├── eda/ # EDA notebooks and scripts │ └── eda_analysis.ipynb │ ├── trend_analysis/ # Scripts for trend analysis │ └── trend_dashboard.py │ ├── root_cause_analysis/ # Root cause visuals and correlation analysis │ └── root_cause.py │ ├── models/ │ ├── classification_model.pkl │ ├── predictive_model.pkl │ └── model_utils.py │ ├── yolo_detection/ │ ├── best.pt # YOLOv8 trained weights │ ├── detect.py # YOLO inference script │ └── sample_images/ │ ├── tableau_dashboard/ │ ├── dashboard_embed_code.html │ └── dashboard_screenshot.png │ ├── dash_app/ │ ├── app.py # Main Dash app │ ├── pages/ │ │ ├── home.py │ │ ├── eda.py │ │ ├── prediction.py │ │ ├── yolo_detection.py │ │ └── tableau.py │ └── assets/ │ └── styles.css │ ├── data/ │ ├── raw/ │ ├── processed/ │ └── sample_input.csv │ ├── requirements.txt ├── README.md └── LICENSE

---

## 🛠️ Tech Stack

- **Python** – Core logic and ML
- **Dash & Plotly** – Interactive UI
- **YOLOv8** – Defect detection (Ultralytics)
- **scikit-learn / XGBoost** – ML models
- **Pandas & Seaborn** – Data wrangling & visualization
- **Tableau Public** – Dashboard for business insights

----

## 📸 Screenshots

![Screenshot 2025-02-22 000705](https://github.com/user-attachments/assets/4ad12b25-e398-4017-9010-8d204be7c00d)

![Screenshot 2025-02-21 235442](https://github.com/user-attachments/assets/152fbc9a-457c-4808-a152-90ed0cfba3d0)

![Screenshot 2025-02-22 002712](https://github.com/user-attachments/assets/8f48686f-5c47-4734-9fa2-dda659814a2a)

![Quality Assurance Team _ Dashboard ](https://github.com/user-attachments/assets/3acf798f-404a-4ff9-b8ea-db276468b519)

-------





