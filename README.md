# EPS Prediction and Financial Analytics

![image](https://github.com/user-attachments/assets/3439b86e-fd30-4743-a4b7-4b2bd21fe120)

This repository contains an ETL pipeline and interactive visualization dashboard designed for financial data analytics, specifically Earnings Per Share (EPS) prediction using a Gradient Boosted Tree regression model built with PySpark, Dash, and Plotly.

## ETL Pipeline
The ETL process uses PySpark for efficient large-scale data processing:

### Extract
- Ingests EPS and income statement data from CSV files.
- Joins datasets on company symbols and dates.

### Transform
- Creates lag features, computes EPS growth, income growth, and EPS surprise metrics.
- Handles missing data via forward-fill.
- Encodes categorical variables (company symbols) using StringIndexer and OneHotEncoder.
- Assembles features and trains a Gradient Boosted Tree regression model.
- Generates comprehensive predictions and prepares results for visualization.

### Load
- Stores enriched data and predictions into MySQL for further analytics and visualization.
- Persists the trained regression model and processed datasets as Parquet files for efficiency.

## Visualization Dashboard
The Dash-based dashboard provides interactive insights into financial metrics and EPS predictions:

### Features:
- **Historical EPS Comparison:** Visualizes actual versus predicted EPS values and prediction errors over time.
- **Future EPS Forecasting:** Allows users to input future financial metrics for generating EPS forecasts.
- **Key Metrics Analysis:** Displays interactive charts for net income, average shares, and income from continuing operations.
- **Company Comparisons:** Facilitates comparison across multiple companies on selected financial metrics.

## Technologies Used
- PySpark (Spark ML, DataFrames)
- Gradient Boosted Trees (GBT) Regression
- Dash and Plotly for interactive visualizations
- MySQL for data persistence
- Pandas for data manipulation within Dash

This project showcases a robust, scalable pipeline for financial forecasting and analytics, offering valuable tools for investors, analysts, and stakeholders.
