Air Quality Index (AQI) Analysis and Forecasting
Overview

This project focuses on analyzing Air Quality Index (AQI) data using data science and machine learning techniques. The goal is to understand pollution trends, identify patterns, perform clustering, and forecast future air quality levels. A dashboard is also developed to visualize insights interactively.

The project demonstrates skills in data preprocessing, exploratory data analysis, dimensionality reduction, clustering, forecasting, and visualization.

Features
Data cleaning and preprocessing
Exploratory Data Analysis (EDA)
Principal Component Analysis (PCA)
K-Means clustering
AQI forecasting using machine learning
Interactive dashboard using Tkinter and Matplotlib
Visualization of trends and pollution patterns
Live AQI fetching using API

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Tkinter
Requests
Installation

Clone the repository:

git clone https://github.com/saum1k/AQI_Project.git
cd AQI_Project

Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn requests
Usage

Run preprocessing:

python src/preprocessing.py

Run exploratory data analysis:

python src/eda.py

Run PCA analysis:

python src/pca_analysis.py

Run clustering:

python src/clustering.py

Run forecasting:

python src/forecasting.py

Launch dashboard:

python src/dashboard.py
Dataset

The dataset contains air quality measurements including pollutant concentrations used to calculate AQI values. The preprocessing step handles missing values, formatting issues, and prepares the data for analysis and modeling.

Machine Learning Methods Used
PCA (Principal Component Analysis)

Used for dimensionality reduction and understanding variance distribution in pollution features.

K-Means Clustering

Used to group locations or observations based on similar pollution characteristics.

Forecasting Model

Used to predict future AQI trends based on historical data.

Dashboard

The dashboard provides:

Visualization of AQI trends
Cluster visualization
PCA visualization
Live AQI data retrieval
Interactive plots
Future Improvements
Add deep learning models for forecasting
Deploy dashboard as web application
Integrate real-time streaming AQI data
Add more advanced feature engineering
Improve visualization interactivity
Author

Saumik Laddha

GitHub:
https://github.com/saum1k
