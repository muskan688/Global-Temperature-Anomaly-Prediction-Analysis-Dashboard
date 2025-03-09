# Global Temperature Anomaly Prediction and Visualization


### Project Overview

This project aims to analyze historical global temperature anomalies and predict future trends using machine learning models. The workflow involves data cleaning, feature engineering, exploratory data analysis (EDA), model training, deployment, and visualization using Tableau.


### How to Run the Project

1. Install Dependencies:
pip install -r requirements.txt

2. Run Data Pipelines:
    - python data_cleaning.py
    - python feature_eng.py
    - python eda.py (for visualization)
    - train.py

3. Generate Data for Tableau:
    - python tableau_data.py

4. Launch Streamlit App:
    - streamlit run deploy.py

5. Visualize Data in Tableau:
    - Open visualization.twb