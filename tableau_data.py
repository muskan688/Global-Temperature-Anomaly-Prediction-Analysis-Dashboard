import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load preprocessed dataset
data = pd.read_csv('preprocessed_file.csv')

# Ensure Month is converted to numbers (1-12)
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
data['Month'] = data['Month'].map(month_mapping)

# Define features used in training
expected_features = ['Year', 'Month', '5-Year_MA', 'Season_Autumn', 'Season_Spring',
                     'Season_Summer', 'Season_Winter', 'Avg_Seasonal_Temp']

# Extract features and target variable
X = data[expected_features]  
y_actual = data['Temperature_Values']

# Load the trained scaler
scaler = joblib.load('scaler.pkl')

# Standardize features
X_scaled = pd.DataFrame(scaler.transform(X), columns=expected_features)

# Load trained models
best_rf_model = joblib.load('best_random_forest_model.pkl')
best_xgb_model = joblib.load('best_xgboost_model.pkl')

# Generate predictions
rf_prediction = best_rf_model.predict(X_scaled)
xgb_prediction = best_xgb_model.predict(X_scaled)

# Save results with correct column names
results_df = pd.DataFrame({
    'Year': data['Year'],  # Ensure it's "Year", not "Year1"
    'Month': data['Month'],  # Ensure it's numeric
    'Temperature_Values': y_actual,
    'RandomForest_Prediction': rf_prediction,
    'XGBoost_Prediction': xgb_prediction
})

#Cleaning
df_cleaned = results_df.dropna()
print('Null removed')

# Save for Tableau
df_cleaned.to_csv('prediction_data.csv', index=False)
print("✅ Prediction results saved to 'prediction_data.csv'")
