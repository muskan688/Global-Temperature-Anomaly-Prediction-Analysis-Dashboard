import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("preprocessed_file.csv")

# Convert Month to numeric
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
data['Month'] = data['Month'].map(month_mapping)

# 🔥 Ensure feature order matches preprocessed_file.csv
feature_cols = ['Year', 'Month', '5-Year_MA', 'Season_Autumn', 'Season_Spring',
                'Season_Summer', 'Season_Winter', 'Avg_Seasonal_Temp']
target_col = 'Temperature_Values'

X = data[feature_cols]
y = data[target_col]

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('features used for fit', X.columns.tolist())

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')  # Save scaler for deployment

# Train and Evaluate Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

evaluation_results = {}
best_models = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    evaluation_results[name] = {
        'MAE': mae, 'MSE': mse, 'R2 Score': r2
    }
    
    # Save the trained model
    model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_filename)
    best_models[name] = model_filename
    print(f"✅ Model Saved: {model_filename}")

# Print Model Performance
result_df = pd.DataFrame(evaluation_results).T
print('\n🔍 Model Performance Comparison:')
print(result_df)

# Hyperparameter Tuning for Random Forest & XGBoost
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Grid Search for Random Forest
print("\n🔍 Tuning Random Forest Model...")
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2')
rf_grid_search.fit(X_train_scaled, y_train)
best_rf = rf_grid_search.best_estimator_
joblib.dump(best_rf, "best_random_forest_model.pkl")
print("✅ Best Random Forest Model Saved: best_random_forest_model.pkl")

# Grid Search for XGBoost
print("\n🔍 Tuning XGBoost Model...")
xgb_grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid_xgb, cv=5, scoring='r2')
xgb_grid_search.fit(X_train_scaled, y_train)
best_xgb = xgb_grid_search.best_estimator_
joblib.dump(best_xgb, "best_xgboost_model.pkl")
print("✅ Best XGBoost Model Saved: best_xgboost_model.pkl")
