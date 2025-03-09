import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load trained models and scaler
random_forest_model = joblib.load('best_random_forest_model.pkl')
xgboost_model = joblib.load('best_xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load preprocessed dataset (to get correct feature order)
preprocessed_data = pd.read_csv("preprocessed_file.csv")

# 🔥 Ensure feature order matches train.py
expected_features = ['Year', 'Month', '5-Year_MA', 'Season_Autumn', 'Season_Spring',
                     'Season_Summer', 'Season_Winter', 'Avg_Seasonal_Temp']

# Convert 'Month' to numeric format (Prevents string-to-float conversion error)
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
preprocessed_data['Month'] = preprocessed_data['Month'].map(month_mapping)

# Streamlit UI
st.title("🌍 Future Monthly Temperature Anomaly Predictions")
st.write("""
This application predicts Monthly Temperature Anomalies using trained models.
Select a future year and month, and visualize the predicted anomalies over time.
""")

# Sidebar Model Selection
st.sidebar.header("🔍 Model Selection")
model_choice = st.sidebar.selectbox('Choose a Model:', ['Random Forest', 'XGBoost'])
model_dict = {"Random Forest": random_forest_model, "XGBoost": xgboost_model}
selected_model = model_dict[model_choice]

# User Input: Future Year & Month
st.header("📅 Enter Prediction Details")
year = st.slider("Select Future Year", min_value=2024, max_value=2100, value=2030)
month = st.selectbox("Select Month", list(month_mapping.keys()))
month_num = month_mapping[month]

# One-Hot Encode Season
season_df = pd.DataFrame([[0, 0, 0, 0]], columns=['Season_Autumn', 'Season_Spring', 'Season_Winter', 'Season_Summer'])

if month_num in [3, 4, 5]:
    season_df['Season_Spring'] = 1
elif month_num in [6, 7, 8]:
    season_df['Season_Summer'] = 1
elif month_num in [9, 10, 11]:
    season_df['Season_Autumn'] = 1
else:
    season_df['Season_Winter'] = 1

# Compute 5-Year Moving Average (MA)
def compute_five_year_ma(selected_year, df):
    past_years = df[df['Year'] < selected_year].tail(5)
    if not past_years.empty:
        past_anomalies = past_years['Temperature_Values']
        yearly_trend = past_anomalies.diff().mean()  # Avg yearly change
        estimated_ma = past_anomalies.mean() + (selected_year - past_years['Year'].max()) * yearly_trend
        return estimated_ma
    return df['Temperature_Values'].mean()

# Compute Avg_Seasonal_Temp for Future Years
def get_avg_seasonal_temp(selected_year, selected_month, df):
    latest_year = df['Year'].max()
    seasonal_trend = df.groupby("Year")['Avg_Seasonal_Temp'].mean().diff().mean()
    projected_temp = df[(df['Year'] == latest_year) & (df['Month'] == selected_month)]
    if not projected_temp.empty:
        return projected_temp.iloc[0]['Avg_Seasonal_Temp'] + ((selected_year - latest_year) * seasonal_trend)
    return df['Avg_Seasonal_Temp'].mean()

# 🛠 Compute missing variables before using them
five_year_ma = compute_five_year_ma(year, preprocessed_data)
avg_seasonal_temp = get_avg_seasonal_temp(year, month_num, preprocessed_data)

# 🔍 Debugging: Print computed feature values
print(f"\n🔍 Debug: Year {year}, 5-Year_MA: {five_year_ma}, Avg_Seasonal_Temp: {avg_seasonal_temp}")

# Create Input DataFrame (Ensure Correct Feature Order)
input_data = pd.DataFrame([[year, month_num, five_year_ma]], columns=['Year', 'Month', '5-Year_MA'])

# Add One-Hot Encoded Season Columns
input_data = pd.concat([input_data, season_df], axis=1)

# Add Avg_Seasonal_Temp at the correct last position
input_data['Avg_Seasonal_Temp'] = avg_seasonal_temp

# 🔥 Ensure input features are in the correct order before prediction
input_data = input_data[expected_features]

# Debugging Output: Print feature order before scaling
print("\n🔍 Expected Feature Order:", expected_features)
print("✅ Final Input Data (Before Scaling):\n", input_data.head())

# Apply Scaling
input_data_scaled = scaler.transform(input_data)

# Predict & Display Results
if st.button("🔮 Predict Temperature Anomaly"):
    prediction = selected_model.predict(input_data_scaled)[0]
    st.subheader(f'📈 Predicted Temperature Anomaly: {prediction:.2f}°C')

    # Display Confidence Interval
    st.write("Confidence in Prediction:")
    if prediction > 0.5:
        st.success("🔥 High Temperature Anomaly Expected!")
    elif prediction < -0.5:
        st.warning("❄ Cooling Trend Expected!")
    else:
        st.info("🌍 Normal Temperature Variability.")

    # Visualization: Prediction vs. Actual
    st.header("📊 Prediction vs. Actual Trends")
    
    # Filter historical data for visualization
    historical_data = preprocessed_data[preprocessed_data['Month'] == month_num]
    
    plt.figure(figsize=(10, 5))
    plt.plot(historical_data['Year'], historical_data['Temperature_Values'], label="Actual Temperature", color='blue')
    plt.axvline(year, linestyle='--', color='red', label="Prediction Year")
    plt.scatter([year], [prediction], color='red', label="Predicted Anomaly", zorder=3)
    
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.title(f"Temperature Trends for {month}")
    plt.legend()
    st.pyplot(plt)

    # Residual Plot
    st.header("📉 Residual Plot (Errors in Predictions)")
    
    # 🔥 Fix: Ensure residuals use scaled data for prediction
    scaled_features = scaler.transform(preprocessed_data[expected_features])
    predicted_values = selected_model.predict(scaled_features)
    residuals = preprocessed_data['Temperature_Values'] - predicted_values

    plt.figure(figsize=(10, 5))
    plt.scatter(preprocessed_data['Year'], residuals, label="Residuals", color='purple')
    plt.axhline(0, linestyle='--', color='black')
    
    plt.xlabel("Year")
    plt.ylabel("Prediction Error")
    plt.title("Residual Plot for Model Predictions")
    plt.legend()
    st.pyplot(plt)

    # 🔥 FUTURE PREDICTION GRAPH NOW INSIDE BUTTON BLOCK
    st.header("📈 Projected Temperature Anomalies for Future Years")

    # Future year range for trend visualization
    future_years = [2030, 2040, 2050]
    future_predictions = []

    for future_year in future_years:
        five_year_ma = compute_five_year_ma(future_year, preprocessed_data)
        avg_seasonal_temp = get_avg_seasonal_temp(future_year, month_num, preprocessed_data)

        input_data_future = pd.DataFrame([[future_year, month_num, five_year_ma]],
                                         columns=['Year', 'Month', '5-Year_MA'])
        input_data_future = pd.concat([input_data_future, season_df], axis=1)
        input_data_future['Avg_Seasonal_Temp'] = avg_seasonal_temp
        input_data_future = input_data_future[expected_features]

        input_data_scaled_future = scaler.transform(input_data_future)
        future_predictions.append(selected_model.predict(input_data_scaled_future)[0])

    # Plot future trend
    plt.figure(figsize=(8, 5))
    plt.plot(future_years, future_predictions, marker='o', linestyle='-', color='red', label="Predicted Anomalies")
    plt.xticks(future_years)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer = True))
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.title("Projected Temperature Anomalies (2030-2050)")
    plt.legend()
    st.pyplot(plt)
