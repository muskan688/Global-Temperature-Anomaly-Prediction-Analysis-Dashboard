#Feature Engineering

#Import the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


#Convert Year column to integer for proper indexing
def moving_average(data, window = 5):
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data['Year'] = data['Year'].astype(int)

    #Set year as index for rolling calculations
    data.set_index('Year', inplace = True)

    #Applying 5-Years moving Average to all numeric columns
    moving_average = data.rolling(window=5, min_periods=1).mean()
    moving_average["5-Year_MA"] = moving_average["J-D"].rolling(window=5, min_periods=1).mean()

    #Reset index to keep year as a column
    moving_average.reset_index(inplace=True)
    return moving_average


#Reshaping the data
def reshape_data(data):
    # Debugging: Check if "Year" exists
    if "Year" not in data.columns:
        print("Error: 'Year' or '5-Year MA' column is missing before reshaping.")
        print("Current columns:", data.columns)
        return None  # Stop execution if "Year" is missing
    
    monthly_columns = [col for col in data.columns if col not in ["Year", "J-D", "D-N", "DJF", "MAM", "JJA", "SON"]]
    data_melted = data.melt(id_vars = ['Year','5-Year_MA'], value_vars = monthly_columns,var_name = 'Month', value_name = 'Temperature_Values')
    data_melted['Season']  = data_melted['Month'].apply(get_season)
    return data_melted


#Defining function to assign seasons based on months
def get_season(month):
        if month in ['Dec','Jan','Feb']:
            return 'Winter'
        elif month in ['Mar','Apr','May']:    
            return 'Spring'
        elif month in ['Jun','July','Aug']:
            return 'Summer'
        else:
            return 'Autumn'


#Data Encoding
def data_encoding(data):
    data_encoded = pd.get_dummies(data, columns=['Season'])
    season_columns = [col for col in data_encoded.columns if 'Season' in col]
    data_encoded[season_columns] = data_encoded[season_columns].astype(int)
    return data_encoded

def add_seasonal_column(data):
    data['Avg_Seasonal_Temp'] = data.groupby(['Year','Month'])['Temperature_Values'].transform('mean')
    return data

#Normalization
def Normalization(data):
    scaler = StandardScaler()
    numeric_columns = [col for col in data.columns if col not in ['Year','Month'] and not col.startswith('Season_')]
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data



if __name__== "__main__":


    #Load data
    data = pd.read_csv('Cleaned_data.csv')

    #Displaying the Moving Average
    data_moving_average = moving_average(data)  
    print('Moving Average Column Added\n',data_moving_average)

    #Displaying Time-Based Features Extraction
    reshaped_data = reshape_data(data_moving_average)
    print('Reshaped data\n',reshaped_data)
   
    #Displaying Encoding data
    encoding_data = data_encoding(reshaped_data)
    print('Encoded data\n',encoding_data)

    #Displaying seasonal data column
    seasonal_data = add_seasonal_column(encoding_data)
    print('Data after adding new seasonal column\n', seasonal_data)

    #Displaying Normalized Data
    normalized_data = Normalization(seasonal_data)
    print('Normalized Encoded data\n',normalized_data)

  

    #Saving the file
    preprocessed_file = normalized_data
    csv_preprocessed = 'preprocessed_file.csv'
    preprocessed_file.to_csv(csv_preprocessed, index=False)
    print(f"Final processed dataset saved as \n : {preprocessed_file}")