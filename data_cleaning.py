import pandas as pd
import numpy as np

#Load the dataset
print('Dataset Loading..')
data = pd.read_csv('Global_Temp.csv', skiprows=1)

#Data Overview
print('Dataset Overview..')
print(data.head(5))

#Replacing ** values
data.replace('***', np.nan, inplace= True)
print('Sorted Data')
print(data.head(5))

#Data Information
print('Dataset Information..')
print(data.info())

#Changing Datatype
data['D-N'] = data['D-N'].astype(float)
data['DJF'] = data['DJF'].astype(float)
print('Changed Datatype..')
print(data.info())

#Data Summary
print('Dataset Summary..')
print(data.describe())

#Missing values
print('Dataset Missing Values Information..')
print(data.isnull().sum())

#Dealing missing values using forward and backward fill
data['D-N'] = data['D-N'].ffill().bfill()
data['DJF'] = data['DJF'].ffill().bfill()
print('Dataset after handling missing values..\n', data.head(5))

#Checked for Duplicates
print(' Sum of Duplicated values\n',data.duplicated().sum())

#Saving the clean file
data.to_csv('Cleaned_data.csv', index = False)
print('Cleaned dataset successfully saved as Cleaned_data.csv.')