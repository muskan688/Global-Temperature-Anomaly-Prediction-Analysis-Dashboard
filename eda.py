import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Load Preprocessed data
data = pd.read_csv('preprocessed_file.csv')

#Time Series Plot
sns.lineplot(data=data, x='Year', y='5-Year_MA', marker = 'o', linestyle = '-', label = '5-Year_MA')

plt.title('Global Temperature Anomaly Trend')
plt.xlabel('Years')
plt.ylabel('Temperature Anomaly')
plt.legend()
plt.show()
'''Pre 1915 -slightly low temp is observed maybe due to climate changes.
around 1950- slightly high (warmer) temp is observed due to world war
post 1980 - rapid increase in temp maybe due to global warming'''

#Monthly Temperature Anomaly over time
plt.figure(figsize=(14,7))
sns.lineplot(data =data,x ='Year', y='Temperature_Values', hue='Month', palette='tab10', linewidth=1)
plt.title('Monthly Temperature Anomalies over the Years')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly')
plt.legend()
plt.show()

#Seasonal Temperature Anomaly over time
seasonal_data = {
    'Autumn' :data[data['Season_Autumn'] == 1].groupby('Year')['Temperature_Values'].mean(),
    'Spring' :data[data['Season_Spring'] == 1].groupby('Year')['Temperature_Values'].mean(),
    'Summer' :data[data['Season_Summer'] == 1].groupby('Year')['Temperature_Values'].mean(),
    'Winter' :data[data['Season_Winter'] == 1].groupby('Year')['Temperature_Values'].mean(),
}
#Convert to Dataframe
seasonal_df = pd.DataFrame(seasonal_data).reset_index()

sns.lineplot(data = seasonal_df, x= 'Year', y='Autumn', label ='Autumn',marker = 'o', linestyle = '--')
sns.lineplot(data = seasonal_df, x= 'Year', y='Spring', label ='Spring',marker = 's', linestyle = '-')
sns.lineplot(data = seasonal_df, x= 'Year', y='Summer', label ='Summer',marker = '^', linestyle = ':')
sns.lineplot(data = seasonal_df, x= 'Year', y='Winter', label ='Winter',marker = 'd', linestyle = '-.')

plt.title('Seasonal Temperature Anomalies')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly')
plt.legend()
plt.show()

#Distribution of Temperature Anomalies
plt.figure(figsize =(12,6))
sns.histplot(data['Temperature_Values'], bins = 20, kde=True, color = 'Blue')
plt.title('Distribution of Temperature Anomalies over the Years')
plt.xlabel('Temperature Values')
plt.ylabel('Frequency')
plt.show()

#Heatmap
month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov':11, 'Dec': 12
}

data['Month_Numbers'] = data['Month'].map(month_mapping)

numeric_columns = [col for col in data.columns if col not in ['Month']]
corr_matrix = data[numeric_columns].corr()

if not corr_matrix.isnull().all().all():
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True,  vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()
else:
    print("No valid correlations to display. All columns may contain constant values or NaNs.")
