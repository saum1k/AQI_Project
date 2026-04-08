print("FILE IS RUNNING")

import pandas as pd

df = pd.read_csv("data/aqi.csv", encoding="latin1")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns in dataset:")
print(df.columns)

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = df[['location','date','pm2_5','rspm','no2','so2']]

df.rename(columns={
    'location':'City',
    'pm2_5':'PM2.5',
    'rspm':'PM10',
    'no2':'NO2',
    'so2':'SO2'
}, inplace=True)

df = df.dropna()

print("\nCleaned Data:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# save cleaned dataset
df.to_csv("outputs/cleaned_aqi.csv", index=False)

print("\nCleaned dataset saved to outputs folder")