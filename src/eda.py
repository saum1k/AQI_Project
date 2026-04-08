import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load cleaned dataset
df = pd.read_csv("outputs/cleaned_aqi.csv")

# convert date column
df['date'] = pd.to_datetime(df['date'])

# 1. AQI proxy trend using PM2.5
plt.figure()
df.groupby('date')['PM2.5'].mean().plot()
plt.title("PM2.5 Trend Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 level")

plt.savefig("outputs/graphs/pm25_trend.png")
plt.show()

# 2. correlation heatmap
plt.figure()
sns.heatmap(df[['PM2.5','PM10','NO2','SO2']].corr(), annot=True)

plt.title("Pollutant Correlation Heatmap")

plt.savefig("outputs/graphs/correlation_heatmap.png")
plt.show()

# 3. pollution distribution
plt.figure()
sns.histplot(df['PM2.5'])

plt.title("PM2.5 Distribution")

plt.savefig("outputs/graphs/pm25_distribution.png")
plt.show()

print("EDA graphs saved in outputs/graphs folder")