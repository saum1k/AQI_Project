import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# load cleaned dataset
df = pd.read_csv("outputs/cleaned_aqi.csv")

# convert date column
df['date'] = pd.to_datetime(df['date'])

# group data by date
daily_data = df.groupby('date')['PM2.5'].mean()

# create ARIMA model
model = ARIMA(daily_data, order=(5,1,0))

# train model
model_fit = model.fit()

# forecast next 30 days
forecast = model_fit.forecast(steps=30)

# plot forecast
plt.figure()

plt.plot(daily_data, label="Actual PM2.5")

future_dates = pd.date_range(start=daily_data.index[-1], periods=30)

plt.plot(future_dates, forecast, label="Predicted PM2.5")

plt.title("PM2.5 Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("PM2.5 Level")

plt.legend()

plt.savefig("outputs/graphs/pm25_forecast.png")

plt.show()

print("Forecast graph saved in outputs/graphs")