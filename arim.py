import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
df=pd.read_csv("GlobalTemperatures.csv")
df['Date']=pd.to_datetime(df['dt'])
df.set_index('Date',inplace=True)
ts=df['LandAndOceanAverageTemperature'].dropna()
ts = ts['1900-01-01':'2005-12-01']
ts.index.freq='MS'
adf=adfuller(ts)
print("ADF Statistics :",adf[0])
print("p-value",adf[1])
if adf[1]<0.05:
    print("series is stationary")
else:
    print("series is not stationary ")
ts_log=np.log(ts)
plot_acf(ts_log,lags=24) 
plt.title("ACF - Autocorrelation")
plt.show()
plot_pacf(ts_log,lags=24,method='ywm')
plt.title("PACF - Partial Autocorrelation")
plt.show()
model=ARIMA(ts_log,order=(1,1,1))
res=model.fit()
plt.figure(figsize=(10,5))
plt.plot(ts_log,label="Log Temperature ")
plt.plot(res.fittedvalues,color='red',label="Fitted")
plt.legend()
plt.show()
fc=res.get_forecast(steps=24)
fc_mean=fc.predicted_mean
fc_ci=fc.conf_int()
fc_series=np.exp(fc_mean)
lower=np.exp(fc_ci['lower LandAndOceanAverageTemperature'])
upper = np.exp(fc_ci['upper LandAndOceanAverageTemperature'])
plt.figure(figsize=(10, 5))
plt.plot(ts, label="Historical Temperature")
plt.plot(fc_series, label="Forecast", color='green')
plt.fill_between(fc_series.index,lower,upper,color='lightgreen',alpha=0.3)
plt.legend()
plt.title("Forecast: Next 24 Months of Global Temperature")
plt.show()
future_forecast=pd.DataFrame({'Forecast Temperature':fc_series, 'Lower Bound (95%)': lower,
    'Upper Bound (95%)': upper})
print("\n=== Future Global Temperature Forecast (Next 24 Months) ===\n")
print(future_forecast.round(3))


