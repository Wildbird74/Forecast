
# coding: utf-8

# In[22]:


#snippet DF_1.1

import pandas as pd
import numpy as np
import dsdtools as tool
import sys
import math as math
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


df = pd.read_csv('Forecast_Books_GMS_2018.csv')

x = df['month_number']
y = df['gms']

# plot the data
plt.figure(figsize=(25,8))
plt.xticks(x,x, rotation='vertical',fontsize = 12)
plt.grid(True)
plt.plot(x,y,"o")
plt.show()


# In[24]:


#snippet DF_1.4
half_df = df[df.month_number <= 24]

x = half_df['month_number']
y = half_df['gms']

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

slope = z[0]
intercept = z[1]

# the line equation:
print("Trendline equation for half dataset: y=%.6fx+(%.6f)"%(z[0],z[1]))


# In[27]:


#snippet DF_1.5
pd.set_option('display.float_format', lambda x: '%.2f' % x)

hesf_df = df.copy()
total_months = 60
alpha = 0.2
beta = 0.2
initial_level = intercept
initial_trend = slope

hesf_df['one_step_ahead_forecast'] = intercept + slope
hesf_df['forecast_error'] = hesf_df['gms'] - hesf_df['one_step_ahead_forecast']
hesf_df['level'] = intercept + slope + alpha * hesf_df['forecast_error']
hesf_df['trend'] = slope + alpha * beta * hesf_df['forecast_error']

hesf_df = hesf_df[['month_number', 'gms', 'level','trend','one_step_ahead_forecast','forecast_error']]

new_df = pd.DataFrame([[49,0,0,0,0,0],
                       [50,0,0,0,0,0],
                       [51,0,0,0,0,0],
                       [52,0,0,0,0,0],
                       [53,0,0,0,0,0],
                       [54,0,0,0,0,0],
                       [55,0,0,0,0,0],
                       [56,0,0,0,0,0],
                       [57,0,0,0,0,0],
                       [58,0,0,0,0,0],
                       [59,0,0,0,0,0],
                       [60,0,0,0,0,0]], 
                      columns=['month_number','gms','level','trend','one_step_ahead_forecast','forecast_error'])

hesf_df = hesf_df.append(new_df,ignore_index=True)

#Applying double exponential smoothing and forecasting orders
for index, row in hesf_df.iterrows():
    if index > 0 and index <= 47:
        hesf_df.loc[index,'one_step_ahead_forecast'] = hesf_df.loc[index-1,'level'] + hesf_df.loc[index-1,'trend']
        hesf_df.loc[index,'forecast_error'] = hesf_df.loc[index, 'gms'] - hesf_df.loc[index,'one_step_ahead_forecast']
        hesf_df.loc[index,'level'] = hesf_df.loc[index-1,'level'] + hesf_df.loc[index-1,'trend'] + alpha * hesf_df.loc[index,'forecast_error']
        hesf_df.loc[index,'trend'] = hesf_df.loc[index-1,'trend'] + alpha * gamma * hesf_df.loc[index,'forecast_error']
        last_known_month = hesf_df.loc[index,'month_number']
        last_known_level = hesf_df.loc[index,'level']
        last_known_trend = hesf_df.loc[index,'trend']
    elif index > 47:
        hesf_df.loc[index,'gms'] = last_known_level + (hesf_df.loc[index,'month_number'] - last_known_month ) * last_known_trend

hesf_df


# In[28]:


#snippet DF_1.6

x = hesf_df['month_number']
y = hesf_df['gms']

# plot the data
plt.figure(figsize=(25,8))
plt.xticks(x,hesf_df['month_number'], rotation='vertical',fontsize = 12)
plt.grid(True)
plt.plot(x,y)

# calc the trendline
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r-")
# the line equation:
print("Trendline equation: y=%.6fx+(%.6f)"%(z[0],z[1]))


# In[29]:


#snippet DF_1.7
#Finding Autocorrelations in time series
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot

autocorr_df = hesf_df[hesf_df.month_number <= 48]
autocorr_df = autocorr_df[['month_number','gms']]

#calculating critical values
max_critical_value = 2/np.sqrt(48)
min_critical_value = -2/np.sqrt(48)

autocorr_value=autocorr_df.gms

#values required for plotting graph
lag_month = [1,2,3,4,5,6,7,8,9,10,11,12]
max_cv_series = [max_critical_value,max_critical_value,max_critical_value,max_critical_value,max_critical_value,max_critical_value,max_critical_value,max_critical_value,max_critical_value,max_critical_value,max_critical_value,max_critical_value]
min_cv_series = [min_critical_value,min_critical_value,min_critical_value,min_critical_value,min_critical_value,min_critical_value,min_critical_value,min_critical_value,min_critical_value,min_critical_value,min_critical_value,min_critical_value]

#Building autocorrelation series
autocorr_series = np.array([autocorr_value.autocorr(lag=1),
                            autocorr_value.autocorr(lag=2),
                            autocorr_value.autocorr(lag=3),
                            autocorr_value.autocorr(lag=4),
                            autocorr_value.autocorr(lag=5),
                            autocorr_value.autocorr(lag=6),
                            autocorr_value.autocorr(lag=7),
                            autocorr_value.autocorr(lag=8),
                            autocorr_value.autocorr(lag=9),
                            autocorr_value.autocorr(lag=10),
                            autocorr_value.autocorr(lag=11),
                            autocorr_value.autocorr(lag=12)]
                          )

#plotting correlogram
plt.figure(figsize=(25,8))
plt.xticks(x,lag_month, rotation='vertical',fontsize = 12)
plt.grid(True)
plt.bar(lag_month,autocorr_series)
plt.plot(max_cv_series)
plt.plot(min_cv_series)

print("Maximum Critical value :%.6f"%(max_critical_value))
print("Minimum Critical value :%.6f"%(min_critical_value))
print(autocorr_series)


# In[15]:


#snippet DF_1.8

pd.set_option('display.float_format', lambda x: '%.6f' % x)

hwsf_df = df[['month_number','gms']]

hwsf_df['smooothed_gms'] = 0
hwsf_df['seasonal_factor_estimate'] = 0
hwsf_df['initial_seasonal_factor'] = 0
hwsf_df['deseasonalised_gms'] = 0
hwsf_df['skew'] = 0

#Seasonal Factor Estimation - 2x12 moving average
for index, row in hwsf_df.iterrows():
    sum1= 0
    sum2= 0
    if index > 5 and index < 41:
        sum1 =sum([hwsf_df.loc[i,'gms'] for i in range(index-6,index+6)])/ 12
        sum2 =sum([hwsf_df.loc[i,'gms'] for i in range(index-5,index+7)])/ 12
        hwsf_df.loc[index,'smooothed_gms'] = (sum1+sum2)/2
        hwsf_df.loc[index,'seasonal_factor_estimate'] = hwsf_df.loc[index,'gms']/hwsf_df.loc[index,'smooothed_gms']

#Calculating initial Seasonal Factor
month_dfs = hwsf_df[hwsf_df.month_number <= 12]
for index,row in month_dfs.iterrows():
    factor_sum = 0
    for i in range(index,48,12):
        factor_sum = factor_sum + hwsf_df.loc[i,'seasonal_factor_estimate']
    hwsf_df.loc[index,'initial_seasonal_factor'] = factor_sum/4
    hwsf_df.loc[index,'skew'] = (1-hwsf_df.loc[index,'initial_seasonal_factor'])*100
    

month_df = hwsf_df[hwsf_df.month_number <= 12]
print('Initail Seasonal Factors:')
print(month_df[['month_number','initial_seasonal_factor','skew']])


# In[16]:


#snippet 1.9

pd.set_option('display.float_format', lambda x: '%.6f' % x)

#Applying seasonal smoothing to historical data
for index,row in hwsf_df.iterrows():
    if i > 11:
        i = 0
    hwsf_df.loc[index,'deseasonalised_gms'] = hwsf_df.loc[index,'gms'] / month_df.iloc[i]['initial_seasonal_factor']
    i = i + 1
    
x = hwsf_df['month_number']
y = hwsf_df['deseasonalised_gms']

hwsf_df



# In[30]:



# plot the seasonally smoothened data
plt.figure(figsize=(25,8))
plt.xticks(x,hwsf_df['month_number'], rotation='vertical',fontsize = 12)
plt.grid(True)
plt.plot(x,y)

# calc the trendline
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r-")

slope = z[0]
intercept = z[1]

# the line equation:
print("Trendline equation: y=%.6fx+(%.6f)"%(slope,intercept))


# In[58]:


#snippet 1.10
pd.set_option('display.float_format', lambda x: '%.6f' % x)

hwf_df = month_df.append(hwsf_df,ignore_index=True)
hwf_df = hwf_df[['month_number','gms','initial_seasonal_factor']]

alpha = 0.1
beta = 0.1
gamma = 0.1

hwf_df['level'] = 0.0
hwf_df['trend'] = 0.0
hwf_df['one_step_forecast'] = 0.0
hwf_df['forecast_error'] = 0.0

hwf_df = hwf_df[['month_number','gms','level','trend','initial_seasonal_factor','one_step_forecast','forecast_error']]

new_df = pd.DataFrame([[49,0,0,0,0,0,0],
                       [50,0,0,0,0,0,0],
                       [51,0,0,0,0,0,0],
                       [52,0,0,0,0,0,0],
                       [53,0,0,0,0,0,0],
                       [54,0,0,0,0,0,0],
                       [55,0,0,0,0,0,0],
                       [56,0,0,0,0,0,0],
                       [57,0,0,0,0,0,0],
                       [58,0,0,0,0,0,0],
                       [59,0,0,0,0,0,0],
                       [60,0,0,0,0,0,0]], 
                      columns=['month_number','gms','level','trend','initial_seasonal_factor','one_step_ahead_forecast','forecast_error'])

hwf_df = hwf_df.append(new_df,ignore_index=True)
hwf_df = hwf_df[['month_number','gms','level','trend','initial_seasonal_factor','one_step_forecast','forecast_error']]

hwf_df = hwf_df.set_value(11, 'level', intercept, takeable=False)
hwf_df = hwf_df.set_value(11, 'trend', slope, takeable=False)

#Applying Triple Exponential Smoothing
for index, row in hwf_df.iterrows():
    if index > 11 and index <= 59:
        hwf_df.loc[index,'one_step_forecast'] = (hwf_df.loc[index-1,'level'] + hwf_df.loc[index-1,'trend']) * hwf_df.loc[index-12,'initial_seasonal_factor']
        hwf_df.loc[index,'forecast_error'] = hwf_df.loc[index, 'gms'] - hwf_df.loc[index,'one_step_forecast']
        hwf_df.loc[index,'initial_seasonal_factor'] = hwf_df.loc[index-12,'initial_seasonal_factor']  + gamma * (1-alpha) * hwf_df.loc[index,'forecast_error']/(hwf_df.loc[index-1,'level'] + hwf_df.loc[index-1,'trend'])
        hwf_df.loc[index,'level'] = hwf_df.loc[index-1,'level'] + hwf_df.loc[index-1,'trend'] + alpha * hwf_df.loc[index,'forecast_error']/hwf_df.loc[index-12,'initial_seasonal_factor']
        hwf_df.loc[index,'trend'] = hwf_df.loc[index-1,'trend'] + alpha * beta * hwf_df.loc[index,'forecast_error']/hwf_df.loc[index-12,'initial_seasonal_factor']  
        
        last_known_month = hwf_df.loc[index,'month_number']
        last_known_level = hwf_df.loc[index,'level']
        last_known_trend = hwf_df.loc[index,'trend']        
    elif index > 59:
        hwf_df.loc[index,'gms'] = (last_known_level + (hwf_df.loc[index,'month_number'] - last_known_month ) * last_known_trend) * hwf_df.loc[index-12,'initial_seasonal_factor']
    else:
        hwf_df.loc[index,'month_number'] = 0
        hwf_df.loc[index,'gms'] = 0 
        
order_forcast = hwf_df[['month_number','gms','forecast_error']][hwf_df.month_number > 0]
order_forcast


# In[55]:


#snippet 1.11
x = order_forcast['month_number']
y = order_forcast['gms']
    
# plot the data itself
plt.figure(figsize=(25,8))
plt.xticks(x,x, rotation='vertical',fontsize = 12)
plt.grid(True)
plt.plot(x,y)

# calc the trendline (it is simply a linear fitting)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r-")

# the line equation:
print("Trendline equation: y=%.6fx+(%.6f)"%(slope,intercept))

