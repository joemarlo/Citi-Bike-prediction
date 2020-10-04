import pandas as pd
import numpy as np
import os
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/home/joemarlo/Dropbox/Data/Projects/NYC-data/Analyses/Citi-Bike-prediction/Citi-Bike-prediction')

# weekday and not holiday
# not raining
# max temperature
# avg wind speed
# snow / precip

# read in the data and remove NAs
station_trips = pd.read_csv("Data/station_day_trips.csv")
station_trips = station_trips.dropna()
station_trips = station_trips.astype({'Station':'int32', 'Trip_count': 'int32', 'Date': 'datetime64'})

# add week and month column
station_trips['Month'] = pd.DatetimeIndex(station_trips['Date']).month
station_trips['Week'] = pd.DatetimeIndex(station_trips['Date']).week
station_trips['Day_of_week'] = pd.DatetimeIndex(station_trips['Date']).dayofweek #monday=0
station_trips['is_weekday'] = station_trips['Day_of_week'].isin(np.linspace(0, 4, 5)) * 1

##### Holidays
cal = calendar()
holidays = cal.holidays(start=station_trips['Date'].min(), end=station_trips['Date'].max(), return_name=True)
holidays = holidays.reset_index(name='Holiday').rename(columns={'index':'Date'})
holidays['is_holiday'] = 1
holidays = holidays.drop('Holiday', axis=1)
station_trips = station_trips.merge(holidays, on='Date', how='left')
station_trips.is_holiday = station_trips.is_holiday.fillna(0)

# add nye
station_trips['holiday_NYE'] = np.where((pd.to_datetime(station_trips['Date']).dt.month == 12) & \
                     (pd.to_datetime(station_trips['Date']).dt.day == 31), 1, 0)
station_trips['is_holiday'] = station_trips[['is_holiday', 'holiday_NYE']].max(axis=1)
station_trips = station_trips.drop('holiday_NYE', axis=1)

# add 'work day'
station_trips['is_workday'] = ((station_trips['is_weekday'] == 1) & (station_trips['is_holiday'] == 0)) * 1

##### Weather
weather = pd.read_csv("Data/weather.csv")
weather = weather.rename(columns={"DATE": "Date", "PRCP": "Precipitation", "SNWD": "Snow_depth", "SNOW": "Snowfall", "TSUN": "Sunshine", "TAVG": "Temp_mean", "TMAX": "Temp_max", "TMIN": "Temp_min", "WSF2": "Wind_speed_max_2min", "WSF5": "Wind_speed_max_5sec", "AWND": "Wind_speed_mean"})
weather = weather.astype({'Date': 'datetime64'})

#sns.distplot(weather.Precipitation)
#plt.show()
weather['is_rainy'] = (weather.Precipitation > 0.1) * 1
weather['Snowfall'] = weather.Snowfall.fillna(0)

#plt.plot(weather.Date, weather.Temp_max - weather.Temp_min)
#plt.plot(weather.Date, weather.Temp_min, color = 'red')
#plt.show()

#sns.distplot(weather.Wind_speed_max_5sec)
#sns.distplot(weather.Wind_speed_max_2min)
#plt.show()
weather['is_windy'] = (weather.Wind_speed_max_5sec > 20) * 1
weather.Wind_speed_max_5sec = weather.Wind_speed_max_5sec.fillna(0)

# select certain variables and merge
weather = weather[['Date', 'Precipitation', 'Snowfall', 'Temp_max', 'Temp_min', 'Wind_speed_max_5sec']]
station_trips = station_trips.merge(weather, on='Date', how='left')

# write out
#station_trips = station_trips.astype({'Month':'int32', 'Week':'int32', 'Day_of_week':'int32', 'is_holiday':'int32', 'is_holiday':'int32'})
station_trips.to_csv("Data/cleaned_data.csv", index=False)
