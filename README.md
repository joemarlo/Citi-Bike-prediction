# Citi-Bike-prediction

Prediction evaluation of daily rides starting from each Citi Bike station. Trained and tested on a 80/20 split of 1.4 million observations representing 95 million bike trips from 2013 to early 2020.

See also [NYC-data](https://github.com/joemarlo/NYC-data)

<br>

<p align="center">
<img src="Plots/y_hat_densities.png" width=80%>
</p>

<p align="center">
<img src="Plots/y_v_y_hat.png" width=80%>
</p>

<br>

Variables used:
- Distance to closest subway station
- Year
- Month of year
- Workday or not (weekdays minus holidays)
- Precipitation
- Max and minimum temperature
- Max wind speed

Data sources:
- https://www.citibikenyc.com/system-data
- https://data.ny.gov/widgets/i9wp-a4ja
- https://www.ncdc.noaa.gov/cdo-web/search
