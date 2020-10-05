import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from boruta import BorutaPy

os.chdir('/home/joemarlo/Dropbox/Data/Projects/NYC-data/Analyses/Citi-Bike-prediction/Citi-Bike-prediction')

# read in the data and convert data tyopes
station_trips = pd.read_csv("Data/cleaned_data.csv")
station_trips = station_trips.astype({'Date': 'datetime64', 'is_holiday': 'int64'})
station_trips = station_trips.rename(columns={'dist_to_subway': 'Dist_to_subway'})
station_trips.isnull().sum()

##### univariate plots

# density plots
sns.kdeplot(station_trips.Trip_count)
plt.show()
sns.kdeplot(station_trips.Dist_to_subway)
plt.show()
sns.kdeplot(station_trips.Precipitation)
plt.show()
sns.kdeplot(station_trips.Snowfall)
plt.show()
sns.kdeplot(station_trips.Temp_max)
sns.kdeplot(station_trips.Temp_min)
plt.show()
sns.kdeplot(station_trips.Wind_speed_max_5sec)
plt.show()

# histograms
sns.distplot(station_trips.Year, kde=False)
plt.show()
sns.distplot(station_trips.Month, kde=False)
plt.show()


##### pairs plot
# first sample for performance
samp = station_trips[['Trip_count', 'Week', 'is_workday', 'Precipitation', 'Snowfall', 'Temp_max', 'Temp_min', "Wind_speed_max_5sec"]].sample(n=1000)
sns.pairplot(samp)
plt.show()

# Compute the correlation matrix
corr = station_trips.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


##### Boruta for feature selection
X = station_trips.drop(["Trip_count", 'Date'], axis=1)
y = station_trips['Trip_count']

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=44, max_iter = 25, perc = 95)

# find all relevant features
feat_selector.fit(np.array(X), y)

# features selected by the Boruta algo
selected_features = X.columns[feat_selector.support_]
#selected_features = ['Station', 'is_workday', 'Precipitation', 'Temp_max', 'Temp_min']
