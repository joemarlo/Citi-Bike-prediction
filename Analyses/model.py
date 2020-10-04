import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as MSE
import xgboost as xgb
from yellowbrick.regressor import ResidualsPlot
from boruta import BorutaPy

os.chdir('/home/joemarlo/Dropbox/Data/Projects/NYC-data/Analyses/Citi-Bike-prediction/Citi-Bike-prediction')

# read in the data and convert data tyopes
station_trips = pd.read_csv("Data/cleaned_data.csv")
station_trips = station_trips.astype({'Date': 'datetime64'})
station_trips.isnull().sum()

#### quick EDA
# pairs pyplot
# first sample for performance
#samp = station_trips[['Trip_count', 'Week', 'is_workday', 'Precipitation', 'Snowfall', 'Temp_max', 'Temp_min', "Wind_speed_max_5sec"]].sample(n=100)
#sns.pairplot(samp)
#plt.show()

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
# first sample for performance
#samp = station_trips.sample(n=100000)
X = station_trips.drop(["Trip_count", 'Date'], axis=1)
y = station_trips['Trip_count']

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=44, max_iter = 25, perc = 95)

# find all relevant features
feat_selector.fit(np.array(X), y)

# check selected features
selected_features = X.columns[feat_selector.support_]
# ['Station', 'is_workday', 'Precipitation', 'Temp_max', 'Temp_min']
X = station_trips[selected_features]

# scale the data
X.iloc[:, 2:5] = preprocessing.scale(X.iloc[:, 2:5])

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)


##### Randomforest
# Instantiate rf
rf = RandomForestRegressor(n_estimators=50,
                           random_state=44)

# Fit rf to the training set
rf.fit(X_train, y_train)

# Predict the trip count
y_pred = rf.predict(X_test)
#sns.kdeplot(y_pred, shade=True)
#plt.show()

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh')
plt.title('Features Importances')
plt.gcf().set_size_inches(5, 15)
plt.show()

# residuals plot
visualizer = ResidualsPlot(rf)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# boxplot of residuals
ax = sns.boxplot(x=y_pred-y_test)
ax.set(xlabel='Difference b/t actual and prediction')
plt.show()

# add results to dataframe
pred_results = pd.DataFrame(y_test).rename(columns={"Trip_count": "y"})
pred_results = pred_results.merge(X[['Station']], left_index=True, right_index=True)
pred_results['y_RF'] = y_pred



##### xgboost
# Instantiate the XGBRegressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=44, max_depth=10)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse
MSE(y_test, preds)**(1/2)

# boxplot of residuals
ax = sns.boxplot(x=y_pred-y_test)
ax.set(xlabel='Difference b/t actual and prediction')
plt.show()

## Cross validate
# Create the DMatrix: housing_dmatrix
dmatrix = xgb.DMatrix(data=X_train, label=y_train)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":6}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=4, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Create a pd.Series of features importances
importances = pd.Series(data=xg_reg.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh')
plt.title('Features Importances')
plt.gcf().set_size_inches(5, 15)
plt.show()

# add results to dataframe
pred_results['y_xgb'] = preds



# net?
# GAM
# KNN


pred_results.to_csv("Predictions/preds.csv", index=False)
