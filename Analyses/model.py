import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import mean_squared_error as MSE
from yellowbrick.regressor import ResidualsPlot
from boruta import BorutaPy

os.chdir('/home/joemarlo/Dropbox/Data/Projects/NYC-data/Analyses/Citi-Bike-prediction/Citi-Bike-prediction')

# read in the data and convert data tyopes
station_trips = pd.read_csv("Data/cleaned_data.csv")
station_trips = station_trips.astype({'Date': 'datetime64', 'is_holiday': 'int64'})
station_trips = station_trips.rename(columns={'dist_to_subway': 'Dist_to_subway'})
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
#rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

# define Boruta feature selection method
#feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=44, max_iter = 25, perc = 95)

# find all relevant features
#feat_selector.fit(np.array(X), y)

# features selected by the Boruta algo
#selected_features = X.columns[feat_selector.support_]
#selected_features = ['Station', 'is_workday', 'Precipitation', 'Temp_max', 'Temp_min']

# Joe's selected features
selected_features = ['Dist_to_subway', 'Station', "Year", 'Month', 'is_workday', 'Precipitation', 'Temp_max', 'Temp_min', 'Wind_speed_max_5sec']
X = station_trips[selected_features]

# dummy code Year and Month
X = pd.get_dummies(X, columns=['Year', 'Month'])

# scale the data
#X.iloc[:, 2:5] = preprocessing.scale(X.iloc[:, 2:5])
X.loc[:, ['Dist_to_subway', 'Precipitation', 'Temp_max', 'Temp_min', 'Wind_speed_max_5sec']] = preprocessing.scale(X.loc[:, ['Dist_to_subway', 'Precipitation', 'Temp_max', 'Temp_min', 'Wind_speed_max_5sec']])

# hash the station column (dummy coding takes too much memory)
#X = X.astype({"Station": 'category'})
#hasher = FeatureHasher(n_features=len(X.Station.unique()))
#hasher.transform(X.Station)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# write out
X_train.to_csv("Predictions/X_train.csv", index=False)
X_test.to_csv("Predictions/X_test.csv", index=False)
y_train.to_csv("Predictions/y_train.csv", index=False)
y_test.to_csv("Predictions/y_test.csv", index=False)

##### Randomforest
# Instantiate rf
rf = RandomForestRegressor(n_estimators=100, random_state=44, n_jobs=-1)

# Fit rf to the training set
rf.fit(X_train, y_train)

# Predict the trip count
y_pred = rf.predict(X_test)
#sns.kdeplot(y_pred, shade=True)
#plt.show()

# Evaluate the test set RMSE
MSE(y_test, y_pred)**(1/2)

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
xg_reg = xgb.XGBRegressor(objective='count:poisson', n_estimators=100, seed=44, max_depth=15, n_jobs=-1)

#RMSE ~ 22 @ max_depth=10
#RMSE ~ 18 @ max_depth=15
#RMSE ~ 19 @ max_depth=20
#RMSE ~ 22 @ max_depth=25

## old
#RMSE ~ 38 @ max_depth=3
#RMSE ~ 27 @ max_depth=10
#RMSE ~ 24 @ max_depth=20
#RMSE ~ 32 @ max_depth=30

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
# Create the DMatrix
dmatrix = xgb.DMatrix(data=X_train, label=y_train)

# Create the parameter dictionary: params
params = {"objective":"reg:poisson", "max_depth":6}

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

# GAM
from pygam import LinearGAM, s, f, GAM
gam = LinearGAM(n_splines=4).gridsearch(X_train.values, y_train.values)

# get the preds
gam_preds = gam.predict(X_test)

# Compute the rmse
MSE(y_test, gam_preds)**(1/2)

# add results to dataframe
pred_results['y_GAM'] = gam_preds

# KNN
from sklearn import neighbors
knn_reg = neighbors.KNeighborsRegressor(n_neighbors=5)

# fit the knn model
knn_reg.fit(X_train, y_train)

# make the preds
knn_preds = knn_reg.predict(X_test)

# Compute the rmse
MSE(y_test, knn_preds)**(1/2)

# add results to dataframe
pred_results['y_knn'] = knn_preds

# lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

n_folds = 5
alphas = np.logspace(-4,-0.5,30)
tunedparameters = [{"C":alphas}]

lassomod = LogisticRegression(penalty = "l1", solver="liblinear", random_state = 0, max_iter = 10000)

clf = GridSearchCV(lassomod, tunedparameters, cv = n_folds)
clf.fit(X_train, y_train)
clf.predict(X_test)

# perceptron regressor
from sklearn.neural_network import MLPRegressor
mlpr = MLPRegressor(random_state=44, max_iter=500, verbose=True, early_stopping=True).fit(X_train, y_train)

# make the preds
mlpr_preds = mlpr.predict(X_test)

# Compute the rmse
MSE(y_test, mlpr_preds)**(1/2)

# add results to dataframe
pred_results['y_mlpr'] = mlpr_preds

# write out dataframe
pred_results.to_csv("Predictions/preds.csv", index=False)



#### TEST: dummy code station by first sampling (otherwise memory problems)
samp = station_trips.sample(n=100000)
X = samp.drop(["Trip_count", 'Date'], axis=1)
y = samp['Trip_count']

# Joe's selected features
selected_features = ['Dist_to_subway', 'Station', "Year", 'Month', 'is_workday', 'Precipitation', 'Temp_max', 'Temp_min', 'Wind_speed_max_5sec']
X = samp[selected_features]

# dummy code Year, Month, and Station
X = pd.get_dummies(X, columns=['Year', 'Month', 'Station'])

# scale the data
X.loc[:, ['Dist_to_subway', 'Precipitation', 'Temp_max', 'Temp_min', 'Wind_speed_max_5sec']] = preprocessing.scale(X.loc[:, ['Dist_to_subway', 'Precipitation', 'Temp_max', 'Temp_min', 'Wind_speed_max_5sec']])

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Instantiate rf
rf = RandomForestRegressor(n_estimators=100, random_state=44, n_jobs=-1)

# Fit rf to the training set
rf.fit(X_train, y_train)

# Predict the trip count
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
MSE(y_test, y_pred)**(1/2)
# RMSE is ~30
