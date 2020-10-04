setwd('/home/joemarlo/Dropbox/Data/Projects/NYC-data/Analyses/Citi-Bike-prediction/Citi-Bike-prediction')
source("Plots/ggplot_settings.R")

# read in the predictions
preds <- read_csv("Predictions/preds.csv") %>% 
  select(Station, y, y_RF, y_xgb)

# densities of each
preds %>% 
  pivot_longer(cols = c("y", "y_RF", "y_xgb")) %>% 
  ggplot(aes(x = value, color = name)) +
  geom_density() +
  scale_x_log10()

# scatter of y_hat vs y by model
preds %>% 
  pivot_longer(cols = c("y_RF", "y_xgb")) %>% 
  ggplot(aes(x = y, y = value, color = name)) +
  geom_point(alpha = 0.05) +
  geom_abline(slope = 1, intercept = 0, color = 'grey30')
  
# RMSE of each
sqrt(mean((preds$y_RF - preds$y)^2))
sqrt(mean((preds$y_xgb - preds$y)^2))

# relative error
mean(preds$y_RF / preds$y) - 1
mean(preds$y_xgb / preds$y) - 1
