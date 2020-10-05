setwd('/home/joemarlo/Dropbox/Data/Projects/NYC-data/Analyses/Citi-Bike-prediction/Citi-Bike-prediction')
source("Plots/ggplot_settings.R")

# read in the predictions and calc mean and median
preds <- read_csv("Predictions/preds.csv") %>% 
  dplyr::select(Station, y, y_RF, y_xgb, y_GAM, y_knn, y_mlpr) #, y_qp, y_nb)

# check RMSE and relative error for each
(RMSE <- apply(preds, 2, function(col) sqrt(mean((col - preds$y)^2))))
apply(preds, 2, function(col) mean(col / preds$y) - 1)
apply(preds, 2, function(col) mean(col <= 0))

# create table of RMSEs and matching labels for plot
RMSE <- enframe(RMSE) %>%
  mutate(label = paste0(
    substr(name, 3, 10), 
    ": RMSE of ", 
    scales::comma_format(accuracy = 0.1)(value))
    ) %>% 
  dplyr::select(name, RMSE = value, label)

# densities of each
preds %>% 
  pivot_longer(cols = starts_with("y_")) %>% 
  left_join(RMSE, by = 'name') %>% 
  mutate(label = fct_reorder(label, RMSE)) %>% 
  # filter(!(name %in% c("y_qp", "y_nb"))) %>% 
  ggplot() +
  geom_density(aes(x = y), fill = 'grey60', color = 'white', alpha = 0.8) +
  geom_density(aes(x = value, fill = name), color = 'white', alpha = 0.5) +
  scale_x_log10(limits = c(1, 1000),
                labels = scales::comma_format(accuracy = 1)) +
  facet_wrap(~label) +
  labs(title = "Comparison of y_hat across various models",
       subtitle = 'Grey density is y',
       x = 'y and y_hat',
       y = NULL) +
  theme(legend.position = 'none')
ggsave("Plots/y_hat_densities.png",
       width = 8,
       height = 6)

# scatter of y_hat vs y by model
preds %>% 
  slice_sample(n = 10000) %>% 
  pivot_longer(cols = starts_with("y_")) %>% 
  left_join(RMSE, by = 'name') %>% 
  mutate(label = fct_reorder(label, RMSE)) %>% 
  ggplot(aes(x = y, y = value, color = name)) +
  geom_point(alpha = 0.05) +
  geom_abline(slope = 1, intercept = 0, 
              color = 'grey50', alpha = 0.7) +
  facet_wrap(~label) +
  labs(title = "y vs. y_hat across various models",
       x = 'y',
       y = 'y_hat') +
  theme(legend.position = 'none')
ggsave("Plots/y_v_y_hat.png",
       width = 8,
       height = 6)
