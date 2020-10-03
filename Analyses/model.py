


# weekday and not holiday
# not raining
# morning commuting hours and evening commuting hours
# month
# day of week
# max temperature
# avg wind speed
# snow / precip
# gender
# age
# greater of start/stop bike station from distance from subway station



# dummy code



# filter outliers
outlier_bounds <- quantile(sampled_data$Trip_duration_seconds, c(0.025, 0.975))
sampled_data <- sampled_data %>%
  filter(Trip_duration_seconds >= outlier_bounds[[1]],
         Trip_duration_seconds <= outlier_bounds[[2]])
