setwd('/home/joemarlo/Dropbox/Data/Projects/NYC-data/Analyses/Citi-Bike-prediction/Citi-Bike-prediction')
source("Plots/ggplot_settings.R")

library(MASS)
library(lme4)
options(mc.cores = parallel::detectCores())
set.seed(44)

# station_trips <- read_csv('Data/cleaned_data.csv')
X_train <- read_csv('Predictions/X_train.csv')
X_test <- read_csv('Predictions/X_test.csv')
y_train <- read_csv('Predictions/y_train.csv')
y_test <- read_csv('Predictions/y_test.csv')

# prep the data -----------------------------------------------------------

# undummy Month
X_train$Month <- X_train %>% 
  dplyr::select(starts_with("Month")) %>%
  apply(., 1, function(row){
    which(row == 1)
  })
X_train <- X_train %>% dplyr::select(!(starts_with("Month_")))
X_test$Month <- X_test %>% 
  dplyr::select(starts_with("Month")) %>%
  apply(., 1, function(row){
    which(row == 1)
  })
X_test <- X_test %>% dplyr::select(!(starts_with("Month_")))


X_train$Station <- as.factor(X_train$Station)
X_train$Month <- as.factor(X_train$Month)
X_test$Station <- as.factor(X_test$Station)
X_test$Month <- as.factor(X_test$Month)

# is this really a poisson variable?
y_train %>% 
  ggplot(aes(x = Trip_count)) +
  geom_density()
mean(y_train$Trip_count)
var(y_train$Trip_count)

X_train$Trip_count <- y_train$Trip_count
X_test$Trip_count <- y_test$Trip_count

# fit models --------------------------------------------------------------

# sample stratified on station for computational performance
samp_train <- X_train %>% 
  group_by(Station) %>% 
  slice_sample(n = 10)

# fit quasipoisson and predict
qpoisson <- glm(Trip_count ~ ., data = samp_train, family = quasipoisson(link = 'log'))
broom::tidy(qpoisson)
qp_preds <- predict(qpoisson, newdata = X_test)
sqrt(mean((qp_preds - y_test$Trip_count)^2))

# fit negative binomial
neg_binom <- MASS::glm.nb(Trip_count ~ ., data = samp_train)
broom::tidy(neg_binom)
nb_preds <- predict(neg_binom, newdata = X_test)
sqrt(mean((qp_preds - y_test$Trip_count)^2))

# fit mlm
# station as random-effect intercept and Year as both fixed and random effect slope
# mlm isn't going to work need many observations per station and performance isn't good enough
# bayesian may work

# neg_binom_mlm <- lme4::glmer.nb(Trip_count ~ Month + is_workday + Precipitation + Temp_max + Temp_min + Wind_speed_max_5sec + (Month | Station), 
#                                 data = samp_train, verbose = TRUE)


# add preds to csv --------------------------------------------------------

read_csv("Predictions/preds.csv") %>% 
  mutate(y_qp = qp_preds,
         y_nb = nb_preds) %>% 
  write_csv("Predictions/preds.csv")
