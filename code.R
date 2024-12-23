################################################################################
# Load Libraries & Data
################################################################################
install.packages("xgboost")
install.packages("dplyr")
install.packages("caret")
install.packages("devtools")
devtools::install_github(repo = "saberpowers/sabRmetrics")

library(dplyr)
library(xgboost)
library(caret)

savant_data <- sabRmetrics::download_baseballsavant(
  start_date = "2024-01-01",
  end_date = "2024-12-01"
)

# Clean
savant_data <- savant_data %>% 
  select(game_id, game_date, year, batter_id, pitcher_id, batter_name,
         pitch_hand, bat_side, pitch_type,
         az, ay, az, vx0, vy0, vz0, pfx_x, pfx_z, spin_axis, release_spin_rate,
         release_speed, launch_speed, launch_angle, arm_angle, delta_run_exp
         ) %>% 
  filter(!is.na(delta_run_exp))



################################################################################
# Data Manipulations
################################################################################
# Select Features
savant_data0 <- savant_data %>% 
  select(az, ay, az, vx0, vy0, vz0, pfx_x, pfx_z, spin_axis, release_spin_rate,
         release_speed, launch_speed, launch_angle, arm_angle, delta_run_exp)

# Train Test Split
trainIndex <- createDataPartition(savant_data0$delta_run_exp, p = 0.8,
                                  list = FALSE,
                                  times = 1)
train <- savant_data0[trainIndex, ]
test <- savant_data0[-trainIndex, ]

# Separate features (X) and label (y)
train_X <- as.matrix(train %>% select(-delta_run_exp))
train_y <- train$delta_run_exp 
test_X <- as.matrix(test %>% select(-delta_run_exp))
test_y <- test$delta_run_exp
rm(train, test)

# Convert to xgb.DMatrix objects
dtrain <- xgb.DMatrix(data = train_X, label = train_y)
dtest <- xgb.DMatrix(data = test_X, label = test_y)
rm(train_X, train_y)



################################################################################
# Train Model
################################################################################
runtine <- system.time({
  StuffModel <- xgb.train(data = dtrain,
                          max.depth = 2,
                          eta = 0.05,
                          tree_method = "hist",
                          nthread = 2,
                          nrounds = 100,
                          objective = "reg:squarederror",
                          verbosity = 2)
})



################################################################################
# Testing
################################################################################
pred <- predict(StuffModel, test_X)
err <- sqrt(mean((pred - test_y)^2)) # RMSE
print(paste("RMSE: ", err))
cat("Elapsed time:", runtime["elapsed"], "seconds\n")





















