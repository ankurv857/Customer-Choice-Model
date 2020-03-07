library(data.table)
library(dplyr)
library(tidyr)
library(sqldf)
library(xgboost)



# Load Data ---------------------------------------------------------------
path <- "/Users/ankur/Documents/Competitions/Insta"
setwd(path)
getwd()


order_train <- fread(file.path(path, "order_train_new.csv"))
order_test <- fread(file.path(path, "order_test_new.csv"))

order_train$dep = ifelse(order_train$reordered == "TRUE" , 1,0)
order_train$reordered = NULL

#train & test

order_train = data.frame(order_train)
train = order_train
train[is.na(train)] = 0

order_test = data.frame(order_test)
test = order_test
test[is.na(test)] = 0

colnames(train)
colnames(test)



# Model -------------------------------------------------------------------
X_features = c("user_product_reordered_ratio", "reordered_sum",
"add_to_cart_order_inverted_mean", "add_to_cart_order_relative_mean",
"reorder_prob",
"last", "prev1", "prev2", "median", "mean",
"dep_reordered_ratio", "aisle_reordered_ratio",
"aisle_products",
"aisle_reordered",
"dep_products",
"dep_reordered",
"prod_users_unq", "prod_users_unq_reordered",
"order_number", "prod_add_to_card_mean",
"days_since_prior_order",
"order_dow", "order_hour_of_day",
"reorder_ration",
"user_orders", "user_order_starts_at", "user_mean_days_since_prior",
"user_average_basket", "user_distinct_products", "user_reorder_ratio", "user_total_products",
"prod_orders", "prod_reorders",
"up_order_rate", "up_orders_since_last_order", "up_order_rate_since_first_order",
"up_orders", "up_first_order", "up_last_order", "up_mean_cart_position",
"days_since_prior_order_mean",
"order_dow_mean",
"order_hour_of_day_mean" , "feature_0" , "feature_1" , "feature_2" ,
"feature_3" , "feature_4" ,"feature_5" , "feature_6" , "feature_7" , "feature_8" , "feature_9" ,"feature_10" ,
"feature_11" , "feature_12" , "feature_13" , "feature_14" , "feature_15" , "feature_16" , "feature_17" ,
"feature_18" ,"feature_19" , "feature_20" , "feature_21" , "feature_22" ,"feature_23" ,"feature_24" ,
"feature_25" ,"feature_26" , "feature_27" , "feature_28" , "feature_29" , "feature_30" , "feature_31")

X_features

subtrain <- train %>% sample_frac(1)
X_target <- subtrain$dep

xgtrain <- xgb.DMatrix(data = data.matrix(subtrain[, X_features]), label = X_target)
xgtest <- xgb.DMatrix(data = data.matrix(test[, X_features]))

params <- list(
  "objective"           = "binary:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.05,
  "max_depth"           = 8,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 0.86,
  "colsample_bytree"    = 0.85,
  "alpha"               = 2e-05,
  "lambda"              = 10
)

watchlist<-list(train=xgtrain)
model_xgb <- xgb.train(params = params, xgtrain, nrounds = 100 ,early_stopping_rounds = 20, prediction = TRUE,watchlist = watchlist )


# Apply model -------------------------------------------------------------

test$reordered <- predict(model_xgb,xgtest )
test$reordered <- (test$reordered > 0.2) * 1

test1 = cbind(order_id = order_test$order_id , product_id = order_test$product_id ,reordered = test$reordered)
test1 = data.frame(test1)

#test1 = cbind(order_id = order_test$order_id , product_id = order_test$product_id ,reordered = test$reordered)
#write.csv(test1 , "test1_v2.csv" , row.names = FALSE)

te1 <- fread(file.path(path, "test1_3845.csv"))
te2 <- fread(file.path(path, "test1_v2.csv"))

te = cbind(order_id = te1$order_id , product_id = te1$product_id , reordered = (0.5* te1$reordered + 0.5*te2$reordered) )
te = data.frame(te)
te$reordered <- (te$reordered > 0.205) * 1

test1 = te

submission <- test1 %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )


missing <- data.frame(
  order_id = unique(test1$order_id[!test1$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit_last.csv", row.names = F)

