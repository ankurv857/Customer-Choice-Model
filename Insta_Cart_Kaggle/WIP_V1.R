
library(data.table)
library(dplyr)
library(tidyr)
library(sqldf)
library(xgboost)



# Load Data ---------------------------------------------------------------
path <- "/Users/ankur/Documents/Competitions/Insta"
setwd(path)
getwd()


aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))
ordert <- fread(file.path(path, "order_products__train.csv"))
orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))

#User Features
products <- products %>% inner_join(aisles) %>% inner_join(departments) %>% select(-aisle_id, -department_id)
rm(aisles, departments)
train_map  =  orders %>% filter(eval_set == "train") %>% select(user_id , flag = eval_set)
test_map  =  orders %>% filter(eval_set == "test") %>% select(user_id , flag = eval_set)
train_test = rbind(train_map,test_map)

prior = orderp %>% inner_join(orders , by = "order_id") %>% inner_join(train_test , by = "user_id") %>% inner_join(products , by = "product_id")
prior = prior %>% group_by(user_id) %>% mutate(rank_order = dense_rank(desc(order_number)))
prior = prior  %>% group_by(user_id, product_id) %>%  mutate(prod_rank_order = dense_rank(desc(order_number)))
#users
uror = prior %>% group_by(user_id) %>% summarise(uror = sum(reordered) / sum(order_number > 1) , ul  = length(reordered) , 
       udl = length(unique(product_id)) , num_baskets = max(order_number) , prod_ro_density = mean(prod_rank_order) ,
       avg_basket_size = length(reordered)/max(order_number) ,basket_size_lag = sum(rank_order == 1)  ,basket_size_lag1 = sum(rank_order == 2) ,  
       aog = mean(days_since_prior_order, na.rm = T))
#prods
pror = prior %>% group_by(product_id) %>% summarise(pror = sum(reordered) / length(reordered) , pl = length(reordered) , 
       pur  = length(unique(user_id*reordered > 0))/length(unique(user_id)) , prod_reorders = sum(reordered),prod_last_orders = sum(prod_rank_order == 1),
       prod_secondlast_orders = sum(prod_rank_order == 2) , ror_prob = sum(prod_rank_order == 2)/sum(prod_rank_order == 1))
#user_prods
upror = prior %>% group_by(user_id,product_id) %>% summarise(upor = length(order_number - 1 ) , upro = sum(reordered) ,latest_or = min(rank_order) , 
        max_prod_ord = max(prod_rank_order) , avg_prod_ord = mean(prod_rank_order) ,  up_first_order = min(order_number), up_last_order = max(order_number), 
        up_average_cart_position = mean(add_to_cart_order) ,up_order_rate = n() , upaog = mean(days_since_prior_order, na.rm = T) )
#user_aisle
uaror = prior %>% group_by(user_id,aisle) %>% summarise(upor_a = length(order_number - 1 ) , upro_a = sum(reordered) ,latest_or_a = min(rank_order) , 
        max_prod_ord_a = max(prod_rank_order) , avg_prod_ord_a = mean(prod_rank_order) ,  up_first_order_a = min(order_number), up_last_order_a = max(order_number), 
        up_average_cart_position_a = mean(add_to_cart_order) ,up_order_rate_a = n() , upaog_a = mean(days_since_prior_order, na.rm = T) )
#user_departments
udror = prior %>% group_by(user_id,department) %>% summarise(upor_d = length(order_number - 1 ) , upro_d = sum(reordered) ,latest_or_d = min(rank_order) , 
        max_prod_ord_d = max(prod_rank_order) , avg_prod_ord_d = mean(prod_rank_order) ,  up_first_order_d = min(order_number), up_last_order_d = max(order_number), 
        up_average_cart_position_d = mean(add_to_cart_order) ,up_order_rate_d = n() , upaog_d = mean(days_since_prior_order, na.rm = T) )

#dataset
data <- upror %>% inner_join(products %>% select(product_id,aisle,department ), by = "product_id") %>% inner_join(pror, by = "product_id") %>% 
        inner_join(uror, by = "user_id") %>% inner_join(uaror, by = c("user_id", "aisle")) %>% inner_join(udror, by = c("user_id", "department")) 

data$up_order_rate_p <- data$up_order_rate / data$num_baskets
data$up_order_rate_a <- data$up_order_rate_a/ data$num_baskets
data$up_order_rate_d <- data$up_order_rate_d / data$num_baskets

data$up_orders_since_last_order_p <- data$num_baskets - data$up_last_order
data$up_orders_since_last_order_a <- data$num_baskets - data$up_last_order_a
data$up_orders_since_last_order_d <- data$num_baskets - data$up_last_order_d

data$up_order_rate_since_first_order_p <- data$up_order_rate / (data$num_baskets - data$up_first_order + 1)
data$up_order_rate_since_first_order_a <- data$up_order_rate_a / (data$num_baskets - data$up_first_order_a + 1)
data$up_order_rate_since_first_order_d <- data$up_order_rate_d / (data$num_baskets - data$up_first_order_d + 1)

ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]
data <- data %>% left_join(ordert %>% select(user_id, product_id, reordered), by = c("user_id", "product_id"))
data = data %>% inner_join(train_test, by = "user_id")
rm(check , orderp, orders , ordert , pror, uaror,udror,upror,uror)

# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$flag == "train",])
train$flag <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$department = as.factor(train$department)
train$aisle = as.factor(train$aisle)
train[is.na(train)] <- 0

test <- as.data.frame(data[data$flag == "test",])
test$flag <- NULL
test$department = as.factor(test$department)
test$aisle = as.factor(test$aisle)
test$reordered <- NULL
test[is.na(test)] <- 0

rm(data)
gc()


# Model -------------------------------------------------------------------

X_features <- colnames(train[c(1:55)])
X_features

subtrain <- train %>% sample_frac(1)
X_target <- subtrain$reordered

xgtrain <- xgb.DMatrix(data = data.matrix(subtrain[, X_features]), label = X_target)
xgtest <- xgb.DMatrix(data = data.matrix(test[, X_features]))

params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.1,
  "max_depth"           = 6,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 0.76,
  "colsample_bytree"    = 0.95,
  "alpha"               = 2e-05,
  "lambda"              = 10
)

watchlist<-list(train=xgtrain)
model_xgb <- xgb.train(params = params, xgtrain, nrounds = 50 ,early_stopping_rounds = 20, prediction = TRUE,watchlist = watchlist )


# Apply model -------------------------------------------------------------

test$reordered <- predict(model_xgb,xgtest )
test$reordered <- (test$reordered > .25) * 1

order_map  =  orders %>% filter(eval_set == "test") %>% select(user_id , order_id)
test = test %>% inner_join(order_map, by = "user_id")

submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )


missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit.csv", row.names = F)
