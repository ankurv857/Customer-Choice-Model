
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

#Preparing prior data for FE

train_map = subset(orders , eval_set == "train")
train_map = data.frame(train_map)
train_map = train_map[c(2)] 
train_map$Flag = "Train"


test_map = subset(orders , orders$eval_set == "test")
test_map = data.frame(test_map)
test_map = test_map[c(2)]
test_map$Flag = "Test"

map_tt = rbind(train_map,test_map)

prior = merge(orderp,orders)
prior = merge(prior,map_tt , by = "user_id")

rm(train_map)
rm(test_map)
rm(map_tt)

#Customer_Segmentation

EDA1 = prior %>% group_by(user_id) %>% summarise(reorder_rate = sum(reordered)/length(reordered) , lengt = length(reordered))
EDA1$dec = ntile(EDA1$reorder_rate, 10)
EDA1 = sqldf("select user_id , reorder_rate , lengt , dec , 
             case when dec in (1,2) then 'Segment1' 
             when dec in (3,4,5) then 'Segment2' 
             when dec in (6,7,8) then 'Segment3' else 'Segment4' end as Segment 
             from EDA1 ")

prior = merge(prior , EDA1 , by = "user_id")
segment_map = EDA1[c(1,2,4,5)]
rm(EDA1)

#Predict the number of prodcts purchased in last order

prior = prior %>% group_by(user_id) %>% mutate(rank_order = dense_rank(desc(order_number)))

dataset = prior %>% group_by(user_id ,rank_order ) %>% summarise(Flag = max(Flag) , Segment = max(Segment) , 
bask_size = length(reordered) , reorder_rate =sum(reordered)/length(reordered) ,order_dow = mean(order_dow),
order_hour_of_day = mean(order_hour_of_day), days_since_prior_order = mean(days_since_prior_order), 
rate = mean(reorder_rate), lengt = mean(lengt)) 

dataset = subset(dataset , Flag == "Test")

dataset_dep = subset(dataset ,rank_order == 1 )
dataset_train = subset(dataset, rank_order > 1 )
dataset_rec = subset(dataset_train, rank_order == 2 )
dataset_lag1 = subset(dataset_train, rank_order == 3 ) 
dataset_lag2 = subset(dataset_train, rank_order == 4 ) 



rm(aisles)
rm(departments)

gc()

data = sqldf("select a.user_id , b.bask_size as bask_size_lag , d.bask_size as bask_size_lag1 , e.bask_size as bask_size_lag2 , avg(a.bask_size) as avg_b,
             max(a.bask_size) as max_b , min(a.bask_size) as min_b , median(a.bask_size) as mode_b ,max(a.Flag)as Flag,
             max(a.Segment) as Segment , max(a.rank_order) as num_orders , avg(a.reorder_rate) as reorder_rate , 
             avg(a.order_dow) as order_dow , avg(a.order_hour_of_day) as order_hour_of_day, 
             avg(a.days_since_prior_order) as days_since_prior_order, avg(a.rate) as rate , avg(a.lengt) as lengt ,
             c.bask_size as dep from dataset_train as a left join dataset_rec as b on a.user_id = b.user_id 
             left join dataset_dep as c on a.user_id = c.user_id  left join dataset_lag1 as d  on a.user_id = d.user_id 
             left join dataset_lag2 as e  on a.user_id = e.user_id group by  a.user_id ")

data$avg_bask_lag = rowMeans(data[c(2:4)] , na.rm = T )
data$bask_size_lag2[is.na(data$bask_size_lag2)] = 0

data1 = sqldf("select a.user_id , b.bask_size as bask_size_lag ,d.bask_size as bask_size_lag1 , e.bask_size as bask_size_lag2, avg(a.bask_size) as avg_b,
              max(a.bask_size) as max_b , min(a.bask_size) as min_b , median(a.bask_size) as mode_b ,max(a.Flag)as Flag,
              max(a.Segment) as Segment , max(a.rank_order) as num_orders , avg(a.reorder_rate) as reorder_rate , 
              avg(a.order_dow) as order_dow , avg(a.order_hour_of_day) as order_hour_of_day, 
              avg(a.days_since_prior_order) as days_since_prior_order, avg(a.rate) as rate , avg(a.lengt) as lengt 
              from dataset as a left join dataset_dep as b on a.user_id = b.user_id  left join dataset_rec as d  on a.user_id = d.user_id
              left join dataset_lag1 as e  on a.user_id = e.user_id group by  a.user_id ")

data1$avg_bask_lag = rowMeans(data1[c(2:4)] , na.rm = T )
data1$bask_size_lag2[is.na(data1$bask_size_lag2)] = 0



colnames(dataset)

rm(dataset_rec)
rm(dataset_train)
rm(dataset_dep)


# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$Flag == "Test",])
train$Flag <- NULL
train$user_id <- NULL
train$Segment = as.factor(train$Segment)

test <- as.data.frame(data1[data1$Flag == "Test",])
test$Flag <- NULL
test$user_id <- NULL
test$Segment = as.factor(test$Segment)

gc()

train = data.frame(train)
test = data.frame(test)
# Model -------------------------------------------------------------------

X_features <- colnames(train[c(1:15,17)])
X_features
X_target <- log(train$dep)

xgtrain <- xgb.DMatrix(data = data.matrix(train[, X_features]), label = X_target)
xgtest <- xgb.DMatrix(data = data.matrix(test[, X_features]))


params <- list()
params$objective <- "reg:linear"
params$booster <- "gbtree"
params$eta <- 0.1
params$max_depth <- 8
params$subsample <- 0.8
params$colsample_bytree <- 0.8
params$min_child_weight <- 2
params$eval_metric <- "rmse"


watchlist<-list(train=xgtrain)

model_xgb_cv <- xgb.cv(params=params, xgtrain, nrounds = 500, nfold = 10, early.stop.round = 30, prediction = TRUE)
model_xgb <- xgb.train(params = params, xgtrain, nrounds = 200 ,early_stopping_rounds = 20, prediction = TRUE,watchlist = watchlist )

pred <- predict(model_xgb, xgtest)
submit <- data.table(user_id = data$user_id ,  predicted = exp(pred))

test_map = subset(orders , orders$eval_set == "test")
test_map = data.frame(test_map)
test_map = test_map[c(1,2)]

submit = merge(submit,test_map)
submit = merge(submit, segment_map)

pred <- predict(model_xgb, xgtrain)
submit <- data.table(actual = train$dep, predicted = exp(pred))
submit$error = (abs((submit$actual) - (submit$predicted))/(submit$actual) ) * 100
mape = mean(submit$error)


colnames(ordert)

# Reshape data ------------------------------------------------------------
aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
orders$eval_set <- as.factor(orders$eval_set)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)
rm(aisles, departments)

ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

orders_products <- orders %>% inner_join(orderp, by = "order_id")

rm(orderp)
gc()


# Products ----------------------------------------------------------------
suppressPackageStartupMessages(library(dplyr))

prd = orders_products %>% arrange(user_id, order_number, product_id) %>% arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>% mutate(product_time = rank(product_id, ties.method = "first"))


prd = prd  %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    prod_orders = n(),
    prod_reorders = sum(reordered),
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time == 2)
  )

prd$prod_reorder_probability <- prd$prod_second_orders / prd$prod_first_orders
prd$prod_reorder_times <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio <- prd$prod_reorders / prd$prod_orders

prd <- prd %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders)

rm(products)
rm(prd1)
gc()

# Users -------------------------------------------------------------------
users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    user_orders = max(order_number),
    user_period = sum(days_since_prior_order, na.rm = T),
    user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T)
  )

us <- orders_products %>%
  group_by(user_id) %>%
  summarise(
    user_total_products = n(),
    user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    user_distinct_products = n_distinct(product_id)
  )

users <- users %>% inner_join(us)
users$user_average_basket <- users$user_total_products / users$user_orders

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

users <- users %>% inner_join(us)

rm(us)
gc()


# Database ----------------------------------------------------------------
data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    up_orders = n(),
    up_first_order = min(order_number),
    up_last_order = max(order_number),
    up_average_cart_position = mean(add_to_cart_order))

rm(orders_products, orders)

data <- data %>% 
  inner_join(prd, by = "product_id") %>%
  inner_join(users, by = "user_id")

data$up_order_rate <- data$up_orders / data$user_orders
data$up_orders_since_last_order <- data$user_orders - data$up_last_order
data$up_order_rate_since_first_order <- data$up_orders / (data$user_orders - data$up_first_order + 1)

data <- data %>% 
  left_join(ordert %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(ordert, prd, users)
gc()



importance <- xgb.importance(colnames(X), model = model)
xgb.ggplot.importance(importance)

rm(X, importance, subtrain)
gc()


# Apply model -------------------------------------------------------------
X <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered <- predict(model, X)

test$reordered <- (test$reordered > 0.21) * 1

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
write.csv(submission, file = "submit1.csv", row.names = F)
