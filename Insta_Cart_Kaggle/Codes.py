import os
import numpy as np
import pandas as pd
os.chdir("/Users/ankur/Documents/Competitions/Insta/")
os.getcwd()

#Import Data
orders = pd.read_csv("orders.csv")
order_products__prior = pd.read_csv("order_products__prior.csv")
order_products__train = pd.read_csv("order_products__train.csv")
products = pd.read_csv("products.csv")
departments = pd.read_csv("departments.csv")
aisles = pd.read_csv("aisles.csv")
order_products__test = pd.read_csv("sample_submission.csv")
#Merge Order Products
order_products__test.drop('products' , axis = 1, inplace=True)
order_products__prior['Flag'] = "P"
order_products__train['Flag'] = "T"
order_products__test['product_id'] = -1
order_products__test['add_to_cart_order'] = -1
order_products__test['reordered'] = -1
order_products__test['Flag'] = "D"
order_products = order_products__prior.append([order_products__train,order_products__test])

trans1 = pd.merge(orders,order_products, how='left' , on= ['order_id'])
trans2 = pd.merge(trans1,products, how='left' , on= ['product_id'])
trans2 = trans2[['user_id' , 'order_id', 'order_number' , 'order_dow' , 'order_hour_of_day', 'add_to_cart_order',
                'product_id' , 'product_name' , 'aisle_id' , 'department_id', 'days_since_prior_order' , 'reordered' ,
                'eval_set', 'Flag']]

#check = trans2[(trans2['user_id'] == 36855) | (trans2['user_id'] ==  35220) | (trans2['user_id']  == 35794 ) |  
#               (trans2['user_id'] == 142891 )]
#check.to_csv('check.csv', index=False, header=True)


trans3 = trans2[trans2['product_id'] != -1]
trans3["Order_Rank"] = trans3.groupby("user_id")["order_number"].rank(ascending = False , method='dense')
dep_sub = trans3[trans3['Order_Rank'] == 1]
train_sub = trans3[trans3['Order_Rank'] != 1]
train_sub1 = train_sub[['user_id','department_id']]
train_sub1.drop_duplicates()
