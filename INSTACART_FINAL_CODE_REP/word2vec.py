import gensim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    np.random.seed(2017)
    products = pd.read_csv('/Users/ankur/Documents/Competitions/Insta/products.csv')
    df = pd.read_pickle('/Users/ankur/Documents/Competitions/Insta/prod2vec.pkl').products
    print('initial size', len(df))
    
    import os
import numpy as np
import pandas as pd
os.chdir("/Users/ankur/Documents/Competitions/Insta/")
os.getcwd()

#Import Data
prior_orders = pd.read_csv("order_products__prior.csv")
train_orders = pd.read_csv("order_products__train.csv")
products = pd.read_csv("products.csv").set_index('product_id')

train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

sentences = prior_products.append(train_products)
longest = np.max(sentences.apply(len))
sentences = sentences.values

model = gensim.models.Word2Vec(sentences, size=100, window=longest, min_count=2, workers=4)

vocab = list(model.wv.vocab.keys())

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(model.wv.syn0)

