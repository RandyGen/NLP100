import pandas as pd
from sklearn.linear_model import LogisticRegression

# データの読み込み
train = pd.read_csv('./data/train.txt', sep='\t')
X_train = pd.read_csv('./data/X_train.txt', sep='\t')


# モデルの学習
lg = LogisticRegression(random_state=123, max_iter=10000)
lg.fit(X_train, train['CATEGORY'])
