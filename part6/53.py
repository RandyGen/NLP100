import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 予測用関数
def score_lg(lg, X):
  return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]

# データの読み込み
train = pd.read_csv('./data/train.txt', sep='\t')
valid = pd.read_csv('./data/valid.txt', sep='\t')
X_train = pd.read_csv('./data/X_train.txt', sep='\t')
X_valid = pd.read_csv('./data/X_valid.txt', sep='\t')


# モデルの学習
lg = LogisticRegression(random_state=123, max_iter=10000)
lg.fit(X_train, train['CATEGORY'])

# モデルの予測
train_pred = score_lg(lg, X_train)
test_pred = score_lg(lg, X_valid)

print(train_pred)
