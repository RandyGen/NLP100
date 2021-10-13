import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# 予測用関数
def score_lg(lg, X):
  return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]

# データの読み込み
train = pd.read_csv('./data/train.txt', sep='\t')
test = pd.read_csv('./data/test.txt', sep='\t')
X_train = pd.read_csv('./data/X_train.txt', sep='\t')
X_test = pd.read_csv('./data/X_test.txt', sep='\t')


# モデルの学習
lg = LogisticRegression(random_state=123, max_iter=10000)
lg.fit(X_train, train['CATEGORY'])

# モデルの予測
test_pred = score_lg(lg, X_test)

# 特徴量の重みの確認
features = X_train.columns.values
index = [i for i in range(1, 11)]
for c, coef in zip(lg.classes_, lg.coef_):
  print(f'【カテゴリ】{c}')
  best10 = pd.DataFrame(features[np.argsort(coef)[::-1][:10]], columns=['重要度上位'], index=index).T
  worst10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=['重要度下位'], index=index).T
  print(pd.concat([best10, worst10], axis=0))
  print('\n')
