from sklearn.metrics import accuracy_score
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
train_pred = score_lg(lg, X_train)
test_pred = score_lg(lg, X_test)

# 正解率の測定
train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1])
test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1])
print(f'正解率（学習データ）：{train_accuracy:.3f}')
print(f'正解率（評価データ）：{test_accuracy:.3f}')