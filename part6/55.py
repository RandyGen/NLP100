from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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

# 学習データ
train_cm = confusion_matrix(train['CATEGORY'], train_pred[1])
print(train_cm)
sns.heatmap(train_cm, annot=True, cmap='Blues')
plt.savefig('output/train_cm.png')

# 評価データ
test_cm = confusion_matrix(test['CATEGORY'], test_pred[1])
print(test_cm)
sns.heatmap(test_cm, annot=True, cmap='Blues')
plt.savefig('output/test_cm.png')
