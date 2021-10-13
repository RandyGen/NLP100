import pandas as pd
from sklearn.model_selection import train_test_split

# データの読込
df = pd.read_csv('./data/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# データの抽出
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]

# データの分割
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])

# 事例数の確認
print('【学習データ】')
print(train['CATEGORY'].value_counts())
print('【検証データ】')
print(valid['CATEGORY'].value_counts())
print('【評価データ】')
print(test['CATEGORY'].value_counts())

from gensim.models import KeyedVectors

# 学習済み単語ベクトルの読み込み
model = KeyedVectors.load_word2vec_format(
    './data/GoogleNews-vectors-negative300.bin.gz',
    binary=True
)

import string
import torch

def transform_w2v(text):
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))  # 記号をスペースに変換するtableの作成（replaceの上位互換）
  words = text.translate(table).split()  # 記号をスペースに置換後、スペースで分割してリスト化
  vec = [model[word] for word in words if word in model]  # 1語ずつベクトル化
  # for word in words:
  #   if word in model:
  #     vec.append(model[word])

  return torch.tensor(sum(vec) / len(vec))  # 平均ベクトルをTensor型に変換して出力

# 特徴ベクトルの作成
# stack -> テンソルの結合
X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])

print(X_train.size())
print(X_train)

# ラベルベクトルの作成
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

print(y_train.size())
print(y_train)

# 保存
torch.save(X_train, './data/X_train.pt')
torch.save(X_valid, './data/X_valid.pt')
torch.save(X_test, './data/X_test.pt')
torch.save(y_train, './data/y_train.pt')
torch.save(y_valid, './data/y_valid.pt')
torch.save(y_test, './data/y_test.pt')
