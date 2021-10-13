from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format(
    './data/GoogleNews-vectors-negative300.bin.gz',
    binary=True
)

ws = []
with open('data/wordsim353/combined.csv', 'r') as f:
    next(f) # 1行目を飛ばす
    for line in f:
        line = [s.strip() for s in line.split(',')]
        line.append(model.similarity(line[0], line[1]))
        ws.append(line)

import numpy as np
from scipy.stats import spearmanr

# スピアマン相関係数の計算
human = np.array(ws).T[2]
w2v = np.array(ws).T[3]
correlation, pvalue = spearmanr(human, w2v)

print(f'スピアマン相関係数: {correlation:.3f}')