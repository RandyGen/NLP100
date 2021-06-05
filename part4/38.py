import collections
import matplotlib.pyplot as plt

fpath = './neko.txt.mecab'

l = []

with open(fpath, 'r') as f:
    for line in f:
        if line != 'EOS\n':
            word = line.split('\t')
            if len(word) == 2 and word[0] != '':
                l.append(word[0])

c = collections.Counter(l)
l_v = []
for k, v in c.most_common():
    l_v.append(v)

# plt.figure(figsize=(8, 4))
plt.hist(l_v, bins=100)
plt.title("c.most_common()")
plt.xlabel('word')
plt.ylabel('frequency')
plt.grid(True)
plt.show()
