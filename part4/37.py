import pprint
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

result = []
for i in range(len(l)):
    if l[i] == '猫':
        result.append(l[i-1])
        result.append(l[i+1])


c = collections.Counter(result)
l_k = []
l_v = []
for k, v in c.most_common(10):
    l_k.append(k)
    l_v.append(v)

plt.bar(l_k, l_v, align="center")
plt.title("c.most_common(10)")
plt.xlabel('word')
plt.ylabel('frequency')
plt.grid(True)
plt.show()
