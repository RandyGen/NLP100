import pprint
import collections

fpath = './neko.txt.mecab'

l = []

with open(fpath, 'r') as f:
    for line in f:
        if line != 'EOS\n':
            word = line.split('\t')
            if len(word) == 2 and word[0] != '':
                l.append(word[0])

c = collections.Counter(l)
pprint.pprint(c.most_common())
