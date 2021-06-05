import pprint

fpath = './neko.txt.mecab'

result = set()

with open(fpath, 'r') as f:
    for line in f:
        if line != 'EOS\n':
            word = line.split('\t')
            if len(word) == 2 and word[0] != '':
                field = word[1].split(',')
                if field[0] == '動詞':
                    result.add(word[0])

print(len(result))
pprint.pprint(result)
