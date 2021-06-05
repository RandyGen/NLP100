import pprint

fpath = './neko.txt.mecab'

l = []
result = set()
first_f = False
second_f = False

with open(fpath, 'r') as f:
    for line in f:
        if line != 'EOS\n':
            word = line.split('\t')
            if len(word) == 2 and word[0] != '':
                field = word[1].split(',')
                if field[0] == '名詞':
                    l.append(word[0])
                    first_f = True
                elif first_f and field[0] != '名詞':
                    result.add(''.join(l))
                    l = []
                    first_f = False
                else:
                    first_f = False
                    l = []


print(len(result))
pprint.pprint(result)
