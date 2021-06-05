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
                if field[0] == '名詞' and not second_f:
                    l.append(word[0])
                    first_f = True
                elif first_f and word[0] == 'の':
                    second_f = True
                    l.append(word[0])
                elif second_f and field[0] == '名詞':
                    l.append(word[0])
                    result.add(''.join(l))
                    l = []
                    second_f = False
                else:
                    first_f = False
                    second_f = False
                    l = []


print(len(result))
pprint.pprint(result)
