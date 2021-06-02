import MeCab

file = 'neko.txt'

with open(file, "r") as f:
	text = f.read()

t = MeCab.Tagger('')

mecab = t.parse(text)

print('fin')
