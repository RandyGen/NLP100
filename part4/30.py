fpath = './neko.txt.mecab'

row = []
result = []

with open(fpath, 'r') as f:
	for line in f:
		if line != 'EOS\n':
			word = line.split('\t')
			if len(word) == 2 and word[0] != '':
				field = word[1].split(',')
				info = {
					'surface': word[0],
					'base': field[-3],
					'pos': field[0],
					'pos1': field[1]
				}
				row.append(info)
		else:
			result.append(row)
			row = []

print(result)
