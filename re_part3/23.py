import json
import re

path = 'eu.json'

data= ''

with open(path, 'r') as f:
    for line in f:
        data += line


pattern = r"^(=+)\s*(.*?)\s*(=+)$"
result = re.findall(pattern, data, re.MULTILINE)
for i in range(len(result)):
    print('{}: {}'.format(result[i][1], len(result[i][0])-1))