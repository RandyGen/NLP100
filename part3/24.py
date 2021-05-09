import json
import re

path = 'eu.json'

data= ''

with open(path, 'r') as f:
    for line in f:
        data += line

pattern = r'\[\[ファイル:(.+?)\|'
result = '\n'.join(re.findall(pattern, data, re.MULTILINE))
print(result)
