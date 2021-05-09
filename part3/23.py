import json
import re

path = 'eu.json'

data= ''

with open(path, 'r') as f:
    for line in f:
        data += line


pattern = r'^(\={2,})\s*(.+?)\s*(\={2,}).*$'
result = '\n'.join(i[1] + ':' + str(len(i[0]) - 1) for i in re.findall(pattern, data, re.MULTILINE))
print(result)
