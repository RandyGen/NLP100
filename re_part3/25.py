# unsolved
import json
import re

path = 'eu.json'

data= ''

with open(path, 'r') as f:
    for line in f:
        data += line


pattern = r"^.*\{\{基礎情報(.*)\}\}.*$"
info = re.findall(pattern, data, re.MULTILINE)

print(info)

pattern = r"^\|(.+?)(?:\s=\s*.*)$"
result = '\n'.join(re.findall(pattern, data))
print(result)
