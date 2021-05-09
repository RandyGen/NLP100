import json
import re
import pprint

path = 'eu.json'

data= ''
result = {}

with open(path, 'r') as f:
    for line in f:
        data += line

pattern = r'^\{\{基礎情報.*?$(.*?)^\}\}'
templete = re.findall(pattern, data, re.MULTILINE + re.DOTALL)

# my result
pattern = r'^\|(.*?)\s*=\s*(.+?)\n'
result = dict(re.findall(pattern, templete[0], re.MULTILINE + re.DOTALL))
pprint.pprint(result)

print('------------------------------------------------')

# sample result
pattern = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
result = dict(re.findall(pattern, templete[0], re.MULTILINE + re.DOTALL))
pprint.pprint(result)
