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

pattern = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
result = dict(re.findall(pattern, templete[0], re.MULTILINE + re.DOTALL))

pattern = r'\'{2,5}'
result_rm = {k: re.sub(pattern, '', v) for k, v in result.items()}
pprint.pprint(result_rm)
