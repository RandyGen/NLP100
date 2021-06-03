# unsolved
import json
import re
import pprint

path = 'eu.json'

data= ''

with open(path, 'r') as f:
    for line in f:
        data += line

# .*? .* の違い
# re.DOTALL $ の使い方

pattern = r"^.*\{\{基礎情報.*?$(.*)^\}\}.*"
info = re.findall(pattern, data, re.MULTILINE + re.DOTALL)

# print(info)


# \s \s+ の違い
pattern = r"^\|(.+?)\s*=\s*(.+?)\n"
result = dict(re.findall(pattern, data, re.MULTILINE + re.DOTALL))
pprint.pprint(result)
