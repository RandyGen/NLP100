import json 

filename = 'jawiki-country.json'
with open(filename, mode='r') as f:
    for line in f:
        line = json.loads(line)
        if line['title'] == 'イギリス':
            text_uk = line['text']
            break

print(type(text_uk))

import re

pattern = r'^(.*\[\[Category:.*\]\].*)$'
result = '\n'.join(re.findall(pattern, text_uk, re.MULTILINE))
print(result)

pattern = r'^.*\[\[Category:(.*?)(?:\|.*)?\]\].*$'
result = '\n'.join(re.findall(pattern, text_uk, re.MULTILINE))
print(result)

pattern = r'^(\={2,})\s*(.+?)\s*(\={2,}).*$'
result = '\n'.join(i[1] + ':' + str(len(i[0]) - 1) for i in re.findall(pattern, text_uk, re.MULTILINE))
print(result)