import json
import gzip
import re

path = 'eu.json.gz'

with gzip.open(path, 'rt') as f:
    for line in f:
        data = json.loads(line)

for row in data['text'].splitlines():
    row_data = row.replace("[", "").replace("]", "")
    if re.match('Category', row_data):
        print(''.join(re.findall('Category:(.*)', row_data)))

