import json
import gzip

path = 'eu.json.gz'

with gzip.open(path, 'rt') as f:
    for line in f:
        data = json.loads(line)

for row in data['text'].splitlines():
    if 'Category' in row:
        print(row)

