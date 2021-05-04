import json
import gzip

path = 'jawiki-country.json.gz'

new_data = {}

with gzip.open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            new_data.update(data)

j = json.dumps(new_data,ensure_ascii=False)

with gzip.open('eu.json.gz', 'wt') as f:
    f.write(j)
