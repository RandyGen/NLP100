# unsolved

import json
import gzip
import re

path = 'eu.json.gz'

with gzip.open(path, 'rt') as f:
    for line in f:
        data = json.loads(line)

for row in data['text'].splitlines():
    if re.findall('^(\={2,})\s*(.+?)\s*(\={2,}).*$', row):
        print(row)
        name = ''.join(re.findall('^(\={2,})\s*(.+?)\s*(\={2,}).*$', row))
        # name_revel = ''.join(re.findall(name + '(.+)', row))
        print(f'{name}')

        # name = ''.join(re.findall('(.+)={2,5}', name))
        # # name_revel = ''.join(re.findall(name + '(.+)', row))
        # print(f'{name}')


