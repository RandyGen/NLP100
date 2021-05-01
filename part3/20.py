import json

json_open = open('jawiki-country.json', 'r')
json_load = json.load(json_open)

for v in json_load.values():
    if v['title'] == 'イギリス':
        print(v['text'])
