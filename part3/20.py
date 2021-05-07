import json

path = 'jawiki-country.json'

new_data = ''

with open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            new_data += data["text"]


with open('eu.json', 'w') as f:
    f.write(new_data)
