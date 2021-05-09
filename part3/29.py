import json
import re
import pprint
import requests

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

url_file = result['国旗画像'].replace(' ', '_')
url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
data = requests.get(url)
print(re.search(r'"url":"(.+?)"', data.text).group(1))