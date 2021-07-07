# README
## ai.ja.text.parsedの作成方法
係り受け解析は__ginza__の__cabocha__を用いて作成しました
  
```
pip install -U ginza
ginza -f cabocha ai.ja.txt > ai.ja.text.parsed
```
  
そのため__cabocha__をインストールして行う従来の方法でやっていないため他のサイトにあるコードを真似てもエラーが発生する可能性があります
  
ex. 40.py line 19  

```  
surface, attr = line.split('\t')
# ValueError too many vallues to unpack (expeted 2)
```
  
```
surface, attr = line.split('\t', 1)
```

README作成日：2021/07/07