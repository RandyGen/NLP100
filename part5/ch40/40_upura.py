class Morph:
    def __init__(self, dc):
        self.surface = dc['surface']
        self.base = dc['base']
        self.pos = dc['pos']
        self.pos1 = dc['pos1']


def parse_cabocha(block):
    res = []
    # 引数である一文から１行ずつ処理
    for line in block.split('\n'):
        # 文末
        if line == '':
            return res

        # 係り受け解析結果行
        elif line[0] == '*':
            continue
        
        # else その他行
        (surface, attr) = line.split('\t', 1)
        attr = attr.split(',')
        lineDict = {
            'surface': surface,
            'base': attr[6],
            'pos': attr[0],
            'pos1': attr[1]
        }
        res.append(Morph(lineDict))


if __name__ == '__main__':
    filename = '../data/src/ai.ja.text.parsed'

    # 係り受け解析結果をEOSごとに読み込み
    with open(filename, mode='rt', encoding='utf-8') as f:
        blocks = f.read().split('EOS\n')

    # リスト中の空の要素を削除
    blocks = list(filter(lambda x: x != '', blocks))

    # 一文ごとに処理しリスト化
    blocks = [parse_cabocha(block) for block in blocks]

    # 出力
    for m in blocks[2]:
        print(vars(m))
