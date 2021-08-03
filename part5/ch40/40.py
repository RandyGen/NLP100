class Morphs:
    def __init__(self, morphs):
        (surface, attr) = morphs.split('\t', 1) # 行を最初の\tで区切る
        attr = attr.split(',')
        self.surface = surface # 表層形
        self.base = attr[5]    # 基本形
        self.pos = attr[0]     # 品詞形
        self.pos1 = attr[1]    # 品詞細分類


if __name__ == '__main__':
    c = 0
    sentence = []
    morphs = []
    fname = '../data/src/ai.ja.text.parsed'

    with open(fname, 'r') as f:
        # 一行ごと処理
        for line in f:
            # 係り受け解析、改行のみの行は無視
            if line[0] == '*' or line == '\n':
                continue

            # 文末以外の行にてMorphsを取得、リストに追加
            elif line != 'EOS\n':
                morphs.append(Morphs(line))

            # 文末（EOS）の行にて取得したMorphsを文ごとのリストにまとめる 
            else:
                sentence.append(morphs)
                morphs = []
                break

    # 一部を出力
    for m in sentence[2]:
        print(vars(m))
