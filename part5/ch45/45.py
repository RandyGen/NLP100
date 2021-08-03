class Morphs:
    def __init__(self, morphs):
        (surface, attr) = morphs.split('\t', 1) # 行を最初の\tで区切る
        attr = attr.split(',')
        self.surface = surface # 表層形
        self.base = attr[5]    # 基本形
        self.pos = attr[0]     # 品詞
        self.pos1 = attr[1]    # 品詞細分類


class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []


class Sentence():
  def __init__(self, chunks):
    self.chunks = chunks
    for i, chunk in enumerate(self.chunks):
      if chunk.dst != -1:
        self.chunks[chunk.dst].srcs.append(i)


if __name__ == '__main__':
    c = 0
    sentences = []
    morphs = []
    chunks = []
    fname = '../data/src/ai.ja.text.parsed'

    with open(fname, 'r') as f:
        # 一行ごと処理
        for line in f:
            # 係り受け解析の行
            if line[0] == '*':
                if morphs != []:
                    chunks.append(Chunk(morphs, dst))
                    morphs = []
                dst = int(line.split()[2].rstrip('D'))

            # 改行のみの行は無視
            elif line == '\n':
                continue

            # 文末（EOS）の行
            elif line != 'EOS\n':
                morphs.append(Morphs(line))
            else:
                chunks.append(Chunk(morphs, dst))
                sentences.append(Sentence(chunks))
                morphs = []
                chunks = []
                dst = None

# 以上は41.pyと同様 ====================================================================================================================

with open('./ans45.txt', 'w') as f:
  for sentence in sentences:
    for chunk in sentence.chunks:
      for morph in chunk.morphs:
        if morph.pos == '動詞':  # chunkの左から順番に動詞を探す
          cases = []
          for src in chunk.srcs:  # 見つけた動詞の係り元chunkから助詞を探す
            cases = cases + [morph.surface for morph in sentence.chunks[src].morphs if morph.pos == '助詞']
          if len(cases) > 0:  # 助詞が見つかった場合は重複除去後辞書順にソートして出力
            cases = sorted(list(set(cases)))
            line = '{}\t{}'.format(morph.base, ' '.join(cases))
            print(line, file=f)
          break
