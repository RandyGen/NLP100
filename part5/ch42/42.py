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

    sentence = sentences[2]
    for chunk in sentence.chunks:
        if int(chunk.dst) != -1:
            modifier = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])
            modifiee = ''.join([morph.surface if morph.pos != '記号' else '' for morph in sentence.chunks[int(chunk.dst)].morphs])
            print(modifier, modifiee, sep='\t')
            