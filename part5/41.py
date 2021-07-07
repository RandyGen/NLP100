class Morphs:
    def __init__(self, morphs):
        (surface, attr) = morphs.split('\t', 1)
        attr = attr.split(',')
        self.surface = surface
        self.base = attr[5]
        self.pos = attr[0]
        self.pos1 = attr[1]


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
    fname = './ai.ja.text.parsed'

    with open(fname, 'r') as f:
        for line in f:
            if line[0] == '*':
                if morphs != []:
                    chunks.append(Chunk(morphs, dst))
                    morphs = []
                dst = int(line.split(' ')[2].rsplit('D'))
            elif line != 'EOS\n':
                morphs.append(Morphs(line))
            else:
                chunks.append(Chunk(morphs, dst))
                sentences.append(Sentence(chunks))
                morphs = []
                chunks = []
                dst = None

    for chunk in sentences[2].chunks:
        print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)
